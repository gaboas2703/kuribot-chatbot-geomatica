"""Gestor de diálogo de KuriBot: estado, llenado de ranuras, políticas y generación de respuestas.

Flujo por mensaje: (1) enmascarar datos personales; (2) red de seguridad de emergencias;
(3) extraer entidades; (4) completar una acción pendiente o (5) clasificar la intención;
(6) aplicar la política de la intención (consultas, geoprocesos, red bayesiana, reglas);
(7) responder con plantillas y devolver una traza auditable.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import entidades as E
from . import geodesia as G
from .conocimiento import BaseConocimiento
from .nlu import ClasificadorIntenciones
from .razonamiento import RedAlertas, motor_alertas, motor_contexto, primera_ventana

warnings.filterwarnings("ignore", category=FutureWarning)

NOMBRES_INTENCION = {
    "saludo": "saludar", "despedida": "despedirse", "ayuda_capacidades": "conocer qué puedo hacer",
    "consultar_concesion": "consultar los datos de una concesión", "verificar_punto": "verificar si un punto cae en una concesión",
    "convertir_coordenadas": "convertir coordenadas entre sistemas", "reportar_alerta": "reportar una posible actividad ilegal",
    "solicitar_vuelo": "solicitar un vuelo de dron", "solicitar_cartografia": "pedir un producto cartográfico",
    "estado_solicitud": "consultar el estado de una solicitud", "consulta_geodesia": "resolver una duda geodésica",
    "hablar_humano": "hablar con una persona", "emergencia_seguridad": "reportar una emergencia",
    "fuera_alcance": "un tema fuera de mi alcance",
}
CAPACIDADES = ("Puedo ayudarle a: consultar una concesión (p. ej., «estado de CM-103»); verificar si un punto cae dentro "
               "de una concesión; convertir coordenadas WGS 84 ↔ PSAD56; reportar una posible actividad minera ilegal; "
               "solicitar un vuelo de dron o un producto cartográfico; consultar el estado de un ticket (GEO-2026-0000); "
               "resolver dudas geodésicas frecuentes, o derivarle con una persona.")
AVISO = ("Soy KuriBot, un asistente automatizado de Geomática (prototipo académico con datos ficticios). "
         "Mis respuestas no sustituyen la revisión de un especialista.")
UBICACION = {"propia": "concesión propia", "tercero": "concesión de un tercero", "amortiguamiento": "franja de 500 m",
             "fuera": "fuera del área de la empresa"}
NEGATIVOS = re.compile(r"^\s*(no s[eé]|nada|ninguna|ninguno|no vi nada|no|desconozco|no tengo)\b", re.I)
CANCELAR = re.compile(r"\b(cancelar|cancela|olv[ií]dalo|dejalo|déjalo)\b", re.I)


_PII_FRASE = re.compile(r"\b(mi|su|el|la)?\s*(c[eé]dula|tel[eé]fono|celular|n[uú]mero|correo)\s*(es|:)?\s*\[(C[ÉE]DULA|TEL[ÉE]FONO|CORREO)\]\s*,?", re.I)
_PII_MARCA = re.compile(r"\[(C[ÉE]DULA|TEL[ÉE]FONO|CORREO)\]")


def sin_pii(t: str) -> str:
    """Texto para el clasificador: los datos personales enmascarados no aportan información de intención (ajuste v2)."""
    t = _PII_MARCA.sub(" ", _PII_FRASE.sub(" ", t))
    return re.sub(r"\s+", " ", t).strip(" ,")


def pct(p: float) -> str:
    return f"{100 * p:.1f} %".replace(".", ",")


def num(x: float, dec: int = 1) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", " ").replace(".", ",")


@dataclass
class Respuesta:
    texto: str
    intencion: str | None = None
    confianza: float | None = None
    decision: str | None = None
    top: list = field(default_factory=list)
    accion: str | None = None
    reglas: list = field(default_factory=list)
    traza_reglas: list = field(default_factory=list)
    bayes: dict | None = None
    entidades: dict = field(default_factory=dict)
    ticket: str | None = None
    mapa: dict | None = None
    vuelo: dict | None = None
    latencia_ms: float = 0.0
    latencia_nlu_ms: float = 0.0
    avisos: list = field(default_factory=list)


@dataclass
class Estado:
    pendiente: dict | None = None
    aclaracion: dict | None = None
    fallos: int = 0
    intentos_ranura: int = 0
    ultima_concesion: str | None = None
    turnos: int = 0


class KuriBot:
    def __init__(self, nlu: ClasificadorIntenciones | None = None, kb: BaseConocimiento | None = None,
                 registro: str | Path | None = None):
        self.nlu = nlu or ClasificadorIntenciones()
        self.kb = kb or BaseConocimiento()
        self.red = RedAlertas()
        self.motor = motor_alertas()
        self.motor_ctx = motor_contexto()
        self.estado = Estado()
        self.registro = Path(registro) if registro else None

    def reiniciar(self):
        self.estado = Estado()

    # ================================================================ entrada principal
    def responder(self, texto: str) -> Respuesta:
        t0 = time.perf_counter()
        self.estado.turnos += 1
        seguro, pii = E.enmascarar_pii(texto)
        ent = self._entidades(seguro)
        r = self._despachar(seguro, ent)
        if pii:
            r.avisos.append("R02: detecté datos personales (" + ", ".join(sorted(set(pii))) + "); no los registro ni los necesito.")
            r.reglas = ["R02"] + r.reglas
        r.entidades = {k: v for k, v in ent.items() if v not in (None, {}, [])}
        r.latencia_ms = (time.perf_counter() - t0) * 1000
        self._registrar(seguro, r)
        return r

    def _entidades(self, t: str) -> dict:
        c = E.extraer_coordenada(t)
        return {"coord": c, "datum": E.extraer_datum(t), "concesion": E.extraer_concesion(t, self.kb.nombres),
                "ticket": E.extraer_ticket(t), "fecha": E.extraer_fecha(t, self.kb.hoy), "producto": E.extraer_producto(t),
                "evidencias": E.extraer_evidencias(t), "urgente": bool(re.search(r"urgent", E.plano(t)))}

    def _despachar(self, t: str, ent: dict) -> Respuesta:
        st = self.estado
        if CANCELAR.search(t) and (st.pendiente or st.aclaracion):
            st.pendiente = st.aclaracion = None
            return Respuesta("De acuerdo, cancelé la solicitud en curso. ¿En qué más puedo ayudarle?", "cancelar", decision="regla")
        pred = self.nlu.predecir(sin_pii(t))
        base = {"confianza": pred.confianza, "top": pred.top, "latencia_nlu_ms": pred.latencia_ms}
        # (2) red de seguridad: emergencias por palabras clave o por la red neuronal
        if E.hay_emergencia(t) or (pred.intencion == "emergencia_seguridad" and pred.confianza >= self.nlu.umbral_abstener):
            st.pendiente = st.aclaracion = None
            return self._con(self._emergencia(t, ent), pred, "emergencia_seguridad", base)
        # (3) respuesta a una aclaración
        if st.aclaracion:
            elegido = self._resolver_aclaracion(t)
            if elegido:
                original = st.aclaracion["texto"]
                st.aclaracion = None
                ent2 = self._entidades(original)
                for k, v in ent.items():
                    if v and not ent2.get(k):
                        ent2[k] = v
                return self._con(self._politica(elegido, original, ent2), pred, elegido, base, decision="aclarada")
            st.aclaracion = None
        # (4) completar acción pendiente
        if st.pendiente:
            p = st.pendiente
            self._fusionar(p["slots"], ent)
            if self._aporta(p["falta"], ent, t) or (pred.intencion == p["intencion"] and pred.decision == "aceptar"):
                st.pendiente = None
                if p["falta"] == "evidencias" and not ent["evidencias"]:
                    p["slots"]["evidencias_sin_dato"] = True
                return self._con(self._politica(p["intencion"], t, p["slots"]), pred, p["intencion"], base, decision="ranura")
            if pred.decision == "aceptar" and pred.intencion not in ("fuera_alcance",):
                st.pendiente = None  # cambio de tema: se abandona la acción pendiente
            else:
                st.intentos_ranura += 1
                if st.intentos_ranura >= 2:
                    st.pendiente = None
                    return self._con(Respuesta("No logré completar el dato requerido. Puedo derivarle con un analista "
                                               "(escriba «quiero hablar con una persona») o puede empezar de nuevo."),
                                     pred, p["intencion"], base, decision="ranura_fallida")
                return self._con(Respuesta(p["pregunta"] + " (o escriba «cancelar»)."), pred, p["intencion"], base, decision="ranura")
        # (5) clasificación de la intención
        if pred.decision == "aclarar":
            a, b = pred.top[0][0], pred.top[1][0]
            st.aclaracion = {"opciones": [a, b], "texto": t}
            return self._con(Respuesta(f"No estoy seguro de haber entendido. ¿Quiere 1) {NOMBRES_INTENCION[a]} o "
                                       f"2) {NOMBRES_INTENCION[b]}? Responda 1 o 2."), pred, a, base, decision="aclarar")
        if pred.decision == "abstener":
            st.fallos += 1
            extra = (" Si lo prefiere, puedo derivarle con un analista: escriba «quiero hablar con una persona»."
                     if st.fallos >= 2 else "")
            return self._con(Respuesta("No tengo suficiente seguridad para responder eso. " + CAPACIDADES + extra),
                             pred, "fuera_alcance", base, decision="abstener")
        st.fallos = 0
        return self._con(self._politica(pred.intencion, t, dict(ent)), pred, pred.intencion, base, decision="aceptar")

    @staticmethod
    def _con(r: Respuesta, pred, intencion, base, decision=None) -> Respuesta:
        r.intencion = r.intencion or intencion
        r.confianza, r.top, r.latencia_nlu_ms = base["confianza"], base["top"], base["latencia_nlu_ms"]
        r.decision = r.decision or decision or pred.decision
        return r

    def _resolver_aclaracion(self, t: str) -> str | None:
        ops = self.estado.aclaracion["opciones"]
        tl = E.plano(t).strip()
        if re.fullmatch(r"(1|uno|la primera|primera|opcion 1|si|sí|a)\.?", tl):
            return ops[0]
        if re.fullmatch(r"(2|dos|la segunda|segunda|opcion 2|b)\.?", tl):
            return ops[1]
        return None

    @staticmethod
    def _fusionar(slots: dict, ent: dict):
        for k, v in ent.items():
            if k == "evidencias":
                slots.setdefault("evidencias", {}).update(v)
            elif v not in (None, {}, False):
                slots[k] = v

    @staticmethod
    def _aporta(falta: str, ent: dict, t: str) -> bool:
        if falta == "ubicacion":
            return bool(ent["coord"] or ent["concesion"])
        if falta == "evidencias":
            return bool(ent["evidencias"]) or bool(NEGATIVOS.search(t))
        return ent.get(falta) is not None and ent.get(falta) != {}

    def _pedir(self, intencion: str, falta: str, slots: dict, pregunta: str) -> Respuesta:
        self.estado.pendiente = {"intencion": intencion, "falta": falta, "slots": slots, "pregunta": pregunta}
        self.estado.intentos_ranura = 0
        return Respuesta(pregunta, intencion, accion=f"solicitar_dato:{falta}")

    # ================================================================ políticas por intención
    def _politica(self, intencion: str, t: str, s: dict) -> Respuesta:
        h = getattr(self, f"_i_{intencion}", None)
        return h(t, s) if h else Respuesta(CAPACIDADES, intencion)

    def _i_saludo(self, t, s):
        return Respuesta(f"¡Hola! {AVISO} {CAPACIDADES}", "saludo", accion="responder")

    def _i_despedida(self, t, s):
        self.estado.pendiente = None
        return Respuesta("Gracias por escribir. Si necesita algo más de Geomática, aquí estaré.", "despedida", accion="cerrar")

    def _i_ayuda_capacidades(self, t, s):
        return Respuesta(CAPACIDADES + " Para emergencias con riesgo para personas escriba «emergencia».", accion="responder")

    def _i_fuera_alcance(self, t, s):
        return Respuesta("Ese tema está fuera de mi alcance: solo atiendo consultas de Geomática y Catastro minero. "
                         + CAPACIDADES, accion="fuera_alcance")

    def _i_consultar_concesion(self, t, s):
        cod = s.get("concesion") or (self.estado.ultima_concesion if re.search(r"\b(esa|esta|la misma)\b", E.plano(t)) else None)
        if not cod:
            return self._pedir("consultar_concesion", "concesion", s, "¿De qué concesión? Indique el código (p. ej., CM-103) o el nombre")
        f = self.kb.ficha(cod)
        if not f:
            return Respuesta(f"No encuentro la concesión {cod} en el catastro de demostración; no puedo inventar sus datos.",
                             accion="no_encontrado")
        self.estado.ultima_concesion = cod
        if f["titular"] != "propia":
            return Respuesta(f"{cod} {f['nombre']} pertenece a un tercero con derecho vigente. Por confidencialidad solo "
                             f"confirmo su existencia y su superficie ({num(f['area_ha'])} ha); para más detalle, consulte a Legal y Catastro.",
                             accion="responder", mapa=self._mapa(cod))
        act = f"; actividad autorizada programada: {f['actividad_programada']}" if f["actividad_programada"] else ""
        return Respuesta(f"Concesión {cod} {f['nombre']}: fase {f['fase']}, estado {f['estado']}, superficie {num(f['area_ha'])} ha, "
                         f"campamento {f['campamento']}. Alertas en los últimos 90 días: {f['alertas_90d']}{act}. "
                         "Fuente: catastro interno de demostración (sintético).", accion="responder", mapa=self._mapa(cod))

    def _coordenada_lista(self, intencion: str, s: dict, pregunta_coord: str):
        c = s.get("coord")
        if c is None:
            return None, self._pedir(intencion, "coord", s, pregunta_coord)
        datum = s.get("datum") or c.punto.datum
        if datum is None:
            return None, self._pedir(intencion, "datum", s, "¿En qué datum están esas coordenadas: WGS 84 o PSAD56? "
                                                            "(en esta zona difieren unos 440 m)")
        p = G.Punto(c.punto.x, c.punto.y, c.punto.tipo, datum)
        e, n = G.a_wgs84_utm(p)
        if not G.dentro_de_ecuador_utm17s(e, n):
            return None, Respuesta("Las coordenadas quedan fuera del rango de Ecuador continental en UTM 17S (G02). "
                                   "Revise si el Este y el Norte están invertidos o si falta un dígito.", intencion, accion="rechazar")
        return (p, e, n, c.avisos), None

    def _i_verificar_punto(self, t, s):
        listo, r = self._coordenada_lista("verificar_punto", s, "Indique las coordenadas del punto (p. ej., E 642350 N 9662120) y su datum")
        if r:
            return r
        p, e, n, avisos = listo
        avisos = list(avisos)
        a = self.kb.analizar_punto(e, n)
        f = self.motor_ctx.inferir({"ubicacion": a.ubicacion, "area_protegida": bool(a.area_protegida),
                                    "comunidad_cercana": bool(a.comunidad)})
        partes = []
        if p.datum == "PSAD56":
            partes.append(f"Transformé el punto de PSAD56 a WGS 84 con {G.descripcion_transformacion()}: E {num(e)} N {num(n)}.")
            if p.tipo == "utm":
                crudo = self.kb.analizar_punto(p.x, p.y)  # lectura errónea: mismas cifras tomadas como WGS 84
                if (crudo.concesion or crudo.ubicacion) != (a.concesion or a.ubicacion):
                    avisos.append(f"Riesgo de datum: si estas cifras se leyeran como WGS 84, el punto caería en "
                                  f"{crudo.concesion or 'una zona sin concesión'} y no en {a.concesion or 'una zona sin concesión'}.")
        if a.ubicacion == "propia":
            c = self.kb.concesiones[a.concesion]
            partes.append(f"El punto está DENTRO de {a.concesion} {c['nombre']} ({c['estado']}), a {num(a.distancia_lindero_m, 0)} m del lindero.")
        elif a.ubicacion == "tercero":
            partes.append(f"El punto está dentro de {a.concesion}, concesión de un tercero con derecho vigente (R12). "
                          "No debe intervenirse sin revisión de Legal.")
        elif a.ubicacion == "amortiguamiento":
            partes.append(f"El punto está FUERA de las concesiones, a {num(a.distancia_lindero_m, 0)} m de {a.cercana} "
                          "(dentro de la franja de 500 m, R11).")
        else:
            partes.append(f"El punto está fuera de las concesiones de la empresa; la más cercana es {a.cercana} a {num(a.distancia_lindero_m, 0)} m.")
        if a.area_protegida:
            partes.append(f"Atención: cae en {a.area_protegida} (R13).")
        if a.comunidad:
            partes.append(f"Hay una comunidad a {num(a.distancia_comunidad_m, 0)} m: {a.comunidad} (R14).")
        partes.append("Verificación sobre el catastro sintético del prototipo; para fines legales use el catastro oficial vigente.")
        self.estado.ultima_concesion = a.concesion or a.cercana
        return Respuesta(" ".join(partes), "verificar_punto", accion="geoproceso", reglas=f.disparadas, traza_reglas=f.traza,
                         mapa=self._mapa(a.concesion or a.cercana, (e, n)), avisos=avisos)

    def _i_convertir_coordenadas(self, t, s):
        listo, r = self._coordenada_lista("convertir_coordenadas", s, "Indique las coordenadas a convertir y su datum de origen")
        if r:
            return r
        p, e, n, avisos = listo
        rep = G.todas_las_representaciones(p)
        (we, wn), (lat, lon), (pe, pn) = rep["WGS84_UTM17S"], rep["WGS84_GEO"], rep["PSAD56_UTM17S"]
        texto = (f"Origen: {p.datum}. Resultados: WGS 84 / UTM 17S (EPSG:32717): E {num(we, 2)} m, N {num(wn, 2)} m · "
                 f"WGS 84 geográficas (EPSG:4326): {lat:.6f}, {lon:.6f} ({G.dms(lat, 'lat')}, {G.dms(lon, 'lon')}) · "
                 f"PSAD56 / UTM 17S (EPSG:24877): E {num(pe, 2)} m, N {num(pn, 2)} m. "
                 f"Transformación de datum: {G.descripcion_transformacion()}. SIRGAS 2000 se considera equivalente a WGS 84 "
                 "al nivel métrico.")
        return Respuesta(texto, "convertir_coordenadas", accion="geoproceso", mapa=self._mapa(None, (we, wn)), avisos=list(avisos))

    def _ubicar_alerta(self, s) -> dict:
        """Contexto geoespacial y nodos raíz (H, A) de la red bayesiana para la alerta."""
        ctx = {"codigo": None, "ubicacion": None, "area_protegida": False, "comunidad_cercana": False, "punto": None}
        ev = {}
        if s.get("coord") is not None:
            datum = s.get("datum") or s["coord"].punto.datum or "WGS84"
            pt = G.Punto(s["coord"].punto.x, s["coord"].punto.y, s["coord"].punto.tipo, datum)
            e, n = G.a_wgs84_utm(pt)
            a = self.kb.analizar_punto(e, n)
            ctx.update(codigo=a.concesion or a.cercana, ubicacion=a.ubicacion, area_protegida=bool(a.area_protegida),
                       comunidad_cercana=bool(a.comunidad), punto=(e, n), datum_supuesto=s.get("datum") is None and s["coord"].punto.datum is None)
            if a.ubicacion in ("propia", "amortiguamiento"):
                ev["H"] = int(self.kb.op["historial_alertas_90d"].get(ctx["codigo"], 0) > 0)
                ev["A"] = int(a.ubicacion == "propia" and ctx["codigo"] in self.kb.op["actividad_autorizada_programada"])
        else:
            cod = s["concesion"]
            c = self.kb.concesiones.get(cod)
            ctx["codigo"] = cod
            if c:
                ctx["ubicacion"] = "propia" if c["titular"] == "propia" else "tercero"
                g = c["geom"]
                ctx["area_protegida"] = any(g.distance(x["geom"]) <= 1.0 for x in self.kb.areas.values())
                ctx["comunidad_cercana"] = any(g.distance(x["geom"]) <= 1000 for x in self.kb.comunidades.values())
                ev["H"] = int(self.kb.op["historial_alertas_90d"].get(cod, 0) > 0)
                if c["titular"] == "propia":
                    ev["A"] = int(cod in self.kb.op["actividad_autorizada_programada"])
        return {"ctx": ctx, "ev_raiz": ev}

    def _i_reportar_alerta(self, t, s):
        if not (s.get("coord") or s.get("concesion")):
            return self._pedir("reportar_alerta", "ubicacion", s, "¿Dónde ocurre? Indique la concesión (código o nombre) o las coordenadas con su datum")
        if s.get("concesion") and not s.get("coord") and s["concesion"] not in self.kb.concesiones:
            return Respuesta(f"No encuentro {s['concesion']} en el catastro; indique coordenadas para ubicar la alerta.", accion="no_encontrado")
        if not s.get("evidencias") and not s.get("evidencias_sin_dato"):
            return self._pedir("reportar_alerta", "evidencias", s, "¿Qué observó? Por ejemplo: maquinaria, personas o campamento, "
                                                                     "desbroce o huecos nuevos, agua turbia, ruido o voladuras de noche, trocha nueva. "
                                                                     "Si no lo sabe, escriba «no sé»")
        u = self._ubicar_alerta(s)
        ctx, ev = u["ctx"], dict(u["ev_raiz"])
        obs = {k: int(v) for k, v in (s.get("evidencias") or {}).items()}
        ev.update(obs)
        p = self.red.posterior(ev)
        contrib = self.red.contribuciones(ev)
        camp = self.kb.campamento(ctx["codigo"])
        ventana = primera_ventana(0, camp, self.kb.op, urgente=True)
        hechos = {"ubicacion": ctx["ubicacion"], "area_protegida": ctx["area_protegida"],
                  "comunidad_cercana": ctx["comunidad_cercana"], "p_intrusion": p, "vuelo_dias": ventana["dia"]}
        inf = self.motor.inferir(hechos)
        f = inf.hechos
        prioridad, accion = f.get("prioridad"), f.get("accion")
        resp = "Seguridad patrimonial" if prioridad not in ("REVISIÓN LEGAL", "INFORMATIVA") else "Legal y Catastro"
        compromiso = self.kb.fecha(ventana["dia"]) if ventana["dia"] is not None and prioridad in ("CRÍTICA", "ALTA") else None
        tk = self.kb.nuevo_ticket("alerta", f"Alerta en {ctx['codigo']} · P={p:.3f}", resp, compromiso, prioridad)
        nombre = self.kb.concesiones.get(ctx["codigo"], {}).get("nombre", "")
        evid = ", ".join(f"{E.EVIDENCIAS[k][0]} ({'sí' if v else 'no'})" for k, v in obs.items()) or "sin evidencias específicas"
        partes = [f"Alerta registrada: {tk} · prioridad {prioridad}.",
                  f"Ubicación: {ctx['codigo']} {nombre} ({UBICACION.get(ctx['ubicacion'], 'sin ubicar')}).", f"Evidencias: {evid}."]
        if "H" in ev:
            partes.append(f"Antecedentes en 90 días: {'sí' if ev['H'] else 'no'}; actividad autorizada programada: "
                          f"{'sí' if ev.get('A') else ('no' if 'A' in ev else 'desconocida')}.")
        partes.append(f"Probabilidad estimada de actividad no autorizada: {pct(p)} (red bayesiana con supuestos expertos; no es una certeza).")
        if ev.get("A") == 1 and obs:
            partes.append("La actividad autorizada programada explica parte de lo observado, por eso la probabilidad no es mayor.")
        partes.append(f"Acción propuesta: {accion}.")
        if ventana["dia"] is not None and prioridad in ("CRÍTICA", "ALTA"):
            partes.append(f"Primera ventana de vuelo apta: {self.kb.fecha(ventana['dia'])} ({ventana['dron']}, {ventana['piloto']}).")
        if f.get("aprobacion_supervisor"):
            partes.append("Antes de ejecutar, el supervisor de turno debe aprobar la acción (R41).")
        if f.get("revision_humana"):
            partes.append("Zona sensible: se requiere revisión humana y protocolo comunitario antes de cualquier acción de campo (R40).")
        if ctx.get("datum_supuesto"):
            partes.append("Asumí WGS 84 para las coordenadas porque no indicó el datum; confírmelo si no es así.")
        bayes = {"p_intrusion": p, "evidencia": ev, "contribuciones": contrib}
        return Respuesta(" ".join(partes), "reportar_alerta", accion=accion, reglas=inf.disparadas, traza_reglas=inf.traza,
                         bayes=bayes, ticket=tk, vuelo=ventana, mapa=self._mapa(ctx["codigo"], ctx["punto"]))

    def _i_solicitar_vuelo(self, t, s):
        cod = s.get("concesion")
        if not cod:
            return self._pedir("solicitar_vuelo", "concesion", s, "¿Sobre qué concesión necesita el vuelo? (código o nombre)")
        if cod not in self.kb.concesiones:
            return Respuesta(f"No encuentro {cod} en el catastro de demostración.", accion="no_encontrado")
        if s.get("fecha") is None:
            return self._pedir("solicitar_vuelo", "fecha", s, "¿Para qué fecha? (hoy, mañana, un día de la semana o una fecha)")
        camp = self.kb.campamento(cod)
        dia = max(0, s["fecha"])
        if dia > self.kb.op["politica_vuelo"]["horizonte_dias"]:
            return Respuesta(f"Solo tengo pronóstico hasta {self.kb.fecha(self.kb.op['politica_vuelo']['horizonte_dias'])}. "
                             "Registre la solicitud más cerca de la fecha o hable con el coordinador de vuelos.", accion="fuera_horizonte")
        v = primera_ventana(dia, camp, self.kb.op, urgente=s.get("urgente", False))
        rechazos = [f"{self.kb.fecha(d['dia'])}: " + "; ".join(m.split(' ', 1)[1] for m in d["motivos"]) for d in v["evaluados"] if not d["apto"]]
        if v["dia"] is None:
            return Respuesta(f"No hay ventana apta en el horizonte de pronóstico para el campamento {camp}. Motivos: "
                             + " | ".join(rechazos) + ". Derivo al coordinador de vuelos.", accion="sin_ventana", vuelo=v)
        tk = self.kb.nuevo_ticket("vuelo", f"Vuelo en {cod}", v["piloto"], self.kb.fecha(v["dia"]))
        cab = "Fecha solicitada apta." if v["dia"] == dia else f"La fecha solicitada no es apta ({' | '.join(rechazos)})."
        return Respuesta(f"{cab} Propuesta: vuelo en {cod} el {self.kb.fecha(v['dia'])} con {v['dron']} y {v['piloto']} "
                         f"(viento {num(v['viento'])} m/s, sin lluvia, visibilidad {v['visibilidad']} km). Solicitud {tk} "
                         "registrada como pendiente de aprobación del coordinador de vuelos.", "solicitar_vuelo",
                         accion="programar_vuelo", ticket=tk, vuelo=v, mapa=self._mapa(cod))

    def _i_solicitar_cartografia(self, t, s):
        prod = s.get("producto")
        if not prod:
            return self._pedir("solicitar_cartografia", "producto", s, "¿Qué producto necesita? (mapa de ubicación, plano topográfico, "
                                                                        "ortofoto, KMZ, shapefile o modelo digital de elevación)")
        cod = s.get("concesion") or self.estado.ultima_concesion
        if not cod:
            return self._pedir("solicitar_cartografia", "concesion", s, "¿De qué concesión o área? (código o nombre)")
        sla = self.kb.op["sla_cartografia_dias_habiles"]
        dias = sla["urgente"] if s.get("urgente") else sla["estandar"]
        tk = self.kb.nuevo_ticket("cartografía", f"{prod} de {cod}", "Analista SIG", None)
        return Respuesta(f"Solicitud {tk} registrada: {prod} de {cod}. Plazo interno de referencia: {dias} día(s) hábil(es)"
                         f"{' (urgente: requiere aprobación de la jefatura)' if s.get('urgente') else ''}. "
                         "El producto se entregará en EPSG:32717 salvo que indique otro sistema.", "solicitar_cartografia",
                         accion="crear_ticket", ticket=tk)

    def _i_estado_solicitud(self, t, s):
        tk = s.get("ticket")
        if not tk:
            return self._pedir("estado_solicitud", "ticket", s, "Indique el número de solicitud (formato GEO-2026-0000)")
        d = self.kb.ticket(tk)
        if not d:
            return Respuesta(f"No encuentro la solicitud {tk}. Revise el número; no puedo suponer su estado.", accion="no_encontrado")
        extra = f" Resolución: {d['resolucion']}." if d.get("resolucion") else ""
        comp = f", fecha comprometida {d['compromiso']}" if d.get("compromiso") else ""
        return Respuesta(f"{tk} ({d['tipo']}: {d['detalle']}): estado «{d['estado']}», responsable {d['asignado']}{comp}.{extra}",
                         "estado_solicitud", accion="consultar_ticket", ticket=tk)

    def _i_consulta_geodesia(self, t, s):
        f, sim = self.kb.buscar_faq(t)
        if sim < 0.30:
            return Respuesta("No tengo una respuesta verificada para esa duda. Para no darle información incorrecta, la derivo "
                             "al analista de geodesia (o reformule la pregunta).", accion="derivar_faq",
                             traza_reglas=[f"FAQ: similitud máxima {sim:.2f} < 0,30"])
        return Respuesta(f"{f['respuesta']} (Fuente: {f['fuente']}.)", "consulta_geodesia", accion="responder_faq",
                         traza_reglas=[f"FAQ {f['id']} · similitud {sim:.2f}"])

    def _i_hablar_humano(self, t, s):
        inf = self.motor.inferir({"intencion": "hablar_humano"})
        tk = self.kb.nuevo_ticket("derivación", "Solicitud de atención humana", "Mesa de Geomática", None)
        return Respuesta(f"Entendido. Registré la derivación {tk}; un analista de la Mesa de Geomática le contactará por el canal "
                         "interno en horario laboral (simulado en este prototipo).", "hablar_humano", accion="derivar_humano",
                         reglas=inf.disparadas, traza_reglas=inf.traza, ticket=tk)

    def _emergencia(self, t, ent):
        u = self._ubicar_alerta(ent) if (ent.get("coord") or ent.get("concesion")) else {"ctx": {"codigo": None}}
        inf = self.motor.inferir({"emergencia": True})
        cod = u["ctx"].get("codigo")
        tk = self.kb.nuevo_ticket("emergencia", f"Emergencia reportada{' en ' + cod if cod else ''}", "Jefe de Seguridad", self.kb.fecha(0), "CRÍTICA")
        return Respuesta("EMERGENCIA: si hay personas en riesgo, llame de inmediato al ECU 911 y al Jefe de Seguridad del campamento. "
                         "Aléjese del lugar si es seguro hacerlo. No se enviará personal ni dron a la zona hasta que Seguridad lo autorice (R01). "
                         f"Registré el caso {tk} con prioridad CRÍTICA y lo derivé a una persona.", "emergencia_seguridad",
                         accion="PROTOCOLO DE EMERGENCIA", reglas=inf.disparadas, traza_reglas=inf.traza, ticket=tk)

    # ================================================================ utilidades
    def _mapa(self, codigo: str | None, punto: tuple[float, float] | None = None) -> dict:
        return {"concesion": codigo, "punto": punto}

    def _registrar(self, texto: str, r: Respuesta):
        if not self.registro:
            return
        fila = {"turno": self.estado.turnos, "texto_enmascarado": texto, "intencion": r.intencion, "confianza": r.confianza,
                "decision": r.decision, "accion": r.accion, "reglas": r.reglas, "ticket": r.ticket,
                "p_intrusion": (r.bayes or {}).get("p_intrusion"), "latencia_ms": round(r.latencia_ms, 2)}
        self.registro.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registro, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fila, ensure_ascii=False) + "\n")


def respuesta_a_dict(r: Respuesta) -> dict:
    d = asdict(r)
    if r.entidades.get("coord") is not None:
        c = r.entidades["coord"]
        d["entidades"]["coord"] = {"x": c.punto.x, "y": c.punto.y, "tipo": c.punto.tipo, "datum": c.punto.datum}
    return d
