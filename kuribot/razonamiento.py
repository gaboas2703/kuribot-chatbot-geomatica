"""Componente de razonamiento de KuriBot (contenidos de las semanas 2 y 3).

1. MotorReglas: sistema de producción con encadenamiento hacia adelante, agenda por nivel y
   prioridad, refracción (cada regla dispara una vez) y traza explicativa. Es la continuación
   del sistema experto SEPRIM de la semana 2, ahora alimentado desde la conversación.
2. RedAlertas: red bayesiana discreta (9 nodos) que estima P(intrusión real | evidencias) con
   eliminación de variables exacta (pgmpy). Las tablas de probabilidad condicional (TPC) son
   juicios expertos del equipo, no frecuencias estimadas: deben recalibrarse con el histórico
   de verificaciones de campo.
3. Factibilidad de vuelo: reglas operativas + búsqueda de la primera ventana apta.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ====================================================================== motor de reglas


@dataclass
class Regla:
    id: str
    nivel: int
    prioridad: int
    descripcion: str
    si: Callable[[dict], bool]
    entonces: Callable[[dict], dict]
    texto_si: str = ""
    texto_entonces: str = ""


@dataclass
class ResultadoInferencia:
    hechos: dict
    disparadas: list[str] = field(default_factory=list)
    traza: list[str] = field(default_factory=list)


class MotorReglas:
    def __init__(self, reglas: list[Regla]):
        ids = [r.id for r in reglas]
        if len(ids) != len(set(ids)):
            raise ValueError("Identificadores de regla duplicados")
        self.reglas = sorted(reglas, key=lambda r: (r.nivel, -r.prioridad, r.id))

    def inferir(self, hechos: dict, max_ciclos: int = 100) -> ResultadoInferencia:
        wm = dict(hechos)
        res = ResultadoInferencia(wm)
        disparadas: set[str] = set()
        for _ in range(max_ciclos):
            agenda = [r for r in self.reglas if r.id not in disparadas and _seguro(r.si, wm)]
            if not agenda:
                break
            r = agenda[0]  # resolución de conflictos: menor nivel, mayor prioridad, id
            nuevos = r.entonces(wm)
            wm.update(nuevos)
            disparadas.add(r.id)
            res.disparadas.append(r.id)
            res.traza.append(f"{r.id} (nivel {r.nivel}): {r.descripcion}")
        res.hechos = wm
        return res


def _seguro(cond, wm) -> bool:
    try:
        return bool(cond(wm))
    except (KeyError, TypeError):
        return False


def _sin_prioridad(f):
    return "prioridad" not in f


UMBRALES = {"critica": 0.80, "alta": 0.50, "media": 0.15}  # supuestos de diseño (ver informe, sección 5.4)

REGLAS_ALERTA: list[Regla] = [
    # ---------------- nivel 0: seguridad, privacidad, autonomía
    Regla("R01", 0, 100, "Riesgo para personas: protocolo de emergencia, sin despliegue de personal ni dron",
          lambda f: f.get("emergencia"), lambda f: {"prioridad": "CRÍTICA", "accion": "PROTOCOLO DE EMERGENCIA",
                                                     "prohibir_despliegue": True, "derivar_humano": True},
          "emergencia = verdadero", "prioridad = CRÍTICA; acción = protocolo; derivar a humano"),
    Regla("R02", 0, 90, "Datos personales detectados: se enmascaran en el registro",
          lambda f: f.get("pii"), lambda f: {"enmascarar_registro": True}, "pii = verdadero", "enmascarar registro"),
    Regla("R03", 0, 80, "Solicitud explícita de atención humana",
          lambda f: f.get("intencion") == "hablar_humano", lambda f: {"derivar_humano": True},
          "intención = hablar_humano", "derivar a humano"),
    # ---------------- nivel 1: contexto geoespacial
    Regla("R10", 1, 50, "El punto está dentro de una concesión propia",
          lambda f: f.get("ubicacion") == "propia", lambda f: {"zona": "concesión propia"}, "ubicación = propia", "zona = concesión propia"),
    Regla("R11", 1, 50, "El punto está en la franja de 500 m alrededor de una concesión propia",
          lambda f: f.get("ubicacion") == "amortiguamiento", lambda f: {"zona": "franja de amortiguamiento"},
          "ubicación = amortiguamiento", "zona = amortiguamiento"),
    Regla("R12", 1, 60, "El punto está en una concesión de un tercero con derecho vigente",
          lambda f: f.get("ubicacion") == "tercero", lambda f: {"derecho_tercero": True}, "ubicación = tercero", "derecho_tercero"),
    Regla("R13", 1, 40, "El punto está dentro de un área protegida",
          lambda f: f.get("area_protegida"), lambda f: {"area_sensible": True}, "área protegida", "área sensible"),
    Regla("R14", 1, 40, "Hay una comunidad a 1 km o menos",
          lambda f: f.get("comunidad_cercana"), lambda f: {"sensibilidad_social": True}, "comunidad ≤ 1 km", "sensibilidad social"),
    Regla("R15", 1, 30, "El punto está fuera del área de responsabilidad de la empresa",
          lambda f: f.get("ubicacion") == "fuera", lambda f: {"fuera_jurisdiccion": True}, "ubicación = fuera", "fuera de jurisdicción"),
    # ---------------- nivel 2: priorización con el posterior bayesiano
    Regla("R20", 2, 100, "Derecho vigente de un tercero: no se califica como intrusión; revisión legal",
          lambda f: _sin_prioridad(f) and f.get("derecho_tercero"), lambda f: {"prioridad": "REVISIÓN LEGAL"},
          "derecho_tercero y sin prioridad", "prioridad = REVISIÓN LEGAL"),
    Regla("R25", 2, 90, "Fuera del área de la empresa: registro informativo",
          lambda f: _sin_prioridad(f) and f.get("fuera_jurisdiccion"), lambda f: {"prioridad": "INFORMATIVA"},
          "fuera_jurisdicción y sin prioridad", "prioridad = INFORMATIVA"),
    Regla("R21", 2, 80, "P(intrusión) ≥ 0,80",
          lambda f: _sin_prioridad(f) and f["p_intrusion"] >= UMBRALES["critica"], lambda f: {"prioridad": "CRÍTICA"},
          "P ≥ 0,80 y sin prioridad", "prioridad = CRÍTICA"),
    Regla("R22", 2, 70, "0,50 ≤ P(intrusión) < 0,80",
          lambda f: _sin_prioridad(f) and f["p_intrusion"] >= UMBRALES["alta"], lambda f: {"prioridad": "ALTA"},
          "P ≥ 0,50 y sin prioridad", "prioridad = ALTA"),
    Regla("R23", 2, 60, "0,15 ≤ P(intrusión) < 0,50",
          lambda f: _sin_prioridad(f) and f["p_intrusion"] >= UMBRALES["media"], lambda f: {"prioridad": "MEDIA"},
          "P ≥ 0,15 y sin prioridad", "prioridad = MEDIA"),
    Regla("R24", 2, 50, "P(intrusión) < 0,15",
          lambda f: _sin_prioridad(f) and "p_intrusion" in f, lambda f: {"prioridad": "BAJA"},
          "sin prioridad (regla por defecto)", "prioridad = BAJA"),
    # ---------------- nivel 3: acción operativa
    Regla("R30", 3, 60, "CRÍTICA con ventana de vuelo hoy o mañana: vuelo prioritario y aviso a Seguridad y Legal",
          lambda f: f.get("prioridad") == "CRÍTICA" and not f.get("emergencia") and f.get("vuelo_dias") is not None and f["vuelo_dias"] <= 1,
          lambda f: {"accion": "Vuelo de verificación prioritario (≤ 24 h) y notificación a Seguridad y Legal"},
          "CRÍTICA y ventana ≤ 1 día", "vuelo prioritario"),
    Regla("R31", 3, 55, "CRÍTICA sin ventana de vuelo inmediata: verificación satelital y patrullaje con seguridad",
          lambda f: f.get("prioridad") == "CRÍTICA" and not f.get("emergencia") and "accion" not in f,
          lambda f: {"accion": "Revisión satelital inmediata y patrullaje con Seguridad; vuelo en la primera ventana apta"},
          "CRÍTICA y sin ventana ≤ 1 día", "satélite + patrullaje"),
    Regla("R32", 3, 50, "ALTA: vuelo en la primera ventana apta (≤ 72 h)",
          lambda f: f.get("prioridad") == "ALTA", lambda f: {"accion": "Programar vuelo de verificación en la primera ventana apta (≤ 72 h)"},
          "ALTA", "vuelo ≤ 72 h"),
    Regla("R33", 3, 40, "MEDIA: revisión satelital y verificación en ronda programada",
          lambda f: f.get("prioridad") == "MEDIA", lambda f: {"accion": "Revisión satelital en el próximo ciclo y verificación en la ronda programada"},
          "MEDIA", "revisión satelital"),
    Regla("R34", 3, 30, "BAJA: registrar y monitorear",
          lambda f: f.get("prioridad") == "BAJA", lambda f: {"accion": "Registrar y mantener monitoreo"}, "BAJA", "registro"),
    Regla("R35", 3, 70, "REVISIÓN LEGAL: Legal y Catastro cotejan derechos del titular vecino",
          lambda f: f.get("prioridad") == "REVISIÓN LEGAL",
          lambda f: {"accion": "Derivar a Legal y Catastro para cotejar los derechos del titular vecino"}, "REVISIÓN LEGAL", "derivar a Legal"),
    Regla("R36", 3, 65, "INFORMATIVA: Legal evalúa si corresponde informar a la autoridad",
          lambda f: f.get("prioridad") == "INFORMATIVA",
          lambda f: {"accion": "Registrar; Legal evalúa si corresponde informar a la autoridad competente"}, "INFORMATIVA", "registro + Legal"),
    # ---------------- nivel 4: salvaguardas con supervisión humana
    Regla("R40", 4, 50, "Área protegida o comunidad cercana: revisión humana y protocolo comunitario antes de actuar",
          lambda f: f.get("area_sensible") or f.get("sensibilidad_social"),
          lambda f: {"revision_humana": True, "protocolo_comunitario": True}, "área sensible o comunidad", "revisión humana"),
    Regla("R41", 4, 40, "Acción de campo (CRÍTICA o ALTA): requiere aprobación del supervisor de turno",
          lambda f: f.get("prioridad") in ("CRÍTICA", "ALTA") and not f.get("emergencia"),
          lambda f: {"aprobacion_supervisor": True}, "CRÍTICA o ALTA", "aprobación humana"),
    Regla("R42", 4, 30, "Toda alerta clasificada genera un ticket trazable",
          lambda f: "prioridad" in f, lambda f: {"crear_ticket": True}, "prioridad asignada", "crear ticket"),
]


def motor_alertas() -> MotorReglas:
    return MotorReglas(REGLAS_ALERTA)


def motor_contexto() -> MotorReglas:
    """Solo reglas de contexto geoespacial y salvaguarda (para verificar puntos, sin priorizar alertas)."""
    return MotorReglas([r for r in REGLAS_ALERTA if r.nivel == 1 or r.id == "R40"])


# ====================================================================== red bayesiana

NODOS = ["H", "A", "I", "M", "P", "S", "R", "T", "V"]
DESCRIPCION_NODOS = {
    "H": "Antecedentes: alertas en el sector en los últimos 90 días",
    "A": "Actividad autorizada programada en el área",
    "I": "Intrusión o actividad minera no autorizada real (variable de interés)",
    "M": "Maquinaria o equipos observados",
    "P": "Personas o campamento no identificados",
    "S": "Alteración del terreno o de la cobertura",
    "R": "Ruido o voladuras nocturnas",
    "T": "Agua turbia aguas abajo",
    "V": "Vía o trocha nueva",
}
ARISTAS = [("H", "I"), ("I", "M"), ("A", "M"), ("I", "P"), ("A", "P"), ("I", "S"), ("A", "S"), ("I", "V"), ("A", "V"),
           ("I", "R"), ("I", "T")]

# TPC elicitadas: P(X = 1 | padres). Para nodos con padres (I, A) el orden es (I=0,A=0), (0,1), (1,0), (1,1).
TPC = {
    "H": 0.30,
    "A": 0.40,
    "I": {(0,): 0.10, (1,): 0.30},
    "M": {(0, 0): 0.03, (0, 1): 0.60, (1, 0): 0.70, (1, 1): 0.85},
    "P": {(0, 0): 0.10, (0, 1): 0.50, (1, 0): 0.75, (1, 1): 0.85},
    "S": {(0, 0): 0.05, (0, 1): 0.45, (1, 0): 0.65, (1, 1): 0.80},
    "V": {(0, 0): 0.03, (0, 1): 0.20, (1, 0): 0.40, (1, 1): 0.45},
    "R": {(0,): 0.05, (1,): 0.40},
    "T": {(0,): 0.15, (1,): 0.40},
}
PADRES = {"H": (), "A": (), "I": ("H",), "M": ("I", "A"), "P": ("I", "A"), "S": ("I", "A"), "V": ("I", "A"),
          "R": ("I",), "T": ("I",)}


class RedAlertas:
    """Red bayesiana de credibilidad de alertas, con inferencia exacta en pgmpy."""

    def __init__(self, tpc: dict | None = None):
        self.tpc = tpc or TPC
        self.modelo = self._construir()
        from pgmpy.inference import VariableElimination
        self.inf = VariableElimination(self.modelo)

    def _construir(self):
        try:
            from pgmpy.models import DiscreteBayesianNetwork as BN
        except ImportError:  # pgmpy < 1.0
            from pgmpy.models import BayesianNetwork as BN
        from pgmpy.factors.discrete import TabularCPD
        m = BN(ARISTAS)
        for nodo in NODOS:
            pad = PADRES[nodo]
            if not pad:
                p1 = self.tpc[nodo]
                cpd = TabularCPD(nodo, 2, [[1 - p1], [p1]])
            else:
                combos = list(itertools.product([0, 1], repeat=len(pad)))
                fila1 = [self.tpc[nodo][c] for c in combos]
                cpd = TabularCPD(nodo, 2, [[1 - v for v in fila1], fila1], evidence=list(pad), evidence_card=[2] * len(pad))
            m.add_cpds(cpd)
        if not m.check_model():
            raise ValueError("Red bayesiana inválida")
        return m

    def posterior(self, evidencia: dict[str, int]) -> float:
        ev = {k: int(v) for k, v in evidencia.items() if k in NODOS and k != "I"}
        q = self.inf.query(["I"], evidence=ev, show_progress=False)
        return float(q.values[1])

    def posterior_enumeracion(self, evidencia: dict[str, int]) -> float:
        """Verificación independiente por enumeración completa de la distribución conjunta."""
        return posterior_enumeracion(self.tpc, evidencia)

    def contribuciones(self, evidencia: dict[str, int]) -> list[tuple[str, float]]:
        """Cambio del posterior al retirar cada evidencia (explicación local tipo «dejar uno fuera»)."""
        base = self.posterior(evidencia)
        salida = []
        for k in evidencia:
            if k in ("H", "A"):
                continue
            resto = {kk: vv for kk, vv in evidencia.items() if kk != k}
            salida.append((k, base - self.posterior(resto)))
        return sorted(salida, key=lambda t: -abs(t[1]))


def posterior_enumeracion(tpc: dict, evidencia: dict[str, int]) -> float:
    num = den = 0.0
    libres = [n for n in NODOS if n not in evidencia]
    for valores in itertools.product([0, 1], repeat=len(libres)):
        asig = dict(evidencia)
        asig.update(dict(zip(libres, valores)))
        p = 1.0
        for nodo in NODOS:
            pad = PADRES[nodo]
            p1 = tpc[nodo] if not pad else tpc[nodo][tuple(asig[x] for x in pad)]
            p *= p1 if asig[nodo] == 1 else 1 - p1
        den += p
        if asig["I"] == 1:
            num += p
    return num / den


def sensibilidad(evidencia: dict[str, int], delta: float = 0.10) -> list[dict]:
    """Análisis de sensibilidad univariante: perturba ±delta cada parámetro de las TPC."""
    base = posterior_enumeracion(TPC, evidencia)
    filas = []
    for nodo, val in TPC.items():
        entradas = [((), val)] if not isinstance(val, dict) else list(val.items())
        for clave, p in entradas:
            extremos = []
            for signo in (-1, 1):
                tpc = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TPC.items()}
                nuevo = float(np.clip(p + signo * delta, 0.01, 0.99))
                if isinstance(val, dict):
                    tpc[nodo][clave] = nuevo
                else:
                    tpc[nodo] = nuevo
                extremos.append(posterior_enumeracion(tpc, evidencia))
            bajo, alto = extremos
            cond = ",".join(f"{pa}={v}" for pa, v in zip(PADRES[nodo], clave))
            filas.append({"parametro": f"P({nodo}=1 | {cond})" if clave else f"P({nodo}=1)",
                          "bajo": bajo, "alto": alto, "rango": abs(alto - bajo), "base": base})
    return sorted(filas, key=lambda f: -f["rango"])


# ====================================================================== factibilidad de vuelo


def evaluar_dia(dia: int, campamento: str, op: dict, urgente: bool) -> tuple[bool, list[str], dict]:
    pol = op["politica_vuelo"]
    viento, lluvia, vis = op["pronostico"][campamento][str(dia)]
    drones = [d for d, v in op["drones"].items() if str(dia) not in v["no_disponible"]]
    pilotos = [p for p, v in op["pilotos"].items() if str(dia) not in v["no_disponible"]]
    motivos = []
    if viento > pol["viento_max_ms"]:
        motivos.append(f"V01 viento {viento:.1f} m/s > {pol['viento_max_ms']:.0f} m/s".replace(".", ","))
    if lluvia > pol["lluvia_max_mmh"]:
        motivos.append(f"V02 lluvia {lluvia:.1f} mm/h".replace(".", ","))
    if vis < pol["visibilidad_min_km"]:
        motivos.append(f"V03 visibilidad {vis:.0f} km < {pol['visibilidad_min_km']:.0f} km")
    if not drones:
        motivos.append("V04 sin dron disponible")
    if not pilotos:
        motivos.append("V05 sin piloto disponible")
    if not urgente and dia < pol["anticipacion_min_dias"]:
        motivos.append("V06 anticipación menor a 1 día para solicitud no urgente")
    return (not motivos), motivos, {"viento": viento, "lluvia": lluvia, "visibilidad": vis,
                                    "dron": drones[0] if drones else None, "piloto": pilotos[0] if pilotos else None}


def primera_ventana(desde: int, campamento: str, op: dict, urgente: bool = False) -> dict:
    """Búsqueda secuencial de la primera ventana apta dentro del horizonte de pronóstico."""
    evaluados = []
    for d in range(max(0, desde), op["politica_vuelo"]["horizonte_dias"] + 1):
        apto, motivos, datos = evaluar_dia(d, campamento, op, urgente)
        evaluados.append({"dia": d, "apto": apto, "motivos": motivos, **datos})
        if apto:
            return {"dia": d, "evaluados": evaluados, **datos}
    return {"dia": None, "evaluados": evaluados}
