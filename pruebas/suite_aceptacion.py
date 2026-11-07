"""Suite de aceptación funcional de KuriBot: 34 escenarios (43 turnos) con criterios explícitos.

Cada escenario empieza con un estado de diálogo limpio. Se evalúa el ÚLTIMO turno de cada
escenario (los turnos previos preparan el contexto) y se verifica, además, cada turno
intermedio que deba pedir un dato. Criterios de aprobación por turno evaluado:
  (a) intención esperada (o conjunto aceptable), (b) acción o regla esperada,
  (c) fragmentos de contenido obligatorios, (d) latencia < 1 500 ms.

Además, CONFIRMACION contiene 8 escenarios redactados DESPUÉS de cerrar la iteración 2 y ejecutados una sola
vez, sin ajustes posteriores, para estimar el desempeño en casos no usados durante el desarrollo.

Uso: python pruebas/suite_aceptacion.py --version v1 | --version v2 [--confirmacion]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from kuribot.conocimiento import BaseConocimiento  # noqa: E402
from kuribot.dialogo import KuriBot  # noqa: E402
from kuribot.nlu import ClasificadorIntenciones  # noqa: E402

# (id, descripción, [turnos], criterios del último turno, criterios de turnos intermedios {indice: dict})
ESCENARIOS = [
    ("A01", "Saludo y aviso de transparencia", ["buenos días, ¿con quién hablo?"],
     {"intencion": ["saludo"], "contiene": ["asistente automatizado"]}),
    ("A02", "Capacidades", ["¿qué cosas puedes resolver?"], {"intencion": ["ayuda_capacidades"], "contiene": ["consultar una concesión"]}),
    ("A03", "Ficha de concesión por nombre", ["dame la ficha de la concesión La Ventana"],
     {"intencion": ["consultar_concesion"], "contiene": ["CM-103", "480"]}),
    ("A04", "Estado de concesión suspendida", ["¿cerro azul está activa?"], {"intencion": ["consultar_concesion"], "contiene": ["suspendida"]}),
    ("A05", "Ranura faltante de concesión", ["quiero consultar una concesión", "CM-106"],
     {"intencion": ["consultar_concesion"], "contiene": ["Los Laureles"]}, {0: {"accion": "solicitar_dato:concesion"}}),
    ("A06", "Punto WGS84 dentro de concesión", ["¿el punto E 642350 N 9662120 WGS84 está dentro de la concesión?"],
     {"intencion": ["verificar_punto"], "contiene": ["DENTRO de CM-101"], "reglas": ["R10"]}),
    ("A07", "Punto sin datum: se pide y luego cerca de comunidad", ["verifica si 644800 9661500 cae en nuestra área", "wgs84"],
     {"intencion": ["verificar_punto"], "contiene": ["DENTRO de CM-103", "comunidad"], "reglas": ["R14"]},
     {0: {"accion": "solicitar_dato:datum"}}),
    ("A08", "Punto PSAD56 transformado antes del cruce", ["¿E 642600 N 9662480 PSAD56 en qué concesión queda?"],
     {"intencion": ["verificar_punto"], "contiene": ["Transformé", "DENTRO de CM-101"]}),
    ("A09", "Punto en franja de amortiguamiento", ["chequea 641100 9664950 wgs84 contra el catastro"],
     {"intencion": ["verificar_punto"], "contiene": ["franja de 500 m"], "reglas": ["R11"]}),
    ("A10", "Punto en concesión de tercero", ["¿la coordenada 642000 9663600 wgs84 es nuestra?"],
     {"intencion": ["verificar_punto"], "contiene": ["tercero"], "reglas": ["R12"]}),
    ("A11", "Conversión PSAD56 -> WGS84", ["convierte E 642600 N 9662480 de PSAD56 a WGS84"],
     {"intencion": ["convertir_coordenadas"], "contiene": ["EPSG:32717", "642 348"]}),
    ("A12", "Conversión geográficas -> UTM", ["pasa -3.0512, -79.7301 a utm, está en wgs84"],
     {"intencion": ["convertir_coordenadas"], "contiene": ["E 641 120"]}),
    ("A13", "Conversión con dos ranuras", ["necesito convertir unas coordenadas", "645210 9663790", "psad56"],
     {"intencion": ["convertir_coordenadas"], "contiene": ["Origen: PSAD56"]},
     {0: {"accion": "solicitar_dato:coord"}, 1: {"accion": "solicitar_dato:datum"}}),
    ("A14", "Alerta con maquinaria y personas en concesión suspendida", ["hay una retroexcavadora y gente extraña trabajando en Cerro Azul"],
     {"intencion": ["reportar_alerta"], "contiene": ["prioridad CRÍTICA"], "reglas": ["R21", "R41", "R42"]}),
    ("A15", "Alerta explicada por actividad autorizada (explaining away)", ["se ve desbroce nuevo y maquinaria en Peña Blanca"],
     {"intencion": ["reportar_alerta"], "contiene": ["prioridad MEDIA", "actividad autorizada programada explica"], "reglas": ["R23"]}),
    ("A16", "Alerta en tres turnos con ranuras", ["quiero reportar una invasión", "en Los Laureles", "el agua baja turbia"],
     {"intencion": ["reportar_alerta"], "contiene": ["prioridad ALTA", "supervisor"], "reglas": ["R22", "R41"]},
     {0: {"accion": "solicitar_dato:ubicacion"}, 1: {"accion": "solicitar_dato:evidencias"}}),
    ("A17", "Alerta en concesión de tercero", ["vimos maquinaria en E 642000 N 9663600 WGS84"],
     {"intencion": ["reportar_alerta"], "contiene": ["REVISIÓN LEGAL"], "reglas": ["R20", "R35"]}),
    ("A18", "Alerta crítica con ventana de vuelo al día siguiente", ["reporto voladuras de noche y una trocha nueva en El Roble"],
     {"intencion": ["reportar_alerta"], "contiene": ["prioridad CRÍTICA", "Vuelo de verificación prioritario"], "reglas": ["R30"]}),
    ("A19", "Alerta con negación y zona sensible", ["hay maquinaria en La Ventana, pero no hay personas"],
     {"intencion": ["reportar_alerta"], "contiene": ["prioridad BAJA", "revisión humana"], "reglas": ["R24", "R40"]}),
    ("A20", "Emergencia con armas", ["hay gente armada disparando cerca del campamento"],
     {"intencion": ["emergencia_seguridad"], "contiene": ["ECU 911", "No se enviará personal ni dron"], "reglas": ["R01"]}),
    ("A21", "Vuelo con fecha no apta por viento", ["necesito un vuelo de dron en San Gregorio", "mañana"],
     {"intencion": ["solicitar_vuelo"], "contiene": ["no es apta", "viento", "2026-11-11"]}, {0: {"accion": "solicitar_dato:fecha"}}),
    ("A22", "Vuelo el mismo día sin urgencia (anticipación)", ["programa un vuelo en Los Laureles para hoy"],
     {"intencion": ["solicitar_vuelo"], "contiene": ["anticipación", "2026-11-10"]}),
    ("A23", "Solicitud de KMZ", ["necesito el KMZ de los linderos de CM-102"],
     {"intencion": ["solicitar_cartografia"], "contiene": ["KMZ", "3 día"], "accion": "crear_ticket"}),
    ("A24", "Cartografía con ranuras", ["quiero pedir cartografía", "una ortofoto", "de El Roble"],
     {"intencion": ["solicitar_cartografia"], "contiene": ["ortofoto de CM-104"]},
     {0: {"accion": "solicitar_dato:producto"}, 1: {"accion": "solicitar_dato:concesion"}}),
    ("A25", "Estado de ticket existente", ["¿cómo va la solicitud GEO-2026-0412?"],
     {"intencion": ["estado_solicitud"], "contiene": ["En elaboración"]}),
    ("A26", "Ticket inexistente (no inventar)", ["estado del ticket GEO-2026-0999"],
     {"intencion": ["estado_solicitud"], "contiene": ["No encuentro"]}),
    ("A27", "FAQ de códigos EPSG", ["¿qué EPSG corresponde a PSAD56 en UTM 17 sur?"],
     {"intencion": ["consulta_geodesia"], "contiene": ["24877"]}),
    ("A28", "FAQ de desplazamiento entre datums", ["¿cuántos metros se mueve un punto entre PSAD56 y WGS84?"],
     {"intencion": ["consulta_geodesia"], "contiene": ["443 m"]}),
    ("A29", "Derivación a humano", ["quiero hablar con un analista de verdad"],
     {"intencion": ["hablar_humano"], "reglas": ["R03"], "accion": "derivar_humano"}),
    ("A30", "Fuera de alcance", ["¿quién ganó la Copa Libertadores?"],
     {"intencion": ["fuera_alcance"], "contiene_alguno": ["fuera de mi alcance", "No tengo suficiente seguridad"]}),
    ("A31", "Dato personal enmascarado", ["mi cédula es 3000000004, ¿cuál es el estado de CM-104?"],
     {"intencion": ["consultar_concesion"], "contiene": ["renovación"], "reglas": ["R02"]}),
    ("A32", "Mensaje con faltas ortográficas", ["ola, nesesito saver el estado de la consecion el robel"],
     {"intencion": ["consultar_concesion"], "contiene": ["El Roble"]}),
    ("A33", "Despedida", ["gracias, eso es todo"], {"intencion": ["despedida"]}),
    ("A34", "Coordenadas sin verbo (mensaje ambiguo)", ["642350 9662120"],
     {"intencion": ["verificar_punto", "convertir_coordenadas"], "accion_alguna": ["solicitar_dato:datum", None]}),
]


CONFIRMACION = [
    ("C01", "Alerta con coordenadas y dos evidencias", ["reporto maquinaria y una trocha nueva en E 646900 N 9660300 WGS84"],
     {"intencion": ["reportar_alerta"], "contiene": ["prioridad"]}),
    ("C02", "Teléfono enmascarado en una solicitud de cartografía",
     ["llámame al 0991234567 para coordinar, necesito un plano topográfico de CM-106"],
     {"intencion": ["solicitar_cartografia"], "reglas": ["R02"], "accion": "crear_ticket"}),
    ("C03", "Emergencia sin la palabra «emergencia»", ["auxilio, hay heridos en la vía al campamento sur"],
     {"intencion": ["emergencia_seguridad"], "reglas": ["R01"]}),
    ("C04", "Conversión desde grados, minutos y segundos", ["pasa 3°03'04\"S 79°43'48\"W a UTM, datum WGS84"],
     {"intencion": ["convertir_coordenadas"], "contiene": ["E 641"]}),
    ("C05", "Vuelo con día de la semana", ["agenda un vuelo sobre La Ventana el viernes"],
     {"intencion": ["solicitar_vuelo"], "contiene": ["2026-11-13"]}),
    ("C06", "FAQ formulada de otra manera", ["¿qué ventaja tiene el RTK frente al GPS navegador?"],
     {"intencion": ["consulta_geodesia"], "contiene": ["centímetros"]}),
    ("C07", "Clima fuera del contexto de vuelos", ["¿va a llover mañana en Machala?"],
     {"intencion": ["fuera_alcance"], "contiene_alguno": ["fuera de mi alcance", "No tengo suficiente seguridad"]}),
    ("C08", "Estado de ticket en registro informal", ["porfa el estatus del GEO-2026-0430"],
     {"intencion": ["estado_solicitud"], "contiene": ["En verificación de campo"]}),
]


def evaluar_turno(r, crit: dict) -> tuple[bool, list[str]]:
    fallos = []
    if "intencion" in crit and r.intencion not in crit["intencion"]:
        fallos.append(f"intención {r.intencion} ∉ {crit['intencion']}")
    if "accion" in crit and r.accion != crit["accion"]:
        fallos.append(f"acción {r.accion} ≠ {crit['accion']}")
    if "accion_alguna" in crit and r.accion not in crit["accion_alguna"] and r.decision != "aclarar":
        fallos.append(f"acción {r.accion} ∉ {crit['accion_alguna']}")
    for frag in crit.get("contiene", []):
        if frag.lower() not in r.texto.lower():
            fallos.append(f"falta «{frag}»")
    if "contiene_alguno" in crit and not any(f.lower() in r.texto.lower() for f in crit["contiene_alguno"]):
        fallos.append("no contiene ninguno de los fragmentos esperados")
    for regla in crit.get("reglas", []):
        if regla not in r.reglas:
            fallos.append(f"no disparó {regla}")
    if r.latencia_ms >= 1500:
        fallos.append(f"latencia {r.latencia_ms:.0f} ms")
    return not fallos, fallos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--modelo", default=None, help="Carpeta del modelo NLU (por defecto, el más reciente)")
    ap.add_argument("--confirmacion", action="store_true", help="Ejecuta solo los 8 escenarios de confirmación")
    args = ap.parse_args()
    nlu = ClasificadorIntenciones(args.modelo)
    filas, aprobados, turnos_total = [], 0, 0
    for esc in (CONFIRMACION if args.confirmacion else ESCENARIOS):
        ident, desc, turnos, crit = esc[:4]
        inter = esc[4] if len(esc) > 4 else {}
        bot = KuriBot(nlu, BaseConocimiento())
        detalle, ok_esc = [], True
        for i, t in enumerate(turnos):
            r = bot.responder(t)
            turnos_total += 1
            c = crit if i == len(turnos) - 1 else inter.get(i)
            ok, fallos = evaluar_turno(r, c) if c else (True, [])
            ok_esc &= ok
            detalle.append({"turno": i + 1, "entrada": t, "respuesta": r.texto, "intencion": r.intencion,
                            "confianza": round(r.confianza or 0, 4), "decision": r.decision, "accion": r.accion,
                            "reglas": r.reglas, "p_intrusion": (r.bayes or {}).get("p_intrusion"), "ticket": r.ticket,
                            "latencia_ms": round(r.latencia_ms, 1), "evaluado": bool(c), "ok": ok, "fallos": fallos})
        aprobados += ok_esc
        filas.append({"id": ident, "descripcion": desc, "aprobado": ok_esc, "turnos": detalle})
        estado = "OK " if ok_esc else "FALLA"
        print(f"{estado} {ident} {desc}" + ("" if ok_esc else "  -> " + " | ".join(f for d in detalle for f in d["fallos"])))
    lat = sorted(d["latencia_ms"] for f in filas for d in f["turnos"])
    resumen = {"version": args.version, "modelo": nlu.nombre, "escenarios": len(filas), "aprobados": aprobados,
               "tasa": aprobados / len(filas), "turnos": turnos_total,
               "latencia_media_ms": sum(lat) / len(lat), "latencia_p95_ms": lat[int(0.95 * (len(lat) - 1))]}
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    salida = RAIZ / "resultados" / (f"confirmacion_{args.version}.json" if args.confirmacion else f"aceptacion_{args.version}.json")
    salida.write_text(json.dumps({"resumen": resumen, "escenarios": filas}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
