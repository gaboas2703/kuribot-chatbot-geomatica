"""Genera la base de conocimiento SINTÉTICA de KuriBot (concesiones, operación, FAQ, tickets).

Todo lo que se escribe aquí es ficticio: nombres, códigos, polígonos, clima, flota y tickets
se construyeron para el prototipo académico. Las respuestas de la FAQ geodésica se limitan a
hechos verificables en la base EPSG (vía pyproj) o a definiciones generales; las políticas
operativas (umbral de viento, plazos, formatos) son supuestos de diseño del equipo.

Uso: python datos/construir_base.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from kuribot.geodesia import descripcion_transformacion, psad56_utm_a_wgs84_utm  # noqa: E402

DATOS = RAIZ / "datos"


def rect(e0, n0, e1, n1):
    return [[e0, n0], [e1, n0], [e1, n1], [e0, n1], [e0, n0]]


CONCESIONES = [
    # codigo, nombre, fase, estado, campamento, polígono UTM17S WGS84, titular
    ("CM-101", "Peña Blanca", "explotación", "vigente", "Norte", rect(641000, 9661500, 643000, 9663000), "propia"),
    ("CM-102", "San Gregorio", "explotación", "vigente", "Norte", rect(643000, 9662000, 644600, 9663500), "propia"),
    ("CM-103", "La Ventana", "exploración avanzada", "vigente", "Central", rect(644600, 9660500, 647000, 9662500), "propia"),
    ("CM-104", "El Roble", "exploración inicial", "vigente (renovación en trámite)", "Central",
     [[640000, 9664000], [641000, 9664000], [641000, 9665400], [640600, 9666000], [640000, 9666000], [640000, 9664000]], "propia"),
    ("CM-105", "Cerro Azul", "explotación", "suspendida temporalmente", "Sur", rect(645500, 9663000, 646500, 9664200), "propia"),
    ("CM-106", "Los Laureles", "exploración inicial", "vigente", "Sur", rect(646500, 9658500, 648500, 9660500), "propia"),
    ("CT-201", "Minas del Valle", "explotación", "vigente", "—", rect(641000, 9663000, 643000, 9664200), "tercero"),
]
AREA_PROTEGIDA = ("BP-01", "Bosque Protector El Mirador (ficticio)", rect(647000, 9660500, 649000, 9662500))
COMUNIDADES = [("COM-01", "Comunidad San José (ficticia)", [644200, 9660800])]

OPERACION = {
    "fecha_referencia": "2026-11-09",
    "nota": "Datos operativos sintéticos para la demostración; no proceden de ninguna empresa.",
    "historial_alertas_90d": {"CM-101": 0, "CM-102": 1, "CM-103": 0, "CM-104": 2, "CM-105": 3, "CM-106": 1, "CT-201": 0},
    "actividad_autorizada_programada": {
        "CM-101": "frente de explotación activo con maquinaria propia",
        "CM-102": "mantenimiento de vías con retroexcavadora propia",
        "CM-103": "campaña de sondajes con perforadora contratista",
    },
    "pronostico": {  # día relativo a fecha_referencia: [viento m/s, lluvia mm/h, visibilidad km]
        "Norte":   {"0": [4.2, 0.0, 10], "1": [12.5, 0.0, 8], "2": [6.0, 0.0, 10], "3": [5.1, 2.4, 4], "4": [3.8, 0.0, 10]},
        "Central": {"0": [7.5, 1.2, 5], "1": [5.0, 0.0, 10], "2": [4.4, 0.0, 10], "3": [11.0, 0.0, 9], "4": [6.2, 0.0, 10]},
        "Sur":     {"0": [3.1, 0.0, 10], "1": [8.9, 0.0, 10], "2": [10.8, 0.5, 6], "3": [4.0, 0.0, 10], "4": [4.5, 0.0, 10]},
    },
    "drones": {"DRN-01": {"tipo": "multirrotor RTK", "no_disponible": ["2"]},
               "DRN-02": {"tipo": "ala fija VTOL", "no_disponible": ["0", "1"]}},
    "pilotos": {"Piloto A": {"no_disponible": ["3"]}, "Piloto B": {"no_disponible": ["0"]}},
    "politica_vuelo": {"viento_max_ms": 10.0, "lluvia_max_mmh": 0.0, "visibilidad_min_km": 5.0,
                       "anticipacion_min_dias": 1, "horizonte_dias": 4},
    "sla_cartografia_dias_habiles": {"estandar": 3, "urgente": 1},
}

TICKETS = [
    {"id": "GEO-2026-0412", "tipo": "cartografía", "detalle": "Mapa topográfico de CM-103", "estado": "En elaboración",
     "asignado": "Analista SIG 1", "compromiso": "2026-11-10"},
    {"id": "GEO-2026-0415", "tipo": "vuelo", "detalle": "Vuelo de inspección en CM-101", "estado": "Programado",
     "asignado": "Piloto A", "compromiso": "2026-11-10"},
    {"id": "GEO-2026-0419", "tipo": "alerta", "detalle": "Desbroce detectado en CM-105", "estado": "Cerrada",
     "asignado": "Seguridad patrimonial", "compromiso": "2026-11-03",
     "resolucion": "Verificada en campo: trabajos de remediación autorizados"},
    {"id": "GEO-2026-0423", "tipo": "cartografía", "detalle": "KMZ de linderos de CM-102", "estado": "Entregado",
     "asignado": "Analista SIG 2", "compromiso": "2026-11-05"},
    {"id": "GEO-2026-0430", "tipo": "alerta", "detalle": "Turbidez aguas abajo de CM-106", "estado": "En verificación de campo",
     "asignado": "Seguridad patrimonial", "compromiso": "2026-11-11"},
]


def faq() -> list[dict]:
    e, n = 642600.0, 9662480.0
    e2, n2 = psad56_utm_a_wgs84_utm(e, n)
    de, dn = e2 - e, n2 - n
    d = (de ** 2 + dn ** 2) ** 0.5
    desplaz = (f"En el área del proyecto, la transformación {descripcion_transformacion()} desplaza un punto "
               f"aproximadamente {d:.0f} m (ΔE ≈ {de:.0f} m, ΔN ≈ {dn:.0f} m al pasar de PSAD56 a WGS 84).")
    return [
        {"id": "F01", "tema": "datum", "preguntas": ["diferencia entre PSAD56 y WGS84", "psad56 vs wgs84", "qué es PSAD56"],
         "respuesta": "PSAD56 (Datum Provisional Sudamericano de 1956) es un datum local basado en el elipsoide Internacional "
                      "de 1924 con punto fundamental en La Canoa (Venezuela); WGS 84 es un sistema geocéntrico global. Por eso "
                      "un mismo punto tiene coordenadas distintas en cada uno. " + desplaz, "fuente": "Base EPSG (pyproj)"},
        {"id": "F02", "tema": "epsg", "preguntas": ["código EPSG UTM 17 sur", "EPSG de PSAD56 UTM 17S", "EPSG WGS84 geográficas",
                                                      "qué EPSG uso"],
         "respuesta": "Códigos de uso frecuente en el proyecto: EPSG:4326 (WGS 84 geográficas), EPSG:32717 (WGS 84 / UTM zona 17S), "
                      "EPSG:24877 (PSAD56 / UTM zona 17S), EPSG:4248 (PSAD56 geográficas) y EPSG:31977 (SIRGAS 2000 / UTM zona 17S).",
         "fuente": "Base EPSG (pyproj)"},
        {"id": "F03", "tema": "marco_oficial", "preguntas": ["datum oficial en Ecuador", "sistema de referencia oficial del país",
                                                              "marco de referencia del IGM"],
         "respuesta": "El marco geodésico nacional es SIRGAS-ECU, la densificación de SIRGAS en el Ecuador a cargo del Instituto "
                      "Geográfico Militar (IGM). La realización y la época vigentes deben tomarse de la resolución vigente del IGM; "
                      "consulte con el analista de geodesia antes de entregar información con fines legales.",
         "fuente": "Política interna (verificar normativa IGM vigente)"},
        {"id": "F04", "tema": "sirgas", "preguntas": ["qué es SIRGAS", "época de referencia en SIRGAS"],
         "respuesta": "SIRGAS (Sistema de Referencia Geocéntrico para las Américas) es la densificación regional del Marco "
                      "Internacional de Referencia Terrestre (ITRF). Como las placas tectónicas se mueven, las coordenadas SIRGAS se "
                      "refieren a una época determinada; comparar coordenadas de épocas distintas exige modelar ese movimiento.",
         "fuente": "Definición general"},
        {"id": "F05", "tema": "zona_utm", "preguntas": ["en qué zona UTM está el proyecto", "qué es la zona 17 sur"],
         "respuesta": "El área del proyecto (alrededor de 79,7° O) está en la zona UTM 17 (de 84° O a 78° O, meridiano central 81° O), "
                      "hemisferio sur. Al este de 78° O comienza la zona 18, por lo que parte de la Amazonía cae en 18S.",
         "fuente": "Definición del sistema UTM"},
        {"id": "F06", "tema": "no_coinciden", "preguntas": ["mis coordenadas no coinciden con el catastro", "el punto sale corrido en el mapa"],
         "respuesta": "Las causas más comunes son: (1) datum distinto, PSAD56 frente a WGS 84 (en esta zona la diferencia es de unos "
                      "cientos de metros); (2) zona UTM equivocada; (3) Este y Norte invertidos; (4) error propio del receptor GNSS. "
                      "Indique el datum de origen y puedo convertir o verificar el punto.", "fuente": "Práctica geodésica"},
        {"id": "F07", "tema": "precision_gnss", "preguntas": ["precisión de un GPS navegador", "diferencia entre RTK y GPS de mano", "qué es RTK"],
         "respuesta": "Un receptor de navegación suele ofrecer exactitudes horizontales del orden de unos metros, y empeora bajo "
                      "cobertura vegetal. El posicionamiento RTK usa correcciones de una base en tiempo real y, con solución fija, "
                      "alcanza exactitud del orden de centímetros. Para linderos y control use GNSS geodésico.",
         "fuente": "Práctica geodésica"},
        {"id": "F08", "tema": "formatos", "preguntas": ["formato de archivo para entregar", "en qué formato entrego la cartografía"],
         "respuesta": "Política interna (supuesto del prototipo): entregar vectores en GeoPackage o Shapefile en EPSG:32717, ráster en "
                      "GeoTIFF, KMZ solo para visualización y mapas en PDF con cuadrícula, escala y metadatos.", "fuente": "Política interna"},
        {"id": "F09", "tema": "politica_vuelo", "preguntas": ["política para vuelos de dron", "reglas para volar dron",
                                                               "anticipación para pedir un vuelo"],
         "respuesta": "Política interna (supuesto del prototipo): vuelos diurnos con línea de vista, viento ≤ 10 m/s, sin lluvia, "
                      "visibilidad ≥ 5 km, piloto habilitado y solicitud con al menos un día de anticipación, salvo alertas de prioridad "
                      "CRÍTICA o ALTA. Toda operación debe cumplir la normativa aeronáutica vigente.", "fuente": "Política interna"},
        {"id": "F10", "tema": "desplazamiento", "preguntas": ["cuánto se desplaza un punto de PSAD56 a WGS84",
                                                               "cuántos metros cambia de PSAD a WGS"],
         "respuesta": desplaz, "fuente": "Cálculo con pyproj"},
        {"id": "F11", "tema": "utm", "preguntas": ["qué es una proyección UTM", "qué es el falso norte"],
         "respuesta": "UTM es una proyección transversa de Mercator dividida en 60 zonas de 6°. Usa factor de escala 0,9996 en el "
                      "meridiano central, falso este de 500 000 m y, en el hemisferio sur, falso norte de 10 000 000 m para evitar "
                      "coordenadas negativas.", "fuente": "Definición del sistema UTM"},
        {"id": "F12", "tema": "datum_def", "preguntas": ["qué es un datum geodésico", "qué es un datum"],
         "respuesta": "Un datum geodésico define la forma y el tamaño de la Tierra (un elipsoide) y su posición respecto del planeta; "
                      "sobre él se expresan las coordenadas. Cambiar de datum cambia los números aunque el punto físico sea el mismo.",
         "fuente": "Definición general"},
        {"id": "F13", "tema": "reporte_alerta", "preguntas": ["información para reportar una alerta", "datos de un reporte de alerta"],
         "respuesta": "Indique la concesión o las coordenadas (con su datum), qué observó (maquinaria, desbroce, agua turbia, ruido "
                      "nocturno, vía nueva), la fecha y la fuente. No incluya datos personales de terceros. Si hay riesgo para "
                      "personas, escriba «emergencia».", "fuente": "Procedimiento interno"},
        {"id": "F14", "tema": "sla_mapa", "preguntas": ["cuánto tarda una solicitud de mapa", "cuánto se demora un mapa"],
         "respuesta": "Plazo interno (supuesto del prototipo): 3 días hábiles para solicitudes estándar y 1 día hábil para urgentes "
                      "aprobadas por la jefatura.", "fuente": "Política interna"},
        {"id": "F15", "tema": "epsg_def", "preguntas": ["qué significa EPSG", "qué es el código EPSG"],
         "respuesta": "EPSG es la base de datos de parámetros geodésicos creada por el European Petroleum Survey Group y mantenida hoy "
                      "por el Comité de Geomática de IOGP. Cada sistema de referencia y cada transformación tiene un código único.",
         "fuente": "IOGP"},
        {"id": "F16", "tema": "ortofoto", "preguntas": ["qué es una ortofoto", "para qué sirve una ortofoto"],
         "respuesta": "Una ortofoto es una imagen aérea corregida por el relieve y la inclinación de la cámara, de modo que tiene "
                      "escala uniforme y permite medir distancias y áreas.", "fuente": "Definición general"},
        {"id": "F17", "tema": "mdt", "preguntas": ["qué es un modelo digital de terreno"],
         "respuesta": "Un modelo digital del terreno (MDT) representa la elevación del suelo desnudo; el modelo digital de superficie "
                      "incluye además vegetación y construcciones.", "fuente": "Definición general"},
        {"id": "F18", "tema": "alturas", "preguntas": ["diferencia entre altura elipsoidal y ortométrica",
                                                        "altura del gps y altura sobre el nivel del mar"],
         "respuesta": "El GNSS entrega altura elipsoidal (h). La altura ortométrica (H), cercana a la altura sobre el nivel medio del "
                      "mar, se obtiene con la ondulación del geoide (N): H = h − N, usando un modelo geoidal.", "fuente": "Definición general"},
        {"id": "F19", "tema": "identificar_datum", "preguntas": ["cómo sé en qué datum están mis coordenadas", "cómo identifico el datum"],
         "respuesta": "Revise los metadatos o la configuración del equipo. Si no existen, compare con un punto de control conocido: en "
                      "esta zona, una diferencia sistemática de unos 440 m respecto del catastro en WGS 84 sugiere coordenadas en PSAD56.",
         "fuente": "Práctica geodésica"},
    ]


def main() -> None:
    feats = []
    for cod, nom, fase, est, camp, pol, tit in CONCESIONES:
        feats.append({"type": "Feature", "properties": {"codigo": cod, "nombre": nom, "fase": fase, "estado": est,
                                                         "campamento": camp, "titular": tit, "capa": "concesion"},
                      "geometry": {"type": "Polygon", "coordinates": [pol]}})
    cod, nom, pol = AREA_PROTEGIDA
    feats.append({"type": "Feature", "properties": {"codigo": cod, "nombre": nom, "capa": "area_protegida"},
                  "geometry": {"type": "Polygon", "coordinates": [pol]}})
    for cod, nom, pt in COMUNIDADES:
        feats.append({"type": "Feature", "properties": {"codigo": cod, "nombre": nom, "capa": "comunidad"},
                      "geometry": {"type": "Point", "coordinates": pt}})
    geo = {"type": "FeatureCollection", "name": "base_sintetica_kuribot",
           "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32717"}}, "features": feats}
    (DATOS / "concesiones.geojson").write_text(json.dumps(geo, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATOS / "operacion.json").write_text(json.dumps(OPERACION, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATOS / "faq.json").write_text(json.dumps(faq(), ensure_ascii=False, indent=2), encoding="utf-8")
    (DATOS / "tickets_semilla.json").write_text(json.dumps(TICKETS, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Base sintética escrita en", DATOS)


if __name__ == "__main__":
    main()
