"""Extracción de entidades basada en reglas (expresiones regulares + diccionario geográfico).

La intención la decide la red neuronal; los valores concretos (coordenadas, códigos, fechas,
evidencias) los extrae este módulo de forma determinista y auditable. También detecta datos
personales para enmascararlos antes de registrar la conversación.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

from .geodesia import Punto

# ------------------------------------------------------------------ utilidades


def plano(texto: str) -> str:
    """Minúsculas y sin tildes (conserva la ñ) para comparar con diccionarios."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn" or c == "̃")
    return unicodedata.normalize("NFC", t)


def _num(s: str) -> float:
    return float(s.replace(",", "."))


# ------------------------------------------------------------------ coordenadas

_UTM_ETIQ = re.compile(r"(?:\b[ex]|este)\s*[:=]?\s*(\d{6}(?:\.\d+)?)\s*[,;]?\s*(?:\b[ny]|norte)\s*[:=]?\s*(\d{7}(?:\.\d+)?)", re.I)
_UTM_PAR = re.compile(r"(?<![\d.])(\d{6}(?:\.\d+)?)\s*[,;]?\s+(\d{7}(?:\.\d+)?)(?![\d])|(?<![\d.])(\d{6}(?:\.\d+)?),(\d{7}(?:\.\d+)?)(?![\d])")
_DEC = re.compile(r"(?:lat(?:itud)?\s*[:=]?\s*)?(-?\d{1,2}\.\d{3,})\s*[,;]?\s+(?:lon(?:gitud)?\s*[:=]?\s*)?(-?\d{2,3}\.\d{3,})|"
                  r"(-?\d{1,2}\.\d{3,}),\s*(-?\d{2,3}\.\d{3,})", re.I)
_DMS = re.compile(r"(\d{1,2})\s*°\s*(\d{1,2})\s*['′]\s*(\d{1,2}(?:[.,]\d+)?)\s*(?:\"|″|'')?\s*([NS])\s*[,;]?\s*"
                  r"(\d{2,3})\s*°\s*(\d{1,2})\s*['′]\s*(\d{1,2}(?:[.,]\d+)?)\s*(?:\"|″|'')?\s*([EWO])", re.I)
_DATUM = [("PSAD56", re.compile(r"psad\s*-?\s*56|psad\b", re.I)),
          ("WGS84", re.compile(r"wgs\s*-?\s*84|wgs\b", re.I)),
          ("SIRGAS", re.compile(r"sirgas", re.I))]


@dataclass
class Coordenada:
    punto: Punto
    texto: str
    inicio: int
    fin: int
    avisos: list[str] = field(default_factory=list)


def extraer_datum(texto: str) -> str | None:
    """Devuelve el datum mencionado. Si hay dos, toma el primero (origen) — el destino lo decide el diálogo."""
    hallados = sorted(((m.start(), nombre) for nombre, rx in _DATUM for m in rx.finditer(texto)))
    return hallados[0][1] if hallados else None


def extraer_coordenada(texto: str) -> Coordenada | None:
    datum = extraer_datum(texto)
    m = _UTM_ETIQ.search(texto)
    if m:
        return Coordenada(Punto(_num(m.group(1)), _num(m.group(2)), "utm", datum), m.group(0), m.start(), m.end())
    m = _DMS.search(texto)
    if m:
        lat = int(m.group(1)) + int(m.group(2)) / 60 + _num(m.group(3)) / 3600
        lon = int(m.group(5)) + int(m.group(6)) / 60 + _num(m.group(7)) / 3600
        lat = -lat if m.group(4).upper() == "S" else lat
        lon = -lon if m.group(8).upper() in ("W", "O") else lon
        return Coordenada(Punto(lon, lat, "geo", datum), m.group(0), m.start(), m.end())
    m = _DEC.search(texto)
    if m:
        a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        lat, lon = _num(a), _num(b)
        avisos = []
        if lon > 0 and 75 <= lon <= 92:
            lon = -lon
            avisos.append("Longitud sin signo: se asumió Oeste.")
        return Coordenada(Punto(lon, lat, "geo", datum), m.group(0), m.start(), m.end(), avisos)
    m = _UTM_PAR.search(texto)
    if m:
        e, n = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return Coordenada(Punto(_num(e), _num(n), "utm", datum), m.group(0), m.start(), m.end())
    return None


# ------------------------------------------------------------------ tickets, concesiones, fechas, productos

_TICKET = re.compile(r"\bgeo[-\s]?(\d{4})[-\s]?(\d{4})\b", re.I)
_CODIGO = re.compile(r"\b(cm|ct)\s*-?\s*(\d{3})\b", re.I)


def extraer_ticket(texto: str) -> str | None:
    m = _TICKET.search(texto)
    return f"GEO-{m.group(1)}-{m.group(2)}" if m else None


def extraer_concesion(texto: str, nombres: dict[str, str]) -> str | None:
    """nombres: {nombre_plano: codigo}. Acepta código (CM-101) o nombre con tolerancia a erratas."""
    m = _CODIGO.search(texto)
    if m:
        return f"{m.group(1).upper()}-{m.group(2)}"
    t = plano(texto)
    for nombre, codigo in nombres.items():
        if re.search(rf"\b{re.escape(nombre)}\b", t):
            return codigo
    palabras = re.findall(r"[a-zñ]+", t)
    mejor, puntaje = None, 0.0
    for nombre, codigo in nombres.items():
        k = len(nombre.split())
        for i in range(len(palabras) - k + 1):
            r = difflib.SequenceMatcher(None, " ".join(palabras[i:i + k]), nombre).ratio()
            if r > puntaje:
                mejor, puntaje = codigo, r
    return mejor if puntaje >= 0.86 else None


_DIAS = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3, "viernes": 4, "sabado": 5, "domingo": 6}
_MESES = {m: i + 1 for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                                           "septiembre", "octubre", "noviembre", "diciembre"])}


def extraer_fecha(texto: str, hoy) -> int | None:
    """Devuelve el desfase en días respecto de 'hoy' (date) o None."""
    import datetime as dt
    t = plano(texto)
    if "pasado mañana" in t or "pasado manana" in t:
        return 2
    if re.search(r"\bmañana\b|\bmanana\b", t):
        return 1
    if re.search(r"\bhoy\b|\bahora\b|\burgente\b", t):
        return 0
    for nombre, wd in _DIAS.items():
        if re.search(rf"\b{nombre}\b", t):
            return (wd - hoy.weekday()) % 7 or 7
    m = re.search(r"\b(\d{1,2})\s+de\s+(" + "|".join(_MESES) + r")\b", t)
    if m:
        try:
            f = dt.date(hoy.year, _MESES[m.group(2)], int(m.group(1)))
            return (f - hoy).days
        except ValueError:
            return None
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        try:
            f = dt.date(hoy.year, int(m.group(2)), int(m.group(1)))
            return (f - hoy).days
        except ValueError:
            return None
    if "esta semana" in t:
        return 1
    return None


_PRODUCTOS = [("ortofoto", r"ortofoto|orto\b"), ("modelo digital de elevación", r"modelo digital|\bmde\b|\bmdt\b|\bdem\b|curvas de nivel"),
              ("KMZ/KML", r"\bkmz\b|\bkml\b"), ("shapefile", r"shape|shp\b|geopackage|\bgpkg\b|capa"),
              ("plano topográfico", r"topograf|plano"), ("mapa temático", r"tematic|cobertura|pendiente"),
              ("mapa de ubicación", r"mapa")]


def extraer_producto(texto: str) -> str | None:
    t = plano(texto)
    for nombre, rx in _PRODUCTOS:
        if re.search(rx, t):
            return nombre
    return None


# ------------------------------------------------------------------ evidencias de alerta (para la red bayesiana)

EVIDENCIAS = {
    "M": ("maquinaria pesada o equipos", r"maquinaria|excavadora|retroexcavadora|volqueta|clasificadora|draga|bomba|"
                                         r"motobomba|manguera|tractor|perforadora|maquina"),
    "S": ("alteración del terreno o la cobertura", r"desbroce|deforest|\btala\b|claro nuevo|cambio de cobertura|"
                                                    r"movimiento de tierra|hueco|\bexcava(?!dor)|labores|bocamina|socavon|imagen|sentinel|satelit"),
    "P": ("personas o campamento no identificados", r"gente|personas|mineros|informales|campamento|carpas"),
    "R": ("ruido o voladuras nocturnas", r"ruido|voladura|explosion|dinamita|de noche|madrugada|nocturn"),
    "T": ("agua turbia aguas abajo", r"turbi|lodo|agua sucia|sediment"),
    "V": ("vía o trocha nueva", r"trocha|camino nuevo|camino abierto|aparecio un camino|via nueva|acceso nuevo|sendero nuevo"),
}
_NEGACION = re.compile(r"\b(no|sin|ningun|ninguna|ninguno|nada de|tampoco|ni)\b")


def extraer_evidencias(texto: str) -> dict[str, bool]:
    """Evidencias observadas: True si se afirma, False si se niega explícitamente. Ausentes = no observadas."""
    t = plano(texto)
    salida: dict[str, bool] = {}
    for clave, (_, rx) in EVIDENCIAS.items():
        for m in re.finditer(rx, t):
            previo = " ".join(t[:m.start()].split()[-3:])
            salida[clave] = not bool(_NEGACION.search(previo))
            break
    return salida


# ------------------------------------------------------------------ seguridad y datos personales

_EMERGENCIA = re.compile(r"armad[oa]s?\b|\barmas\b|disparo|dispararon|\btiros\b|balacera|herid[oa]|lesionad|secuestr|"
                         r"retuvieron|retenid|amenaz|atrapad|incendio|agred|agresi|asalt|desaparecid|violencia|"
                         r"enfrentamiento|atacaron|auxilio|encapuchad")


def hay_emergencia(texto: str) -> bool:
    return bool(_EMERGENCIA.search(plano(texto)))


def cedula_valida(d: str) -> bool:
    """Validación del dígito verificador de la cédula ecuatoriana (módulo 10)."""
    if len(d) != 10 or not d.isdigit():
        return False
    prov, tercero = int(d[:2]), int(d[2])
    if not (1 <= prov <= 24 or prov == 30) or tercero >= 6:
        return False
    suma = 0
    for i, c in enumerate(d[:9]):
        v = int(c) * (2 if i % 2 == 0 else 1)
        suma += v - 9 if v > 9 else v
    return (10 - suma % 10) % 10 == int(d[9])


_DIEZ = re.compile(r"(?<!\d)(\d{10})(?!\d)")
_TELEFONO = re.compile(r"(?<!\d)(?:\+?593\s?|0)9\d{8}(?!\d)")
_CORREO = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def enmascarar_pii(texto: str) -> tuple[str, list[str]]:
    tipos: list[str] = []

    def _ced(m):
        if cedula_valida(m.group(1)):
            tipos.append("cédula")
            return "[CÉDULA]"
        return m.group(0)

    t = _DIEZ.sub(_ced, texto)
    t = _TELEFONO.sub(lambda m: (tipos.append("teléfono"), "[TELÉFONO]")[1], t)
    t = _CORREO.sub(lambda m: (tipos.append("correo"), "[CORREO]")[1], t)
    return t, tipos


# ------------------------------------------------------------------ enmascarado para el clasificador


def enmascarar_entidades(texto: str) -> str:
    """Sustituye valores por su tipo antes de clasificar la intención (mismo paso para todos los modelos)."""
    t = texto
    c = extraer_coordenada(t)
    while c is not None:
        t = t[:c.inicio] + " COORDENADA " + t[c.fin:]
        c = extraer_coordenada(t)
    t = _TICKET.sub(" TICKET ", t)
    t = _CODIGO.sub(" CODIGO ", t)
    return re.sub(r"\s+", " ", t).strip()
