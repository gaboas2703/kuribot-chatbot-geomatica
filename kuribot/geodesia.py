"""Operaciones geodésicas y espaciales de KuriBot (pyproj + Shapely).

Convenciones
- Sistema de trabajo: WGS 84 / UTM zona 17S (EPSG:32717), en metros.
- PSAD56 / UTM 17S (EPSG:24877) se transforma con la operación EPSG:3990
  «PSAD56 to WGS 84 (14)», Helmert de 7 parámetros (convención coordinate frame) válida
  para Ecuador continental, con exactitud declarada de 5 m en la base EPSG. En producción
  deben confirmarse los parámetros oficiales vigentes del IGM antes de usar los resultados
  con fines legales o catastrales.
- SIRGAS 2000 / UTM 17S (EPSG:31977) se trata como equivalente a WGS 84 al nivel métrico
  (transformación nula de la base EPSG); la diferencia real depende de la época de referencia.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from pyproj import CRS, Transformer
from pyproj.transformer import TransformerGroup

EPSG = {
    "WGS84_GEO": 4326,
    "WGS84_UTM17S": 32717,
    "PSAD56_GEO": 4248,
    "PSAD56_UTM17S": 24877,
    "SIRGAS2000_UTM17S": 31977,
}
OPERACION_PSAD56_WGS84 = 3990  # PSAD56 to WGS 84 (14), Ecuador continental


@dataclass(frozen=True)
class Punto:
    """Coordenada de entrada tal como la expresó el usuario."""
    x: float              # Este (m) o longitud (°)
    y: float              # Norte (m) o latitud (°)
    tipo: str             # "utm" o "geo"
    datum: str | None     # "WGS84", "PSAD56", "SIRGAS" o None si no se indicó


@lru_cache(maxsize=None)
def _transformador(origen: int, destino: int) -> Transformer:
    return Transformer.from_crs(origen, destino, always_xy=True)


@lru_cache(maxsize=None)
def _psad56_a_wgs84_geo() -> Transformer:
    """Transformación PSAD56 (geográficas) -> WGS 84 (geográficas) fijada en EPSG:3990."""
    grupo = TransformerGroup(EPSG["PSAD56_GEO"], EPSG["WGS84_GEO"], always_xy=True)
    for t in grupo.transformers:
        ident = t.to_json_dict().get("id", {})
        if ident.get("code") == OPERACION_PSAD56_WGS84:
            return t
    raise RuntimeError("La base EPSG local no contiene la operación 3990.")


def descripcion_transformacion() -> str:
    t = _psad56_a_wgs84_geo()
    nombre = t.description.split(" (with axis")[0]
    return f"{nombre} (EPSG:{OPERACION_PSAD56_WGS84}; exactitud declarada {t.accuracy:.0f} m)"


def psad56_utm_a_wgs84_utm(e: float, n: float) -> tuple[float, float]:
    lon, lat = _transformador(EPSG["PSAD56_UTM17S"], EPSG["PSAD56_GEO"]).transform(e, n)
    lon2, lat2 = _psad56_a_wgs84_geo().transform(lon, lat)
    return _transformador(EPSG["WGS84_GEO"], EPSG["WGS84_UTM17S"]).transform(lon2, lat2)


def wgs84_utm_a_psad56_utm(e: float, n: float) -> tuple[float, float]:
    lon, lat = _transformador(EPSG["WGS84_UTM17S"], EPSG["WGS84_GEO"]).transform(e, n)
    lon2, lat2 = _psad56_a_wgs84_geo().transform(lon, lat, direction="INVERSE")
    return _transformador(EPSG["PSAD56_GEO"], EPSG["PSAD56_UTM17S"]).transform(lon2, lat2)


def a_wgs84_utm(p: Punto) -> tuple[float, float]:
    """Lleva cualquier entrada soportada a WGS 84 / UTM 17S."""
    datum = (p.datum or "WGS84").upper()
    if p.tipo == "geo":
        if datum == "PSAD56":
            lon, lat = _psad56_a_wgs84_geo().transform(p.x, p.y)
        else:
            lon, lat = p.x, p.y
        return _transformador(EPSG["WGS84_GEO"], EPSG["WGS84_UTM17S"]).transform(lon, lat)
    if datum == "PSAD56":
        return psad56_utm_a_wgs84_utm(p.x, p.y)
    return p.x, p.y  # WGS84 o SIRGAS (equivalencia métrica)


def todas_las_representaciones(p: Punto) -> dict[str, tuple[float, float]]:
    e, n = a_wgs84_utm(p)
    lon, lat = _transformador(EPSG["WGS84_UTM17S"], EPSG["WGS84_GEO"]).transform(e, n)
    pe, pn = wgs84_utm_a_psad56_utm(e, n)
    return {"WGS84_UTM17S": (e, n), "WGS84_GEO": (lat, lon), "PSAD56_UTM17S": (pe, pn)}


def dms(valor: float, eje: str) -> str:
    hemis = ("N" if valor >= 0 else "S") if eje == "lat" else ("E" if valor >= 0 else "W")
    v = abs(valor)
    g = int(v)
    m = int((v - g) * 60)
    s = (v - g - m / 60) * 3600
    return f"{g}°{m:02d}'{s:05.2f}\"{hemis}"


def dentro_de_ecuador_utm17s(e: float, n: float) -> bool:
    """Control de rango aproximado para Ecuador continental expresado en UTM 17S."""
    return 450_000 <= e <= 1_100_000 and 9_440_000 <= n <= 10_170_000


def distancia(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nombre_crs(epsg: int) -> str:
    return CRS.from_epsg(epsg).name
