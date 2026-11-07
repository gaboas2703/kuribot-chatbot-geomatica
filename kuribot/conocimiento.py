"""Base de conocimiento de KuriBot: catastro sintético, operación, FAQ y tickets (SQLite)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Point, shape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .entidades import plano

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
FRANJA_M = 500.0      # franja de amortiguamiento (supuesto de diseño heredado de SEPRIM)
RADIO_COMUNIDAD_M = 1000.0


@dataclass
class AnalisisPunto:
    e: float
    n: float
    ubicacion: str                 # propia | amortiguamiento | tercero | fuera
    concesion: str | None          # código de la concesión que contiene el punto
    cercana: str | None            # concesión propia más cercana
    distancia_lindero_m: float     # distancia al lindero de la concesión que lo contiene o de la más cercana
    area_protegida: str | None
    comunidad: str | None
    distancia_comunidad_m: float | None


class BaseConocimiento:
    def __init__(self, ruta_datos: Path = DATOS, db: str = ":memory:"):
        geo = json.loads((ruta_datos / "concesiones.geojson").read_text(encoding="utf-8"))
        self.concesiones, self.areas, self.comunidades = {}, {}, {}
        for f in geo["features"]:
            p, g = f["properties"], shape(f["geometry"])
            if p["capa"] == "concesion":
                self.concesiones[p["codigo"]] = {**p, "geom": g, "area_ha": round(g.area / 1e4, 1)}
            elif p["capa"] == "area_protegida":
                self.areas[p["codigo"]] = {**p, "geom": g}
            else:
                self.comunidades[p["codigo"]] = {**p, "geom": g}
        self.op = json.loads((ruta_datos / "operacion.json").read_text(encoding="utf-8"))
        self.hoy = dt.date.fromisoformat(self.op["fecha_referencia"])
        self.faq = json.loads((ruta_datos / "faq.json").read_text(encoding="utf-8"))
        self.nombres = {plano(c["nombre"]): cod for cod, c in self.concesiones.items()}
        self._indexar_faq()
        self.db = sqlite3.connect(db, check_same_thread=False)
        self._crear_tickets(json.loads((ruta_datos / "tickets_semilla.json").read_text(encoding="utf-8")))

    # ------------------------------------------------------------ catastro
    def ficha(self, codigo: str) -> dict | None:
        c = self.concesiones.get(codigo)
        if not c:
            return None
        return {k: c[k] for k in ("codigo", "nombre", "fase", "estado", "campamento", "titular", "area_ha")} | {
            "alertas_90d": self.op["historial_alertas_90d"].get(codigo, 0),
            "actividad_programada": self.op["actividad_autorizada_programada"].get(codigo)}

    def analizar_punto(self, e: float, n: float) -> AnalisisPunto:
        pt = Point(e, n)
        contiene = [c for c in self.concesiones.values() if c["geom"].covers(pt)]
        propias = [c for c in self.concesiones.values() if c["titular"] == "propia"]
        cercana = min(propias, key=lambda c: c["geom"].distance(pt))
        if contiene:
            c = sorted(contiene, key=lambda c: c["titular"] != "propia")[0]
            ubic = "propia" if c["titular"] == "propia" else "tercero"
            cod, dist = c["codigo"], c["geom"].exterior.distance(pt)
        else:
            dist = cercana["geom"].distance(pt)
            ubic, cod = ("amortiguamiento" if dist <= FRANJA_M else "fuera"), None
        area = next((a["nombre"] for a in self.areas.values() if a["geom"].covers(pt)), None)
        com = min(self.comunidades.values(), key=lambda c: c["geom"].distance(pt)) if self.comunidades else None
        dcom = com["geom"].distance(pt) if com else None
        return AnalisisPunto(e, n, ubic, cod, cercana["codigo"], round(dist, 1), area,
                             com["nombre"] if com and dcom <= RADIO_COMUNIDAD_M else None,
                             round(dcom, 1) if dcom is not None else None)

    def campamento(self, codigo: str | None) -> str:
        c = self.concesiones.get(codigo or "")
        return c["campamento"] if c and c["campamento"] in self.op["pronostico"] else "Central"

    # ------------------------------------------------------------ FAQ (recuperación por similitud TF-IDF)
    def _indexar_faq(self):
        self._faq_textos, self._faq_ids = [], []
        for f in self.faq:
            for q in f["preguntas"]:
                self._faq_textos.append(q)
                self._faq_ids.append(f["id"])
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, strip_accents="unicode")
        self._mat = self._vec.fit_transform(self._faq_textos)

    def buscar_faq(self, consulta: str) -> tuple[dict | None, float]:
        s = cosine_similarity(self._vec.transform([consulta]), self._mat)[0]
        i = int(s.argmax())
        f = next(x for x in self.faq if x["id"] == self._faq_ids[i])
        return f, float(s[i])

    # ------------------------------------------------------------ tickets
    def _crear_tickets(self, semilla):
        cur = self.db.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, tipo TEXT, detalle TEXT, estado TEXT, "
                    "asignado TEXT, compromiso TEXT, resolucion TEXT, prioridad TEXT, creado TEXT)")
        for t in semilla:
            cur.execute("INSERT OR REPLACE INTO tickets VALUES (?,?,?,?,?,?,?,?,?)",
                        (t["id"], t["tipo"], t["detalle"], t["estado"], t["asignado"], t["compromiso"],
                         t.get("resolucion"), t.get("prioridad"), self.op["fecha_referencia"]))
        self.db.commit()

    def ticket(self, ident: str) -> dict | None:
        cur = self.db.execute("SELECT * FROM tickets WHERE id = ?", (ident,))
        fila = cur.fetchone()
        if not fila:
            return None
        return dict(zip([c[0] for c in cur.description], fila))

    def nuevo_ticket(self, tipo: str, detalle: str, asignado: str, compromiso: str | None, prioridad: str | None = None) -> str:
        ultimos = [int(r[0].split("-")[-1]) for r in self.db.execute("SELECT id FROM tickets")]
        ident = f"GEO-2026-{max(ultimos) + 1:04d}"
        self.db.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?)",
                        (ident, tipo, detalle, "Registrado", asignado, compromiso, None, prioridad, self.op["fecha_referencia"]))
        self.db.commit()
        return ident

    def fecha(self, desfase: int) -> str:
        return (self.hoy + dt.timedelta(days=desfase)).isoformat()
