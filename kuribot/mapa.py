"""Mapa estático (matplotlib) del catastro sintético: funciona sin conexión a servicios de teselas."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

COLORES = {"propia": "#2E7D5B", "tercero": "#9A6B2F", "resaltado": "#C0392B", "protegida": "#5B8DB8"}


def dibujar(kb, codigo: str | None = None, punto: tuple[float, float] | None = None, ancho: float = 5.2, alto: float = 4.2):
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=110)
    for a in kb.areas.values():
        ax.add_patch(MplPolygon(list(a["geom"].exterior.coords), closed=True, fc=COLORES["protegida"], alpha=0.18,
                                ec=COLORES["protegida"], lw=1, hatch="..", label="Área protegida (ficticia)"))
    for cod, c in kb.concesiones.items():
        color = COLORES["resaltado"] if cod == codigo else COLORES[c["titular"]]
        ax.add_patch(MplPolygon(list(c["geom"].exterior.coords), closed=True, fc=color, alpha=0.30 if cod == codigo else 0.12,
                                ec=color, lw=2.2 if cod == codigo else 1))
        x, y = c["geom"].centroid.coords[0]
        ax.text(x, y, cod, ha="center", va="center", fontsize=7, color="#222")
    for com in kb.comunidades.values():
        x, y = com["geom"].coords[0]
        ax.plot(x, y, marker="^", color="#6C3483", ms=7)
        ax.text(x + 120, y - 160, "Comunidad", fontsize=6, color="#6C3483")
    if punto is not None:
        ax.plot(*punto, marker="*", color="#111", ms=14, mec="white", mew=1)
    ax.set_xlim(639_400, 649_600)
    ax.set_ylim(9_658_000, 9_666_600)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Este (m) · WGS 84 / UTM 17S", fontsize=7)
    ax.set_ylabel("Norte (m)", fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("Catastro sintético del prototipo", fontsize=8)
    fig.tight_layout()
    return fig
