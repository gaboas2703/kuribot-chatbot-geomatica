"""Genera las figuras del informe a partir de los resultados reproducibles (resultados/*.json).

Uso: python experimentos/figuras.py --version v1 [--v2 v2]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
RES, FIG = RAIZ / "resultados", RAIZ / "docs" / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

# Paleta de referencia validada (dataviz): categórica en orden fijo y rampa secuencial azul
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e6e5e1"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
from matplotlib.ticker import FuncFormatter  # noqa: E402

COMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
plt.rcParams.update({"axes.formatter.use_locale": False, "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#b9b8b3", "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
NOMBRE = {"M0": "M0 TF-IDF + RL", "M1": "M1 fastText + BiLSTM", "M2": "M2 BETO congelado + MLP", "M3": "M3 BETO ajustado"}
ETQ = {"ayuda_capacidades": "ayuda", "consulta_geodesia": "consulta geodesia", "consultar_concesion": "consultar concesión",
       "convertir_coordenadas": "convertir coord.", "despedida": "despedida", "emergencia_seguridad": "emergencia",
       "estado_solicitud": "estado solicitud", "fuera_alcance": "fuera de alcance", "hablar_humano": "hablar con humano",
       "reportar_alerta": "reportar alerta", "saludo": "saludo", "solicitar_cartografia": "solicitar cartografía",
       "solicitar_vuelo": "solicitar vuelo", "verificar_punto": "verificar punto"}


def _comas(fig):
    for ax in fig.axes:
        if ax.get_xaxis().get_major_formatter().__class__.__name__ == "ScalarFormatter":
            ax.xaxis.set_major_formatter(COMA)
        if ax.get_yaxis().get_major_formatter().__class__.__name__ == "ScalarFormatter":
            ax.yaxis.set_major_formatter(COMA)


_savefig = matplotlib.figure.Figure.savefig


def _savefig_comas(self, *a, **k):
    _comas(self)
    return _savefig(self, *a, **k)


matplotlib.figure.Figure.savefig = _savefig_comas


def barras_modelos(r, cv):
    mods = list(r["modelos"])
    grupos = [("Validación cruzada\n(5 pliegues, media)", lambda m: cv[m]["f1_macro_media"] if cv else np.nan),
              ("Prueba limpia\n(n = 252)", lambda m: r["modelos"][m]["prueba"]["prueba_limpia"]["f1_macro"]),
              ("Prueba con ruido\n(n = 252)", lambda m: r["modelos"][m]["prueba"]["prueba_ruido"]["f1_macro"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.8 / len(mods)
    for j, m in enumerate(mods):
        vals = [g[1](m) for g in grupos]
        x = np.arange(len(grupos)) + (j - (len(mods) - 1) / 2) * w
        ax.bar(x, vals, w * 0.92, color=C[j], label=NOMBRE[m], zorder=3)
        if cv and j < 4:
            de = [cv[m]["f1_macro_de"], 0, 0]
            ax.errorbar(x[0], vals[0], yerr=de[0], color=TINTA2, capsize=2, lw=0.8, zorder=4)
        extra = [cv[m]["f1_macro_de"] if cv else 0, 0, 0]
        for xi, v, e in zip(x, vals, extra):
            ax.text(xi, v + e + 0.012, f"{v:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=6.5, color=TINTA)
    ax.set_xticks(range(len(grupos)), [g[0] for g in grupos])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 macro")
    ax.yaxis.grid(True, color=REJILLA, zorder=0)
    ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.savefig(FIG / "comparacion_modelos.png")
    plt.close(fig)


def matriz(r, modelo, clave, nombre):
    et = r["etiquetas"]
    M = np.array(r["modelos"][modelo]["prueba"][clave]["matriz_confusion"], dtype=float)
    Mn = M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(Mn, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(et)):
        for j in range(len(et)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if Mn[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(len(et)), [ETQ[e] for e in et], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(et)), [ETQ[e] for e in et], fontsize=7)
    ax.set_xlabel("Intención predicha")
    ax.set_ylabel("Intención real")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(FIG / nombre)
    plt.close(fig)


def f1_por_clase(r):
    et = r["etiquetas"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), sharey=True)
    for ax, clave, tit in zip(axes, ["prueba_limpia", "prueba_ruido"], ["Prueba limpia", "Prueba con ruido"]):
        y = np.arange(len(et))
        for j, (m, col) in enumerate([("M0", C[0]), ("M3", C[1])]):
            rep = r["modelos"][m]["prueba"][clave]["reporte"]
            vals = [rep[e]["f1-score"] for e in et]
            ax.barh(y + (j - 0.5) * 0.38, vals, 0.36, color=col, label=NOMBRE[m], zorder=3)
        ax.set_yticks(y, [ETQ[e] for e in et], fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_title(tit)
        ax.xaxis.grid(True, color=REJILLA, zorder=0)
        ax.set_xlabel("F1")
        ax.invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "f1_por_intencion.png")
    plt.close(fig)


def curva_entrenamiento(r):
    h = r["modelos"]["M3"]["info"]["historia"]
    ep = [x["epoca"] for x in h]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    axes[0].plot(ep, [x["perdida_entrenamiento"] for x in h], color=C[0], lw=2, marker="o", ms=4)
    axes[0].set_title("Pérdida de entrenamiento (M3)")
    axes[0].set_xlabel("Época")
    axes[1].plot(ep, [x["f1_macro_validacion"] for x in h], color=C[1], lw=2, marker="o", ms=4)
    axes[1].set_title("F1 macro en validación (M3)")
    axes[1].set_xlabel("Época")
    for ax in axes:
        ax.yaxis.grid(True, color=REJILLA)
        ax.set_xticks(ep)
    fig.tight_layout()
    fig.savefig(FIG / "curva_entrenamiento.png")
    plt.close(fig)


def fiabilidad(r):
    p = r["modelos"]["M3"]["prueba"]["prueba_limpia"]
    et = r["etiquetas"]
    import csv
    filas = list(csv.DictReader(open(RES / f"predicciones_prueba_{r['version']}.csv", encoding="utf-8")))
    filas = [f for f in filas if f["conjunto"] == "prueba_limpia"]
    conf = np.array(p["conf_cal"])
    ok = np.array([f["pred_M3"] == f["real"] for f in filas])
    bordes = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    xs, ys, ns = [], [], []
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(ok[m].mean())
            ns.append(m.sum())
    axes[0].plot([0, 1], [0, 1], ls="--", color=TINTA2, lw=1)
    axes[0].plot(xs, ys, color=C[0], lw=1.5, zorder=2)
    axes[0].scatter(xs, ys, s=[18 + 2.2 * n for n in ns], color=C[0], edgecolor="white", linewidth=1.5, zorder=3)
    for xx, yy, n in zip(xs, ys, ns):
        axes[0].annotate(f"n={n}", (xx, yy), xytext=(4, -11), textcoords="offset points", fontsize=6, color=TINTA2)
    axes[0].set_xlabel("Confianza calibrada")
    axes[0].set_ylabel("Exactitud observada")
    axes[0].set_title("Diagrama de fiabilidad (M3)")
    curva = r["modelos"]["M3"]["umbral"]["curva_prueba"]
    axes[1].plot([c["cobertura"] for c in curva], [c["exactitud_aceptadas"] for c in curva], color=C[1], lw=2, marker="o", ms=4)
    tau = r["modelos"]["M3"]["umbral"]["tau_elegido_en_validacion"]
    sel = min(curva, key=lambda c: abs(c["tau"] - tau))
    axes[1].plot(sel["cobertura"], sel["exactitud_aceptadas"], marker="o", ms=9, color=C[1], mec="white", mew=2)
    axes[1].annotate(f"τ = {tau:.2f}".replace(".", ","), (sel["cobertura"], sel["exactitud_aceptadas"]), xytext=(-18, 10),
                     textcoords="offset points", fontsize=7, color=TINTA)
    axes[1].set_xlabel("Cobertura (fracción aceptada)")
    axes[1].set_ylabel("Exactitud en aceptadas")
    axes[1].set_title("Riesgo–cobertura (prueba limpia)")
    for ax in axes:
        ax.grid(True, color=REJILLA)
    fig.tight_layout()
    fig.savefig(FIG / "calibracion_umbral.png")
    plt.close(fig)


def tornado():
    from kuribot.razonamiento import sensibilidad
    ev = {"H": 0, "A": 1, "M": 1, "S": 1}
    filas = [f for f in sensibilidad(ev) if f["rango"] > 1e-3][:8][::-1]
    base = filas[0]["base"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, f in enumerate(filas):
        lo, hi = min(f["bajo"], f["alto"]), max(f["bajo"], f["alto"])
        ax.barh(i, hi - lo, left=lo, color=C[0], height=0.55, zorder=3)
        ax.text(hi + 0.005, i, f"{lo:.2f}–{hi:.2f}".replace(".", ","), va="center", fontsize=6.5, color=TINTA)
    ax.axvline(base, color=C[1], lw=1.5, zorder=4)
    ax.text(base, len(filas) - 0.3, f" posterior base {base:.3f}".replace(".", ","), color=TINTA, fontsize=7)
    ax.set_yticks(range(len(filas)), [f["parametro"] for f in filas], fontsize=7)
    ax.set_xlabel("P(intrusión | evidencias) al variar cada parámetro ±0,10")
    ax.xaxis.grid(True, color=REJILLA, zorder=0)
    fig.savefig(FIG / "sensibilidad_rb.png")
    plt.close(fig)


def grafos():
    from kuribot.razonamiento import ARISTAS, DESCRIPCION_NODOS
    etiqueta = {k: f"{k}\\n{v}" for k, v in DESCRIPCION_NODOS.items()}
    import textwrap
    lineas = ["digraph RB { rankdir=TB; nodesep=0.25; ranksep=0.5; bgcolor=white;",
              'node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=10, color="#9ec5f4", fillcolor="#eef5fd"];']
    for k, v in DESCRIPCION_NODOS.items():
        txt = "\\n".join(textwrap.wrap(v, 24))
        extra = ', fillcolor="#fde6dc", color="#eb6834", penwidth=2' if k == "I" else (', fillcolor="#f3f2ee", color="#b9b8b3"' if k in "HA" else "")
        lineas.append(f'{k} [label="{k}: {txt}"{extra}];')
    lineas.append("{rank=same; H; A;}")
    for a, b in ARISTAS:
        lineas.append(f'{a} -> {b} [color="#52514e"];')
    lineas.append("}")
    (FIG / "red_bayesiana.dot").write_text("\n".join(lineas), encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", str(FIG / "red_bayesiana.dot"), "-o", str(FIG / "red_bayesiana.png")], check=True)

    arq = r'''digraph ARQ { rankdir=LR; compound=true; nodesep=0.3; ranksep=0.45; fontname="DejaVu Sans"; bgcolor=white;
node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=9, fillcolor="#eef5fd", color="#6da7ec"];
edge [color="#52514e", fontname="DejaVu Sans", fontsize=7];
U [label="Usuario interno\n(Geomática, Seguridad,\nOperaciones)", shape=ellipse, fillcolor="#fde6dc", color="#eb6834"];
UI [label="Interfaz\nStreamlit (web) · consola", fillcolor="#f3f2ee", color="#b9b8b3"];
subgraph cluster_nlu { label="1. Comprensión (NLU)"; style="rounded"; color="#9ec5f4"; fontsize=9;
  PII [label="Enmascarado de datos\npersonales (R02)"]; NER [label="Extractor de entidades\n(regex + diccionario:\ncoordenadas, datum,\nconcesión, ticket, fecha,\nevidencias con negación)"];
  NLU [label="Clasificador de intenciones\nBETO ajustado + softmax\ncalibrado (T) · 14 clases", fillcolor="#fde6dc", color="#eb6834", penwidth=2]; }
subgraph cluster_dm { label="2. Gestión del diálogo y razonamiento"; style="rounded"; color="#9ec5f4"; fontsize=9;
  DM [label="Gestor de diálogo\nestado · ranuras · aclaración\n· cambio de tema · abstención"];
  RB [label="Red bayesiana (9 nodos)\nP(intrusión | evidencias)\npgmpy · eliminación de variables", fillcolor="#e3f6ee", color="#1baf7a", penwidth=2];
  MR [label="Motor de reglas\nencadenamiento hacia adelante\nR01–R42 (5 niveles)\n+ reglas de vuelo V01–V06", fillcolor="#e3f6ee", color="#1baf7a", penwidth=2];
  GEO [label="Geoprocesos\npyproj (EPSG:3990) · Shapely\npunto en polígono · distancias"]; }
subgraph cluster_kb { label="3. Base de conocimiento (sintética)"; style="rounded"; color="#9ec5f4"; fontsize=9;
  KB1 [label="Catastro GeoJSON\n(concesiones, área\nprotegida, comunidad)", shape=cylinder, fillcolor="#f3f2ee", color="#b9b8b3"];
  KB2 [label="Operación JSON\n(pronóstico, drones,\npilotos, historial)", shape=cylinder, fillcolor="#f3f2ee", color="#b9b8b3"];
  KB3 [label="FAQ geodésica\n(recuperación TF-IDF)", shape=cylinder, fillcolor="#f3f2ee", color="#b9b8b3"];
  KB4 [label="Tickets SQLite", shape=cylinder, fillcolor="#f3f2ee", color="#b9b8b3"]; }
NLG [label="4. Generación (NLG)\nplantillas con datos\nverificados (sin LLM)"];
LOG [label="Registro auditable\n(JSONL, texto enmascarado)", fillcolor="#f3f2ee", color="#b9b8b3"];
HUM [label="Derivación a persona\n(analista, Seguridad, Legal)", fillcolor="#fde6dc", color="#eb6834"];
U -> UI [label="mensaje"]; UI -> PII; PII -> NER; PII -> NLU; NER -> DM; NLU -> DM [label="intención,\nconfianza"];
DM -> GEO; DM -> RB [label="evidencias"]; RB -> MR [label="posterior"]; GEO -> MR [label="contexto"]; DM -> MR;
GEO -> KB1 [dir=both]; RB -> KB2 [dir=both, label="H, A"]; MR -> KB2 [dir=both]; DM -> KB3 [dir=both]; DM -> KB4 [dir=both];
MR -> NLG [label="prioridad,\nacción"]; DM -> NLG; NLG -> UI [label="respuesta +\ntrazabilidad"]; DM -> LOG [style=dashed]; MR -> HUM [style=dashed, label="R01, R03,\nR40, R41"];
}'''
    (FIG / "arquitectura.dot").write_text(arq, encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", str(FIG / "arquitectura.dot"), "-o", str(FIG / "arquitectura.png")], check=True)


def iteraciones(v1, v2):
    """Comparación v1 -> v2 del modelo de producción."""
    if not v2:
        return
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    etiquetas = ["Prueba limpia", "Prueba con ruido"]
    for j, (r, nombre) in enumerate([(v1, "v1 (sin aumento)"), (v2, "v2 (con aumento de ruido)")]):
        vals = [r["modelos"]["M3"]["prueba"][k]["f1_macro"] for k in ("prueba_limpia", "prueba_ruido")]
        x = np.arange(2) + (j - 0.5) * 0.36
        ax.bar(x, vals, 0.34, color=C[j], label=nombre, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}".replace(".", ","), ha="center", fontsize=7, color=TINTA)
    ax.set_xticks(range(2), etiquetas)
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("F1 macro (M3)")
    ax.yaxis.grid(True, color=REJILLA, zorder=0)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.savefig(FIG / "iteraciones_m3.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--v2", default=None)
    args = ap.parse_args()
    r = json.loads((RES / f"nlp_final_{args.version}.json").read_text(encoding="utf-8"))
    barras_modelos(r, r.get("validacion_cruzada"))          # comparación de los cuatro modelos (iteración 1)
    prod = json.loads((RES / f"nlp_final_{args.v2}.json").read_text(encoding="utf-8")) if args.v2 else r
    matriz(prod, "M3", "prueba_limpia", "matriz_confusion_m3_limpia.png")   # modelo de producción
    matriz(prod, "M3", "prueba_ruido", "matriz_confusion_m3_ruido.png")
    f1_por_clase(prod)
    curva_entrenamiento(prod)
    fiabilidad(prod)
    tornado()
    grafos()
    if args.v2:
        iteraciones(r, prod)
    print("Figuras en", FIG)


if __name__ == "__main__":
    main()
