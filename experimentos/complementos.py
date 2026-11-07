"""Cálculos complementarios del informe:
(1) latencia de M2 sin caché de vectores (en el experimento principal la caché la subestimaba);
(2) McNemar del modelo de producción (M3 v2) frente a la mejor línea base (M0 v1) sobre los mismos conjuntos de prueba.
Uso: python experimentos/complementos.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "experimentos"))
import entrenar_nlp as EN  # noqa: E402
from kuribot import nlu_modelos as NM  # noqa: E402

torch.set_num_threads(2)
RES = RAIZ / "resultados"


def latencia_m2():
    etiquetas, idx, parts = EN.cargar(1)
    pool = parts["pool"]
    x = [f["texto"] for f in pool]
    y = np.array([idx[f["intencion"]] for f in pool])
    g = [f["grupo"] for f in pool]
    tr, dev = EN.separar_dev(y, g, EN.SEMILLA)
    m = NM.ModeloM2(len(etiquetas), EN.SEMILLA, ruta_beto=str(RAIZ / "modelos" / "beto_base"))
    m.entrenar([x[i] for i in tr], y[tr], [x[i] for i in dev], y[dev])
    xs = [f["texto"] for f in parts["prueba_limpia"]]
    ys = np.array([idx[f["intencion"]] for f in parts["prueba_limpia"]])
    f1 = f1_score(ys, m.logits(xs).argmax(1), average="macro")
    m.memo.clear()
    lat = NM.latencia_ms(m, xs[:60])
    out = {"f1_macro_prueba_limpia_reentrenado": f1, **lat,
           "nota": "M2 reentrenado con la misma partición y semilla de v1; latencia medida sin caché de vectores."}
    (RES / "latencia_m2_v1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("M2", out)


def mcnemar_produccion():
    from scipy.stats import binomtest
    f1 = list(csv.DictReader(open(RES / "predicciones_prueba_v1.csv", encoding="utf-8")))
    f2 = list(csv.DictReader(open(RES / "predicciones_prueba_v2.csv", encoding="utf-8")))
    out = {}
    for conj in ("prueba_limpia", "prueba_ruido"):
        a = [r for r in f2 if r["conjunto"] == conj]
        b = [r for r in f1 if r["conjunto"] == conj]
        assert [r["texto"] for r in a] == [r["texto"] for r in b]
        ok_a = np.array([r["pred_M3"] == r["real"] for r in a])
        ok_b = np.array([r["pred_M0"] == r["real"] for r in b])
        ok_c = np.array([r["pred_M3"] == r["real"] for r in b])
        n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
        m01, m10 = int((ok_a & ~ok_c).sum()), int((~ok_a & ok_c).sum())
        out[conj] = {"M3v2_vs_M0v1": {"solo_M3v2": n01, "solo_M0v1": n10, "p_valor": binomtest(n10, n01 + n10, 0.5).pvalue},
                     "M3v2_vs_M3v1": {"solo_M3v2": m01, "solo_M3v1": m10, "p_valor": binomtest(m10, m01 + m10, 0.5).pvalue}}
    (RES / "mcnemar_produccion.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    mcnemar_produccion()
    latencia_m2()
