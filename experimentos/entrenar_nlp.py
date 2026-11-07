"""Experimento NLP de KuriBot: validación cruzada, entrenamiento final, evaluación y exportación.

Uso:
    python experimentos/entrenar_nlp.py --version v1            # CV de 4 modelos + modelo final
    python experimentos/entrenar_nlp.py --version v2 --aumento  # iteración 2 (sin CV), solo M0 y M3

Salidas en resultados/ (JSON y CSV) y modelos/ (artefactos de inferencia).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "datos"))
from kuribot import nlu_modelos as NM  # noqa: E402

torch.set_num_threads(2)
RES = RAIZ / "resultados"
MOD = RAIZ / "modelos"
RES.mkdir(exist_ok=True)
SEMILLA = 42


def cargar(iteracion: int = 1):
    d = json.loads((RAIZ / "datos" / "corpus.json").read_text(encoding="utf-8"))["datos"]
    d = [f for f in d if f.get("iteracion", 1) <= iteracion]
    etiquetas = sorted({f["intencion"] for f in d})
    idx = {e: i for i, e in enumerate(etiquetas)}
    parts = {p: [f for f in d if f["particion"] == p] for p in ("pool", "prueba_limpia", "prueba_ruido")}
    return etiquetas, idx, parts


def vectores_fasttext():
    import spacy
    nlp = spacy.load("es_core_news_lg", exclude=["tok2vec", "morphologizer", "parser", "senter", "attribute_ruler",
                                                  "lemmatizer", "ner"])
    vocab = nlp.vocab

    def f(t):
        return vocab[t].vector.copy() if vocab.has_vector(t) else None
    return f


def aumentar(textos, etiquetas, grupos, semilla=7):
    """Iteración 2: añade una copia perturbada (sin tildes, abreviaturas, errata) de cada frase de entrenamiento."""
    from construir_corpus import ruido
    rng = random.Random(semilla)
    xt, yt, gt = list(textos), list(etiquetas), list(grupos)
    for t, y, g in zip(textos, etiquetas, grupos):
        xt.append(ruido(t, rng))
        yt.append(y)
        gt.append(g)
    return xt, np.array(yt), gt


EPOCAS_M3 = 6
PACIENCIA_M3 = 2


def fabricar(nombre, n, semilla, vec):
    if nombre == "M0":
        return NM.ModeloM0(n, semilla)
    if nombre == "M1":
        return NM.ModeloM1(n, semilla, vectores=vec)
    if nombre == "M2":
        return NM.ModeloM2(n, semilla, ruta_beto=str(MOD / "beto_base"))
    return NM.ModeloM3(n, semilla, ruta_beto=str(MOD / "beto_base"), epocas=EPOCAS_M3, paciencia=PACIENCIA_M3)


def separar_dev(y, grupos, semilla):
    sgk = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=semilla)
    tr, dev = next(sgk.split(np.zeros(len(y)), y, grupos))
    return tr, dev


def validacion_cruzada(modelos, pool, idx, n, vec):
    x = [f["texto"] for f in pool]
    y = np.array([idx[f["intencion"]] for f in pool])
    g = [f["grupo"] for f in pool]
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    out = {m: [] for m in modelos}
    for k, (tr, te) in enumerate(sgk.split(np.zeros(len(y)), y, g)):
        tr_in, dev_in = separar_dev(y[tr], [g[i] for i in tr], SEMILLA + k)
        tr_idx, dev_idx = tr[tr_in], tr[dev_in]
        for m in modelos:
            t0 = time.time()
            mod = fabricar(m, n, SEMILLA + k, vec)
            mod.entrenar([x[i] for i in tr_idx], y[tr_idx], [x[i] for i in dev_idx], y[dev_idx])
            pred = mod.logits([x[i] for i in te]).argmax(1)
            r = {"pliegue": k + 1, "exactitud": accuracy_score(y[te], pred), "f1_macro": f1_score(y[te], pred, average="macro"),
                 "n_prueba": int(len(te)), "segundos": round(time.time() - t0, 1)}
            out[m].append(r)
            print(f"[CV] pliegue {k+1} {m}: F1={r['f1_macro']:.3f} acc={r['exactitud']:.3f} ({r['segundos']} s)", flush=True)
            del mod
    resumen = {m: {"f1_macro_media": float(np.mean([r["f1_macro"] for r in v])),
                   "f1_macro_de": float(np.std([r["f1_macro"] for r in v], ddof=1)),
                   "exactitud_media": float(np.mean([r["exactitud"] for r in v])),
                   "exactitud_de": float(np.std([r["exactitud"] for r in v], ddof=1)), "pliegues": v}
               for m, v in out.items()}
    return resumen


def bootstrap_f1(y, pred, B=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    n = len(y)
    vals = [f1_score(y[i], pred[i], average="macro") for i in (rng.integers(0, n, n) for _ in range(B))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mcnemar(y, a, b):
    """Prueba exacta de McNemar (binomial sobre pares discordantes)."""
    a_ok, b_ok = a == y, b == y
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"solo_A_acierta": n01, "solo_B_acierta": n10, "p_valor": float(p)}


def evaluar(nombre, mod, T, sets, etiquetas):
    res = {}
    for clave, (xs, ys) in sets.items():
        z = mod.logits(xs)
        p1, pT = NM.softmax(z), NM.softmax(z, T)
        pred = z.argmax(1)
        k = int((pred == ys).sum())
        res[clave] = {
            "exactitud": accuracy_score(ys, pred), "exactitud_ic95_wilson": NM.wilson(k, len(ys)),
            "f1_macro": f1_score(ys, pred, average="macro"), "f1_macro_ic95_bootstrap": bootstrap_f1(ys, pred),
            "f1_ponderado": f1_score(ys, pred, average="weighted"),
            "ece_sin_calibrar": NM.ece(p1, ys), "ece_calibrado": NM.ece(pT, ys),
            "reporte": classification_report(ys, pred, labels=list(range(len(etiquetas))), target_names=etiquetas,
                                             output_dict=True, zero_division=0),
            "matriz_confusion": confusion_matrix(ys, pred, labels=list(range(len(etiquetas)))).tolist(),
            "pred": pred.tolist(), "conf_cal": pT.max(1).tolist(), "probs_cal": pT.round(5).tolist(),
        }
        print(f"[TEST] {nombre} {clave}: acc={res[clave]['exactitud']:.3f} F1={res[clave]['f1_macro']:.3f} "
              f"ECE {res[clave]['ece_sin_calibrar']:.3f}->{res[clave]['ece_calibrado']:.3f}", flush=True)
    return res


def analisis_umbral(z_dev, y_dev, z_test, y_test, T, i_oos):
    """Selección del umbral de confianza en validación y su efecto en prueba (abstención -> fuera_alcance)."""
    def aplicar(z, tau):
        p = NM.softmax(z, T)
        pred = p.argmax(1).copy()
        pred[p.max(1) < tau] = i_oos
        return pred
    grid = np.round(np.arange(0.20, 0.91, 0.05), 2)
    f_dev = [f1_score(y_dev, aplicar(z_dev, t), average="macro") for t in grid]
    tau = float(grid[int(np.argmax(f_dev))])
    curva = []
    for t in grid:
        p = NM.softmax(z_test, T)
        acept = p.max(1) >= t
        pred = aplicar(z_test, t)
        en_alcance = y_test != i_oos
        curva.append({"tau": float(t), "cobertura": float(acept.mean()),
                      "exactitud_aceptadas": float((p.argmax(1)[acept] == y_test[acept]).mean()) if acept.any() else float("nan"),
                      "f1_macro_con_abstencion": float(f1_score(y_test, pred, average="macro")),
                      "recall_fuera_alcance": float((pred[~en_alcance] == i_oos).mean()),
                      "exactitud_en_alcance": float((pred[en_alcance] == y_test[en_alcance]).mean())})
    return {"tau_elegido_en_validacion": tau, "f1_dev_por_tau": dict(zip(map(float, grid), map(float, f_dev))), "curva_prueba": curva}


def exportar_m3(mod, etiquetas, T, tau, destino: Path, extra: dict):
    from safetensors.torch import save_file
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    estado = {k: v.detach().to(torch.float16).contiguous() for k, v in mod.red.state_dict().items()}
    save_file(estado, str(destino / "clasificador.safetensors"))
    for f in ("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(MOD / "beto_base" / f, destino / ("bert_config.json" if f == "config.json" else f))
    (destino / "kuribot_nlu.json").write_text(json.dumps({
        "etiquetas": etiquetas, "temperatura": T, "umbral_confianza": tau, "max_len": 48,
        "preprocesamiento": "kuribot.entidades.enmascarar_entidades", "pesos": "float16", **extra}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--aumento", action="store_true")
    ap.add_argument("--sin-cv", action="store_true")
    ap.add_argument("--modelos", default="M0,M1,M2,M3")
    ap.add_argument("--iteracion", type=int, default=1, help="1: corpus original; 2: con plantillas agregadas tras el análisis de fallos")
    ap.add_argument("--epocas-m3", type=int, default=6)
    ap.add_argument("--paciencia-m3", type=int, default=2)
    args = ap.parse_args()
    global EPOCAS_M3, PACIENCIA_M3
    EPOCAS_M3, PACIENCIA_M3 = args.epocas_m3, args.paciencia_m3
    modelos = args.modelos.split(",")
    etiquetas, idx, parts = cargar(args.iteracion)
    n = len(etiquetas)
    vec = vectores_fasttext() if "M1" in modelos else None
    salida = {"version": args.version, "aumento": args.aumento, "iteracion_corpus": args.iteracion,
              "epocas_max_m3": args.epocas_m3, "paciencia_m3": args.paciencia_m3, "etiquetas": etiquetas, "semilla": SEMILLA,
              "entorno": {"torch": torch.__version__, "hilos_cpu": torch.get_num_threads()}}
    if not args.sin_cv:
        salida["validacion_cruzada"] = validacion_cruzada(modelos, parts["pool"], idx, n, vec)
        (RES / f"nlp_cv_{args.version}.json").write_text(json.dumps(salida["validacion_cruzada"], indent=1), encoding="utf-8")

    pool = parts["pool"]
    x = [f["texto"] for f in pool]
    y = np.array([idx[f["intencion"]] for f in pool])
    g = [f["grupo"] for f in pool]
    tr, dev = separar_dev(y, g, SEMILLA)
    xtr, ytr, gtr = [x[i] for i in tr], y[tr], [g[i] for i in tr]
    if args.aumento:
        xtr, ytr, gtr = aumentar(xtr, ytr, gtr)
    xdev, ydev = [x[i] for i in dev], y[dev]
    sets = {k: ([f["texto"] for f in parts[k]], np.array([idx[f["intencion"]] for f in parts[k]]))
            for k in ("prueba_limpia", "prueba_ruido")}
    salida["particion_final"] = {"entrenamiento": len(xtr), "validacion": len(xdev), "prueba_limpia": len(sets["prueba_limpia"][1]),
                                 "prueba_ruido": len(sets["prueba_ruido"][1])}
    finales, preds = {}, {}
    for m in modelos:
        t0 = time.time()
        mod = fabricar(m, n, SEMILLA, vec)
        info = mod.entrenar(xtr, ytr, xdev, ydev)
        z_dev = mod.logits(xdev)
        T = NM.ajustar_temperatura(z_dev, ydev)
        r = evaluar(m, mod, T, sets, etiquetas)
        if m == "M2":
            mod.memo.clear()  # sin caché: la latencia debe incluir el paso por BETO
        lat = NM.latencia_ms(mod, sets["prueba_limpia"][0][:60])
        finales[m] = {"nombre": mod.nombre, "temperatura": T, "parametros_entrenables": NM.n_parametros(mod),
                      "segundos_entrenamiento": round(time.time() - t0, 1), "latencia": lat,
                      "f1_macro_validacion": float(f1_score(ydev, z_dev.argmax(1), average="macro")),
                      "info": {k: v for k, v in info.items()}, "prueba": r}
        preds[m] = {k: np.array(v["pred"]) for k, v in r.items()}
        if m in ("M3", "M0"):
            finales[m]["umbral"] = analisis_umbral(z_dev, ydev, mod.logits(sets["prueba_limpia"][0]), sets["prueba_limpia"][1], T,
                                                   etiquetas.index("fuera_alcance"))
        if m == "M3":
            tau = finales[m]["umbral"]["tau_elegido_en_validacion"]
            exportar_m3(mod, etiquetas, T, tau, MOD / f"intenciones_beto_{args.version}",
                        {"version": args.version, "aumento": args.aumento,
                         "f1_macro_prueba_limpia": r["prueba_limpia"]["f1_macro"],
                         "f1_macro_prueba_ruido": r["prueba_ruido"]["f1_macro"]})
        if m == "M0":
            import joblib
            joblib.dump({"pipe": mod.pipe, "etiquetas": etiquetas, "temperatura": T}, MOD / f"intenciones_tfidf_{args.version}.joblib")
        del mod
    comparaciones = {}
    if "M3" in preds:
        for otro in [m for m in modelos if m != "M3"]:
            for k, (xs, ys) in sets.items():
                comparaciones[f"M3_vs_{otro}_{k}"] = mcnemar(ys, preds["M3"][k], preds[otro][k])
    salida["modelos"] = finales
    salida["mcnemar"] = comparaciones
    (RES / f"nlp_final_{args.version}.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # Predicciones de prueba en CSV (para el informe y la auditoría de errores)
    import csv
    with open(RES / f"predicciones_prueba_{args.version}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["conjunto", "texto", "real"] + [f"pred_{m}" for m in modelos] + [f"conf_{m}" for m in modelos])
        for k, (xs, ys) in sets.items():
            for i, (t, yv) in enumerate(zip(xs, ys)):
                w.writerow([k, t, etiquetas[yv]] + [etiquetas[preds[m][k][i]] for m in modelos]
                           + [round(finales[m]["prueba"][k]["conf_cal"][i], 4) for m in modelos])
    print("Listo:", RES / f"nlp_final_{args.version}.json")


if __name__ == "__main__":
    main()
