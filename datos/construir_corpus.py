"""Construye corpus.json a partir de corpus_fuente.py (semilla fija, sin fuga entre particiones).

- POOL: cada plantilla con marcadores se instancia 3 veces con valores distintos; sin
  marcadores, una vez. Todas las instancias de una plantilla comparten «grupo», de modo que
  la validación cruzada (StratifiedGroupKFold) nunca separa variantes de la misma plantilla.
- PRUEBA: plantillas redactadas aparte, instanciadas una vez («prueba_limpia»).
- PRUEBA CON RUIDO: la misma prueba con perturbaciones ortográficas sembradas (sin tildes,
  minúsculas, abreviaturas de chat y un error de tecleo por frase con probabilidad 0,7).
  El ruido se aplica a la plantilla antes de insertar entidades, para no corromper códigos.

Uso: python datos/construir_corpus.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from corpus_fuente import POOL, POOL_ITERACION_2, PRUEBA, VALORES  # noqa: E402

SEMILLA = 2026
MARCADOR = re.compile(r"\{(\w+)\}")
VECINAS = {"a": "sq", "e": "wr", "i": "uo", "o": "ip", "u": "yi", "s": "ad", "n": "bm", "r": "et",
           "t": "ry", "l": "kñ", "c": "xv", "d": "sf", "m": "n", "p": "o"}


def instanciar(plantilla: str, rng: random.Random) -> str:
    return MARCADOR.sub(lambda m: rng.choice(VALORES[m.group(1)]), plantilla)


def sin_tildes(t: str) -> str:
    nfkd = unicodedata.normalize("NFD", t)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn" or c == "̃").replace("ñ", "ñ")


def error_tecleo(palabra: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(palabra) - 1)
    op = rng.choice(["intercambio", "omision", "duplicado", "vecina"])
    if op == "intercambio":
        return palabra[:i] + palabra[i + 1] + palabra[i] + palabra[i + 2:]
    if op == "omision":
        return palabra[:i] + palabra[i + 1:]
    if op == "duplicado":
        return palabra[:i] + palabra[i] + palabra[i:]
    c = palabra[i]
    return palabra[:i] + (rng.choice(VECINAS[c]) if c in VECINAS else c) + palabra[i + 1:]


def ruido(plantilla: str, rng: random.Random) -> str:
    t = plantilla.lower()
    t = re.sub(r"[¿¡]", "", t)
    if rng.random() < 0.8:
        t = sin_tildes(t)
    for patron, reemplazo, p in [(r"\bque\b", "q", 0.5), (r"\bpor favor\b", "porfa", 0.7),
                                 (r"\bpara\b", "pa", 0.3), (r"\bestá\b|\besta\b", "sta", 0.2)]:
        if rng.random() < p:
            t = re.sub(patron, reemplazo, t)
    if rng.random() < 0.7:
        tokens = t.split(" ")
        idx = [i for i, w in enumerate(tokens) if len(w) >= 5 and w.isalpha()]
        if idx:
            j = rng.choice(idx)
            tokens[j] = error_tecleo(tokens[j], rng)
        t = " ".join(tokens)
    return t


def main() -> None:
    rng = random.Random(SEMILLA)
    filas = []
    for intencion, plantillas in POOL.items():
        for k, pl in enumerate(plantillas):
            n = 3 if MARCADOR.search(pl) else 1
            vistos = set()
            for _ in range(n * 4):
                texto = instanciar(pl, rng)
                if texto not in vistos:
                    vistos.add(texto)
                if len(vistos) == n:
                    break
            for texto in sorted(vistos):
                filas.append({"texto": texto, "intencion": intencion, "particion": "pool",
                              "grupo": f"{intencion}:{k:02d}"})
    for intencion, plantillas in PRUEBA.items():
        for k, pl in enumerate(plantillas):
            filas.append({"texto": instanciar(pl, rng), "intencion": intencion, "particion": "prueba_limpia",
                          "grupo": f"T-{intencion}:{k:02d}"})
            filas.append({"texto": instanciar(ruido(pl, rng), rng), "intencion": intencion,
                          "particion": "prueba_ruido", "grupo": f"T-{intencion}:{k:02d}"})
    # Plantillas agregadas en la iteración 2, con un generador propio para no alterar las particiones previas.
    rng2 = random.Random(SEMILLA + 2)
    for intencion, plantillas in POOL_ITERACION_2.items():
        for k, pl in enumerate(plantillas):
            vistos = set()
            for _ in range(12):
                vistos.add(instanciar(pl, rng2))
                if len(vistos) == 3:
                    break
            for texto in sorted(vistos):
                filas.append({"texto": texto, "intencion": intencion, "particion": "pool",
                              "grupo": f"{intencion}:it2-{k:02d}", "iteracion": 2})
    for f in filas:
        f.setdefault("iteracion", 1)
    # Control de fuga: ninguna frase de prueba puede aparecer en el pool.
    pool_textos = {f["texto"].lower() for f in filas if f["particion"] == "pool"}
    fugas = [f["texto"] for f in filas if f["particion"] != "pool" and f["texto"].lower() in pool_textos]
    meta = {"semilla": SEMILLA, "intenciones": sorted(POOL), "fugas_exactas_detectadas": fugas,
            "conteos": {p: sum(1 for f in filas if f["particion"] == p) for p in ("pool", "prueba_limpia", "prueba_ruido")},
            "plantillas_pool": sum(len(v) for v in POOL.values()),
            "plantillas_iteracion_2": sum(len(v) for v in POOL_ITERACION_2.values()),
            "pool_iteracion_2": sum(1 for f in filas if f.get("iteracion") == 2)}
    (RAIZ / "corpus.json").write_text(json.dumps({"meta": meta, "datos": filas}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
