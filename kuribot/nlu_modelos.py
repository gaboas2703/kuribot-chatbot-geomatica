"""Modelos de clasificación de intenciones comparados en el proyecto.

M0  TF-IDF (palabras 1-2 + caracteres 2-5) + regresión logística   -> ML clásico (línea base)
M1  Embeddings fastText preentrenados (spaCy es_core_news_lg) + BiLSTM + atención + MLP
M2  BETO congelado (media de la última capa) + MLP
M3  BETO ajustado de extremo a extremo (fine-tuning) + capa lineal   -> modelo de producción

Todos reciben el mismo texto preprocesado (entidades enmascaradas) y las mismas particiones.
"""
from __future__ import annotations

import copy
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from torch import nn

from .entidades import enmascarar_entidades

TOKEN = re.compile(r"[a-záéíóúüñ]+|\d+|[A-ZÁÉÍÓÚÑ]+", re.I)


def fijar_semilla(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def preprocesar(textos: list[str]) -> list[str]:
    return [enmascarar_entidades(t) for t in textos]


def pesos_clase(y: np.ndarray, n: int) -> torch.Tensor:
    conteo = np.bincount(y, minlength=n).astype(float)
    w = conteo.sum() / (n * np.maximum(conteo, 1))
    return torch.tensor(w, dtype=torch.float32)


# ====================================================================== M0


def crear_m0(C: float = 10.0) -> Pipeline:
    feats = FeatureUnion([
        ("palabras", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(lowercase=True, strip_accents="unicode", analyzer="char_wb",
                                       ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", feats), ("lr", LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))])


class ModeloM0:
    nombre = "M0 TF-IDF + RL"

    def __init__(self, n_clases: int, semilla: int = 0):
        self.n = n_clases
        self.pipe = crear_m0()

    def entrenar(self, xtr, ytr, xdev=None, ydev=None):
        self.pipe.fit(preprocesar(xtr), ytr)
        return {}

    def logits(self, textos):
        # log-probabilidades como logits: el softmax recupera las probabilidades de la RL
        return np.log(np.clip(self.pipe.predict_proba(preprocesar(textos)), 1e-12, 1))


# ====================================================================== utilidades de entrenamiento neuronal


def _entrenar_torch(modelo, lotes_tr, evaluar_dev, epocas, lr, pesos, paciencia, wd=0.01, calentamiento=0.0,
                    params=None):
    params = params if params is not None else [p for p in modelo.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    pasos_total = epocas * len(lotes_tr())
    pasos_cal = int(calentamiento * pasos_total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, pasos_cal) if s < pasos_cal else max(0.0, (pasos_total - s) / max(1, pasos_total - pasos_cal)))
    perdida = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.05)
    mejor, mejor_estado, sin_mejora, historia = -1.0, None, 0, []
    for ep in range(epocas):
        modelo.train()
        total, n = 0.0, 0
        for entrada, y in lotes_tr():
            opt.zero_grad()
            salida = modelo(*entrada)
            l = perdida(salida, y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            total += float(l.detach()) * len(y)
            n += len(y)
        f1 = evaluar_dev() if evaluar_dev else float("nan")
        historia.append({"epoca": ep + 1, "perdida_entrenamiento": total / n, "f1_macro_validacion": f1})
        if evaluar_dev is None:
            continue
        if f1 > mejor + 1e-4:
            mejor, mejor_estado, sin_mejora = f1, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return historia


# ====================================================================== M1


def tokenizar(texto: str) -> list[str]:
    return [t.lower() for t in TOKEN.findall(texto)]


class RedBiLSTM(nn.Module):
    def __init__(self, emb: np.ndarray, n_clases: int, oculto: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32), freeze=False, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb.shape[1], oculto, batch_first=True, bidirectional=True)
        self.atencion = nn.Linear(2 * oculto, 1)
        self.mlp = nn.Sequential(nn.Linear(4 * oculto, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_clases))

    def forward(self, ids, mascara):
        h, _ = self.lstm(self.drop(self.embedding(ids)))
        puntaje = self.atencion(h).squeeze(-1).masked_fill(~mascara, -1e9)
        alfa = torch.softmax(puntaje, dim=1).unsqueeze(-1)
        atendido = (alfa * h).sum(1)
        maximo = h.masked_fill(~mascara.unsqueeze(-1), -1e9).max(1).values
        return self.mlp(torch.cat([atendido, maximo], dim=1))


class ModeloM1:
    nombre = "M1 fastText + BiLSTM"

    def __init__(self, n_clases: int, semilla: int = 0, vectores=None, max_len: int = 32):
        self.n, self.semilla, self.max_len = n_clases, semilla, max_len
        self.vectores = vectores  # función token -> vector (np.ndarray) o None
        self.vocab: dict[str, int] = {}
        self.red: RedBiLSTM | None = None
        self.cobertura = float("nan")

    def _ids(self, textos):
        filas = [[self.vocab.get(t, 1) for t in tokenizar(x)][: self.max_len] or [1] for x in preprocesar(textos)]
        L = max(len(f) for f in filas)
        ids = torch.zeros(len(filas), L, dtype=torch.long)
        for i, f in enumerate(filas):
            ids[i, : len(f)] = torch.tensor(f)
        return ids, ids != 0

    def entrenar(self, xtr, ytr, xdev, ydev):
        fijar_semilla(self.semilla)
        toks = sorted({t for x in preprocesar(xtr) for t in tokenizar(x)})
        self.vocab = {"<pad>": 0, "<unk>": 1, **{t: i + 2 for i, t in enumerate(toks)}}
        emb = np.random.default_rng(self.semilla).normal(0, 0.1, (len(self.vocab), 300)).astype(np.float32)
        emb[0] = 0
        cubiertos = 0
        if self.vectores is not None:
            for t, i in self.vocab.items():
                v = self.vectores(t) if i > 1 else None
                if v is not None:
                    emb[i], cubiertos = v, cubiertos + 1
        self.cobertura = cubiertos / max(1, len(self.vocab) - 2)
        self.red = RedBiLSTM(emb, self.n)
        ytr_t = torch.tensor(ytr)
        idx = np.arange(len(xtr))

        epoca = [0]

        def lotes_lista():
            rng = np.random.default_rng(self.semilla + epoca[0])
            epoca[0] += 1
            orden = rng.permutation(idx)
            salida = []
            for i in range(0, len(orden), 32):
                b = orden[i:i + 32]
                ids, m = self._ids([xtr[j] for j in b])
                salida.append(((ids, m), ytr_t[b]))
            return salida

        def dev():
            return f1_score(ydev, self.logits(xdev).argmax(1), average="macro")

        hist = _entrenar_torch(self.red, lotes_lista, dev, epocas=40, lr=2e-3, pesos=pesos_clase(ytr, self.n), paciencia=6)
        return {"historia": hist, "cobertura_embeddings": self.cobertura, "vocabulario": len(self.vocab)}

    @torch.no_grad()
    def logits(self, textos):
        self.red.eval()
        salida = []
        for i in range(0, len(textos), 64):
            ids, m = self._ids(textos[i:i + 64])
            salida.append(self.red(ids, m).numpy())
        return np.concatenate(salida)


# ====================================================================== BETO (M2 y M3)


class CodificadorBETO(nn.Module):
    def __init__(self, ruta: str | Path | None = None, config=None):
        super().__init__()
        from transformers import BertModel
        # Con 'config' se crea la arquitectura vacía (los pesos se cargan después desde el modelo exportado).
        self.bert = (BertModel(config, add_pooling_layer=False) if config is not None
                     else BertModel.from_pretrained(str(ruta), add_pooling_layer=False))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class ClasificadorBETO(nn.Module):
    def __init__(self, ruta, n_clases: int, dropout: float = 0.1, config=None):
        super().__init__()
        self.cod = CodificadorBETO(ruta, config)
        self.drop = nn.Dropout(dropout)
        self.salida = nn.Linear(768, n_clases)

    def forward(self, input_ids, attention_mask):
        return self.salida(self.drop(self.cod(input_ids, attention_mask)))


def _tok(ruta):
    from transformers import BertTokenizerFast
    return BertTokenizerFast.from_pretrained(str(ruta))


class ModeloM2:
    nombre = "M2 BETO congelado + MLP"
    _cache: dict = {}

    def __init__(self, n_clases: int, semilla: int = 0, ruta_beto: str = "modelos/beto_base"):
        self.n, self.semilla, self.ruta = n_clases, semilla, ruta_beto
        self.tok = _tok(ruta_beto)
        if ruta_beto not in ModeloM2._cache:
            ModeloM2._cache[ruta_beto] = (CodificadorBETO(ruta_beto).eval(), {})
        self.cod, self.memo = ModeloM2._cache[ruta_beto]
        self.mlp = None

    @torch.no_grad()
    def _vec(self, textos):
        pre = preprocesar(textos)
        faltan = [t for t in dict.fromkeys(pre) if t not in self.memo]
        for i in range(0, len(faltan), 32):
            enc = self.tok(faltan[i:i + 32], padding=True, truncation=True, max_length=48, return_tensors="pt")
            v = self.cod(enc["input_ids"], enc["attention_mask"])
            for t, fila in zip(faltan[i:i + 32], v):
                self.memo[t] = fila
        return torch.stack([self.memo[t] for t in pre])

    def entrenar(self, xtr, ytr, xdev, ydev):
        fijar_semilla(self.semilla)
        Xtr = self._vec(xtr)
        # Estandarización con media y desviación de entrenamiento: las salidas de BERT son muy anisótropas
        # (coseno medio ≈ 0,9 entre frases), lo que impide aprender a una MLP sin normalizar.
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-4)
        Xtr = (Xtr - self.mu) / self.sd
        Xdev = (self._vec(xdev) - self.mu) / self.sd
        self.mlp = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, self.n))
        ytr_t = torch.tensor(ytr)

        def lotes():
            orden = torch.randperm(len(ytr))
            return [((Xtr[orden[i:i + 32]],), ytr_t[orden[i:i + 32]]) for i in range(0, len(ytr), 32)]

        def dev():
            self.mlp.eval()
            with torch.no_grad():
                return f1_score(ydev, self.mlp(Xdev).argmax(1).numpy(), average="macro")

        hist = _entrenar_torch(self.mlp, lotes, dev, epocas=150, lr=1e-3, pesos=pesos_clase(ytr, self.n), paciencia=15)
        return {"historia": hist}

    @torch.no_grad()
    def logits(self, textos):
        self.mlp.eval()
        return self.mlp((self._vec(textos) - self.mu) / self.sd).numpy()


class ModeloM3:
    nombre = "M3 BETO ajustado (fine-tuning)"

    def __init__(self, n_clases: int, semilla: int = 0, ruta_beto: str = "modelos/beto_base", epocas: int = 6,
                 lr: float = 3e-5, lote: int = 16, paciencia: int = 2):
        self.n, self.semilla, self.ruta = n_clases, semilla, ruta_beto
        self.epocas, self.lr, self.lote, self.paciencia = epocas, lr, lote, paciencia
        self.tok = _tok(ruta_beto)
        self.red = None

    def _enc(self, textos):
        return self.tok(preprocesar(textos), padding=True, truncation=True, max_length=48, return_tensors="pt")

    def entrenar(self, xtr, ytr, xdev, ydev):
        fijar_semilla(self.semilla)
        self.red = ClasificadorBETO(self.ruta, self.n)
        ytr_t = torch.tensor(ytr)

        def lotes():
            orden = torch.randperm(len(ytr)).tolist()
            out = []
            for i in range(0, len(orden), self.lote):
                b = orden[i:i + self.lote]
                enc = self._enc([xtr[j] for j in b])
                out.append(((enc["input_ids"], enc["attention_mask"]), ytr_t[b]))
            return out

        def dev():
            return f1_score(ydev, self.logits(xdev).argmax(1), average="macro") if xdev is not None else float("nan")

        hist = _entrenar_torch(self.red, lotes, dev if xdev is not None else None, epocas=self.epocas, lr=self.lr,
                               pesos=pesos_clase(ytr, self.n), paciencia=self.paciencia, wd=0.01, calentamiento=0.1)
        return {"historia": hist}

    @torch.no_grad()
    def logits(self, textos):
        self.red.eval()
        salida = []
        for i in range(0, len(textos), 32):
            enc = self._enc(textos[i:i + 32])
            salida.append(self.red(enc["input_ids"], enc["attention_mask"]).numpy())
        return np.concatenate(salida)


# ====================================================================== calibración y métricas


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ajustar_temperatura(logits_dev: np.ndarray, y_dev: np.ndarray) -> float:
    """Escalado de temperatura (Guo et al., 2017): minimiza la log-verosimilitud negativa en validación."""
    z = torch.tensor(logits_dev, dtype=torch.float64)
    y = torch.tensor(y_dev)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def cierre():
        opt.zero_grad()
        l = nn.functional.cross_entropy(z / logT.exp(), y)
        l.backward()
        return l

    opt.step(cierre)
    return float(logT.detach().exp().clamp(0.05, 20))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bordes = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(bordes[:-1], bordes[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def latencia_ms(modelo, textos: list[str], repeticiones: int = 1) -> dict:
    tiempos = []
    for _ in range(repeticiones):
        for t in textos:
            t0 = time.perf_counter()
            modelo.logits([t])
            tiempos.append((time.perf_counter() - t0) * 1000)
    a = np.array(tiempos)
    return {"media_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}


def n_parametros(modelo) -> int:
    if isinstance(modelo, ModeloM0):
        lr = modelo.pipe.named_steps["lr"]
        return int(lr.coef_.size + lr.intercept_.size)
    if isinstance(modelo, ModeloM1):
        return sum(p.numel() for p in modelo.red.parameters())
    if isinstance(modelo, ModeloM2):
        return sum(p.numel() for p in modelo.mlp.parameters())
    if isinstance(modelo, ModeloM3):
        return sum(p.numel() for p in modelo.red.parameters())
    return -1


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)
