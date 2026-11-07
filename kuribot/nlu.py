"""Inferencia de intenciones para el ChatBot (carga el modelo exportado por experimentos/entrenar_nlp.py)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .entidades import enmascarar_entidades

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "modelos"


@dataclass
class Prediccion:
    intencion: str
    confianza: float
    top: list[tuple[str, float]]
    decision: str          # aceptar | aclarar | abstener
    modelo: str
    latencia_ms: float


def _softmax(z: np.ndarray, T: float) -> np.ndarray:
    z = z / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class ClasificadorIntenciones:
    """Envuelve el BETO ajustado (preferido) o la línea base TF-IDF si no hay pesos disponibles."""

    def __init__(self, ruta: str | Path | None = None, umbral_aclarar: float | None = None, umbral_abstener: float = 0.30):
        ruta = Path(ruta) if ruta else self._ultimo("intenciones_beto_v*")
        if ruta is not None and (ruta / "clasificador.safetensors").exists():
            self._cargar_beto(ruta)
        else:
            base = self._ultimo("intenciones_tfidf_v*.joblib")
            if base is None:
                raise FileNotFoundError("No hay modelos entrenados en modelos/. Ejecute experimentos/entrenar_nlp.py")
            self._cargar_tfidf(base)
        self.umbral_aclarar = umbral_aclarar if umbral_aclarar is not None else max(0.5, self.tau)
        self.umbral_abstener = umbral_abstener
        self._logits("hola")  # calentamiento: la primera inferencia de PyTorch tras cargar el modelo es lenta

    @staticmethod
    def _ultimo(patron: str) -> Path | None:
        c = sorted(MODELOS.glob(patron))
        return c[-1] if c else None

    def _cargar_beto(self, ruta: Path):
        import torch
        from safetensors.torch import load_file
        from transformers import BertConfig, BertTokenizerFast

        from .nlu_modelos import ClasificadorBETO

        meta = json.loads((ruta / "kuribot_nlu.json").read_text(encoding="utf-8"))
        self.etiquetas, self.T, self.tau = meta["etiquetas"], meta["temperatura"], meta["umbral_confianza"]
        self.max_len = meta.get("max_len", 48)
        cfg = BertConfig.from_json_file(str(ruta / "bert_config.json"))
        red = ClasificadorBETO(None, len(self.etiquetas), config=cfg)
        estado = {k: v.float() for k, v in load_file(str(ruta / "clasificador.safetensors")).items()}
        red.load_state_dict(estado)
        red.eval()
        self.red, self.tok, self.torch = red, BertTokenizerFast.from_pretrained(str(ruta)), torch
        self.nombre = f"BETO ajustado ({ruta.name})"
        self._logits = self._logits_beto

    def _cargar_tfidf(self, ruta: Path):
        import joblib
        d = joblib.load(ruta)
        self.pipe, self.etiquetas, self.T, self.tau = d["pipe"], d["etiquetas"], d["temperatura"], 0.5
        self.nombre = f"TF-IDF + RL ({ruta.name}, modo de respaldo)"
        self._logits = lambda t: np.log(np.clip(self.pipe.predict_proba([t])[0], 1e-12, 1))

    def _logits_beto(self, texto: str) -> np.ndarray:
        with self.torch.no_grad():
            enc = self.tok([texto], truncation=True, max_length=self.max_len, return_tensors="pt")
            return self.red(enc["input_ids"], enc["attention_mask"])[0].numpy()

    def predecir(self, texto: str) -> Prediccion:
        t0 = time.perf_counter()
        p = _softmax(self._logits(enmascarar_entidades(texto)), self.T)
        orden = np.argsort(-p)
        top = [(self.etiquetas[i], float(p[i])) for i in orden[:3]]
        conf = top[0][1]
        if conf >= self.umbral_aclarar:
            decision = "aceptar"
        elif conf >= self.umbral_abstener and top[0][1] + top[1][1] >= 0.70:
            decision = "aclarar"
        else:
            decision = "abstener"
        return Prediccion(top[0][0], conf, top, decision, self.nombre, (time.perf_counter() - t0) * 1000)
