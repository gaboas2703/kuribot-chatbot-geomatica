"""Extrae el codificador BETO del pipeline oficial de spaCy y lo guarda en formato Hugging Face.

Motivo: el entorno de desarrollo no tiene acceso a huggingface.co, pero sí a las versiones
publicadas de los modelos de spaCy en GitHub. El pipeline ``es_dep_news_trf`` 3.8.0 incluye
el transformer BETO (dccuchile/bert-base-spanish-wwm-cased; Cañete et al., 2020), ajustado
por Explosion durante el entrenamiento del analizador morfosintáctico de spaCy.

El script:
1. lee el archivo serializado ``transformer/model`` (msgpack de Thinc);
2. convierte los nombres de parámetros de curated-transformers a ``BertModel`` de HF
   (la proyección fusionada QKV se divide en query, key y value, en ese orden);
3. guarda config.json, vocab.txt, tokenizer y model.safetensors;
4. con ``--verificar`` compara las salidas contra la implementación original de
   curated-transformers 0.1.1 (diferencia absoluta máxima esperada < 1e-4).

Uso:
    pip install es_dep_news_trf-3.8.0-py3-none-any.whl --no-deps
    python scripts/extraer_beto.py --salida modelos/beto_base [--verificar RUTA_CT011]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path

import srsly
import torch


def ubicar_modelo_spacy() -> Path:
    spec = importlib.util.find_spec("es_dep_news_trf")
    if spec is None or spec.origin is None:
        raise SystemExit("Instale primero es_dep_news_trf-3.8.0 (GitHub explosion/spacy-models).")
    raiz = Path(spec.origin).parent
    candidatos = list(raiz.glob("es_dep_news_trf-*/transformer/model"))
    if not candidatos:
        raise SystemExit("No se encontró transformer/model dentro del paquete.")
    return candidatos[0]


def leer_pesos(ruta: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    datos = srsly.msgpack_loads(ruta.read_bytes())
    shim = srsly.msgpack_loads(datos["shims"][6][0])
    estado = torch.load(io.BytesIO(shim["state"]), map_location="cpu", weights_only=False)
    vocab = datos["attrs"][3]["wordpiece_processor"].decode("utf-8").split("\n")
    return estado, vocab


def convertir(estado: dict[str, torch.Tensor], n_capas: int = 12) -> dict[str, torch.Tensor]:
    pref = "curated_encoder."
    hf: dict[str, torch.Tensor] = {}
    emb = {
        "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
    }
    for destino, origen in emb.items():
        hf[destino] = estado[pref + origen]
    for i in range(n_capas):
        o = f"{pref}layers.{i}."
        d = f"encoder.layer.{i}."
        w_qkv, b_qkv = estado[o + "mha.input.weight"], estado[o + "mha.input.bias"]
        for nombre, w, b in zip(("query", "key", "value"), w_qkv.chunk(3, dim=0), b_qkv.chunk(3, dim=0)):
            hf[d + f"attention.self.{nombre}.weight"] = w.clone()
            hf[d + f"attention.self.{nombre}.bias"] = b.clone()
        pares = {
            "attention.output.dense": "mha.output",
            "attention.output.LayerNorm": "attn_output_layernorm",
            "intermediate.dense": "ffn.intermediate",
            "output.dense": "ffn.output",
            "output.LayerNorm": "ffn_output_layernorm",
        }
        for destino, origen in pares.items():
            hf[d + destino + ".weight"] = estado[o + origen + ".weight"]
            hf[d + destino + ".bias"] = estado[o + origen + ".bias"]
    return hf


def guardar(hf_estado: dict[str, torch.Tensor], vocab: list[str], salida: Path) -> None:
    from transformers import BertConfig, BertModel, BertTokenizerFast

    salida.mkdir(parents=True, exist_ok=True)
    (salida / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    config = BertConfig(
        vocab_size=len(vocab), hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2,
        layer_norm_eps=1e-12, pad_token_id=vocab.index("[PAD]"),
    )
    modelo = BertModel(config, add_pooling_layer=False)
    faltan, sobran = modelo.load_state_dict(hf_estado, strict=False)
    faltan = [k for k in faltan if "position_ids" not in k]
    if faltan or sobran:
        raise RuntimeError(f"Conversión incompleta. Faltan: {faltan}; sobran: {sobran}")
    modelo.save_pretrained(salida)
    import inspect
    opciones = dict(do_lower_case=False, strip_accents=False, tokenize_chinese_chars=True)
    if "vocab" in inspect.signature(BertTokenizerFast.__init__).parameters:   # transformers >= 5
        tok = BertTokenizerFast(vocab={p: i for i, p in enumerate(vocab)}, **opciones)
    else:                                                                       # transformers 4.x
        tok = BertTokenizerFast(vocab_file=str(salida / "vocab.txt"), **opciones)
    prueba = tok("Hay maquinaria cerca del lindero de la concesión")["input_ids"]
    if tok.unk_token_id in prueba or tok.cls_token_id != vocab.index("[CLS]"):
        raise RuntimeError("El tokenizador no reproduce el vocabulario de BETO.")
    tok.save_pretrained(salida)
    (salida / "PROCEDENCIA.json").write_text(json.dumps({
        "origen": "spaCy es_dep_news_trf 3.8.0 (Explosion), componente transformer",
        "modelo_base": "dccuchile/bert-base-spanish-wwm-cased (BETO; Cañete et al., 2020)",
        "nota": "Pesos ajustados por spaCy para análisis morfosintáctico (UD AnCora); "
                "no son idénticos al BETO original publicado en Hugging Face.",
        "licencia_pipeline": "GNU GPL 3.0 (por los datos AnCora del pipeline)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def verificar(estado, salida: Path, ruta_ct011: str) -> float:
    """Compara la salida HF con la implementación original curated-transformers 0.1.1."""
    import sys
    from transformers import BertModel, BertTokenizerFast

    sys.path.insert(0, ruta_ct011)
    from curated_transformers.models.bert.config import BertConfig as CTConfig  # type: ignore
    from curated_transformers.models.bert.encoder import BertEncoder  # type: ignore
    from curated_transformers.models.attention import AttentionMask  # type: ignore

    cfg = CTConfig(vocab_size=31002, hidden_width=768, intermediate_width=3072,
                   num_attention_heads=12, num_hidden_layers=12, hidden_act="gelu",
                   layer_norm_eps=1e-12, max_position_embeddings=512, type_vocab_size=2,
                   padding_idx=1, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
    ct = BertEncoder(cfg)
    ct.load_state_dict({k.replace("curated_encoder.", ""): v for k, v in estado.items()})
    ct.eval()
    hf = BertModel.from_pretrained(salida, add_pooling_layer=False).eval()
    tok = BertTokenizerFast.from_pretrained(salida)
    frases = ["Necesito convertir coordenadas de PSAD56 a WGS84.",
              "Hay maquinaria no autorizada cerca del lindero de la concesión."]
    enc = tok(frases, padding=True, return_tensors="pt")
    with torch.no_grad():
        salida_hf = hf(**enc).last_hidden_state
        mascara = AttentionMask(enc["attention_mask"].bool())
        salida_ct = ct(enc["input_ids"], mascara, token_type_ids=enc["token_type_ids"])
        ultimo = salida_ct.last_hidden_layer_states
        salida_ct = ultimo() if callable(ultimo) else ultimo
    m = enc["attention_mask"].bool()
    return float((salida_hf[m] - salida_ct[m]).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="modelos/beto_base")
    ap.add_argument("--verificar", default=None, help="Ruta a curated-transformers 0.1.1 instalado con --target")
    args = ap.parse_args()
    estado, vocab = leer_pesos(ubicar_modelo_spacy())
    hf_estado = convertir(estado)
    guardar(hf_estado, vocab, Path(args.salida))
    print(f"Modelo guardado en {args.salida} ({len(vocab)} piezas de vocabulario).")
    if args.verificar:
        dif = verificar(estado, Path(args.salida), args.verificar)
        print(f"Diferencia absoluta máxima HF vs curated-transformers: {dif:.2e}")
        if dif > 1e-3:
            raise SystemExit("La conversión no reproduce la salida original.")


if __name__ == "__main__":
    main()
