"""Reconstruye el modelo BETO ajustado a partir de las partes publicadas en el repositorio.

GitHub no admite archivos de más de 100 MB, por lo que KuriBot_Modelo_BETO_v2.zip (≈ 200 MB) se guarda en cinco
partes en modelos/partes_beto_v2/. Este script las une, verifica la suma SHA-256 de cada parte y del archivo
completo, y descomprime el resultado en modelos/intenciones_beto_v2/.

Uso (Windows, macOS o Linux): python scripts/unir_modelo.py
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PARTES = RAIZ / "modelos" / "partes_beto_v2"
DESTINO = RAIZ / "modelos" / "intenciones_beto_v2"
NOMBRE = "KuriBot_Modelo_BETO_v2.zip"


def sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def main() -> None:
    if (DESTINO / "clasificador.safetensors").exists():
        print(f"El modelo ya existe en {DESTINO.relative_to(RAIZ)}; no se hace nada.")
        return
    esperadas = {}
    for linea in (PARTES / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if linea.strip():
            suma, nombre = linea.split()
            esperadas[nombre] = suma
    partes = sorted(PARTES.glob(f"{NOMBRE}.part*"), key=lambda p: int(p.suffix.replace(".part", "")))
    if not partes:
        sys.exit(f"No se encontraron partes en {PARTES}")
    contenido = bytearray()
    for p in partes:
        datos = p.read_bytes()
        if sha256(datos) != esperadas.get(p.name):
            sys.exit(f"La parte {p.name} está incompleta o dañada (SHA-256 distinta). Vuelva a descargarla.")
        contenido.extend(datos)
    if sha256(bytes(contenido)) != esperadas[NOMBRE]:
        sys.exit("El archivo reconstruido no coincide con la suma SHA-256 publicada.")
    zip_tmp = PARTES / NOMBRE
    zip_tmp.write_bytes(contenido)
    with zipfile.ZipFile(zip_tmp) as z:
        z.extractall(RAIZ)
    zip_tmp.unlink()
    print(f"Modelo reconstruido y verificado en {DESTINO.relative_to(RAIZ)} ({len(partes)} partes, {len(contenido) / 1e6:.0f} MB).")


if __name__ == "__main__":
    main()
