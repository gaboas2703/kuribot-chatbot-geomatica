"""Capturas reales de la interfaz Streamlit (Playwright) y de la consola (salida real renderizada como imagen).

Requiere la aplicación en ejecución:  streamlit run app_streamlit.py --server.headless true --server.port 8501
Uso: python docs/capturar.py
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
CAP = RAIZ / "docs" / "capturas"
CAP.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"

ESCENAS = [
    ("captura_1_consulta", ["Hola", "¿Cuál es el estado de la concesión La Ventana?"]),
    ("captura_2_alerta", ["Hay una retroexcavadora y gente extraña trabajando en Cerro Azul"]),
    ("captura_3_punto", ["¿El punto E 643100 N 9662100 PSAD56 está dentro de alguna concesión?"]),
    ("captura_4_vuelo", ["Necesito un vuelo de dron en San Gregorio", "mañana"]),
    ("captura_5_emergencia", ["Hay gente armada disparando cerca del campamento de Los Laureles"]),
]


def esperar_respuesta(page, n_mensajes: int):
    page.wait_for_function(f"document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= {n_mensajes}", timeout=120000)
    page.wait_for_timeout(2500)


def recortar(ruta: Path):
    """Elimina el espacio vacío inferior de la captura (la conversación no ocupa todo el alto de la ventana)."""
    from PIL import Image, ImageChops
    img = Image.open(ruta).convert("RGB")
    ancho, alto = img.size
    zona = img.crop((int(ancho * 0.22), 0, ancho, alto))
    fondo = Image.new("RGB", zona.size, zona.getpixel((zona.size[0] - 5, alto - 5)))
    caja = ImageChops.difference(zona, fondo).getbbox()
    if caja:
        img.crop((0, 0, ancho, min(alto, caja[3] + 40))).save(ruta)


def web():
    with sync_playwright() as p:
        nav = p.chromium.launch()
        page = nav.new_page(viewport={"width": 1440, "height": 2400}, device_scale_factor=1.25)
        for nombre, mensajes in ESCENAS:
            page.goto(URL)
            page.wait_for_selector("[data-testid='stChatInput'] textarea", timeout=180000)
            boton = page.get_by_role("button", name="🔄 Nueva conversación")
            if boton.count():
                boton.click()
                page.wait_for_timeout(1500)
            for i, m in enumerate(mensajes):
                caja = page.locator("[data-testid='stChatInput'] textarea")
                caja.fill(m)
                caja.press("Enter")
                esperar_respuesta(page, 2 * (i + 1))
            page.evaluate("""() => { for (const s of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main'])
                                 { const e = document.querySelector(s); if (e) e.scrollTo(0, 0); } window.scrollTo(0, 0); }""")
            page.wait_for_timeout(800)
            ruta = CAP / f"{nombre}.png"
            fondo = page.evaluate("""() => Math.max(...[...document.querySelectorAll('[data-testid="stChatMessage"]')]
                                      .map(e => e.getBoundingClientRect().bottom))""")
            page.screenshot(path=str(ruta), clip={"x": 0, "y": 0, "width": 1440, "height": min(2400, int(fondo) + 30)})
            print("captura", nombre)
        nav.close()


def consola():
    """Ejecuta la consola real con entradas predefinidas y dibuja la salida en una imagen."""
    entradas = "\n".join(["buenos días", "convierte E 642600 N 9662480 de PSAD56 a WGS84",
                          "se ve desbroce nuevo y maquinaria en Peña Blanca", "estado del ticket GEO-2026-0999", "salir"]) + "\n"
    r = subprocess.run([sys.executable, "cli.py"], input=entradas, capture_output=True, text=True, cwd=RAIZ, timeout=600)
    texto = r.stdout
    (CAP / "consola_salida.txt").write_text(texto, encoding="utf-8")
    from PIL import Image, ImageDraw, ImageFont
    lineas = []
    for l in texto.splitlines():
        lineas.extend(textwrap.wrap(l, 118) or [""])
    fuente = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 15)
    alto = 24 * len(lineas) + 60
    img = Image.new("RGB", (1240, alto), "#1a1a19")
    d = ImageDraw.Draw(img)
    d.text((20, 14), "$ python cli.py", fill="#86b6ef", font=fuente)
    for i, l in enumerate(lineas):
        color = "#f0efec" if not l.strip().startswith("[") else "#c3c2b7"
        if l.startswith("Usted ›"):
            color = "#eda100"
        d.text((20, 44 + 24 * i), l, fill=color, font=fuente)
    img.save(CAP / "captura_6_consola.png")
    print("captura consola")


if __name__ == "__main__":
    web()
    consola()
