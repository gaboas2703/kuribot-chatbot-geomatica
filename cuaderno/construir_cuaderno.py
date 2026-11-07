"""Genera cuaderno/KuriBot_Colab.ipynb: reproduce el proyecto completo en Google Colab."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
ARCHIVOS = ["kuribot/__init__.py", "kuribot/geodesia.py", "kuribot/entidades.py", "kuribot/conocimiento.py", "kuribot/nlu_modelos.py",
            "kuribot/nlu.py", "kuribot/razonamiento.py", "kuribot/dialogo.py", "kuribot/mapa.py", "datos/corpus_fuente.py",
            "datos/construir_base.py", "datos/construir_corpus.py", "scripts/extraer_beto.py", "experimentos/entrenar_nlp.py",
            "experimentos/figuras.py", "pruebas/test_unitarias.py", "pruebas/suite_aceptacion.py", "cli.py", "app_streamlit.py",
            "requirements.txt"]

celdas = [
    md("""# KuriBot · ChatBot híbrido de Geomática (UEES · Inteligencia Artificial · Semana 4)

Este cuaderno reproduce el proyecto de principio a fin en Google Colab (CPU o GPU):

| Paso | Qué hace |
|---|---|
| 1 | Instala dependencias y descarga BETO desde la versión oficial de spaCy en GitHub |
| 2 | Escribe el código fuente del proyecto (idéntico al Anexo C del informe) |
| 3 | Construye la base de conocimiento y el corpus sintéticos |
| 4 | Entrena y compara los cuatro modelos (iteración 1) y el modelo ajustado (iteración 2) |
| 5 | Ejecuta las pruebas unitarias y la suite de aceptación de 34 escenarios |
| 6 | Conversa con el ChatBot |

**Datos:** todos son sintéticos. **Tiempo de referencia:** en el entorno de desarrollo (CPU de 2 hilos, como el de Colab gratuito) la validación cruzada de los cuatro modelos tardó unos 17 min y los modelos finales de la iteración 1, unos 5 min más; la iteración 2 duplica los datos de entrenamiento y tarda algo más. Con GPU (*Entorno de ejecución → Cambiar tipo*) el tiempo baja."""),
    md("## 1. Dependencias"),
    code("""!apt-get -qq install -y graphviz > /dev/null
!pip -q install pgmpy pyproj shapely streamlit safetensors
!pip -q install --no-deps https://github.com/explosion/spacy-models/releases/download/es_dep_news_trf-3.8.0/es_dep_news_trf-3.8.0-py3-none-any.whl
!pip -q install --no-deps https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.8.0/es_core_news_lg-3.8.0-py3-none-any.whl
import os; os.makedirs("kuribot/kuribot", exist_ok=True)
for d in ["datos", "scripts", "experimentos", "pruebas", "modelos", "resultados", "docs/figuras"]:
    os.makedirs(f"kuribot/{d}", exist_ok=True)
%cd kuribot"""),
    md("## 2. Código fuente"),
]
for a in ARCHIVOS:
    celdas.append(code(f"%%writefile {a}\n" + (RAIZ / a).read_text(encoding="utf-8")))
celdas += [
    md("## 3. BETO, base de conocimiento y corpus"),
    code("""!python scripts/extraer_beto.py --salida modelos/beto_base
!python datos/construir_base.py
!python datos/construir_corpus.py"""),
    md("## 4. Modelado\nIteración 1: validación cruzada de M0–M3 y modelos finales. Iteración 2: ajustes tras el análisis de fallos."),
    code("""!python experimentos/entrenar_nlp.py --version v1
!python experimentos/entrenar_nlp.py --version v2 --aumento --sin-cv --modelos M0,M3 --iteracion 2 --epocas-m3 10 --paciencia-m3 3"""),
    md("## 5. Pruebas"),
    code("""!python -m pytest pruebas/test_unitarias.py -q
!python pruebas/suite_aceptacion.py --version v2
!python experimentos/figuras.py --version v1 --v2 v2"""),
    md("## 6. Conversar con KuriBot"),
    code("""from kuribot.dialogo import KuriBot
bot = KuriBot()
for t in ["hola", "hay una retroexcavadora y gente extraña trabajando en Cerro Azul",
          "¿el punto E 642600 N 9662480 PSAD56 en qué concesión queda?", "necesito un vuelo de dron en San Gregorio", "mañana"]:
    r = bot.responder(t)
    print("Usted  ›", t)
    print("KuriBot›", r.texto)
    print("        [", r.intencion, f"{r.confianza:.2f}", r.decision, r.reglas, "]\\n")"""),
]
nb = nbf.v4.new_notebook(cells=celdas, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "language_info": {"name": "python"}, "colab": {"provenance": []}})
nbf.write(nb, RAIZ / "cuaderno" / "KuriBot_Colab.ipynb")
print("Cuaderno generado con", len(celdas), "celdas")
