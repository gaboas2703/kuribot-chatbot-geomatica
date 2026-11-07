# KuriBot · Asistente de Geomática y Catastro minero

Prototipo académico de ChatBot híbrido (Maestría en Inteligencia de Negocios y Ciencia de Datos, Universidad
Espíritu Santo, asignatura Inteligencia Artificial, Semana 4). Atiende consultas internas del área de Geomática de
una empresa minera **ficticia** del sur del Ecuador: fichas de concesiones, verificación de puntos en el catastro,
conversión de coordenadas WGS 84 ↔ PSAD56, reportes de posible minería ilegal, solicitudes de vuelo de dron y de
cartografía, seguimiento de tickets y dudas geodésicas frecuentes.

**Todos los datos (concesiones, polígonos, clima, flota, tickets) son sintéticos.**

Equipo: Gabriel Ayala Sánchez, Daniel Castro M. y Ricardo Ruiz L. · Docente: Jorge Luis Zambrano Martínez, Ph.D.

Informe del proyecto: [`docs/informe/informe.pdf`](docs/informe/informe.pdf) (versión editable en
[`docs/informe/informe.docx`](docs/informe/informe.docx)).

## Arquitectura en una línea
NLU con **BETO ajustado** (red neuronal transformer, 14 intenciones) + extractor de entidades por reglas
→ gestor de diálogo con ranuras → **red bayesiana** de 9 nodos (pgmpy) + **motor de reglas** con
encadenamiento hacia adelante (R01–R42, V01–V06) + geoprocesos (pyproj EPSG:3990, Shapely) →
respuestas por plantillas con datos verificados y trazabilidad completa.

## Ejecución rápida (Windows, macOS o Linux, Python 3.10–3.13)
```bash
python -m pip install -r requirements.txt
python scripts/unir_modelo.py         # una sola vez: reconstruye el modelo BETO ajustado
streamlit run app_streamlit.py        # interfaz web en http://localhost:8501
python cli.py                         # interfaz de consola
```
GitHub no admite archivos de más de 100 MB, así que el modelo ajustado (≈ 220 MB, float16) se guarda en cinco
partes en `modelos/partes_beto_v2/`, con sus sumas SHA-256. `scripts/unir_modelo.py` las une, verifica la
integridad y crea `modelos/intenciones_beto_v2/`. Si el modelo no está, la aplicación usa automáticamente la
línea base TF-IDF (`modelos/intenciones_tfidf_v2.joblib`) y lo indica en la barra lateral.

## Reproducir los experimentos
```bash
# 1) Obtener BETO desde la versión oficial de spaCy en GitHub y convertirlo a formato Hugging Face
pip install https://github.com/explosion/spacy-models/releases/download/es_dep_news_trf-3.8.0/es_dep_news_trf-3.8.0-py3-none-any.whl --no-deps
pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_lg-3.8.0/es_core_news_lg-3.8.0-py3-none-any.whl --no-deps
python scripts/extraer_beto.py --salida modelos/beto_base
# 2) Datos sintéticos
python datos/construir_base.py && python datos/construir_corpus.py
# 3) NLP: validación cruzada de 4 modelos + modelo final (iteración 1) y ajuste (iteración 2)
python experimentos/entrenar_nlp.py --version v1
python experimentos/entrenar_nlp.py --version v2 --aumento --sin-cv --modelos M0,M3 --iteracion 2 --epocas-m3 10 --paciencia-m3 3
python experimentos/complementos.py   # latencia de M2 sin caché y McNemar del modelo final
# 4) Pruebas
python -m pytest pruebas/test_unitarias.py -v
python pruebas/suite_aceptacion.py --version v2
python pruebas/suite_aceptacion.py --version v2 --confirmacion   # 8 escenarios nuevos
# 5) Figuras, capturas e informe
python experimentos/figuras.py --version v1 --v2 v2
streamlit run app_streamlit.py --server.headless true &   # y luego: python docs/capturar.py
python docs/informe/construir_informe.py                  # DOCX con el formato UEES y PDF (Pandoc + LibreOffice)
```
En Google Colab: abrir `cuaderno/KuriBot_Colab.ipynb` y ejecutar todas las celdas.

## Estructura
| Ruta | Contenido |
|---|---|
| `kuribot/entidades.py` | Extracción de coordenadas, datum, concesión, ticket, fecha, evidencias (con negación) y datos personales |
| `kuribot/nlu_modelos.py` | Modelos M0–M3, calibración por temperatura, ECE, latencia |
| `kuribot/nlu.py` | Inferencia de intenciones con decisión aceptar / aclarar / abstener |
| `kuribot/razonamiento.py` | Motor de reglas, red bayesiana, sensibilidad y ventanas de vuelo |
| `kuribot/geodesia.py` | Transformaciones PSAD56 ↔ WGS 84 (EPSG:3990) y utilidades UTM |
| `kuribot/conocimiento.py` | Catastro, operación, FAQ y tickets (SQLite) |
| `kuribot/dialogo.py` | Gestor de diálogo, políticas por intención y generación de respuestas |
| `app_streamlit.py`, `cli.py` | Interfaces web y de consola |
| `datos/` | Corpus fuente, constructor del corpus y base sintética |
| `experimentos/` | Entrenamiento, evaluación y figuras |
| `pruebas/` | Pruebas unitarias (34, pytest), suite de aceptación (34 escenarios) y confirmación (8) |
| `resultados/` | Métricas, predicciones y evidencias de cada versión |
| `modelos/` | Línea base TF-IDF y partes del modelo BETO ajustado |
| `scripts/` | Extracción de BETO desde spaCy y reconstrucción del modelo ajustado |
| `docs/` | Figuras, capturas e informe (fuentes Markdown, plantilla de formato y generador) |
| `cuaderno/` | Cuaderno para Google Colab |

## Límites
No usar en producción. No conecta con catastro oficial, ARCOM, servicios meteorológicos ni sistemas de
seguridad reales; los tickets y derivaciones son simulados. Las TPC de la red bayesiana y los umbrales
son supuestos expertos a calibrar. El pipeline de spaCy del que se extrae BETO se distribuye con
licencia GPL-3.0, condición que hereda el modelo ajustado de `modelos/partes_beto_v2/`.
