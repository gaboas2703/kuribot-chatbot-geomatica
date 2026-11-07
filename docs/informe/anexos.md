# Anexos

## A Manual básico de usuario

### Instalación e inicio

1. Instale Python 3.10 o superior (en Windows, marque «Add Python to PATH»).
2. Descargue el proyecto desde el repositorio (botón *Code* → *Download ZIP*, o `git clone`).
3. En una terminal abierta en la carpeta del proyecto, ejecute `python -m pip install -r requirements.txt` y, una sola vez, `python scripts/unir_modelo.py`. Este script une las cinco partes del modelo guardadas en `modelos/partes_beto_v2/`, porque GitHub no admite archivos de más de 100 MB, verifica sus sumas SHA-256 y crea `modelos/intenciones_beto_v2/`. Sin el modelo, la aplicación funciona con la línea base TF-IDF y lo indica en la barra lateral.
4. Interfaz web: `streamlit run app_streamlit.py` (se abre `http://localhost:8501`). Consola: `python cli.py`; escriba `salir` para terminar.

### Uso de la interfaz web

La pantalla tiene tres zonas ({{figura:captura1}}). La barra lateral muestra el aviso de asistente automatizado, el modelo cargado, el interruptor de trazabilidad, mensajes de ejemplo que se envían con un clic y el botón «Nueva conversación». La zona central contiene la conversación y el cuadro de mensajes. Bajo cada respuesta, el desplegable «Trazabilidad de la respuesta» muestra la intención y su confianza, las tres intenciones más probables, las entidades, las reglas disparadas, la red bayesiana (posterior y evidencias), las ventanas de vuelo y, cuando corresponde, un mapa del catastro sintético ({{figura:captura2}} a {{figura:captura5}}). La consola imprime la misma información en una línea técnica ({{figura:captura6}}).

<<figura:captura1|../capturas/captura_1_consulta.png|66|Interfaz web: saludo, aviso de transparencia y consulta de una concesión>>
<<nota|Captura de la ejecución real del prototipo.>>

<<figura:captura2|../capturas/captura_2_alerta_recorte.png|44|Reporte de alerta con prioridad CRÍTICA: posterior bayesiano, evidencias, reglas disparadas y mapa>>
<<nota|Captura de la ejecución real del prototipo.>>

<<figura:captura3|../capturas/captura_3_punto_recorte.png|72|Verificación de un punto en PSAD56 con advertencia de riesgo de datum>>
<<nota|Leídas como WGS 84, las mismas cifras caerían en otra concesión. Captura de la ejecución real del prototipo.>>

<<figura:captura4|../capturas/captura_4_vuelo_recorte.png|62|Solicitud de vuelo en dos turnos: el bot pide la fecha, descarta el día con viento y propone la primera ventana apta>>
<<nota|Captura de la ejecución real del prototipo.>>

<<figura:captura5|../capturas/captura_5_emergencia_recorte.png|70|Emergencia: la regla R01 prevalece, no se despacha personal ni dron y el caso se deriva a una persona>>
<<nota|Captura de la ejecución real del prototipo.>>

<<figura:captura6|../capturas/captura_6_consola.png|72|Interfaz de consola con la línea técnica de cada respuesta>>
<<nota|Captura de la ejecución real del prototipo.>>

### Datos de demostración y buenas prácticas

<<tabla:demo|Concesiones, coordenadas y tickets sintéticos para la demostración>>

| Elemento | Valor | Qué permite probar |
|:--|:--|:--|
| CM-101 Peña Blanca | Explotación, vigente, actividad autorizada programada | Explicación alternativa: maquinaria y desbroce → prioridad MEDIA |
| CM-103 La Ventana | Exploración avanzada; colinda con bosque protector y comunidad | Regla R40 de revisión humana |
| CM-104 El Roble | Renovación en trámite; 2 alertas en 90 días | Alerta CRÍTICA con vuelo al día siguiente |
| CM-105 Cerro Azul | Suspendida; 3 alertas en 90 días | Alerta CRÍTICA con vuelo el mismo día |
| CM-106 Los Laureles | Exploración inicial; 1 alerta en 90 días | Turbidez → prioridad ALTA |
| CT-201 Minas del Valle | Concesión de un tercero | Regla R20: revisión legal, no intrusión |
| E 642600 N 9662480 PSAD56 | Se transforma a WGS 84 (≈ 443 m) | Conversión y verificación de datum |
| E 641100 N 9664950 WGS84 | 100 m fuera de CM-104 | Franja de amortiguamiento (R11) |
| GEO-2026-0412 / 0415 / 0419 / 0423 / 0430 | Tickets semilla | Consulta de estado; GEO-2026-0999 no existe |

- Indique siempre el datum de las coordenadas (WGS 84 o PSAD56); si lo omite, el asistente lo pedirá.
- No escriba cédulas, teléfonos ni datos de terceros; si lo hace, el sistema los enmascara y no los registra.
- Ante riesgo para personas, escriba «emergencia» y llame al ECU 911: el asistente no reemplaza los canales de emergencia.
- Escriba «cancelar» para abandonar una solicitud y «quiero hablar con una persona» para derivar.
- Si `streamlit` no se reconoce como comando, ejecute `python -m streamlit run app_streamlit.py`.

## B Ficha del modelo, reproducción y correspondencia con la rúbrica

<<tabla:ficha|Ficha del modelo de intenciones (formato adaptado de Mitchell et al., 2019)>>

| Campo | Descripción |
|:--|:--|
| Modelo | @@FICHA_MODELO@@ |
| Uso previsto | Clasificar mensajes internos de la mesa de Geomática en 14 intenciones para un ChatBot con reglas y red bayesiana |
| Usos no previstos | Decisiones sobre personas, atención al público, idiomas distintos del español, emergencias sin canal humano |
| Datos de entrenamiento | @@FICHA_DATOS@@ |
| Datos de evaluación | 252 frases de prueba redactadas aparte y su versión con ruido ortográfico (252) |
| Métricas | @@FICHA_METRICAS@@ |
| Calibración | Escalado de temperatura en validación (T = @@FICHA_T@@); umbral de aceptación 0,50 |
| Factores no evaluados | Variantes dialectales, kichwa y shuar, voz, usuarios reales, deriva temporal |
| Licencia del modelo base | Pipeline es_dep_news_trf de spaCy (GPL-3.0); BETO original con su propia licencia |

**Reproducción.** El código, los datos sintéticos, los resultados y el cuaderno para Google Colab están en https://github.com/gaboas2703/kuribot-chatbot-geomatica. La secuencia es: extraer BETO (`scripts/extraer_beto.py`), construir la base y el corpus (`datos/`), entrenar las iteraciones 1 y 2 (`experimentos/entrenar_nlp.py`), ejecutar las pruebas (`pruebas/`) y generar las figuras (`experimentos/figuras.py`). El archivo `README.md` contiene los comandos exactos, y `cuaderno/KuriBot_Colab.ipynb` ejecuta la misma secuencia en Colab.

<<tabla:rubrica|Correspondencia con los criterios de evaluación>>

| Criterio (puntos) | Dónde se evidencia |
|:--|:--|
| Caso de negocio y diferenciación IA vs. ML (2) | Caso de negocio: diagnóstico, actores, {{tabla:metricas}}, {{tabla:enfoques}} y escenario económico; arquitectura en la {{figura:arquitectura}} |
| Modelado técnico: NLP y razonamiento (4) | Metodología; clasificador, red bayesiana ({{figura:red}} y {{tabla:tpc}}), motor de reglas ({{tabla:reglas}}); comparación de cuatro modelos y ajustes ({{tabla:modelos}} a {{tabla:clases}}) |
| Funcionalidad demostrable (3,5) | Interfaces web y de consola (Anexo A); 34 escenarios y 43 turnos documentados, 8 de confirmación y 34 pruebas unitarias (Anexo C); análisis de fallos y ajustes |
| Ética, regulación y sesgos (2) | Análisis crítico: ética, sesgos, marco regulatorio y lineamientos de uso responsable |
| Rigor metodológico (2) | Particiones sin fuga, validación cruzada agrupada, intervalos de confianza, McNemar, calibración, dos iteraciones y confirmación; limitaciones declaradas |
| Normas APA 7 (1,5) | Citas autor-fecha en el texto y 43 referencias verificadas |

## C Evidencia de pruebas

<<tabla:aceptacion|Suite de aceptación: última entrada de cada escenario, decisión del sistema y resultado por iteración>>

@@TABLA_ACEPTACION@@
<<nota|Sí = cumple todos los criterios (intención, acción o regla, contenido obligatorio y latencia < 1 500 ms). Las respuestas completas de cada turno están en los archivos JSON de la carpeta resultados del repositorio.>>

<<tabla:ejecuciones|Resumen de las ejecuciones de la suite funcional>>

@@ACEPTACION_RESUMEN@@

**Pruebas unitarias** (`python -m pytest pruebas/test_unitarias.py -v`):

@@PYTEST@@

## D Código fuente completo

Se reproducen los módulos del ChatBot, la construcción de datos, el entrenamiento y evaluación, las pruebas y la extracción de BETO. El corpus fuente (`datos/corpus_fuente.py`), los datos sintéticos y los scripts auxiliares de figuras y capturas están en el repositorio.

@@CODIGO_ANEXO@@
