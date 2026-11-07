# Resultados

## Implementación

KuriBot está escrito en Python y organizado en módulos con responsabilidades separadas: entidades, geodesia, base de conocimiento, modelos e inferencia de intenciones, razonamiento y gestor de diálogo (Anexo D). Tiene dos interfaces sobre el mismo motor. La web, construida con Streamlit, muestra la conversación y, bajo cada respuesta, un panel de trazabilidad con la intención, las probabilidades, las entidades, las reglas disparadas, el posterior bayesiano, las ventanas de vuelo y un mapa del catastro; la de consola imprime la misma información en una línea técnica (Anexo A). El modelo de producción ocupa 219 MB en media precisión y corre en CPU sin servicios externos; si faltan sus pesos, la aplicación usa la línea base TF-IDF y lo advierte.

## Desempeño del componente NLP

En la validación cruzada agrupada por plantilla, la prueba más exigente porque cada pliegue evalúa formulaciones nuevas, BETO ajustado (M3) superó con claridad a los demás: F1 macro de 0,848 frente a 0,762 de la línea base ({{tabla:modelos}} y {{figura:comparacion}}). En la prueba limpia, en cambio, M3 y la línea base TF-IDF quedaron prácticamente empatados (0,917 y 0,922; McNemar p = 0,86), y con ruido ortográfico la línea base fue mejor (0,881 frente a 0,805; p = 0,005): los n-gramas de caracteres toleran una letra cambiada, mientras que BETO divide una palabra con errata en piezas que nunca vio juntas. La BiLSTM con fastText (cobertura del vocabulario de 96,9 %) perdió 13 puntos con ruido porque trabaja con palabras completas, y BETO congelado fue el peor (0,764), coherente con que sus pesos se ajustaron en spaCy para sintaxis y no para semántica.

<<tabla:modelos|Comparación de modelos de intenciones, iteración 1>>

@@TABLA_MODELOS@@
<<nota|VC = validación cruzada de cinco pliegues agrupada por plantilla; IC = intervalo *bootstrap* de 2 000 remuestreos; la latencia es por mensaje en CPU de dos hilos.>>

<<figura:comparacion|../figuras/comparacion_modelos.png|80|F1 macro de los cuatro modelos en validación cruzada y en los conjuntos de prueba, iteración 1>>
<<nota|Las barras de error muestran una desviación estándar entre pliegues.>>

La decisión no podía tomarse solo por el F1 en limpio. M3 generalizó mejor a formulaciones nuevas, detectó mucho mejor los mensajes fuera de alcance (exhaustividad de 0,833 frente a 0,611 en la prueba limpia), que es la clase donde un error produce una respuesta impropia [@larson2019], y su debilidad, los errores de tecleo, tenía un remedio conocido. Por eso se eligió como modelo de producción y se planteó una segunda iteración para mejorar su robustez.

## Iteración 2: ajustes del modelo

Casi todos los errores de M3 con ruido ocurrían en palabras con errata o sin tilde («agenda drrone para San Gregorio» se clasificó como consulta de concesión). Se aplicaron tres ajustes, todos con datos de entrenamiento y validación: (1) una copia perturbada de cada frase de entrenamiento, con una semilla distinta de la de prueba, siguiendo las técnicas de aumento simple para texto [@wei2019]; (2) ocho plantillas nuevas de alertas con coordenadas, motivadas por el fallo A17 de la suite de aceptación; y (3) hasta diez épocas con paciencia de tres, porque el F1 de validación seguía subiendo en la sexta.

<<tabla:versiones|Efecto de los ajustes de la iteración 2 sobre la línea base y el modelo de producción>>

@@TABLA_VERSIONES@@
<<nota|v1: corpus original y hasta 6 épocas; v2: plantillas nuevas, una copia con ruido por frase y hasta 10 épocas. τ es el umbral elegido en validación; el bot acepta con confianza ≥ máx(0,50; τ).>>

El aumento benefició al *transformer* y no a la línea base ({{tabla:versiones}}). M3 pasó de 0,805 a 0,874 con ruido, una mejora significativa (25 aciertos nuevos frente a 9 perdidos; McNemar p = 0,009), y de 0,917 a 0,929 en limpio, mientras M0 bajó levemente en ambos conjuntos. Frente a la mejor línea base de la iteración 1, el modelo final ya no difiere de forma significativa (p = 1,00 en limpio y p = 0,74 con ruido), pero conserva la ventaja en detección fuera de alcance (0,833 y 0,778 de exhaustividad frente a 0,556 y 0,444 de M0). La parada temprana se activó en la séptima época, con el mejor F1 de validación en la cuarta.

## Desempeño por intención y calibración

Las clases más difíciles son «fuera de alcance» (F1 de 0,769), que agrupa temas heterogéneos, y «solicitar vuelo» (0,848), que se confunde con cartografía cuando se pide una ortofoto, que en efecto exige volar ({{tabla:clases}}). La matriz de confusión ({{figura:confusion}}) muestra que los errores se concentran en pares semánticamente cercanos: verificar frente a convertir coordenadas, y cartografía frente a vuelo o concesión.

<<tabla:clases|Precisión, exhaustividad y F1 por intención del modelo de producción (M3, iteración 2)>>

@@TABLA_CLASES@@

<<figura:confusion|../figuras/matriz_confusion_m3_limpia.png|62|Matriz de confusión del modelo de producción en la prueba limpia (n = 252)>>

El escalado de temperatura (T = 0,81) redujo el error de calibración esperado de 0,066 a 0,042 en la prueba limpia y de 0,085 a 0,058 con ruido. Con el umbral elegido en validación (τ = 0,40), el bot acepta el 98,4 % de los mensajes con 93,1 % de exactitud; exigir 0,80 elevaría esa exactitud a 98,2 % a costa de pedir aclaración en el 12 % de los casos. Se optó por un umbral de aceptación de 0,50 y por la aclaración entre las dos intenciones más probables, porque preguntar es barato y equivocarse en una alerta no lo es. La latencia del clasificador fue de 53,6 ms en el percentil 95.

## Pruebas funcionales documentadas

La suite de aceptación ejecuta 34 escenarios (@@TURNOS@@ turnos) que cubren las 14 intenciones, el llenado de ranuras en dos y tres turnos, los geoprocesos, las seis ramas de priorización de alertas, la emergencia, la protección de datos, los mensajes ambiguos y fuera de alcance, y la regla de no inventar datos; el Anexo C documenta cada escenario con su entrada, la decisión del sistema y su resultado en ambas iteraciones. La primera iteración aprobó @@ACEPT_V1@@ escenarios (@@ACEPT_TASA_V1@@ %) y la segunda @@ACEPT_V2@@ (@@ACEPT_TASA_V2@@ %), con una latencia de extremo a extremo de @@ACEPT_LAT_MEDIA@@ ms en promedio y @@ACEPT_LAT_P95@@ ms en el percentil 95.

Los escenarios de alertas muestran el razonamiento completo ({{tabla:bayes}}). La misma combinación de maquinaria y desbroce produce prioridad MEDIA en Peña Blanca, donde hay un frente de explotación autorizado (A15), mientras que maquinaria y personas en Cerro Azul, suspendida y con tres alertas recientes, llevan el posterior a 0,987 y a prioridad CRÍTICA (A14). Cuando el reporte niega una evidencia («pero no hay personas», A19), el posterior cae a 0,045, pero R40 exige revisión humana porque La Ventana colinda con el bosque protector y la comunidad. En A17 el posterior es 0,359, pero el punto está en la concesión de un tercero y R20 lo deriva a Legal sin calificarlo como intrusión.

<<tabla:bayes|Decisiones del componente de razonamiento en los escenarios de alerta (iteración 2)>>

@@TABLA_BAYES@@
<<nota|P(I | e) = probabilidad posterior de actividad no autorizada dada la evidencia, calculada con la red bayesiana.>>

## Análisis de fallos y ajustes

La primera ejecución falló en dos escenarios. En **A17** («vimos maquinaria en E 642000 N 9663600 WGS84») el modelo predijo «verificar punto»: el corpus original no tenía alertas con coordenadas y la red aprendió una correlación espuria entre el marcador de coordenadas y la verificación de puntos; se agregaron ocho plantillas de alertas con coordenadas. En **A31** («mi cédula es [CÉDULA], ¿cuál es el estado de CM-104?») el modelo confundió la consulta con el estado de un ticket, porque el marcador de dato personal no existía en el entrenamiento; se retiraron esos marcadores del texto que recibe el clasificador, ya que no aportan información sobre la intención. Ambos pasaron en la iteración 2 sin afectar a los demás. Un tercer problema apareció al reproducir el proyecto en un directorio limpio: el primer turno tardó una vez 2,2 s por el arranque en frío de PyTorch y A01 falló por latencia; el clasificador ahora ejecuta una inferencia de calentamiento al cargarse, y la suite repetida con ese cambio no varió ninguna intención, regla ni respuesta.

Ajustar el sistema hasta que la suite pase no garantiza que mejore de verdad. Para controlarlo, después de cerrar la iteración 2 se redactaron ocho escenarios de confirmación nuevos (C01–C08), ejecutados una sola vez y sin ajustes posteriores: aprobaron siete (87,5 %). El fallo, C07 («¿va a llover mañana en Machala?»), se clasificó como solicitud de vuelo por la presencia de clima y fecha; la respuesta fue inocua, porque el bot preguntó por la concesión del vuelo, pero confirma que la frontera entre el clima como condición de vuelo y el clima como charla no está bien aprendida. Las 34 pruebas unitarias (Anexo C) verifican, entre otras cosas, la ida y vuelta PSAD56–WGS 84 con error inferior a 1 cm, la coincidencia entre pgmpy y la enumeración exacta, la prevalencia de la regla de emergencia, el determinismo del motor y la negativa a inventar tickets.

# Evaluación integral contra las métricas de éxito

El sistema cumplió todas las metas de sistema y las cinco técnicas en la prueba limpia ({{tabla:evaluacion}}). Con ruido ortográfico, el F1 macro superó su meta, pero la exhaustividad fuera de alcance (0,778) y el error de calibración (0,058) quedaron ligeramente por debajo de lo exigido. Las métricas de negocio no pueden evaluarse con datos sintéticos.

<<tabla:evaluacion|Evaluación del sistema frente a las métricas de éxito>>

| Indicador | Meta | Resultado | Estado |
|:--|:--|:--|:--|
| F1 macro, prueba limpia | ≥ 0,90 | 0,929 [IC 0,89–0,96] | Cumplida |
| F1 macro, prueba con ruido | ≥ 0,85 | 0,874 [IC 0,83–0,91] | Cumplida tras la iteración 2 |
| Exhaustividad fuera de alcance | ≥ 0,80 | 0,833 limpia; 0,778 con ruido | Cumplida solo en limpia |
| ECE calibrado | ≤ 0,05 | 0,042 limpia; 0,058 con ruido | Cumplida solo en limpia |
| Latencia del clasificador, p95 | ≤ 300 ms | 53,6 ms | Cumplida |
| Escenarios de aceptación | ≥ 90 % | @@ACEPT_TASA_V2@@ % (iteración 2); 87,5 % en confirmación | Cumplida; confirmación levemente por debajo |
| Emergencias derivadas a una persona | 100 % | 2 de 2 escenarios (A20, C03) y prueba unitaria | Cumplida |
| Alertas en terceros calificadas como intrusión | 0 | 0 (A17 y prueba unitaria) | Cumplida |
| Acciones de campo sin aprobación humana | 0 | 0: R41 se dispara en toda prioridad CRÍTICA o ALTA | Cumplida |
| Datos inventados | 0 | 0: tickets y concesiones inexistentes se declaran no encontrados | Cumplida |
| Latencia de extremo a extremo, p95 | ≤ 1 500 ms | @@ACEPT_LAT_P95@@ ms | Cumplida |
| Contención, tiempo a decisión, CSAT | Ver {{tabla:metricas}} | No medibles con datos sintéticos | Pendiente de piloto |

Dos resultados merecen leerse con cautela. La ventaja del *deep learning* sobre una buena línea base de n-gramas es real en generalización y en detección fuera de alcance, pero no en todos los conjuntos: en la prueba limpia ambos modelos rinden igual. Y la suite fue escrita por el mismo equipo que construyó el sistema; la confirmación con escenarios nuevos (87,5 %) es una estimación más honesta del desempeño ante consultas no previstas.
