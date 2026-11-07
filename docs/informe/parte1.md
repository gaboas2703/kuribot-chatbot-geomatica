# Introducción

En una operación minera, el área de Geomática funciona como una mesa de servicios silenciosa. Operaciones pregunta si un frente de trabajo sigue dentro del lindero, Seguridad reporta maquinaria desconocida cerca de una quebrada, Legal necesita el estado de una concesión antes de una inspección y Medio Ambiente pide un mapa para el informe del mes. Buena parte de esas consultas se repite y llega por canales informales; otra parte, menos frecuente, es delicada: una coordenada expresada en el datum equivocado puede ubicar un pozo a más de cuatrocientos metros de su posición real, y una alerta de minería ilegal mal priorizada puede exponer al personal de campo o dejar sin verificar una intrusión.

Este proyecto aborda ese problema con KuriBot (*kuri* significa «oro» en kichwa), un asistente conversacional para la mesa de servicios de Geomática de una empresa minera ficticia del sur del Ecuador. El caso se inspira en la práctica profesional de uno de los integrantes en geomática y catastro minero, pero todos los datos son sintéticos. KuriBot integra los contenidos del módulo: un clasificador de intenciones basado en una red neuronal *transformer* preentrenada en español y ajustada al dominio, un motor de reglas que hereda la lógica del sistema experto de priorización de alertas construido en la semana 2, una red bayesiana que estima la credibilidad de una alerta bajo incertidumbre (semana 3) y los criterios de diseño, evaluación y riesgo de los chatbots empresariales de la semana 4 [@zambrano2026].

La pregunta que guía el trabajo es práctica: ¿puede una arquitectura híbrida, que combina comprensión neuronal del lenguaje con razonamiento simbólico y probabilístico, resolver de forma confiable y trazable las consultas recurrentes de Geomática y priorizar alertas de campo sin delegar en la máquina las decisiones que corresponden a personas? El objetivo general es diseñar, implementar y evaluar un ChatBot funcional que combine técnicas de IA apropiadas al contexto minero ecuatoriano. Los objetivos específicos son:

1. Formalizar el caso de negocio y sus métricas de éxito, y justificar por qué exige IA y no solo analítica tradicional.
2. Comparar cuatro clasificadores de intenciones, uno de *machine learning* clásico y tres redes neuronales, sin fuga de información.
3. Implementar un motor de reglas y una red bayesiana que respalden y expliquen las decisiones del diálogo.
4. Construir el prototipo, documentar al menos veinte interacciones de prueba y evaluar sus implicaciones éticas y regulatorias.

El documento presenta primero el caso de negocio y el estado del arte; luego la metodología, la arquitectura y el modelado; después los resultados y su contraste con las métricas de éxito, y cierra con el análisis ético, las limitaciones y las conclusiones. Los anexos contienen el manual de usuario, la ficha del modelo, la evidencia de pruebas y el código completo.

# Caso de negocio

## Diagnóstico

Minera del Austro S.A. (nombre ficticio) es una empresa de mediana minería aurífera con seis concesiones propias en tres campamentos, colindante con la concesión de un tercero, un bosque protector y una comunidad. Su área de Geomática, con dos o tres analistas, atiende a cinco áreas internas. Para dimensionar el problema se plantea una línea base **hipotética**, que deberá medirse en un piloto: 240 solicitudes al mes, de las cuales cerca del 55 % son repetitivas (fichas de concesiones, conversiones de coordenadas, estado de pedidos y dudas geodésicas) y toman unos diez minutos cada una, y unas quince alertas mensuales de posible actividad minera no autorizada.

El diagnóstico identifica cinco fricciones. La **dependencia de pocas personas**: cuando el analista está en campo, las consultas simples esperan. La **falta de trazabilidad**: las solicitudes llegan por mensajería y llamadas, sin número de caso. El **riesgo geodésico**: en la zona coexisten coordenadas en PSAD56 y en WGS 84, y la transformación oficial de la base EPSG desplaza un punto unos 443 m en el área del proyecto, suficiente para ubicarlo en la concesión equivocada. La **priorización intuitiva de alertas**: dos reportes con la misma redacción pueden merecer respuestas distintas según los antecedentes del sector o los trabajos autorizados en el área. Y la **seguridad del personal**: una alerta con personas armadas no debe tratarse como una solicitud de vuelo más.

## Actores y alcance

Los usuarios son colaboradores de Operaciones, Seguridad patrimonial, Legal y Catastro, Medio Ambiente y Relaciones Comunitarias; intervienen además el analista de Geomática, que recibe las derivaciones, el coordinador de vuelos, el supervisor de turno, que aprueba toda acción de campo, y el responsable de protección de datos. El prototipo reconoce 14 intenciones: saludo, despedida, ayuda, consultar una concesión, verificar un punto, convertir coordenadas, reportar una alerta, solicitar un vuelo de dron, solicitar cartografía, consultar una solicitud, resolver una duda geodésica, hablar con una persona, reportar una emergencia y mensajes fuera de alcance. Quedan excluidos de forma deliberada emitir certificados, modificar el catastro, autorizar vuelos o intervenciones (solo propone y registra), identificar personas, conectarse con la Agencia de Regulación y Control (ARCOM) o con servicios meteorológicos reales y dar asesoría legal.

## Métricas de éxito

Las metas de la {{tabla:metricas}} se fijaron antes de evaluar los modelos y combinan métricas técnicas, de sistema y de negocio [@zambrano2026]. Las de negocio quedan como objetivos de un piloto, porque un prototipo con datos sintéticos no puede demostrarlas.

<<tabla:metricas|Métricas de éxito definidas antes de la evaluación>>

| Dimensión | Indicador | Meta |
|:--|:--|:--|
| NLP | F1 macro en prueba limpia (frases redactadas aparte) | ≥ 0,90 |
| NLP | F1 macro en prueba con ruido ortográfico | ≥ 0,85 |
| NLP | Exhaustividad de la clase «fuera de alcance» | ≥ 0,80 |
| NLP | Error de calibración esperado (ECE) tras escalado de temperatura | ≤ 0,05 |
| NLP | Latencia del clasificador, percentil 95, en CPU | ≤ 300 ms |
| Sistema | Escenarios de aceptación aprobados | ≥ 90 % |
| Sistema | Emergencias con riesgo para personas derivadas a una persona | 100 % |
| Sistema | Alertas en concesiones de terceros calificadas como intrusión | 0 |
| Sistema | Acciones de campo propuestas sin aprobación humana | 0 |
| Sistema | Respuestas con datos inventados (tickets o concesiones inexistentes) | 0 |
| Sistema | Latencia de extremo a extremo, percentil 95 | ≤ 1 500 ms |
| Negocio (piloto) | Contención de consultas repetitivas; tiempo del reporte a la decisión; satisfacción (CSAT) | ≥ 50 %; < 5 min; ≥ 4/5 |

## Inteligencia artificial, *machine learning* y analítica tradicional

La IA busca comportamiento inteligente en máquinas: percibir, representar conocimiento, razonar, aprender y actuar [@russell2021]. El *machine learning* es el subconjunto que induce ese comportamiento a partir de datos: un programa aprende de la experiencia *E* respecto de una tarea *T* y una medida *P* si su rendimiento en *T*, medido por *P*, mejora con *E* [@mitchell1997]. El *deep learning* es la familia del *machine learning* basada en redes con múltiples capas de representación [@goodfellow2016]. La pregunta útil no es «¿IA o ML?», sino qué parte del problema conviene aprender y qué parte conviene programar [@zambrano2026].

En KuriBot solo un componente aprende de datos: el clasificador de intenciones, cuya tarea *T* es asignar a cada mensaje una de 14 intenciones, con la experiencia *E* de 744 expresiones etiquetadas y la medida *P* del F1 macro en frases nunca vistas. El resto es IA que no es *machine learning*: reglas explícitas, una red bayesiana con parámetros elicitados por expertos, una búsqueda de ventanas de vuelo y geoprocesos deterministas ({{tabla:enfoques}}). Una consulta SQL puede devolver el estado de la concesión CM-105, pero no entender que «¿cerro azul está activa?» pide exactamente eso, ni advertir que a unas coordenadas les falta el datum. Tampoco conviene resolver todo con *machine learning*: las alertas verificadas son pocas y no existe un histórico etiquetado para entrenar un modelo de priorización, y la seguridad del personal y los derechos de terceros no pueden quedar a merced de una probabilidad.

<<tabla:enfoques|Diferenciación de enfoques y su papel en KuriBot>>

| Enfoque | Qué aporta | Qué no puede hacer por sí solo | Uso en KuriBot |
|:--|:--|:--|:--|
| Analítica tradicional (SQL, SIG) | Consultas exactas sobre datos estructurados | Interpretar texto libre o razonar con evidencia incierta | Consultas al catastro una vez entendida la intención |
| *Machine learning* clásico | Patrones léxicos con pocos datos | Generalizar a paráfrasis y errores ortográficos | Línea base TF-IDF + regresión logística (M0) |
| *Deep learning* | Representaciones contextuales del lenguaje | Garantizar políticas, conocer hechos, explicar decisiones | BETO ajustado para clasificar intenciones (M3) |
| IA simbólica (reglas) | Decisiones auditables y restricciones de seguridad | Cubrir la variabilidad del lenguaje | Motor de reglas R01–R42 y reglas de vuelo V01–V06 |
| IA probabilística | Combinar evidencias inciertas con conocimiento experto | Obtener probabilidades válidas sin datos ni elicitación | Red bayesiana de credibilidad de alertas |

## Escenario económico

Con la línea base hipotética, si el asistente resolviera sin intervención la mitad de las 132 consultas repetitivas mensuales, liberaría unas 11 horas de analista al mes (132 × 0,5 × 10 min); a un costo cargado supuesto de USD 10 por hora, el ahorro directo sería de unos USD 1 300 al año, modesto frente al costo de integrar y mantener el sistema. El argumento económico principal es otro: reducir los errores de datum en entregas con efectos legales y acortar el tiempo entre un reporte de campo y la decisión de verificarlo. Ninguno de esos beneficios puede cuantificarse con datos sintéticos, de modo que la decisión de inversión debe tomarse después de un piloto.

# Estado del arte

## ChatBots en la empresa y en la industria

Adamopoulou y Moussiades [-@adamopoulou2020] clasifican los chatbots por dominio, servicio, objetivo y método de respuesta (reglas, recuperación o generación) y muestran que su calidad depende tanto de la integración con fuentes de conocimiento como del modelo de lenguaje. Caldarini et al. [-@caldarini2022] documentan el paso hacia arquitecturas de *deep learning*, junto con la falta de criterios de evaluación comparables y el predominio de recursos en inglés, límites que este proyecto enfrenta con métricas explícitas y un corpus propio en español. Følstad et al. [-@folstad2021] proponen una agenda centrada en el usuario, el impacto, la ética y la confianza, y Adam et al. [-@adam2021] advierten que las señales antropomórficas aumentan la complacencia del usuario, razón por la que KuriBot se presenta siempre como asistente automatizado. En la industria, Colabianchi et al. [-@colabianchi2023] proponen una arquitectura de cinco módulos para agentes conversacionales industriales y Wellsandt et al. [-@wellsandt2022] describen asistentes de mantenimiento con inteligencia híbrida aumentada, que acercan la información al operador sin retirarle la decisión. No se identificaron estudios sobre asistentes conversacionales para catastro minero o geomática en español; esa es la brecha aplicada que atiende este trabajo.

## Clasificación de intenciones con modelos preentrenados

Larson et al. [-@larson2019] mostraron, con 150 intenciones, que los clasificadores aciertan dentro del dominio pero fallan con consultas fuera de alcance, y que entrenar con ejemplos de esa clase mejora su detección; por eso KuriBot tiene una clase «fuera de alcance» y un umbral de abstención. Casanueva et al. [-@casanueva2020] encontraron que los codificadores preentrenados detectan intenciones con pocas decenas de ejemplos por clase, el escenario de este proyecto. La arquitectura *transformer* [@vaswani2017] y el preentrenamiento bidireccional de BERT [@devlin2019] son la base de BETO, el BERT entrenado con un corpus grande en español [@canete2020]. Como alternativa liviana se evaluaron vectores fastText con información de subpalabras [@bojanowski2017] y una red LSTM [@hochreiter1997]. Guo et al. [-@guo2017] demostraron que las redes modernas son sobreconfiadas y que un escalado de temperatura corrige gran parte del problema, algo indispensable cuando la confianza decide si el bot responde, pregunta o se abstiene.

## Razonamiento híbrido y minería ilegal

Una red bayesiana representa una distribución conjunta mediante un grafo acíclico dirigido y tablas de probabilidad condicional, y permite inferencia exacta bajo incertidumbre [@pearl1988; @koller2009]. En minería, Mishra et al. [-@mishra2021] combinaron juicio experto y datos de sitio en una red bayesiana de riesgo geotécnico, precedente directo de parámetros elicitados que se actualizan con datos. Garcez y Lamb [-@garcez2023] describen la integración de aprendizaje y razonamiento como la tercera ola de la IA; KuriBot aplica esa idea de forma modesta: la red neuronal interpreta el lenguaje y los componentes simbólicos y probabilísticos deciden. En el Ecuador, Mestanza-Ramón et al. [-@mestanza2022] señalan la débil coordinación entre Estado, comunidades y mineros y la insuficiente fiscalización de la minería aurífera, y Chuizaca-Espinoza et al. [-@chuizaca2026] documentaron con Sentinel-2 y *random forest* la expansión de la minería ilegal en el río Nangaritza. Ese tipo de alerta satelital es una de las evidencias que recibe la red bayesiana, con una condición: la detección remota orienta la verificación, no la sustituye.

# Metodología

## Diseño, corpus y particiones

El trabajo sigue un ciclo de ciencia del diseño adaptado de CRISP-DM y alineado con las cuatro fases de la actividad: negocio y arquitectura; datos y modelado; implementación y pruebas en dos iteraciones; evaluación, ética y defensa. Es un estudio de ingeniería con datos sintéticos, no un experimento con usuarios.

El equipo redactó 406 plantillas para las 14 intenciones, en español de uso laboral ecuatoriano y con registros formales e informales. Las plantillas con marcadores (concesión, coordenadas, ticket, producto, fecha) se instanciaron tres veces con valores del catastro sintético, lo que produjo 744 expresiones de entrenamiento y validación. Aparte, y con otra formulación, se redactaron 18 frases de prueba por intención (252). Para medir robustez, cada frase de prueba se perturbó con una semilla fija (minúsculas, sin tildes, abreviaturas de chat como «q» o «porfa» y un error de tecleo con probabilidad 0,7), lo que dio la **prueba con ruido** de 252 frases. Ninguna frase de prueba coincide con una de entrenamiento, y todas las variantes de una plantilla comparten un identificador de grupo para que la validación cruzada nunca las separe (StratifiedGroupKFold). El desbalance entre clases (de 24 a 90 ejemplos) se compensa con pesos de clase.

Antes de clasificar, cada mensaje pasa por dos pasos deterministas comunes a todos los modelos: el enmascarado de datos personales (cédulas con dígito verificador válido, teléfonos y correos) y el de entidades, que sustituye coordenadas, tickets y códigos por su tipo. Así el clasificador aprende la intención por el contexto y el extractor se encarga de los valores.

## Modelos comparados

- **M0, TF-IDF + regresión logística:** unigramas y bigramas de palabras más n-gramas de caracteres de 2 a 5, con pesos balanceados; es la línea base de *machine learning* clásico.
- **M1, fastText + BiLSTM:** vectores de 300 dimensiones del modelo es_core_news_lg de spaCy [@honnibal2020], una LSTM bidireccional de 128 unidades, agregación por atención y por máximo, y un perceptrón multicapa.
- **M2, BETO congelado + MLP:** media estandarizada de la última capa de BETO y un perceptrón de 256 unidades ocultas.
- **M3, BETO ajustado:** el *transformer* completo se ajusta de extremo a extremo con una capa lineal de salida.

Los pesos de BETO se obtuvieron del pipeline oficial es_dep_news_trf 3.8.0 de spaCy, publicado en GitHub, porque el entorno de desarrollo no tenía acceso a otros repositorios de modelos. Ese pipeline contiene BETO ajustado por spaCy para análisis morfosintáctico, de modo que no es idéntico al original; la conversión se verificó contra la implementación de origen, con una diferencia máxima de 4,1 × 10⁻⁶.

## Protocolo de evaluación

Los modelos se comparan con validación cruzada de cinco pliegues estratificada por intención y agrupada por plantilla, con una parte de cada pliegue reservada para la parada temprana. Después, cada modelo se entrena con el 85 % del corpus y se valida con el 15 % restante, que también fija la temperatura de calibración y el umbral de confianza; ninguna decisión se toma con el conjunto de prueba. Se reportan exactitud con intervalo de Wilson, F1 macro con intervalo *bootstrap* de 2 000 remuestreos, métricas por clase, matriz de confusión, error de calibración esperado, latencia y parámetros entrenables, y las diferencias se contrastan con la prueba exacta de McNemar [@dietterich1998]. Se usaron scikit-learn [@pedregosa2011], PyTorch [@paszke2019], Transformers [@wolf2020] y pgmpy [@ankan2015], con semillas fijas y CPU de dos hilos; las tablas y figuras se generan desde los archivos de resultados, sin transcripción manual.

La evaluación funcional usa una suite de 34 escenarios (43 turnos) con criterios fijados antes de ejecutarla: intención esperada, acción o regla esperada, fragmentos obligatorios en la respuesta y latencia inferior a 1 500 ms. Se ejecuta en dos iteraciones, la primera para revelar fallos y la segunda para verificar los ajustes, y se complementa con 34 pruebas unitarias y con ocho escenarios de confirmación redactados al final y ejecutados una sola vez.

# Arquitectura del sistema

La arquitectura sigue la separación clásica entre comprensión, gestión del diálogo y generación [@adamopoulou2020; @zambrano2026], con dos decisiones propias: la red neuronal solo decide la **intención**, nunca los hechos ni las acciones, y toda respuesta se construye con plantillas a partir de datos verificados, sin un modelo generativo que pueda inventar cifras ({{figura:arquitectura}}).

<<figura:arquitectura|../figuras/arquitectura.png|100|Arquitectura de componentes de KuriBot>>
<<nota|Las flechas discontinuas indican registro y derivación a personas. Elaboración propia.>>

Cada mensaje sigue siete pasos: (1) se enmascaran los datos personales; (2) una red de seguridad busca señales de emergencia; (3) el extractor obtiene coordenadas, datum, concesión, ticket, fecha, producto y evidencias, con detección de negaciones; (4) si había una acción pendiente, se intenta completar la ranura que faltaba; (5) si no, el clasificador predice la intención y su confianza calibrada; (6) la política de la intención consulta el catastro, ejecuta geoprocesos e invoca la red bayesiana y el motor de reglas; (7) la respuesta se genera con plantillas y se acompaña de una traza auditable (intención, confianza, reglas, posterior bayesiano y latencia). La base de conocimiento es sintética: un catastro en GeoJSON (seis concesiones propias, una de un tercero, un bosque protector y una comunidad), un archivo de operación (pronóstico de cinco días, drones, pilotos, historial de alertas y actividades autorizadas), una FAQ geodésica de 19 entradas y una base SQLite de tickets.

# Modelado del componente inteligente

## Clasificador de intenciones

Para un mensaje con piezas $x_1,\dots,x_n$, BETO produce estados contextuales $\mathbf{h}_i\in\mathbb{R}^{768}$; la representación del mensaje es su media y la distribución sobre las 14 intenciones se obtiene con una capa lineal y la temperatura $T$:

$$\bar{\mathbf{h}}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{h}_i,\qquad \hat{\mathbf{p}}=\operatorname{softmax}\!\left(\frac{\mathbf{W}\bar{\mathbf{h}}+\mathbf{b}}{T}\right)$$

El ajuste minimiza la entropía cruzada ponderada por clase, con suavizado de etiquetas de 0,05, AdamW (tasa 3 × 10⁻⁵), lotes de 16, hasta 48 piezas por mensaje y parada temprana sobre el F1 macro de validación. $T$ se estima después del entrenamiento minimizando la log-verosimilitud negativa en validación [@guo2017]. La decisión usa la confianza calibrada $p_1$ y la segunda probabilidad $p_2$, pues la probabilidad máxima del softmax es una base sencilla y eficaz para detectar errores y entradas ajenas al entrenamiento [@hendrycks2017]: si $p_1$ ≥ máx(0,50; $\tau$), el bot **acepta** la intención; si $p_1$ ≥ 0,30 y $p_1+p_2$ ≥ 0,70, **pide aclaración** entre las dos opciones; en otro caso **se abstiene**, explica lo que puede hacer y, al segundo fallo seguido, ofrece una persona. El umbral $\tau$ se elige en validación.

## Red bayesiana de credibilidad de alertas

El nodo de interés $I$ representa una actividad minera no autorizada real ({{figura:red}}). Su causa son los antecedentes del sector ($H$, alertas en los últimos 90 días). La actividad autorizada programada ($A$) es una causa alternativa de varias evidencias: maquinaria ($M$), personas o campamento ($P$), alteración del terreno ($S$) y vía nueva ($V$); el ruido nocturno ($R$) y la turbidez ($T$) dependen solo de $I$. $H$ y $A$ se observan en la base de conocimiento; las evidencias se extraen del mensaje, incluidas las negadas («no hay personas» fija $P=0$), y las no mencionadas se marginalizan. Las tablas de probabilidad condicional de la {{tabla:tpc}} son juicios expertos del equipo, no frecuencias estimadas.

<<figura:red|../figuras/red_bayesiana.png|78|Grafo de la red bayesiana de credibilidad de alertas>>
<<nota|Elaboración propia, generado desde el código.>>

<<tabla:tpc|Tablas de probabilidad condicional elicitadas, P(X = 1 | padres)>>

| Nodo | Padres | I=0, A=0 | I=0, A=1 | I=1, A=0 | I=1, A=1 |
|:--|:--|--:|--:|--:|--:|
| M: maquinaria | I, A | 0,03 | 0,60 | 0,70 | 0,85 |
| P: personas o campamento | I, A | 0,10 | 0,50 | 0,75 | 0,85 |
| S: alteración del terreno | I, A | 0,05 | 0,45 | 0,65 | 0,80 |
| V: vía o trocha nueva | I, A | 0,03 | 0,20 | 0,40 | 0,45 |
| R: ruido o voladuras nocturnas | I | 0,05 | — | 0,40 | — |
| T: agua turbia | I | 0,15 | — | 0,40 | — |
<<nota|Para el nodo de interés, P(I = 1 | H = 0) = 0,10 y P(I = 1 | H = 1) = 0,30. Las probabilidades a priori P(H = 1) = 0,30 y P(A = 1) = 0,40 solo se usan cuando H o A no se observan en la base de conocimiento.>>

La inferencia es exacta por eliminación de variables (pgmpy): para una evidencia $\mathbf{e}$, el posterior normaliza la distribución conjunta sobre las variables no observadas $\mathbf{z}$, y se verificó por enumeración completa con diferencias menores que 10⁻⁹.

$$P(I=1\mid\mathbf{e})=\frac{\sum_{\mathbf{z}}P(I=1,\mathbf{e},\mathbf{z})}{\sum_{i\in\{0,1\}}\sum_{\mathbf{z}}P(I=i,\mathbf{e},\mathbf{z})},\qquad P(\mathbf{x})=\prod_{k}P(x_k\mid \mathrm{pa}(x_k))$$

La estructura captura la explicación alternativa (*explaining away*): maquinaria y desbroce en Peña Blanca, sin antecedentes, dan un posterior de 0,219 porque hay un frente de explotación autorizado; la misma evidencia en un área sin actividad programada lo eleva a 0,971. Un análisis de sensibilidad (±0,10 en cada parámetro) mostró que el posterior depende mucho más del prior $P(I=1\mid H=0)$ (rango 0,02–0,39) que de cualquier verosimilitud (rangos menores de 0,08); por eso lo primero que debe calibrarse con verificaciones de campo es la tasa base de intrusiones por sector.

## Motor de reglas y umbrales de decisión

El motor es un sistema de producción con encadenamiento hacia adelante: en cada ciclo arma la agenda con las reglas aplicables, dispara la de menor nivel y mayor prioridad, agrega sus conclusiones a la memoria de trabajo y no la repite (refracción). Las prioridades son excluyentes y una prueba automatizada comprueba que barajar el orden de las reglas no cambia la conclusión ({{tabla:reglas}}). R12 y R20 impiden tratar como intrusión lo que ocurre en la concesión de un tercero, y R40 obliga a la revisión humana cerca de comunidades o áreas protegidas, criterios de equidad heredados de la semana 2.

<<tabla:reglas|Base de reglas del motor de decisiones (resumen)>>

| Nivel | Reglas | Condición → conclusión |
|:--|:--|:--|
| 0. Seguridad | R01–R03 | Emergencia con riesgo para personas → CRÍTICA, protocolo, sin despliegue y derivación humana; datos personales → enmascarar; pedido de una persona → derivar |
| 1. Contexto geoespacial | R10–R15 | Concesión propia; franja de 500 m; concesión de tercero; área protegida; comunidad a 1 km o menos; fuera del área |
| 2. Priorización | R20–R25 | Derecho de tercero → REVISIÓN LEGAL; P(I) ≥ 0,80 CRÍTICA; ≥ 0,50 ALTA; ≥ 0,15 MEDIA; resto BAJA; fuera del área → INFORMATIVA |
| 3. Acción | R30–R36 | Vuelo prioritario si hay ventana en 24 h, si no satélite y patrullaje; vuelo en 72 h; revisión satelital; registro; derivación a Legal |
| 4. Salvaguardas | R40–R42 | Zona sensible → revisión humana; CRÍTICA o ALTA → aprobación del supervisor; toda alerta → ticket |
| Vuelos | V01–V06 | Viento > 10 m/s, lluvia, visibilidad < 5 km, sin dron, sin piloto o menos de un día de anticipación → no apto; búsqueda de la primera ventana apta |
<<nota|Los umbrales de 500 m, 1 km, 10 m/s y 5 km son supuestos de diseño del equipo.>>

El umbral inferior de acción se deriva de una razón de costos: si verificar con dron cuesta $C$ y no detectar a tiempo una intrusión real cuesta $L$, conviene verificar cuando $P(I\mid\mathbf{e})\,L \ge C$, es decir, cuando $P(I\mid\mathbf{e}) \ge C/L$. Con el supuesto $L/C$ ≈ 6,7, el umbral queda en 0,15, que separa BAJA de MEDIA; 0,50 y 0,80 escalan el plazo de verificación. Con costos asimétricos, la métrica y el umbral son decisiones de negocio [@zambrano2026].

El gestor de diálogo guarda la acción pendiente, sus ranuras, la última concesión mencionada y los fallos seguidos. Si falta un dato (concesión, coordenadas, datum, fecha, producto o evidencias), lo pide y completa la acción cuando llega; si llega otra intención con confianza suficiente, abandona la pendiente; tras dos intentos fallidos ofrece una persona, y «cancelar» anula la acción en cualquier momento.
