# 1. Problema de negocio
El caso seleccionado para el proyecto corresponde a la **inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)**.

Las plataformas de streaming musical generan grandes volúmenes de información sobre canciones, artistas, géneros y características de audio. Estos datos pueden ser utilizados para identificar patrones asociados al nivel de popularidad de las canciones y apoyar la toma de decisiones relacionadas con el análisis y selección de contenido musical.

En este contexto, el problema de negocio consiste en analizar las características de las canciones disponibles en el dataset de Spotify para identificar patrones asociados a sus niveles de popularidad. El análisis busca generar información para entrenar y evaluar modelos de Machine Learning que permitan estimar la popularidad, clasificar las canciones en niveles de popularidad y explorar perfiles musicales mediante segmentación.

Para abordar este problema se realizó un proceso de exploración, evaluación de calidad y preparación de los datos, seguido del entrenamiento, evaluación y optimización de modelos supervisados y no supervisados. Se consideran variables como género musical, contenido explícito, duración y características de audio.

# 2. Objetivos
## 2.1 Objetivo general
Analizar las características musicales de las canciones del dataset Spotify Tracks mediante técnicas de exploración, regresión, clasificación y clustering, con el fin de identificar patrones relacionados con la popularidad, comparar el desempeño de modelos de Machine Learning y describir segmentos musicales interpretables.

## 2.2 Objetivos específicos
- Explorar la estructura y composición del dataset de canciones de Spotify.

- Identificar problemas de calidad de los datos, considerando valores nulos, registros duplicados y variables que requieran limpieza.

- Realizar la limpieza y preparación de los datos mediante Python, asegurando su consistencia para las etapas posteriores del proyecto.

- Analizar las principales características de las canciones, considerando variables como género musical, duración, contenido explícito, danceability y energy.

- Examinar la distribución de la variable `popularity` e identificar características relevantes para su posterior predicción.

- Generar un conjunto de datos limpio y estructurado para el entrenamiento y evaluación de modelos.

- Entrenar y evaluar un modelo de regresión para estimar la popularidad numérica de las canciones.

- Construir un clasificador que categorice la popularidad en niveles bajo, medio y alto.

- Aplicar K-Means para identificar perfiles de canciones a partir de características de audio.

- Optimizar los modelos mediante búsqueda de hiperparámetros o comparación de cantidades de clústeres, y documentar sus resultados y limitaciones.

# 3. Fuentes de datos
Para el desarrollo del proyecto se utilizó el archivo `Spotify_Tracks_Dataset.csv`, correspondiente al caso de **Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)** definido por la asignatura.

El dataset contiene información de canciones de Spotify, incluyendo datos descriptivos de las pistas, artistas, álbumes, géneros musicales y características de audio. Además, incorpora la variable `popularity`, que presenta valores entre 0 y 100 y se utiliza como variable objetivo en la regresión y como base para derivar las clases de popularidad del clasificador.

El conjunto de datos original contiene **114.000 registros y 21 variables**. Tras retirar el índice original, tratar el registro incompleto y eliminar duplicados exactos, el archivo limpio quedó con **113.549 registros y 20 variables**. Las variables incluyen diferentes tipos de información, entre ellas:

- `track_id`: identificador de la canción.

- `artists`: artista o artistas asociados.

- `album_name`: álbum al que pertenece la canción.

- `track_name`: nombre de la canción.

- `popularity`: nivel de popularidad de la canción.

- `duration_ms`: duración de la canción en milisegundos.

- `explicit`: indicador de contenido explícito.

- `danceability`: nivel de bailabilidad.

- `energy`: nivel de energía.

- `track_genre`: género musical.

### Herramientas utilizadas
Para la exploración y preparación de los datos se utilizó **Python**, principalmente mediante la biblioteca **Pandas**, permitiendo cargar, inspeccionar, limpiar y transformar el conjunto de datos.

La exploración inicial se documentó en un Notebook de Python y el entrenamiento, evaluación y optimización se organizaron en scripts dentro de `src/`.

# 4. Preparación de datos y EDA
## 4.1 Exploración inicial de los datos
La exploración inicial permitió conocer la estructura, dimensiones y tipos de variables presentes en el dataset antes de realizar el proceso de limpieza.

El conjunto de datos original está compuesto por **114.000 registros y 21 variables**. Las variables presentan diferentes tipos de datos, incluyendo variables numéricas, categóricas y booleanas.

Entre las variables disponibles se encuentran información de identificación de las pistas (`track_id`), artista (`artists`), álbum (`album_name`), nombre de la canción (`track_name`), popularidad (`popularity`), características de audio y género musical (`track_genre`).

La variable `popularity` presenta valores entre 0 y 100 y se considera una variable relevante para el análisis, debido a que permite representar el nivel de popularidad de las canciones.

Durante esta etapa también se identificó la presencia de una columna denominada `Unnamed: 0`, correspondiente a un índice original del archivo. Debido a que esta columna no aporta información relevante para el análisis, fue considerada posteriormente para su eliminación durante el proceso de limpieza.

## 4.2 Calidad de los datos
### 4.2.1 Valores nulos
Se realizó una revisión de los valores nulos presentes en el conjunto de datos. El análisis identificó únicamente **3 valores nulos**, distribuidos en las variables `artists`, `album_name` y `track_name`, todos correspondientes al mismo registro.

El registro afectado representa aproximadamente un **0,0009 %** del total de registros (1 de 114.000; los 3 valores nulos corresponden a ese mismo registro), por lo que su presencia es mínima. Sin embargo, debido a que el registro afectado no contiene información textual fundamental de la canción, se decidió eliminar dicho registro durante el proceso de limpieza.

Tras eliminar el registro con valores nulos, el conjunto quedó en 113.999 filas; posteriormente, la eliminación de 450 duplicados exactos dejó el dataset final en 113.549 registros.

### 4.2.2 Registros duplicados
Se verificó la existencia de registros completamente duplicados mediante la función `duplicated()`. La revisión definitiva del proceso de limpieza identificó **450 registros duplicados exactos**, que fueron eliminados para evitar que esas filas se contabilizaran más de una vez.

Adicionalmente, se revisó la variable `track_id`, identificándose **24.259 registros con un identificador repetido** en el dataset original (antes de eliminar los duplicados exactos). Sin embargo, estos casos no fueron eliminados, debido a que un mismo `track_id` puede aparecer asociado a diferentes valores de `track_genre`.

Por lo tanto, los registros repetidos por `track_id` no fueron considerados duplicados erróneos y se conservaron para evitar la pérdida de información.

### 4.2.3 Columna de índice
Durante la exploración se identificó la variable `Unnamed: 0`, la cual corresponde a un índice generado al guardar previamente el archivo.

Esta variable presenta una secuencia correlativa de registros y no aporta información relevante para el análisis de las canciones. Por esta razón, se decidió eliminarla durante el proceso de limpieza.

## 4.3 Proceso de limpieza de datos
A partir de los resultados obtenidos durante la evaluación de la calidad de los datos, se realizó un proceso de limpieza con el objetivo de obtener un conjunto de datos consistente y adecuado para las etapas posteriores del proyecto.

El proceso de limpieza fue implementado mediante la función `limpieza_datos()`, desarrollada en el archivo `limpieza.py`. Esta función permite aplicar de manera ordenada y reproducible las transformaciones necesarias sobre el dataset.

Las principales actividades realizadas fueron:

1\\. **Normalización de nombres de columnas:** se eliminaron espacios innecesarios y se estandarizaron los nombres de las variables utilizando minúsculas.

2\\. **Eliminación del índice original:** se eliminó la columna `Unnamed: 0`, debido a que correspondía al índice generado originalmente en el archivo y no aportaba información relevante para el análisis.

3\\. **Validación de las columnas requeridas:** se verificó que las variables necesarias para el análisis estuvieran presentes en el dataset.

4\\. **Limpieza de variables de texto:** se eliminaron espacios innecesarios en variables como `track_id`, `artists`, `album_name`, `track_name` y `track_genre`.

5\\. **Conversión de variables numéricas:** las variables numéricas fueron convertidas a sus tipos correspondientes para asegurar un tratamiento adecuado durante el análisis.

6\\. **Tratamiento de valores nulos:** se eliminó el registro que presentaba valores nulos en las variables `artists`, `album_name` y `track_name`.

7\\. **Eliminación de duplicados exactos:** se aplicó una revisión de registros completamente duplicados. Se eliminaron 450 duplicados exactos durante la limpieza definitiva.

8\\. **Reinicio del índice:** finalmente, se reinició el índice del dataset limpio para mantener una numeración correlativa de los registros.

### Resultado de la limpieza
Como resultado del proceso, el dataset pasó de **114.000 registros y 21 variables** a **113.549 registros y 20 variables**. Se eliminó la columna `Unnamed: 0`, el registro con valores nulos en campos descriptivos y 450 filas duplicadas exactas.

El dataset resultante se utilizó en el análisis exploratorio y como base para los scripts de entrenamiento de los modelos.

## 4.4 Análisis exploratorio de variables relevantes
El análisis exploratorio de datos (EDA) permitió estudiar el comportamiento de las principales variables del dataset y detectar patrones relevantes antes de las etapas posteriores del proyecto.

Para esta primera etapa se seleccionaron las variables consideradas más relevantes para el problema de negocio: `popularity`, `danceability`, `energy`, `duration_ms`, `track_genre` y `explicit`.

La selección busca concentrar el análisis en variables relacionadas con la popularidad y con las principales características de las canciones, evitando incorporar variables que no aporten directamente a esta primera exploración.

### 4.4.1 Distribución de la popularidad
La variable `popularity` representa el nivel de popularidad de cada canción y presenta valores entre 0 y 100.

El análisis descriptivo muestra una media de **33,32** y una mediana de **35**. Además, se identificó una concentración importante de registros con popularidad igual a 0, correspondiente aproximadamente al **13,95 %** del dataset limpio (15.843 de 113.549 registros).

El histograma permite observar que, excluyendo la concentración de valores en 0, gran parte de las canciones se encuentra en niveles de popularidad medios, mientras que los valores superiores a 80 presentan una menor frecuencia.

Esta distribución evidencia que la popularidad no se encuentra distribuida de manera uniforme y presenta una concentración importante en determinados rangos de valores.

![Distribución de la popularidad](../images/distribucion_popularity.png)

**Figura 1.** Distribución de la popularidad de las canciones.

### 4.4.2 Distribución de danceability
La variable `danceability` representa el nivel de bailabilidad de una canción y presenta valores en un rango de 0 a 1.

A partir del análisis exploratorio se observa una concentración de los registros principalmente entre valores aproximados de 0,4 y 0,8. La mayor frecuencia se encuentra alrededor de 0,6, lo que indica que una proporción importante de las canciones presenta niveles medios y altos de bailabilidad.

No se identifican valores fuera del rango esperado para esta variable, por lo que no se considera necesario realizar una corrección asociada a sus límites.

![Distribución de danceability](../images/distribucion_danceability.png)

**Figura 2.** Distribución de la variable danceability.

### 4.4.3 Distribución de energy
La variable `energy` representa el nivel de energía de una canción y presenta valores en un rango de 0 a 1.

La distribución observada muestra una mayor concentración de canciones en valores medios y altos de energía, principalmente entre 0,6 y 1,0. En contraste, las canciones con niveles bajos de energía presentan una menor frecuencia dentro del conjunto de datos.

Este comportamiento permite observar que existe una presencia importante de canciones con características de mayor intensidad dentro del dataset. No se identifican valores fuera del rango esperado para esta variable.

![Distribución de energy](../images/distribucion_energy.png)

**Figura 3.** Distribución de la variable energy.

### 4.4.4 Distribución de la duración de las canciones
La variable `duration_ms` representa la duración de las canciones en milisegundos. Para facilitar su interpretación, durante el análisis exploratorio se realizó una conversión temporal de milisegundos a minutos.

La distribución presenta una concentración importante de canciones de duración relativamente corta, mientras que hacia valores superiores se observa una disminución progresiva de la frecuencia.

También se identifican algunos registros con duraciones considerablemente mayores que el resto. Al revisar estos casos, se encontraron canciones correspondientes a mezclas continuas, grabaciones en vivo y otros contenidos de larga duración. Por esta razón, estos valores no fueron eliminados automáticamente, ya que pueden corresponder a registros válidos dentro del catálogo musical.

En consecuencia, la variable presenta una distribución asimétrica hacia la derecha, producto de la presencia de algunas pistas con duraciones significativamente superiores a la mayoría de las observaciones.

![Distribución de la duración de las canciones](../images/distribucion_duracion.png)

**Figura 4.** Distribución de la duración de las canciones.

### 4.4.5 Distribución por género musical
El análisis de la variable `track_genre` permite conocer la distribución de las canciones según su género musical.

Debido a la cantidad de géneros disponibles en el dataset, se seleccionaron los 10 géneros con mayor número de registros para facilitar la visualización.

Los resultados muestran una distribución relativamente equilibrada entre los géneros con mayor representación, sin observarse una concentración extrema de registros en un único género. Esto permite contar con una representación diversa de categorías musicales dentro del conjunto de datos.

La variable `track_genre` resulta relevante para el análisis, ya que permite estudiar posteriormente si existen diferencias en los niveles de popularidad según el género musical.

![Top 10 géneros musicales](../images/top_10_generos.png)

**Figura 5.** Top 10 géneros musicales con mayor cantidad de registros.

### 4.4.6 Distribución del contenido explícito
La variable `explicit` indica si una canción contiene contenido explícito. Al analizar su distribución, se observa un claro predominio de canciones clasificadas como no explícitas.

Del total de registros, aproximadamente el **91,44 %** corresponde a canciones sin contenido explícito (`False`), mientras que el **8,56 %** corresponde a canciones clasificadas como explícitas (`True`).

Esta distribución muestra que la categoría no explícita es considerablemente más frecuente dentro del dataset. Sin embargo, ambas categorías se mantienen en el conjunto de datos, ya que la variable puede resultar relevante para analizar posteriormente su relación con la popularidad de las canciones.

![Distribución del contenido explícito](../images/distribucion_explicit.png)

**Figura 6.** Distribución de canciones según contenido explícito.

### 4.4.7 Identificación de anomalías y observaciones relevantes
Durante el análisis exploratorio se identificaron algunos valores que requieren consideración antes de utilizar los datos en las etapas posteriores del proyecto.

En primer lugar, se encontraron **157 registros con un valor de `tempo` igual a 0**, equivalente aproximadamente al 0,14% del dataset. Este valor puede corresponder a información no detectada o no disponible para determinadas canciones. Debido a su baja frecuencia y a que no se cuenta con información suficiente para determinar un valor correcto, estos registros fueron conservados.

También se identificaron canciones con **duraciones considerablemente superiores al promedio**. Al revisar estos registros se encontraron contenidos como mezclas continuas, grabaciones en vivo y sonidos de larga duración. Por esta razón, estos valores no fueron considerados errores automáticamente y se mantuvieron en el dataset.

Respecto de la variable `time_signature`, se observaron valores de 0, 1, 3, 4 y 5. La mayor concentración corresponde al valor 4, con 101.486 registros. Los valores menos frecuentes no fueron eliminados, debido a que su baja frecuencia por sí sola no permite determinar que correspondan a errores.

Por otra parte, la variable `popularity` presenta **15.843 registros con valor 0**, equivalentes aproximadamente al 13,95% de las observaciones. Estos valores se conservaron, ya que corresponden a valores posibles dentro de la escala de popularidad definida para el dataset y no existe evidencia suficiente para considerarlos registros erróneos.

En consecuencia, las anomalías y valores poco frecuentes identificados fueron evaluados individualmente. Se evitó eliminar registros únicamente por presentar valores extremos o poco frecuentes, priorizando la conservación de información válida para las etapas posteriores del proyecto.

# 5. KPIs del proyecto
Los KPIs permiten evaluar tanto la calidad de la preparación como el desempeño de los modelos. Los indicadores de calidad describen la confiabilidad de los datos; las métricas de modelamiento evalúan cada tarea por separado y no deben compararse directamente entre tipos de modelos distintos.

\| KPI | Qué mide | Resultado / criterio | Importancia |

\|---|---|---|---|

\| Retención de registros | Proporción de registros que se conserva tras la limpieza | 113.549 de 114.000 registros originales (aprox. 99,60 %) | Controla la pérdida de información durante la preparación |

\| Completitud del dataset limpio | Presencia de valores nulos en el dataset usado | 0 valores nulos pendientes tras la limpieza | Reduce errores en entrenamiento y análisis |

\| Consistencia de popularidad | Tipo y rango de `popularity` | Variable numérica entre 0 y 100 | Asegura que la variable objetivo sea interpretable |

\| Error de regresión | Error absoluto y cuadrático medio de la predicción | MAE = 14,2308; RMSE = 19,3945 | Cuantifica el error de predicción de popularidad |

\| Capacidad explicativa de regresión | Proporción de variabilidad explicada | R² de prueba = 0,2501 | Indica que el modelo lineal explica cerca de una cuarta parte de la variabilidad de la popularidad |

\| Desempeño del clasificador | Proporción de aciertos y F1 macro | Accuracy = 84,04 %; F1 macro = 0,7456 en el modelo optimizado v2 (versión inicial: 57,29 % y 0,4757; ver 9.3.1) | Resume el rendimiento y ayuda a considerar el desbalance entre clases |

\| Calidad de segmentación | Separación y cohesión de los clústeres | Silhouette = 0,1616 para K=4; mejor valor comparado = 0,2499 para K=2 | Orienta la elección de número de grupos junto con su interpretabilidad |

Los resultados indican que el árbol de decisión optimizado (v2) mejora de forma importante respecto del modelo base y de su versión inicial, superando el 80 % de accuracy, mientras que la regresión lineal explica cerca de un 25 % de la variabilidad de la popularidad, en su mayor parte gracias al género musical. En K-Means, el mejor Silhouette Score corresponde a K=2, aunque se conserva K=4 por la utilidad analítica de describir cuatro perfiles musicales.

# 6. Aplicación de CRISP-DM
## 6.1 Business Understanding
El problema de negocio es comprender qué características musicales se relacionan con la popularidad de una canción y evaluar enfoques de Machine Learning que permitan estimar la popularidad, clasificar sus niveles y descubrir segmentos musicales. El resultado es un análisis académico exploratorio que aporta evidencia cuantitativa, sin afirmar causalidad ni representar una solución desplegada en Spotify.

## 6.2 Data Understanding
Se cargó el archivo `Spotify_Tracks_Dataset.csv` y se revisaron sus dimensiones, tipos, valores nulos, duplicados, rangos y distribuciones. Se identificó `popularity` como variable objetivo y se analizaron variables de audio, contenido explícito y género musical. También se revisaron casos especiales como `tempo = 0`, duraciones extensas y popularidad igual a 0.

## 6.3 Data Preparation
La función `limpieza_datos()` normaliza los nombres de columnas, elimina el índice original, valida las columnas requeridas, limpia textos, convierte variables numéricas, elimina filas con valores nulos y elimina duplicados exactos. Para el árbol de decisión se derivó `popularity_level` a partir de `popularity`. En los pipelines supervisados, `track_genre` se codificó mediante One-Hot Encoding y, en la regresión lineal, las variables numéricas se estandarizaron con `StandardScaler`; para K-Means, las diez variables seleccionadas se estandarizaron mediante `StandardScaler`.

## 6.4 Modeling
En la Evaluación 2 se entrenaron tres enfoques de Machine Learning: regresión lineal para estimar `popularity` como valor numérico; árbol de decisión para clasificar las canciones en niveles de popularidad baja, media y alta; y K-Means para identificar grupos de canciones según sus características de audio. En los modelos supervisados, la variable objetivo se mantiene separada de los predictores para evitar filtraciones de información.

## 6.5 Evaluation
La evaluación incluye métricas de calidad de datos y métricas específicas de cada modelo. Para regresión se calcularon MAE, MSE, RMSE y R²; para clasificación, accuracy, precision, recall, F1 y matriz de confusión; y para K-Means, Silhouette Score, tamaños de clúster y perfiles promedio. Los resultados y sus limitaciones se detallan en la sección 9. La evaluación muestra que el árbol de decisión optimizado mejora el desempeño respecto del modelo base, la regresión lineal alcanza un R² cercano a 0,25 y K-Means presenta separación limitada entre grupos.

## 6.6 Deployment
El despliegue en producción no forma parte del trabajo realizado. Sí se guardaron los modelos entrenados y los artefactos necesarios en la carpeta `models/`, y se generaron archivos de datos y gráficos en `data/` e `images/` para facilitar la revisión y reproducibilidad local. Un despliegue como servicio batch o aplicación sería una etapa futura.

# 7. Segunda parte del EDA: relaciones y comparaciones
El EDA avanzado compara la popularidad según género, contenido explícito y características de audio. La popularidad promedio más alta entre los grupos mostrados corresponde a `pop-film`, con **59,28**, seguido por `k-pop`, con **56,96**. Estas diferencias sugieren que el género debe incorporarse como variable categórica en el análisis posterior.

![Popularidad promedio por género](../images/popularidad_promedio_genero.png)

**Figura 7.** Géneros con mayor popularidad promedio.

Las canciones con contenido explícito presentan una popularidad promedio de **36,52**, mientras que las canciones sin contenido explícito alcanzan **33,02**. La diferencia es descriptiva y no permite afirmar que el contenido explícito cause una mayor popularidad.

![Distribución de popularidad según contenido explícito](../images/popularidad_por_contenido_explicito.png)

**Figura 8.** Distribución de popularidad según contenido explícito.

La matriz de correlación muestra relaciones fuertes entre algunas variables de audio entre sí, especialmente `energy` y `loudness` (**0,76**) y `energy` y `acousticness` (**-0,73**). Estas relaciones deben considerarse para evitar interpretar cada variable como completamente independiente.

![Relaciones entre audio y popularidad](../images/relaciones_audio_popularidad.png)

**Figura 9.** Relaciones entre `energy`, `danceability` y `popularity`.

![Matriz de correlación](../images/matriz_correlacion_audio_popularidad.png)

**Figura 10.** Matriz de correlación entre características de audio y popularidad.

En relación directa con `popularity`, las correlaciones lineales son débiles. La mayor asociación absoluta corresponde a `instrumentalness` (**-0,0947**), seguida por `loudness` (**0,0474**) y `speechiness` (**-0,0455**). Esto indica que la popularidad probablemente depende de combinaciones de variables, categorías y factores no incluidos en el dataset, por lo que no se debe basar la selección únicamente en correlaciones simples.

# 8. Hallazgos principales
- La popularidad está concentrada en niveles bajos y medios; las canciones con `popularity >= 75` representan **2.829 de 113.549 registros**, aproximadamente **2,49%**.

- El grupo de popularidad alta presenta mayor `danceability` promedio (**0,632**) que el resto (**0,565**) y mayor `energy` promedio (**0,679** frente a **0,641**).

- Las canciones con popularidad alta tienen una proporción de contenido explícito de **19,30%**, superior al **8,28%** observado en el resto.

- `track_genre` presenta diferencias relevantes entre grupos: `pop-film` y `k-pop` aparecen entre los géneros con mayor popularidad promedio dentro del análisis.

- Se observan relaciones importantes entre predictores, pero las asociaciones lineales individuales con `popularity` son débiles. Por ello, las conclusiones deben complementarse con modelos y análisis por grupos.

- La duración presenta una asociación prácticamente nula con la popularidad en este conjunto, con correlación **-0,0079**. Los valores extremos de duración deben conservarse mientras correspondan a contenidos válidos, pero conviene revisar su influencia durante el modelamiento.

- El dataset limpio quedó preparado para el entrenamiento. Las transformaciones se aplicaron según el modelo: codificación de género en los modelos supervisados, y estandarización de variables numéricas en la regresión lineal y en K-Means.

# 9. Modelamiento, evaluación y optimización
Esta sección documenta los modelos desarrollados en la Evaluación 2. Los tres enfoques responden preguntas diferentes y sus métricas no son directamente comparables entre sí: la regresión estima un valor numérico, la clasificación predice una categoría y K-Means agrupa canciones sin utilizar una etiqueta objetivo.

## 9.1 Preparación común y reproducibilidad
El dataset utilizado por los scripts es `data/Spotify_Tracks_Limpio.csv`, con 113.549 filas y 20 columnas. Los scripts se organizan bajo `src/entrenamiento/`, `src/evaluacion/` y `src/optimizacion/`. Los modelos serializados se guardan en `models/`, mientras que las visualizaciones y resultados auxiliares se guardan en `images/` y `data/`.

Para los modelos supervisados se utilizaron transformaciones de variables categóricas mediante `OneHotEncoder(handle_unknown="ignore")` y pipelines de scikit-learn. En la regresión lineal, el codificador se usa en formato denso (`sparse_output=False`) y las variables numéricas se estandarizan con `StandardScaler`; sin este escalado el solver no converge y el modelo queda subajustado. La partición de entrenamiento y prueba utiliza `test_size=0.2` y `random_state=42`, con 90.839 registros para entrenamiento y 22.710 para prueba. Para el árbol de decisión se utiliza además una partición estratificada, dado el desbalance entre los niveles de popularidad.

## 9.2 Regresión lineal: predicción de popularidad
**Objetivo.** Estimar el valor numérico de `popularity` en una escala de 0 a 100.

**Predictores.** Se utilizaron características numéricas de audio y duración (`duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo` y `time_signature`), estandarizadas con `StandardScaler`, junto con `track_genre` codificado mediante One-Hot Encoding denso.

**Nota metodológica.** Una versión previa del pipeline no escalaba las variables numéricas y generaba una matriz dispersa. Al combinar variables en escalas muy distintas (por ejemplo, `duration_ms` en cientos de miles frente a valores entre 0 y 1), el solver iterativo no convergía y el modelo quedaba subajustado (R² ≈ 0,003). Esa versión fue descartada; los resultados de esta sección corresponden al pipeline corregido.

**Evaluación del modelo base.**

\| Métrica | Resultado |

\|---|---:|

\| MAE | 14,2308 |

\| MSE | 376,1475 |

\| RMSE | 19,3945 |

\| R² en prueba | 0,2501 |

\| R² en entrenamiento | 0,2593 |

El error absoluto medio es de aproximadamente 14,23 puntos de popularidad. El R² de 0,25 indica que el modelo explica cerca de una cuarta parte de la variabilidad de la popularidad; el 75 % restante no es explicado por los predictores disponibles. La diferencia entre R² de entrenamiento (0,2593) y de prueba (0,2501) es pequeña, por lo que no se observa un sobreajuste marcado.

**Aporte de cada grupo de predictores.** Como análisis complementario (`src/evaluacion/ev_regresion_referencia.py`, misma partición), se compararon modelos lineales con distintos predictores:

\| Predictores | R² en prueba | MAE en prueba |

\|---|---:|---:|

\| Solo `track_genre` | 0,2437 | 14,2880 |

\| Solo variables numéricas | 0,0266 | 18,4790 |

\| Género + numéricas (modelo final) | 0,2501 | 14,2308 |

Casi toda la capacidad explicativa proviene del género musical. Las variables de audio y duración, por sí solas, explican apenas un 2,7 % de la variabilidad, lo que es coherente con las correlaciones lineales débiles observadas en el EDA.

**Optimización.** Se compararon Ridge mediante `GridSearchCV` (`alpha` en 0,01; 0,1; 1; 10; 100) y Lasso mediante `RandomizedSearchCV` (`alpha` con distribución logarítmica entre 0,001 y 10).

\| Método | Mejor R² en validación cruzada | R² en prueba | RMSE en prueba | MAE en prueba | Parámetro destacado |

\|---|---:|---:|---:|---:|---|

\| Ridge + GridSearchCV | 0,2569 | 0,2501 | 19,3944 | 14,2335 | `alpha=1` |

\| Lasso + RandomizedSearchCV | 0,2569 | 0,2502 | 19,3935 | 14,2553 | `alpha≈0,0012` |

Ambas alternativas empatan en R² de validación cruzada (al cuarto decimal) y obtienen resultados prácticamente idénticos en prueba. Ridge fue la alternativa seleccionada por el criterio del script (mejor o igual R² de validación cruzada). Respecto del modelo base, la regularización no aporta una mejora relevante (RMSE de 19,3945 a 19,3944), lo que indica que el límite del modelo lineal está en la información disponible en los predictores y no en el sobreajuste. El R² de validación cruzada (0,2569) es consistente con el de prueba (0,2501).

**Artefactos.** `models/regresion_lineal.pkl` y `models/regresion_lineal_optimizado.pkl`.

## 9.3 Árbol de decisión: clasificación de popularidad
**Objetivo.** Clasificar las canciones en tres niveles mediante la variable `popularity_level`, derivada de `popularity` con los intervalos baja (0–33), media (34–66) y alta (67–100), implementados con los cortes `[-1, 33, 66, 100]`.

**Distribución de clases.** Baja: 55.179 registros (48,59 %); media: 50.743 (44,69 %); alta: 7.627 (6,72 %). La clase alta está subrepresentada, por lo que se utilizó `class_weight="balanced"` y partición estratificada. Las características predictoras son las mismas características musicales descritas para regresión, con `track_genre` codificado.

**Modelo base.** Se utilizó `DecisionTreeClassifier(random_state=42, max_depth=4, class_weight="balanced")`.

\| Indicador | Resultado |

\|---|---:|

\| Accuracy | 0,2849 (28,49 %) |

\| F1 macro | 0,2823 |

\| F1 ponderado | 0,3288 |

El modelo base presentó un desempeño bajo y un tratamiento desigual de las clases. En particular, su precisión para la clase alta fue 0,0924, aunque su recall fue 0,7843, lo que significa que identificó una proporción importante de canciones realmente altas, pero también asignó esa clase a muchos ejemplos que no lo eran.

**Optimización.** Se compararon GridSearchCV y RandomizedSearchCV utilizando `f1_macro` como criterio de selección, para dar peso equivalente al desempeño en cada clase. La búsqueda aleatoria fue la alternativa seleccionada, con parámetros `criterion='gini'`, `max_depth=14`, `min_samples_leaf=9` y `min_samples_split=18`.

\| Indicador | Modelo base | Modelo optimizado |

\|---|---:|---:|

\| Accuracy | 28,49 % | 57,29 % |

\| F1 macro | 0,2823 | 0,4757 |

Reporte del modelo optimizado:

\| Clase | Precision | Recall | F1-score | Soporte |

\|---|---:|---:|---:|---:|

\| Alta | 0,35 | 0,28 | 0,31 | 1.525 |

\| Baja | 0,83 | 0,30 | 0,44 | 11.036 |

\| Media | 0,53 | 0,91 | 0,67 | 10.149 |

\| Macro promedio | 0,57 | 0,50 | 0,48 | 22.710 |

\| Promedio ponderado | 0,66 | 0,57 | 0,54 | 22.710 |

La optimización elevó la accuracy de 28,49 % a 57,29 % y el F1 macro de 0,2823 a 0,4757. Aun así, el recall de la clase alta es 0,28, de modo que el clasificador no identifica todas las canciones populares. El resultado global debe interpretarse junto con el desempeño por clase y no como evidencia de una solución perfecta.

**Artefactos de la versión inicial.** `models/arbol_decision.pkl` y `models/arbol_decision_optimizado_v1.pkl` (copia de respaldo del optimizado original, que corresponde a la primera entrega).

### 9.3.1 Mejora posterior: árbol optimizado v2 (accuracy > 80 %)

**Motivación.** Tras la primera entrega, el mejor árbol alcanzaba 57,29 % de accuracy. Se estableció como meta que el modelo optimizado superara el 80 %. La mejora se abordó revisando primero la calidad de la evaluación y luego la información disponible para el modelo, sin modificar la definición de la variable objetivo (`popularity_level` con los mismos cortes `[-1, 33, 66, 100]`) ni el dataset limpio.

**Diagnóstico previo.**

1. *Fuga entre copias de una misma canción.* El dataset limpio contiene 23.809 filas con `track_id` repetido (la misma canción listada en varios géneros). Con la partición aleatoria original, copias de una misma canción podían caer en entrenamiento y en prueba. Esto infla la accuracy de árboles profundos: un árbol sin límite de profundidad llegaba a 70,5 % simplemente memorizando esas copias, sin aprender nada generalizable. Por ello, ese camino no se consideró una mejora válida.
2. *Poca información en las variables de audio.* Con solo variables de audio y género, incluso un árbol bien ajustado no pasa de ~60 % cuando se evita la fuga. Coincide con lo observado en la regresión lineal: las características de audio explican poco de la popularidad.
3. *Información de contexto no utilizada.* Las columnas `artists` y `album_name` estaban en el dataset pero no se usaban. La popularidad de un artista o álbum es información disponible antes de lanzar una canción nueva, y es la que más relación tiene con su nivel de popularidad.

**Cambios realizados.**

| # | Cambio | Archivo | Motivo |
|---|---|---|---|
| 1 | Partición 80/20 **agrupada por `track_id`** (`GroupShuffleSplit`, `random_state=42`): 90.922 registros de entrenamiento y 22.627 de prueba | `src/preparacion_arbol.py` (nuevo) | Evita que una misma canción esté en entrenamiento y prueba; la métrica es más exigente y fiable |
| 2 | Nueva variable `first_artist` (artista principal, primer valor de `artists` separado por `;`) | `src/preparacion_arbol.py` | Permite medir el historial del artista |
| 3 | Transformador `HistorialPopularidad`: *target encoding* de `first_artist` y `album_name`. Por cada uno genera la proporción suavizada de canciones Baja/Media/Alta y la cantidad de canciones conocidas (`_n`) | `src/transformadores.py` (nuevo) | Incorpora el historial de popularidad como variables numéricas que el árbol puede usar |
| 4 | El encoding se calcula **fuera de pliegue** (5 pliegues agrupados por `track_id`) durante el ajuste; en prueba se usan solo estadísticas de entrenamiento; entidades no vistas reciben la proporción global y `n = 0` | `src/transformadores.py` | Impide que una canción se codifique con su propia etiqueta (fuga del objetivo). Va dentro del `Pipeline`, por lo que también se aplica dentro de la validación cruzada |
| 5 | Validación cruzada agrupada (`GroupKFold`, 3 pliegues) en lugar de `cv=5` aleatorio | `src/optimizacion/op_arbol_decision.py` | Consistencia con la partición; evita estimaciones optimistas en la selección de hiperparámetros |
| 6 | Criterio de selección: **accuracy** (`refit="accuracy"`), registrando también F1 macro | `src/optimizacion/op_arbol_decision.py` | La meta es accuracy; el F1 macro se mantiene como control del desempeño por clase |
| 7 | `class_weight` (`None` / `balanced`) pasa a formar parte de la búsqueda; antes estaba fijo en `balanced` | `src/optimizacion/op_arbol_decision.py` | `balanced` penalizaba la accuracy global; se deja que la búsqueda decida |
| 8 | Nueva rejilla: `criterion` {gini, entropy}, `max_depth` {8, 10, 12}, `min_samples_leaf` {5, 20}, `min_samples_split` = 2, `class_weight` {None, balanced} (24 combinaciones) y búsqueda aleatoria de 10 iteraciones | `src/optimizacion/op_arbol_decision.py` | Rango de profundidad centrado en la zona donde el árbol rinde sin sobreajustarse |
| 9 | Script de evaluación del modelo v2 y comparación justa con los modelos anteriores | `src/evaluacion/ev_arbol_decision_optimizado.py` (nuevo) | Reproduce todas las métricas de esta sección |
| 10 | Respaldo del modelo optimizado de la primera entrega | `models/arbol_decision_optimizado_v1.pkl` | Conserva la versión previa; `arbol_decision_optimizado.pkl` pasa a ser la v2 |

**Modelo seleccionado.** GridSearchCV, con `criterion='gini'`, `max_depth=8`, `min_samples_leaf=20`, `min_samples_split=2` y `class_weight=None`. Accuracy promedio en validación cruzada: 0,8295.

**Resultados.** Todos los modelos se evaluaron sobre el **mismo conjunto de prueba agrupado** (22.627 registros), para que la comparación sea justa:

| Indicador | Base (`max_depth=4`) | Optimizado v1 | **Optimizado v2** |
|---|---:|---:|---:|
| Accuracy | 28,70 % | 57,96 % | **84,04 %** |
| F1 macro | 0,2851 | 0,4885 | **0,7456** |

(Con la partición aleatoria original, el base y el v1 obtuvieron 28,49 % y 57,29 %; en la partición agrupada sus resultados son casi iguales, de modo que la mejora de la v2 no se debe al cambio de partición.)

Reporte del modelo optimizado v2:

| Clase | Precision | Recall | F1-score | Soporte |
|---|---:|---:|---:|---:|
| Alta | 0,64 | 0,43 | 0,52 | 1.579 |
| Baja | 0,89 | 0,88 | 0,89 | 10.846 |
| Media | 0,81 | 0,86 | 0,83 | 10.202 |
| Macro promedio | 0,78 | 0,72 | 0,75 | 22.627 |
| Promedio ponderado | 0,84 | 0,84 | 0,84 | 22.627 |

Matriz de confusión (filas: real; columnas: predicha):

| | Pred. Baja | Pred. Media | Pred. Alta |
|---|---:|---:|---:|
| **Real Baja** | 9.567 | 1.248 | 31 |
| **Real Media** | 1.085 | 8.766 | 351 |
| **Real Alta** | 101 | 795 | 683 |

**Análisis de la mejora.** La accuracy subió de 57,29 % (v1, partición original) a 84,04 %, superando la meta de 80 %. El F1 macro pasó de 0,4757 a 0,7456, y la precisión de la clase alta aumentó de 0,35 a 0,64, por lo que la mejora no se concentra solo en las clases mayoritarias. La mejora proviene principalmente del historial de artista y álbum, no de un ajuste más fino de hiperparámetros.

**Limitaciones y condiciones de uso.**

- *El resultado depende de conocer el artista o el álbum.* En las canciones de prueba cuyo artista principal sí aparece en entrenamiento, la accuracy es 85,93 %; en las 1.982 canciones (8,8 % de la prueba) de artistas **no vistos** baja a 64,33 %, y con solo variables de audio el árbol queda cerca de 60 %. El modelo es adecuado para clasificar canciones de artistas o álbumes con historial, pero no debe presentarse como un predictor de popularidad basado en las características del audio.
- La clase alta sigue siendo la más difícil: su recall es 0,43, es decir, el modelo aún no detecta más de la mitad de las canciones realmente altas (795 de 1.579 se clasifican como medias). Se prefirió accuracy como criterio de selección, con `class_weight=None`, a costa de este recall; si el objetivo de negocio fuera detectar canciones muy populares, convendría priorizar el recall de esa clase.
- Parte de la señal puede reflejar que las canciones de un mismo álbum tienden a tener popularidad parecida (por ejemplo, álbumes completos con popularidad 0). Es un patrón real del dataset, pero no una relación causal con el contenido musical.
- Como un mismo artista puede aparecer en entrenamiento y prueba con canciones distintas, la métrica refleja el escenario «artista conocido». La evaluación con artistas completamente nuevos se reporta aparte (64,33 %) por ser el caso más exigente.
- Por limitaciones de cómputo (un solo núcleo), la búsqueda se realizó con 3 pliegues y una rejilla de 24 combinaciones; una búsqueda más amplia podría ajustar levemente los resultados.

**Artefactos de la v2.** `models/arbol_decision_optimizado.pkl` (modelo v2, un `Pipeline` único; requiere `src/transformadores.py` para cargarse), `src/transformadores.py`, `src/preparacion_arbol.py`, `src/optimizacion/op_arbol_decision.py` y `src/evaluacion/ev_arbol_decision_optimizado.py`.

## 9.4 K-Means: segmentación musical
**Objetivo.** Identificar grupos de canciones con características de audio similares, sin utilizar `popularity` como variable para formar los clústeres.

**Variables utilizadas.** `duration_ms`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` y `tempo`. Antes del entrenamiento se aplicó `StandardScaler`, debido a que las variables están en escalas diferentes.

**Configuración final.** `KMeans(n_clusters=4, random_state=42, n_init=10)`. Se entrenó también una versión sin escalar únicamente con fines exploratorios y comparativos; esta versión no se utiliza como modelo definitivo. La segmentación final corresponde al modelo entrenado sobre las diez variables después de aplicar `StandardScaler`. El Silhouette Score de la solución de cuatro clústeres fue 0,1616, lo que indica una separación limitada entre grupos; por ello, los perfiles se deben interpretar como tendencias y no como categorías musicales perfectas. Este valor se calcula sobre una muestra aleatoria de 10.000 canciones (`random_state=42`), por lo que puede variar levemente (aproximadamente entre 0,159 y 0,163) con otra muestra.

\| Clúster | Registros | Proporción | Danceability media | Energy media | Acousticness media | Liveness media | Popularidad media |

\|---|---:|---:|---:|---:|---:|---:|---:|

\| 0 | 8.702 | 7,66 % | 0,53 | 0,75 | 0,34 | 0,73 | 34,14 |

\| 1 | 47.411 | 41,75 % | 0,69 | 0,69 | 0,25 | 0,16 | 33,49 |

\| 2 | 34.461 | 30,35 % | 0,49 | 0,80 | 0,08 | 0,19 | 34,14 |

\| 3 | 22.975 | 20,23 % | 0,46 | 0,27 | 0,78 | 0,16 | 31,46 |

**Interpretación exploratoria de los clústeres:**

- **Clúster 0:** alta energía y liveness, compatible con pistas con rasgos de interpretación en vivo.

- **Clúster 1:** mayor bailabilidad y valence relativamente alta, compatible con canciones rítmicas y positivas.

- **Clúster 2:** energía y tempo elevados, con baja acousticness y valence relativamente baja (0,33).

- **Clúster 3:** baja energía y mayor acousticness, compatible con canciones acústicas o suaves.

Estos nombres son interpretaciones descriptivas de promedios de variables; no corresponden a etiquetas de género verificadas. Las medias de popularidad entre clústeres son relativamente cercanas, por lo que la segmentación no demuestra por sí sola que un perfil sea mucho más popular que otro.

**Optimización del número de clústeres.** Se compararon valores de K entre 2 y 7 mediante Silhouette Score, además del método del codo basado en la inercia.

\| K | Silhouette Score | Inercia |

\|---:|---:|---:|

\| 2 | 0,2499 | 916.631,5443 |

\| 3 | 0,1556 | 823.371,4685 |

\| 4 | 0,1616 | 754.444,3254 |

\| 5 | 0,1662 | 691.752,5114 |

\| 6 | 0,1662 | 639.591,6314 |

\| 7 | 0,1767 | 590.211,3983 |

K=2 obtuvo el mejor Silhouette Score de las alternativas evaluadas. No obstante, se mantuvo K=4 como configuración final porque permite describir cuatro perfiles diferenciados con mayor granularidad para el análisis. Esta decisión prioriza la utilidad interpretativa, no la máxima separación estadística. Si el criterio principal fuera exclusivamente maximizar Silhouette Score, K=2 sería la alternativa preferible.

**Artefactos.** `models/modelo_kmeans.pkl`, `models/scaler_kmeans.pkl`, `data/Spotify_Tracks_Clusters.csv`, `data/resultados_optimizacion_kmeans.csv`, `images/clusters_spotify.png` e `images/metodo_codo_kmeans.png`. El script `src/entrenamiento/kmeans.py` genera la segmentación final con K=4; `src/optimizacion/op_kmeans.py` compara K=2 a K=7 y genera la tabla de resultados y el gráfico del método del codo.

## 9.5 Comparación y conclusiones de modelamiento
Los resultados muestran que no existe un único modelo que resuelva todas las preguntas del proyecto. La regresión lineal explica cerca de un 25 % de la variabilidad de la popularidad (casi por completo gracias al género musical) con un error medio de unos 14 puntos; el árbol optimizado v2 supera el 80 % de accuracy (84,04 %) gracias al historial de artista y álbum, aunque mantiene dificultades con la clase alta (recall 0,43) y baja a 64,33 % con artistas no vistos; y K-Means permite explorar perfiles musicales, con separación limitada y diferencias pequeñas en popularidad promedio.

Por tanto, los hallazgos deben comunicarse como resultados analíticos exploratorios, no como evidencia de causalidad ni como garantía de predicción precisa. La popularidad puede depender de factores no incluidos en las variables del dataset, como exposición, promoción, trayectoria del artista, actualidad o contexto de escucha. Estos factores son hipótesis plausibles, no variables comprobadas por este análisis.

## 9.6 Estructura del proyecto y ejecución
La estructura utilizada para organizar el proyecto es:

```text

eva1_machine_learning/

├── data/

│   ├── Spotify_Tracks_Dataset.csv

│   ├── Spotify_Tracks_Limpio.csv

│   ├── Spotify_Tracks_Clusters.csv

│   └── resultados_optimizacion_kmeans.csv

├── images/

├── models/

├── src/

│   ├── limpieza.py

│   ├── entrenamiento/

│   │   ├── regresion_lineal.py

│   │   ├── arbol_decision.py

│   │   └── kmeans.py

│   ├── evaluacion/

│   │   ├── ev_regresion_lineal.py

│   │   ├── ev_arbol_decision.py

│   │   ├── ev_kmeans.py

│   │   └── ev_regresion_referencia.py

│   └── optimizacion/

│       ├── op_regresion_lineal.py

│       ├── op_arbol_decision.py

│       └── op_kmeans.py

└── docs/

    └── informe_tecnico.md

```

El proyecto utiliza Python y bibliotecas como pandas, NumPy, scikit-learn, matplotlib y joblib. Para reproducir los resultados, se debe contar con las dependencias instaladas y el archivo de datos original en `data/`. Los scripts deben ejecutarse desde el proyecto respetando el orden lógico: limpieza, entrenamiento, evaluación y optimización. Para K-Means, `src/entrenamiento/kmeans.py` entrena y guarda el modelo final con cuatro clústeres, mientras que `src/optimizacion/op_kmeans.py` evalúa distintas cantidades de clústeres sin sobrescribir el modelo definitivo ni el dataset final de clústeres. La disponibilidad exacta de versiones depende del entorno donde se ejecute el proyecto; para una reproducibilidad estricta se recomienda registrar las versiones instaladas en un archivo `requirements.txt`.
