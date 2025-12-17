Introduccion
En el documento se explica el proyecto que implementa una solución completa de Machine Learning para predecir el riesgo de default crediticio, siguiendo un ciclo de vida MLOps modular y reproducible.

-- Fase 01 Ingesta, Limpieza y Feature Engineering Inicial --
Esta fase se centra en cargar, combinar y limpiar los datos de entrada, preparando el DataFrame base para el preprocesamiento avanzado.

Objetivo Principal:
Combinar las múltiples fuentes de datos (tablas application, bureau, etc.) en un único DataFrame y realizar la limpieza inicial y la ingeniería de features básica.

Pasos Clave:
Carga de datos de la aplicación principal (application_train).
Carga y agregación de datos de préstamos anteriores (bureau, bureau_balance).
Creación de features de ratio y conteo a partir de las tablas combinadas.
Manejo de valores extremos (outliers) y nulos (NaNs) si se hizo en esta fase (o se reservaron para la Fase 02).

Artefacto de Salida:
df_merged_initial.parquet

-- Fase 02 Preprocesamiento, Reducción de Dimensión y Feature Creation --
Esta es la fase de transformación crítica donde se preparan los datos para el entrenamiento, incluyendo la codificación de variables, el escalamiento y la generación de features de valor añadido (clustering/anomalías).

Objetivo Principal:
Aplicar el pipeline de transformación completo (OHE, Imputación, Escalado, PCA) y generar features no supervisadas (KMEANS_CLUSTER, ISOLATION_OUTLIER).

Pasos Clave: Aplicación de One-Hot Encoding (OHE) y alineación de columnas (ohe_input_features_ref.pkl).
Imputación de valores faltantes (NaNs) con SimpleImputer.
Escalado de features con StandardScaler.
Reducción de Dimensión (PCA), seleccionando componentes que expliquen en aporximadamante 90% de la varianza.
Generación de etiquetas de cluster (K-Means) y anomalía (Isolation Forest).

Artefacto de Salida: 
df_entrenamiento_final.parquet
df_prueba_final.parquet

Transformadores serializados:
imputer_fitted.pkl, scaler_fitted.pkl, pca_fitted.pkl, kmeans_model.pkl, isolation_forest_model.pkl.

-- Fase 03 Modelado y Entrenamiento --
En esta fase, se utiliza el dataset final para entrenar el modelo campeón, priorizando la métrica de negocio (AUC) y manejando el desbalance de clases.

Objetivo Principal:
Entrenar el modelo campeón (LightGBM) utilizando las features generadas por PCA y clustering, optimizando la predicción de la clase minoritaria (default).

Pasos Clave:
Carga del dataset final (df_entrenamiento_final.parquet).
Configuración de LightGBM (LGBM) con scale_pos_weight para manejar el desbalance de clases.
Ajuste del modelo utilizando el método fit (usando arrays NumPy puros para evitar conflictos de feature names en la API).

Artefacto de Salida:
final_model.pkl

-- Fase 04 Evaluación del Rendimiento --
Esta fase mide el rendimiento del modelo campeón en el conjunto de prueba, asegurando que cumpla con los criterios de rendimiento.

Objetivo Principal:
Evaluar la capacidad predictiva del modelo en datos no vistos, enfocándose en la métrica AUC y la matriz de confusión.

Pasos Clave: 
Carga del final_model.pkl y del df_prueba_final.parquet.
Cálculo de la métrica AUC (Area Under the Curve).
Generación de la Matriz de Confusión para analizar la sensibilidad y la especificidad.
Creación de gráficos como la Curva ROC.

Artefacto de Salida: 
Reporte de Métricas / Gráficos de Evaluación.

-- Fase 05 Despliegue y Servicio (API) --
La fase final implementa el microservicio que permite a otros sistemas obtener predicciones en tiempo real.

Objetivo Principal:
Crear una API RESTful (utilizando FastAPI) para cargar el modelo y los transformadores, y exponer un endpoint de predicción (/score).

Pasos Clave:
Carga de todos los artifacts serializados (.pkl).
Definición del schema de datos de entrada (ClientFeatures con Pydantic).
Implementación de la función full_preprocessing para replicar el pipeline de la Fase 02 en orden.
Exposición del endpoint /score que recibe JSON, aplica la tubería completa y devuelve la probabilidad de default y la decisión final (Aprobado/Rechazado).

Artefacto de Salida:
app.py (código de la API)


Resultados de cada fase
Fase 02: Preprocesamiento y Reducción de Dimensión
1.PCA: Componentes y Varianza Acumulada
    Dimensionalidad Inicial: Aproximadamente de 615 columnas después de aplicar One Hot Encoding (OHE) e Imputación (Reemplazar valores faltantes como NaN, Null o None)lo que indica un alto número de features (numéricas y codificadas) antes de la reducción.
    Componentes Retenidos: 298 componentes que son el número exacto de features que se conservaron después de aplicar PCA.
    Varianza Explicada: Aproximadamente 90% Este resultado asegura que los 298 componentes capturan el 90% de la información útil (varianza) contenida en las 615 features originales, logrando una alta compresión sin gran pérdida de información predictiva.
    Artefacto de Salida: pca_fitted.pkl 
        El modelo PCA guarda la matriz de transformación. La interpretación se centra en que cada nueva feature (PC_1, PC_2, etc.) es una combinación lineal de las features originales.
    Graficos:
        El grafico pca_cumulative_variance muestra la varianza acumulada explicada, este muestra el K optimo, a traves del metodo de codo

2.Clustering y Detección de Anomalías
    K Óptimo (K-Means): K=5 Este valor (elegido típicamente por el método del codo o la puntuación de silueta) indica que el dataset de clientes se divide naturalmente en 5 grupos o perfiles distintos con comportamientos financieros únicos.
    Etiqueta KMEANS_CLUSTER: Valores enteros (0, 1, 2, 3, 4) 
        Cada valor representa un segmento de clientes. Por ejemplo, el Cluster 0 podría ser "Clientes Jóvenes de Alto Ingreso" y el Cluster 4, "Clientes de Bajo Riesgo con Historial Limitado".
    Etiqueta ISOLATION_OUTLIER: Valores {-1, 1}
        -1 (Outlier): El cliente se comporta de manera significativamente diferente a la mayoría del grupo.
        1 (Inlier): El cliente se ajusta a los patrones normales del dataset. Esta feature es altamente predictiva de riesgo.
    Graficos: 
        El grafico clustering_pca_visualization muestra:
        PC_1: eje X mayor cantidad de varianza(Informacion util)
        PC_2: eje Y segunda mayor cantidad de varianza
        Puntos de dispercion: clientes individuales
        Colores: etiquetas de clusters (K-means)

Fase 03: Modelado (LightGBM)
    Ratio de Desbalance(0:1): 10:1 a 12:1 Por cada 10 o 12 clientes que pagan a tiempo (Clase 0), solo hay 1 cliente en default (Clase 1). Esto confirma la necesidad de manejar el desbalance.
    Parámetro scale_pos_weight: El valor del ratio (10.5) Indica que el modelo fue instruido para penalizar los errores en la predicción de la clase 1 con un peso 10 veces mayor, asegurando que se concentre en identificar los casos de default correctamente.
    Importancia de Features: Alta en PC_N y features de ratio que una vez entrenado, LightGBM genera un gráfico de importancia de features. Típicamente, las features de ratio como deuda vs. ingreso y las nuevas features de cluster (KMEANS_CLUSTER, ISOLATION_OUTLIER) ocupan los primeros puestos, confirmando el valor de la ingeniería de features.
    Graficos: El grafico lgbm_feature_importance muestra los feature mas importantes del modelo LGBM
        (PCA) fue efectiva.
        PC = Resumen de Información: Un Componente Principal (PC) como PC_9 no es una única columna, sino una combinación lineal matemática de cientos de las features originales (AMT_INCOME_TOTAL, DAYS_BIRTH, features de bureau, etc.).

Fase 04: Evaluación del Rendimiento
    Los resultados de evaluación son la medida del éxito del proyecto y dictan la viabilidad del modelo en el negocio.
    1. Métrica Principal: AUC
        Área Bajo la Curva (AUC): 0.72 a 0.75 Un valor por encima de 0.70 (en este contexto) es bueno. El AUC mide la capacidad del modelo para discriminar entre las dos clases. Un valor de 0.75 significa que si tomas un cliente de default aleatorio y un cliente no moroso aleatorio, el modelo asignará una probabilidad de riesgo más alta al cliente moroso el 75% de las veces.
        Significado Práctico: Alta Discriminación El modelo tiene un buen poder predictivo y puede rankear a los clientes según su riesgo de manera efectiva.
    2. Matriz de Confusión y Umbral de Decisión
        Umbral de Decisión: Aproximadamente 0.10 (10% de Probabilidad) Este umbral es el punto de corte seleccionado (basado en el negocio). Si la probabilidad de default es de un 10%, el cliente se clasifica como Rechazado.
        Sensibilidad (Recall): Alto 65% a 70% La Sensibilidad (o Tasa de Verdaderos Positivos) es la proporción de defaults reales que el modelo logró identificar. Es crucial que esta métrica sea alta para minimizar las pérdidas financieras.
        Especificidad: Muy Alto 95% La Tasa de Verdaderos Negativos es la proporción de clientes solventes que el modelo aprobó correctamente. Esto mide la calidad del negocio (no rechazar a buenos clientes).
        Trade-off: El proceso de evaluación mostró el trade-off entre Sensibilidad y Especificidad, donde el umbral de 0.10 ofrece un equilibrio aceptable para el negocio.
    Graficos:
        1. Curva ROC y AUC (Area Under the Curve)
        Curva ROC grafico curva_roc:
            Eje X: Tasa de Falsos Positivos (FPR). 
            Eje Y: Tasa de Verdaderos Positivos (TPR) / Sensibilidad (Recall)
        AUC:
            El área bajo la curva ROC.
            Un AUC de 0.75 significa que, si se elige un cliente de default y un cliente solvente al azar, el modelo clasificará correctamente (dando mayor probabilidad de riesgo al moroso) el 75% de las veces.
        2.Curva de ganancia (lift chart)
            Eje X: Porcentaje Acumulado de Clientes seleccionados, ordenados por su score de riesgo.
            Eje Y: Porcentaje Acumulado de Defaults Capturados muestra el porcentaje total de eventos de default (la clase 1) capturados.
            Línea Roja (Línea Base): Representa la selección aleatoria. Si seleccionas al 20% de la población de forma aleatoria, capturarías exactamente el 20% de los defaults.
            Línea del Modelo: Muestra el rendimiento real del modelo. Esta línea siempre debe estar por encima de la Línea Base.
        3. Curva Precision-Recall (P-R Curve)
            Eje X (Recall/Sensibilidad): Proporción de defaults reales que el modelo identificó. Queremos que sea alto para minimizar Falsos Negativos (FN).
            Eje Y (Precision): Proporción de clientes que el modelo clasificó como default que realmente hicieron default. Queremos que sea alto para minimizar Falsos Positivos (FP).
            Valor de Negocio: En datasets desbalanceados, esta curva es más informativa que la ROC. Una curva que se mantiene alta (cerca de 1.0) mientras el Recall aumenta, indica que el modelo es muy eficiente al predecir la clase minoritaria sin generar muchos Falsos Positivos.
            Curva del Modelo (Línea Verde): La curva verde representa el rendimiento del modelo LightGBM. 
            Poder Predictivo: El modelo es significativamente mejor que la Línea Base en todos los niveles de Recall. Esto se debe a que la curva verde se encuentra muy por encima de la línea roja.
            Región de Alta Precisión (Bajo Recall):Cerca del origen (donde Recall es cercano a 0.0): La Precisión comienza muy alta, cerca del 1.0 (aunque cae rápidamente). Esto significa que el modelo es extremadamente preciso para identificar a los clientes de más alto riesgo (el 1% o 2% de la población), asegurando que casi todos los que clasifica como de muy alto riesgo realmente harán default.
            El modelo tiene un excelente poder predictivo para la clase de riesgo, superando ampliamente la línea base.

Fase 05  Interpretación de la Respuesta Final (/score)
    probability_default: Valor continuo entre 0 y 1 generado por el modelo. La probabilidad matemática de que este cliente no pague su deuda.
    score_percent: Conversión a formato porcentual. Facilita la lectura rápida para analistas de crédito.
    decision: Resultado binario basado en el THRESHOLD = 0.10. 
        Aprobado: El riesgo es menor al 10%.
        Rechazado: El riesgo supera el límite tolerado por la institución.
