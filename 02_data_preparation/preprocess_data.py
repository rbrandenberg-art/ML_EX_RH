import pandas as pd
import numpy as np
import pyarrow.parquet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import warnings

#Cargar el DataFrame final fusionado (salida de la fase 01)
df_final = pd.read_parquet("df_merged_initial.parquet")

# Definir la variable objetivo (Y) y las características (X)
# TARGET es la variable de incumplimiento (0 o 1)
X = df_final.drop(columns=['TARGET', 'SK_ID_CURR']) # Excluir TARGET y el ID del cliente
Y = df_final['TARGET']
SK_ID_CURR = df_final['SK_ID_CURR'] # Guardar IDs por separado si es necesario para el merge final
# Realizar la división (80% entrenamiento, 20% prueba)
# stratify=Y es crucial para mantener la proporción de la clase minoritaria (incumplimiento)
# en ambos conjuntos, ya que el dataset está desbalanceado.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42, 
    stratify=Y
)
# Crear el DataFrame de TRABAJO (solo entrenamiento)
# Ahora trabajaremos exclusivamente con X_train. Volvemos a juntar TARGET solo por conveniencia 
# para el análisis no supervisado (K-Means/IF), pero solo debe ser usado el X_train en el modelado.
df_entrenamiento = X_train.copy()
df_entrenamiento['TARGET'] = Y_train

print("División train/test realizada. El preprocesamiento se realizará solo en el conjunto de entrenamiento.")
print(f"Dimensiones del Conjunto de Entrenamiento: {df_entrenamiento.shape}")

#PCA Preparacion -------------------------------------------------------------------------------------------------------
# Excluir IDs y TARGET
df_pca = df_final.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

# Seleccionar solo variables numéricas (ya que la mayoría de categóricas fueron convertidas en el merge)
df_pca_numeric = df_entrenamiento.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
df_train_numeric = df_pca_numeric.select_dtypes(include=np.number)
print(f"DataFrame de ENTRENAMIENTO para PCA: {df_train_numeric.shape}")

# Reemplazar NaN con la media
imputer = SimpleImputer(strategy='mean')

# Ajustar y transformar (fit_transform) el DataFrame
df_imputed_train_array = imputer.fit_transform(df_train_numeric)

joblib.dump(imputer, 'artifacts/imputer_fitted.pkl')

# Volver a convertir a DataFrame para el siguiente paso (PCA)
df_pca_imputed = pd.DataFrame(df_imputed_train_array, columns=df_train_numeric.columns)
print("Valores faltantes imputados (Media aprendida solo del entrenamiento).")

# Escalar los datos imputados
scaler = StandardScaler()
df_scaled_train_array = scaler.fit_transform(df_pca_imputed)

joblib.dump(scaler, 'artifacts/scaler_fitted.pkl')

# Volver a convertir a DataFrame
df_pca_scaled = pd.DataFrame(df_scaled_train_array, columns=df_pca_imputed.columns)
print("Datos escalados (μ y σ aprendidas solo del entrenamiento).")
print(f"DataFrame listo para PCA: {df_pca_scaled.shape}")


# Aplicar PCA sin límite de componentes para capturar la varianza total
X_train_pca_ready = df_pca_scaled.values 
feature_names = df_pca_scaled.columns

pca = PCA()
pca.fit(X_train_pca_ready)

# Calcular la varianza acumulada
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Encontrar el número de componentes que explican el 90% de la varianza
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

# Transformar los datos al número óptimo de componentes
pca_final = PCA(n_components=n_components_90)
pca_final.fit(X_train_pca_ready)

df_pca_train_array = pca_final.transform(X_train_pca_ready)

# Volver a convertir a DataFrame con nombres de componentes
pca_cols = [f'PC_{i+1}' for i in range(n_components_90)]
df_pca_final = pd.DataFrame(df_pca_train_array, index=df_pca_scaled.index, columns=pca_cols)

joblib.dump(pca_final, 'artifacts/pca_fitted.pkl')

print("\n--- Resultados del Análisis de PCA ---")
print(f"Se necesitan {n_components_90} componentes (de las {df_pca_numeric.shape[1]} iniciales) para explicar el 90% de la varianza total de los datos.")
print(f"DataFrame reducido (df_pca_final) tiene: {df_pca_final.shape} columnas.")

# --- Transformar el Conjunto de Prueba (X_test) ---

# 1. Aplicar imputación y escalado aprendidos al X_test
X_test_imputed_array = imputer.transform(X_test.select_dtypes(include=np.number))
X_test_scaled_array = scaler.transform(X_test_imputed_array)

# 2. Aplicar la transformación PCA aprendida al X_test
df_pca_test_array = pca_final.transform(X_test_scaled_array)

# Crear DataFrame de prueba final (también necesario para la evaluación final)
df_pca_test_final = pd.DataFrame(df_pca_test_array, columns=pca_cols)

# Guardar el conjunto de prueba transformado (Artifact)
df_pca_test_final.to_parquet("artifacts/df_prueba_processed.parquet", index=False)


# --- APLICACIÓN DE K-MEANS (K=5) -----------------------------------------------------------

# X para K-Means es el conjunto de entrenamiento reducido por PCA
X_pca_train = df_pca_final.values
K_optimo = 5 # Usamos el K validado en la fase de análisis

# Ajustar y predecir SOLO en el entrenamiento
final_kmeans = KMeans(n_clusters=K_optimo, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_pca_train)

# --- Guardar el modelo K-Means ajustado (ARTIFACT) ---
joblib.dump(final_kmeans, 'artifacts/kmeans_model.pkl')
print(f"✅ Modelo K-Means (K={K_optimo}) guardado en /artifacts.")

# --- APLICACIÓN DE ISOLATION FOREST ---------------------------------------------------------

contamination_rate = 0.01

# Ajustar y predecir SOLO en el entrenamiento
if_model = IsolationForest(
    n_estimators=100,
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1
)
outlier_labels = if_model.fit_predict(X_pca_train)

# --- Guardar el modelo Isolation Forest ajustado (ARTIFACT) ---
joblib.dump(if_model, 'artifacts/isolation_forest_model.pkl')
print("✅ Modelo Isolation Forest guardado en /artifacts.")


# --- CREACIÓN DE FEATURES E INTEGRACIÓN AL DATASET DE ENTRENAMIENTO ---

# Las nuevas etiquetas se añaden al DataFrame de ENTRENAMIENTO original, 
# ya que comparten el mismo índice y orden.
df_entrenamiento['KMEANS_CLUSTER'] = cluster_labels
df_entrenamiento['ISOLATION_OUTLIER'] = outlier_labels

# El DataFrame de Entrenamiento ahora tiene todas las features, incluidas las no supervisadas.
print("✅ Features KMEANS_CLUSTER e ISOLATION_OUTLIER creadas en df_entrenamiento.")


# --- GUARDAR EL DATASET FINAL DE ENTRENAMIENTO PARA MODELADO ---

# Este es el dataset que la fase /03_modeling utilizará para entrenar el LightGBM/XGBoost.
df_entrenamiento.to_parquet("artifacts/df_entrenamiento_final.parquet", index=True) 
print("✅ Dataset de entrenamiento FINAL (con todas las features) guardado en /artifacts.")
