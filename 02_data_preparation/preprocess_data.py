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
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

#Cargar el DataFrame final fusionado (salida de la fase 01)
df_final_merged = pd.read_parquet("df_merged_initial.parquet")

X = df_final_merged.drop(columns=['TARGET', 'SK_ID_CURR']) 
Y = df_final_merged['TARGET']

# Aplicar One-Hot Encoding (OHE) a las categóricas tipo 'object'
categorical_cols_object = X.select_dtypes(include='object').columns.tolist()
print(f"Features categóricas originales a codificar: {categorical_cols_object}")

# Aplicar OHE a X completo (temporalmente)
X_ohe = pd.get_dummies(X, columns=categorical_cols_object, dummy_na=False)
print(f"Dimensionalidad después de OHE: {X_ohe.shape}")

# Realizar la división (80% entrenamiento, 20% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_ohe, Y, 
    test_size=0.2, 
    random_state=42, 
    stratify=Y
)

# Realineación (Asegura que X_train y X_test tengan las MISMAS columnas)
# Esto resuelve el error: NAME_FAMILY_STATUS_Unknown (si aparece solo en test)
train_cols = X_train.columns
joblib.dump(train_cols, 'artifacts/ohe_input_features_ref.pkl')
names = joblib.load('artifacts/ohe_input_features_ref.pkl')
print(names[-5:])
X_test = X_test.reindex(columns=train_cols, fill_value=0)

# Definir el DataFrame de trabajo de entrenamiento y las referencias
df_entrenamiento = X_train.copy()
df_entrenamiento['TARGET'] = Y_train

# Lista de los nombres de TODAS las columnas (originales + OHE)
NUMERIC_COLUMNS_REF = X_train.columns 

print("✅ División train/test y One-Hot Encoding completados.")
# ==========================================================
# El set de entrenamiento listo para Imputación/Escalado/PCA
X_train_imput = X_train # DataFrame completo de entrenamiento (OHE + Numéricas)
X_test_imput = X_test # DataFrame completo de prueba (OHE + Numéricas)


# --- IMPUTACIÓN ---

imputer = SimpleImputer(strategy='mean')
try:
    # Anula la validación de features de Pandas para la API
    imputer = imputer.set_output(transform="default") 
except:
    pass 

X_train_imputed_array = imputer.fit_transform(X_train_imput)
joblib.dump(imputer, 'artifacts/imputer_fitted.pkl')

# Convertir a DataFrame después de imputar (Input para Scaler)
# Esto asegura que el Scaler reciba la variable esperada y mantenga los nombres de columna.
X_train_imputed = pd.DataFrame(
    X_train_imputed_array, 
    columns=train_cols, 
    index=X_train.index
)
print(f"✅ Imputación de NaNs completada. {X_train_imputed.shape[1]} columnas imputadas.")

# --- ESCALAMIENTO ---
scaler = StandardScaler()
try:
    scaler = scaler.set_output(transform="default")
except:
    pass

X_train_scaled_array = scaler.fit_transform(X_train_imputed)
joblib.dump(scaler, 'artifacts/scaler_fitted.pkl')

# Convertir la salida del Scaler a DataFrame (Input para PCA)
X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=NUMERIC_COLUMNS_REF, index=X_train.index)
print("✅ Escalado completado.")


# --- PCA Y REDUCCIÓN DE DIMENSIÓN ---
# X_train_pca_ready es el DataFrame escalado (X_train_scaled)

pca = PCA()
pca.fit(X_train_scaled) # Ajustar PCA CON nombres (resuelve UserWarning 2)

# Cálculo de la varianza acumulada
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

pca_final = PCA(n_components=n_components_90)
pca_final.fit(X_train_scaled) # Ajustar PCA final CON nombres
joblib.dump(pca_final, 'artifacts/pca_fitted.pkl')

# Transformar entrenamiento
pca_cols = [f'PC_{i+1}' for i in range(n_components_90)]
df_pca_final = pd.DataFrame(pca_final.transform(X_train_scaled), columns=pca_cols, index=X_train.index)
print(f"✅ PCA completado. Dimensionalidad reducida a {n_components_90} componentes.")


# --- TRANSFORMAR EL CONJUNTO DE PRUEBA (Flujo Completo) ---
# Aplicar SOLO transform() de cada artifact ajustado.

# Imputar (salida array)
X_test_imputed_array = imputer.transform(X_test)

# Convertir a DataFrame (para mantener nombres)
X_test_imputed_df = pd.DataFrame(X_test_imputed_array, columns=NUMERIC_COLUMNS_REF, index=X_test.index)

# Escalar (salida array)
X_test_scaled_array = scaler.transform(X_test_imputed_df) 

# Convertir a DataFrame (para PCA)
X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=NUMERIC_COLUMNS_REF, index=X_test.index)

# Transformar PCA (salida array)
df_pca_test_array = pca_final.transform(X_test_scaled)

# Crear DataFrame de prueba final
df_pca_test_final = pd.DataFrame(df_pca_test_array, columns=pca_cols, index=X_test.index)
# ==========================================================
# --- K-MEANS (Asignar Cluster) ---
K_optimo = 5 
final_kmeans = KMeans(n_clusters=K_optimo, random_state=42, n_init=10)

# Ajustar y predecir en Entrenamiento (df_pca_final)
train_cluster_labels = final_kmeans.fit_predict(df_pca_final) 
# Predecir en Prueba (usando el modelo ajustado)
test_cluster_labels = final_kmeans.predict(df_pca_test_final) 

joblib.dump(final_kmeans, 'artifacts/kmeans_model.pkl')
print("✅ Modelo K-Means guardado.")


# --- ISOLATION FOREST (Asignar Anomalía) ---
contamination_rate = 0.01

if_model = IsolationForest(
    n_estimators=100, 
    contamination=contamination_rate, 
    random_state=42, 
    n_jobs=-1
)

# Ajustar y predecir en Entrenamiento (df_pca_final)
train_outlier_labels = if_model.fit_predict(df_pca_final) 
# Predecir en Prueba (usando el modelo ajustado)
test_outlier_labels = if_model.predict(df_pca_test_final) 

joblib.dump(if_model, 'artifacts/isolation_forest_model.pkl')
print("✅ Modelo Isolation Forest guardado.")


# --- CREAR DATASETS FINALES PARA MODELADO ---

# Añadir las etiquetas a los conjuntos transformados por PCA
df_pca_final['KMEANS_CLUSTER'] = train_cluster_labels
df_pca_final['ISOLATION_OUTLIER'] = train_outlier_labels

df_pca_test_final['KMEANS_CLUSTER'] = test_cluster_labels
df_pca_test_final['ISOLATION_OUTLIER'] = test_outlier_labels


# --- GUARDAR DATASETS FINALES (CON PCA + CLUSTERS) ---

# Entrenamiento: Incluye X_train y Y_train (añadir Y_train de vuelta)
df_pca_final['TARGET'] = Y_train
df_pca_final.to_parquet("artifacts/df_entrenamiento_final.parquet", index=True) 

# Prueba: Incluye X_test y Y_test
df_pca_test_final['TARGET'] = Y_test
df_pca_test_final.to_parquet("artifacts/df_prueba_final.parquet", index=True)

print("✅ Scripts de Preprocesamiento completados. Listo para la Fase 03.")
# ==========================================================