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

#Se especifica la ruta
file_application = "../ML_EX_RH/datos_examen/application_.parquet"
file_bureau = "../ML_EX_RH/datos_examen/bureau.parquet"
file_bureau_balance = "../ML_EX_RH/datos_examen/bureau_balance.parquet"
file_previous_application = "../ML_EX_RH/datos_examen/previous_application.parquet"

# Se cargan los DataFrames
df_application = pd.read_parquet(file_application)
df_bureau = pd.read_parquet(file_bureau)
df_bureau_balance = pd.read_parquet(file_bureau_balance)
df_previous_application = pd.read_parquet(file_previous_application)

print("¡Archivos cargados con éxito!")
print(f"application_ rows: {len(df_application)}")
print(f"bureau rows: {len(df_bureau)}")
print(f"bureau_balance rows: {len(df_bureau_balance)}")
print(f"previous_application rows: {len(df_previous_application)}")

#Se une el bureau_balance al bureau
# Agregación de bureau_balance: Conteo de STATUS (categórica)
df_bureau_balance_status = pd.get_dummies(df_bureau_balance, columns=['STATUS'], prefix='STATUS_BB')
bureau_balance_counts = df_bureau_balance_status.groupby('SK_ID_BUREAU').sum()

# Agregación de meses (numérica)
bureau_balance_counts['MONTHS_COUNT'] = df_bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].count()

# Unión de bureau_balance al df_bureau
df_bureau_merged = df_bureau.merge(
    bureau_balance_counts,
    on='SK_ID_BUREAU',
    how='left' # Mantenemos todos los créditos de 'bureau'
)

#Union del "bureau unido" al application_
# --- Resumen de Variables Numéricas de Bureau ---
numerical_cols = df_bureau_merged.select_dtypes(include=np.number).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != 'SK_ID_CURR' and col != 'SK_ID_BUREAU']

# Definir funciones de agregación
agg_funcs = ['mean', 'sum', 'min', 'max', 'count']

# Agrupar y agregar
df_bureau_num_agg = df_bureau_merged.groupby('SK_ID_CURR').agg({
    col: agg_funcs for col in numerical_cols
})

# Renombrar columnas
df_bureau_num_agg.columns = [f'BUREAU_{col[0]}_{col[1].upper()}' for col in df_bureau_num_agg.columns]
df_bureau_num_agg = df_bureau_num_agg.reset_index()

# --- Resumen de Variables Categóricas de Bureau ---
categorical_cols = df_bureau_merged.select_dtypes(include='object').columns.tolist()

df_bureau_cat_agg = pd.DataFrame({'SK_ID_CURR': df_bureau_merged['SK_ID_CURR'].unique()})

for col in categorical_cols:
    # Aplicar One-Hot Encoding
    dummies = pd.get_dummies(df_bureau_merged[['SK_ID_CURR', col]], columns=[col], prefix=f'BUREAU_CAT_{col}')
    
    # Agrupar y sumar (para contar las ocurrencias por cliente)
    dummies_agg = dummies.groupby('SK_ID_CURR').sum().reset_index()
    
    # Unir al DataFrame de categorías
    df_bureau_cat_agg = df_bureau_cat_agg.merge(dummies_agg, on='SK_ID_CURR', how='left')
    
# --- Unión de las Agregaciones ---
df_bureau_final = df_bureau_num_agg.merge(
    df_bureau_cat_agg,
    on='SK_ID_CURR',
    how='outer'
)

# --- Unión Final al DataFrame Principal (application_) ---
df_application_merged = df_application.merge(
    df_bureau_final,
    on='SK_ID_CURR',
    how='left'
)

#Union del application_ al previous_application
# --- Separación de Variables Numéricas y Categóricas ---
df_prev_agg = df_previous_application.copy()
df_prev_agg = df_prev_agg.drop(columns=['SK_ID_PREV']) # Eliminar ID de solicitud previa

# --- Agregación de Variables Numéricas (df_previous_application) ---
numerical_cols = df_prev_agg.select_dtypes(include=np.number).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != 'SK_ID_CURR']

agg_funcs = ['mean', 'sum', 'min', 'max', 'count']
agg_dict_num = {col: agg_funcs for col in numerical_cols}

df_prev_num_agg = df_prev_agg.groupby('SK_ID_CURR').agg(agg_dict_num)

# Renombrar columnas con prefijo PREV_
df_prev_num_agg.columns = [f'PREV_{col[0]}_{col[1].upper()}' for col in df_prev_num_agg.columns]
df_prev_num_agg = df_prev_num_agg.reset_index()

# --- Agregación de Variables Categóricas (df_previous_application) ---
categorical_cols = df_prev_agg.select_dtypes(include='object').columns.tolist()

df_prev_cat_agg = pd.DataFrame({'SK_ID_CURR': df_prev_agg['SK_ID_CURR'].unique()})

for col in categorical_cols:
    # Aplicar One-Hot Encoding
    dummies = pd.get_dummies(df_prev_agg[['SK_ID_CURR', col]], columns=[col], prefix=f'PREV_CAT_{col}')
    
    # Agrupar y sumar
    dummies_agg = dummies.groupby('SK_ID_CURR').sum().reset_index()
    
    # Unir
    df_prev_cat_agg = df_prev_cat_agg.merge(dummies_agg, on='SK_ID_CURR', how='left')


# --- Definición de df_prev_final ---
# Unión Final de Agregaciones Numéricas + Categóricas
df_prev_final = df_prev_num_agg.merge(
    df_prev_cat_agg,
    on='SK_ID_CURR',
    how='outer'
)

# --- Unión al DataFrame principal (application_merged del paso 2) ---
df_final_merged = df_application_merged.merge(
    df_prev_final,
    on='SK_ID_CURR',
    how='left'
)

# Guardar el resultado final
df_final_merged.to_parquet("df_merged_initial.parquet", index=False)

# --- 2. PREPARACIÓN NECESARIA PARA ANÁLISIS DE PCA (SIN PARTITION: ¡SOLO ANÁLISIS!) ---
# Para el análisis EDA/PCA, es aceptable usar el dataset completo, pues no estamos ajustando un modelo supervisado.
# La partición se hará en la Fase 02.

df_pca = df_final_merged.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
df_pca_numeric = df_pca.select_dtypes(include=np.number)

# Imputación
imputer = SimpleImputer(strategy='mean')
df_pca_imputed = imputer.fit_transform(df_pca_numeric)
df_pca_imputed = pd.DataFrame(df_pca_imputed, columns=df_pca_numeric.columns)

# Escalamiento
scaler = StandardScaler()
df_pca_scaled = scaler.fit_transform(df_pca_imputed)
df_pca_scaled = pd.DataFrame(df_pca_scaled, columns=df_pca_imputed.columns)
print("Datos listos para PCA (Análisis de Varianza).")

# --- 3. APLICACIÓN DE PCA PARA VISUALIZACIÓN Y JUSTIFICACIÓN ---
pca = PCA()
pca.fit(df_pca_scaled)

# Calcular la Varianza Acumulada
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

# Mostrar la varianza explicada (Tu código original)
print("\n--- Varianza explicada por los primeros 10 Componentes (PC) ---")
variance_explained = pca.explained_variance_ratio_[:10]
for i, var in enumerate(variance_explained):
    print(f"PC_{i+1}: {var:.4f}")

# Graficar la Varianza Acumulada (Tu código original) 
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% de Varianza Explicada')
plt.axvline(x=n_components_90, color='g', linestyle='--', label=f'{n_components_90} Componentes')
plt.title('Varianza Acumulada Explicada por los Componentes Principales')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada Explicada')
plt.legend()
plt.grid(True)
plt.show()

# --- 4. SALIDA PARA LA FASE 02 ---

# Guardar el dataset fusionado (df_final_merged) para que la Fase 02 pueda dividirlo y transformarlo.
print("\n EDA y Análisis de Dimensionalidad completados.")
df_final_merged.to_parquet("df_merged_initial.parquet", index=False) 
print("Dataset de Features inicial guardado para la fase de Preprocesamiento (02).")