from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
from typing import List

# --- CONFIGURACIÓN Y DECLARACIÓN DE VARIABLES GLOBALES ---

# Usar ruta absoluta para robustez, asumiendo que el script se ejecuta desde /05_deployment
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts')

# Declaración de variables globales (Inicialmente None, se llenan en load_artifacts)
MODEL = None
SCALER = None
PCA = None
IMPUTER = None
KMEANS = None
IFOREST = None
OHE_FEATURE_NAMES = None

# Lista de las features categóricas originales (¡REEMPLAZA ESTO CON TUS COLUMNAS REALES!)
CATEGORICAL_COLS = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
# Lista de las features numéricas originales (¡REEMPLAZA ESTO CON TUS COLUMNAS REALES!)
NUMERIC_COLS = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH'] # EJEMPLO

# --- FUNCIÓN DE CARGA DE ARTIFACTS (CRÍTICA) ---

def load_artifacts():
    """Carga todos los modelos y transformadores al inicio de la API."""
    global MODEL, SCALER, PCA, IMPUTER, KMEANS, IFOREST, OHE_FEATURE_NAMES
    try:
        # Modelos y Transformadores
        MODEL = joblib.load(os.path.join(ARTIFACTS_PATH, 'final_model.pkl'))
        SCALER = joblib.load(os.path.join(ARTIFACTS_PATH, 'scaler_fitted.pkl'))
        PCA = joblib.load(os.path.join(ARTIFACTS_PATH, 'pca_fitted.pkl'))
        IMPUTER = joblib.load(os.path.join(ARTIFACTS_PATH, 'imputer_fitted.pkl'))
        KMEANS = joblib.load(os.path.join(ARTIFACTS_PATH, 'kmeans_model.pkl'))
        IFOREST = joblib.load(os.path.join(ARTIFACTS_PATH, 'isolation_forest_model.pkl'))
        
        # Lista de Features OHE
        OHE_FEATURE_NAMES = joblib.load(os.path.join(ARTIFACTS_PATH, 'ohe_input_features_ref.pkl'))

        print("✅ Todos los modelos y transformadores cargados con éxito.")

    except FileNotFoundError as e:
        print(f"Error al cargar artifacts: {e}. Deteniendo API.")
        raise RuntimeError(f"Fallo al cargar artifacts: {e}")

# Llamar a la función de carga al inicio
try:
    load_artifacts()
except RuntimeError:
    pass

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Scoring Crediticio (MLOps Project)",
    description="Microservicio para predecir la probabilidad de incumplimiento de pago (default).",
)


# --- DEFINICIÓN DEL ESQUEMA DE DATOS (Pydantic) ---

# CRÍTICO: Debes reemplazar este esquema con TODAS las features originales 
# numéricas y categóricas que usaste como entrada a la FASE 02.
class ClientFeatures(BaseModel):
    # EJEMPLO DE FEATURES ORIGINALES: REEMPLAZA ESTO
    AMT_CREDIT: float = Field(200000.0, description="Monto del crédito solicitado.")
    AMT_ANNUITY: float = Field(20000.0, description="Monto de la anualidad.")
    AMT_INCOME_TOTAL: float = Field(450000.0, description="Ingreso total anual.")
    DAYS_BIRTH: float = Field(-10000.0, description="Edad del cliente en días (negativo).")
    NAME_CONTRACT_TYPE: str = Field('Cash loans', description="Tipo de contrato ('Cash loans', 'Revolving loans').")
    CODE_GENDER: str = Field('F', description="Género ('M', 'F', 'XNA').")
    FLAG_OWN_CAR: str = Field('N', description="Dueño de coche ('Y', 'N').")
    FLAG_OWN_REALTY: str = Field('Y', description="Dueño de propiedad ('Y', 'N').")


# --- FUNCIÓN DE PREPROCESAMIENTO COMPLETO ---

def full_preprocessing(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Aplica la tubería completa de preprocesamiento a una nueva fila de datos:
    OHE -> Alineación -> Imputación -> Escalado -> PCA -> K-Means/IForest -> Output Array.
    """
    
    if OHE_FEATURE_NAMES is None:
        raise RuntimeError("No se cargó la lista de nombres de features OHE.")
    
    # --- One-Hot Encoding (OHE) y Alineación ---
    
    # Aplicar OHE solo a las columnas categóricas que definiste globalmente
    df_cat = df_raw[CATEGORICAL_COLS]
    df_ohe = pd.get_dummies(df_cat, drop_first=False)
    
    # Reintegrar features numéricas originales
    df_num = df_raw.drop(columns=CATEGORICAL_COLS, errors='ignore')
    
    # Combinar numéricas y OHE
    df_combined = pd.concat([df_num.reset_index(drop=True), df_ohe.reset_index(drop=True)], axis=1)
    
    # PASO CRÍTICO: ALINEAR (reindex) con los ~615 nombres de features del Train Set
    X_aligned = df_combined.reindex(columns=OHE_FEATURE_NAMES, fill_value=0)

    # ----------------------------------------------------------------------
    # LIMPIEZA FORZADA (Necesaria, ya que el error KMEANS_CLUSTER persiste)
    # ----------------------------------------------------------------------
    if 'KMEANS_CLUSTER' in X_aligned.columns:
        X_aligned = X_aligned.drop(columns=['KMEANS_CLUSTER'], errors='ignore')

    if 'ISOLATION_OUTLIER' in X_aligned.columns:
        X_aligned = X_aligned.drop(columns=['ISOLATION_OUTLIER'], errors='ignore')

    print(f"\n--- DEBUG IMPUTER INPUT ---")
    print(f"Número de columnas antes de Imputer: {len(X_aligned.columns)}")
    print(f"Columna KMEANS_CLUSTER presente?: {'KMEANS_CLUSTER' in X_aligned.columns}")
    # Imprimir las primeras 5 y últimas 5 columnas para verificar:
    print(f"Primeras 5 columnas: {X_aligned.columns[:5].tolist()}")
    print(f"Últimas 5 columnas: {X_aligned.columns[-5:].tolist()}")
    
    # Convertir a array NumPy para DESACTIVAR la validación de nombres de Scikit-learn
    X_aligned_array = X_aligned.values
    
    # --- Imputación y Escalado (Transformación) ---
    X_imputed_array = IMPUTER.transform(X_aligned_array)
    X_scaled_array = SCALER.transform(X_imputed_array)
    
    # --- PCA (Reducción de Dimensionalidad) ---
    X_pca_array = PCA.transform(X_scaled_array)
    N_components = PCA.n_components_
    pca_cols = [f'PC_{i+1}' for i in range(N_components)]

    X_pca_df = pd.DataFrame(X_pca_array, columns=pca_cols, index=df_raw.index)
    
    # --- Clustering y Anomalía (Generación de Features Finales) ---
    
    # La predicción del K-Means e IForest requiere el vector de PCs
    X_pca_df['KMEANS_CLUSTER'] = KMEANS.predict(X_pca_df)
    X_pca_df['ISOLATION_OUTLIER'] = IFOREST.predict(X_pca_df)
    
    # Forzar los tipos a float32 para evitar el error categórico de LightGBM (solución final de la Fase 04)
    X_pca_df['KMEANS_CLUSTER'] = X_pca_df['KMEANS_CLUSTER'].astype(np.float32)
    X_pca_df['ISOLATION_OUTLIER'] = X_pca_df['ISOLATION_OUTLIER'].astype(np.float32)
    
    X_final = X_pca_df.values 

    return X_final

# --- ENDPOINT DE PREDICCIÓN ---

@app.get("/")
def home():
    """Endpoint de bienvenida."""
    return {"message": "API de Scoring Crediticio activa. Use /score para predicciones."}


@app.post("/score")
def predict_risk(client_data: ClientFeatures):
    """
    Recibe los datos de un cliente, aplica el preprocesamiento completo 
    y devuelve la probabilidad de default (TARGET=1).
    """
    if MODEL is None or OHE_FEATURE_NAMES is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado o la configuración de features es ausente.")

    # Convertir el JSON de Pydantic a un DataFrame de Pandas (1 fila)
    data_dict = client_data.dict()
    df_raw = pd.DataFrame([data_dict])

    try:
        # Aplicar la tubería de preprocesamiento completa
        X_final_vector = full_preprocessing(df_raw)
        
        # --- Predicción ---
        prediction_proba = MODEL.predict_proba(X_final_vector,validate_features=False)[0, 1]
        
    except Exception as e:
        # Captura errores que puedan ocurrir durante el preprocesamiento (ej. datos faltantes o malos)
        raise HTTPException(status_code=400, detail=f"Error durante el preprocesamiento/predicción: {e}")

    # --- Generar Respuesta y Decisión ---
    
    # Umbral de riesgo (Ejemplo: 10% de probabilidad de default)
    THRESHOLD = 0.10
    decision = "Rechazado" if prediction_proba >= THRESHOLD else "Aprobado"
    
    return {
        "probability_default": round(prediction_proba, 4),
        "score_percent": f"{round(prediction_proba*100, 2)}%",
        "decision": decision,
        "detail": f"Riesgo de {decision} con probabilidad de default de {round(prediction_proba*100, 2)}%"
    }