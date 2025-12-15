import joblib
import os
import pandas as pd

ARTIFACTS_PATH = 'artifacts/'
MODEL_PATH = ARTIFACTS_PATH + 'final_model.pkl'

try:
    final_lgbm = joblib.load(MODEL_PATH)
    # Intenta obtener los nombres de features que el modelo usó para ajustarse
    model_features = final_lgbm.booster_.feature_name()
    print(f"Número de features grabadas en el modelo: {len(model_features)}")
    print(f"Features grabadas: {model_features[-5:]}")
    
    # Si esta lista AÚN contiene nombres de PCs (PC_1, PC_2, etc.), significa que el .values no funcionó.

except Exception as e:
    print(f"No se pudo cargar o diagnosticar el modelo: {e}")