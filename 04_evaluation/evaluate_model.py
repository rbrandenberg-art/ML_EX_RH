import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

# --- Configuración de Rutas ---
# Asumiendo que el script se ejecuta desde /04_evaluation/
ARTIFACTS_PATH = 'artifacts/'
MODEL_PATH = ARTIFACTS_PATH + 'final_model.pkl'
TEST_DATA_PATH = ARTIFACTS_PATH + 'df_prueba_final.parquet'
PLOTS_PATH = 'plots/'

# Crear carpeta 'plots' si no existe
os.makedirs(PLOTS_PATH, exist_ok=True)
print(f"✅ Carpeta de plots creada en: {PLOTS_PATH}")

# --- Carga de Artifacts y Datos ---
try:
    # Cargar el modelo final entrenado
    final_lgbm = joblib.load(MODEL_PATH)
    print(f"✅ Modelo LightGBM cargado desde: {MODEL_PATH}")

    # Cargar el conjunto de prueba (preprocesado en fase 02)
    df_test = pd.read_parquet(TEST_DATA_PATH)
    print(f"✅ Datos de prueba cargados desde: {TEST_DATA_PATH}")

except FileNotFoundError as e:
    print(f"Error al cargar archivos: {e}")
    exit()

# Separar features (X) y target (Y)
X_test = df_test.drop(columns=['TARGET'])
Y_test = df_test['TARGET']
# ==========================================================

# Obtener los nombres de las features usadas durante el entrenamiento
# Se accede al modelo interno (booster_) para obtener los nombres grabados.
try:
    model_features = list(final_lgbm.booster_.feature_name())
    N_model = len(model_features)
    N_test = len(X_test.columns)

    print("\n==============================================")
    print("🔬 VERIFICACIÓN DE INCONSISTENCIA DE FEATURES")
    print(f"Número de Features usadas para entrenar el modelo (N): {N_model}")
    print(f"Número de Features en el conjunto de prueba (X_test): {N_test}")
    print("==============================================")

    if N_model != N_test:
        print("❌ ALERTA: EL NÚMERO DE COLUMNAS NO COINCIDE.")
        print("Esto causará fallos en producción si no se usa .values.")
    else:
        print("✅ EL NÚMERO DE FEATURES COINCIDE. El flujo de datos es consistente.")

except AttributeError:
    # Si el modelo no tiene el atributo booster_ (versión muy antigua o pipeline)
    print("⚠️ No se pudo verificar el número de features del modelo (Atributo no encontrado).")
    print(f"Simplemente verificando el set de prueba: {len(X_test.columns)} columnas.")

# ==========================================================
# Identificar las columnas que fueron numéricas/clusters/outliers
# Si tu dataset es solo PCs, Clusters y Outliers:
cluster_cols = ['KMEANS_CLUSTER', 'ISOLATION_OUTLIER']
pc_cols = [col for col in X_test.columns if col.startswith('PC_')]

# Forzar las columnas de clusters/outliers a tipo float para evitar la inferencia categórica
for col in cluster_cols:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype(np.float32)

# Asegurar que las PCs sean float
for col in pc_cols:
    X_test[col] = X_test[col].astype(np.float32)

# --- Predicción de Probabilidades ---
# Predict_proba devuelve la probabilidad de clase 0 y clase 1
# La predicción ahora se realiza sobre un DataFrame limpio de tipos que LightGBM confunde.
X_test_array = X_test.values

Y_pred_proba = final_lgbm.predict_proba(X_test_array)[:, 1]

# --- Cálculo de Métricas ---

# AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
# Métrica principal: mide la capacidad de discriminación del modelo.
roc_auc = roc_auc_score(Y_test, Y_pred_proba)

# Coeficiente de Gini
# Es la métrica estándar de la industria crediticia: Gini = 2 * AUC - 1
gini_coefficient = 2 * roc_auc - 1

print("\n==============================================")
print("📊 RESULTADOS DE LA EVALUACIÓN EN CONJUNTO DE PRUEBA")
print("==============================================")
print(f"➡️ Área bajo la curva ROC (AUC-ROC): {roc_auc:.4f}")
print(f"➡️ Coeficiente Gini: {gini_coefficient:.4f}")
print("==============================================")

if gini_coefficient >= 0.6:
    print("👍 El modelo muestra un poder discriminatorio (Gini > 0.6) considerado bueno en la industria.")
elif gini_coefficient >= 0.5:
    print("⚠️ El modelo tiene un poder discriminatorio marginal (Gini > 0.5).")
else:
    print("❌ El modelo no discrimina mejor que la aleatoriedad.")

# --- Generación de Curva ROC ---
def plot_roc_curve(Y_test, Y_pred_proba, roc_auc, path):
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC del Modelo LightGBM')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(path + 'curva_roc.png')
    plt.close()
    print(f"✅ Curva ROC guardada en: {path}curva_roc.png")

# --- Generación de Curva Precision-Recall ---
def plot_precision_recall_curve(Y_test, Y_pred_proba, path):
    # La función precision_recall_curve ya fue importada al inicio.
    precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', linestyle='-', color='darkgreen')
    
    # La línea base para la P-R Curve es la proporción de positivos
    proportion_positives = Y_test.sum() / len(Y_test)
    plt.axhline(y=proportion_positives, color='r', linestyle='--', label=f'Línea Base (Ratio Positivos: {proportion_positives:.2f})')
    
    plt.xlabel('Recall (Sensibilidad o Tasa de Verdaderos Positivos)')
    plt.ylabel('Precision (Valor Predictivo Positivo)')
    plt.title('Curva Precision-Recall para la Clase de Default (TARGET=1)')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.savefig(path + 'curva_precision_recall.png')
    plt.close()
    print(f"✅ Curva Precision-Recall guardada en: {path}curva_precision_recall.png")

plot_roc_curve(Y_test, Y_pred_proba, roc_auc, PLOTS_PATH)

# --- Generación de Curva de Ganancia (Lift Chart) ---
def plot_lift_chart(Y_test, Y_pred_proba, path):
    df_results = pd.DataFrame({'Target': Y_test, 'Score': Y_pred_proba})
    # Ordenar por score descendente
    df_results = df_results.sort_values(by='Score', ascending=False).reset_index(drop=True)
    
    total_defaults = df_results['Target'].sum()
    total_clients = len(df_results)

    # Crear deciles y calcular la ganancia (lift)
    df_results['Decil'] = pd.qcut(df_results.index, q=10, labels=False, duplicates='drop')
    
    decile_summary = df_results.groupby('Decil').agg(
        num_clientes=('Target', 'count'),
        num_defaults=('Target', 'sum')
    ).reset_index()

    decile_summary['cumulative_defaults'] = decile_summary['num_defaults'].cumsum()
    decile_summary['cumulative_pct_defaults'] = decile_summary['cumulative_defaults'] / total_defaults
    decile_summary['cumulative_pct_clients'] = (decile_summary.index + 1) / len(decile_summary)
    
    # Lift = Ganancia / Línea base (porcentaje de clientes)
    decile_summary['Lift'] = decile_summary['cumulative_pct_defaults'] / decile_summary['cumulative_pct_clients']

    plt.figure(figsize=(8, 6))
    plt.plot(decile_summary['cumulative_pct_clients'], decile_summary['cumulative_pct_defaults'], marker='o', label='Modelo')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Línea Base (Aleatorio)')
    plt.xlabel('Porcentaje Acumulado de Clientes')
    plt.ylabel('Porcentaje Acumulado de Defaults Capturados')
    plt.title('Curva de Ganancia (Lift Chart)')
    plt.grid(True)
    plt.legend()
    plt.savefig(path + 'curva_ganancia.png')
    plt.close()
    print(f"✅ Curva de Ganancia guardada en: {path}curva_ganancia.png")

plot_roc_curve(Y_test, Y_pred_proba, roc_auc, PLOTS_PATH)

# AÑADIR LA LLAMADA AQUÍ:
plot_precision_recall_curve(Y_test, Y_pred_proba, PLOTS_PATH)

# --- Generación de Curva de Ganancia (Lift Chart) ---
def plot_lift_chart(Y_test, Y_pred_proba, path):
    plot_lift_chart(Y_test, Y_pred_proba, PLOTS_PATH)

print("\n--- Evaluación Finalizada ---")