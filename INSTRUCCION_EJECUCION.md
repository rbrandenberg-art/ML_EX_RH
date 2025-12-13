# Instrucciones de Ejecución Claras del Código
El código de implementación se encuentra en el notebook adjunto (EA3.ipynb).
# Instalar las librerías necesarias:
pip install pandas numpy scikit-learn pyarrow matplot libscikit-learn joblib lightgbm
# Carga de Datos: 
Asegurarse de que los archivos .parquet estén ubicados en el mismo directorio que el notebook.Ejecución: Ejecutar el notebook de forma secuencial. Los pasos clave son: 
Unión de Tablas: Agregación de bureau, bureau_balance, y previous_application al nivel de SK_ID_CURR.PCA y Escalamiento: Imputación, escalado y reducción de la dimensionalidad (de aproximadamente 900 variables a un poco mas de 150 componentes).
K-Means: Cálculo del K óptimo (Método del Codo) y asignación de etiquetas de cluster.
Isolation Forest: Detección de outliers con tasa de contaminación al 1%.