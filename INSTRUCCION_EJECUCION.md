# Instrucciones de Ejecución Claras del Código
El código de implementación se encuentra dividido en 5 modulos de ejecucion distintos mas los agregados necesarios para que funcione:
ML_EX_RH/
|-01_data_understanding/
||-preprocess.py
|-02_data_preparation/
||-preprocess_data.py
|-03_modeling/
||-train_model.py
|-04_evaluatio/
||-evaluate_model.py
|-05_deployment/
||-app.py
||-predictor.py
|-artifacts/
|-gitignore
|-INSTRUCCIONES_EJECUCION.MD
|-README.MD
|-requirements.txt
# Instalar las librerías necesarias:
Copiar el siguiente comando y pegarlo en la rama raiz.
pip install pandas numpy scikit-learn pyarrow matplot joblib lightgbm fastapi uvicorn pydantic
# Carga de Datos: 
Asegurarse de que los archivos .parquet estén ubicados en la carpeta raiz.
Asegurarse de que la consola este ubicada en la carpeta raiz antes de iniciar comandos y ejecuciones.
Ejecución: Ejecutar los archivos .py en orden secuencial(del 01 al 05). 
Ejecutar la API: - cd 05_deployment 
Dentro de la carpeta ejecutar: - uvicorn app:app --reload, - python -m uvicorn app:app --reload
