import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Crear carpeta para guardar los archivos si no existe
os.makedirs('model', exist_ok=True)

# Cargar los datos
df = pd.read_excel('C:/Users/apena/proyecto_AI/src/model/data_V2.xlsx')

# Separar variables independientes y la variable objetivo
X = df.drop(['Nombre_CIE10', 'duracion_horas'], axis=1)
Y = df['duracion_horas']

# Codificación ordinal para CIE10
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoded = encoder.fit_transform(X[['CIE10']])
df_encoded = pd.DataFrame(encoded, columns=['CIE10_codificado'])

# Concatenar con variables numéricas
X_final = pd.concat([
    df_encoded,
    X[['Sexo', 'Edad', 'Tipo_de_atencion']].reset_index(drop=True)
], axis=1)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_final, Y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = RandomForestRegressor(max_leaf_nodes=3, max_depth=10)
modelo.fit(X_train, y_train)

# Guardar el modelo y el codificador
joblib.dump(modelo, 'model/modelo_entrenado.pkl')
joblib.dump(encoder, 'model/vector_entrenado.pkl')

# Evaluar y mostrar el error
y_predict = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_predict)
print("Modelo y codificador guardados correctamente.")
print("Mean Squared Error:", mse)
