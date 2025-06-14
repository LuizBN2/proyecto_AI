import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Cargar los datos
df = pd.read_excel('data_V2.xlsx')

# Separar variables independientes y la variable objetivo
X = df.drop(['Nombre_CIE10', 'duracion_horas'], axis=1)
Y = df['duracion_horas']

# Codificación one-hot
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(X[['CIE10']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['CIE10']))

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_encoded, Y, test_size=0.2, random_state=42)

# Entrenar el modelo
modelo = RandomForestRegressor(max_leaf_nodes=3, max_depth=10)
modelo.fit(X_train, y_train)

# Predicción
y_predict = modelo.predict(X_test)

# Métrica MSE
mse = mean_squared_error(y_test, y_predict)
joblib.dump(modelo, 'modelo_regresion_lineal.pkl')
print("Mean Squared Error:", mse)
