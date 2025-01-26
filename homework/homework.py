#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
import pandas as pd
import gzip
import json
import os
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
train = pd.read_csv("/content/files/input/train_data.csv.zip",index_col=False,compression="zip")
test = pd.read_csv("/content/files/input/test_data.csv.zip",index_col=False,compression="zip")

train['Age'] = 2021 - train['Year']
test['Age'] = 2021 - test['Year']

train.drop(columns=['Year', 'Car_Name'], inplace=True)
test.drop(columns=['Year', 'Car_Name'], inplace=True)

print(train.head())
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train = train.drop(columns=['Selling_Price'])  
y_train = train['Selling_Price']                

x_test = test.drop(columns=['Selling_Price'])   
y_test = test['Selling_Price']                  

# Mostrar un vistazo de las dimensiones de los datasets divididos
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
x_train = train.drop(columns=['Selling_Price'])  
y_train = train['Selling_Price']                

x_test = test.drop(columns=['Selling_Price'])   
y_test = test['Selling_Price']                  

# Mostrar un vistazo de las dimensiones de los datasets divididos
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
param_grid = {
    'feature_selector__k':range(1,15),
    'regressor__fit_intercept':[True,False],
    
}

model=GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,  
    scoring="neg_mean_absolute_error",
    n_jobs=-1,   
)



model.fit(x_train, y_train)

print(f"Mejores hiperparámetros: {model.best_params_}")
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
param_grid = {
    'feature_selector__k':range(1,15),
    'regressor__fit_intercept':[True,False],
    
}

model=GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,  
    scoring="neg_mean_absolute_error",
    n_jobs=-1,   
)



model.fit(x_train, y_train)

print(f"Mejores hiperparámetros: {model.best_params_}")
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import json
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,median_absolute_error

best_model=model.best_estimator_
# Calcular métricas para el conjunto de entrenamiento
y_train_pred = best_model.predict(x_train)
train_metrics = {
    'type': 'metrics',
    'dataset': 'train',
    'r2': r2_score(y_train, y_train_pred),
    'mse': mean_squared_error(y_train, y_train_pred),
    'mad': median_absolute_error(y_train, y_train_pred),
}

# Calcular métricas para el conjunto de prueba
y_test_pred = best_model.predict(x_test)
test_metrics = {
    'type': 'metrics',
    'dataset': 'test',
    'r2': r2_score(y_test, y_test_pred),
    'mse': mean_squared_error(y_test, y_test_pred),
    'mad': median_absolute_error(y_test, y_test_pred),
}

output_path = "/content/files/output/metrics.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)  

with open(output_path, 'w') as f:
    f.write(json.dumps(train_metrics) + '\n')  
    f.write(json.dumps(test_metrics) + '\n')  

print(f"Métricas guardadas en: {output_path}")
