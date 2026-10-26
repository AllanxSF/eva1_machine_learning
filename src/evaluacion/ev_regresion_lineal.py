
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


# Definir rutas del proyecto
base_dir = Path(__file__).resolve().parents[2]
ruta_datos = base_dir / "data" / "Spotify_Tracks_Limpio.csv"
ruta_modelo = base_dir / "models" / "regresion_lineal.pkl"


# Cargar el dataset limpio
df = pd.read_csv(ruta_datos, encoding="utf-8")


# Preparar las variables
columnas_numericas = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature"
]

columnas_categoricas = ["track_genre"]

x = df[columnas_numericas + columnas_categoricas]
y = df["popularity"]


# Dividir los datos igual que en el entrenamiento
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# Cargar el modelo entrenado
modelo = joblib.load(ruta_modelo)


# Realizar predicciones
y_pred = modelo.predict(x_test)
y_train_pred = modelo.predict(x_train)


# Calcular metricas de prueba
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_test = r2_score(y_test, y_pred)


# Calcular metrica de entrenamiento
r2_train = r2_score(y_train, y_train_pred)


# Crear tabla de resultados
df_report = pd.DataFrame({
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R² (Test)": [r2_test],
    "R² (Train)": [r2_train]
}).T

df_report.columns = ["Regresion Lineal"]

print("\nResultados de evaluacion:")
print(df_report.round(4))


# Guardar resultados para utilizarlos posteriormente
metrics = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "r2_test": r2_test,
    "r2_train": r2_train,
    "y_test": y_test,
    "y_pred": y_pred,
    "y_train": y_train,
    "y_train_pred": y_train_pred
}
