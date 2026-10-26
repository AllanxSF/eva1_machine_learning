
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


# Definir las rutas del proyecto
base_dir = Path(__file__).resolve().parents[2]
ruta_datos = base_dir / "data" / "Spotify_Tracks_Limpio.csv"
ruta_modelo = base_dir / "models" / "arbol_decision.pkl"


# Cargar el dataset limpio
df = pd.read_csv(ruta_datos, encoding="utf-8")
df = df.copy()


# Crear la variable objetivo categorica
df["popularity_level"] = pd.cut(
    df["popularity"],
    bins=[-1, 33, 66, 100],
    labels=["Baja", "Media", "Alta"]
)

df = df.dropna(subset=["popularity_level"])


# Definir las variables del modelo
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
columnas_modelo = columnas_numericas + columnas_categoricas

x = df[columnas_modelo]
y = df["popularity_level"]


# Dividir los datos igual que en el entrenamiento
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Cargar el modelo entrenado y realizar predicciones
modelo = joblib.load(ruta_modelo)
y_pred = modelo.predict(x_test)


# Calcular las metricas
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Baja", "Media", "Alta"]
)

report_dict = classification_report(
    y_test,
    y_pred,
    labels=["Baja", "Media", "Alta"],
    output_dict=True,
    zero_division=0
)

df_report = pd.DataFrame(report_dict).transpose()


# Guardar los resultados en un diccionario
metrics = {
    "accuracy": accuracy,
    "confusion_matrix": cm,
    "classification_report": report_dict,
}


# Mostrar los resultados
print("\nAccuracy:", round(accuracy, 4))

print("\nReporte de clasificacion:")
print(df_report.round(4))

print("\nMatriz de confusion:")
print(pd.DataFrame(
    cm,
    index=["Real Baja", "Real Media", "Real Alta"],
    columns=["Predicha Baja", "Predicha Media", "Predicha Alta"]
))
