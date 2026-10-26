
# Preparacion del entorno
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


# Definir la ruta principal del proyecto
base_dir = Path(__file__).resolve().parent.parent.parent

# Cargar el dataset limpio
ruta = (
    base_dir
    / "data"
    / "Spotify_Tracks_Limpio.csv"
)

df = pd.read_csv(
    ruta,
    encoding="utf-8"
)

# Copiamos el dataframe para no modificar el original
df = df.copy()

print("Dimensiones del dataset:", df.shape)


# Preparacion de los datos
# Variables predictoras: caracteristicas musicales y genero
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

# Variable objetivo: popularidad numerica
y = df["popularity"]


# Preprocesamiento de la variable categorica
preprocesador = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            columnas_categoricas
        )
    ],
    remainder="passthrough"
)


# Crear el modelo de regresion lineal
modelo = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("regresor", LinearRegression())
])


# Dividir los datos para entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# Entrenar el modelo
modelo.fit(x_train, y_train)


# Guardar el modelo entrenado
modelo_dir = base_dir / "models"
modelo_dir.mkdir(parents=True, exist_ok=True)

ruta_modelo = modelo_dir / "regresion_lineal.pkl"

joblib.dump(
    modelo,
    ruta_modelo
)


# Mostrar resultados del entrenamiento
print("Modelo de regresion lineal entrenado correctamente.")
print("Registros de entrenamiento:", len(x_train))
print("Registros de prueba:", len(x_test))
print("Modelo guardado en:", ruta_modelo)
