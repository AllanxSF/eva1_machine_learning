
# Preparacion del entorno
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# Definir las rutas del proyecto
base_dir = Path(__file__).resolve().parents[2]
ruta = base_dir / "data" / "Spotify_Tracks_Limpio.csv"

# Cargar el dataset limpio
df = pd.read_csv(ruta, encoding="utf-8")

# Copiamos el dataframe para no modificar el original
df = df.copy()


# Crear la variable objetivo categorica
df["popularity_level"] = pd.cut(
    df["popularity"],
    bins=[-1, 33, 66, 100],
    labels=["Baja", "Media", "Alta"]
)

# Eliminar registros sin categoria asignada
df = df.dropna(subset=["popularity_level"])


# Definir las variables predictoras
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


# Preprocesamiento de las variables categoricas
preprocesador = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            columnas_categoricas
        ),
    ],
    remainder="passthrough",
)


# Crear el modelo de arbol de decision
modelo = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    (
        "clasificador",
        DecisionTreeClassifier(
            random_state=42,
            max_depth=4,
            class_weight="balanced"
        )
    ),
])


# Dividir los datos para entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Entrenar el modelo
modelo.fit(x_train, y_train)


# Guardar el modelo entrenado
modelo_dir = base_dir / "models"
modelo_dir.mkdir(parents=True, exist_ok=True)

ruta_modelo = modelo_dir / "arbol_decision.pkl"
joblib.dump(modelo, ruta_modelo)


# Mostrar resultados
print("Modelo de arbol de decision entrenado correctamente.")
print("Registros de entrenamiento:", len(x_train))
print("Registros de prueba:", len(x_test))
print("Distribucion de categorias:")
print(y.value_counts())
print("Modelo guardado en:", ruta_modelo)
