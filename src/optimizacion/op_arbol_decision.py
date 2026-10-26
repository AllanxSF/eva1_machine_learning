
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# Cargar datos
base_dir = Path(__file__).resolve().parents[2]
df = pd.read_csv(base_dir / "data" / "Spotify_Tracks_Limpio.csv")

# Crear categoria de popularidad
df["popularity_level"] = pd.cut(
    df["popularity"],
    bins=[-1, 33, 66, 100],
    labels=["Baja", "Media", "Alta"]
)

# Definir variables
numericas = [
    "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature"
]
categoricas = ["track_genre"]

df = df.dropna(subset=["popularity_level"])
x = df[numericas + categoricas]
y = df["popularity_level"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocesamiento y modelo
preprocesador = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocesamiento", preprocesador),
    ("clasificador", DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    ))
])

# Parametros para optimizar
parametros_grid = {
    "clasificador__criterion": ["gini", "entropy"],
    "clasificador__max_depth": [3, 4, 5, 8, 10],
    "clasificador__min_samples_split": [2, 10, 20],
    "clasificador__min_samples_leaf": [1, 4, 8]
}

parametros_random = {
    "clasificador__criterion": ["gini", "entropy", "log_loss"],
    "clasificador__max_depth": randint(3, 16),
    "clasificador__min_samples_split": randint(2, 25),
    "clasificador__min_samples_leaf": randint(1, 12)
}

# Ejecutar las busquedas
grid = GridSearchCV(
    pipeline, parametros_grid, cv=5, n_jobs=-1, scoring="f1_macro"
)

random = RandomizedSearchCV(
    pipeline, parametros_random, n_iter=15, cv=5,
    n_jobs=-1, scoring="f1_macro", random_state=42
)

print("Optimizando con GridSearchCV...")
grid.fit(x_train, y_train)

print("Optimizando con RandomizedSearchCV...")
random.fit(x_train, y_train)

# Elegir el mejor modelo por validacion cruzada
mejor = grid if grid.best_score_ >= random.best_score_ else random
modelo = mejor.best_estimator_

# Evaluar en datos de prueba
predicciones = modelo.predict(x_test)

print("\nMejor metodo:", mejor.__class__.__name__)
print("Mejor F1 macro en validacion:", round(mejor.best_score_, 4))
print("Mejores parametros:", mejor.best_params_)
print("Accuracy en prueba:", round(accuracy_score(y_test, predicciones), 4))
print("F1 macro en prueba:", round(f1_score(y_test, predicciones, average="macro"), 4))

print("\nReporte de clasificacion:")
print(classification_report(y_test, predicciones, zero_division=0))

print("\nMatriz de confusion:")
print(confusion_matrix(
    y_test, predicciones, labels=["Baja", "Media", "Alta"]
))

# Guardar modelo optimizado
ruta_modelo = base_dir / "models" / "arbol_decision_optimizado.pkl"
ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, ruta_modelo)

print("\nModelo guardado en:", ruta_modelo)
