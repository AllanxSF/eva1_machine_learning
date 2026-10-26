
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==================================
# CARGAR DATOS
# ==================================

base_dir = Path(__file__).resolve().parents[2]

ruta_datos = base_dir / "data" / "Spotify_Tracks_Limpio.csv"

df = pd.read_csv(ruta_datos, encoding="utf-8")


# ==================================
# VARIABLES DEL MODELO
# ==================================

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


# ==================================
# DIVIDIR DATOS
# ==================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================
# PREPROCESAMIENTO
# ==================================

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


# ==================================
# PIPELINE RIDGE
# ==================================

pipeline_ridge = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("regresor", Ridge()),
    ]
)


# ==================================
# PIPELINE LASSO
# ==================================

pipeline_lasso = Pipeline(
    steps=[
        ("preprocesamiento", preprocesador),
        ("regresor", Lasso(max_iter=10000)),
    ]
)


# ==================================
# BUSQUEDA DE HIPERPARAMETROS
# ==================================

grid_parametros = {
    "regresor__alpha": [0.01, 0.1, 1, 10, 100]
}

random_parametros = {
    "regresor__alpha": uniform(0.01, 100)
}


grid_search = GridSearchCV(
    pipeline_ridge,
    grid_parametros,
    cv=5,
    n_jobs=-1,
    scoring="r2",
    refit=True,
)

random_search = RandomizedSearchCV(
    pipeline_lasso,
    random_parametros,
    n_iter=15,
    cv=5,
    n_jobs=-1,
    scoring="r2",
    refit=True,
    random_state=42,
)


# ==================================
# ENTRENAR LAS BUSQUEDAS
# ==================================

print("Optimizando Ridge con GridSearchCV...")
grid_search.fit(x_train, y_train)

print("\nOptimizando Lasso con RandomizedSearchCV...")
random_search.fit(x_train, y_train)


# ==================================
# SELECCIONAR EL MEJOR MODELO
# ==================================

mejor_busqueda = (
    grid_search
    if grid_search.best_score_ >= random_search.best_score_
    else random_search
)

modelo_optimizado = mejor_busqueda.best_estimator_


# ==================================
# EVALUAR LOS MODELOS
# ==================================

y_pred_ridge = grid_search.best_estimator_.predict(x_test)
y_pred_lasso = random_search.best_estimator_.predict(x_test)

resumen_busquedas = pd.DataFrame([
    {
        "metodo": "GridSearchCV (Ridge)",
        "mejor_score_cv": grid_search.best_score_,
        "r2_test": r2_score(y_test, y_pred_ridge),
        "rmse_test": np.sqrt(
            mean_squared_error(y_test, y_pred_ridge)
        ),
        "mejor_alpha": grid_search.best_params_["regresor__alpha"],
    },
    {
        "metodo": "RandomizedSearchCV (Lasso)",
        "mejor_score_cv": random_search.best_score_,
        "r2_test": r2_score(y_test, y_pred_lasso),
        "rmse_test": np.sqrt(
            mean_squared_error(y_test, y_pred_lasso)
        ),
        "mejor_alpha": random_search.best_params_["regresor__alpha"],
    },
])


print("\nResumen de las busquedas:")
print(resumen_busquedas.round(4).to_string(index=False))


# ==================================
# GUARDAR RESULTADOS
# ==================================

metrics = {
    "r2": r2_score(
        y_test,
        modelo_optimizado.predict(x_test)
    ),
    "rmse": np.sqrt(
        mean_squared_error(
            y_test,
            modelo_optimizado.predict(x_test)
        )
    ),
    "best_method": mejor_busqueda.__class__.__name__,
    "best_cv_score": mejor_busqueda.best_score_,
    "best_params": mejor_busqueda.best_params_,
}

modelo_dir = base_dir / "models"
modelo_dir.mkdir(parents=True, exist_ok=True)

ruta_modelo = modelo_dir / "regresion_lineal_optimizado.pkl"

joblib.dump(modelo_optimizado, ruta_modelo)

print("\nMejor método:", metrics["best_method"])
print("Mejor score de validación cruzada:", round(metrics["best_cv_score"], 4))
print("Mejores parámetros:", metrics["best_params"])
print("R² en prueba:", round(metrics["r2"], 4))
print("RMSE en prueba:", round(metrics["rmse"], 4))
print("Modelo optimizado guardado en:", ruta_modelo)
