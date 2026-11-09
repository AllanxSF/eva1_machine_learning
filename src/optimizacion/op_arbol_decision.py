import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir / "src"))

from preparacion_arbol import (  # noqa: E402
    CATEGORICAS, COL_GRUPO, HISTORIAL, NUMERICAS, cargar_datos, dividir,
)
from transformadores import HistorialPopularidad  # noqa: E402


# Cargar datos y dividir (split agrupado por track_id, sin fuga entre copias)
df = cargar_datos(base_dir)
x_train, x_test, y_train, y_test = dividir(df)
print("Entrenamiento:", len(x_train), "| Prueba:", len(x_test))


# Preprocesamiento: genero (OHE) + numericas + historial de artista/album
preprocesador = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ("historial", HistorialPopularidad(columnas=tuple(HISTORIAL), col_grupo=COL_GRUPO), HISTORIAL + [COL_GRUPO]),
        ("num", "passthrough", NUMERICAS),
    ]
)

pipeline = Pipeline([
    ("preprocesamiento", preprocesador),
    ("clasificador", DecisionTreeClassifier(random_state=42)),
])


# Parametros a optimizar (class_weight ahora forma parte de la busqueda)
parametros_grid = {
    "clasificador__criterion": ["gini", "entropy"],
    "clasificador__max_depth": [8, 10, 12],
    "clasificador__min_samples_split": [2],
    "clasificador__min_samples_leaf": [5, 20],
    "clasificador__class_weight": [None, "balanced"],
}

parametros_random = {
    "clasificador__criterion": ["gini", "entropy", "log_loss"],
    "clasificador__max_depth": randint(5, 16),
    "clasificador__min_samples_split": randint(2, 40),
    "clasificador__min_samples_leaf": randint(3, 30),
    "clasificador__class_weight": [None, "balanced"],
}

# Validacion cruzada agrupada por cancion; se selecciona por accuracy
# y se registra tambien f1_macro para vigilar el desempeno por clase.
cv = GroupKFold(n_splits=3)
scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

grid = GridSearchCV(pipeline, parametros_grid, cv=cv, n_jobs=-1,
                    scoring=scoring, refit="accuracy")
random = RandomizedSearchCV(pipeline, parametros_random, n_iter=10, cv=cv,
                            n_jobs=-1, scoring=scoring, refit="accuracy",
                            random_state=42)

groups = x_train[COL_GRUPO]
print("Optimizando con GridSearchCV...")
grid.fit(x_train, y_train, groups=groups)

print("Optimizando con RandomizedSearchCV...")
random.fit(x_train, y_train, groups=groups)

mejor = grid if grid.best_score_ >= random.best_score_ else random
modelo = mejor.best_estimator_

predicciones = modelo.predict(x_test)

print("\nMejor metodo:", mejor.__class__.__name__)
print("Mejor accuracy en validacion cruzada:", round(mejor.best_score_, 4))
print("Mejores parametros:", mejor.best_params_)
print("Accuracy en prueba:", round(accuracy_score(y_test, predicciones), 4))
print("F1 macro en prueba:", round(f1_score(y_test, predicciones, average="macro"), 4))

print("\nReporte de clasificacion:")
print(classification_report(y_test, predicciones, zero_division=0))

print("\nMatriz de confusion (filas reales Baja/Media/Alta):")
print(confusion_matrix(y_test, predicciones, labels=["Baja", "Media", "Alta"]))

ruta_modelo = base_dir / "models" / "arbol_decision_optimizado.pkl"
ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(modelo, ruta_modelo)
print("\nModelo guardado en:", ruta_modelo)
