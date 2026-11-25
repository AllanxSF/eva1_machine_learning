"""Tecnicas de ensamble para clasificar popularidad (Baja / Media / Alta).

Compara, sobre el MISMO split agrupado por track_id y el MISMO preprocesamiento
que el arbol v2:
  - Arbol de decision v2 (referencia, hiperparametros ya optimizados)
  - Random Forest            (bagging)
  - HistGradientBoosting     (boosting)
  - VotingClassifier soft    (arbol v2 + Random Forest + Gradient Boosting)
Los hiperparametros de RF y GB se optimizan con RandomizedSearchCV y
GroupKFold(3), seleccionando por accuracy y registrando f1_macro.
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,RandomForestClassifier, VotingClassifier)
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir / "src"))

from preparacion_arbol import (  # noqa: E402
    CATEGORICAS, COL_GRUPO, HISTORIAL, NUMERICAS, cargar_datos, dividir,
)
from transformadores import HistorialPopularidad  # noqa: E402

ETIQUETAS = ["Baja", "Media", "Alta"]

df = cargar_datos(base_dir)
x_train, x_test, y_train, y_test = dividir(df)
print("Entrenamiento:", len(x_train), "| Prueba:", len(x_test))


def preprocesador():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("historial", HistorialPopularidad(columnas=tuple(HISTORIAL), col_grupo=COL_GRUPO),HISTORIAL + [COL_GRUPO]),
        ("num", "passthrough", NUMERICAS),
    ])


def pipeline(clasificador):
    return Pipeline([("preprocesamiento", preprocesador()), ("clasificador", clasificador)])


cv = GroupKFold(n_splits=3)
scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
groups = x_train[COL_GRUPO]

# ---------------- Bagging: Random Forest ----------------
rf_search = RandomizedSearchCV(
    pipeline(RandomForestClassifier(random_state=42, n_jobs=1)),
    {
        "clasificador__n_estimators": [100],
        "clasificador__max_depth": randint(12, 23),
        "clasificador__min_samples_leaf": randint(3, 16),
        "clasificador__max_features": ["sqrt"],
        "clasificador__class_weight": [None, "balanced_subsample"],
    },
    n_iter=5, cv=cv, scoring=scoring, refit="accuracy", random_state=42, n_jobs=1,
)
t = time.time()
print("Optimizando Random Forest...")
rf_search.fit(x_train, y_train, groups=groups)
print(f"  listo en {time.time() - t:.0f}s | CV accuracy {rf_search.best_score_:.4f} | {rf_search.best_params_}")

# ---------------- Boosting: HistGradientBoosting ----------------
gb_search = RandomizedSearchCV(
    pipeline(HistGradientBoostingClassifier(random_state=42)),
    {
        "clasificador__learning_rate": [0.05, 0.1, 0.2],
        "clasificador__max_iter": [100, 150],
        "clasificador__max_depth": [4, 6, 8, None],
        "clasificador__min_samples_leaf": [20, 50, 100],
        "clasificador__l2_regularization": [0.0, 1.0, 10.0],
    },
    n_iter=8, cv=cv, scoring=scoring, refit="accuracy", random_state=42, n_jobs=1,
)
t = time.time()
print("Optimizando Gradient Boosting...")
gb_search.fit(x_train, y_train, groups=groups)
print(f"  listo en {time.time() - t:.0f}s | CV accuracy {gb_search.best_score_:.4f} | {gb_search.best_params_}")

# ---------------- Arbol v2 de referencia ----------------
arbol = pipeline(DecisionTreeClassifier(
    random_state=42, criterion="gini", max_depth=8, min_samples_leaf=20,
    min_samples_split=2, class_weight=None))

# ---------------- Voting soft: arbol v2 + RF + GB ----------------
def sin_prepro(search):
    return clone(search.best_estimator_.named_steps["clasificador"])

voting = pipeline(VotingClassifier(
    estimators=[("arbol", DecisionTreeClassifier(random_state=42, max_depth=8, min_samples_leaf=20)),
                ("rf", sin_prepro(rf_search)),
                ("gb", sin_prepro(gb_search))],
    voting="soft", n_jobs=1))

modelos = {
    "Arbol de decision v2": arbol,
    "Random Forest (bagging)": rf_search.best_estimator_,
    "Gradient Boosting (boosting)": gb_search.best_estimator_,
    "Voting soft (arbol+RF+GB)": voting,
}
cv_scores = {
    "Arbol de decision v2": 0.8295,
    "Random Forest (bagging)": rf_search.best_score_,
    "Gradient Boosting (boosting)": gb_search.best_score_,
    "Voting soft (arbol+RF+GB)": float("nan"),
}

# ---------------- Evaluacion comparable ----------------
vistos = set(x_train["first_artist"])
mask_nuevo = ~x_test["first_artist"].isin(vistos)
filas, preds = [], {}
for nombre, m in modelos.items():
    if nombre in ("Arbol de decision v2", "Voting soft (arbol+RF+GB)"):
        t = time.time()
        m.fit(x_train, y_train)
        print(f"Entrenado {nombre} en {time.time() - t:.0f}s")
    p = m.predict(x_test)
    preds[nombre] = p
    filas.append({
        "modelo": nombre,
        "cv_accuracy": round(cv_scores[nombre], 4),
        "accuracy": round(accuracy_score(y_test, p), 4),
        "f1_macro": round(f1_score(y_test, p, average="macro"), 4),
        "recall_alta": round(recall_score(y_test, p, labels=["Alta"], average=None)[0], 4),
        "accuracy_artista_nuevo": round(accuracy_score(y_test[mask_nuevo], p[mask_nuevo.values]), 4),
    })

resultados = pd.DataFrame(filas)
print("\nComparacion de modelos (mismo conjunto de prueba agrupado):")
print(resultados.to_string(index=False))

mejor_nombre = resultados.sort_values("accuracy", ascending=False).iloc[0]["modelo"]
print("\nMejor modelo por accuracy en prueba:", mejor_nombre)
p = preds[mejor_nombre]
print(pd.DataFrame(classification_report(y_test, p, labels=ETIQUETAS, output_dict=True,
                                          zero_division=0)).transpose().round(4))
print(pd.DataFrame(confusion_matrix(y_test, p, labels=ETIQUETAS),
                   index=["Real Baja", "Real Media", "Real Alta"],
                   columns=["Pred. Baja", "Pred. Media", "Pred. Alta"]))

# Importancia de variables (Random Forest)
rf_final = rf_search.best_estimator_
nombres = rf_final.named_steps["preprocesamiento"].get_feature_names_out()
imp = pd.Series(rf_final.named_steps["clasificador"].feature_importances_, index=nombres)
imp.index = [i.split("__", 1)[1] for i in imp.index]
imp = imp.groupby(lambda n: n if not n.startswith("track_genre_") else "track_genre (OHE)").sum()
imp = imp.sort_values(ascending=False)
print("\nImportancia de variables (Random Forest, genero OHE agrupado):")
print(imp.head(12).round(4))

# ---------------- Artefactos ----------------
(base_dir / "models").mkdir(exist_ok=True)
joblib.dump(rf_search.best_estimator_, base_dir / "models" / "random_forest_optimizado.pkl", compress=3)
joblib.dump(gb_search.best_estimator_, base_dir / "models" / "gradient_boosting_optimizado.pkl", compress=3)
joblib.dump(voting, base_dir / "models" / "voting_ensamble.pkl", compress=3)
resultados.to_csv(base_dir / "data" / "resultados_ensamble.csv", index=False)
imp.round(6).to_csv(base_dir / "data" / "importancia_variables_rf.csv", header=["importancia"])

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
x = range(len(resultados))
ax[0].bar([i - 0.2 for i in x], resultados["accuracy"], 0.4, label="Accuracy")
ax[0].bar([i + 0.2 for i in x], resultados["f1_macro"], 0.4, label="F1 macro")
ax[0].set_xticks(list(x))
ax[0].set_xticklabels([n.replace(" (", "\n(") for n in resultados["modelo"]], fontsize=8)
ax[0].set_ylim(0, 1); ax[0].set_title("Clasificacion de popularidad: arbol vs ensambles")
ax[0].legend(loc="lower right")
for i, v in enumerate(resultados["accuracy"]):
    ax[0].text(i - 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
top = imp.head(10)[::-1]
ax[1].barh(top.index, top.values, color="#4C72B0")
ax[1].set_title("Importancia de variables (Random Forest)")
plt.tight_layout()
(base_dir / "images").mkdir(exist_ok=True)
plt.savefig(base_dir / "images" / "comparacion_ensambles.png", dpi=130)
print("\nArtefactos guardados.")
