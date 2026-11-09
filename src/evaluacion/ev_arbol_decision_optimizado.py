"""Evalua el arbol optimizado (v2) y el optimizado anterior (v1) con el mismo split agrupado."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir / "src"))

from preparacion_arbol import COL_GRUPO, cargar_datos, dividir  # noqa: E402
import transformadores  # noqa: E402,F401  (necesario para deserializar el modelo)

df = cargar_datos(base_dir)
x_train, x_test, y_train, y_test = dividir(df)

modelo = joblib.load(base_dir / "models" / "arbol_decision_optimizado.pkl")
y_pred = modelo.predict(x_test)

etiquetas = ["Baja", "Media", "Alta"]
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("F1 macro:", round(f1_score(y_test, y_pred, average="macro"), 4))
print("\nReporte de clasificacion:")
print(pd.DataFrame(classification_report(
    y_test, y_pred, labels=etiquetas, output_dict=True, zero_division=0
)).transpose().round(4))
print("\nMatriz de confusion:")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=etiquetas),
    index=["Real Baja", "Real Media", "Real Alta"],
    columns=["Predicha Baja", "Predicha Media", "Predicha Alta"],
))

# Comparacion justa: modelo v1 (optimizado anterior) y base, con el MISMO split agrupado.
cols_v1 = [c for c in x_test.columns if c not in ("first_artist", "album_name", COL_GRUPO)]
for nombre, archivo in [("Base (max_depth=4)", "arbol_decision.pkl"),
                        ("Optimizado v1", "arbol_decision_optimizado_v1.pkl")]:
    m = joblib.load(base_dir / "models" / archivo)
    p = m.predict(x_test[cols_v1])
    print(f"\n{nombre} en split agrupado -> accuracy {accuracy_score(y_test, p):.4f}, "
          f"F1 macro {f1_score(y_test, p, average='macro'):.4f}")

# Escenario exigente: artistas nunca vistos en entrenamiento
vistos = set(x_train["first_artist"])
mask = ~x_test["first_artist"].isin(vistos)
print(f"\nCanciones de prueba con artista no visto: {mask.sum()} ({mask.mean():.1%})")
print("Accuracy artista visto   :", round(accuracy_score(y_test[~mask], y_pred[~mask.values]), 4))
print("Accuracy artista no visto:", round(accuracy_score(y_test[mask], y_pred[mask.values]), 4))
