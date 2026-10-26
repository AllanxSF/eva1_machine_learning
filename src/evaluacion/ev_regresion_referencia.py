
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Analisis complementario: cuanto aporta cada grupo de predictores a la
# regresion lineal (misma particion que el entrenamiento).
base_dir = Path(__file__).resolve().parents[2]
df = pd.read_csv(base_dir / "data" / "Spotify_Tracks_Limpio.csv", encoding="utf-8")

columnas_numericas = [
    "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature"
]
columnas_categoricas = ["track_genre"]

x = df[columnas_numericas + columnas_categoricas]
y = df["popularity"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

cat = ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columnas_categoricas)
num = ("num", StandardScaler(), columnas_numericas)

configuraciones = {
    "Solo track_genre": [cat],
    "Solo variables numericas": [num],
    "Genero + numericas (modelo final)": [cat, num],
}

filas = []
for nombre, transformadores in configuraciones.items():
    modelo = Pipeline([
        ("pre", ColumnTransformer(transformadores)),
        ("reg", LinearRegression()),
    ])
    modelo.fit(x_train, y_train)
    pred = modelo.predict(x_test)
    filas.append({
        "Predictores": nombre,
        "R2 (Test)": r2_score(y_test, pred),
        "MAE (Test)": mean_absolute_error(y_test, pred),
    })

print("\nAporte de cada grupo de predictores:")
print(pd.DataFrame(filas).round(4).to_string(index=False))
