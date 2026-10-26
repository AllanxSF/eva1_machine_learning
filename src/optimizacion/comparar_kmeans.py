
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Cargar datos
base_dir = Path(__file__).resolve().parents[2]
df = pd.read_csv(base_dir / "data" / "Spotify_Tracks_Limpio.csv")

# Variables musicales
columnas = [
    "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]

X = df[columnas]

# Escalar variables
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

# Comparar K = 2 y K = 4
for k in [2, 4]:
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df[f"Cluster_{k}"] = modelo.fit_predict(X_escalado)

    print(f"\n========== K = {k} ==========")

    print("\nCantidad de canciones por cluster:")
    print(df[f"Cluster_{k}"].value_counts().sort_index())

    print("\nPromedios musicales:")
    print(
        df.groupby(f"Cluster_{k}")[columnas]
        .mean()
        .round(2)
    )

    print("\nPopularidad por cluster:")
    print(
        df.groupby(f"Cluster_{k}")["popularity"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
