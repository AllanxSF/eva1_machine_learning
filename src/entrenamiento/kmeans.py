
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import joblib
from pathlib import Path


# ==================================
# CARGAR DATOS
# ==================================

base_dir = Path(__file__).resolve().parents[2]

ruta = (
    base_dir
    / "data"
    / "Spotify_Tracks_Limpio.csv"
)

df = pd.read_csv(
    ruta,
    encoding="utf-8"
)

df = df.copy()


# ==================================
# VARIABLES PARA KMEANS
# ==================================

columnas_modelo = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

X = df[columnas_modelo]


# ==================================
# MODELO SIN ESCALAR
# ==================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster_SinEscalar"] = kmeans.fit_predict(X)

print("\nPrimeros registros sin escalar:")
print(df[["track_name", "Cluster_SinEscalar"]].head())


# ==================================
# ESCALADO
# ==================================

scaler = StandardScaler()

X_escalado = scaler.fit_transform(X)


# ==================================
# MODELO ESCALADO
# ==================================

kmeans_escalado = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans_escalado.fit_predict(X_escalado)

print("\nPrimeros registros con escalado:")
print(df[["track_name", "Cluster"]].head())


# ==================================
# RESUMEN DE CLUSTERS
# ==================================

print("\nCantidad de canciones por cluster:")
print(df["Cluster"].value_counts().sort_index())


print("\nPromedios por cluster:")

print(
    df.groupby("Cluster")[columnas_modelo]
    .mean()
    .round(2)
)


# ==================================
# RELACION CON LA POPULARIDAD
# ==================================

print("\nPopularidad promedio por cluster:")

print(
    df.groupby("Cluster")["popularity"]
    .agg(["mean", "median", "count"])
    .round(2)
)


# ==================================
# GUARDAR MODELO Y ESCALADOR
# ==================================

ruta_modelos = base_dir / "models"
ruta_modelos.mkdir(parents=True, exist_ok=True)

joblib.dump(
    kmeans_escalado,
    ruta_modelos / "modelo_kmeans.pkl"
)

joblib.dump(
    scaler,
    ruta_modelos / "scaler_kmeans.pkl"
)


# ==================================
# GUARDAR DATASET CON CLUSTERS
# ==================================

ruta_clusters = (
    base_dir
    / "data"
    / "Spotify_Tracks_Clusters.csv"
)

df.to_csv(
    ruta_clusters,
    index=False,
    encoding="utf-8"
)


print("\nModelo y escalador guardados correctamente.")
print("Modelo:", ruta_modelos / "modelo_kmeans.pkl")
print("Escalador:", ruta_modelos / "scaler_kmeans.pkl")
print("Dataset con clusters:", ruta_clusters)
