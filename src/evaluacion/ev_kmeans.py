
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ==================================
# CARGAR DATOS
# ==================================

base_dir = Path(__file__).resolve().parents[2]

ruta_datos = (
    base_dir
    / "data"
    / "Spotify_Tracks_Clusters.csv"
)

ruta_modelo = base_dir / "models" / "modelo_kmeans.pkl"
ruta_scaler = base_dir / "models" / "scaler_kmeans.pkl"

df = pd.read_csv(ruta_datos, encoding="utf-8")


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
# CARGAR MODELO Y ESCALADOR
# ==================================

kmeans_escalado = joblib.load(ruta_modelo)
scaler = joblib.load(ruta_scaler)

X_escalado = scaler.transform(X)


# ==================================
# SILHOUETTE SCORE
# ==================================

# Usar una muestra para reducir el tiempo de calculo
score = silhouette_score(
    X_escalado,
    df["Cluster"],
    sample_size=10000,
    random_state=42
)

print("\nSilhouette Score:")
print(round(score, 4))


# ==================================
# PROMEDIOS POR CLUSTER
# ==================================

resumen = df.groupby("Cluster")[columnas_modelo].mean()

print("\nPromedios por cluster:")
print(resumen.round(2))


# ==================================
# VISUALIZACION DE CLUSTERS
# ==================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["danceability"],
    df["energy"],
    c=df["Cluster"],
    cmap="viridis",
    alpha=0.5,
    s=10
)

plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.title("Clusters de canciones de Spotify")
plt.tight_layout()

# Guardar el grafico
ruta_grafico = base_dir / "images" / "clusters_spotify.png"
ruta_grafico.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(ruta_grafico, dpi=150)
plt.show()

print("\nGrafico guardado en:", ruta_grafico)


# ==================================
# DISTRIBUCION DE CLUSTERS
# ==================================

print("\nCantidad de canciones por cluster:")
print(df["Cluster"].value_counts().sort_index())


# ==================================
# PORCENTAJE DE CLUSTERS
# ==================================

print("\nPorcentaje por cluster:")
print(
    (
        df["Cluster"].value_counts(normalize=True).sort_index() * 100
    ).round(2)
)


# ==================================
# POPULARIDAD POR CLUSTER
# ==================================

print("\nPopularidad por cluster:")
print(
    df.groupby("Cluster")["popularity"]
    .agg(["mean", "median", "count"])
    .round(2)
)


print("\nEvaluacion de K-Means completada correctamente")
