
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Rutas del proyecto
base_dir = Path(__file__).resolve().parents[2]
ruta_datos = base_dir / "data" / "Spotify_Tracks_Limpio.csv"
ruta_imagenes = base_dir / "images"
ruta_imagenes.mkdir(parents=True, exist_ok=True)

# Cargar dataset limpio
df = pd.read_csv(ruta_datos, encoding="utf-8")

# Variables utilizadas por K-Means
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
    "tempo",
]

X = df[columnas_modelo]

# Escalar las variables
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

# Muestra reproducible para calcular Silhouette
muestra = pd.DataFrame(X_escalado).sample(
    n=min(10000, len(df)),
    random_state=42
).to_numpy()

resultados = []
inercias = []

# Método del codo: K de 1 a 10
for k in range(1, 11):
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    modelo.fit(X_escalado)
    inercias.append(modelo.inertia_)

# Silhouette: K de 2 a 7
for k in range(2, 8):
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    modelo.fit(X_escalado)

    etiquetas_muestra = modelo.predict(muestra)
    score = silhouette_score(muestra, etiquetas_muestra)

    resultados.append({
        "K": k,
        "silhouette_score": score,
        "inercia": modelo.inertia_,
    })

    print(
        f"K={k} | Silhouette Score={score:.4f} "
        f"| Inercia={modelo.inertia_:.2f}"
    )

# Guardar tabla de resultados
df_resultados = pd.DataFrame(resultados)
ruta_resultados = base_dir / "data" / "resultados_optimizacion_kmeans.csv"
df_resultados.to_csv(ruta_resultados, index=False, encoding="utf-8")

mejor = df_resultados.loc[df_resultados["silhouette_score"].idxmax()]
print("\nMejor K según Silhouette Score:", int(mejor["K"]))
print(f"Mejor Silhouette Score: {mejor['silhouette_score']:.4f}")
print("Resultados guardados en:", ruta_resultados)

# Gráfico del método del codo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inercias, marker="o")
plt.xticks(range(1, 11))
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.title("Método del codo para K-Means")
plt.grid(True, alpha=0.3)
plt.tight_layout()

ruta_grafico = ruta_imagenes / "metodo_codo_kmeans.png"
plt.savefig(ruta_grafico, dpi=150)
plt.close()

print("Gráfico guardado en:", ruta_grafico)
