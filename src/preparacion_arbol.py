"""Preparacion de datos compartida por la optimizacion y evaluacion del arbol."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

NUMERICAS = [
    "duration_ms", "explicit", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]
CATEGORICAS = ["track_genre"]
HISTORIAL = ["first_artist", "album_name"]   # codificadas por HistorialPopularidad
COL_GRUPO = "track_id"                        # evita fuga entre copias de una cancion


def cargar_datos(base_dir):
    df = pd.read_csv(base_dir / "data" / "Spotify_Tracks_Limpio.csv")
    df["popularity_level"] = pd.cut(
        df["popularity"], bins=[-1, 33, 66, 100], labels=["Baja", "Media", "Alta"]
    )
    df = df.dropna(subset=["popularity_level"])
    # Artista principal (el campo `artists` separa colaboraciones con ';')
    df["first_artist"] = df["artists"].astype(str).str.split(";").str[0].str.strip()
    return df


def dividir(df):
    """Split 80/20 agrupado por track_id (una cancion nunca queda en ambos conjuntos)."""
    columnas = NUMERICAS + CATEGORICAS + HISTORIAL + [COL_GRUPO]
    x, y = df[columnas], df["popularity_level"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_tr, idx_te = next(gss.split(x, y, groups=df[COL_GRUPO]))
    return x.iloc[idx_tr], x.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
