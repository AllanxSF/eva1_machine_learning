import pandas as pd

COLS_TEXTO = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "track_genre"
]


COLS_REQUERIDAS = {
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
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
    "time_signature",
    "track_genre",
}


def limpieza_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y prepara el dataset de Spotify Tracks.
    """

    df = df.copy()

    # ------------------------------------------
    # Normalizar nombres de columnas
    # ------------------------------------------

    df.columns = df.columns.str.lower().str.strip()


    # ------------------------------------------
    # Eliminar índice original
    # ------------------------------------------

    if "unnamed: 0" in df.columns:
        df = df.drop(columns=["unnamed: 0"])


    # ------------------------------------------
    # Validar columnas requeridas
    # ------------------------------------------

    faltantes = COLS_REQUERIDAS - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )


    # ------------------------------------------
    # Limpiar variables de texto
    # ------------------------------------------

    for col in COLS_TEXTO:
        df[col] = df[col].astype("string").str.strip()


    # ------------------------------------------
    # Convertir variables numéricas
    # ------------------------------------------

    columnas_numericas = [
        "popularity",
        "duration_ms",
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

    for col in columnas_numericas:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    # ------------------------------------------
    # Tratamiento de valores nulos
    # ------------------------------------------

    df = df.dropna()


    # ------------------------------------------
    # Eliminar duplicados exactos
    # ------------------------------------------

    df = df.drop_duplicates()


    # ------------------------------------------
    # Reiniciar índice
    # ------------------------------------------

    df = df.reset_index(drop=True)


    return df