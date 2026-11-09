"""Transformadores propios para los modelos de clasificacion de popularidad."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold

CLASES = ["Baja", "Media", "Alta"]


class HistorialPopularidad(BaseEstimator, TransformerMixin):
    """
    Codificacion por objetivo (target encoding) del historial de popularidad
    de una entidad (artista principal, album) para cada nivel de popularidad.

    Por cada columna genera:
      - <col>_p_baja, <col>_p_media, <col>_p_alta: proporcion suavizada de
        canciones de esa entidad en cada nivel (suavizado hacia la proporcion global).
      - <col>_n: cantidad de canciones conocidas de esa entidad en entrenamiento.

    Para evitar fuga de informacion (data leakage), durante el ajuste las
    variables de entrenamiento se calculan "fuera de pliegue": cada fila se
    codifica con estadisticas de OTROS pliegues, agrupados por `col_grupo`
    (track_id), de modo que una cancion nunca se codifica con su propia
    etiqueta ni con la de sus copias en otros generos. En transform() se usan
    las estadisticas completas de entrenamiento; entidades no vistas reciben
    la proporcion global y n = 0.
    """

    def __init__(self, columnas=("first_artist", "album_name"),
                 col_grupo="track_id", suavizado=5, n_pliegues=5):
        self.columnas = columnas
        self.col_grupo = col_grupo
        self.suavizado = suavizado
        self.n_pliegues = n_pliegues

    # --- utilidades -------------------------------------------------------
    def _conteos(self, valores, y):
        tabla = pd.crosstab(valores, y).reindex(columns=CLASES, fill_value=0)
        return tabla

    def _codificar(self, valores, tabla):
        cnt = tabla.reindex(valores.values).fillna(0).to_numpy()
        n = cnt.sum(axis=1, keepdims=True)
        prior = self.prior_.reshape(1, -1)
        prop = (cnt + self.suavizado * prior) / (n + self.suavizado)
        return np.hstack([prop, n])

    def _nombres(self, col):
        return [f"{col}_p_baja", f"{col}_p_media", f"{col}_p_alta", f"{col}_n"]

    # --- API sklearn ------------------------------------------------------
    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.prior_ = y.value_counts(normalize=True).reindex(CLASES).to_numpy()
        self.tablas_ = {c: self._conteos(X[c], y) for c in self.columnas}
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        y = pd.Series(np.asarray(y), index=X.index)
        salida = np.zeros((len(X), 4 * len(self.columnas)))
        pliegues = GroupKFold(n_splits=self.n_pliegues).split(
            X, groups=X[self.col_grupo]
        )
        for idx_fit, idx_apl in pliegues:
            for k, c in enumerate(self.columnas):
                tabla = self._conteos(X[c].iloc[idx_fit], y.iloc[idx_fit])
                salida[idx_apl, 4 * k:4 * k + 4] = self._codificar(
                    X[c].iloc[idx_apl], tabla
                )
        return salida

    def transform(self, X):
        partes = [self._codificar(X[c], self.tablas_[c]) for c in self.columnas]
        return np.hstack(partes)

    def get_feature_names_out(self, input_features=None):
        return np.array([n for c in self.columnas for n in self._nombres(c)])
