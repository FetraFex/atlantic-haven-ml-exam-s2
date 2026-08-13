"""Pipelines de prétraitement (fit uniquement sur le train)."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import drop_non_feature_columns


def _is_categorical_series(s: pd.Series) -> bool:
    """Détecte object / category / StringDtype (pandas 2+/3+)."""
    dtype = s.dtype
    if dtype == "object" or str(dtype) == "category":
        return True
    if pd.api.types.is_string_dtype(dtype):
        return True
    if pd.api.types.is_bool_dtype(dtype):
        return False
    # fallback : majorité de valeurs non convertibles en float
    if dtype == "object" or getattr(dtype, "kind", None) in {"O", "U"}:
        return True
    sample = s.dropna().head(50)
    if len(sample) == 0:
        return False
    try:
        pd.to_numeric(sample)
        return False
    except (ValueError, TypeError):
        return True


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Sépare numériques et catégorielles après exclusion des colonnes non-features."""
    X = drop_non_feature_columns(df)
    cat_cols, num_cols = [], []
    for c in X.columns:
        if _is_categorical_series(X[c]):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, cat_cols


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Prétraitement scikit-learn :
    - numériques : imputation médiane (+ standardisation si scale_numeric) ;
    - catégorielles : imputation mode + OneHotEncoder(handle_unknown='ignore').

    handle_unknown='ignore' gère les catégories absentes du train.
    """
    num_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric = Pipeline(steps=num_steps)
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=10,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


def prepare_xy(df: pd.DataFrame, target: str = "reservation_annulee"):
    """Retourne X (features brutes + engineered) et y si disponible."""
    X = drop_non_feature_columns(df)
    y = df[target].astype(int) if target in df.columns else None
    return X, y
