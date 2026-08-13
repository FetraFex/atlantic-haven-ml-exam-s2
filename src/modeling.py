"""Construction, entraînement et comparaison des modèles."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, get_feature_columns, prepare_xy
from .utils import RANDOM_STATE
from .validation import evaluate_predictions, optimize_threshold

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def build_logistic_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(num_cols, cat_cols, scale_numeric=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_random_forest_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(num_cols, cat_cols, scale_numeric=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=16,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_hgb_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(num_cols, cat_cols, scale_numeric=False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=8,
                    learning_rate=0.06,
                    max_iter=350,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_xgb_pipeline(
    num_cols: list[str], cat_cols: list[str], scale_pos_weight: float
) -> Pipeline:
    if not HAS_XGB:
        raise ImportError("xgboost non disponible")
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(num_cols, cat_cols, scale_numeric=False)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_lgbm_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    if not HAS_LGBM:
        raise ImportError("lightgbm non disponible")
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(num_cols, cat_cols, scale_numeric=False)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=48,
                    max_depth=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def fit_and_score(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    name: str,
) -> dict[str, Any]:
    """Entraîne, optimise le seuil sur validation, retourne métriques."""
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    proba = pipeline.predict_proba(X_valid)[:, 1]
    metrics_05 = evaluate_predictions(y_valid, proba, threshold=0.5)
    thr, f1_opt, thr_grid, f1_grid = optimize_threshold(y_valid, proba)
    metrics_opt = evaluate_predictions(y_valid, proba, threshold=thr)

    return {
        "name": name,
        "pipeline": pipeline,
        "train_time_sec": float(train_time),
        "proba_valid": proba,
        "threshold": thr,
        "metrics_05": metrics_05,
        "metrics_opt": metrics_opt,
        "threshold_grid": thr_grid,
        "f1_grid": f1_grid,
        "f1": metrics_opt["f1"],
    }


def compare_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compare LR + RF + HGB (+ XGB/LGBM si installés) sur le même split temporel."""
    X_tr, y_tr = prepare_xy(train_df)
    X_va, y_va = prepare_xy(valid_df)
    num_cols, cat_cols = get_feature_columns(train_df)

    pos = float(y_tr.mean())
    scale = (1.0 - pos) / max(pos, 1e-9)

    builders: list[tuple[str, Any]] = [
        ("logistic_regression", lambda: build_logistic_pipeline(num_cols, cat_cols)),
        ("random_forest", lambda: build_random_forest_pipeline(num_cols, cat_cols)),
        ("hist_gradient_boosting", lambda: build_hgb_pipeline(num_cols, cat_cols)),
    ]
    if HAS_XGB:
        builders.append(
            ("xgboost", lambda: build_xgb_pipeline(num_cols, cat_cols, scale))
        )
    if HAS_LGBM:
        builders.append(("lightgbm", lambda: build_lgbm_pipeline(num_cols, cat_cols)))

    results = {}
    for name, builder in builders:
        print(f"\n--- Entraînement : {name} ---")
        pipe = builder()
        results[name] = fit_and_score(pipe, X_tr, y_tr, X_va, y_va, name)
        m = results[name]["metrics_opt"]
        print(
            f"{name}: thr={m['threshold']:.2f} F1={m['f1']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} "
            f"PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
            f"time={results[name]['train_time_sec']:.1f}s"
        )

    best_name = max(results, key=lambda k: results[k]["f1"])
    return {
        "results": results,
        "best_name": best_name,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "scale_pos_weight": scale,
    }


def rebuild_pipeline(name: str, num_cols, cat_cols, scale_pos_weight: float) -> Pipeline:
    mapping = {
        "logistic_regression": lambda: build_logistic_pipeline(num_cols, cat_cols),
        "random_forest": lambda: build_random_forest_pipeline(num_cols, cat_cols),
        "hist_gradient_boosting": lambda: build_hgb_pipeline(num_cols, cat_cols),
    }
    if HAS_XGB:
        mapping["xgboost"] = lambda: build_xgb_pipeline(num_cols, cat_cols, scale_pos_weight)
    if HAS_LGBM:
        mapping["lightgbm"] = lambda: build_lgbm_pipeline(num_cols, cat_cols)
    if name not in mapping:
        raise KeyError(f"Modèle inconnu : {name}")
    return mapping[name]()


def assert_submission(
    submission: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_path=None,
) -> None:
    """Assertions de conformité du fichier de soumission."""
    expected_cols = ["reservation_id", "probabilite_annulation", "reservation_annulee"]
    assert list(submission.columns) == expected_cols, (
        f"Colonnes incorrectes : {list(submission.columns)}"
    )
    assert len(submission) == len(test_df), (
        f"Nb lignes {len(submission)} != {len(test_df)}"
    )
    assert submission["reservation_id"].tolist() == test_df["reservation_id"].tolist(), (
        "Ordre des reservation_id non respecté"
    )
    assert submission.isna().sum().sum() == 0, "Valeurs manquantes dans submission"
    proba = submission["probabilite_annulation"].astype(float)
    assert ((proba >= 0) & (proba <= 1)).all(), "Probabilités hors [0, 1]"
    preds = submission["reservation_annulee"]
    assert set(preds.unique()).issubset({0, 1}), "Décisions non binaires"
    if sample_path is not None:
        sample = pd.read_csv(sample_path)
        assert list(sample.columns) == expected_cols
        assert len(sample) == len(submission)
