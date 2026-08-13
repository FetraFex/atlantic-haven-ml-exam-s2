"""Validation temporelle et métriques d'évaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .utils import TARGET_COL, TIME_COL


@dataclass
class TemporalSplitInfo:
    train: pd.DataFrame
    valid: pd.DataFrame
    cut_index: int
    train_period: tuple[str, str]
    valid_period: tuple[str, str]
    train_cancel_rate: float
    valid_cancel_rate: float


def temporal_holdout(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    val_ratio: float = 0.20,
    verbose: bool = True,
) -> TemporalSplitInfo:
    """
    Holdout temporel strict :
    - trie chronologiquement par time_col (pas de shuffle) ;
    - observations anciennes → entraînement ;
    - observations les plus récentes → validation ;
    - aucune période mélangée.

    val_ratio=0.20 : les 20 % les plus récentes (par date_reservation).
    """
    if time_col not in df.columns:
        raise KeyError(f"Colonne temporelle absente : {time_col}")

    sorted_df = df.sort_values(time_col).reset_index(drop=True)
    n = len(sorted_df)
    cut = int(n * (1.0 - val_ratio))
    if cut <= 0 or cut >= n:
        raise ValueError(f"val_ratio={val_ratio} produit un découpage invalide (n={n})")

    train = sorted_df.iloc[:cut].copy()
    valid = sorted_df.iloc[cut:].copy()

    # Vérification anti-fuite temporelle
    assert train[time_col].max() <= valid[time_col].min(), (
        "Chevauchement temporel train/valid détecté"
    )

    info = TemporalSplitInfo(
        train=train,
        valid=valid,
        cut_index=cut,
        train_period=(
            str(train[time_col].min().date()),
            str(train[time_col].max().date()),
        ),
        valid_period=(
            str(valid[time_col].min().date()),
            str(valid[time_col].max().date()),
        ),
        train_cancel_rate=float(train[TARGET_COL].mean()),
        valid_cancel_rate=float(valid[TARGET_COL].mean()),
    )

    if verbose:
        print_split_summary(info)
    return info


def print_split_summary(info: TemporalSplitInfo) -> None:
    """Affiche dates, dimensions et taux d'annulation de chaque partie."""
    print("=== Validation temporelle (holdout) ===")
    print(
        f"Train : {len(info.train):,} lignes | "
        f"{info.train_period[0]} → {info.train_period[1]} | "
        f"taux annulation = {info.train_cancel_rate:.4f}"
    )
    print(
        f"Valid : {len(info.valid):,} lignes | "
        f"{info.valid_period[0]} → {info.valid_period[1]} | "
        f"taux annulation = {info.valid_cancel_rate:.4f}"
    )


def optimize_threshold(
    y_true,
    y_proba,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Parcourt une grille de seuils et maximise le F1 de la classe positive.
    Ne doit JAMAIS être appelé sur le jeu de test.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    scores = np.array(
        [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    )
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx]), thresholds, scores


def evaluate_predictions(
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> dict:
    """Calcule F1, précision, rappel, PR-AUC, ROC-AUC et matrice de confusion."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0
        ),
    }
