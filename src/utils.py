"""Utilitaires communs : constantes, chargement, seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET_COL = "reservation_annulee"
ID_COL = "reservation_id"
TIME_COL = "date_reservation"  # colonne temporelle retenue (création de la réservation)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fixe les graines NumPy (et tente celles des libs optionnelles)."""
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass


def load_raw_data(
    parse_dates: bool = True,
    sort_train: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge train et test depuis data/raw sans modifier les fichiers sources.

    - Le test conserve l'ordre original des lignes (obligatoire pour submission.csv).
    - sort_train=True trie uniquement le train par TIME_COL (utile hors pipeline).
    """
    train = pd.read_csv(RAW_DIR / "reservations_train.csv")
    test = pd.read_csv(RAW_DIR / "reservations_test.csv")

    if parse_dates:
        for df in (train, test):
            df["date_reservation"] = pd.to_datetime(df["date_reservation"])
            df["date_arrivee"] = pd.to_datetime(df["date_arrivee"])

    if sort_train:
        train = train.sort_values(TIME_COL).reset_index(drop=True)

    return train, test


def ensure_dirs() -> None:
    """Crée les dossiers de sortie s'ils n'existent pas."""
    for d in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_figure(fig, name: str) -> Path:
    """Enregistre une figure matplotlib dans report/figures."""
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
