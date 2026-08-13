#!/usr/bin/env python3
"""
Pipeline principal du projet Atlantic Haven Hotels.

Ce module orchestre les étapes suivantes :
- chargement et contrôle des données ;
- analyse exploratoire ;
- création des variables ;
- validation temporelle ;
- comparaison des modèles ;
- optimisation du seuil ;
- génération des métriques, figures et prédictions finales.

Cette documentation n'affecte pas le comportement du pipeline.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_GROUPS, engineer_features, leakage_checklist
from src.modeling import assert_submission, compare_models, rebuild_pipeline
from src.preprocessing import get_feature_columns, prepare_xy
from src.utils import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    TARGET_COL,
    TIME_COL,
    ensure_dirs,
    load_raw_data,
    save_figure,
    set_seed,
)
from src.validation import evaluate_predictions, optimize_threshold, temporal_holdout

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")


def run_eda(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Réalise l'analyse exploratoire et sauvegarde les graphiques principaux.

    Args:
        train: Données d'entraînement contenant la variable cible.
        test: Données de test utilisées uniquement pour les contrôles descriptifs.

    Returns:
        Un dictionnaire contenant les statistiques principales de l'EDA.
    """
    ensure_dirs()

    eda = {}
    eda["train_shape"] = list(train.shape)
    eda["test_shape"] = list(test.shape)
    eda["cancel_rate"] = float(train[TARGET_COL].mean())

    eda["train_period"] = [
        str(train[TIME_COL].min().date()),
        str(train[TIME_COL].max().date()),
    ]

    eda["test_period"] = [
        str(test[TIME_COL].min().date()),
        str(test[TIME_COL].max().date()),
    ]

    eda["missing_train"] = (
        train.isna()
        .sum()[train.isna().sum() > 0]
        .astype(int)
        .to_dict()
    )

    # Comptage des valeurs vides de la colonne agent_id.
    agent_empty = int(
        (
            train["agent_id"].isna()
            | (train["agent_id"].astype(str).str.strip() == "")
        ).sum()
    )

    eda["agent_id_empty_train"] = agent_empty
    eda["dup_id_train"] = int(
        train["reservation_id"].duplicated().sum()
    )
    eda["dup_id_test"] = int(
        test["reservation_id"].duplicated().sum()
    )

    # Visualisation de la distribution de la variable cible.
    fig, ax = plt.subplots(figsize=(6, 4))

    counts = (
        train[TARGET_COL]
        .value_counts()
        .sort_index()
    )

    ax.bar(
        ["Maintenue (0)", "Annulée (1)"],
        counts.values,
        color=["#2a9d8f", "#e76f51"],
    )

    ax.set_title(
        f"Répartition de la cible "
        f"(taux={eda['cancel_rate']:.1%})"
    )
    ax.set_ylabel("Nombre de réservations")

    for index, value in enumerate(counts.values):
        ax.text(
            index,
            value + 50,
            str(value),
            ha="center",
        )

    save_figure(fig, "01_cible_distribution.png")
    plt.close(fig)

    # Évolution mensuelle du taux d'annulation.
    tmp = train.copy()
    tmp["mois"] = (
        tmp[TIME_COL]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        tmp.groupby("mois")[TARGET_COL]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
        color="#264653",
    )

    ax.set_title(
        "Évolution temporelle du taux d'annulation (train)"
    )
    ax.set_xlabel("Mois de réservation")
    ax.set_ylabel("Taux d'annulation")
    ax.tick_params(axis="x", rotation=45)

    save_figure(fig, "02_taux_annulation_temporel.png")
    plt.close(fig)

    # Analyse du taux d'annulation par variable catégorielle.
    categorical_analyses = [
        (
            "canal_reservation",
            "03_annulation_par_canal.png",
        ),
        (
            "type_acompte",
            "04_annulation_par_acompte.png",
        ),
        (
            "tarif_remboursable",
            "05_annulation_tarif_remboursable.png",
        ),
        (
            "type_destination",
            "06_annulation_type_destination.png",
        ),
        (
            "client_type",
            "07_annulation_client_type.png",
        ),
    ]

    for column, filename in categorical_analyses:
        rates = (
            train.groupby(column)[TARGET_COL]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.barh(
            rates.index.astype(str),
            rates["mean"],
            color="#457b9d",
        )

        ax.set_xlabel("Taux d'annulation")
        ax.set_title(
            f"Taux d'annulation selon {column}"
        )

        save_figure(fig, filename)
        plt.close(fig)

        eda[f"rate_by_{column}"] = {
            str(key): {
                "rate": float(value["mean"]),
                "n": int(value["count"]),
            }
            for key, value in rates.iterrows()
        }

    # Comparaison du délai de réservation selon la cible.
    fig, ax = plt.subplots(figsize=(7, 4))

    sns.boxplot(
        data=train,
        x=TARGET_COL,
        y="delai_reservation_jours",
        ax=ax,
        palette=["#2a9d8f", "#e76f51"],
    )

    ax.set_xticklabels(
        ["Maintenue", "Annulée"]
    )
    ax.set_title(
        "Délai de réservation selon la cible"
    )

    save_figure(fig, "08_delai_vs_cible.png")
    plt.close(fig)

    return eda


def ablation_study(
    train_fe: pd.DataFrame,
    valid_fe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mesure le gain de F1-score apporté par chaque groupe de variables.

    La régression logistique sans feature engineering est utilisée
    comme référence.

    Args:
        train_fe: Données d'entraînement avec les variables créées.
        valid_fe: Données de validation avec les variables créées.

    Returns:
        Un tableau comparant les différentes configurations.
    """
    from src.modeling import (
        build_logistic_pipeline,
        fit_and_score,
    )

    base_cols = [
        column
        for column in train_fe.columns
        if column
        not in {
            "reservation_id",
            "date_reservation",
            "date_arrivee",
            TARGET_COL,
            "agent_id",
        }
        and column not in sum(
            FEATURE_GROUPS.values(),
            [],
        )
    ]

    rows = []

    configs = {
        "sans_fe": base_cols,
    }

    for group_name, features in FEATURE_GROUPS.items():
        configs[f"groupe_{group_name}"] = (
            base_cols
            + [
                feature
                for feature in features
                if feature in train_fe.columns
            ]
        )

    configs["toutes_fe"] = (
        base_cols
        + [
            feature
            for feature in sum(
                FEATURE_GROUPS.values(),
                [],
            )
            if feature in train_fe.columns
        ]
    )

    for configuration_name, columns in configs.items():
        use_cols = [
            column
            for column in columns
            if column in train_fe.columns
        ]

        X_train = train_fe[use_cols]
        X_valid = valid_fe[use_cols]

        y_train = train_fe[TARGET_COL].astype(int)
        y_valid = valid_fe[TARGET_COL].astype(int)

        temporary_data = train_fe[
            use_cols + [TARGET_COL]
        ]

        numeric_columns, categorical_columns = (
            get_feature_columns(temporary_data)
        )

        pipeline = build_logistic_pipeline(
            numeric_columns,
            categorical_columns,
        )

        result = fit_and_score(
            pipeline,
            X_train,
            y_train,
            X_valid,
            y_valid,
            configuration_name,
        )

        rows.append(
            {
                "config": configuration_name,
                "n_features": len(use_cols),
                "f1": result["f1"],
                "precision": (
                    result["metrics_opt"]["precision"]
                ),
                "recall": (
                    result["metrics_opt"]["recall"]
                ),
                "threshold": result["threshold"],
            }
        )

        print(
            f"Ablation {configuration_name}: "
            f"F1={result['f1']:.4f}"
        )

    table = pd.DataFrame(rows)

    base_f1 = float(
        table.loc[
            table["config"] == "sans_fe",
            "f1",
        ].iloc[0]
    )

    table["delta_f1"] = (
        table["f1"] - base_f1
    )

    table.to_csv(
        PROCESSED_DIR / "ablation_features.csv",
        index=False,
    )

    return table


def main() -> dict:
    """
    Exécute le pipeline complet du projet.

    Returns:
        Un dictionnaire regroupant les métriques, le modèle sélectionné,
        les erreurs analysées et les informations de soumission.
    """
    set_seed()
    ensure_dirs()

    print("=== Chargement ===")

    train_raw, test_raw = load_raw_data(
        parse_dates=True,
        sort_train=False,
    )

    # Conservation de l'ordre original des identifiants du test.
    test_order_ids = (
        test_raw["reservation_id"]
        .tolist()
    )

    print(
        f"Train {train_raw.shape} | "
        f"Test {test_raw.shape}"
    )

    eda = run_eda(
        train_raw,
        test_raw,
    )

    print("\n=== Feature engineering ===")

    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)

    print("\n=== Split temporel ===")

    split = temporal_holdout(
        train_fe,
        time_col=TIME_COL,
        val_ratio=0.20,
        verbose=True,
    )

    train_split = split.train
    valid_split = split.valid

    print(
        "\n=== Ablation FE "
        "(régression logistique) ==="
    )

    ablation = ablation_study(
        train_split,
        valid_split,
    )

    print("\n=== Comparaison modèles ===")

    comparison = compare_models(
        train_split,
        valid_split,
    )

    results = comparison["results"]
    best_name = comparison["best_name"]
    best = results[best_name]

    print(
        f"\n>>> Modèle sélectionné : {best_name} "
        f"| F1={best['f1']:.4f} "
        f"| thr={best['threshold']:.2f}"
    )

    # Courbe du F1-score selon le seuil de décision.
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        best["threshold_grid"],
        best["f1_grid"],
        color="#1d3557",
    )

    ax.axvline(
        best["threshold"],
        color="#e63946",
        linestyle="--",
        label=f"seuil={best['threshold']:.2f}",
    )

    ax.set_xlabel("Seuil")
    ax.set_ylabel("F1-score (classe 1)")
    ax.set_title(
        f"Optimisation du seuil — {best_name}"
    )
    ax.legend()

    save_figure(
        fig,
        "09_f1_vs_seuil.png",
    )
    plt.close(fig)

    # Génération des matrices de confusion.
    confusion_configurations = [
        (
            "0.5",
            best["metrics_05"],
            "10_confusion_seuil_05.png",
        ),
        (
            f"{best['threshold']:.2f}",
            best["metrics_opt"],
            "11_confusion_seuil_opt.png",
        ),
    ]

    for label, metrics, filename in confusion_configurations:
        confusion_matrix = np.array(
            metrics["confusion_matrix"]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Préd 0", "Préd 1"],
            yticklabels=["Réel 0", "Réel 1"],
        )

        ax.set_title(
            f"Matrice de confusion "
            f"(seuil={label})"
        )

        save_figure(fig, filename)
        plt.close(fig)

    # Création du tableau comparatif des modèles.
    rows = []

    for model_name, result in results.items():
        metrics = result["metrics_opt"]

        rows.append(
            {
                "modele": model_name,
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "seuil": metrics["threshold"],
                "temps_s": result["train_time_sec"],
                "f1_seuil_05": (
                    result["metrics_05"]["f1"]
                ),
            }
        )

    metrics_table = (
        pd.DataFrame(rows)
        .sort_values(
            "f1",
            ascending=False,
        )
    )

    metrics_table.to_csv(
        PROCESSED_DIR / "model_comparison.csv",
        index=False,
    )

    # Analyse des erreurs de classification.
    validation_errors = valid_split.copy()

    validation_errors["proba"] = (
        best["proba_valid"]
    )

    validation_errors["pred"] = (
        validation_errors["proba"]
        >= best["threshold"]
    ).astype(int)

    false_positives = validation_errors[
        (validation_errors[TARGET_COL] == 0)
        & (validation_errors["pred"] == 1)
    ].nlargest(5, "proba")

    false_negatives = validation_errors[
        (validation_errors[TARGET_COL] == 1)
        & (validation_errors["pred"] == 0)
    ].nsmallest(5, "proba")

    # Analyse des performances par région et destination.
    region_stats = []

    for column in [
        "region_hotel",
        "type_destination",
    ]:
        for key, group in validation_errors.groupby(column):
            if len(group) < 40:
                continue

            region_stats.append(
                {
                    "groupe": column,
                    "valeur": str(key),
                    "n": int(len(group)),
                    "f1": float(
                        f1_score(
                            group[TARGET_COL],
                            group["pred"],
                            zero_division=0,
                        )
                    ),
                    "cancel_rate": float(
                        group[TARGET_COL].mean()
                    ),
                }
            )

    pd.DataFrame(region_stats).to_csv(
        PROCESSED_DIR / "perf_sous_groupes.csv",
        index=False,
    )

    # Calcul de l'importance des variables.
    X_valid, y_valid = prepare_xy(valid_split)

    try:
        permutation_result = permutation_importance(
            best["pipeline"],
            X_valid,
            y_valid,
            n_repeats=5,
            random_state=42,
            scoring="f1",
            n_jobs=-1,
        )

        feature_names = list(
            best["pipeline"]
            .named_steps["prep"]
            .get_feature_names_out()
        )

        raw_names = list(X_valid.columns)

        importance = pd.DataFrame(
            {
                "feature": raw_names,
                "importance": (
                    permutation_result.importances_mean
                ),
            }
        )

        importance = importance.sort_values(
            "importance",
            ascending=False,
        )

        importance.to_csv(
            PROCESSED_DIR
            / "permutation_importance.csv",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        top_importance = (
            importance
            .head(15)
            .iloc[::-1]
        )

        ax.barh(
            top_importance["feature"],
            top_importance["importance"],
            color="#1d3557",
        )

        ax.set_title(
            "Permutation importance (F1) — top 15"
        )

        save_figure(
            fig,
            "12_permutation_importance.png",
        )
        plt.close(fig)

        top_features = (
            importance
            .head(20)
            .to_dict(orient="records")
        )

    except Exception as error:
        print(
            "Permutation importance échouée:",
            error,
        )

        top_features = []
        feature_names = []

    # Réentraînement du modèle final.
    print(
        "\n=== Réentraînement final "
        "sur train complet ==="
    )

    X_full, y_full = prepare_xy(train_fe)

    final_pipeline = rebuild_pipeline(
        best_name,
        comparison["num_cols"],
        comparison["cat_cols"],
        comparison["scale_pos_weight"],
    )

    final_pipeline.fit(
        X_full,
        y_full,
    )

    model_path = (
        MODELS_DIR / "modele_final.pkl"
    )

    joblib.dump(
        {
            "pipeline": final_pipeline,
            "threshold": best["threshold"],
            "model_name": best_name,
        },
        model_path,
    )

    print(
        f"Modèle sauvé : {model_path}"
    )

    # Rechargement du test dans son ordre original.
    test_ordered = pd.read_csv(
        RAW_DIR / "reservations_test.csv"
    )

    test_ordered["date_reservation"] = pd.to_datetime(
        test_ordered["date_reservation"]
    )

    test_ordered["date_arrivee"] = pd.to_datetime(
        test_ordered["date_arrivee"]
    )

    X_test = prepare_xy(
        engineer_features(test_ordered)
    )[0]

    probability_test = (
        final_pipeline
        .predict_proba(X_test)[:, 1]
    )

    prediction_test = (
        probability_test
        >= best["threshold"]
    ).astype(int)

    submission = pd.DataFrame(
        {
            "reservation_id": (
                test_ordered["reservation_id"]
            ),
            "probabilite_annulation": np.round(
                probability_test,
                6,
            ),
            "reservation_annulee": prediction_test,
        }
    )

    assert_submission(
        submission,
        test_ordered,
        RAW_DIR / "sample_submission.csv",
    )

    assert (
        submission["reservation_id"].tolist()
        == test_order_ids
    )

    submission_path = (
        PROJECT_ROOT / "submission.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    print(
        f"Submission : {submission_path} "
        f"({len(submission)} lignes)"
    )

    # Regroupement de tous les résultats.
    payload = {
        "eda": eda,
        "split": {
            "time_col": TIME_COL,
            "val_ratio": 0.20,
            "train_period": list(
                split.train_period
            ),
            "valid_period": list(
                split.valid_period
            ),
            "train_n": len(train_split),
            "valid_n": len(valid_split),
            "train_cancel_rate": (
                split.train_cancel_rate
            ),
            "valid_cancel_rate": (
                split.valid_cancel_rate
            ),
        },
        "ablation": ablation.to_dict(
            orient="records"
        ),
        "leakage_checklist": leakage_checklist(),
        "metrics_table": metrics_table.to_dict(
            orient="records"
        ),
        "best_model": best_name,
        "threshold": best["threshold"],
        "best_metrics": best["metrics_opt"],
        "baseline_lr": (
            results["logistic_regression"]
            ["metrics_opt"]
        ),
        "baseline_lr_05": (
            results["logistic_regression"]
            ["metrics_05"]
        ),
        "top_features": top_features,
        "false_positives": false_positives[
            [
                "reservation_id",
                "region_hotel",
                "type_destination",
                "canal_reservation",
                "tarif_remboursable",
                "type_acompte",
                "delai_reservation_jours",
                "client_type",
                "proba",
            ]
        ].to_dict(orient="records"),
        "false_negatives": false_negatives[
            [
                "reservation_id",
                "region_hotel",
                "type_destination",
                "canal_reservation",
                "tarif_remboursable",
                "type_acompte",
                "delai_reservation_jours",
                "client_type",
                "proba",
            ]
        ].to_dict(orient="records"),
        "region_stats": region_stats,
        "submission": {
            "n_rows": len(submission),
            "positive_rate": float(
                prediction_test.mean()
            ),
            "proba_min": float(
                probability_test.min()
            ),
            "proba_max": float(
                probability_test.max()
            ),
        },
    }

    results_path = (
        PROCESSED_DIR / "results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
            default=float,
        )

    print(
        "Résultats → "
        "data/processed/results.json"
    )

    return payload


if __name__ == "__main__":
    main()
