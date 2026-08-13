#!/usr/bin/env python3
"""Pipeline complet Atlantic Haven Hotels — génère métriques, figures et submission."""

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
    eda["missing_train"] = train.isna().sum()[train.isna().sum() > 0].astype(int).to_dict()
    # agent_id empty strings
    agent_empty = int(
        (train["agent_id"].isna() | (train["agent_id"].astype(str).str.strip() == "")).sum()
    )
    eda["agent_id_empty_train"] = agent_empty
    eda["dup_id_train"] = int(train["reservation_id"].duplicated().sum())
    eda["dup_id_test"] = int(test["reservation_id"].duplicated().sum())

    # Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train[TARGET_COL].value_counts().sort_index()
    ax.bar(["Maintenue (0)", "Annulée (1)"], counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title(f"Répartition de la cible (taux={eda['cancel_rate']:.1%})")
    ax.set_ylabel("Nombre de réservations")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, str(v), ha="center")
    save_figure(fig, "01_cible_distribution.png")
    plt.close(fig)

    # Temporal cancellation rate
    tmp = train.copy()
    tmp["mois"] = tmp[TIME_COL].dt.to_period("M").astype(str)
    monthly = tmp.groupby("mois")[TARGET_COL].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color="#264653")
    ax.set_title("Évolution temporelle du taux d'annulation (train)")
    ax.set_xlabel("Mois de réservation")
    ax.set_ylabel("Taux d'annulation")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "02_taux_annulation_temporel.png")
    plt.close(fig)

    # Cancellation by key categoricals
    for col, fname in [
        ("canal_reservation", "03_annulation_par_canal.png"),
        ("type_acompte", "04_annulation_par_acompte.png"),
        ("tarif_remboursable", "05_annulation_tarif_remboursable.png"),
        ("type_destination", "06_annulation_type_destination.png"),
        ("client_type", "07_annulation_client_type.png"),
    ]:
        rates = train.groupby(col)[TARGET_COL].agg(["mean", "count"]).sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(rates.index.astype(str), rates["mean"], color="#457b9d")
        ax.set_xlabel("Taux d'annulation")
        ax.set_title(f"Taux d'annulation selon {col}")
        save_figure(fig, fname)
        plt.close(fig)
        eda[f"rate_by_{col}"] = {
            str(k): {"rate": float(v["mean"]), "n": int(v["count"])}
            for k, v in rates.iterrows()
        }

    # Lead time vs cancel
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=train,
        x=TARGET_COL,
        y="delai_reservation_jours",
        ax=ax,
        palette=["#2a9d8f", "#e76f51"],
    )
    ax.set_xticklabels(["Maintenue", "Annulée"])
    ax.set_title("Délai de réservation selon la cible")
    save_figure(fig, "08_delai_vs_cible.png")
    plt.close(fig)

    return eda


def ablation_study(train_fe: pd.DataFrame, valid_fe: pd.DataFrame) -> pd.DataFrame:
    """Mesure le gain F1 de chaque groupe de features vs baseline sans FE."""
    from src.modeling import build_logistic_pipeline, fit_and_score

    base_cols = [
        c
        for c in train_fe.columns
        if c
        not in {
            "reservation_id",
            "date_reservation",
            "date_arrivee",
            TARGET_COL,
            "agent_id",
        }
        and c not in sum(FEATURE_GROUPS.values(), [])
    ]

    rows = []
    configs = {"sans_fe": base_cols}
    for g, feats in FEATURE_GROUPS.items():
        configs[f"groupe_{g}"] = base_cols + [f for f in feats if f in train_fe.columns]
    configs["toutes_fe"] = base_cols + [
        f for f in sum(FEATURE_GROUPS.values(), []) if f in train_fe.columns
    ]

    for name, cols in configs.items():
        use_cols = [c for c in cols if c in train_fe.columns]
        X_tr = train_fe[use_cols]
        X_va = valid_fe[use_cols]
        y_tr = train_fe[TARGET_COL].astype(int)
        y_va = valid_fe[TARGET_COL].astype(int)
        # rebuild column types from this subset
        tmp = train_fe[use_cols + [TARGET_COL]]
        num_cols, cat_cols = get_feature_columns(tmp)
        pipe = build_logistic_pipeline(num_cols, cat_cols)
        res = fit_and_score(pipe, X_tr, y_tr, X_va, y_va, name)
        rows.append(
            {
                "config": name,
                "n_features": len(use_cols),
                "f1": res["f1"],
                "precision": res["metrics_opt"]["precision"],
                "recall": res["metrics_opt"]["recall"],
                "threshold": res["threshold"],
            }
        )
        print(f"Ablation {name}: F1={res['f1']:.4f}")

    table = pd.DataFrame(rows)
    base_f1 = float(table.loc[table["config"] == "sans_fe", "f1"].iloc[0])
    table["delta_f1"] = table["f1"] - base_f1
    table.to_csv(PROCESSED_DIR / "ablation_features.csv", index=False)
    return table


def main() -> dict:
    set_seed()
    ensure_dirs()

    print("=== Chargement ===")
    train_raw, test_raw = load_raw_data(parse_dates=True, sort_train=False)
    # conserver ordre test original
    test_order_ids = test_raw["reservation_id"].tolist()

    print(f"Train {train_raw.shape} | Test {test_raw.shape}")
    eda = run_eda(train_raw, test_raw)

    print("\n=== Feature engineering ===")
    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)

    print("\n=== Split temporel ===")
    split = temporal_holdout(train_fe, time_col=TIME_COL, val_ratio=0.20, verbose=True)
    tr, va = split.train, split.valid

    print("\n=== Ablation FE (régression logistique) ===")
    ablation = ablation_study(tr, va)

    print("\n=== Comparaison modèles ===")
    comparison = compare_models(tr, va)
    results = comparison["results"]
    best_name = comparison["best_name"]
    best = results[best_name]
    print(f"\n>>> Modèle sélectionné : {best_name} | F1={best['f1']:.4f} | thr={best['threshold']:.2f}")

    # Courbe F1 vs seuil
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(best["threshold_grid"], best["f1_grid"], color="#1d3557")
    ax.axvline(best["threshold"], color="#e63946", linestyle="--", label=f"seuil={best['threshold']:.2f}")
    ax.set_xlabel("Seuil")
    ax.set_ylabel("F1-score (classe 1)")
    ax.set_title(f"Optimisation du seuil — {best_name}")
    ax.legend()
    save_figure(fig, "09_f1_vs_seuil.png")
    plt.close(fig)

    # Matrices de confusion
    for label, metrics, fname in [
        ("0.5", best["metrics_05"], "10_confusion_seuil_05.png"),
        (f"{best['threshold']:.2f}", best["metrics_opt"], "11_confusion_seuil_opt.png"),
    ]:
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Préd 0", "Préd 1"], yticklabels=["Réel 0", "Réel 1"])
        ax.set_title(f"Matrice de confusion (seuil={label})")
        save_figure(fig, fname)
        plt.close(fig)

    # Tableau comparatif
    rows = []
    for name, r in results.items():
        m = r["metrics_opt"]
        rows.append(
            {
                "modele": name,
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "seuil": m["threshold"],
                "temps_s": r["train_time_sec"],
                "f1_seuil_05": r["metrics_05"]["f1"],
            }
        )
    metrics_table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics_table.to_csv(PROCESSED_DIR / "model_comparison.csv", index=False)

    # Erreurs
    va_err = va.copy()
    va_err["proba"] = best["proba_valid"]
    va_err["pred"] = (va_err["proba"] >= best["threshold"]).astype(int)
    fp = va_err[(va_err[TARGET_COL] == 0) & (va_err["pred"] == 1)].nlargest(5, "proba")
    fn = va_err[(va_err[TARGET_COL] == 1) & (va_err["pred"] == 0)].nsmallest(5, "proba")

    # Perf par région / destination
    region_stats = []
    for col in ["region_hotel", "type_destination"]:
        for key, g in va_err.groupby(col):
            if len(g) < 40:
                continue
            region_stats.append(
                {
                    "groupe": col,
                    "valeur": str(key),
                    "n": int(len(g)),
                    "f1": float(f1_score(g[TARGET_COL], g["pred"], zero_division=0)),
                    "cancel_rate": float(g[TARGET_COL].mean()),
                }
            )
    pd.DataFrame(region_stats).to_csv(PROCESSED_DIR / "perf_sous_groupes.csv", index=False)


    X_va, y_va = prepare_xy(va)
    try:
        pi = permutation_importance(
            best["pipeline"],
            X_va,
            y_va,
            n_repeats=5,
            random_state=42,
            scoring="f1",
            n_jobs=-1,
        )
        feat_names = list(best["pipeline"].named_steps["prep"].get_feature_names_out())
        # permutation on raw X columns
        raw_names = list(X_va.columns)
        imp = pd.DataFrame({"feature": raw_names, "importance": pi.importances_mean})
        imp = imp.sort_values("importance", ascending=False)
        imp.to_csv(PROCESSED_DIR / "permutation_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        top = imp.head(15).iloc[::-1]
        ax.barh(top["feature"], top["importance"], color="#1d3557")
        ax.set_title("Permutation importance (F1) — top 15")
        save_figure(fig, "12_permutation_importance.png")
        plt.close(fig)
        top_features = imp.head(20).to_dict(orient="records")
    except Exception as e:
        print("Permutation importance échouée:", e)
        top_features = []
        feat_names = []

    # Réentraînement final
    print("\n=== Réentraînement final sur train complet ===")
    X_full, y_full = prepare_xy(train_fe)
    final_pipe = rebuild_pipeline(
        best_name,
        comparison["num_cols"],
        comparison["cat_cols"],
        comparison["scale_pos_weight"],
    )
    final_pipe.fit(X_full, y_full)
    model_path = MODELS_DIR / "modele_final.pkl"
    joblib.dump(
        {"pipeline": final_pipe, "threshold": best["threshold"], "model_name": best_name},
        model_path,
    )
    print(f"Modèle sauvé : {model_path}")

    # Submission — ordre original du test
    test_ordered = pd.read_csv(RAW_DIR / "reservations_test.csv")
    test_ordered["date_reservation"] = pd.to_datetime(test_ordered["date_reservation"])
    test_ordered["date_arrivee"] = pd.to_datetime(test_ordered["date_arrivee"])
    X_test = prepare_xy(engineer_features(test_ordered))[0]
    proba_test = final_pipe.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= best["threshold"]).astype(int)
    submission = pd.DataFrame(
        {
            "reservation_id": test_ordered["reservation_id"],
            "probabilite_annulation": np.round(proba_test, 6),
            "reservation_annulee": pred_test,
        }
    )
    assert_submission(submission, test_ordered, RAW_DIR / "sample_submission.csv")
    assert submission["reservation_id"].tolist() == test_order_ids
    sub_path = PROJECT_ROOT / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission : {sub_path} ({len(submission)} lignes)")

    payload = {
        "eda": eda,
        "split": {
            "time_col": TIME_COL,
            "val_ratio": 0.20,
            "train_period": list(split.train_period),
            "valid_period": list(split.valid_period),
            "train_n": len(tr),
            "valid_n": len(va),
            "train_cancel_rate": split.train_cancel_rate,
            "valid_cancel_rate": split.valid_cancel_rate,
        },
        "ablation": ablation.to_dict(orient="records"),
        "leakage_checklist": leakage_checklist(),
        "metrics_table": metrics_table.to_dict(orient="records"),
        "best_model": best_name,
        "threshold": best["threshold"],
        "best_metrics": best["metrics_opt"],
        "baseline_lr": results["logistic_regression"]["metrics_opt"],
        "baseline_lr_05": results["logistic_regression"]["metrics_05"],
        "top_features": top_features,
        "false_positives": fp[
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
        "false_negatives": fn[
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
            "positive_rate": float(pred_test.mean()),
            "proba_min": float(proba_test.min()),
            "proba_max": float(proba_test.max()),
        },
    }
    with open(PROCESSED_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    print("Résultats → data/processed/results.json")
    return payload


if __name__ == "__main__":
    main()
