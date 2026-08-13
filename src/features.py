"""Feature engineering sans fuite de cible."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Groupes pour l'ablation expérimentale
FEATURE_GROUPS: dict[str, list[str]] = {
    "dates": [
        "annee_reservation",
        "mois_reservation",
        "mois_arrivee",
        "trimestre_arrivee",
        "saison_arrivee",
        "jour_semaine_arrivee",
        "arrivee_weekend_calc",
        "delai_calcule",
        "delai_court",
        "delai_long",
    ],
    "sejour": [
        "taille_groupe",
        "enfants_presents",
        "personnes_par_chambre",
        "nuits_par_chambre",
        "sejour_long",
    ],
    "prix": [
        "prix_par_personne",
        "prix_par_nuit_calc",
        "montant_apres_remise",
        "remise_forte",
        "tarif_flexible",
        "sans_acompte",
    ],
    "historique": [
        "taux_annulation_passe",
        "client_jamais_reserve",
        "a_deja_annule",
    ],
    "interactions": [
        "lead_x_remboursable",
        "saison_x_nuits",
        "nouveau_x_plateforme",
        "acompte_x_delai",
        "engagement_score",
        "agent_manquant",
    ],
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables dérivées à partir des colonnes présentes uniquement.

    Règles anti-fuite :
    - aucune agrégation globale apprise hors split ;
    - aucune utilisation de reservation_annulee ;
    - historique = ratio row-wise à partir de colonnes déjà disponibles
      (annulations_passees / reservations_passees), sans la cible courante.
    """
    out = df.copy()

    # --- Dates / calendrier ---
    out["annee_reservation"] = out["date_reservation"].dt.year
    out["mois_reservation"] = out["date_reservation"].dt.month
    out["mois_arrivee"] = out["date_arrivee"].dt.month
    out["trimestre_arrivee"] = out["date_arrivee"].dt.quarter
    out["jour_semaine_arrivee"] = out["date_arrivee"].dt.dayofweek
    out["arrivee_weekend_calc"] = (out["jour_semaine_arrivee"] >= 5).astype(int)
    # saison météo simple (hémisphère nord)
    out["saison_arrivee"] = out["mois_arrivee"].map(
        {
            12: "hiver",
            1: "hiver",
            2: "hiver",
            3: "printemps",
            4: "printemps",
            5: "printemps",
            6: "ete",
            7: "ete",
            8: "ete",
            9: "automne",
            10: "automne",
            11: "automne",
        }
    )

    out["delai_calcule"] = (out["date_arrivee"] - out["date_reservation"]).dt.days
    out["delai_court"] = (out["delai_reservation_jours"] <= 7).astype(int)
    out["delai_long"] = (out["delai_reservation_jours"] >= 90).astype(int)

    # --- Séjour ---
    enfants = out["enfants"].fillna(0)
    out["taille_groupe"] = out["adultes"].fillna(0) + enfants
    out["enfants_presents"] = (enfants > 0).astype(int)
    chambres = out["chambres"].replace(0, np.nan)
    out["personnes_par_chambre"] = out["taille_groupe"] / chambres
    out["nuits_par_chambre"] = out["nuits"] / chambres
    out["sejour_long"] = (out["nuits"] >= 7).astype(int)

    # --- Prix ---
    taille = out["taille_groupe"].replace(0, np.nan)
    out["prix_par_personne"] = out["montant_total_eur"] / taille
    out["prix_par_nuit_calc"] = out["montant_total_eur"] / out["nuits"].replace(0, np.nan)
    out["montant_apres_remise"] = out["montant_total_eur"] * (1 - out["remise_pct"].fillna(0) / 100.0)
    out["remise_forte"] = (out["remise_pct"].fillna(0) >= 10).astype(int)
    out["tarif_flexible"] = (
        out["tarif_remboursable"].astype(str).str.lower().eq("oui").astype(int)
    )
    out["sans_acompte"] = out["type_acompte"].astype(str).str.lower().eq("aucun").astype(int)

    # --- Historique client (row-wise, sans cible) ---
    past = out["reservations_passees"].fillna(0)
    canc = out["annulations_passees"].fillna(0)
    out["taux_annulation_passe"] = np.where(past > 0, canc / past, 0.0)
    out["client_jamais_reserve"] = (past == 0).astype(int)
    out["a_deja_annule"] = (canc > 0).astype(int)

    # --- Interactions / friction ---
    out["lead_x_remboursable"] = out["delai_reservation_jours"] * out["tarif_flexible"]
    out["saison_x_nuits"] = out["haute_saison_regionale"].fillna(0) * out["nuits"]
    out["nouveau_x_plateforme"] = (
        out["client_type"].astype(str).str.lower().eq("nouveau").astype(int)
        * out["canal_reservation"]
        .astype(str)
        .str.contains("plateforme", case=False, na=False)
        .astype(int)
    )
    out["acompte_x_delai"] = out["sans_acompte"] * out["delai_reservation_jours"]
    out["engagement_score"] = (
        out["demandes_speciales"].fillna(0)
        + out["modifications_reservation"].fillna(0)
        - out["jours_liste_attente"].fillna(0)
    )
    agent = out["agent_id"]
    out["agent_manquant"] = (
        agent.isna() | (agent.astype(str).str.strip() == "") | (agent.astype(str) == "nan")
    ).astype(int)

    return out


def leakage_checklist() -> list[dict]:
    """Documentation explicite anti-fuite pour chaque groupe de features."""
    rows = []
    for group, feats in FEATURE_GROUPS.items():
        rows.append(
            {
                "groupe": group,
                "variables": ", ".join(feats),
                "utilise_cible": False,
                "utilise_futur": False,
                "commentaire": (
                    "Calculées uniquement à partir des colonnes disponibles au moment "
                    "de la réservation / sur la ligne courante."
                ),
            }
        )
    return rows


def drop_non_feature_columns(df: pd.DataFrame, drop_high_card: bool = True) -> pd.DataFrame:
    """Retire identifiants, dates brutes et cible."""
    drop_cols = {
        "reservation_id",
        "date_reservation",
        "date_arrivee",
        "reservation_annulee",
    }
    if drop_high_card:
        drop_cols.add("agent_id")  # cardinalité élevée ; flag agent_manquant conserve l'info
    return df.drop(columns=[c for c in drop_cols if c in df.columns])
