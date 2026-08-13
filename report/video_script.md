# Script vidéo — Atlantic Haven Hotels (3 à 5 minutes)

Durée cible : **~4 minutes**. Les chiffres ci-dessous sont alignés sur `notebooks/notebook.ipynb` et `README.md`.

---

## 0:00–0:30 — Présentation de l’équipe

Bonjour, nous sommes l’équipe Atlantic Haven Hotels, Master 1 Machine Learning & Data Science à l’ISPM.

Sept rôles :

1. Data Engineer & chef de projet  
2. Data Analyst — EDA  
3. ML Engineer — baseline et validation temporelle  
4. Feature Engineer  
5. Lead Data Scientist — modèles et seuil  
6. Analyste métier & rédaction  
7. Analyse d’erreurs & réalisation vidéo  

---

## 0:30–1:00 — Problème métier

Atlantic Haven Hotels doit anticiper les annulations (`reservation_annulee = 1`) pour protéger le remplissage, sans pénaliser automatiquement les clients qui maintiendront leur séjour.

Le modèle fournit une **probabilité** et une **décision binaire** au seuil choisi. La métrique principale est le **F1 de la classe annulation**.

---

## 1:00–1:45 — Constats EDA

- Train : **8 000** réservations ; test : **2 000** (plus récentes).  
- Taux d’annulation global train : **25,8 %**.  
- Scénarios fréquents d’annulation :
  - tarif **remboursable** : 31,3 % vs 14,0 % ;
  - **sans acompte** : 34,0 % vs 10,4 % (acompte total) ;
  - canal **plateforme en ligne** : 30,4 %.  
*(Montrer les figures `03`–`05` et la courbe temporelle `02`.)*

---

## 1:45–2:15 — Validation temporelle

Pas de split aléatoire : le test est plus récent.

- Colonne : **`date_reservation`**  
- Holdout 80/20 :
  - train : **2023-01-01 → 2024-11-28** (6 400)  
  - valid : **2024-11-28 → 2025-05-24** (1 600)  
Prétraitements appris uniquement sur le train. Le fichier test n’entre jamais dans le tuning.

---

## 2:15–2:45 — Baseline et feature engineering

Baseline **régression logistique** : F1 = **0.4759** (seuil 0.43).

Feature engineering (délai, saison, prix/personne, historique client, interactions acompte×délai).  
Ablation : le groupe **historique** apporte le meilleur gain (**+0.0022** F1).

---

## 2:45–3:30 — Modèles, modèle final, seuil

Comparaison : Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM.

**Modèle final : Random Forest**  
- F1 = **0.4775**  
- précision = 0.34 · rappel = 0.80  
- **seuil optimal = 0.33** (grille sur validation, pas sur le test)

*(Montrer courbe F1 vs seuil et matrices de confusion 0.5 vs 0.33.)*

---

## 3:30–4:15 — Erreurs et recommandation métier

- **Faux positif** : client prédit annulant alors qu’il reste — souvent profil « flexible / sans acompte / OTA ».  
- **Faux négatif** : annulation réelle non détectée — souvent acompte élevé / tarif non remboursable.

Recommandation : utiliser la proba en **zones** (surveillance → confirmation douce → suivi renforcé), **jamais** comme annulation automatique du client.

---

## 4:15–4:30 — Conclusion

Solution reproductible, validation temporelle respectée, `submission.csv` de 2 000 lignes généré.  
Merci pour votre attention.

---

## Notes de tournage

- Préparer les slides/figures : `report/figures/01_cible_distribution.png`, `02_taux_annulation_temporel.png`, `04_annulation_par_acompte.png`, `09_f1_vs_seuil.png`, `11_confusion_seuil_opt.png`, `12_permutation_importance.png`.
- Enregistrer dès que le modèle est figé ; vérifier le lien public avant le commit final.
- **À faire manuellement :** enregistrement et publication de la vidéo (Drive/YouTube), puis mise à jour du lien dans `README.md`.
