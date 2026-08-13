# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

Merci de lister tous les membres de l’équipe ayant effectivement participé au Hackathon.

#### Membre 1

- nom : ANDRIAMAHEFA
- prénom(s) : Ny Fetra Phanoël
- classe : IGGLIA 4
- numéro : 16
- rôle : Data Engineer & Chef de projet (structure dépôt, utils, intégration notebook, checklist)

#### Membre 2

- nom : ANDRIANTSOA
- prénom(s) : Velotiana Todisoa Angelo
- classe : IGGLIA 4
- numéro : 22
- rôle : Data Analyst — EDA (cible, manquants, scénarios d’annulation, figures)

#### Membre 3

- nom : ANDRIANARAHINJAKA
- prénom(s) : Yohannee Aintsoa
- classe : IGGLIA 4
- numéro : 54
- rôle : ML Engineer — baseline & validation temporelle (pipeline, logistic regression, split)

#### Membre 4

- nom : LAPORTE
- prénom(s) :  Hantaharimanana Marie Fabia
- classe : IGGLIA 4
- numéro : 53
- rôle : Feature Engineer (variables dérivées, anti-fuite, ablation F1)

#### Membre 5

- nom : RASOAMAHAZOMANANA
- prénom(s) :  Tsitoniaina Rogella
- classe : IGGLIA 4
- numéro : 15
- rôle : Lead Data Scientist (modèles alternatifs, seuil, sélection, submission)

#### Membre 6

- nom : NOMESAHANINA
- prénom(s) : Aiky
- classe : IGGLIA 4
- numéro : 35
- rôle : Analyste métier & rédacteur technique (README, Q1–Q9, recommandations)

#### Membre 7

- nom : RAKOTOARISOA
- prénom(s) : Fanaja Manoa Ny Avo
- classe : IGGLIA 4
- numéro : 32
- rôle : Analyse d’erreurs & communicant vidéo (FP/FN, script 3–5 min)

---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels subit des annulations qui laissent des chambres inoccupées et dégradent la planification. L’objectif est de prédire `reservation_annulee` suffisamment tôt pour déclencher une action proportionnée (confirmation, rappel), sans pénaliser automatiquement les clients susceptibles de maintenir leur séjour.

#### Méthodologie adoptée

EDA complète, holdout temporel strict sur `date_reservation` (80 % anciennes / 20 % plus récentes), pipeline scikit-learn (imputation + OneHot `handle_unknown="ignore"`), baseline régression logistique, feature engineering avec ablation, comparaison de cinq modèles, optimisation du seuil sur la validation uniquement, réentraînement final sur tout le train, génération de `submission.csv`.

#### Résultats obtenus

Sur la validation temporelle (2024-11-28 → 2025-05-24) :

- **Modèle final : Random Forest** — F1 = **0.4775**, précision = 0.3396, rappel = 0.8042, PR-AUC = 0.3548, ROC-AUC = 0.6409, **seuil = 0.33**
- **Baseline logistic regression** — F1 = **0.4759** (seuil 0.43), PR-AUC = 0.3932
- Constat clé : les conditions commerciales (acompte, tarif remboursable, canal) dominent le risque d’annulation ; le groupe de features **historique** apporte le meilleur gain d’ablation (+0.0022 F1).

#### Mots-clés

classification binaire, annulation hôtelière, validation temporelle, F1-score, feature engineering, optimisation de seuil, Random Forest, régression logistique

---

### **3. Contenu du Repository**

- **notebooks/notebook.ipynb** : livrable principal exécutable de bout en bout
- **notebooks/01_eda.ipynb … 06_interpretation_erreurs.ipynb** : notebooks de travail par rôle
- **src/** : `utils.py`, `validation.py`, `preprocessing.py`, `features.py`, `modeling.py`
- **submission.csv** : prédictions sur `reservations_test.csv` (2 000 lignes)
- **models/modele_final.pkl** : pipeline final + seuil
- **report/figures/** : graphiques EDA / seuil / confusion / importance
- **report/video_script.md** : script de présentation orale
- **requirements.txt**, **.gitignore**
- **data/raw/** : sources non modifiées (`reservations_train.csv`, `reservations_test.csv`, `data_dictionary.csv`)
- **scripts/run_pipeline.py** : pipeline batch reproductible

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION**](https://drive.google.com/file/d/1pKy7dsmJyHRQVWXqkMIkwZRIISPwsC9D/view?usp=sharing)
- [Lien vers le dépôt GitHub](https://github.com/FetraFex/atlantic-haven-ml-exam-s2)

---

### **4. Résultats de Modélisation**

Protocole unique : holdout temporel `date_reservation`, train = 6 400 lignes (2023-01-01 → 2024-11-28), valid = 1 600 lignes (2024-11-28 → 2025-05-24). Métriques au **seuil optimisé** (sauf mention contraire).

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline | class_weight=balanced, max_iter=2000, seuil=0.43 | 0.4759 | 0.3489 | 0.7483 | 0.6544 |
| Random Forest | n_estimators=400, max_depth=16, min_samples_leaf=4, seuil=0.33 | **0.4775** | 0.3396 | 0.8042 | 0.6409 |
| HistGradientBoosting | max_iter=350, lr=0.06, max_depth=8, seuil=0.21 | 0.4577 | 0.3166 | 0.8252 | 0.6269 |
| XGBoost | n_estimators=500, lr=0.05, max_depth=6, seuil=0.17 | 0.4583 | 0.3118 | 0.8648 | 0.6264 |
| LightGBM | n_estimators=500, num_leaves=48, lr=0.05, seuil=0.13 | 0.4558 | 0.3094 | 0.8648 | 0.6220 |
| **Modèle final = Random Forest** | idem + réentraîné sur 8 000 lignes | **0.4775** | 0.3396 | 0.8042 | 0.6409 |

**PR-AUC (complément) :** LR 0.3932 · RF 0.3548 · HGB 0.3591 · XGB 0.3542 · LGBM 0.3536

**Seuil de décision retenu :** **0.33** (optimisé sur la validation temporelle pour maximiser le F1 ; jamais sur le test).

**Justification du choix du modèle final :**

Le Random Forest obtient le meilleur F1 temporel (0.4775), légèrement au-dessus de la régression logistique (0.4759), avec un rappel plus élevé (0.8042 vs 0.7483). Cela réduit les faux négatifs (annulations non détectées), cohérent avec l’enjeu métier de protéger le remplissage. La LR reste une référence solide (meilleur PR-AUC et meilleure précision) et plus interprétable ; le RF est retenu pour le F1 demandé par le sujet, tout en gardant un seuil métier explicite.

---

### **5. Réponses aux Questions d’Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l’accuracy pour cette tâche ?**

La classe positive représente **25,84 %** des réservations train. Une accuracy élevée pourrait être obtenue en prédisant presque toujours « maintenue ». Le F1 de la classe 1 combine précision et rappel sur les annulations, ce qui mesure réellement la capacité à détecter correctement les annulations sans ignorer le coût des fausses alertes.

#### **Q2. Dans ce contexte, qu’est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

- **Faux positif :** client prédit comme annulant alors qu’il maintient — risque de relance intrusive ou de surbooking défensif injustifié.
- **Faux négatif :** annulation non détectée — chambre potentiellement vide, perte de revenu et de planification.

Les deux sont coûteux. Une position nuancée : le **faux négatif** est souvent plus grave pour le remplissage, d’où un seuil abaissé (0.33) favorisant le rappel (0.80). Mais un trop grand volume de FP dégrade l’expérience client : d’où l’usage de **seuils gradués** (surveillance / confirmation) plutôt qu’une pénalisation automatique.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

Ablation (F1 LR, même split temporel) :

| Config | F1 | Δ F1 vs sans FE |
|---|---:|---:|
| sans_fe | 0.4736 | 0.0000 |
| groupe_dates | 0.4710 | −0.0026 |
| groupe_sejour | 0.4712 | −0.0024 |
| groupe_prix | 0.4711 | −0.0025 |
| **groupe_historique** | **0.4758** | **+0.0022** |
| groupe_interactions | 0.4730 | −0.0006 |
| toutes_fe | 0.4759 | +0.0023 |

Le groupe **historique** (`taux_annulation_passe`, `client_jamais_reserve`, `a_deja_annule`) apporte le gain le plus clair. Les interactions commerciales (`acompte_x_delai`, `lead_x_remboursable`, `sans_acompte`) apparaissent aussi dans la permutation importance du modèle final. Les gains restent modestes car beaucoup d’information est déjà dans les variables brutes (acompte, remboursabilité, délai).

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Les fichiers sont ordonnés dans le temps : le test couvre des réservations **plus récentes** (2025-05-24 → 2025-12-31) que le train (2023-01-01 → 2025-05-24). Un split aléatoire mélange passé et futur, fuit des régimes temporels et surestime les performances.

**Protocole retenu :**

- colonne : `date_reservation` (moment où la réservation existe réellement)
- tri chronologique, **sans shuffle**
- holdout 80/20 : train **2023-01-01 → 2024-11-28** (6 400, taux 25,59 %) ; valid **2024-11-28 → 2025-05-24** (1 600, taux 26,81 %)
- prétraitements fit **uniquement** sur le train de chaque étape
- le fichier test n’a servi ni au tuning ni au seuil

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- **Tarif remboursable = oui** : taux d’annulation **31,3 %** vs **14,0 %** si non remboursable.
- **Acompte = aucun** : **34,0 %** vs **10,4 %** pour acompte total.
- **Canal plateforme_en_ligne** : **30,4 %** (vs 14,5 % canal entreprise).
- **Combinaison fréquente à risque :** délai long + tarif flexible + sans acompte + canal OTA (profil dominant des faux positifs à haute proba).

Ces scénarios décrivent des **conditions commerciales et opérationnelles**, non des populations intrinsèques.

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l’entraînement ?**

- Numériques (`enfants`, `prix_moyen_nuit_eur`, `demandes_speciales`, …) : `SimpleImputer(strategy="median")` (+ `StandardScaler` pour la LR).
- Catégorielles : `SimpleImputer(strategy="most_frequent")` puis `OneHotEncoder(handle_unknown="ignore", min_frequency=10)`.
- `agent_id` : retiré (haute cardinalité) au profit du flag `agent_manquant`.
- Catégorie vue au test mais absente du train : ex. `canal_reservation = assistant_vocal` — encodée à zéro grâce à `handle_unknown="ignore"`.
- Fit exclusivement sur le train du split (aucune statistique globale avant split).

#### **Q7. Selon vous, quelle action l’hôtel devrait-il entreprendre lorsqu’une réservation en cours présente une forte probabilité d’annulation ?**

Action **proportionnée** selon la probabilité :

1. **Zone haute (ex. p ≥ 0.50)** : contact humain ou message de confirmation douce, proposition d’option flexible / avantage fidélité, **sans annuler** ni surtaxer automatiquement.
2. **Zone intermédiaire (seuil métier ~0.33–0.50)** : suivi CRM léger et monitoring du remplissage.
3. Ne jamais transformer la proba en sanction automatique du client.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Non parfaitement. Exemples sur la validation (n ≥ 40) :

| Région | n | F1 | Taux annulation |
|---|---:|---:|---:|
| Lombardia | 216 | 0.505 | 0.278 |
| Liguria | 127 | 0.482 | 0.268 |
| Lazio | 258 | 0.444 | 0.229 |
| Campania | 169 | 0.403 | 0.231 |

Les écarts peuvent refléter des mélanges de canaux/tarifs différents et la **taille limitée** de certains sous-groupes : les F1 régionaux sont moins stables que le F1 global.

#### **Q9. Analyse des erreurs**

**Faux positifs (exemples) :** R005310, R000682, R001206, R002897, R006211 — souvent tarif remboursable, acompte aucun, délai long, canal OTA/agence. Le modèle voit un profil « à risque commercial » alors que le séjour a été maintenu.

**Faux négatifs (exemples) :** R009204, R001185, R001223, R001479, R002592 — souvent tarif non remboursable et/ou acompte total/partiel, canal site hôtel/entreprise : signaux de « engagement » qui masquent une annulation réelle.

**Pistes d’amélioration :** features de comportement post-réservation autorisées (ouvertures d’e-mails, modifications tardives horodatées), calibration des probabilités, seuils différenciés par canal, et enrichissement météo/événements locaux horodatés sans fuite.

---

### **6. Conclusion et Recommandations**

Le pipeline temporel produit un F1 de **0.4775** (Random Forest, seuil 0.33) sur des données plus récentes, proche de la baseline LR (0.4759). Les leviers métier les plus clairs sont l’**acompte**, la **remboursabilité** et le **canal**. Le modèle est utile comme **score de priorisation**, pas comme décision automatique.

**Recommandation opérationnelle finale :**

Déployer le score en trois zones (surveillance / confirmation douce / suivi renforcé), mesurer l’impact sur le taux de no-show et la satisfaction client, et recalibrer périodiquement le seuil sur une fenêtre temporelle glissante.

---

### **7. Reproductibilité**

- version de Python : 3.12
- principales bibliothèques : pandas 3.0.5, scikit-learn 1.7.2, matplotlib 3.11, seaborn 0.13, lightgbm 4.7, xgboost (si installé), joblib, nbconvert
- graine(s) aléatoire(s) : **RANDOM_STATE = 42**
- commande d’installation :
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- commande d’exécution du notebook :
  ```bash
  cd atlantic-haven-hotels
  .venv/bin/jupyter nbconvert --to notebook --execute notebooks/notebook.ipynb --output notebook_executed.ipynb
  ```
  ou pipeline batch :
  ```bash
  .venv/bin/python scripts/run_pipeline.py
  ```
- durée approximative d’entraînement : ~3–5 minutes (comparaison multi-modèles + permutation importance)
- environnement utilisé : local (Linux)

---

### **8. Bibliographie**

- Documentation scikit-learn — Pipelines, OneHotEncoder, RandomForest, métriques
- Documentation pandas — time series / groupby
- Sujet officiel ISPM — Atlantic Haven Hotels (examen S2)
- Document de répartition des tâches (équipe de 7)
- Outil d’IA générative (Cursor / Composer) : aide à la structuration du dépôt, rédaction du README et génération des notebooks ; tous les scores et analyses proviennent d’exécutions réelles du code sur les CSV fournis
