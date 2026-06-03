<<<<<<< HEAD
# TP IA - Agriculture au Burundi

Ce projet repond au TP de prediction des bonnes et mauvaises recoltes au Burundi a partir du fichier `agriculture_burundi.csv`.

## Contenu

- `agriculture_burundi.csv` : dataset du TP.
- `TP_Agriculture_Burundi.pdf` : enonce.
- `src/train_models.py` : pipeline complet d'analyse, preprocessing, entrainement, evaluation et sauvegarde.
- `models/` : modeles `.pkl` generes.
- `figures/` : visualisations demandees dans le TP.
- `reports/` : tableaux de resultats, predictions de scenarios et rapport court.
- `app.py` : application web Streamlit.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Entrainer les modeles

```powershell
.\.venv\Scripts\python.exe src\train_models.py
```

Cette commande cree :

- `models/decision_tree.pkl`
- `models/random_forest.pkl`
- `models/logistic_regression.pkl`
- `models/artifacts.pkl`
- les figures dans `figures/`
- les resultats dans `reports/`

## Lancer l'application web

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

L'application contient maintenant :

- tableau de bord d'accueil avec KPI, meilleur modele et dernier historique ;
- themes clair/sombre ;
- prediction individuelle avec choix du modele ;
- comparaison instantanee des 3 modeles ;
- score de consensus entre les modeles ;
- niveau de risque et recommandations agronomiques selon les conditions saisies ;
- analyse de sensibilite locale sur la prediction ;
- simulation `what-if` sur la pluie, la temperature, les engrais et l'irrigation ;
- courbe de sensibilite a la pluviometrie ;
- historique persistant des predictions dans `reports/prediction_history.csv` ;
- export CSV et rapport HTML d'une prediction ;
- prediction par fichier CSV ;
- scenarios du TP precharges ;
- metriques, matrices de confusion, courbes ROC et graphiques de donnees.

## Choix methodologiques

Les colonnes `rendement_t_ha` et `production_totale_t` sont exclues des variables d'entree pour eviter le data leakage : la cible `bonne_recolte` est definie a partir du rendement, et la production totale depend directement du rendement.

Les lignes sans `bonne_recolte` sont supprimees pour l'entrainement. Les autres valeurs manquantes sont traitees dans les pipelines par imputation : mediane pour les variables numeriques continues et valeur la plus frequente pour les variables binaires/categorielles. Les variables categorielles sont encodees par One-Hot Encoding avec `drop='first'`.

La division train/test utilise `stratify=y` et `random_state=42` pour conserver la distribution des classes et rendre les resultats reproductibles.
=======
# AIProject
projet de prediction des revenus des recoltes de l'agriculture au burundi
>>>>>>> 1bc643a0bcda25fe382d9265a87f41094b52fb35
