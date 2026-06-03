from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "agriculture_burundi.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42

TARGET = "bonne_recolte"
LEAKAGE_COLUMNS = ["rendement_t_ha", "production_totale_t"]
CATEGORICAL_FEATURES = ["saison", "province", "culture"]
CONTINUOUS_FEATURES = [
    "annee",
    "altitude_m",
    "pluviometrie_mm",
    "temperature_moy_C",
    "superficie_ha",
    "nb_menages",
]
BINARY_FEATURES = ["utilisation_engrais", "acces_irrigation"]
FEATURES = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES + BINARY_FEATURES


def ensure_dirs() -> None:
    for directory in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="utf-8")


def clean_for_training(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean[TARGET] = clean[TARGET].astype("float")
    clean = clean.dropna(subset=[TARGET])
    clean[TARGET] = clean[TARGET].astype(int)
    return clean


def build_preprocessor(scale_continuous: bool) -> ColumnTransformer:
    if scale_continuous:
        continuous_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        continuous_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )

    binary_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    tree_preprocessor = build_preprocessor(scale_continuous=False)
    logistic_preprocessor = build_preprocessor(scale_continuous=True)

    return {
        "decision_tree": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=4,
                        criterion="gini",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", logistic_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def model_label(model_key: str) -> str:
    return {
        "decision_tree": "Arbre de decision",
        "random_forest": "Foret aleatoire",
        "logistic_regression": "Regression logistique",
    }[model_key]


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out()


def positive_probability(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]


def evaluate_model(
    model_key: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = positive_probability(pipeline, X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "model_key": model_key,
        "model": model_label(model_key),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "auc": auc(fpr, tpr),
        "confusion_matrix": cm.tolist(),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "classification_report": report,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
    }


def save_data_quality_reports(df: pd.DataFrame, clean: pd.DataFrame) -> dict:
    missing = pd.DataFrame(
        {
            "colonne": df.columns,
            "nb_manquants": df.isna().sum().values,
            "pct_manquants": (df.isna().mean().values * 100).round(2),
        }
    )
    missing.to_csv(REPORTS_DIR / "valeurs_manquantes.csv", index=False)

    numeric_stats = df.describe().T
    numeric_stats.to_csv(REPORTS_DIR / "statistiques_numeriques.csv")

    target_distribution = (
        df[TARGET]
        .value_counts(dropna=False)
        .rename_axis("classe")
        .reset_index(name="effectif")
    )
    target_distribution["pourcentage"] = (
        target_distribution["effectif"] / len(df) * 100
    ).round(2)
    target_distribution.to_csv(REPORTS_DIR / "distribution_cible.csv", index=False)

    yield_by_culture = (
        df.groupby("culture", dropna=False)["rendement_t_ha"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    yield_by_culture.to_csv(REPORTS_DIR / "rendement_moyen_par_culture.csv", index=False)

    rain_by_province = (
        df.groupby("province", dropna=False)["pluviometrie_mm"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    rain_by_province.to_csv(REPORTS_DIR / "pluviometrie_moyenne_par_province.csv", index=False)

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "trainable_shape_after_drop_missing_target": [
            int(clean.shape[0]),
            int(clean.shape[1]),
        ],
        "years": [int(df["annee"].min()), int(df["annee"].max())],
        "n_provinces": int(df["province"].nunique()),
        "provinces": sorted(df["province"].dropna().unique().tolist()),
        "n_cultures": int(df["culture"].nunique()),
        "cultures": sorted(df["culture"].dropna().unique().tolist()),
        "missing": missing.to_dict(orient="records"),
        "target_distribution": target_distribution.to_dict(orient="records"),
        "highest_mean_yield_culture": str(yield_by_culture.iloc[0]["culture"]),
        "lowest_mean_yield_culture": str(yield_by_culture.iloc[-1]["culture"]),
        "highest_mean_rain_province": str(rain_by_province.iloc[0]["province"]),
    }


def save_exploration_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="culture", y="rendement_t_ha")
    plt.title("Distribution du rendement par culture")
    plt.xlabel("Culture")
    plt.ylabel("Rendement (t/ha)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplot_rendement_par_culture.png", dpi=180)
    plt.close()

    yearly_production = (
        df.groupby("annee", as_index=False)["production_totale_t"].sum()
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=yearly_production, x="annee", y="production_totale_t", marker="o")
    plt.title("Evolution de la production totale par annee")
    plt.xlabel("Annee")
    plt.ylabel("Production totale (tonnes)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lineplot_production_par_annee.png", dpi=180)
    plt.close()

    fertilizer_target = (
        df.dropna(subset=["utilisation_engrais", TARGET])
        .groupby("utilisation_engrais", as_index=False)[TARGET]
        .mean()
    )
    fertilizer_target["utilisation_engrais"] = fertilizer_target[
        "utilisation_engrais"
    ].map({0.0: "Sans engrais", 1.0: "Avec engrais"})
    plt.figure(figsize=(7, 5))
    sns.barplot(data=fertilizer_target, x="utilisation_engrais", y=TARGET)
    plt.title("Proportion de bonnes recoltes selon l'utilisation d'engrais")
    plt.xlabel("")
    plt.ylabel("Proportion de bonnes recoltes")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "barplot_bonnes_recoltes_engrais.png", dpi=180)
    plt.close()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap="vlag", center=0, annot=False)
    plt.title("Matrice de correlation des variables numeriques")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heatmap_correlation.png", dpi=180)
    plt.close()


def save_model_figures(
    models: dict[str, Pipeline],
    metrics: dict[str, dict],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    for model_key, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["Mauvaise", "Bonne"],
            cmap="Blues",
            colorbar=False,
        )
        display.ax_.set_title(f"Matrice de confusion - {model_label(model_key)}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_key}.png", dpi=180)
        plt.close()

    plt.figure(figsize=(8, 6))
    for model_key, metric in metrics.items():
        plt.plot(metric["roc_fpr"], metric["roc_tpr"], label=f"{model_label(model_key)} (AUC={metric['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title("Courbes ROC des trois modeles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_comparaison_modeles.png", dpi=180)
    plt.close()

    tree_pipeline = models["decision_tree"]
    tree_feature_names = get_feature_names(tree_pipeline)
    plt.figure(figsize=(18, 10))
    plot_tree(
        tree_pipeline.named_steps["model"],
        feature_names=tree_feature_names,
        class_names=["Mauvaise", "Bonne"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "arbre_decision.png", dpi=180)
    plt.close()

    tree_importance = save_feature_importance(
        models["decision_tree"],
        "decision_tree",
        top_n=20,
    )
    forest_importance = save_feature_importance(
        models["random_forest"],
        "random_forest",
        top_n=20,
    )
    logistic_coefficients = save_logistic_coefficients(models["logistic_regression"], top_n=20)

    depth_rows = []
    for depth in range(1, 21):
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale_continuous=False)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=depth,
                        criterion="gini",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        depth_rows.append(
            {
                "max_depth": depth,
                "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
                "test_accuracy": accuracy_score(y_test, pipeline.predict(X_test)),
            }
        )
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(REPORTS_DIR / "accuracy_par_profondeur.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.plot(depth_df["max_depth"], depth_df["train_accuracy"], marker="o", label="Train")
    plt.plot(depth_df["max_depth"], depth_df["test_accuracy"], marker="o", label="Test")
    plt.xlabel("Profondeur maximale")
    plt.ylabel("Accuracy")
    plt.title("Impact de max_depth sur l'arbre de decision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "overfitting_max_depth.png", dpi=180)
    plt.close()

    estimator_rows = []
    for n_estimators in [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]:
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(scale_continuous=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        estimator_rows.append(
            {
                "n_estimators": n_estimators,
                "test_accuracy": accuracy_score(y_test, pipeline.predict(X_test)),
            }
        )
    estimator_df = pd.DataFrame(estimator_rows)
    estimator_df.to_csv(REPORTS_DIR / "accuracy_par_nombre_arbres.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.plot(estimator_df["n_estimators"], estimator_df["test_accuracy"], marker="o")
    plt.xlabel("Nombre d'arbres")
    plt.ylabel("Accuracy test")
    plt.title("Impact du nombre d'arbres sur la foret aleatoire")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "random_forest_n_estimators.png", dpi=180)
    plt.close()

    return {
        "decision_tree_importance": tree_importance,
        "random_forest_importance": forest_importance,
        "logistic_coefficients": logistic_coefficients,
        "best_depth_by_test_accuracy": int(
            depth_df.sort_values("test_accuracy", ascending=False).iloc[0]["max_depth"]
        ),
        "depth_overfitting_table": depth_rows,
        "n_estimators_table": estimator_rows,
    }


def save_feature_importance(pipeline: Pipeline, model_key: str, top_n: int) -> list[dict]:
    feature_names = get_feature_names(pipeline)
    importance = pipeline.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    importance_df.to_csv(REPORTS_DIR / f"importance_variables_{model_key}.csv", index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=importance_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orient="h",
    )
    plt.title(f"Importance des variables - {model_label(model_key)}")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"importance_variables_{model_key}.png", dpi=180)
    plt.close()

    return importance_df.to_dict(orient="records")


def save_logistic_coefficients(pipeline: Pipeline, top_n: int) -> list[dict]:
    feature_names = get_feature_names(pipeline)
    coefficients = pipeline.named_steps["model"].coef_[0]
    coefficients_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    )
    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    top_df = coefficients_df.sort_values("abs_coefficient", ascending=False).head(top_n)
    top_df.to_csv(REPORTS_DIR / "coefficients_regression_logistique.csv", index=False)

    plot_df = top_df.sort_values("coefficient", ascending=True)
    colors = ["#b22222" if value < 0 else "#1f7a4d" for value in plot_df["coefficient"]]
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Coefficients les plus influents - Regression logistique")
    plt.xlabel("Coefficient")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "coefficients_regression_logistique.png", dpi=180)
    plt.close()

    return top_df.to_dict(orient="records")


def build_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    default_year = int(df["annee"].max())
    default_season = "A"
    default_area_by_culture = df.groupby("culture")["superficie_ha"].median().to_dict()
    default_households_by_province = df.groupby("province")["nb_menages"].median().to_dict()

    maize_label = next(
        (culture for culture in df["culture"].dropna().unique() if culture.startswith("Ma") and culture != "Manioc"),
        "Mais",
    )

    raw = [
        {
            "scenario": "Kayanza - Mais",
            "province": "Kayanza",
            "culture": maize_label,
            "altitude_m": 1980,
            "pluviometrie_mm": 920,
            "temperature_moy_C": 17.8,
            "utilisation_engrais": 1,
            "acces_irrigation": 0,
        },
        {
            "scenario": "Bubanza - Manioc",
            "province": "Bubanza",
            "culture": "Manioc",
            "altitude_m": 790,
            "pluviometrie_mm": 550,
            "temperature_moy_C": 25.4,
            "utilisation_engrais": 0,
            "acces_irrigation": 1,
        },
        {
            "scenario": "Gitega - Haricot",
            "province": "Gitega",
            "culture": "Haricot",
            "altitude_m": 1720,
            "pluviometrie_mm": 430,
            "temperature_moy_C": 18.2,
            "utilisation_engrais": 0,
            "acces_irrigation": 0,
        },
        {
            "scenario": "Cibitoke - Patate douce",
            "province": "Cibitoke",
            "culture": "Patate douce",
            "altitude_m": 810,
            "pluviometrie_mm": 810,
            "temperature_moy_C": 24.1,
            "utilisation_engrais": 1,
            "acces_irrigation": 1,
        },
    ]

    scenarios = []
    for row in raw:
        scenarios.append(
            {
                **row,
                "annee": default_year,
                "saison": default_season,
                "superficie_ha": float(default_area_by_culture.get(row["culture"], df["superficie_ha"].median())),
                "nb_menages": int(default_households_by_province.get(row["province"], df["nb_menages"].median())),
            }
        )
    return pd.DataFrame(scenarios)


def predict_scenarios(models: dict[str, Pipeline], scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    X_scenarios = scenarios[FEATURES]
    for _, scenario in scenarios.iterrows():
        output = {
            "scenario": scenario["scenario"],
            "province": scenario["province"],
            "culture": scenario["culture"],
        }
        for model_key, pipeline in models.items():
            prob = float(pipeline.predict_proba(pd.DataFrame([scenario[FEATURES]]))[0, 1])
            pred = int(prob >= 0.5)
            output[f"{model_key}_prediction"] = "Bonne" if pred == 1 else "Mauvaise"
            output[f"{model_key}_prob_bonne"] = round(prob, 4)
        rows.append(output)
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS_DIR / "predictions_scenarios.csv", index=False)
    return result


def write_markdown_report(
    data_summary: dict,
    metrics: dict[str, dict],
    scenario_predictions: pd.DataFrame,
    extra: dict,
) -> None:
    best_accuracy = max(metrics.values(), key=lambda row: row["accuracy"])
    best_auc = max(metrics.values(), key=lambda row: row["auc"])
    scenario_3 = scenario_predictions[
        scenario_predictions["scenario"].str.contains("Gitega")
    ].iloc[0]
    scenario_3_table = "\n".join(
        [
            "| Modele | Prediction | Probabilite bonne |",
            "|---|---:|---:|",
            f"| Arbre de decision | {scenario_3['decision_tree_prediction']} | {scenario_3['decision_tree_prob_bonne'] * 100:.1f}% |",
            f"| Foret aleatoire | {scenario_3['random_forest_prediction']} | {scenario_3['random_forest_prob_bonne'] * 100:.1f}% |",
            f"| Regression logistique | {scenario_3['logistic_regression_prediction']} | {scenario_3['logistic_regression_prob_bonne'] * 100:.1f}% |",
        ]
    )

    lines = [
        "# Rapport court - Agriculture au Burundi",
        "",
        "## Questions 29 et 30",
        "",
        "### Q29 - Scenario Gitega - Haricot",
        "",
        (
            "Le scenario Gitega - Haricot combine une pluviometrie tres faible "
            "(430 mm), pas d'engrais et pas d'irrigation. Les predictions obtenues "
            "sont les suivantes :"
        ),
        "",
        scenario_3_table,
        "",
        (
            "Cette situation est agronomiquement risquee : le haricot est sensible "
            "au stress hydrique, surtout lorsque la pluie est faible et que "
            "l'irrigation n'est pas disponible. Un agronome pourrait recommander "
            "une meilleure conservation de l'humidite du sol, l'acces a une source "
            "d'irrigation si possible, l'ajustement du calendrier cultural, "
            "l'utilisation de varietes plus tolerantes et un appui en fertilisation."
        ),
        "",
        "### Q30 - Choix du modele pour le Ministere",
        "",
        (
            f"Le modele recommande est la {best_auc['model']} si l'objectif est "
            f"la robustesse globale, car elle obtient l'AUC la plus elevee "
            f"({best_auc['auc']:.3f}). En accuracy pure, le meilleur modele est "
            f"{best_accuracy['model']} ({best_accuracy['accuracy']:.3f})."
        ),
        "",
        (
            "Pour une utilisation publique, la foret aleatoire est souvent le "
            "meilleur compromis : elle gere bien les interactions non lineaires "
            "entre pluie, altitude, temperature, culture et pratiques agricoles, "
            "tout en restant interpretable avec l'importance des variables."
        ),
        "",
        (
            "Des donnees supplementaires amelioreraient le modele : type de sol, "
            "dates exactes de semis, maladies et ravageurs, qualite des semences, "
            "prix/acces aux intrants, donnees meteo plus fines, pratiques culturales "
            "et historiques de rendement par exploitation."
        ),
        "",
        (
            "Limites : le dataset est simule, la cible est construite a partir du "
            "rendement, les classes sont desequilibrees, certaines valeurs sont "
            "manquantes et les scenarios incomplets utilisent des valeurs par "
            "defaut. Le modele doit donc aider la decision, pas remplacer "
            "l'expertise agronomique."
        ),
        "",
        "## Resume technique",
        "",
        f"- Lignes/colonnes : {data_summary['shape'][0]} / {data_summary['shape'][1]}",
        f"- Periode : {data_summary['years'][0]}-{data_summary['years'][1]}",
        f"- Provinces : {data_summary['n_provinces']}",
        f"- Cultures : {', '.join(data_summary['cultures'])}",
        f"- Meilleure culture en rendement moyen : {data_summary['highest_mean_yield_culture']}",
        f"- Province avec pluie moyenne la plus forte : {data_summary['highest_mean_rain_province']}",
        f"- Profondeur avec meilleure accuracy test : {extra['best_depth_by_test_accuracy']}",
    ]
    (REPORTS_DIR / "rapport_reflexion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_data()
    clean = clean_for_training(df)

    data_summary = save_data_quality_reports(df, clean)
    save_exploration_figures(df)

    X = clean[FEATURES]
    y = clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    metrics = {}
    for model_key, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics[model_key] = evaluate_model(
            model_key,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        joblib.dump(pipeline, MODELS_DIR / f"{model_key}.pkl")

    forest_cv_scores = cross_val_score(
        models["random_forest"],
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    metrics["random_forest"]["cv_accuracy_mean"] = float(forest_cv_scores.mean())
    metrics["random_forest"]["cv_accuracy_std"] = float(forest_cv_scores.std())

    metrics_df = pd.DataFrame(
        [
            {
                "model_key": key,
                "model": value["model"],
                "accuracy": value["accuracy"],
                "train_accuracy": value["train_accuracy"],
                "precision_macro": value["precision_macro"],
                "recall_macro": value["recall_macro"],
                "f1_macro": value["f1_macro"],
                "auc": value["auc"],
                "false_positives": value["false_positives"],
                "false_negatives": value["false_negatives"],
                "cv_accuracy_mean": value.get("cv_accuracy_mean", np.nan),
                "cv_accuracy_std": value.get("cv_accuracy_std", np.nan),
            }
            for key, value in metrics.items()
        ]
    )
    metrics_df.to_csv(REPORTS_DIR / "metriques_modeles.csv", index=False)

    extra = save_model_figures(models, metrics, X_train, X_test, y_train, y_test)
    scenarios = build_scenarios(df)
    scenarios.to_csv(REPORTS_DIR / "scenarios_utilises.csv", index=False)
    scenario_predictions = predict_scenarios(models, scenarios)

    artifacts = {
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "continuous_features": CONTINUOUS_FEATURES,
        "binary_features": BINARY_FEATURES,
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "data_summary": data_summary,
        "metrics": metrics,
        "extra": extra,
        "categories": {
            "saison": sorted(df["saison"].dropna().unique().tolist()),
            "province": sorted(df["province"].dropna().unique().tolist()),
            "culture": sorted(df["culture"].dropna().unique().tolist()),
        },
        "defaults": {
            "annee": int(df["annee"].max()),
            "saison": "A",
            "altitude_m": float(df["altitude_m"].median()),
            "pluviometrie_mm": float(df["pluviometrie_mm"].median()),
            "temperature_moy_C": float(df["temperature_moy_C"].median()),
            "superficie_ha": float(df["superficie_ha"].median()),
            "utilisation_engrais": int(df["utilisation_engrais"].mode(dropna=True).iloc[0]),
            "acces_irrigation": int(df["acces_irrigation"].mode(dropna=True).iloc[0]),
            "nb_menages": int(df["nb_menages"].median()),
        },
        "scenario_predictions": scenario_predictions.to_dict(orient="records"),
    }
    joblib.dump(artifacts, MODELS_DIR / "artifacts.pkl")

    serializable_artifacts = json.loads(json.dumps(artifacts, default=str))
    (REPORTS_DIR / "resume_pipeline.json").write_text(
        json.dumps(serializable_artifacts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_report(data_summary, metrics, scenario_predictions, extra)

    print("Pipeline termine.")
    print(metrics_df.round(4).to_string(index=False))
    print()
    print(scenario_predictions.to_string(index=False))


if __name__ == "__main__":
    main()
