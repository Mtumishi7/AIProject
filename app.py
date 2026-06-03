from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "agriculture_burundi.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"
ARTIFACTS_PATH = MODELS_DIR / "artifacts.pkl"
HISTORY_PATH = REPORTS_DIR / "prediction_history.csv"

MODEL_FILES = {
    "Arbre de decision": "decision_tree.pkl",
    "Foret aleatoire": "random_forest.pkl",
    "Regression logistique": "logistic_regression.pkl",
}

MODEL_KEYS = {
    "Arbre de decision": "decision_tree",
    "Foret aleatoire": "random_forest",
    "Regression logistique": "logistic_regression",
}

MODEL_LABELS = {value: key for key, value in MODEL_KEYS.items()}

MAIZE_RAW_VALUES = {
    "Mais",
    "mais",
    "MAIS",
    "Ma\u00efs",
    "ma\u00efs",
    "Ma\u00c3\u00afs",
    "ma\u00c3\u00afs",
    "Ma\u00c4\u00bcs",
}

SCENARIO_VALUES = {
    "Saisie libre": None,
    "Kayanza - Mais": {
        "province": "Kayanza",
        "culture_prefix": "Ma",
        "altitude_m": 1980.0,
        "pluviometrie_mm": 920.0,
        "temperature_moy_C": 17.8,
        "utilisation_engrais": True,
        "acces_irrigation": False,
    },
    "Bubanza - Manioc": {
        "province": "Bubanza",
        "culture": "Manioc",
        "altitude_m": 790.0,
        "pluviometrie_mm": 550.0,
        "temperature_moy_C": 25.4,
        "utilisation_engrais": False,
        "acces_irrigation": True,
    },
    "Gitega - Haricot": {
        "province": "Gitega",
        "culture": "Haricot",
        "altitude_m": 1720.0,
        "pluviometrie_mm": 430.0,
        "temperature_moy_C": 18.2,
        "utilisation_engrais": False,
        "acces_irrigation": False,
    },
    "Cibitoke - Patate douce": {
        "province": "Cibitoke",
        "culture": "Patate douce",
        "altitude_m": 810.0,
        "pluviometrie_mm": 810.0,
        "temperature_moy_C": 24.1,
        "utilisation_engrais": True,
        "acces_irrigation": True,
    },
}


st.set_page_config(
    page_title="AgriPredict Burundi",
    page_icon="AP",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = joblib.load(ARTIFACTS_PATH)
    models = {
        label: joblib.load(MODELS_DIR / filename)
        for label, filename in MODEL_FILES.items()
    }
    return artifacts, models


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding="utf-8")


@st.cache_data
def load_report_csv(filename: str) -> pd.DataFrame:
    path = REPORTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def inject_css(theme: str) -> None:
    dark = theme == "Sombre"
    colors = {
        "bg": "#0f1411" if dark else "#f7f9f4",
        "bg_top": "#17231b" if dark else "#edf6e9",
        "surface": "#171c19" if dark else "#fffefa",
        "surface_2": "#202820" if dark else "#eef3ea",
        "surface_3": "#2a3327" if dark else "#e3eadf",
        "text": "#f7f3ea" if dark else "#182116",
        "muted": "#b9bdad" if dark else "#657064",
        "border": "#343d35" if dark else "#d9dfd3",
        "primary": "#75c044" if dark else "#235f3a",
        "accent": "#16b8a6" if dark else "#0d8f85",
        "blue": "#6da7ff" if dark else "#2f68c5",
        "green": "#55c56a" if dark else "#2f8b43",
        "amber": "#f6c65b" if dark else "#c27a13",
        "red": "#ff7a6b" if dark else "#c4493d",
        "shadow": "rgba(0, 0, 0, .34)" if dark else "rgba(54, 70, 49, .12)",
    }

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {colors["bg"]};
            --bg-top: {colors["bg_top"]};
            --surface: {colors["surface"]};
            --surface-2: {colors["surface_2"]};
            --surface-3: {colors["surface_3"]};
            --text: {colors["text"]};
            --muted: {colors["muted"]};
            --border: {colors["border"]};
            --primary: {colors["primary"]};
            --accent: {colors["accent"]};
            --blue: {colors["blue"]};
            --green: {colors["green"]};
            --amber: {colors["amber"]};
            --red: {colors["red"]};
            --shadow: {colors["shadow"]};
        }}

        * {{
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            letter-spacing: 0;
        }}

        .stApp {{
            color: var(--text);
            background:
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg) 310px),
                var(--bg);
        }}

        .main .block-container {{
            max-width: 1340px;
            padding-top: 1rem;
            padding-bottom: 2.5rem;
        }}

        h1, h2, h3, h4, p, label, span {{
            color: var(--text);
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.1rem;
        }}

        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: .85rem .95rem;
            box-shadow: 0 12px 26px var(--shadow);
            border-top: 3px solid var(--accent);
        }}

        div[data-testid="stMetricLabel"] {{
            color: var(--muted);
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--text);
        }}

        .app-header {{
            border: 1px solid var(--border);
            border-radius: 8px;
            background:
                linear-gradient(135deg, color-mix(in srgb, var(--primary) 17%, transparent) 0%, transparent 38%),
                linear-gradient(125deg, var(--surface) 0%, var(--surface-2) 56%, var(--surface-3) 100%);
            padding: 1.15rem 1.25rem;
            box-shadow: 0 16px 36px var(--shadow);
            margin-bottom: 1rem;
        }}

        .eyebrow {{
            color: var(--accent);
            font-size: .76rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }}

        .app-title {{
            font-size: 2.05rem;
            line-height: 1.13;
            font-weight: 850;
            margin: 0 0 .35rem 0;
            color: var(--text);
        }}

        .app-subtitle {{
            color: var(--muted);
            font-size: 1.01rem;
            max-width: 920px;
            margin: 0;
        }}

        .panel {{
            border: 1px solid var(--border);
            border-radius: 8px;
            background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
            padding: .95rem;
            box-shadow: 0 12px 26px var(--shadow);
        }}

        .soft-panel {{
            border: 1px solid var(--border);
            border-radius: 8px;
            background: color-mix(in srgb, var(--surface-2) 88%, var(--accent) 12%);
            padding: .85rem;
        }}

        .section-label {{
            color: var(--muted);
            font-size: .76rem;
            font-weight: 850;
            text-transform: uppercase;
            margin: .35rem 0 .55rem 0;
        }}

        .badge-row {{
            display: flex;
            gap: .42rem;
            flex-wrap: wrap;
            margin-top: .75rem;
        }}

        .badge {{
            border: 1px solid var(--border);
            border-radius: 999px;
            background: color-mix(in srgb, var(--surface-2) 82%, var(--accent) 18%);
            color: var(--text);
            padding: .28rem .62rem;
            font-size: .82rem;
            font-weight: 650;
        }}

        .risk-good, .risk-watch, .risk-bad {{
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem;
            background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
            box-shadow: 0 12px 26px var(--shadow);
        }}

        .risk-good {{ border-left: 7px solid var(--green); }}
        .risk-watch {{ border-left: 7px solid var(--amber); }}
        .risk-bad {{ border-left: 7px solid var(--red); }}

        .risk-title {{
            font-size: 1.5rem;
            line-height: 1.15;
            font-weight: 850;
            margin: 0 0 .2rem 0;
            color: var(--text);
        }}

        .risk-copy {{
            color: var(--muted);
            margin: 0;
        }}

        .mini-kpi {{
            border: 1px solid var(--border);
            border-radius: 8px;
            background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
            padding: .78rem;
            min-height: 88px;
        }}

        .mini-kpi-label {{
            color: var(--muted);
            font-size: .76rem;
            margin-bottom: .22rem;
        }}

        .mini-kpi-value {{
            color: var(--text);
            font-weight: 800;
            font-size: 1.08rem;
            line-height: 1.25;
        }}

        .status-ok {{ color: var(--green); font-weight: 800; }}
        .status-warn {{ color: var(--amber); font-weight: 800; }}
        .status-bad {{ color: var(--red); font-weight: 800; }}
        .muted {{ color: var(--muted); }}

        .stButton button, .stDownloadButton button {{
            border-radius: 8px;
            font-weight: 750;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #ffffff;
            box-shadow: 0 10px 22px var(--shadow);
            transition: transform .14s ease, box-shadow .14s ease, filter .14s ease;
        }}

        .stButton button:hover, .stDownloadButton button:hover {{
            border-color: var(--accent);
            filter: saturate(1.08) brightness(1.04);
            transform: translateY(-1px);
            box-shadow: 0 14px 30px var(--shadow);
        }}

        div[data-testid="stTabs"] button[role="tab"] {{
            border-radius: 8px;
            color: var(--muted);
            font-weight: 750;
        }}

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            background: color-mix(in srgb, var(--surface-2) 70%, var(--accent) 30%);
            color: var(--text);
            border-bottom-color: var(--accent);
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {{
            border-radius: 8px;
            border-color: var(--border);
            background: var(--surface);
            color: var(--text);
        }}

        div[data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 24px var(--shadow);
        }}

        .stProgress > div > div {{
            background: linear-gradient(90deg, var(--red), var(--amber), var(--green), var(--accent));
        }}

        a {{
            color: var(--primary);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text in MAIZE_RAW_VALUES:
        return "Mais"
    return text


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    shown = frame.copy()
    if "culture" in shown.columns:
        shown["culture"] = shown["culture"].map(display_value)
    return shown


def clean_display_text(text: str) -> str:
    return (
        text.replace("Ma\u00c3\u00afs", "Mais")
        .replace("ma\u00c3\u00afs", "mais")
        .replace("temp\u00c3\u00a9rature", "temperature")
        .replace("r\u00c3\u00a9colte", "recolte")
        .replace("r\u00c3\u00a9coltes", "recoltes")
    )


def index_of(values: list[str], value: str) -> int:
    return values.index(value) if value in values else 0


def culture_default(categories: dict[str, list[str]], scenario: dict[str, Any] | None) -> str:
    cultures = categories["culture"]
    if scenario is None:
        return cultures[0]
    if scenario.get("culture") in cultures:
        return scenario["culture"]
    prefix = scenario.get("culture_prefix")
    if prefix:
        for culture in cultures:
            if culture.startswith(prefix) and culture != "Manioc":
                return culture
    return cultures[0]


def metric_table(artifacts: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model_key, metrics in artifacts["metrics"].items():
        rows.append(
            {
                "Modele": MODEL_LABELS.get(model_key, metrics["model"]),
                "Accuracy": metrics["accuracy"],
                "Accuracy train": metrics.get("train_accuracy"),
                "F1 macro": metrics["f1_macro"],
                "AUC": metrics["auc"],
                "Faux positifs": metrics["false_positives"],
                "Faux negatifs": metrics["false_negatives"],
                "CV moyenne": metrics.get("cv_accuracy_mean"),
                "CV ecart-type": metrics.get("cv_accuracy_std"),
            }
        )
    return pd.DataFrame(rows)


def styled_metrics(frame: pd.DataFrame):
    return frame.style.format(
        {
            "Accuracy": "{:.3f}",
            "Accuracy train": "{:.3f}",
            "F1 macro": "{:.3f}",
            "AUC": "{:.3f}",
            "CV moyenne": "{:.3f}",
            "CV ecart-type": "{:.3f}",
        },
        na_rep="-",
    )


def clean_feature_name(name: str) -> str:
    return (
        display_value(name)
        .replace("continuous__", "")
        .replace("binary__", "")
        .replace("categorical__", "")
        .replace("_", " ")
    )


def feature_chart(model_label: str, artifacts: dict[str, Any]) -> pd.DataFrame:
    key = {
        "Arbre de decision": "decision_tree_importance",
        "Foret aleatoire": "random_forest_importance",
        "Regression logistique": "logistic_coefficients",
    }[model_label]
    df = pd.DataFrame(artifacts["extra"].get(key, []))
    if df.empty:
        return df
    value_col = "importance" if "importance" in df.columns else "coefficient"
    sort_col = value_col if value_col == "importance" else "abs_coefficient"
    chart_df = df.sort_values(sort_col, ascending=False)[["feature", value_col]].head(14).copy()
    chart_df["feature"] = chart_df["feature"].map(clean_feature_name)
    return chart_df.set_index("feature").sort_values(value_col)


def prediction_state(probability: float, threshold: float) -> tuple[str, str, str]:
    if probability >= max(0.75, threshold + 0.15):
        return "Bonne recolte", "risk-good", "Confiance elevee"
    if probability >= threshold:
        return "Bonne recolte", "risk-watch", "Signal positif a confirmer"
    if probability >= max(0.0, threshold - 0.15):
        return "Mauvaise recolte", "risk-watch", "Zone incertaine"
    return "Mauvaise recolte", "risk-bad", "Risque eleve"


def risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "Risque faible"
    if probability >= 0.55:
        return "Risque moyen"
    return "Risque eleve"


def format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.1f}{suffix}".replace(",", " ")


def predict_all(
    models: dict[str, Any],
    input_df: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for model_label, pipeline in models.items():
        probs = pipeline.predict_proba(input_df[features])[:, 1]
        rows.extend(
            {
                "Modele": model_label,
                "Prediction": "Bonne" if prob >= threshold else "Mauvaise",
                "Probabilite bonne": float(prob),
                "Risque mauvaise": 1 - float(prob),
            }
            for prob in probs
        )
    return pd.DataFrame(rows)


def consensus_text(predictions: pd.DataFrame) -> tuple[str, float]:
    counts = predictions["Prediction"].value_counts()
    top_prediction = counts.index[0]
    ratio = counts.iloc[0] / len(predictions)
    if ratio == 1:
        return f"Les {len(predictions)} modeles sont unanimes : {top_prediction}.", ratio
    return f"Les modeles ne sont pas unanimes : {top_prediction} domine a {ratio:.0%}.", ratio


def profile_selection(data: pd.DataFrame, province: str, culture: str) -> dict[str, Any]:
    scope = data[(data["province"] == province) & (data["culture"] == culture)]
    if scope.empty:
        scope = data[data["province"] == province]
    if scope.empty:
        scope = data

    clean_target = scope.dropna(subset=["bonne_recolte"])
    return {
        "observations": int(len(scope)),
        "pluie_mediane": scope["pluviometrie_mm"].median(),
        "rendement_moyen": scope["rendement_t_ha"].mean(),
        "production_moyenne": scope["production_totale_t"].mean(),
        "taux_bonne": clean_target["bonne_recolte"].mean() if not clean_target.empty else None,
    }


def exact_dataset_match(reference: pd.DataFrame, input_row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    row = input_row.iloc[0]
    mask = pd.Series(True, index=reference.index)
    for column in features:
        if column not in reference.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference[column]):
            mask &= (reference[column].astype(float) - float(row[column])).abs() <= 1e-9
        else:
            mask &= reference[column].astype(str) == str(row[column])
    return reference.loc[mask]


def out_of_domain_messages(reference: pd.DataFrame, input_row: pd.DataFrame, artifacts: dict[str, Any]) -> list[str]:
    messages = []
    row = input_row.iloc[0]
    for column in artifacts["continuous_features"]:
        if column not in reference.columns:
            continue
        observed = reference[column].dropna()
        if observed.empty:
            continue
        value = float(row[column])
        low = float(observed.min())
        high = float(observed.max())
        if value < low or value > high:
            messages.append(f"{column}={value:g} est hors de l'intervalle observe dans le CSV [{low:g}, {high:g}].")
    return messages


def nearest_historical_cases(
    reference: pd.DataFrame,
    input_row: pd.DataFrame,
    artifacts: dict[str, Any],
    k: int = 25,
) -> tuple[pd.DataFrame, str]:
    row = input_row.iloc[0]
    known = reference.dropna(subset=["bonne_recolte"]).copy()
    if known.empty:
        return pd.DataFrame(), "Aucune cible connue"

    scope = known[(known["province"] == row["province"]) & (known["culture"] == row["culture"])].copy()
    scope_label = f"{display_value(row['culture'])} a {row['province']}"
    if len(scope) < 12:
        scope = known[known["culture"] == row["culture"]].copy()
        scope_label = f"culture {display_value(row['culture'])}"
    if len(scope) < 12:
        scope = known[known["province"] == row["province"]].copy()
        scope_label = f"province {row['province']}"
    if len(scope) < 12:
        scope = known.copy()
        scope_label = "ensemble du CSV"

    distance = pd.Series(0.0, index=scope.index)
    weights = {
        "annee": 0.25,
        "altitude_m": 0.75,
        "pluviometrie_mm": 1.15,
        "temperature_moy_C": 0.85,
        "superficie_ha": 0.35,
        "nb_menages": 0.30,
    }
    for column in artifacts["continuous_features"]:
        observed = known[column].dropna()
        if observed.empty:
            continue
        span = max(float(observed.max() - observed.min()), 1.0)
        values = scope[column].fillna(observed.median()).astype(float)
        distance += (values - float(row[column])).abs() / span * weights.get(column, 0.5)

    for column in artifacts["binary_features"]:
        values = scope[column].fillna(scope[column].mode().iloc[0] if not scope[column].mode().empty else 0)
        distance += (values.astype(float) - float(row[column])).abs() * 0.45

    distance += (scope["saison"].astype(str) != str(row["saison"])).astype(float) * 0.18
    distance += (scope["province"].astype(str) != str(row["province"])).astype(float) * 0.55
    distance += (scope["culture"].astype(str) != str(row["culture"])).astype(float) * 0.70

    nearest = scope.assign(distance_reference=distance).sort_values("distance_reference").head(k)
    return nearest, scope_label


def historical_anchor(
    reference_data: pd.DataFrame,
    input_row: pd.DataFrame,
    artifacts: dict[str, Any],
    model_probability: float,
) -> dict[str, Any]:
    reference = normalize_categories(reference_data, artifacts)
    known = reference.dropna(subset=["bonne_recolte"]).copy()
    if known.empty:
        return {
            "final_probability": model_probability,
            "historical_probability": None,
            "source": "Modele ML uniquement",
            "conformity": "Non disponible",
            "scope_label": "Aucune cible connue",
            "nearest_cases": pd.DataFrame(),
            "out_of_domain": [],
            "exact_match": False,
            "nearest_distance": None,
        }

    exact = exact_dataset_match(known, input_row, artifacts["features"])
    nearest, scope_label = nearest_historical_cases(known, input_row, artifacts)
    out_of_domain = out_of_domain_messages(known, input_row, artifacts)

    if not exact.empty:
        observed_rate = float(exact["bonne_recolte"].mean())
        return {
            "final_probability": observed_rate,
            "historical_probability": observed_rate,
            "source": "Observation exacte trouvee dans agriculture_burundi.csv",
            "conformity": "Exacte",
            "scope_label": "ligne exacte du CSV",
            "nearest_cases": exact.assign(distance_reference=0.0).head(10),
            "out_of_domain": out_of_domain,
            "exact_match": True,
            "nearest_distance": 0.0,
        }

    historical_probability = float(nearest["bonne_recolte"].mean()) if not nearest.empty else float(known["bonne_recolte"].mean())
    nearest_distance = float(nearest["distance_reference"].mean()) if not nearest.empty else None
    if nearest_distance is None:
        historical_weight = 0.25
        conformity = "Faible"
    elif nearest_distance <= 0.20:
        historical_weight = 0.50
        conformity = "Forte"
    elif nearest_distance <= 0.45:
        historical_weight = 0.40
        conformity = "Moyenne"
    else:
        historical_weight = 0.30
        conformity = "Faible"

    final_probability = (1 - historical_weight) * model_probability + historical_weight * historical_probability
    return {
        "final_probability": float(final_probability),
        "historical_probability": historical_probability,
        "source": "Modele ML ajuste par les cas proches du CSV",
        "conformity": conformity,
        "scope_label": scope_label,
        "nearest_cases": nearest,
        "out_of_domain": out_of_domain,
        "exact_match": False,
        "nearest_distance": nearest_distance,
    }


def recommendations(
    input_row: pd.Series,
    probability: float,
    threshold: float,
    disagreement: bool,
) -> pd.DataFrame:
    rows = []
    culture = display_value(input_row["culture"])
    rain = float(input_row["pluviometrie_mm"])
    temp = float(input_row["temperature_moy_C"])
    fertilizer = int(input_row["utilisation_engrais"])
    irrigation = int(input_row["acces_irrigation"])

    def add(priority: str, action: str, reason: str) -> None:
        rows.append({"Priorite": priority, "Action": action, "Raison": reason})

    if probability < threshold:
        add(
            "Haute",
            "Classer cette parcelle en suivi prioritaire.",
            "La probabilite de bonne recolte est sous le seuil de decision.",
        )
    elif probability < threshold + 0.12:
        add(
            "Moyenne",
            "Confirmer la prediction avec une observation terrain.",
            "La prediction est positive mais proche du seuil.",
        )

    if rain < 500:
        add(
            "Haute",
            "Mettre en place une strategie anti-stress hydrique.",
            "La pluviometrie est tres faible pour la saison.",
        )
    elif rain < 650:
        add(
            "Moyenne",
            "Renforcer le paillage et ajuster le calendrier de semis.",
            "La pluie disponible reste limitee.",
        )
    elif rain > 1050:
        add(
            "Moyenne",
            "Surveiller drainage, maladies et pertes par exces d'eau.",
            "Une pluviometrie tres elevee peut aussi fragiliser la recolte.",
        )

    if fertilizer == 0:
        add(
            "Moyenne",
            f"Evaluer une fertilisation raisonnee pour {culture}.",
            "Le modele donne un signal positif a l'utilisation d'engrais.",
        )

    if irrigation == 0 and rain < 850:
        add(
            "Moyenne",
            "Chercher une source d'eau complementaire ou une variete plus tolerante.",
            "L'absence d'irrigation augmente la vulnerabilite en saison seche.",
        )

    if temp > 24:
        add(
            "Moyenne",
            "Conserver l'humidite du sol et surveiller le stress thermique.",
            "La temperature moyenne est elevee.",
        )
    elif temp < 16:
        add(
            "Basse",
            "Verifier que la variete choisie est adaptee aux zones plus fraiches.",
            "Les temperatures basses peuvent ralentir la croissance.",
        )

    if disagreement:
        add(
            "Haute",
            "Demander une verification terrain avant une decision couteuse.",
            "Les trois modeles ne donnent pas exactement le meme signal.",
        )

    if not rows:
        add(
            "Basse",
            "Maintenir le suivi de la pluie, des intrants et de l'etat sanitaire.",
            "Aucun facteur de risque majeur n'est ressorti pour cette saisie.",
        )
    return pd.DataFrame(rows)


def local_sensitivity(pipeline: Any, input_row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    base = float(pipeline.predict_proba(input_row[features])[0, 1])
    changes = [
        ("Pluie +100 mm", {"pluviometrie_mm": float(input_row.iloc[0]["pluviometrie_mm"]) + 100}),
        ("Pluie -100 mm", {"pluviometrie_mm": max(0.0, float(input_row.iloc[0]["pluviometrie_mm"]) - 100)}),
        ("Temperature +1 C", {"temperature_moy_C": float(input_row.iloc[0]["temperature_moy_C"]) + 1}),
        ("Temperature -1 C", {"temperature_moy_C": max(0.0, float(input_row.iloc[0]["temperature_moy_C"]) - 1)}),
        ("Engrais active", {"utilisation_engrais": 1}),
        ("Engrais retire", {"utilisation_engrais": 0}),
        ("Irrigation activee", {"acces_irrigation": 1}),
        ("Irrigation retiree", {"acces_irrigation": 0}),
    ]
    rows = []
    for label, updates in changes:
        modified = input_row.copy()
        for column, value in updates.items():
            modified.loc[modified.index[0], column] = value
        prob = float(pipeline.predict_proba(modified[features])[0, 1])
        rows.append({"Changement": label, "Probabilite": prob, "Impact": prob - base})
    return pd.DataFrame(rows).sort_values("Impact", key=lambda series: series.abs(), ascending=False)


def rainfall_curve(models: dict[str, Any], input_row: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for rain in range(350, 1351, 50):
        modified = input_row.copy()
        modified.loc[modified.index[0], "pluviometrie_mm"] = float(rain)
        row = {"Pluviometrie": rain}
        for model_label, pipeline in models.items():
            row[model_label] = float(pipeline.predict_proba(modified[features])[0, 1])
        rows.append(row)
    return pd.DataFrame(rows).set_index("Pluviometrie")


def intervention_table(
    pipeline: Any,
    base_input: pd.DataFrame,
    features: list[str],
    threshold: float,
    reference_data: pd.DataFrame | None = None,
    artifacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    base_model_prob = float(pipeline.predict_proba(base_input[features])[0, 1])
    if reference_data is not None and artifacts is not None:
        base_prob = float(historical_anchor(reference_data, base_input, artifacts, base_model_prob)["final_probability"])
    else:
        base_prob = base_model_prob
    scenarios = [
        ("Situation actuelle", {}),
        ("Pluie +100 mm", {"pluviometrie_mm": float(base_input.iloc[0]["pluviometrie_mm"]) + 100}),
        ("Engrais", {"utilisation_engrais": 1}),
        ("Irrigation", {"acces_irrigation": 1}),
        ("Engrais + irrigation", {"utilisation_engrais": 1, "acces_irrigation": 1}),
        (
            "Pluie + engrais + irrigation",
            {
                "pluviometrie_mm": float(base_input.iloc[0]["pluviometrie_mm"]) + 100,
                "utilisation_engrais": 1,
                "acces_irrigation": 1,
            },
        ),
    ]
    rows = []
    for label, updates in scenarios:
        modified = base_input.copy()
        for column, value in updates.items():
            modified.loc[modified.index[0], column] = value
        model_prob = float(pipeline.predict_proba(modified[features])[0, 1])
        if reference_data is not None and artifacts is not None:
            prob = float(historical_anchor(reference_data, modified, artifacts, model_prob)["final_probability"])
        else:
            prob = model_prob
        rows.append(
            {
                "Scenario": label,
                "Prediction": "Bonne" if prob >= threshold else "Mauvaise",
                "Probabilite bonne": prob,
                "Gain vs base": prob - base_prob,
            }
        )
    return pd.DataFrame(rows)


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(HISTORY_PATH)


def append_history(record: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    frame = pd.DataFrame([record])
    frame.to_csv(
        HISTORY_PATH,
        index=False,
        mode="a",
        header=not HISTORY_PATH.exists(),
        encoding="utf-8",
    )


def normalize_categories(data: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    normalized = data.copy()
    cultures = artifacts["categories"]["culture"]
    maize = next((value for value in cultures if value.startswith("Ma") and value != "Manioc"), None)
    if maize and "culture" in normalized.columns:
        normalized["culture"] = normalized["culture"].replace({value: maize for value in MAIZE_RAW_VALUES})

    yes_values = {"Oui": 1, "oui": 1, "Yes": 1, "yes": 1, "TRUE": 1, "true": 1, True: 1}
    no_values = {"Non": 0, "non": 0, "No": 0, "no": 0, "FALSE": 0, "false": 0, False: 0}
    for column in ["utilisation_engrais", "acces_irrigation"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].replace({**yes_values, **no_values})
    return normalized


def validate_batch_input(data: pd.DataFrame, artifacts: dict[str, Any]) -> list[str]:
    notes = []
    features = artifacts["features"]
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        notes.append("Colonnes absentes completees avec les valeurs par defaut : " + ", ".join(missing))

    extra = [column for column in data.columns if column not in features]
    if extra:
        notes.append("Colonnes supplementaires conservees dans la sortie : " + ", ".join(extra[:8]))

    normalized = normalize_categories(data, artifacts)
    for column in ["saison", "province", "culture"]:
        if column in normalized.columns:
            allowed = set(artifacts["categories"][column])
            unknown = sorted(set(normalized[column].dropna().astype(str)) - allowed)
            if unknown:
                notes.append(
                    f"Valeurs inconnues pour {column} : {', '.join(map(display_value, unknown[:5]))}. "
                    "Le modele les ignore ou utilise l'imputation."
                )
    return notes


def predict_batch(
    models: dict[str, Any],
    data: pd.DataFrame,
    artifacts: dict[str, Any],
    threshold: float,
    reference_data: pd.DataFrame,
) -> pd.DataFrame:
    prepared = normalize_categories(data, artifacts)
    for feature in artifacts["features"]:
        if feature not in prepared.columns:
            prepared[feature] = artifacts["defaults"].get(feature)

    output = prepared.copy()
    model_decisions = {}
    model_probs = {}
    for model_label, pipeline in models.items():
        model_key = MODEL_KEYS[model_label]
        probs = pipeline.predict_proba(prepared[artifacts["features"]])[:, 1]
        decisions = ["Bonne" if prob >= threshold else "Mauvaise" for prob in probs]
        output[f"{model_key}_prediction"] = decisions
        output[f"{model_key}_prob_bonne"] = probs
        model_decisions[model_key] = decisions
        model_probs[model_key] = probs

    decisions_df = pd.DataFrame(model_decisions)
    probs_df = pd.DataFrame(model_probs)
    output["prob_bonne_moyenne_modeles"] = probs_df.mean(axis=1)
    output["consensus_prediction_modeles"] = decisions_df.mode(axis=1)[0]
    output["consensus_ratio"] = decisions_df.apply(lambda row: row.value_counts(normalize=True).iloc[0], axis=1)

    final_probs = []
    final_predictions = []
    conformity_sources = []
    observed_labels = []
    for index, row in prepared.iterrows():
        observed = row.get("bonne_recolte") if "bonne_recolte" in prepared.columns else None
        if observed is not None and not pd.isna(observed):
            final_prob = float(observed)
            source = "Cible observee du CSV importe"
            observed_labels.append("Bonne" if final_prob >= threshold else "Mauvaise")
        else:
            row_input = prepared.loc[[index], artifacts["features"]]
            anchor = historical_anchor(
                reference_data,
                row_input,
                artifacts,
                float(output.loc[index, "prob_bonne_moyenne_modeles"]),
            )
            final_prob = float(anchor["final_probability"])
            source = anchor["source"]
            observed_labels.append("")
        final_probs.append(final_prob)
        final_predictions.append("Bonne" if final_prob >= threshold else "Mauvaise")
        conformity_sources.append(source)

    output["prediction_conforme_donnees"] = final_predictions
    output["prob_bonne_moyenne"] = final_probs
    output["source_conformite"] = conformity_sources
    if "bonne_recolte" in prepared.columns:
        output["cible_observee"] = observed_labels
    output["niveau_risque"] = output["prob_bonne_moyenne"].map(risk_level)
    return output


def build_input_frame(
    saison: str,
    province: str,
    culture: str,
    annee: int,
    altitude_m: float,
    pluviometrie_mm: float,
    temperature_moy_C: float,
    superficie_ha: float,
    nb_menages: int,
    utilisation_engrais: bool,
    acces_irrigation: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "saison": saison,
                "province": province,
                "culture": culture,
                "annee": int(annee),
                "altitude_m": float(altitude_m),
                "pluviometrie_mm": float(pluviometrie_mm),
                "temperature_moy_C": float(temperature_moy_C),
                "superficie_ha": float(superficie_ha),
                "nb_menages": int(nb_menages),
                "utilisation_engrais": int(utilisation_engrais),
                "acces_irrigation": int(acces_irrigation),
            }
        ]
    )


def build_html_report(
    record: dict[str, Any],
    predictions: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    selected_metrics: dict[str, Any],
) -> bytes:
    prediction_rows = "\n".join(
        (
            f"<tr><td>{escape(row['Modele'])}</td>"
            f"<td>{escape(row['Prediction'])}</td>"
            f"<td>{row['Probabilite bonne'] * 100:.1f}%</td>"
            f"<td>{row['Risque mauvaise'] * 100:.1f}%</td></tr>"
        )
        for _, row in predictions.iterrows()
    )
    rec_rows = "\n".join(
        (
            f"<tr><td>{escape(str(row['Priorite']))}</td>"
            f"<td>{escape(str(row['Action']))}</td>"
            f"<td>{escape(str(row['Raison']))}</td></tr>"
        )
        for _, row in recommendations_df.iterrows()
    )
    field_rows = "\n".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(display_value(value))}</td></tr>"
        for key, value in record.items()
    )
    html = f"""
    <!doctype html>
    <html lang="fr">
    <head>
      <meta charset="utf-8">
      <title>Rapport de prediction agricole</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 34px; color: #162027; }}
        h1 {{ color: #0f766e; margin-bottom: 6px; }}
        h2 {{ margin-top: 28px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; }}
        th, td {{ border: 1px solid #d9e2e0; padding: 8px 10px; text-align: left; }}
        th {{ background: #f1f5f4; }}
        .kpi {{ display: inline-block; border: 1px solid #d9e2e0; padding: 10px 14px; margin: 8px 8px 8px 0; border-radius: 8px; }}
        .note {{ color: #60717a; }}
      </style>
    </head>
    <body>
      <h1>Rapport de prediction agricole</h1>
      <p class="note">Genere le {escape(record['date'])}</p>
      <div class="kpi">Prediction : <strong>{escape(record['prediction'])}</strong></div>
      <div class="kpi">Probabilite bonne : <strong>{float(record['probabilite_bonne']) * 100:.1f}%</strong></div>
      <div class="kpi">AUC modele : <strong>{selected_metrics['auc']:.3f}</strong></div>
      <h2>Comparaison des modeles</h2>
      <table><tr><th>Modele</th><th>Prediction</th><th>Prob. bonne</th><th>Risque mauvaise</th></tr>{prediction_rows}</table>
      <h2>Recommandations</h2>
      <table><tr><th>Priorite</th><th>Action</th><th>Raison</th></tr>{rec_rows}</table>
      <h2>Donnees saisies</h2>
      <table>{field_rows}</table>
      <p class="note">Ce rapport aide la decision. Il ne remplace pas une expertise agronomique terrain.</p>
    </body>
    </html>
    """
    return html.encode("utf-8")


def classification_report_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    report = metrics.get("classification_report", {})
    labels = {
        "0": "Mauvaise recolte",
        "1": "Bonne recolte",
        "macro avg": "Moyenne macro",
        "weighted avg": "Moyenne ponderee",
    }
    rows = []
    for key, label in labels.items():
        values = report.get(key)
        if isinstance(values, dict):
            rows.append(
                {
                    "Classe": label,
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1": values.get("f1-score"),
                    "Support": values.get("support"),
                }
            )
    return pd.DataFrame(rows)


def course_checklist() -> pd.DataFrame:
    rows = [
        ("Ex. 1", "Exploration, valeurs manquantes, statistiques, visualisations", "Couverte"),
        ("Ex. 2", "Pretraitement, encodage, normalisation, train/test stratifie", "Couverte"),
        ("Ex. 3", "Arbre de decision, matrice de confusion, importance, overfitting", "Couverte"),
        ("Ex. 4", "Foret aleatoire, validation croisee, n_estimators", "Couverte"),
        ("Ex. 5", "Regression logistique, coefficients, ROC/AUC", "Couverte"),
        ("Ex. 6", "Scenarios reels du TP et recommandations", "Couverte"),
        ("Ex. 7", "Application web avec saisie, prediction, metriques et exports", "Couverte"),
    ]
    return pd.DataFrame(rows, columns=["Exercice", "Element demande", "Statut"])


def read_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    return clean_display_text(path.read_text(encoding="utf-8", errors="ignore"))


if not ARTIFACTS_PATH.exists():
    st.error("Les modeles ne sont pas encore entraines. Lancez `python src/train_models.py`.")
    st.stop()

artifacts, models = load_artifacts()
data = load_dataset()
categories = artifacts["categories"]
defaults = artifacts["defaults"]
features = artifacts["features"]
overview = metric_table(artifacts)
best_auc = overview.sort_values("AUC", ascending=False).iloc[0]
history = load_history()

with st.sidebar:
    st.markdown("### AgriPredict")
    theme = st.radio("Theme", ["Clair", "Sombre"], horizontal=True)
    default_model = st.selectbox("Modele principal", list(MODEL_FILES.keys()), index=1)
    decision_threshold = st.slider(
        "Seuil de bonne recolte",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Probabilite minimale pour classer une recolte comme bonne.",
    )
    st.markdown("---")
    st.caption("Donnees agricoles du Burundi, 2015-2023.")
    st.caption("Les colonnes rendement et production sont exclues du modele pour eviter le data leakage.")
    if not history.empty:
        st.download_button(
            "Exporter l'historique",
            data=display_frame(history).to_csv(index=False).encode("utf-8"),
            file_name="historique_predictions.csv",
            mime="text/csv",
            width="stretch",
        )

inject_css(theme)

st.markdown(
    """
    <div class="app-header">
      <div class="eyebrow">IA appliquee a l'agriculture / Burundi</div>
      <div class="app-title">AgriPredict Burundi</div>
      <p class="app-subtitle">
        Application Streamlit pour explorer les donnees agricoles, comparer trois modeles,
        predire les bonnes ou mauvaises recoltes et transformer les resultats en recommandations terrain.
      </p>
      <div class="badge-row">
        <span class="badge">Decision tree</span>
        <span class="badge">Random forest</span>
        <span class="badge">Logistic regression</span>
        <span class="badge">Scenarios du TP</span>
        <span class="badge">Export HTML / CSV</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

head_cols = st.columns(5)
head_cols[0].metric("Observations", f"{artifacts['data_summary']['shape'][0]:,}".replace(",", " "))
head_cols[1].metric("Provinces", artifacts["data_summary"]["n_provinces"])
head_cols[2].metric("Cultures", artifacts["data_summary"]["n_cultures"])
head_cols[3].metric("Meilleur AUC", f"{best_auc['AUC']:.3f}", best_auc["Modele"])
head_cols[4].metric("Historique", len(history))

(
    dashboard_tab,
    prediction_tab,
    simulator_tab,
    batch_tab,
    models_tab,
    data_tab,
    report_tab,
) = st.tabs(
    ["Accueil", "Prediction", "Simulation", "Import CSV", "Modeles", "Donnees", "Synthese TP"]
)

with dashboard_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Vue generale")
        st.dataframe(styled_metrics(overview.sort_values("AUC", ascending=False)), width="stretch", hide_index=True)

        chart_metrics = overview.set_index("Modele")[["Accuracy", "F1 macro", "AUC"]]
        st.bar_chart(chart_metrics)

        scenarios = load_report_csv("scenarios_utilises.csv")
        if not scenarios.empty:
            st.subheader("Scenarios du TP")
            scenario_predictions = predict_batch(models, scenarios, artifacts, decision_threshold, data)
            compact_cols = [
                "scenario",
                "province",
                "culture",
                "prediction_conforme_donnees",
                "consensus_ratio",
                "prob_bonne_moyenne",
                "source_conformite",
            ]
            compact = display_frame(scenario_predictions[compact_cols].copy())
            st.dataframe(
                compact.style.format({"consensus_ratio": "{:.0%}", "prob_bonne_moyenne": "{:.1%}"}),
                width="stretch",
                hide_index=True,
            )

    with right:
        st.subheader("Lecture rapide")
        target_rows = artifacts["data_summary"]["target_distribution"]
        class_1 = next((row for row in target_rows if row.get("classe") == 1.0), {})
        class_0 = next((row for row in target_rows if row.get("classe") == 0.0), {})
        missing_target = next((row for row in target_rows if pd.isna(row.get("classe"))), {})

        kpi_a, kpi_b = st.columns(2)
        with kpi_a:
            st.markdown(
                f"""
                <div class="mini-kpi">
                  <div class="mini-kpi-label">Modele conseille par AUC</div>
                  <div class="mini-kpi-value">{escape(str(best_auc['Modele']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_b:
            st.markdown(
                f"""
                <div class="mini-kpi">
                  <div class="mini-kpi-label">Classes positives</div>
                  <div class="mini-kpi-value">{class_1.get('pourcentage', 0):.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="soft-panel">
              <div class="section-label">Qualite des donnees</div>
              <p class="muted">
                Le dataset est desequilibre : {class_0.get('pourcentage', 0):.1f}% de mauvaises recoltes
                connues contre {class_1.get('pourcentage', 0):.1f}% de bonnes recoltes.
                Les lignes sans cible representent {missing_target.get('pourcentage', 0):.1f}%.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if history.empty:
            st.markdown('<div class="soft-panel">Aucune prediction sauvegardee pour le moment.</div>', unsafe_allow_html=True)
        else:
            last = history.tail(1).iloc[0]
            st.markdown(
                f"""
                <div class="soft-panel">
                  <div class="section-label">Derniere prediction</div>
                  <div class="mini-kpi-value">{escape(str(last['prediction']))}</div>
                  <p class="muted">
                    {escape(str(last['province']))} - {escape(display_value(last['culture']))},
                    {float(last['probabilite_bonne']) * 100:.1f}% de probabilite bonne.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(display_frame(history.tail(6)), width="stretch", hide_index=True)

        st.subheader("Couverture du TP")
        st.dataframe(course_checklist(), width="stretch", hide_index=True)

with prediction_tab:
    form_col, result_col = st.columns([1.04, 1])
    with form_col:
        st.subheader("Nouvelle prediction")
        preset_name = st.segmented_control(
            "Scenario",
            list(SCENARIO_VALUES.keys()),
            default="Saisie libre",
        )
        preset = SCENARIO_VALUES[preset_name]
        province_default = preset["province"] if preset else "Gitega"
        culture_default_value = culture_default(categories, preset)

        with st.form("prediction_form"):
            st.markdown('<div class="section-label">Modele</div>', unsafe_allow_html=True)
            selected_model = st.segmented_control(
                "Modele a utiliser",
                list(MODEL_FILES.keys()),
                default=default_model,
            )

            st.markdown('<div class="section-label">Localisation et culture</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            province = c1.selectbox(
                "Province",
                categories["province"],
                index=index_of(categories["province"], province_default),
            )
            culture = c2.selectbox(
                "Culture",
                categories["culture"],
                index=index_of(categories["culture"], culture_default_value),
                format_func=display_value,
            )
            saison = c3.selectbox(
                "Saison",
                categories["saison"],
                index=index_of(categories["saison"], defaults["saison"]),
            )

            st.markdown('<div class="section-label">Climat et parcelle</div>', unsafe_allow_html=True)
            c4, c5, c6 = st.columns(3)
            annee = c4.number_input("Annee", min_value=2015, max_value=2035, value=int(defaults["annee"]), step=1)
            altitude_m = c5.number_input(
                "Altitude (m)",
                min_value=0.0,
                max_value=3000.0,
                value=float(preset["altitude_m"] if preset else defaults["altitude_m"]),
                step=10.0,
            )
            pluviometrie_mm = c6.number_input(
                "Pluviometrie (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=float(preset["pluviometrie_mm"] if preset else defaults["pluviometrie_mm"]),
                step=10.0,
            )

            c7, c8, c9 = st.columns(3)
            temperature_moy_C = c7.number_input(
                "Temperature moyenne (C)",
                min_value=0.0,
                max_value=45.0,
                value=float(preset["temperature_moy_C"] if preset else defaults["temperature_moy_C"]),
                step=0.1,
            )
            superficie_ha = c8.number_input(
                "Superficie (ha)",
                min_value=0.01,
                max_value=100.0,
                value=float(defaults["superficie_ha"]),
                step=0.1,
            )
            nb_menages = c9.number_input(
                "Nombre de menages",
                min_value=1,
                max_value=10000,
                value=int(defaults["nb_menages"]),
                step=1,
            )

            st.markdown('<div class="section-label">Intrants et eau</div>', unsafe_allow_html=True)
            c10, c11 = st.columns(2)
            utilisation_engrais = c10.toggle(
                "Engrais utilise",
                value=bool(preset["utilisation_engrais"] if preset else defaults["utilisation_engrais"]),
            )
            acces_irrigation = c11.toggle(
                "Acces a l'irrigation",
                value=bool(preset["acces_irrigation"] if preset else defaults["acces_irrigation"]),
            )
            st.form_submit_button("Predire la recolte", width="stretch")

        benchmark = profile_selection(data, province, culture)
        b1, b2, b3 = st.columns(3)
        b1.metric("Ref. pluie mediane", format_number(benchmark["pluie_mediane"], " mm"))
        b2.metric("Ref. rendement", format_number(benchmark["rendement_moyen"], " t/ha"))
        b3.metric("Ref. bonne recolte", format_percent(benchmark["taux_bonne"]))

    input_row = build_input_frame(
        saison,
        province,
        culture,
        annee,
        altitude_m,
        pluviometrie_mm,
        temperature_moy_C,
        superficie_ha,
        nb_menages,
        utilisation_engrais,
        acces_irrigation,
    )

    pipeline = models[selected_model]
    model_probability = float(pipeline.predict_proba(input_row[features])[0, 1])
    anchor = historical_anchor(data, input_row, artifacts, model_probability)
    probability = float(anchor["final_probability"])
    prediction_label, risk_css, confidence = prediction_state(probability, decision_threshold)
    all_predictions = predict_all(models, input_row, features, decision_threshold)
    consensus, consensus_ratio = consensus_text(all_predictions)
    selected_metrics = artifacts["metrics"][MODEL_KEYS[selected_model]]
    model_disagreement = consensus_ratio < 1
    recs = recommendations(input_row.iloc[0], probability, decision_threshold, model_disagreement)
    sensitivity = local_sensitivity(pipeline, input_row, features)

    record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modele": selected_model,
        "seuil": decision_threshold,
        "prediction": prediction_label,
        "niveau_risque": risk_level(probability),
        "probabilite_bonne": round(probability, 4),
        "probabilite_modele_brut": round(model_probability, 4),
        "probabilite_historique_csv": round(anchor["historical_probability"], 4)
        if anchor["historical_probability"] is not None
        else None,
        "source_conformite": anchor["source"],
        "conformite_csv": anchor["conformity"],
        "consensus": round(consensus_ratio, 4),
        **input_row.iloc[0].to_dict(),
    }

    with result_col:
        st.subheader("Resultat")
        st.markdown(
            f"""
            <div class="{risk_css}">
              <div class="risk-title">{prediction_label}</div>
              <p class="risk-copy">{probability * 100:.1f}% de probabilite ancree aux donnees - {confidence}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{selected_metrics['accuracy']:.3f}")
        k2.metric("F1 macro", f"{selected_metrics['f1_macro']:.3f}")
        k3.metric("AUC", f"{selected_metrics['auc']:.3f}")
        k4.metric("Consensus", f"{consensus_ratio:.0%}")

        st.subheader("Conformite aux donnees du CSV")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Modele brut", format_percent(model_probability))
        a2.metric("Cas proches CSV", format_percent(anchor["historical_probability"]))
        a3.metric("Conformite", anchor["conformity"])
        a4.metric(
            "Distance moyenne",
            "-" if anchor["nearest_distance"] is None else f"{anchor['nearest_distance']:.3f}",
        )
        st.markdown(
            f"""
            <div class="soft-panel">
              Source de decision : <strong>{escape(anchor['source'])}</strong>.
              Reference utilisee : {escape(anchor['scope_label'])}.
            </div>
            """,
            unsafe_allow_html=True,
        )
        for message in anchor["out_of_domain"]:
            st.warning(message)
        nearest_display = anchor["nearest_cases"].copy()
        nearest_columns = [
            "annee",
            "saison",
            "province",
            "culture",
            "pluviometrie_mm",
            "temperature_moy_C",
            "utilisation_engrais",
            "acces_irrigation",
            "rendement_t_ha",
            "bonne_recolte",
            "distance_reference",
        ]
        nearest_display = display_frame(nearest_display[[column for column in nearest_columns if column in nearest_display.columns]])
        with st.expander("Voir les observations du CSV les plus proches"):
            st.dataframe(nearest_display, width="stretch", hide_index=True)

        st.markdown(f'<div class="soft-panel">{escape(consensus)}</div>', unsafe_allow_html=True)
        st.write("")
        prediction_display = all_predictions.copy()
        prediction_display["Probabilite bonne"] = prediction_display["Probabilite bonne"].map(format_percent)
        prediction_display["Risque mauvaise"] = prediction_display["Risque mauvaise"].map(format_percent)
        st.dataframe(prediction_display, width="stretch", hide_index=True)

        st.subheader("Recommandations terrain")
        st.dataframe(recs, width="stretch", hide_index=True)

        actions = st.columns(2)
        if actions[0].button("Enregistrer", width="stretch"):
            append_history(record)
            st.success("Prediction sauvegardee dans l'historique.")
        actions[1].download_button(
            "Rapport HTML",
            data=build_html_report(record, all_predictions, recs, selected_metrics),
            file_name="rapport_prediction_agricole.html",
            mime="text/html",
            width="stretch",
        )

    lower_left, lower_right = st.columns([1, 1])
    with lower_left:
        st.subheader("Sensibilite locale")
        sens_display = sensitivity.copy()
        sens_display["Probabilite"] = sens_display["Probabilite"].map(format_percent)
        sens_display["Impact"] = sens_display["Impact"].map(lambda value: f"{value * 100:+.1f} pts")
        st.dataframe(sens_display, width="stretch", hide_index=True)

    with lower_right:
        st.subheader("Variables globales influentes")
        chart = feature_chart(selected_model, artifacts)
        if not chart.empty:
            st.bar_chart(chart)

with simulator_tab:
    st.subheader("Simulation what-if")
    st.markdown(
        """
        <div class="soft-panel">
          Modifiez la pluie, la temperature, les engrais ou l'irrigation pour observer
          comment la probabilite de bonne recolte evolue.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    sim_model = st.segmented_control("Modele de simulation", list(MODEL_FILES.keys()), default=default_model)
    base_input = input_row.copy()

    s1, s2, s3, s4 = st.columns(4)
    rain_delta = s1.slider("Variation pluie (mm)", -400, 400, 0, 25)
    temp_delta = s2.slider("Variation temperature (C)", -6.0, 6.0, 0.0, 0.5)
    sim_fertilizer = s3.toggle("Avec engrais", value=bool(base_input.iloc[0]["utilisation_engrais"]))
    sim_irrigation = s4.toggle("Avec irrigation", value=bool(base_input.iloc[0]["acces_irrigation"]))

    simulated = base_input.copy()
    simulated.loc[simulated.index[0], "pluviometrie_mm"] = max(0.0, float(base_input.iloc[0]["pluviometrie_mm"]) + rain_delta)
    simulated.loc[simulated.index[0], "temperature_moy_C"] = max(0.0, float(base_input.iloc[0]["temperature_moy_C"]) + temp_delta)
    simulated.loc[simulated.index[0], "utilisation_engrais"] = int(sim_fertilizer)
    simulated.loc[simulated.index[0], "acces_irrigation"] = int(sim_irrigation)

    sim_pipeline = models[sim_model]
    base_model_prob = float(sim_pipeline.predict_proba(base_input[features])[0, 1])
    sim_model_prob = float(sim_pipeline.predict_proba(simulated[features])[0, 1])
    base_prob = float(historical_anchor(data, base_input, artifacts, base_model_prob)["final_probability"])
    sim_prob = float(historical_anchor(data, simulated, artifacts, sim_model_prob)["final_probability"])
    delta = sim_prob - base_prob

    sim_cols = st.columns(4)
    sim_cols[0].metric("Probabilite initiale", format_percent(base_prob))
    sim_cols[1].metric("Probabilite simulee", format_percent(sim_prob))
    sim_cols[2].metric("Impact", f"{delta * 100:+.1f} pts")
    sim_cols[3].metric("Risque simule", risk_level(sim_prob))

    sim_left, sim_right = st.columns([1, 1])
    with sim_left:
        st.subheader("Interventions rapides")
        quick = intervention_table(sim_pipeline, base_input, features, decision_threshold, data, artifacts)
        quick_display = quick.copy()
        quick_display["Probabilite bonne"] = quick_display["Probabilite bonne"].map(format_percent)
        quick_display["Gain vs base"] = quick_display["Gain vs base"].map(lambda value: f"{value * 100:+.1f} pts")
        st.dataframe(quick_display, width="stretch", hide_index=True)

    with sim_right:
        st.subheader("Parcelle simulee")
        st.dataframe(display_frame(simulated), width="stretch", hide_index=True)

    st.subheader("Sensibilite a la pluviometrie")
    st.line_chart(rainfall_curve(models, base_input, features))

with batch_tab:
    st.subheader("Prediction par fichier CSV")
    sample_row = pd.DataFrame(
        [
            {
                "saison": "A",
                "province": "Gitega",
                "culture": "Mais",
                "annee": int(defaults["annee"]),
                "altitude_m": float(defaults["altitude_m"]),
                "pluviometrie_mm": float(defaults["pluviometrie_mm"]),
                "temperature_moy_C": float(defaults["temperature_moy_C"]),
                "superficie_ha": float(defaults["superficie_ha"]),
                "nb_menages": int(defaults["nb_menages"]),
                "utilisation_engrais": "Oui",
                "acces_irrigation": "Non",
            }
        ]
    )

    c1, c2 = st.columns([.75, 1.25])
    with c1:
        st.download_button(
            "Telecharger un modele CSV",
            data=sample_row.to_csv(index=False).encode("utf-8"),
            file_name="modele_prediction_recoltes.csv",
            mime="text/csv",
            width="stretch",
        )
    with c2:
        uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded is not None:
        batch_input = pd.read_csv(uploaded)
        notes = validate_batch_input(batch_input, artifacts)
        for note in notes:
            st.warning(note)

        batch_output = predict_batch(models, batch_input, artifacts, decision_threshold, data)
        high_risk = int((batch_output["prob_bonne_moyenne"] < decision_threshold).sum())
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Lignes traitees", len(batch_output))
        bc2.metric("A surveiller", high_risk)
        bc3.metric("Consensus moyen", format_percent(batch_output["consensus_ratio"].mean()))

        batch_display = display_frame(batch_output.copy())
        prob_cols = [column for column in batch_display.columns if column.endswith("prob_bonne")]
        prob_cols += ["prob_bonne_moyenne_modeles", "prob_bonne_moyenne", "consensus_ratio"]
        st.dataframe(
            batch_display.style.format({column: "{:.1%}" for column in prob_cols}),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Telecharger les predictions",
            data=display_frame(batch_output).to_csv(index=False).encode("utf-8"),
            file_name="predictions_recoltes.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.dataframe(sample_row, width="stretch", hide_index=True)

with models_tab:
    st.subheader("Modeles et interpretation")
    m1, m2 = st.columns([1.08, 1])
    with m1:
        st.dataframe(styled_metrics(overview.sort_values("AUC", ascending=False)), width="stretch", hide_index=True)

        selected_report_model = st.selectbox("Rapport de classification", list(MODEL_FILES.keys()), index=1)
        report_df = classification_report_frame(artifacts["metrics"][MODEL_KEYS[selected_report_model]])
        st.dataframe(
            report_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Support": "{:.0f}"}),
            width="stretch",
            hide_index=True,
        )

        roc_path = FIGURES_DIR / "roc_comparaison_modeles.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Courbes ROC des trois modeles")

    with m2:
        selected_chart_model = st.selectbox("Importance / coefficients", list(MODEL_FILES.keys()), index=1)
        chart_df = feature_chart(selected_chart_model, artifacts)
        if not chart_df.empty:
            st.bar_chart(chart_df)
        st.markdown(
            """
            <div class="soft-panel">
              <strong>Note expert.</strong> La cible est tres desequilibree. L'accuracy seule peut etre
              flatteuse ; l'AUC, le F1 macro et les matrices de confusion donnent une lecture plus utile.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Matrices et hyperparametres")
    cm_cols = st.columns(3)
    for column, model_key in zip(cm_cols, ["decision_tree", "random_forest", "logistic_regression"]):
        path = FIGURES_DIR / f"confusion_matrix_{model_key}.png"
        if path.exists():
            column.image(str(path), caption=MODEL_LABELS[model_key])

    hp_left, hp_right = st.columns(2)
    with hp_left:
        depth_df = load_report_csv("accuracy_par_profondeur.csv")
        if not depth_df.empty:
            st.line_chart(depth_df.set_index("max_depth")[["train_accuracy", "test_accuracy"]])
    with hp_right:
        estimators_df = load_report_csv("accuracy_par_nombre_arbres.csv")
        if not estimators_df.empty:
            st.line_chart(estimators_df.set_index("n_estimators")[["test_accuracy"]])

with data_tab:
    st.subheader("Donnees et qualite")
    filter_cols = st.columns([1, 1, 1])
    province_filter = filter_cols[0].multiselect("Provinces", categories["province"])
    culture_filter = filter_cols[1].multiselect("Cultures", categories["culture"], format_func=display_value)
    year_min = int(data["annee"].min())
    year_max = int(data["annee"].max())
    year_range = filter_cols[2].slider("Annees", year_min, year_max, (year_min, year_max))

    filtered = data.copy()
    if province_filter:
        filtered = filtered[filtered["province"].isin(province_filter)]
    if culture_filter:
        filtered = filtered[filtered["culture"].isin(culture_filter)]
    filtered = filtered[(filtered["annee"] >= year_range[0]) & (filtered["annee"] <= year_range[1])]

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Lignes filtrees", len(filtered))
    dc2.metric("Pluie moyenne", format_number(filtered["pluviometrie_mm"].mean(), " mm"))
    dc3.metric("Rendement moyen", format_number(filtered["rendement_t_ha"].mean(), " t/ha"))
    dc4.metric("Bonne recolte", format_percent(filtered["bonne_recolte"].mean()))

    st.dataframe(display_frame(filtered.head(350)), width="stretch", hide_index=True)

    missing = load_report_csv("valeurs_manquantes.csv")
    target = load_report_csv("distribution_cible.csv")
    stats = load_report_csv("statistiques_numeriques.csv")
    yield_by_crop = load_report_csv("rendement_moyen_par_culture.csv")
    rain_by_province = load_report_csv("pluviometrie_moyenne_par_province.csv")

    dq1, dq2 = st.columns(2)
    with dq1:
        st.subheader("Valeurs manquantes")
        if not missing.empty:
            st.dataframe(missing, width="stretch", hide_index=True)
    with dq2:
        st.subheader("Distribution cible")
        if not target.empty:
            target_display = target.copy()
            target_display["classe"] = target_display["classe"].map(
                lambda value: "Manquante" if pd.isna(value) else ("Bonne" if float(value) == 1 else "Mauvaise")
            )
            st.dataframe(target_display, width="stretch", hide_index=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Rendement moyen par culture")
        if not yield_by_crop.empty:
            yc = display_frame(yield_by_crop).set_index("culture")
            st.bar_chart(yc)
    with chart_right:
        st.subheader("Pluviometrie moyenne par province")
        if not rain_by_province.empty:
            st.bar_chart(rain_by_province.set_index("province"))

    if not stats.empty:
        st.subheader("Statistiques numeriques")
        st.dataframe(stats, width="stretch")

    st.subheader("Figures du pipeline")
    figure_paths = [
        ("Rendement par culture", "boxplot_rendement_par_culture.png"),
        ("Production par annee", "lineplot_production_par_annee.png"),
        ("Engrais et bonnes recoltes", "barplot_bonnes_recoltes_engrais.png"),
        ("Correlations", "heatmap_correlation.png"),
    ]
    fig_cols = st.columns(2)
    for index, (caption, filename) in enumerate(figure_paths):
        path = FIGURES_DIR / filename
        if path.exists():
            fig_cols[index % 2].image(str(path), caption=caption)

    st.download_button(
        "Telecharger les donnees filtrees",
        data=display_frame(filtered).to_csv(index=False).encode("utf-8"),
        file_name="donnees_agriculture_burundi_filtrees.csv",
        mime="text/csv",
        width="stretch",
    )

with report_tab:
    st.subheader("Synthese du travail demande")
    st.dataframe(course_checklist(), width="stretch", hide_index=True)

    st.subheader("Reponses rapides aux points importants")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Periode", f"{artifacts['data_summary']['years'][0]}-{artifacts['data_summary']['years'][1]}")
    summary_cols[1].metric("Meilleure culture rendement", display_value(artifacts["data_summary"]["highest_mean_yield_culture"]))
    summary_cols[2].metric("Province plus pluvieuse", artifacts["data_summary"]["highest_mean_rain_province"])

    st.markdown(
        """
        <div class="soft-panel">
          Les colonnes <strong>rendement_t_ha</strong> et <strong>production_totale_t</strong>
          sont exclues de l'entrainement car elles revelent directement la cible ou une consequence
          directe du rendement. Les modeles utilisent donc les facteurs disponibles avant la recolte.
        </div>
        """,
        unsafe_allow_html=True,
    )

    report_text = read_markdown(REPORTS_DIR / "rapport_reflexion.md")
    notes_text = read_markdown(REPORTS_DIR / "notes_initiales.md")
    if report_text:
        with st.expander("Rapport court Q29-Q30", expanded=True):
            st.markdown(report_text)
    if notes_text:
        with st.expander("Notes initiales du pipeline"):
            st.markdown(notes_text)

    st.subheader("Livrables")
    st.markdown(
        """
        - Modele arbre de decision : `models/decision_tree.pkl`
        - Modele foret aleatoire : `models/random_forest.pkl`
        - Modele regression logistique : `models/logistic_regression.pkl`
        - Figures : dossier `figures/`
        - Rapports CSV et historique : dossier `reports/`
        - Application : `streamlit run app.py`
        """
    )
