"""
explainability.py

SHAP explainability layer for your existing Keras ANN churn model
(model.h5 + label_encoder_gender.pkl + onehot_encoder_geo.pkl + scaler.pkl).

Uses shap.KernelExplainer, which is model-agnostic — it works directly
against model.predict() without needing a tree-based or TF-specific
explainer, so it drops in next to your existing ANN with no retraining.

IMPORTANT: Set FEATURE_COLUMNS below to match the exact column order your
app.py already builds before calling scaler.transform(). This must match
what the model was trained on, or the SHAP attributions will be wrong.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------
# EDIT THIS to match your app.py's actual column order exactly.
# This is the standard order used by the common ANN churn tutorial
# structure (CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
# HasCrCard, IsActiveMember, EstimatedSalary, Geography_Germany,
# Geography_Spain) - adjust if yours differs.
# ----------------------------------------------------------------
FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
]

FEATURE_LABELS = {
    "CreditScore": "Credit Score",
    "Gender": "Gender (Male=1)",
    "Age": "Age",
    "Tenure": "Tenure (years)",
    "Balance": "Account Balance",
    "NumOfProducts": "Number of Products",
    "HasCrCard": "Has Credit Card",
    "IsActiveMember": "Active Member",
    "EstimatedSalary": "Estimated Salary",
    "Geography_Germany": "Located in Germany",
    "Geography_Spain": "Located in Spain",
}


def load_artifacts():
    model = keras.models.load_model(BASE_DIR / "model.h5")
    with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open(BASE_DIR / "onehot_encoder_geo.pkl", "rb") as f:
        ohe_geo = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, le_gender, ohe_geo, scaler


def build_input_row(raw: dict, le_gender, ohe_geo) -> pd.DataFrame:
    """Turn raw human-readable inputs into the encoded row your model expects.
    Mirrors the same encoding your app.py already does for prediction."""
    geo_encoded = ohe_geo.transform([[raw["Geography"]]])
    geo_cols = ohe_geo.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)
    # normalize to Geography_Germany / Geography_Spain naming if needed
    geo_df.columns = [c.replace("Geography_", "Geography_") for c in geo_df.columns]

    row = {
        "CreditScore": raw["CreditScore"],
        "Gender": le_gender.transform([raw["Gender"]])[0],
        "Age": raw["Age"],
        "Tenure": raw["Tenure"],
        "Balance": raw["Balance"],
        "NumOfProducts": raw["NumOfProducts"],
        "HasCrCard": raw["HasCrCard"],
        "IsActiveMember": raw["IsActiveMember"],
        "EstimatedSalary": raw["EstimatedSalary"],
    }
    df = pd.DataFrame([row])
    df = pd.concat([df, geo_df], axis=1)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


def get_background(scaler, n=30) -> np.ndarray:
    """Background dataset for KernelExplainer. Loads data/background_sample.pkl
    if you've saved one (recommended - a real sample of scaled X_train rows
    gives more faithful explanations). Falls back to a zero-centered synthetic
    background, which is a reasonable approximation since StandardScaler
    centers features at 0."""
    bg_path = BASE_DIR / "data" / "background_sample.pkl"
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            return pickle.load(f)
    n_features = len(FEATURE_COLUMNS)
    return np.random.normal(0, 0.5, size=(n, n_features))


def explain_prediction(raw: dict, model, le_gender, ohe_geo, scaler, nsamples: int = 100):
    """
    Returns:
        churn_probability: float
        shap_df: DataFrame with columns [Feature, Label, Value, SHAP]
                 sorted by SHAP contribution (descending = pushes toward churn)
    """
    input_df = build_input_row(raw, le_gender, ohe_geo)
    scaled_input = scaler.transform(input_df)

    proba = float(model.predict(scaled_input, verbose=0)[0][0])

    background = get_background(scaler)
    explainer = shap.KernelExplainer(
        lambda x: model.predict(x, verbose=0), background
    )
    shap_values = explainer.shap_values(scaled_input, nsamples=nsamples)

    # shap_values shape can be (1, n_features, 1) or (1, n_features) depending
    # on shap/keras version - normalize to a flat (n_features,) array
    sv = np.array(shap_values)
    sv = sv.reshape(-1)[: len(FEATURE_COLUMNS)]

    shap_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Label": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLUMNS],
            "Value": input_df.iloc[0].values,
            "SHAP": sv,
        }
    ).sort_values("SHAP", ascending=False)

    return proba, shap_df
