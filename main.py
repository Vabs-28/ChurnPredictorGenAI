import json
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="ChurnGuard AI - Prediction API",
    description="Predicts bank customer churn and explains the prediction with SHAP.",
    version="1.0",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

model = joblib.load(ARTIFACTS_DIR / "churn_model.pkl")
with open(ARTIFACTS_DIR / "model_columns.json") as f:
    model_columns = json.load(f)
with open(ARTIFACTS_DIR / "model_config.json") as f:
    model_config = json.load(f)

threshold = model_config.get("threshold", 0.5)
explainer = shap.TreeExplainer(model)


class CustomerData(BaseModel):
    CreditScore: int = Field(..., example=650)
    Country: str = Field(..., example="Germany")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., example=45)
    Tenure: int = Field(..., example=3)
    Balance: float = Field(..., example=120000)
    NumOfProducts: int = Field(..., example=2)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=0)
    EstimatedSalary: float = Field(..., example=100000)


def get_risk_level(p: float) -> str:
    if p < 0.30:
        return "Low Risk"
    if p < 0.50:
        return "Moderate Risk"
    if p < 0.70:
        return "High Risk"
    return "Critical Risk"


def preprocess_input(c: CustomerData) -> pd.DataFrame:
    row = {
        "CreditScore": c.CreditScore,
        "Age": c.Age,
        "Tenure": c.Tenure,
        "Balance": c.Balance,
        "NumOfProducts": c.NumOfProducts,
        "HasCrCard": c.HasCrCard,
        "IsActiveMember": c.IsActiveMember,
        "EstimatedSalary": c.EstimatedSalary,
        "Country_Germany": 1 if c.Country == "Germany" else 0,
        "Country_Spain": 1 if c.Country == "Spain" else 0,
        "Gender_Male": 1 if c.Gender == "Male" else 0,
    }
    df = pd.DataFrame([row]).reindex(columns=model_columns, fill_value=0)
    return df


@app.get("/")
def home():
    return {
        "message": "ChurnGuard AI Prediction API is running.",
        "model": model_config.get("model_name"),
        "threshold": threshold,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    X = preprocess_input(customer)

    proba = float(model.predict_proba(X)[0][1])
    prediction = "Churn" if proba >= threshold else "No Churn"
    risk_level = get_risk_level(proba)

    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        churn_shap = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        churn_shap = shap_vals[0, :, 1]
    else:
        churn_shap = shap_vals[0]

    impacts = sorted(
        zip(model_columns, churn_shap), key=lambda x: x[1], reverse=True
    )[:5]
    top_drivers = [
        {"feature": k, "value": float(X[k].iloc[0]), "impact_score": round(float(v), 4)}
        for k, v in impacts
        if v > 0
    ]

    return {
        "churn_probability": round(proba, 4),
        "churn_probability_percentage": round(proba * 100, 2),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold_used": threshold,
        "top_churn_drivers": top_drivers,
    }
