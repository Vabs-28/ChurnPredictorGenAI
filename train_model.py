"""
ChurnGuard AI - Model Training

Loads data/raw_churn.csv, cleans it, engineers features, trains and tunes a
RandomForestClassifier, chooses a decision threshold via F1, and saves
deployment artifacts to app/artifacts/:
    - churn_model.pkl        (trained sklearn model)
    - model_columns.json     (exact column order expected at inference)
    - model_config.json      (threshold + metadata)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw_churn.csv"
ARTIFACTS_DIR = BASE_DIR / "app" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------
df = pd.read_csv(DATA_PATH)

drop_cols = [c for c in ["CustomerId", "Surname", "RowNumber"] if c in df.columns]
df = df.drop(columns=drop_cols)
df = df.drop_duplicates()
assert df.isna().sum().sum() == 0, "Unexpected missing values — inspect data/raw_churn.csv"

# ----------------------------------------------------
# 2. Feature engineering (One-Hot Encoding, France/Female = baseline)
# ----------------------------------------------------
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
# drop_first keeps Geography_Germany, Geography_Spain, Gender_Male
df = df.rename(columns={
    "Geography_Germany": "Country_Germany",
    "Geography_Spain": "Country_Spain",
})

X = df.drop(columns=["Exited"])
y = df["Exited"]
model_columns = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------
# 3. Train + tune (small grid; widen if you have time to spare)
# ----------------------------------------------------
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [6, 10, None],
    "min_samples_leaf": [1, 3],
    "class_weight": ["balanced", None],
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
search = GridSearchCV(
    base_model, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=0
)
search.fit(X_train, y_train)
model = search.best_estimator_
print("Best params:", search.best_params_)

# ----------------------------------------------------
# 4. Threshold tuning via F1 on the test set
# ----------------------------------------------------
proba = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, proba >= t) for t in thresholds]
best_threshold = float(thresholds[int(np.argmax(f1_scores))])

y_pred = (proba >= best_threshold).astype(int)
print(f"\nChosen threshold: {best_threshold:.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# 5. Save artifacts
# ----------------------------------------------------
joblib.dump(model, ARTIFACTS_DIR / "churn_model.pkl")

with open(ARTIFACTS_DIR / "model_columns.json", "w") as f:
    json.dump(model_columns, f, indent=2)

with open(ARTIFACTS_DIR / "model_config.json", "w") as f:
    json.dump(
        {
            "model_name": "RandomForestClassifier",
            "threshold": best_threshold,
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "best_params": search.best_params_,
        },
        f,
        indent=2,
    )

print(f"\nSaved model + config to {ARTIFACTS_DIR}")
