"""
Generates a synthetic bank-customer-churn dataset that matches the schema
of the standard 'Churn_Modelling.csv' dataset (CustomerId, Surname, CreditScore,
Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, Exited).

Use this ONLY as a placeholder to get the pipeline running end-to-end.
Replace data/raw_churn.csv with your real Churn_Modelling.csv (same column
names) before training the final model.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 10000

geographies = RNG.choice(["France", "Germany", "Spain"], size=N, p=[0.5, 0.25, 0.25])
genders = RNG.choice(["Male", "Female"], size=N, p=[0.55, 0.45])
age = RNG.normal(38, 10, N).clip(18, 92).round().astype(int)
credit_score = RNG.normal(650, 96, N).clip(350, 850).round().astype(int)
tenure = RNG.integers(0, 11, N)
num_products = RNG.choice([1, 2, 3, 4], size=N, p=[0.5, 0.4, 0.08, 0.02])
has_cr_card = RNG.choice([0, 1], size=N, p=[0.3, 0.7])
is_active = RNG.choice([0, 1], size=N, p=[0.48, 0.52])
salary = RNG.uniform(11, 200, N) * 1000

# Balance: many customers with 0 balance (single-product accounts), others with a real balance
has_balance = RNG.choice([0, 1], size=N, p=[0.36, 0.64])
balance = np.where(
    has_balance == 1,
    RNG.normal(100000, 45000, N).clip(1000, 250000),
    0.0,
).round(2)

# --- Churn probability model (drives realistic, learnable signal) ---
logit = (
    -1.55
    + 0.9 * (geographies == "Germany")
    + 0.30 * (age > 50)
    + 0.55 * (age > 60)
    - 0.9 * (is_active == 1)
    + 0.5 * (num_products >= 3)
    - 0.35 * (num_products == 2)
    + 0.25 * (balance > 100000)
    + 0.15 * (genders == "Female")
    - 0.10 * (tenure >= 7)
    + RNG.normal(0, 0.6, N)
)
prob = 1 / (1 + np.exp(-logit))
exited = RNG.binomial(1, prob)

df = pd.DataFrame(
    {
        "CustomerId": np.arange(15600000, 15600000 + N),
        "Surname": [f"Cust{i}" for i in range(N)],
        "CreditScore": credit_score,
        "Geography": geographies,
        "Gender": genders,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary.round(2),
        "Exited": exited,
    }
)

df.to_csv("data/raw_churn.csv", index=False)
print(f"Wrote data/raw_churn.csv -> {len(df)} rows, churn rate = {df['Exited'].mean():.2%}")
