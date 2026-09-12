"""
genai_retention.py

RAG layer: retrieves relevant bank retention policies for a given customer
(rule-based retriever - transparent and easy to explain in an interview),
then asks Gemini to write a personalized, policy-grounded retention strategy
using the SHAP drivers as context.
"""

import json
import os
from pathlib import Path

from google import genai

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "data" / "policy_knowledge_base.json") as f:
    BANK_POLICIES = json.load(f)


def retrieve_relevant_policies(raw: dict, top_k: int = 4) -> list:
    scored = []
    for policy_id, policy in BANK_POLICIES.items():
        score = 0
        if policy_id == "BALANCE_RETENTION" and raw["Balance"] >= 100000:
            score += 3
        if policy_id == "LOYALTY_REWARDS" and raw["Tenure"] >= 5:
            score += 3
        if policy_id == "SENIOR_BANKING" and raw["Age"] >= 55:
            score += 3
        if policy_id in ("REACTIVATION_PROGRAM", "DIGITAL_BANKING") and raw["IsActiveMember"] == 0:
            score += 4
        if policy_id == "LOW_PRODUCT_EXPANSION" and raw["NumOfProducts"] == 1:
            score += 3
        if policy_id == "RELATIONSHIP_PRICING" and raw["NumOfProducts"] >= 3:
            score += 3
        if policy_id == "CREDIT_BUILDING" and raw["CreditScore"] < 650:
            score += 2
        if policy_id == "CUSTOMER_CARE":
            score += 1
        scored.append((score, policy))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


def generate_retention_strategy(api_key: str, raw: dict, shap_df, churn_probability: float) -> str:
    client = genai.Client(api_key=api_key)

    policies = retrieve_relevant_policies(raw)
    policy_context = "\n----------------------------------------\n".join(
        f"Policy: {p['title']}\nEligibility: {p['eligibility']}\nObjective: {p['objective']}\n"
        f"Benefits: {', '.join(p['benefits'])}\nRecommended Action: {p['action']}"
        for p in policies
    )

    top_drivers = shap_df[shap_df["SHAP"] > 0].head(3)
    drivers_text = "\n\n".join(
        f"Feature: {r.Label}\nCustomer Value: {r.Value}\nSHAP Contribution: +{r.SHAP:.3f}"
        for r in top_drivers.itertuples()
    )

    prompt = f"""You are a Senior Customer Retention Manager at a multinational retail bank.

Your job is to retain valuable customers who are at risk of leaving.

CUSTOMER PROFILE
-----------------
{json.dumps(raw, indent=2)}
Churn Probability: {churn_probability:.1%}

KEY CHURN DRIVERS (from SHAP explainability)
-----------------------------------------------
{drivers_text if drivers_text else "No strong positive churn drivers identified."}

INTERNAL BANK POLICIES (only use these, do not invent offers)
-----------------------------------------------
{policy_context}

TASK
-----
Write a concise, manager-friendly retention strategy (150-200 words) that:
1. Briefly explains why this customer is at risk, referencing the drivers above.
2. Recommends specific retention actions drawn ONLY from the policies listed.
3. Avoids inventing any offer, rate, or benefit not stated in the policies.
4. Ends with a clear, prioritized next action.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text
