# 🏦 ChurnPredictor GenAI

### Bank Customer Churn Prediction with Explainable AI & GenAI-Powered Retention Strategies

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/Model-Keras%20ANN-orange)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-success)
![RAG](https://img.shields.io/badge/RAG-Policy%20Retrieval-purple)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

A churn model tells you *who* is likely to leave. This project goes further —
it also explains *why*, and generates a personalized, policy-grounded
retention strategy a bank's relationship manager could act on immediately.

**[Live Demo →](#)** *(add your Streamlit Cloud link here after deployment)*

---

## 🎯 The Problem

Most churn projects stop at a probability score. That's useful for a
dashboard, but not for a retention team — a manager still has to figure out
*why* the model flagged the customer and *what to actually do about it*.
This project closes that gap end-to-end.

## 🧠 How It Works

```
Customer data
      │
      ▼
Keras ANN  ──────────────▶  Churn Probability
      │
      ▼
SHAP (KernelExplainer) ──▶  Top drivers behind THIS prediction
      │
      ▼
Rule-based Retriever ────▶  Relevant internal retention policies
      │
      ▼
Gemini 2.5 Flash ────────▶  Personalized, policy-grounded retention strategy
```

Each stage feeds the next. The LLM at the end never predicts or decides
anything on its own — it only synthesizes what the ML and retrieval stages
already determined into a readable, actionable recommendation, and it's
explicitly instructed not to invent offers outside the retrieved policies.

## ✨ Features

- **Churn prediction** — Keras ANN (dense layers + dropout, class-weighted
  for imbalance) trained on standard bank-churn features
- **Explainability** — SHAP `KernelExplainer` shows exactly which features
  pushed each individual prediction up or down, not just global feature
  importance
- **Retrieval-Augmented Generation** — a transparent, rule-based retriever
  matches the customer's profile (balance, tenure, age, activity, product
  count, credit score) against a knowledge base of bank retention policies
- **GenAI-generated strategy** — Gemini writes a manager-ready retention
  plan grounded only in the retrieved policies and SHAP drivers
- **Interactive dashboard** — Streamlit app with a live risk gauge and a
  SHAP contribution chart per customer

## 🏗️ Project Structure

```
ChurnPredictorGenAI/
├── data/
│   ├── raw_churn.csv              # training data
│   ├── generate_data.py           # synthetic data generator (dev/demo)
│   ├── policy_knowledge_base.json # retention policies for the RAG layer
│   └── background_sample.pkl      # SHAP background (generated)
├── train_model.py                 # trains the ANN, saves all artifacts
├── explainability.py              # SHAP explanation layer
├── genai_retention.py             # policy retrieval + Gemini generation
├── app.py                         # Streamlit app
├── requirements.txt
└── .env.example
```

## 🚀 Quickstart

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
cp .env.example .env              # add your GEMINI_API_KEY
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey.

## 📊 Model Performance

*(Fill in with your numbers after training on your real dataset — printed
automatically by `train_model.py`.)*

| Metric | Value |
|---|---|
| ROC-AUC | — |
| Precision (Churn) | — |
| Recall (Churn) | — |
| F1-score (Churn) | — |

## 🔍 Explainability Example

For a given customer, the app shows the top SHAP-attributed drivers, e.g.:

| Feature | Value | Impact |
|---|---|---|
| Located in Germany | Yes | +0.14 (↑ churn risk) |
| Active Member | No | +0.06 (↑ churn risk) |
| Age | 58 | +0.03 (↑ churn risk) |

## 🤖 How the RAG + GenAI System Actually Works

This is the part worth understanding in detail — not just "it calls an LLM."

### 1. The knowledge base

`data/policy_knowledge_base.json` holds 9 internal bank retention policies,
each with an eligibility rule, an objective, concrete benefits, and a
recommended action — e.g. **High Balance Retention Program** (balance >
100,000 → dedicated relationship manager + preferential rates) or
**Inactive Customer Reactivation Program** (inactive members → reactivation
cashback + personal outreach). This is the fixed, ground-truth corpus the
LLM is allowed to draw from — it's never allowed to invent an offer outside
this file.

### 2. Retrieval — matching policies to the customer

`retrieve_relevant_policies()` in `genai_retention.py` scores every policy
against the customer's actual attributes and returns the top 4:

| Policy | Triggers when |
|---|---|
| Balance Retention | `Balance >= 100000` |
| Loyalty Rewards | `Tenure >= 5` |
| Senior Banking | `Age >= 55` |
| Reactivation / Digital Banking | `IsActiveMember == 0` |
| Low-Product Expansion | `NumOfProducts == 1` |
| Relationship Pricing | `NumOfProducts >= 3` |
| Credit Building | `CreditScore < 650` |
| Customer Care | always (baseline fallback) |

This is deliberately a **rule-based retriever**, not embedding-based
semantic search — every match is traceable to a single `if` condition,
which means you can explain, line by line, exactly why any given policy
was or wasn't retrieved for a customer. (See Future Improvements for the
semantic-search upgrade path.)

### 3. Grounding — what gets sent to the LLM

The generation prompt is built from three things, all computed *before*
the LLM is ever called:

- the raw customer profile (the form inputs)
- the top 3 **positive** SHAP drivers (features actually pushing this
  customer toward churn — not just globally important features)
- the retrieved policies from step 2, formatted as structured text

### 4. Generation — the actual prompt

```
You are a Senior Customer Retention Manager at a multinational retail bank.

CUSTOMER PROFILE
-----------------
{customer profile JSON}
Churn Probability: {proba:.1%}

KEY CHURN DRIVERS (from SHAP explainability)
-----------------------------------------------
{top 3 SHAP drivers with values and contribution scores}

INTERNAL BANK POLICIES (only use these, do not invent offers)
-----------------------------------------------
{retrieved policies: eligibility, objective, benefits, action}

TASK
-----
Write a concise, manager-friendly retention strategy (150-200 words) that:
1. Briefly explains why this customer is at risk, referencing the drivers above.
2. Recommends specific retention actions drawn ONLY from the policies listed.
3. Avoids inventing any offer, rate, or benefit not stated in the policies.
4. Ends with a clear, prioritized next action.
```

Sent to **Gemini 2.5 Flash** via `client.models.generate_content()`.

### 5. Worked example

**Input:** Age 58, Germany, Balance ₹150,000, 1 product, inactive member.

**SHAP output:** `Germany (+0.14)`, `Inactive (+0.08)`, `Age (+0.05)` → 77%
churn probability.

**Retrieved policies:** Reactivation Program, Digital Banking Adoption,
High Balance Retention, Senior Banking.

**Gemini's output** (illustrative — actual wording varies per run):
> This 58-year-old customer in Germany shows a 77% churn risk, primarily
> driven by inactivity and a high account balance with only one product —
> a profile at real risk of full attrition. Recommended action: enroll the
> customer in the **Inactive Customer Reactivation Program** with a
> reactivation cashback offer, paired with the **High Balance Retention
> Program** to assign a dedicated relationship manager given their
> ₹150,000 balance. Given their age, also flag them for **Senior Banking
> Benefits**. Priority: schedule a personal outreach call within 7 days,
> before pursuing digital onboarding.

Notice every recommendation traces back to a retrieved policy — that
traceability is the entire point of the RAG step, not a side effect of it.

### Why this design, not a raw LLM call

A prompt like *"suggest a retention offer for this customer"* with no
retrieval step invites the model to hallucinate benefits the bank doesn't
actually offer. Retrieval-then-generate keeps every recommendation
auditable against a real policy — which is what "grounded" actually means
in RAG, not just a buzzword.

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `SHAP` · `Streamlit` ·
`Plotly` · `Google Gemini (google-genai SDK)`

## 📈 Future Improvements

- Embedding-based semantic retrieval instead of rule-based scoring
- LLM-as-a-judge evaluation of generated strategies (relevance,
  groundedness, faithfulness, personalization, helpfulness)
- A/B testing framework to measure real retention-offer acceptance rates
- Batch scoring endpoint for the full customer base, not just one at a time

## 📄 License

MIT

## 👤 Author

Built by **Vaibhav** — engineering student focused on data science, ML,
and analytics.
