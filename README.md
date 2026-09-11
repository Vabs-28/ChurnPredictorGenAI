# ChurnGuard AI

### End-to-End Customer Churn Prediction & AI-Powered Retention Strategy Generation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![RandomForest](https://img.shields.io/badge/ML-RandomForest-success)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-orange)
![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4)
![RAG](https://img.shields.io/badge/RAG-Retrieval--Augmented%20Generation-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Most churn models stop at a probability score. **ChurnGuard AI** goes further: it predicts churn, explains *why* using SHAP, retrieves relevant internal retention policies, and asks Gemini to write a personalized, policy-grounded retention strategy for a relationship manager to act on.

## Pipeline

```
Raw customer data
      |
Data cleaning & feature engineering
      |
SQL business analytics
      |
RandomForest churn model (tuned)
      |
SHAP explainability
      |
Rule-based policy retrieval (RAG)
      |
Gemini 1.5 Flash
      |
Personalized retention strategy
```

## Project Structure

```
ChurnGuardAI/
├── data/
│   ├── generate_data.py          # synthetic dataset (swap for your real csv)
│   ├── raw_churn.csv
│   └── policy_knowledge_base.json
├── sql/
│   └── churn_analysis_queries.sql
├── src/
│   └── train_model.py            # trains model, saves artifacts
├── app/
│   ├── main.py                   # FastAPI backend
│   ├── app_standalone.py         # Streamlit app (SHAP + Gemini RAG)
│   └── artifacts/                # churn_model.pkl, model_columns.json, model_config.json
├── evaluation/
│   └── gemini_judge.py           # LLM-as-a-judge evaluation
├── requirements.txt
├── .env.example
└── GUIDE.md                      # full stepwise build guide
```

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_data.py      # or drop in your own raw_churn.csv
python src/train_model.py
cp .env.example .env              # add your GEMINI_API_KEY
streamlit run app/app_standalone.py
```

See `GUIDE.md` for the full step-by-step walkthrough, including how to get a Gemini API key, run the FastAPI + Streamlit split version, and run the evaluation framework.

## Author

Built by Vaibhav.
