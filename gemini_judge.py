"""
ChurnGuard AI - LLM-as-a-Judge Evaluation

Reads evaluation/generation_results.csv (columns: CustomerID, ChurnProbability,
RetrievedPolicies, GeneratedStrategy) and asks Gemini to score each generated
strategy on Relevance, Groundedness, Faithfulness, Personalization, Helpfulness.

Run this AFTER you've generated a batch of strategies for a sample of
customers (e.g. by looping app_standalone.py's generate_retention_strategy()
over ~50-70 rows and saving the results to generation_results.csv).
"""

import json
import os
import time
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=API_KEY)
judge_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"},
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "generation_results.csv"
OUTPUT_PATH = BASE_DIR / "judge_scores.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"{INPUT_PATH} not found. Generate strategies for a sample of "
        "customers first and save them here (see module docstring)."
    )

df = pd.read_csv(INPUT_PATH)
results = []

print(f"Starting Gemini judge evaluation for {len(df)} customers...\n")

for idx, row in df.iterrows():
    print(f"Evaluating Customer {row['CustomerID']} ({idx + 1}/{len(df)})...")

    prompt = f"""You are an expert evaluator for banking RAG systems.

Evaluate ONLY the generated retention strategy below.

Customer Churn Probability: {row['ChurnProbability']}

Retrieved Policies:
{row['RetrievedPolicies']}

Generated Strategy:
{row['GeneratedStrategy']}

Score the strategy from 1 to 5 on each dimension:
- relevance: Does it address the customer's actual churn situation?
- groundedness: Is it supported by the retrieved policies?
- faithfulness: Does it avoid inventing offers or unsupported claims?
- personalization: Does it use customer-specific information?
- helpfulness: Would a retention manager find this practically useful?

Return ONLY a JSON object with these five integer keys."""

    start = time.perf_counter()
    try:
        response = judge_model.generate_content(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        scores = json.loads(response.text)

        results.append(
            {
                "CustomerID": row["CustomerID"],
                "Relevance": scores.get("relevance", 0),
                "Groundedness": scores.get("groundedness", 0),
                "Faithfulness": scores.get("faithfulness", 0),
                "Personalization": scores.get("personalization", 0),
                "Helpfulness": scores.get("helpfulness", 0),
                "JudgeLatency(ms)": round(latency_ms, 2),
            }
        )
        time.sleep(1)  # be polite to the free-tier rate limit

    except Exception as e:
        print(f"Failed Customer {row['CustomerID']}: {e}")

judge_df = pd.DataFrame(results)
judge_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("GEMINI JUDGE COMPLETE")
print("=" * 60)
print(f"Customers evaluated: {len(judge_df)}")
if len(judge_df):
    for col in ["Relevance", "Groundedness", "Faithfulness", "Personalization", "Helpfulness"]:
        print(f"Average {col}: {judge_df[col].mean():.2f} / 5")
print(f"\nSaved -> {OUTPUT_PATH}")
