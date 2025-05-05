#!/usr/bin/env python3
"""
Script to generate predictions for new customer data.
"""

import argparse
import pandas as pd
import joblib
from pathlib import Path

from src.utils.config import CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="ChurnSense Prediction")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to CSV file with customer data",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file. If not provided, uses latest production model",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save prediction results",
    )

    return parser.parse_args()


def find_latest_production_model():
    model_dir = Path(CONFIG.models_path)
    production_models = list(model_dir.glob("production_*.pkl"))

    if not production_models:
        raise FileNotFoundError("No production models found")

    latest_model = max(production_models, key=lambda p: p.stat().st_mtime)
    return latest_model


def main():
    args = parse_args()
    customer_data = pd.read_csv(args.data_path)

    if args.model_path:
        model_path = args.model_path

    else:
        model_path = find_latest_production_model()

    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    try:
        if CONFIG.target_column in customer_data.columns:
            customer_data = customer_data.drop(columns=[CONFIG.target_column])

        print("Generating predictions")
        customer_ids = None

        if CONFIG.id_column in customer_data.columns:
            customer_ids = customer_data[CONFIG.id_column]
            X = customer_data.drop(columns=[CONFIG.id_column])

        else:
            X = customer_data

        churn_prob = model.predict_proba(X)[:, 1]
        churn_pred = (churn_prob >= 0.5).astype(int)

        results = pd.DataFrame(
            {
                "CustomerID": (
                    customer_ids if customer_ids is not None else range(len(X))
                ),
                "ChurnProbability": churn_prob,
                "ChurnPrediction": churn_pred,
                "RiskLevel": pd.cut(
                    churn_prob,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True,
                ),
            }
        )

        results.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")

        # Print summary
        print("\nPrediction Summary:")
        print(f"Total customers: {len(results)}")
        print(
            f"Predicted to churn: {results['ChurnPrediction'].sum()} ({results['ChurnPrediction'].mean()*100:.1f}%)"
        )
        print("\nRisk level distribution:")
        print(results["RiskLevel"].value_counts().sort_index())

    except Exception as e:
        print(f"Error generating predictions: {str(e)}")


if __name__ == "__main__":
    main()
