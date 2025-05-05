#!/usr/bin/env python3
"""
Main execution script for the ChurnSense project.
This script runs the entire data and model pipeline.
"""

import argparse
import pandas as pd
from pathlib import Path

from src.utils.config import CONFIG
from src.pipeline.data_pipeline import run_data_pipeline
from src.pipeline.model_pipeline import run_model_pipeline
from src.data.data_loader import get_feature_names


def parse_args():
    parser = argparse.ArgumentParser(description="ChurnSense Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=CONFIG.data_path,
        help="Path to the raw data file",
    )

    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning",
    )

    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        choices=[
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "svc",
            "knn",
            "naive_bayes",
        ],
        default=None,
        help="Types of models to train",
    )

    return parser.parse_args()


def main():
    print("ChurnSense - Customer Churn Prediction Pipeline")

    args = parse_args()
    data_pipeline = run_data_pipeline(data_path=args.data_path)
    df = data_pipeline.get_data("featured")
    categorical_cols, numerical_cols = get_feature_names(df)

    print("Running model pipeline")
    model_pipeline = run_model_pipeline(
        df=df,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        model_types=args.model_types,
        tune_best=not args.no_tune,
    )

    print("\nPipeline completed!")
    print(f"Best model: {model_pipeline.best_model_name}")

    if hasattr(model_pipeline, "production_model_name"):
        print(f"Production model saved as: {model_pipeline.production_model_name}")

    print("You can now run the dashboard with: `python dashboard.py`")


if __name__ == "__main__":
    main()
