# churnsense/cli/main.py
"""Command-line interface for ChurnSense."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from churnsense.config import config
from churnsense.data.loader import load_data, get_feature_types
from churnsense.models.factory import ModelFactory
from churnsense.pipelines.data_pipeline import DataPipeline
from churnsense.pipelines.model_pipeline import ModelPipeline
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="ChurnSense - Customer Churn Prediction"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    data_parser = subparsers.add_parser("process", help="Process data")
    data_parser.add_argument(
        "--input", "-i", type=str, help="Input data file path", default=config.data_path
    )

    data_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output processed data file path",
        default=config.processed_data_path,
    )

    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--data",
        "-d",
        type=str,
        help="Processed data file path",
        default=config.processed_data_path,
    )

    train_parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        choices=[
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "knn",
            "all",
        ],
        default=["all"],
        help="Models to train",
    )

    train_parser.add_argument(
        "--tune", "-t", action="store_true", help="Tune hyperparameters"
    )


    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--model", type=str, help="Model file path", required=True
    )

    predict_parser.add_argument(
        "--data", "-d", type=str, help="Input data file path", required=True
    )

    predict_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output predictions file path",
        default="predictions.csv",
    )

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the dashboard")
    dashboard_parser.add_argument(
        "--port", type=int, default=8050, help="Dashboard port"
    )

    return parser.parse_args(args)


def process_data_command(args: argparse.Namespace) -> None:
    """
    Run the data processing pipeline.
    """

    logger.info("Running data processing pipeline")
    data_pipeline = DataPipeline()

    try:
        df = data_pipeline.run_pipeline(
            data_path=args.input,
            output_path=args.output,
        )
        
        logger.info(f"Data processing completed. Processed {len(df)} records.")

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        sys.exit(1)


def train_models_command(args: argparse.Namespace) -> None:
    """
    Train models on the processed data.
    """

    logger.info("Training models")

    try:
        df = load_data(args.data)
        feature_types = get_feature_types(df)

        if "all" in args.models:
            model_types = None 

        else:
            model_types = args.models

        model_pipeline = ModelPipeline()
        results = model_pipeline.run_pipeline(
            df=df,
            categorical_cols=feature_types["categorical"],
            numerical_cols=feature_types["numerical"],
            model_types=model_types,
            tune_best=args.tune,
        )

        logger.info(
            f"Model training completed. Best model: {results['best_model_name']}"
        )

    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        sys.exit(1)


def predict_command(args: argparse.Namespace) -> None:
    """
    Make predictions using a trained model.
    """

    import joblib

    logger.info(f"Making predictions using model {args.model}")

    try:
        model = joblib.load(args.model)
        df = load_data(args.data)

        X = df.drop(columns=[config.id_column, config.target_column], errors="ignore")

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        output_df = pd.DataFrame(
            {
                "CustomerID": (
                    df[config.id_column]
                    if config.id_column in df.columns
                    else range(len(df))
                ),
                "ChurnPrediction": predictions,
                "ChurnProbability": probabilities,
            }
        )

        output_df["RiskLevel"] = pd.cut(
            output_df["ChurnProbability"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        output_df.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        sys.exit(1)


def dashboard_command(args: argparse.Namespace) -> None:
    """
    Launch the ChurnSense dashboard.
    """
    try:
        from churnsense.dashboard.app import run_dashboard

        logger.info(f"Launching dashboard on port {args.port}")
        run_dashboard(port=args.port)

    except ImportError:
        logger.error(
            "Dashboard dependencies not installed. Install with 'pip install churnsense[dashboard]'"
        )

        sys.exit(1)

    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")
        sys.exit(1)


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the ChurnSense CLI.
    """
    
    parsed_args = parse_args(args)

    if parsed_args.command == "process":
        process_data_command(parsed_args)

    elif parsed_args.command == "train":
        train_models_command(parsed_args)

    elif parsed_args.command == "predict":
        predict_command(parsed_args)

    elif parsed_args.command == "dashboard":
        dashboard_command(parsed_args)

    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
