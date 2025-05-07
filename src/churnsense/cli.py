"""
Command-line interface for ChurnSense.

This module provides CLI functionality for running the dashboard,
training models, and making predictions.
"""

import argparse
import sys
import os
from pathlib import Path

from churnsense.config import config
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


def run_dashboard():
    """
    Run the ChurnSense dashboard web application.

    This function starts with the Dash web server to serve
    the ChurnSense dashboard application.
    """

    try:
        from churnsense.dashboard.app import app

        parser = argparse.ArgumentParser(description="Run the ChurnSense Dashboard")

        parser.add_argument(
            "--host",
            type=str,
            default=config.host,
            help=f"Host to run the dashboard on (default: {config.host})",
        )

        parser.add_argument(
            "--port",
            type=int,
            default=config.port,
            help=f"Port to run the dashboard on (default: {config.port})",
        )

        parser.add_argument(
            "--debug",
            action="store_true",
            default=config.debug,
            help="Run in debug mode",
        )

        args = parser.parse_args()

        logger.info(f"Starting ChurnSense Dashboard at http://{args.host}:{args.port}/")
        logger.info(f"Debug mode: {args.debug}")

        app.run(host=args.host, port=args.port, debug=args.debug)

    except ImportError as e:
        logger.error(f"Failed to import dashboard components: {str(e)}")

        print(
            f"Error: Could not start dashboard. Make sure all dependencies are installed."
        )
        print(f"Missing dependency: {str(e)}")
        print("Try running: pip install -e .[dash]")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Error starting dashboard: {str(e)}")

        print(f"Error: {str(e)}")
        sys.exit(1)


def train_model():
    """
    Train a churn prediction model using the ML pipeline.

    This function runs the model training pipeline with
    specified configuration parameters.
    """

    try:
        from churnsense.pipeline.model_pipeline import ModelPipeline

        parser = argparse.ArgumentParser(description="Train a churn prediction model")
        parser.add_argument(
            "--data",
            type=str,
            default=str(config.data_path),
            help="Path to the data file",
        )

        parser.add_argument(
            "--save-dir",
            type=str,
            default=str(config.models_dir),
            help="Directory to save the trained model",
        )

        parser.add_argument(
            "--feature-engineering",
            action="store_true",
            default=True,
            help="Enable feature engineering",
        )

        parser.add_argument(
            "--tune-hyperparameters",
            action="store_true",
            default=True,
            help="Enable hyperparameter tuning",
        )

        args = parser.parse_args()
        logger.info(f"Starting model training with data from {args.data}")

        pipeline = ModelPipeline(data_path=args.data, models_path=args.save_dir)
        results = pipeline.run_pipeline(
            feature_engineering=args.feature_engineering,
            tune_hyperparameters=args.tune_hyperparameters,
        )

        logger.info(f"Model training completed successfully")
        logger.info(f"Best model: {results.get('best_model_name', 'Unknown')}")
        logger.info(f"Model saved to {args.save_dir}")

        print(f"✓ Model training completed successfully")
        print(f"  Best model: {results.get('best_model_name', 'Unknown')}")
        print(f"  Model saved to {args.save_dir}")

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


def predict():
    """
    Make predictions using a trained model.

    This function loads a trained model and makes predictions
    on new data.
    """

    try:
        import pandas as pd
        from churnsense.model.predictor import ChurnPredictor

        parser = argparse.ArgumentParser(description="Make churn predictions")
        parser.add_argument(
            "--data",
            type=str,
            required=True,
            help="Path to the data file for predictions",
        )

        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Path to the model file (default: latest production model)",
        )

        parser.add_argument(
            "--output",
            type=str,
            default="predictions.csv",
            help="Path to save prediction results",
        )

        parser.add_argument(
            "--threshold",
            type=float,
            default=None,
            help="Custom threshold for churn classification",
        )

        args = parser.parse_args()

        logger.info(f"Loading data from {args.data}")
        data = pd.read_csv(args.data)

        logger.info(f"Initializing predictor")
        predictor = ChurnPredictor(model_path=args.model)

        logger.info(f"Making predictions")
        results = predictor.batch_predict(data, threshold=args.threshold)

        results.to_csv(args.output, index=False)
        logger.info(f"Predictions saved to {args.output}")
        print(f"✓ Predictions completed successfully")
        print(f"  Predictions saved to {args.output}")
        print(
            f"  Predicted {results['ChurnPredicted'].sum()} customers likely to churn out of {len(results)}"
        )

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Use one of the following commands:")
    print("  churnsense-dashboard - Run the dashboard")
    print("  churnsense-train - Train a model")
    print("  churnsense-predict - Make predictions")
    sys.exit(1)
