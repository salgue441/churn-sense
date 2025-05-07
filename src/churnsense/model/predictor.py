"""Model Prediction module for ChurnSense"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator

from churnsense.config import config
from churnsense.utils.logging import setup_logger, JsonLogger
from churnsense.utils.exceptions import PredictionError, ModelLoadError

logger = setup_logger(__name__)
prediction_logger = JsonLogger("predictions")


class ChurnPredictor:
    """Class for making predictions with trained churn models"""

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the churn predictor

        Args:
            model: Pre-loaded model instance
            model_path: Path to saved model
            threshold: Probability threshold for churn classification
        """

        self.model = model
        self.threshold = threshold
        self.feature_names = None
        self.model_metadata = {}

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk

        Args:
            model_path: Path to the saved model file

        Raises:
            ModelLoadError: If loading the model fails
        """

        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        try:
            logger.info(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)

            if isinstance(model_data, dict) and "model" in model_data:
                self.model = model_data["model"]
                self.model_metadata = model_data.get("metadata", {})
                self.feature_names = self.model_metadata.get("feature_names")

            else:
                self.model = model_data

            logger.info(f"Model loaded successfully from {model_path}")

        except Exception as e:
            error_msg = f"Error loading model from {model_path}: {str(e)}"

            logger.error(error_msg)
            raise ModelLoadError(error_msg, {"path": str(model_path)}) from e

    def predict(
        self,
        data: pd.DataFrame,
        return_proba: bool = True,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make predictions on customer data

        Args:
            data: DataFrame with customer features
            return_proba: Whether to include probabilities in the result
            threshold: Custom threshold for this prediction (overrides instance threshold)

        Returns:
            Dictionary with prediction results

        Raises:
            PredictionError: If prediction fails
        """

        if self.model is None:
            raise PredictionError("No model loaded for prediction")

        if threshold is None:
            threshold = self.threshold

        if self.feature_names:
            missing_features = [f for f in self.feature_names if f not in data.columns]
            if missing_features:
                raise PredictionError(
                    f"Missing required features for prediction",
                    {"missing_features": missing_features},
                )

            data = data[self.feature_names]

        try:
            logger.info(f"Making predictions on {len(data)} samples")

            prediction_start = pd.Timestamp.now()
            y_pred_proba = self.model.predict_proba(data)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            prediction_time = (pd.Timestamp.now() - prediction_start).total_seconds()

            result = {
                "predictions": y_pred.tolist(),
                "prediction_time": prediction_time,
                "threshold": threshold,
                "n_samples": len(data),
                "churn_count": int(y_pred.sum()),
                "churn_rate": float(y_pred.mean()),
            }

            if return_proba:
                result["probabilities"] = y_pred_proba.tolist()

            logger.info(
                f"Predictions complete: "
                f"{result['churn_count']} predicted churns "
                f"({result['churn_rate']*100:.2f}%)"
            )

            prediction_logger.log_event(
                "prediction",
                "Churn prediction complete",
                {
                    "n_samples": len(data),
                    "churn_count": int(y_pred.sum()),
                    "churn_rate": float(y_pred.mean()),
                    "threshold": threshold,
                    "prediction_time": prediction_time,
                },
            )

            return result

        except Exception as e:
            error_msg = f"Error making predictions: {str(e)}"

            logger.error(error_msg)
            raise PredictionError(error_msg) from e

    def predict_customer(
        self,
        customer_data: pd.Series,
        return_proba: bool = True,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make prediction for a single customer

        Args:
            customer_data: Series with customer features
            return_proba: Whether to include probability in the result
            threshold: Custom threshold for this prediction

        Returns:
            Dictionary with prediction result
        """

        df = pd.DataFrame([customer_data])
        result = self.predict(df, return_proba, threshold)
        customer_result = {
            "churn_predicted": bool(result["predictions"][0]),
            "prediction_time": result["prediction_time"],
            "threshold": result["threshold"],
        }

        if return_proba:
            customer_result["churn_probability"] = float(result["probabilities"][0])

        return customer_result

    def batch_predict(
        self,
        data: pd.DataFrame,
        id_column: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of customers and return as DataFrame

        Args:
            data: DataFrame with customer features
            id_column: Column to use as customer ID
            threshold: Custom threshold for this batch

        Returns:
            DataFrame with predictions and IDs
        """

        result = self.predict(data, return_proba=True, threshold=threshold)
        if id_column is None:
            id_column = config.id_column

        if id_column in data.columns:
            ids = data[id_column]

        else:
            ids = pd.Series(range(len(data)), name="CustomerID")

        df_result = pd.DataFrame(
            {
                "CustomerID": ids,
                "ChurnProbability": result["probabilities"],
                "ChurnPredicted": result["predictions"],
            }
        )

        df_result["RiskLevel"] = pd.cut(
            df_result["ChurnProbability"],
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        return df_result

    def explain_prediction(
        self,
        customer_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        top_features: int = 0,
    ) -> Dict[str, Any]:
        """
        Explain the factors contributing to a customer's churn prediction

        Args:
            customer_data: DataFrame with customer features (single row)
            feature_names: List of feature names (uses model metadata if None)
            top_features: Number of top contributing features to return

        Returns:
            Dictionary with explanation

        Note:
            This is a simple implementation using feature importance.
            For more advanced explanations, consider using SHAP or LIME.
        """

        if len(customer_data) != 1:
            raise ValueError("Customer data must contain exactly one row")

        if feature_names is None:
            feature_names = self.feature_names

        if feature_names is None:
            feature_names = customer_data.columns.tolist()

        try:
            prediction_result = self.predict(customer_data, return_proba=True)
            churn_predicted = bool(prediction_result["predictions"][0])
            churn_probability = float(prediction_result["probabilities"][0])
            feature_importance = {}

            if hasattr(self.model, "feature_importances_"):
                importance = self.model.feature_importances_
                feature_importance = dict(zip(feature_names, importance))

            elif hasattr(self.model, "coef_"):
                importance = np.abs(self.model.coef_[0])
                feature_importance = dict(zip(feature_names, importance))

            contributions = []
            for feature, value in customer_data.iloc[0].items():
                if feature not in feature_names:
                    continue

                importance = feature_importance.get(feature, 0)
                direction = "positive"
                if isinstance(value, (int, float)):
                    if value < customer_data[feature].median():
                        direction = "negative"

                contribution = {
                    "feature": feature,
                    "value": value,
                    "importance": importance,
                    "direction": direction,
                }

                contributions.append(contribution)

            contributions.sort(key=lambda x: x["importance"], reverse=True)
            top_contributions = contributions[:top_features]
            explanation = {
                "churn_predicted": churn_predicted,
                "churn_probability": churn_probability,
                "top_contributing_features": top_contributions,
                "feature_coverage": (
                    sum(c["importance"] for c in top_contributions)
                    / sum(feature_importance.values())
                    if feature_importance
                    else None
                ),
            }

            return explanation

        except Exception as e:
            error_msg = f"Error explaining prediction: {str(e)}"

            logger.error(error_msg)
            raise PredictionError(error_msg) from e

    def get_threshold_metrics(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Calculate metrics at different probability thresholds

        Args:
            X_test: Test feature data
            y_test: True target values
            thresholds: List of threshold values to evaluate

        Returns:
            DataFrame with metrics at each threshold
        """

        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            accuracy_score,
            confusion_matrix,
        )

        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics = {
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            }

            avg_customer_value = config.business_metrics.avg_customer_value
            retention_cost = config.business_metrics.retention_campaign_cost
            retention_success_rate = config.business_metrics.retention_success_rate

            # True positives: Correct churn predictions that we can act on
            retained_customers = tp * retention_success_rate
            retention_value = retained_customers * avg_customer_value

            # False positives: Incorrect churn predictions we waste resources on
            wasted_cost = fp * retention_cost

            # False negatives: Missed churn predictions
            missed_value = fn * avg_customer_value

            # Total cost and ROI
            total_cost = (tp + fp) * retention_cost
            total_benefit = retention_value
            net_benefit = total_benefit - total_cost
            roi = (net_benefit / total_cost) if total_cost > 0 else 0

            metrics.update(
                {
                    "retained_customers": retained_customers,
                    "retention_value": retention_value,
                    "wasted_cost": wasted_cost,
                    "missed_value": missed_value,
                    "total_cost": total_cost,
                    "total_benefit": total_benefit,
                    "net_benefit": net_benefit,
                    "roi": roi,
                }
            )

            results.append(metrics)

        return pd.DataFrame(results)

    def find_optimal_threshold(
        self, X_test: pd.DataFrame, y_test: pd.Series, metric: str = "roi"
    ) -> float:
        """
        Find the optimal threshold based on a specific metric

        Args:
            X_test: Test feature data
            y_test: True target values
            metric: Metric to optimize ('roi', 'f1', 'precision', 'recall')

        Returns:
            Optimal threshold value
        """

        metrics_df = self.get_threshold_metrics(X_test, y_test)

        if metric == "roi":
            best_idx = metrics_df["roi"].idxmax()

        elif metric == "f1":
            best_idx = metrics_df["f1"].idxmax()

        elif metric == "precision":
            best_idx = metrics_df["precision"].idxmax()

        elif metric == "recall":
            best_idx = metrics_df["recall"].idxmax()

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        optimal_threshold = metrics_df.loc[best_idx, "threshold"]
        optimal_value = metrics_df.loc[best_idx, metric]

        logger.info(
            f"Optimal threshold: {optimal_threshold:.2f} with {metric}={optimal_value:.4f}"
        )
        self.threshold = optimal_threshold

        return optimal_threshold
