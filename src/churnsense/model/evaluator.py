"""Model evaluation module for ChurnSense"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from churnsense.config import config
from churnsense.utils.logging import setup_logger
from churnsense.utils.exceptions import ModelEvaluationError

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Class for evaluating Churn prediction models
    """

    def __init__(self, results_path: Optional[Union[str, Path]] = None):
        """
        Initialize the model evaluator

        Args:
            results_path: Path to save evaluation results
        """

        if results_path is None:
            self.results_path = Path(config.evaluation_path)

        else:
            self.results_path = Path(results_path)

        self.results_path.mkdir(parents=True, exist_ok=True)
        self.evaluation_results = {}

    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test target values
            model_name: Name of the model for reporting

        Returns:
            Dictionary with evaluation metrics

        Raises:
            ModelEvaluationError: If evaluation fails
        """

        logger.info(f"Evaluating {model_name}")

        try:
            start_time = time.time()

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics["eval_time"] = time.time() - start_time

            logger.info(f"Evaluation results for {model_name}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

            self.evaluation_results[model_name] = metrics
            return metrics

        except Exception as e:
            error_msg = f"Error evaluating model: {str(e)}"

            logger.error(error_msg)
            raise ModelEvaluationError(error_msg, {"model": model_name}) from e

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation

        Args:
            y_true: True target values
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary with metrics
        """

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "average_precision": average_precision_score(y_true, y_pred_proba),
            "brier_score": brier_score_loss(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update(
            {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "npv": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
            }
        )

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)

        metrics.update(
            {
                "roc_curve": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist(),
                },
                "pr_curve": {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": (
                        pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
                    ),
                },
            }
        )

        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        metrics["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

        return metrics

    def compare_models(
        self, models: Dict[str, BaseEstimator], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test data

        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test: Test target values

        Returns:
            DataFrame with comparison results
        """

        logger.info(f"Comparing {len(models)} models")

        comparison_results = []
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)

            comparison_results.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "log_loss": metrics["log_loss"],
                    "kappa": metrics["kappa"],
                    "mcc": metrics["mcc"],
                }
            )

        df_comparison = pd.DataFrame(comparison_results)
        df_comparison = df_comparison.sort_values("roc_auc", ascending=False)

        comparison_path = self.results_path / "model_comparison.csv"
        df_comparison.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")

        return df_comparison

    def evaluate_feature_importance(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Evaluate feature importance using permutation importance

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            feature_names: List of feature names
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            n_jobs: Number of parallel jobs

        Returns:
            DataFrame with feature importance results
        """

        logger.info("Evaluating feature importance")

        if feature_names is None:
            feature_names = X_test.columns.tolist()

        if random_state is None:
            random_state = config.random_seed

        if n_jobs is None:
            n_jobs = config.n_jobs

        try:
            perm_importance = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs,
            )

            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": perm_importance.importances_mean,
                    "importance_std": perm_importance.importances_std,
                }
            )

            importance_df = importance_df.sort_values(
                "importance_mean", ascending=False
            )

            importance_path = self.results_path / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")

            return importance_df

        except Exception as e:
            error_msg = f"Error evaluating feature importance: {str(e)}"

            logger.error(error_msg)
            raise ModelEvaluationError(error_msg) from e

    def plot_roc_curve(
        self,
        models: Dict[str, BaseEstimator],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models

        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test: Test target values
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """

        plt.figure(figsize=(10, 8))
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Random Classifier (AUC = 0.5)",
        )

        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc_score:.3f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix == "":
                save_path = save_path / "roc_curve_comparison.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curve plot saved to {save_path}")

        return plt.gcf()

    def plot_pr_curve(
        self,
        models: Dict[str, BaseEstimator],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models

        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test: Test target values
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """

        plt.figure(figsize=(10, 8))

        no_skill = len(y_test[y_test == 1]) / len(y_test)
        plt.plot(
            [0, 1],
            [no_skill, no_skill],
            linestyle="--",
            color="gray",
            label=f"No Skill Classifier (AP = {no_skill:.3f})",
        )

        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)

            plt.plot(
                recall, precision, lw=2, label=f"{model_name} (AP = {ap_score:.3f})"
            )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves Comparison")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)

        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix == "":
                save_path = save_path / "pr_curve_comparison.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"PR curve plot saved to {save_path}")

        return plt.gcf()

    def plot_calibration_curve(
        self,
        models: Dict[str, BaseEstimator],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot calibration curves for multiple models

        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test: Test target values
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """

        plt.figure(figsize=(10, 8))
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated"
        )

        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
            brier = brier_score_loss(y_test, y_pred_proba)

            plt.plot(
                prob_pred,
                prob_true,
                lw=2,
                marker="o",
                label=f"{model_name} (Brier = {brier:.3f})",
            )

        plt.xlabel("Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curves Comparison")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)

        if save_path is not None:
            save_path = Path(save_path)

            if save_path.suffix == "":
                save_path = save_path / "calibration_curve_comparison.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Calibration curve plot saved to {save_path}")

        return plt.gcf()

    def plot_confusion_matrix(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        normalize: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix for a model

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            model_name: Name of the model
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """

        plt.figure(figsize=(8, 6))

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        if normalize:
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_data = cm_norm
            fmt = ".2f"
            title_suffix = " (Normalized)"

        else:
            cm_data = cm
            fmt = "d"
            title_suffix = ""

        sns.heatmap(
            cm_data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            cbar=False,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )

        if normalize:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j + 0.5,
                        i + 0.7,
                        f"({cm[i, j]})",
                        ha="center",
                        va="center",
                        color="black" if cm_norm[i, j] < 0.5 else "white",
                    )

        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"Confusion Matrix - {model_name}{title_suffix}")

        if save_path is not None:
            save_path = Path(save_path)
            if save_path.suffix == "":
                save_path = (
                    save_path
                    / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
                )

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix plot saved to {save_path}")

        return plt.gcf()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot feature importance

        Args:
            importance_df: DataFrame with feature importance values
            top_n: Number of top features to display
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """

        top_features = importance_df.head(top_n).copy()

        plt.figure(figsize=(12, 8))
        sns.barplot(
            x="importance_mean",
            y="feature",
            data=top_features,
            xerr=top_features["importance_std"],
        )

        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Features by Importance")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)

            if save_path.suffix == "":
                save_path = save_path / "feature_importance.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")

        return plt.gcf()

    def plot_metric_at_thresholds(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metrics: Optional[List[str]] = None,
        thresholds: Optional[np.ndarray] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot multiple metrics at different probability thresholds

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            metrics: List of metrics to plot
            thresholds: Array of threshold values
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """

        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1", "specificity"]

        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 19)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            result = {
                "threshold": threshold,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            }

            results.append(result)

        df_metrics = pd.DataFrame(results)
        plt.figure(figsize=(12, 8))

        for metric in metrics:
            plt.plot(
                df_metrics["threshold"],
                df_metrics[metric],
                marker="o",
                label=metric.capitalize(),
            )

        plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Metrics at Different Probability Thresholds")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)

        if save_path is not None:
            save_path = Path(save_path)

            if save_path.suffix == "":
                save_path = save_path / "threshold_metrics.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Threshold metrics plot saved to {save_path}")

        return plt.gcf()

    def evaluate_business_impact(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        thresholds: Optional[np.ndarray] = None,
        avg_customer_value: Optional[float] = None,
        retention_cost: Optional[float] = None,
        retention_success_rate: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Evaluate the business impact of the model

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target values
            thresholds: Array of threshold values
            avg_customer_value: Average value of a customer
            retention_cost: Cost of retention campaign per customer
            retention_success_rate: Success rate of retention campaign

        Returns:
            DataFrame with business metrics at different thresholds
        """

        logger.info("Evaluating business impact")

        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 19)

        if avg_customer_value is None:
            avg_customer_value = config.business_metrics.avg_customer_value

        if retention_cost is None:
            retention_cost = config.business_metrics.retention_campaign_cost

        if retention_success_rate is None:
            retention_success_rate = config.business_metrics.retention_success_rate

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        results = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Business metrics
            true_positives = tp  # Correctly predicted churns
            false_positives = fp  # Incorrectly predicted churns
            false_negatives = fn  # Missed churn predictions

            # True positives: Correct churn predictions we can act on
            retained_customers = true_positives * retention_success_rate
            retention_value = retained_customers * avg_customer_value

            # False positives: Incorrect churn predictions we waste resources on
            wasted_cost = false_positives * retention_cost

            # False negatives: Missed churn predictions
            missed_value = false_negatives * avg_customer_value

            # Total cost and ROI
            total_cost = (true_positives + false_positives) * retention_cost
            total_benefit = retention_value
            net_benefit = total_benefit - total_cost
            roi = (net_benefit / total_cost) if total_cost > 0 else 0

            results.append(
                {
                    "threshold": threshold,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
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

        df_business = pd.DataFrame(results)
        best_roi_idx = df_business["roi"].idxmax()
        best_roi_threshold = df_business.loc[best_roi_idx, "threshold"]
        best_roi = df_business.loc[best_roi_idx, "roi"]

        logger.info(
            f"Optimal threshold for ROI: {best_roi_threshold:.2f} with ROI = {best_roi:.2f}"
        )

        business_path = self.results_path / "business_impact.csv"
        df_business.to_csv(business_path, index=False)
        logger.info(f"Business impact analysis saved to {business_path}")

        return df_business

    def plot_business_impact(
        self,
        business_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        Plot business impact metrics

        Args:
            business_df: DataFrame with business metrics
            metrics: List of metrics to plot
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ["roi", "net_benefit", "retention_value", "wasted_cost"]

        plt.figure(figsize=(12, 8))

        fig, ax1 = plt.subplots(figsize=(12, 8))
        color = "tab:blue"
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("ROI", color=color)
        ax1.plot(business_df["threshold"], business_df["roi"], color=color, marker="o")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Value ($)", color=color)

        for i, metric in enumerate(metrics):
            if metric == "roi":
                continue

            ax2.plot(
                business_df["threshold"],
                business_df[metric],
                marker="x",
                label=metric.replace("_", " ").title(),
            )

        best_roi_idx = business_df["roi"].idxmax()
        best_roi_threshold = business_df.loc[best_roi_idx, "threshold"]

        plt.axvline(
            x=best_roi_threshold,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal ROI Threshold = {best_roi_threshold:.2f}",
        )

        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(loc="center right")

        plt.title("Business Impact at Different Thresholds")
        plt.grid(alpha=0.3)

        if save_path is not None:
            save_path = Path(save_path)

            if save_path.suffix == "":
                save_path = save_path / "business_impact.png"

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Business impact plot saved to {save_path}")

        return plt.gcf()

    def save_evaluation_results(
        self, model_name: str, output_format: str = "json"
    ) -> None:
        """
        Save evaluation results to file

        Args:
            model_name: Name of the model
            output_format: Output format ('json' or 'csv')
        """

        if model_name not in self.evaluation_results:
            logger.warning(f"No evaluation results found for {model_name}")
            return

        results = self.evaluation_results[model_name]

        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        results[key][subkey] = subvalue.tolist()

                    elif isinstance(subvalue, (np.integer, np.floating)):
                        results[key][subkey] = float(subvalue
                                                     )
            elif isinstance(value, np.ndarray):
                results[key] = value.tolist()

            elif isinstance(value, (np.integer, np.floating)):
                results[key] = float(value)

        if output_format.lower() == "json":
            output_path = self.results_path / f"{model_name}_evaluation.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        else:
            flat_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if not isinstance(subvalue, (list, dict)):
                            flat_results[f"{key}_{subkey}"] = subvalue

                elif not isinstance(value, (list, dict)):
                    flat_results[key] = value

            output_path = self.results_path / f"{model_name}_evaluation.csv"
            pd.DataFrame([flat_results]).to_csv(output_path, index=False)

        logger.info(f"Evaluation results saved to {output_path}")

    def generate_evaluation_report(
        self, model_name: str, show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report

        Args:
            model_name: Name of the model
            show_plots: Whether to display plots

        Returns:
            Dictionary with report data
        """

        if model_name not in self.evaluation_results:
            logger.warning(f"No evaluation results found for {model_name}")
            return {}

        results = self.evaluation_results[model_name]
        report = {
            "model_name": model_name,
            "evaluation_time": results.get("eval_time", None),
            "metrics": {
                "accuracy": results.get("accuracy", None),
                "precision": results.get("precision", None),
                "recall": results.get("recall", None),
                "f1": results.get("f1", None),
                "roc_auc": results.get("roc_auc", None),
                "avg_precision": results.get("average_precision", None),
                "log_loss": results.get("log_loss", None),
                "brier_score": results.get("brier_score", None),
            },
            "confusion_matrix": {
                "true_positives": results.get("true_positives", None),
                "false_positives": results.get("false_positives", None),
                "true_negatives": results.get("true_negatives", None),
                "false_negatives": results.get("false_negatives", None),
                "specificity": results.get("specificity", None),
                "npv": results.get("npv", None),
            },
        }

        report_path = self.results_path / f"{model_name}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {report_path}")

        return report
