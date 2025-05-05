# churnsense/models/evaluation.py
"""Model evaluation module for ChurnSense."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churnsense.config import config
from churnsense.utils.logging import setup_logger
from churnsense.utils.visualization import save_fig

logger = setup_logger(__name__)


class ModelEvaluator:
    """
    Class for comprehensive model evaluation.
    """

    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize the model evaluator.

        Args:
            save_path: Path to save evaluation results. If None, uses config.
        """

        self.save_path = save_path or Path(config.evaluation_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        threshold: float = 0.5,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensively evaluate a model's performance.

        Args:
            model: Trained model to evaluate.
            X_test: Test features.
            y_test: Test target.
            model_name: Name of the model.
            threshold: Classification threshold.
            save_results: Whether to save evaluation results.

        Returns:
            Dictionary with evaluation metrics.
        """

        start_time = time.time()

        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            return {"error": str(e)}

        results = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        results["model_name"] = model_name
        results["threshold"] = threshold
        results["evaluation_time"] = time.time() - start_time

        try:
            self._generate_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)

        except Exception as e:
            logger.error(f"Error generating evaluation plots: {str(e)}")
            results["plot_error"] = str(e)

        try:
            feature_importance = self._calculate_feature_importance(
                model, X_test, y_test, model_name
            )

            if feature_importance is not None:
                results["feature_importance"] = feature_importance

        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")

        try:
            business_impact = self._calculate_business_impact(
                y_test, y_pred, y_pred_proba
            )

            results.update(business_impact)

        except Exception as e:
            logger.error(f"Error calculating business impact: {str(e)}")

        if save_results:
            self._save_evaluation_results(results, model_name)

        return results

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Dictionary with metrics.
        """

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "average_precision": average_precision_score(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba),
            "brier_score": brier_score_loss(y_true, y_pred_proba),
        }

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update(
            {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0),
                "npv": float(tn / (tn + fn) if (tn + fn) > 0 else 0),
                "balanced_accuracy": float(
                    (metrics["recall"] + metrics["specificity"]) / 2
                ),
            }
        )

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        }

        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        metrics["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["classification_report"] = report

        return metrics

    def _calculate_feature_importance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        n_repeats: int = 10,
    ) -> Optional[Dict[str, List[float]]]:
        """
        Calculate feature importance using permutation importance.

        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test target.
            model_name: Name of the model.
            n_repeats: Number of permutation repeats.

        Returns:
            Dictionary with feature importance scores or None if not applicable.
        """

        if not hasattr(model, "predict_proba"):
            return None

        try:
            result = permutation_importance(
                model,
                X_test,
                y_test,
                scoring="roc_auc",
                n_repeats=n_repeats,
                random_state=config.random_seed,
                n_jobs=config.n_jobs,
            )

            importance_dict = {
                "features": X_test.columns.tolist(),
                "importances_mean": result.importances_mean.tolist(),
                "importances_std": result.importances_std.tolist(),
            }

            self._plot_feature_importance(
                importance_dict["features"],
                importance_dict["importances_mean"],
                importance_dict["importances_std"],
                model_name,
            )

            return importance_dict

        except Exception as e:
            logger.error(f"Error calculating permutation importance: {str(e)}")
            return None

    def _calculate_business_impact(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate business impact metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities.

        Returns:
            Dictionary with business impact metrics.
        """

        avg_customer_value = config.avg_customer_value
        retention_campaign_cost = config.retention_campaign_cost
        retention_success_rate = config.retention_success_rate

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_customers = len(y_true)
        actual_churners = y_true.sum()
        predicted_churners = y_pred.sum()

        retained_customers = tp * retention_success_rate
        saved_revenue = retained_customers * avg_customer_value
        campaign_cost = predicted_churners * retention_campaign_cost
        wasted_campaign_cost = fp * retention_campaign_cost
        missed_churn_cost = fn * avg_customer_value

        net_benefit = saved_revenue - campaign_cost
        roi = (net_benefit / campaign_cost) if campaign_cost > 0 else 0

        return {
            "business_impact": {
                "total_customers": int(total_customers),
                "actual_churners": int(actual_churners),
                "predicted_churners": int(predicted_churners),
                "retained_customers": float(retained_customers),
                "saved_revenue": float(saved_revenue),
                "campaign_cost": float(campaign_cost),
                "wasted_campaign_cost": float(wasted_campaign_cost),
                "missed_churn_cost": float(missed_churn_cost),
                "net_benefit": float(net_benefit),
                "roi": float(roi),
            }
        }

    def _generate_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
    ) -> None:
        """
        Generate evaluation plots.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Predicted probabilities.
            model_name: Name of the model.
        """

        fig, axs = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Confusion Matrix (top left)
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(axs[0, 0], cm)

        # 2. ROC Curve (top right)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        self._plot_roc_curve(axs[0, 1], fpr, tpr, roc_auc)

        # 3. Precision-Recall Curve (bottom left)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        self._plot_pr_curve(axs[1, 0], precision, recall, avg_precision)

        # 4. Calibration Curve (bottom right)
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        self._plot_calibration_curve(axs[1, 1], prob_true, prob_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics_text = (
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
            f"ROC AUC: {roc_auc:.4f}\n"
            f"Avg Precision: {avg_precision:.4f}\n"
            f"Specificity: {specificity:.4f}"
        )

        axs[1, 1].text(
            0.05,
            0.50,
            metrics_text,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        fig.suptitle(f"Model Evaluation: {model_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        save_fig(fig, f"{model_name}_evaluation.png")

    def _plot_confusion_matrix(self, ax: plt.Axes, cm: np.ndarray) -> None:
        """
        Plot confusion matrix.

        Args:
            ax: Matplotlib axes.
            cm: Confusion matrix.
        """

        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        ax.set_title("Confusion Matrix")
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        thresh = cm.max() / 2

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No Churn", "Churn"])

    def _plot_roc_curve(
        self, ax: plt.Axes, fpr: np.ndarray, tpr: np.ndarray, roc_auc: float
    ) -> None:
        """
        Plot ROC curve.

        Args:
            ax: Matplotlib axes.
            fpr: False positive rates.
            tpr: True positive rates.
            roc_auc: ROC AUC score.
        """

        ax.plot(
            fpr,
            tpr,
            color="#4e79a7",
            lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            lw=1,
            linestyle="--",
            label="Random Classifier (AUC = 0.5)",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    def _plot_pr_curve(
        self,
        ax: plt.Axes,
        precision: np.ndarray,
        recall: np.ndarray,
        avg_precision: float,
    ) -> None:
        """
        Plot precision-recall curve.

        Args:
            ax: Matplotlib axes.
            precision: Precision values.
            recall: Recall values.
            avg_precision: Average precision score.
        """

        ax.plot(
            recall,
            precision,
            color="#e15759",
            lw=2,
            label=f"PR Curve (AP = {avg_precision:.3f})",
        )

        no_skill = np.sum(precision) / len(precision)
        ax.plot(
            [0, 1],
            [no_skill, no_skill],
            color="gray",
            lw=1,
            linestyle="--",
            label=f"No Skill (AP = {no_skill:.3f})",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    def _plot_calibration_curve(
        self, ax: plt.Axes, prob_true: np.ndarray, prob_pred: np.ndarray
    ) -> None:
        """
        Plot calibration curve.

        Args:
            ax: Matplotlib axes.
            prob_true: True probabilities.
            prob_pred: Predicted probabilities.
        """

        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=1,
            color="#4e79a7",
            label="Calibration Curve",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            linestyle="--",
            label="Perfect Calibration",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    def _plot_feature_importance(
        self,
        features: List[str],
        importances: List[float],
        std_devs: List[float],
        model_name: str,
        top_n: int = 15,
    ) -> None:
        """
        Plot feature importance.

        Args:
            features: Feature names.
            importances: Importance scores.
            std_devs: Standard deviations of importance scores.
            model_name: Name of the model.
            top_n: Number of top features to display.
        """

        importance_df = pd.DataFrame(
            {
                "Feature": features,
                "Importance": importances,
                "StdDev": std_devs,
            }
        )

        importance_df = importance_df.sort_values("Importance", ascending=False)
        importance_df = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            xerr=importance_df["StdDev"],
            color="#4e79a7",
            alpha=0.8,
            error_kw={"ecolor": "#e15759"},
        )

        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance: {model_name}")
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        save_fig(fig, f"{model_name}_feature_importance.png")

    def _save_evaluation_results(
        self, results: Dict[str, Any], model_name: str
    ) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results.
            model_name: Name of the model.
        """

        json_results = self._prepare_for_json(results)
        file_path = self.save_path / f"{model_name}_evaluation.json"

        with open(file_path, "w") as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Evaluation results saved to {file_path}")

    def _prepare_for_json(self, obj: Any) -> Any:
        """
        Prepare object for JSON serialization.

        Args:
            obj: Object to prepare.

        Returns:
            JSON-serializable object.
        """

        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)

        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)

        return obj
