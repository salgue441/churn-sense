"""
Model evaluation module for the ChurnSense project.
This module handles evaluating model performance using various metrics.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from src.utils.config import CONFIG
from src.utils.helpers import timer_decorator, save_fig, save_evaluation_result


@timer_decorator
def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.

    Args:
        model (Any): Trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        model_name (str): Name of the model for reporting.

    Returns:
        Dict[str, Any]: Dictionary of performance metrics.
    """

    print(f"Model Evaluation: {model_name}")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "avg_precision": average_precision_score(y_test, y_pred_proba),
        "log_loss": log_loss(y_test, y_pred_proba),
        "brier_score": brier_score_loss(y_test, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results["tn"] = tn
    results["fp"] = fp
    results["fn"] = fn
    results["tp"] = tp
    results["specificity"] = tn / (tn + fp)
    results["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Print metrics
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Average Precision: {results['avg_precision']:.4f}")
    print(f"Specificity: {results['specificity']:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    fig = plot_model_evaluation(y_test, y_pred, y_pred_proba, model_name)
    save_fig(fig, f"model_evaluation_{model_name.lower().replace(' ', '_')}.png")

    save_evaluation_result(results, model_name.lower().replace(" ", "_"))

    return results


def plot_model_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
) -> plt.Figure:
    """
    Plot comprehensive model evaluation visualizations.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        model_name (str, optional): Name of the model for reporting.

    Returns:
        plt.Figure: Matplotlib figure with evaluation plots.
    """

    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(f"Model Evaluation: {model_name}", fontsize=20)

    # 1. Confusion Matrix (top left)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    axs[0, 0].set_title("Confusion Matrix")
    cm_plot = axs[0, 0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[0, 0].text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    axs[0, 0].set_xlabel("Predicted")
    axs[0, 0].set_ylabel("Actual")
    axs[0, 0].set_xticks([0, 1])
    axs[0, 0].set_xticklabels(["No Churn", "Churn"])
    axs[0, 0].set_yticks([0, 1])
    axs[0, 0].set_yticklabels(["No Churn", "Churn"])

    # 2. ROC Curve (top right)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    axs[0, 1].plot(
        fpr, tpr, color="#4e79a7", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})"
    )

    axs[0, 1].plot(
        [0, 1],
        [0, 1],
        color="gray",
        lw=1,
        linestyle="--",
        label="Random Classifier (AUC = 0.5)",
    )

    axs[0, 1].set_xlim([0.0, 1.0])
    axs[0, 1].set_ylim([0.0, 1.05])
    axs[0, 1].set_xlabel("False Positive Rate")
    axs[0, 1].set_ylabel("True Positive Rate")
    axs[0, 1].set_title("ROC Curve")
    axs[0, 1].legend(loc="lower right")
    axs[0, 1].grid(alpha=0.3)

    # 3. Precision-Recall Curve (bottom left)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    axs[1, 0].plot(
        recall,
        precision,
        color="#e15759",
        lw=2,
        label=f"PR Curve (AP = {avg_precision:.3f})",
    )

    no_skill = len(y_true[y_true == 1]) / len(y_true)
    axs[1, 0].plot(
        [0, 1],
        [no_skill, no_skill],
        color="gray",
        lw=1,
        linestyle="--",
        label=f"No Skill Classifier (AP = {no_skill:.3f})",
    )

    axs[1, 0].set_xlim([0.0, 1.0])
    axs[1, 0].set_ylim([0.0, 1.05])
    axs[1, 0].set_xlabel("Recall")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].set_title("Precision-Recall Curve")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].grid(alpha=0.3)

    # 4. Performance Metrics Table (bottom right)
    axs[1, 1].axis("off")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    metrics_data = [
        ["Metric", "Value"],
        ["Accuracy", f"{accuracy_score(y_true, y_pred):.4f}"],
        ["Precision", f"{precision_score(y_true, y_pred):.4f}"],
        ["Recall", f"{recall_score(y_true, y_pred):.4f}"],
        ["F1 Score", f"{f1_score(y_true, y_pred):.4f}"],
        ["ROC AUC", f"{roc_auc:.4f}"],
        ["Avg Precision", f"{avg_precision:.4f}"],
        ["Specificity", f"{specificity:.4f}"],
    ]

    table = axs[1, 1].table(
        cellText=metrics_data, loc="center", cellLoc="center", colWidths=[0.4, 0.3]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor("#4e79a7")
        table[(0, i)].set_text_props(color="white", fontweight="bold")

    axs[1, 1].set_title("Performance Metrics", pad=30)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    title: str = "Feature Importance",
    top_n: int = 20,
) -> Optional[plt.Figure]:
    """
    Plot feature importances for a model.

    Args:
        model (Any): Trained model with feature_importances_ attribute or coefficients.
        feature_names (List[str]): List of feature names.
        title (str, optional): Title for the plot.
        top_n (int, optional): Number of top features to show.

    Returns:
        Optional[plt.Figure]: Matplotlib figure with feature importance plot or None if not supported.
    """

    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            
        elif hasattr(model, "named_steps") and hasattr(
            model.named_steps["classifier"], "feature_importances_"
        ):
            importances = model.named_steps["classifier"].feature_importances_

        elif hasattr(model, "named_steps") and hasattr(
            model.named_steps["classifier"], "coef_"
        ):
            importances = np.abs(model.named_steps["classifier"].coef_[0])

        else:
            print("Model doesn't support direct feature importance extraction")
            return None

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )

        importance_df = importance_df.sort_values("Importance", ascending=False).head(
            top_n
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color="#4e79a7")

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Importance Score", fontsize=14)
        ax.set_ylabel("", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)

        ax.invert_yaxis()
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Error plotting feature importance: {str(e)}")
        return None


def analyze_feature_impact(
    model: Any, X_sample: pd.DataFrame, feature_names: List[str], top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze the impact of features on predictions by perturbing each feature.

    Args:
        model (Any): Trained model.
        X_sample (pd.DataFrame): Sample data to analyze.
        feature_names (List[str]): Names of features.
        top_n (int, optional): Number of top features to show.

    Returns:
        pd.DataFrame: DataFrame with feature impact analysis.
    """

    impacts = []
    baseline_pred = model.predict_proba(X_sample)[:, 1].mean()

    for feature in feature_names:
        if X_sample[feature].dtype == "object":
            continue

        X_perturbed = X_sample.copy()
        std_dev = X_sample[feature].std()
        X_perturbed[feature] = X_perturbed[feature] + std_dev

        new_pred = model.predict_proba(X_perturbed)[:, 1].mean()
        impact = new_pred - baseline_pred
        impacts.append(
            {
                "Feature": feature,
                "Impact": impact,
                "Direction": "Positive" if impact > 0 else "Negative",
            }
        )

    impact_df = pd.DataFrame(impacts)
    impact_df = impact_df.sort_values("Impact", key=abs, ascending=False).head(top_n)

    return impact_df


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute permutation feature importance for a model.

    Args:
        model (Any): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        n_repeats (int, optional): Number of times to permute each feature.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with permutation importance results.
    """

    if random_state is None:
        random_state = CONFIG["random_seed"]

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=CONFIG["n_jobs"],
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std
    perm_importance_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": importances_mean,
            "Std_Dev": importances_std,
        }
    )

    perm_importance_df = perm_importance_df.sort_values("Importance", ascending=False)

    return perm_importance_df


def evaluate_business_impact(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate the business impact of the model.

    Args:
        model (Any): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        threshold (float, optional): Probability threshold for positive class.

    Returns:
        Dict[str, Any]: Dictionary with business impact metrics.
    """

    avg_customer_value = CONFIG["business_metrics"]["avg_customer_value"]
    retention_campaign_cost = CONFIG["business_metrics"]["retention_campaign_cost"]
    retention_success_rate = CONFIG["business_metrics"]["retention_success_rate"]

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_customers = len(y_test)
    actual_churners = y_test.sum()
    predicted_churners = y_pred.sum()

    loss_without_model = actual_churners * avg_customer_value

    retained_customers = tp * retention_success_rate
    saved_revenue = retained_customers * avg_customer_value
    wasted_campaign_cost = fp * retention_campaign_cost
    missed_churn_cost = fn * avg_customer_value
    total_campaign_cost = predicted_churners * retention_campaign_cost

    net_benefit = saved_revenue - total_campaign_cost
    roi = (
        (saved_revenue - total_campaign_cost) / total_campaign_cost
        if total_campaign_cost > 0
        else 0
    )

    results = {
        "total_customers": total_customers,
        "actual_churners": int(actual_churners),
        "predicted_churners": int(predicted_churners),
        "retained_customers": float(retained_customers),
        "saved_revenue": float(saved_revenue),
        "total_campaign_cost": float(total_campaign_cost),
        "wasted_campaign_cost": float(wasted_campaign_cost),
        "missed_churn_cost": float(missed_churn_cost),
        "net_benefit": float(net_benefit),
        "roi": float(roi),
    }

    return results
