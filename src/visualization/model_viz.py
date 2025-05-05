"""
Model visualization module for the ChurnSense project.
This module provides functions for visualizing model performance and insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from src.utils.config import CONFIG
from src.utils.helpers import save_fig


def plot_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save: bool = False,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot comprehensive model evaluation visualizations.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class
        model_name (str, optional): Name of the model for reporting
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. If None, uses model_name.

    Returns:
        plt.Figure: Matplotlib figure with evaluation plots
    """
    if filename is None:
        filename = f"model_evaluation_{model_name.lower().replace(' ', '_')}.png"

    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle(f"Model Evaluation: {model_name}", fontsize=20)

    # 1. Confusion Matrix (top left)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[0, 0], cbar=False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[0, 0].text(
                j + 0.5,
                i + 0.7,
                f"{cm_norm[i, j]:.1%}",
                ha="center",
                va="center",
                color="black" if cm_norm[i, j] < 0.5 else "white",
            )

    axs[0, 0].set_xlabel("Predicted", fontsize=14)
    axs[0, 0].set_ylabel("Actual", fontsize=14)
    axs[0, 0].set_title("Confusion Matrix", fontsize=16)
    axs[0, 0].set_xticklabels(["No Churn", "Churn"])
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
    axs[0, 1].set_xlabel("False Positive Rate", fontsize=14)
    axs[0, 1].set_ylabel("True Positive Rate", fontsize=14)
    axs[0, 1].set_title("ROC Curve", fontsize=16)
    axs[0, 1].legend(loc="lower right", fontsize=12)
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
    axs[1, 0].set_xlabel("Recall", fontsize=14)
    axs[1, 0].set_ylabel("Precision", fontsize=14)
    axs[1, 0].set_title("Precision-Recall Curve", fontsize=16)
    axs[1, 0].legend(loc="upper right", fontsize=12)
    axs[1, 0].grid(alpha=0.3)

    # 4. Performance Metrics Table (bottom right)
    axs[1, 1].axis("off")

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

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

    axs[1, 1].set_title("Performance Metrics", pad=30, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save:
        save_fig(fig, filename)

    return fig


def plot_feature_importance(
    model,
    feature_names: List[str],
    title: str = "Feature Importance",
    top_n: int = 20,
    save: bool = False,
    filename: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot feature importances for a model.

    Args:
        model: Trained model with feature_importances_ attribute or coefficients
        feature_names (List[str]): List of feature names
        title (str, optional): Title for the plot
        top_n (int, optional): Number of top features to show
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. If None, generates from title.

    Returns:
        Optional[plt.Figure]: Matplotlib figure with feature importance plot or None if not supported
    """

    if filename is None:
        filename = f"{title.lower().replace(' ', '_')}.png"

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

        if len(feature_names) != len(importances):
            print(
                f"Warning: Feature names length ({len(feature_names)}) doesn't match importances length ({len(importances)})"
            )

            feature_names = [f"Feature {i}" for i in range(len(importances))]

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )

        importance_df = importance_df.sort_values("Importance", ascending=False).head(
            top_n
        )

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.barplot(
            x="Importance", y="Feature", data=importance_df, ax=ax, palette="viridis"
        )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Importance Score", fontsize=14)
        ax.set_ylabel("Feature", fontsize=14)
        ax.tick_params(axis="y", labelsize=12)

        plt.tight_layout()

        if save:
            save_fig(fig, filename)

        return fig

    except Exception as e:
        print(f"Error plotting feature importance: {str(e)}")
        return None


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save: bool = False,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        normalize (bool, optional): Whether to normalize the confusion matrix. Default is True.
        title (str, optional): Title for the plot
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. If None, generates from title.

    Returns:
        plt.Figure: Matplotlib figure with confusion matrix plot
    """

    if filename is None:
        filename = f"{title.lower().replace(' ', '_')}.png"

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"

    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        square=True,
        cbar=True,
        ax=ax,
    )

    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save: bool = False,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve for a model.

    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class
        model_name (str, optional): Name of the model for the plot title
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. If None, generates from model_name.

    Returns:
        plt.Figure: Matplotlib figure with ROC curve plot
    """

    if filename is None:
        filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(10, 8))

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
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save: bool = False,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot precision-recall curve for a model.

    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class
        model_name (str, optional): Name of the model for the plot title
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. If None, generates from model_name.

    Returns:
        plt.Figure: Matplotlib figure with precision-recall curve plot
    """

    if filename is None:
        filename = f"precision_recall_curve_{model_name.lower().replace(' ', '_')}.png"

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        recall,
        precision,
        color="#e15759",
        lw=2,
        label=f"PR Curve (AP = {avg_precision:.3f})",
    )

    no_skill = len(y_true[y_true == 1]) / len(y_true)
    ax.plot(
        [0, 1],
        [no_skill, no_skill],
        color="gray",
        lw=1,
        linestyle="--",
        label=f"No Skill Classifier (AP = {no_skill:.3f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(f"Precision-Recall Curve - {model_name}", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    save: bool = False,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot calibration curve for a model.

    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class
        model_name (str, optional): Name of the model for the plot title
        n_bins (int, optional): Number of bins for the calibration curve. Default is 10.
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. If None, generates from model_name.

    Returns:
        plt.Figure: Matplotlib figure with calibration curve plot
    """
    
    if filename is None:
        filename = f"calibration_curve_{model_name.lower().replace(' ', '_')}.png"

    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=2,
        label=model_name,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect Calibration",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Predicted Probability", fontsize=14)
    ax.set_ylabel("Fraction of Positives", fontsize=14)
    ax.set_title(f"Calibration Curve - {model_name}", fontsize=16)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    sort_by: Optional[str] = None,
    ascending: bool = False,
    save: bool = False,
    filename: str = "model_comparison.png",
) -> plt.Figure:
    """
    Plot model comparison based on a specific metric.

    Args:
        results_df (pd.DataFrame): DataFrame containing model results with 'model_name' column
        metric (str, optional): Metric to compare. Default is "roc_auc".
        sort_by (str, optional): Metric to sort by. If None, uses the comparison metric.
        ascending (bool, optional): Whether to sort in ascending order. Default is False.
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is "model_comparison.png".

    Returns:
        plt.Figure: Matplotlib figure with model comparison plot
    """

    if sort_by is None:
        sort_by = metric

    if "model_name" not in results_df.columns:
        if "Model" in results_df.columns:
            results_df = results_df.rename(columns={"Model": "model_name"})

        else:
            raise ValueError("DataFrame must contain 'model_name' or 'Model' column")

    results_df = results_df.sort_values(sort_by, ascending=ascending)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        results_df["model_name"],
        results_df[metric],
        color=plt.cm.viridis(np.linspace(0, 1, len(results_df))),
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontweight="bold",
        )

    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    ax.set_title(f"Model Comparison by {metric.replace('_', ' ').title()}", fontsize=16)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_threshold_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[List[float]] = None,
    save: bool = False,
    filename: str = "threshold_metrics.png",
) -> plt.Figure:
    """
    Plot various metrics at different probability thresholds.

    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class
        thresholds (List[float], optional): List of thresholds to evaluate.
                                          If None, uses np.arange(0.1, 1.0, 0.1).
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is "threshold_metrics.png".

    Returns:
        plt.Figure: Matplotlib figure with threshold metrics plot
    """

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    metrics = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)

        metrics.append(
            {
                "Threshold": threshold,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Specificity": specificity,
            }
        )

    metrics_df = pd.DataFrame(metrics)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        metrics_df["Threshold"],
        metrics_df["Accuracy"],
        marker="o",
        label="Accuracy",
        linewidth=2,
    )

    ax.plot(
        metrics_df["Threshold"],
        metrics_df["Precision"],
        marker="s",
        label="Precision",
        linewidth=2,
    )

    ax.plot(
        metrics_df["Threshold"],
        metrics_df["Recall"],
        marker="^",
        label="Recall",
        linewidth=2,
    )

    ax.plot(
        metrics_df["Threshold"],
        metrics_df["F1 Score"],
        marker="D",
        label="F1 Score",
        linewidth=2,
    )

    ax.plot(
        metrics_df["Threshold"],
        metrics_df["Specificity"],
        marker="*",
        label="Specificity",
        linewidth=2,
    )

    ax.set_xlabel("Probability Threshold", fontsize=14)
    ax.set_ylabel("Metric Value", fontsize=14)
    ax.set_title(
        "Model Performance Metrics at Different Probability Thresholds", fontsize=16
    )
    ax.set_xlim([thresholds[0], thresholds[-1]])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)
    ax.text(0.51, 0.5, "Default Threshold (0.5)", rotation=90, alpha=0.7)
    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig
