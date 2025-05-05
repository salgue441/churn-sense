# churnsense/utils/visualization.py
"""Visualization utilities for ChurnSense."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from churnsense.config import config
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


def save_fig(fig: Figure, filename: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to disk.

    Args:
        fig: Figure to save.
        filename: Name of the file.
        dpi: Resolution in dots per inch.
    """

    figures_dir = Path(config.figures_path)
    figures_dir.mkdir(parents=True, exist_ok=True)

    file_path = figures_dir / filename
    fig.savefig(file_path, bbox_inches="tight", dpi=dpi)

    logger.info(f"Figure saved to {file_path}")


def plot_churn_distribution(df: pd.DataFrame) -> Figure:
    """
    Plot churn distribution.

    Args:
        df: DataFrame with churn data.

    Returns:
        Matplotlib figure.
    """

    if config.target_column not in df.columns:
        logger.warning(f"Target column '{config.target_column}' not found in data")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No target column found", ha="center", va="center")
        return fig

    churn_counts = df[config.target_column].value_counts()
    churn_pct = df[config.target_column].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        churn_counts.index,
        churn_counts.values,
        color=(
            ["#3498db", "#e74c3c"]
            if config.positive_class in churn_counts.index
            else None
        ),
    )

    for i, (count, pct) in enumerate(zip(churn_counts.values, churn_pct.values)):
        ax.text(
            i,
            count + (max(churn_counts.values) * 0.03),
            f"{pct:.1f}%",
            ha="center",
            fontweight="bold",
        )

    ax.set_xlabel(config.target_column)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {config.target_column}", fontsize=14)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{int(height):,}",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    ax.yaxis.grid(alpha=0.3)
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    max_features: int = 15,
) -> Figure:
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with feature importance scores.
        max_features: Maximum number of features to display.

    Returns:
        Matplotlib figure.
    """

    required_cols = ["Feature", "Importance"]
    if not all(col in importance_df.columns for col in required_cols):
        logger.warning("importance_df must have 'Feature' and 'Importance' columns")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Invalid data format", ha="center", va="center")
        return fig

    df = importance_df.sort_values("Importance", ascending=False).head(max_features)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        df["Feature"],
        df["Importance"],
        color="#3498db",
        alpha=0.8,
    )

    if "StdDev" in df.columns:
        ax.barh(
            df["Feature"],
            df["Importance"],
            xerr=df["StdDev"],
            ecolor="#e74c3c",
            capsize=5,
            alpha=0.8,
        )

    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance", fontsize=14)
    ax.invert_yaxis()
    ax.xaxis.grid(alpha=0.3)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["Importance"] + (max(df["Importance"]) * 0.02),
            i,
            f"{row['Importance']:.4f}",
            va="center",
        )

    return fig


def plot_customer_segments(
    df: pd.DataFrame,
    x_feature: str = "tenure",
    y_feature: str = "MonthlyCharges",
    cluster_col: str = "Cluster",
    churn_col: Optional[str] = None,
) -> Figure:
    """
    Plot customer segments.

    Args:
        df: DataFrame with customer data and cluster assignments.
        x_feature: Feature for x-axis.
        y_feature: Feature for y-axis.
        cluster_col: Column with cluster assignments.
        churn_col: Column with churn status.

    Returns:
        Matplotlib figure.
    """

    if cluster_col not in df.columns:
        logger.warning(f"Cluster column '{cluster_col}' not found in data")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cluster information found", ha="center", va="center")
        return fig

    if x_feature not in df.columns or y_feature not in df.columns:
        logger.warning(f"Features '{x_feature}' or '{y_feature}' not found in data")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Required features not found", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(12, 8))
    cluster_colors = plt.cm.tab10.colors

    if churn_col is not None and churn_col in df.columns:
        for cluster in sorted(df[cluster_col].unique()):
            cluster_df = df[df[cluster_col] == cluster]

            non_churned = cluster_df[cluster_df[churn_col] != config.positive_class]
            ax.scatter(
                non_churned[x_feature],
                non_churned[y_feature],
                s=50,
                c=[cluster_colors[cluster % len(cluster_colors)]],
                marker="o",
                alpha=0.7,
                label=f"Cluster {cluster} (Retained)",
            )

            churned = cluster_df[cluster_df[churn_col] == config.positive_class]
            ax.scatter(
                churned[x_feature],
                churned[y_feature],
                s=50,
                c=[cluster_colors[cluster % len(cluster_colors)]],
                marker="x",
                alpha=0.7,
                label=f"Cluster {cluster} (Churned)",
            )

    else:
        for cluster in sorted(df[cluster_col].unique()):
            cluster_df = df[df[cluster_col] == cluster]
            ax.scatter(
                cluster_df[x_feature],
                cluster_df[y_feature],
                s=50,
                c=[cluster_colors[cluster % len(cluster_colors)]],
                alpha=0.7,
                label=f"Cluster {cluster}",
            )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title("Customer Segments", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Figure:
    """
    Plot correlation matrix.

    Args:
        df: DataFrame with numerical features.
        columns: List of columns to include in correlation matrix.

    Returns:
        Matplotlib figure.
    """

    if columns is not None:
        valid_columns = [
            col
            for col in columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not valid_columns:
            logger.warning("No valid numerical columns specified")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No valid numerical columns", ha="center", va="center")
            return fig
        
        data = df[valid_columns]

    else:
        data = df.select_dtypes(include=["int64", "float64"])

        if data.empty:
            logger.warning("No numerical columns found in data")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numerical columns", ha="center", va="center")
            return fig

    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14)
    return fig


def plot_churn_risk_distribution(
    df: pd.DataFrame,
    risk_col: str = "ChurnProbability",
    risk_level_col: str = "RiskLevel",
) -> Figure:
    """
    Plot churn risk distribution.

    Args:
        df: DataFrame with churn risk data.
        risk_col: Column with churn risk scores.
        risk_level_col: Column with risk level categories.

    Returns:
        Matplotlib figure.
    """

    if risk_col not in df.columns:
        logger.warning(f"Risk column '{risk_col}' not found in data")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No risk data found", ha="center", va="center")
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Histogram of risk scores
    sns.histplot(df[risk_col], bins=20, kde=True, ax=ax1, color="#3498db")
    ax1.set_xlabel("Churn Risk Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Churn Risk Scores")

    threshold = 0.5  
    ax1.axvline(
        threshold, color="#e74c3c", linestyle="--", label=f"Threshold ({threshold})"
    )
    ax1.legend()

    # Plot 2: Risk level distribution
    if risk_level_col in df.columns:
        risk_counts = df[risk_level_col].value_counts().sort_index()
        risk_colors = {"Low": "#3498db", "Medium": "#f39c12", "High": "#e74c3c"}
        colors = [risk_colors.get(level, "#7f8c8d") for level in risk_counts.index]

        bars = ax2.bar(risk_counts.index, risk_counts.values, color=colors)
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max(risk_counts.values) * 0.03),
                f"{int(height):,}",
                ha="center",
            )

        ax2.set_xlabel("Risk Level")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Risk Levels")

        risk_pct = df[risk_level_col].value_counts(normalize=True).sort_index() * 100
        for i, (level, pct) in enumerate(risk_pct.items()):
            ax2.text(
                i,
                bars[i].get_height() / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    else:
        ax2.text(
            0.5,
            0.5,
            "No risk level data found",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    plt.tight_layout()
    return fig
