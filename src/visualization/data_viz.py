"""
Data visualization module for the ChurnSense project.
This module provides functions for visualizing customer data and churn patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path

from src.utils.config import CONFIG
from src.utils.helpers import save_fig


def plot_churn_distribution(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    save: bool = False,
    filename: str = "churn_distribution.png",
) -> plt.Figure:
    """
    Plot the distribution of churn vs. non-churn customers.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        positive_class (str, optional): Value representing the positive class. If None, uses CONFIG["positive_class"].
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is "churn_distribution.png".

    Returns:
        plt.Figure: The matplotlib figure object
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if positive_class is None:
        positive_class = CONFIG.positive_class

    churn_counts = df[target_col].value_counts()
    churn_pct = df[target_col].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4ECDC4", "#FF6B6B"]

    if positive_class in churn_counts.index:
        colors = [
            "#4ECDC4" if idx != positive_class else "#FF6B6B"
            for idx in churn_counts.index
        ]

    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax[0], palette=colors)
    ax[0].set_title("Churn Distribution (Count)", fontsize=14)
    ax[0].set_xlabel(target_col, fontsize=12)
    ax[0].set_ylabel("Count", fontsize=12)

    for i, v in enumerate(churn_counts.values):
        ax[0].text(i, v + 0.1, f"{v:,}", ha="center", fontsize=12)

    ax[1].pie(
        churn_counts.values,
        labels=[
            f"{idx}\n({pct:.1f}%)"
            for idx, pct in zip(churn_counts.index, churn_pct.values)
        ],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "w", "linewidth": 1},
        textprops={"fontsize": 12},
    )

    ax[1].set_title("Churn Distribution (%)", fontsize=14)
    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_categorical_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    save: bool = False,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the relationship between a categorical feature and churn.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data
        feature (str): Name of the categorical feature to plot
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        positive_class (str, optional): Value representing the positive class. If None, uses CONFIG["positive_class"].
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is None (generates from feature name).

    Returns:
        plt.Figure: The matplotlib figure object
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if positive_class is None:
        positive_class = CONFIG.positive_class

    if filename is None:
        filename = f"categorical_feature_{feature}.png"

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    sns.countplot(
        x=feature, hue=target_col, data=df, ax=ax[0], palette=["#4ECDC4", "#FF6B6B"]
    )
    ax[0].set_title(f"Distribution by {feature}", fontsize=14)
    ax[0].set_xlabel(feature, fontsize=12)
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].tick_params(axis="x", rotation=45 if len(df[feature].unique()) > 5 else 0)

    churn_rate = (
        df.groupby(feature)[target_col]
        .apply(lambda x: (x == positive_class).mean() * 100)
        .reset_index()
    )
    churn_rate.columns = [feature, "Churn Rate (%)"]

    churn_rate = churn_rate.sort_values("Churn Rate (%)", ascending=False)
    overall_churn_rate = (df[target_col] == positive_class).mean() * 100

    sns.barplot(
        x=feature, y="Churn Rate (%)", data=churn_rate, ax=ax[1], palette="RdYlGn_r"
    )
    ax[1].axhline(
        y=overall_churn_rate,
        color="r",
        linestyle="--",
        label=f"Overall Churn Rate: {overall_churn_rate:.1f}%",
    )
    ax[1].set_title(f"Churn Rate by {feature}", fontsize=14)
    ax[1].set_xlabel(feature, fontsize=12)
    ax[1].set_ylabel("Churn Rate (%)", fontsize=12)
    ax[1].tick_params(axis="x", rotation=45 if len(df[feature].unique()) > 5 else 0)
    ax[1].legend()

    for i, v in enumerate(churn_rate["Churn Rate (%)"]):
        ax[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    contingency_table = pd.crosstab(df[feature], df[target_col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    plt.figtext(
        0.5,
        0.01,
        f"Chi-square test: χ² = {chi2:.2f}, p-value = {p:.4f}"
        + f" ({'Significant' if p < 0.05 else 'Not significant'} at α=0.05)",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save:
        save_fig(fig, filename)

    return fig


def plot_numerical_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    bins: int = 10,
    save: bool = False,
    filename: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the relationship between a numerical feature and churn.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data
        feature (str): Name of the numerical feature to plot
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        positive_class (str, optional): Value representing the positive class. If None, uses CONFIG["positive_class"].
        bins (int, optional): Number of bins for histograms. Default is 10.
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is None (generates from feature name).

    Returns:
        plt.Figure: The matplotlib figure object
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if positive_class is None:
        positive_class = CONFIG.positive_class

    if filename is None:
        filename = f"numerical_feature_{feature}.png"

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    churned = df[df[target_col] == positive_class][feature]
    retained = df[df[target_col] != positive_class][feature]

    sns.histplot(
        churned,
        ax=ax[0],
        kde=True,
        label=positive_class,
        alpha=0.6,
        color="#FF6B6B",
        bins=bins,
    )
    sns.histplot(
        retained,
        ax=ax[0],
        kde=True,
        label=f"Not {positive_class}",
        alpha=0.6,
        color="#4ECDC4",
        bins=bins,
    )

    ax[0].set_title(f"Distribution of {feature} by Churn Status", fontsize=14)
    ax[0].set_xlabel(feature, fontsize=12)
    ax[0].set_ylabel("Count", fontsize=12)
    ax[0].legend()

    sns.boxplot(
        x=target_col, y=feature, data=df, ax=ax[1], palette=["#4ECDC4", "#FF6B6B"]
    )
    ax[1].set_title(f"Box Plot of {feature} by Churn Status", fontsize=14)
    ax[1].set_xlabel(target_col, fontsize=12)
    ax[1].set_ylabel(feature, fontsize=12)

    mean_churned = churned.mean()
    mean_retained = retained.mean()
    se_churned = stats.sem(churned)
    se_retained = stats.sem(retained)

    ci_churned = se_churned * stats.t.ppf((1 + 0.95) / 2, len(churned) - 1)
    ci_retained = se_retained * stats.t.ppf((1 + 0.95) / 2, len(retained) - 1)
    barplot_data = pd.DataFrame(
        {
            "Churn Status": [positive_class, f"Not {positive_class}"],
            "Mean": [mean_churned, mean_retained],
            "CI": [ci_churned, ci_retained],
        }
    )

    sns.barplot(
        x="Churn Status",
        y="Mean",
        data=barplot_data,
        ax=ax[2],
        palette=["#FF6B6B", "#4ECDC4"],
    )
    ax[2].errorbar(
        x=[0, 1],
        y=[mean_churned, mean_retained],
        yerr=[ci_churned, ci_retained],
        fmt="none",
        color="black",
        capsize=5,
    )

    ax[2].set_title(f"Mean {feature} with 95% CI", fontsize=14)
    ax[2].set_xlabel("Churn Status", fontsize=12)
    ax[2].set_ylabel(f"Mean {feature}", fontsize=12)

    for i, v in enumerate([mean_churned, mean_retained]):
        ax[2].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)

    u_stat, p_value = stats.mannwhitneyu(churned, retained)
    plt.figtext(
        0.5,
        0.01,
        f"Mann-Whitney U test: U = {u_stat:.2f}, p-value = {p_value:.4f}"
        + f" ({'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05)",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save:
        save_fig(fig, filename)

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    save: bool = False,
    filename: str = "correlation_matrix.png",
) -> plt.Figure:
    """
    Plot a correlation matrix of numerical features.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data
        numerical_cols (List[str], optional): List of numerical columns to include.
                                             If None, uses all numeric columns.
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is "correlation_matrix.png".

    Returns:
        plt.Figure: The matplotlib figure object
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if df[target_col].dtype == "object":
        df_temp = df.copy()
        df_temp[target_col] = (df_temp[target_col] == CONFIG.positive_class).astype(int)

        corr_matrix = df_temp[numerical_cols + [target_col]].corr()

    else:
        corr_matrix = df[numerical_cols + [target_col]].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(12, 10))
    fig = plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_customer_segments(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    pca_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    save: bool = False,
    filename: str = "customer_segments.png",
) -> plt.Figure:
    """
    Plot customer segments from clustering.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data with cluster assignments
        cluster_col (str, optional): Name of the column containing cluster labels. Default is "Cluster".
        pca_cols (List[str], optional): List of PCA component columns for visualization.
                                       If None, uses "PCA1" and "PCA2" if available.
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        positive_class (str, optional): Value representing the positive class. If None, uses CONFIG["positive_class"].
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is "customer_segments.png".

    Returns:
        plt.Figure: The matplotlib figure object
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if positive_class is None:
        positive_class = CONFIG.positive_class

    if pca_cols is None:
        if "PCA1" in df.columns and "PCA2" in df.columns:
            pca_cols = ["PCA1", "PCA2"]

        else:
            if "TSNE1" in df.columns and "TSNE2" in df.columns:
                pca_cols = ["TSNE1", "TSNE2"]

            elif "UMAP1" in df.columns and "UMAP2" in df.columns:
                pca_cols = ["UMAP1", "UMAP2"]

            else:
                raise ValueError("No PCA, t-SNE, or UMAP columns found in DataFrame")

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    scatter1 = ax[0].scatter(
        df[pca_cols[0]],
        df[pca_cols[1]],
        c=df[cluster_col].astype("category").cat.codes,
        cmap="viridis",
        alpha=0.7,
        s=50,
    )

    legend1 = ax[0].legend(
        *scatter1.legend_elements(), title="Cluster", loc="upper right"
    )

    ax[0].add_artist(legend1)
    ax[0].set_title(f"Customer Segments ({pca_cols[0]} vs {pca_cols[1]})", fontsize=14)
    ax[0].set_xlabel(pca_cols[0], fontsize=12)
    ax[0].set_ylabel(pca_cols[1], fontsize=12)
    ax[0].grid(alpha=0.3)

    churn_map = {positive_class: 1, f"Not {positive_class}": 0}
    if df[target_col].dtype == "object":
        colors = np.where(df[target_col] == positive_class, "#FF6B6B", "#4ECDC4")

    else:
        colors = np.where(df[target_col] == 1, "#FF6B6B", "#4ECDC4")

    scatter2 = ax[1].scatter(
        df[pca_cols[0]], df[pca_cols[1]], c=colors, alpha=0.7, s=50
    )

    custom_legend = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF6B6B",
            markersize=10,
            label=positive_class,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4ECDC4",
            markersize=10,
            label=f"Not {positive_class}",
        ),
    ]

    ax[1].legend(handles=custom_legend, title=target_col, loc="upper right")
    ax[1].set_title(f"Customer Segments by Churn Status", fontsize=14)
    ax[1].set_xlabel(pca_cols[0], fontsize=12)
    ax[1].set_ylabel(pca_cols[1], fontsize=12)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig


def plot_churn_rate_by_segment(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    target_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    save: bool = False,
    filename: str = "churn_by_segment.png",
) -> plt.Figure:
    """
    Plot churn rate by customer segment.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data with cluster assignments
        cluster_col (str, optional): Name of the column containing cluster labels. Default is "Cluster".
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        positive_class (str, optional): Value representing the positive class. If None, uses CONFIG["positive_class"].
        save (bool, optional): Whether to save the plot. Default is False.
        filename (str, optional): Filename to save the plot. Default is "churn_by_segment.png".

    Returns:
        plt.Figure: The matplotlib figure object
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if positive_class is None:
        positive_class = CONFIG.positive_class

    if df[target_col].dtype == "object":
        churn_by_cluster = (
            df.groupby(cluster_col)[target_col]
            .apply(lambda x: (x == positive_class).mean() * 100)
            .reset_index()
        )

    else:
        churn_by_cluster = (
            df.groupby(cluster_col)[target_col]
            .apply(lambda x: x.mean() * 100)
            .reset_index()
        )

    churn_by_cluster.columns = [cluster_col, "Churn Rate (%)"]

    churn_by_cluster = churn_by_cluster.sort_values(cluster_col)
    cluster_sizes = df[cluster_col].value_counts().reset_index()
    cluster_sizes.columns = [cluster_col, "Count"]
    cluster_sizes["Percentage"] = (cluster_sizes["Count"] / len(df) * 100).round(1)

    cluster_data = churn_by_cluster.merge(cluster_sizes, on=cluster_col)
    cluster_data = cluster_data.sort_values("Churn Rate (%)", ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    bar1 = ax[0].bar(
        cluster_data[cluster_col].astype(str),
        cluster_data["Churn Rate (%)"],
        color=plt.cm.RdYlGn_r(cluster_data["Churn Rate (%)"] / 100),
    )

    for i, v in enumerate(cluster_data["Churn Rate (%)"]):
        ax[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    ax[0].set_title("Churn Rate by Customer Segment", fontsize=14)
    ax[0].set_xlabel("Segment", fontsize=12)
    ax[0].set_ylabel("Churn Rate (%)", fontsize=12)
    ax[0].set_ylim(0, max(cluster_data["Churn Rate (%)"]) * 1.1)
    ax[0].grid(axis="y", alpha=0.3)

    overall_rate = (
        (df[target_col] == positive_class).mean() * 100
        if df[target_col].dtype == "object"
        else df[target_col].mean() * 100
    )
    ax[0].axhline(
        y=overall_rate, color="r", linestyle="--", label=f"Overall: {overall_rate:.1f}%"
    )
    ax[0].legend()

    bar2 = ax[1].bar(
        cluster_data[cluster_col].astype(str), cluster_data["Count"], color="skyblue"
    )

    for i, (count, pct) in enumerate(
        zip(cluster_data["Count"], cluster_data["Percentage"])
    ):
        ax[1].text(i, count + 0.1, f"{count:,}\n({pct:.1f}%)", ha="center", fontsize=10)

    ax[1].set_title("Segment Sizes", fontsize=14)
    ax[1].set_xlabel("Segment", fontsize=12)
    ax[1].set_ylabel("Count", fontsize=12)
    ax[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save:
        save_fig(fig, filename)

    return fig
