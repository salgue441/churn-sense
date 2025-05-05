"""
Interactive visualization module for the ChurnSense project.
This module provides functions for creating interactive plots using Plotly.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from src.utils.config import CONFIG


def create_churn_dashboard(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    positive_class: Optional[str] = None,
) -> Dict[str, go.Figure]:
    """
    Create a set of interactive plots for a churn dashboard.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        positive_class (str, optional): Value representing the positive class. If None, uses CONFIG["positive_class"].

    Returns:
        Dict[str, go.Figure]: Dictionary of Plotly figures for dashboard
    """
    
    if target_col is None:
        target_col = CONFIG.target_column

    if positive_class is None:
        positive_class = CONFIG.positive_class

    dashboard_plots = {}

    # 1. Churn Distribution Pie Chart
    churn_counts = df[target_col].value_counts().reset_index()
    churn_counts.columns = [target_col, "Count"]

    pie_fig = px.pie(
        churn_counts,
        names=target_col,
        values="Count",
        title="Customer Churn Distribution",
        color=target_col,
        color_discrete_map={positive_class: "#FF6B6B", "No": "#4ECDC4"},
        hole=0.4,
    )

    pie_fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}",
    )

    pie_fig.update_layout(
        title_font_size=16,
        legend_title_font_size=14,
        legend_font_size=12,
    )

    dashboard_plots["churn_distribution"] = pie_fig

    # 2. Monthly Charges vs. Tenure Scatter Plot
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        scatter_fig = px.scatter(
            df,
            x="tenure",
            y="MonthlyCharges",
            color=target_col,
            color_discrete_map={positive_class: "#FF6B6B", "No": "#4ECDC4"},
            opacity=0.7,
            title="Monthly Charges vs. Tenure",
            labels={
                "tenure": "Tenure (months)",
                "MonthlyCharges": "Monthly Charges ($)",
            },
        )

        scatter_fig.update_layout(
            title_font_size=16,
            legend_title_font_size=14,
            legend_font_size=12,
        )

        dashboard_plots["monthly_charges_vs_tenure"] = scatter_fig

    # 3. Churn Rate by Contract Type
    if "Contract" in df.columns:
        contract_churn = (
            df.groupby("Contract")[target_col]
            .apply(lambda x: (x == positive_class).mean() * 100)
            .reset_index()
        )

        contract_churn.columns = ["Contract", "Churn Rate (%)"]
        contract_churn = contract_churn.sort_values("Churn Rate (%)", ascending=False)

        contract_fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate (%)",
            color="Churn Rate (%)",
            color_continuous_scale="RdYlGn_r",
            title="Churn Rate by Contract Type",
            text_auto=".1f",
            labels={"Churn Rate (%)": "Churn Rate (%)"},
        )

        contract_fig.update_layout(
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
        )

        dashboard_plots["churn_by_contract"] = contract_fig

    # 4. Top Features (counts and churn rates)
    categorical_features = [
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
    ]
    available_features = [f for f in categorical_features if f in df.columns]

    if available_features:
        feature = available_features[0]  
        counts = df[feature].value_counts().reset_index()
        counts.columns = [feature, "Count"]

        churn_rate = (
            df.groupby(feature)[target_col]
            .apply(lambda x: (x == positive_class).mean() * 100)
            .reset_index()
        )

        churn_rate.columns = [feature, "Churn Rate (%)"]

        feature_data = counts.merge(churn_rate, on=feature)
        feature_data = feature_data.sort_values("Churn Rate (%)", ascending=False)
        feature_fig = make_subplots(specs=[[{"secondary_y": True}]])

        feature_fig.add_trace(
            go.Bar(
                x=feature_data[feature],
                y=feature_data["Count"],
                name="Count",
                marker_color="#4ECDC4",
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>",
            ),
            secondary_y=False,
        )

        feature_fig.add_trace(
            go.Scatter(
                x=feature_data[feature],
                y=feature_data["Churn Rate (%)"],
                name="Churn Rate (%)",
                mode="lines+markers",
                marker=dict(color="#FF6B6B", size=10),
                line=dict(color="#FF6B6B", width=3),
                hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>",
            ),
            secondary_y=True,
        )

        feature_fig.update_layout(
            title_text=f"Customer Count and Churn Rate by {feature}",
            title_font_size=16,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        feature_fig.update_xaxes(title_text=feature, title_font_size=14)
        feature_fig.update_yaxes(
            title_text="Count", secondary_y=False, title_font_size=14
        )

        feature_fig.update_yaxes(
            title_text="Churn Rate (%)", secondary_y=True, title_font_size=14
        )

        dashboard_plots["feature_analysis"] = feature_fig

    return dashboard_plots


def create_feature_explorer(
    df: pd.DataFrame,
    feature: str,
    feature_type: str,
    target_col: Optional[str] = None,
    positive_class: Optional[str] = None,
    bins: int = 10,
) -> go.Figure:
    """
    Create an interactive plot exploring a feature's relationship with churn.

    Args:
        df (pd.DataFrame): DataFrame containing the customer data
        feature (str): Name of the feature to explore
        feature_type (str): Type of feature - 'categorical' or 'numerical'
        target_col (str, optional): Name of the target column. If None, uses CONFIG["target_column"].
        positive_class (str, optional): Value representing the positive class. If None, uses CONFIG["positive_class"].
        bins (int, optional): Number of bins for numerical features. Default is 10.

    Returns:
        go.Figure: Plotly figure for feature exploration
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if positive_class is None:
        positive_class = CONFIG.positive_class

    if feature_type == "categorical":
        value_counts = df[feature].value_counts().reset_index()
        value_counts.columns = [feature, "Count"]

        value_counts["Percentage"] = (value_counts["Count"] / len(df) * 100).round(1)

        churn_rate = (
            df.groupby(feature)[target_col]
            .apply(lambda x: (x == positive_class).mean() * 100)
            .reset_index()
        )
        churn_rate.columns = [feature, "Churn Rate (%)"]
        merged_data = value_counts.merge(churn_rate, on=feature)
        merged_data = merged_data.sort_values("Churn Rate (%)", ascending=False)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=merged_data[feature],
                y=merged_data["Count"],
                name="Count",
                marker_color="#4ECDC4",
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: "
                + merged_data["Percentage"].apply(lambda x: f"{x:.1f}%").tolist()[0]
                + "<extra></extra>",
                text=merged_data["Percentage"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=merged_data[feature],
                y=merged_data["Churn Rate (%)"],
                name="Churn Rate (%)",
                mode="lines+markers",
                marker=dict(color="#FF6B6B", size=10),
                line=dict(color="#FF6B6B", width=3),
                hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title_text=f"Analysis of {feature}",
            title_font_size=16,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=500,
        )

        fig.update_xaxes(title_text=feature, title_font_size=14)
        fig.update_yaxes(title_text="Count", secondary_y=False, title_font_size=14)
        fig.update_yaxes(
            title_text="Churn Rate (%)",
            secondary_y=True,
            title_font_size=14,
            range=[0, max(merged_data["Churn Rate (%)"]) * 1.1],
        )

        return fig

    elif feature_type == "numerical":
        df_temp = df.copy()
        df_temp[f"{feature}_bin"] = pd.cut(df_temp[feature], bins=bins)

        bin_counts = df_temp[f"{feature}_bin"].value_counts().reset_index()
        bin_counts.columns = ["Bin", "Count"]
        bin_counts["Percentage"] = (bin_counts["Count"] / len(df_temp) * 100).round(1)

        churn_by_bin = (
            df_temp.groupby(f"{feature}_bin")[target_col]
            .apply(lambda x: (x == positive_class).mean() * 100)
            .reset_index()
        )

        churn_by_bin.columns = ["Bin", "Churn Rate (%)"]

        merged_data = bin_counts.merge(churn_by_bin, on="Bin")
        merged_data["Bin_Label"] = merged_data["Bin"].astype(str)
        merged_data = merged_data.sort_values("Bin")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=merged_data["Bin_Label"],
                y=merged_data["Count"],
                name="Count",
                marker_color="#4ECDC4",
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Percentage: "
                + merged_data["Percentage"].apply(lambda x: f"{x:.1f}%")
                + "<extra></extra>",
                text=merged_data["Percentage"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=merged_data["Bin_Label"],
                y=merged_data["Churn Rate (%)"],
                name="Churn Rate (%)",
                mode="lines+markers",
                marker=dict(color="#FF6B6B", size=10),
                line=dict(color="#FF6B6B", width=3),
                hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title_text=f"Analysis of {feature}",
            title_font_size=16,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=500,
        )

        fig.update_xaxes(title_text=feature, title_font_size=14)
        fig.update_yaxes(title_text="Count", secondary_y=False, title_font_size=14)
        fig.update_yaxes(
            title_text="Churn Rate (%)",
            secondary_y=True,
            title_font_size=14,
            range=[0, max(merged_data["Churn Rate (%)"]) * 1.1],
        )

        return fig

    else:
        raise ValueError("feature_type must be 'categorical' or 'numerical'")


def create_model_comparison_plot(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    sort_by: str = "roc_auc",
    ascending: bool = False,
) -> go.Figure:
    """
    Create an interactive model comparison plot.

    Args:
        results_df (pd.DataFrame): DataFrame containing model evaluation results
        metrics (List[str], optional): List of metrics to include in the comparison.
                                      If None, uses ["accuracy", "precision", "recall", "f1", "roc_auc"].
        sort_by (str, optional): Metric to sort models by. Default is "roc_auc".
        ascending (bool, optional): Whether to sort in ascending order. Default is False.

    Returns:
        go.Figure: Plotly figure for model comparison
    """

    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    if "model_name" not in results_df.columns:
        if "Model" in results_df.columns:
            results_df = results_df.rename(columns={"Model": "model_name"})
          
        else:
            raise ValueError("DataFrame must contain 'model_name' or 'Model' column")

    sorted_df = results_df.sort_values(sort_by, ascending=ascending)
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly[: len(metrics)]

    for i, metric in enumerate(metrics):
        if metric in sorted_df.columns:
            fig.add_trace(
                go.Bar(
                    name=metric.replace("_", " ").title(),
                    x=sorted_df["model_name"],
                    y=sorted_df[metric],
                    text=[f"{val:.4f}" for val in sorted_df[metric]],
                    textposition="auto",
                    marker_color=colors[i % len(colors)],
                )
            )

    fig.update_layout(
        title="Model Performance Comparison",
        title_font_size=16,
        xaxis_title="Model",
        xaxis_title_font_size=14,
        yaxis_title="Score",
        yaxis_title_font_size=14,
        legend_title="Metric",
        legend_title_font_size=14,
        barmode="group",
        height=500,
    )

    return fig


def create_risk_distribution_plot(
    df: pd.DataFrame,
    churn_proba_col: str = "ChurnProbability",
    risk_level_col: str = "RiskLevel",
    threshold: float = 0.5,
) -> Dict[str, go.Figure]:
    """
    Create plots showing the distribution of churn risk.

    Args:
        df (pd.DataFrame): DataFrame containing churn probabilities and risk levels
        churn_proba_col (str, optional): Name of the column with churn probabilities.
                                        Default is "ChurnProbability".
        risk_level_col (str, optional): Name of the column with risk levels.
                                       Default is "RiskLevel".
        threshold (float, optional): Probability threshold for churn classification.
                                    Default is 0.5.

    Returns:
        Dict[str, go.Figure]: Dictionary of Plotly figures for risk distribution
    """

    risk_plots = {}

    # 1. Histogram of churn probabilities
    hist_fig = px.histogram(
        df,
        x=churn_proba_col,
        nbins=30,
        title="Distribution of Churn Probabilities",
        color_discrete_sequence=["#4ECDC4"],
        opacity=0.7,
    )

    hist_fig.add_shape(
        type="line",
        x0=threshold,
        y0=0,
        x1=threshold,
        y1=hist_fig.data[0].y.max(),
        line=dict(color="red", width=2, dash="dash"),
    )

    hist_fig.add_annotation(
        x=threshold + 0.02,
        y=hist_fig.data[0].y.max() * 0.95,
        text=f"Threshold: {threshold}",
        showarrow=False,
        font=dict(color="red"),
    )

    hist_fig.update_layout(
        title_font_size=16,
        xaxis_title="Churn Probability",
        xaxis_title_font_size=14,
        yaxis_title="Count",
        yaxis_title_font_size=14,
    )

    risk_plots["probability_histogram"] = hist_fig

    # 2. Risk level pie chart
    if risk_level_col in df.columns:
        risk_counts = df[risk_level_col].value_counts().reset_index()
        risk_counts.columns = [risk_level_col, "Count"]

        pie_fig = px.pie(
            risk_counts,
            names=risk_level_col,
            values="Count",
            title="Customer Risk Level Distribution",
            color=risk_level_col,
            color_discrete_map={
                "High": "#FF6B6B",
                "Medium": "#FFD166",
                "Low": "#4ECDC4",
            },
            hole=0.4,
        )

        pie_fig.update_traces(
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}",
        )

        pie_fig.update_layout(
            title_font_size=16,
            legend_title_font_size=14,
            legend_font_size=12,
        )

        risk_plots["risk_level_distribution"] = pie_fig

    if risk_level_col in df.columns:
        numerical_features = [
            col
            for col in df.columns
            if df[col].dtype in ["int64", "float64"] and col not in [churn_proba_col]
        ]

        if numerical_features:
            feature = numerical_features[0]  
            box_fig = px.box(
                df,
                x=risk_level_col,
                y=feature,
                color=risk_level_col,
                title=f"{feature} by Risk Level",
                color_discrete_map={
                    "High": "#FF6B6B",
                    "Medium": "#FFD166",
                    "Low": "#4ECDC4",
                },
            )

            box_fig.update_layout(
                title_font_size=16,
                xaxis_title="Risk Level",
                xaxis_title_font_size=14,
                yaxis_title=feature,
                yaxis_title_font_size=14,
            )

            risk_plots["feature_by_risk"] = box_fig

    return risk_plots
