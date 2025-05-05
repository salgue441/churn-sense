#!/usr/bin/env python3
"""
ChurnSense Interactive Dashboard

This script creates an interactive web dashboard for exploring customer churn data,
visualizing model performance, and generating predictions for at-risk customers.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils.config import CONFIG
from src.data.data_loader import load_data, get_feature_names
from src.data.feature_engineering import (
    create_engineered_features,
    identify_at_risk_customers,
    generate_retention_recommendations,
)
from src.visualization.interactive_viz import (
    create_churn_dashboard,
    create_feature_explorer,
    create_model_comparison_plot,
    create_risk_distribution_plot,
)


# ================ DATA LOADING FUNCTIONS ================

def load_project_data():
    """
    Load data and models for the dashboard.
    
    Returns:
        tuple: (df, model, model_name, evaluation_results)
    """
    # Load processed data
    try:
        df = pd.read_csv(CONFIG.processed_data_path)
        print(f"Loaded processed data from {CONFIG.processed_data_path}")
    except FileNotFoundError:
        # Fall back to raw data
        df = load_data(CONFIG.data_path)
        print(f"Loaded raw data from {CONFIG.data_path}")
    
    # Always apply feature engineering to ensure we have all required features
    try:
        df = create_engineered_features(df)
        print("Applied feature engineering to data")
    except Exception as e:
        print(f"Warning: Error during feature engineering: {e}")
        traceback.print_exc()
    
    # Load model
    models_dir = Path(CONFIG.models_path)
    production_models = list(models_dir.glob("production_*.pkl"))
    
    if production_models:
        # Find latest production model
        model_path = max(production_models, key=lambda p: p.stat().st_mtime)
        model = joblib.load(model_path)
        model_name = model_path.stem
        print(f"Loaded model from {model_path}")
    else:
        # Try to find any model
        model_files = list(models_dir.glob("*.pkl"))
        if model_files:
            model_path = model_files[0]
            model = joblib.load(model_path)
            model_name = model_path.stem
            print(f"Loaded model from {model_path}")
        else:
            model = None
            model_name = None
            print("No models found")
    
    # Load evaluation results
    eval_dir = Path(CONFIG.evaluation_path)
    eval_files = list(eval_dir.glob("*.json"))
    evaluation_results = {}
    
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)
                model_name = results.get('model_name', eval_file.stem)
                evaluation_results[model_name] = results
        except Exception as e:
            print(f"Error loading {eval_file}: {e}")
    
    return df, model, model_name, evaluation_results


# ================ DASHBOARD LAYOUT COMPONENTS ================

def create_overview_tab(df, model_name, evaluation_results):
    """
    Create the Overview tab layout.
    
    Args:
        df: DataFrame with customer data
        model_name: Name of the loaded model
        evaluation_results: Model evaluation results
        
    Returns:
        dbc.Tab: The Overview tab component
    """
    # Get churn rate
    target_col = CONFIG.target_column
    positive_class = CONFIG.positive_class
    churn_rate = (df[target_col] == positive_class).mean() * 100
    
    # Calculate at-risk revenue
    at_risk_revenue = 0
    if 'MonthlyCharges' in df.columns:
        at_risk_revenue = df[df[target_col] == positive_class]['MonthlyCharges'].sum()
    
    # Create dashboard plots
    dashboard_plots = create_churn_dashboard(df, target_col, positive_class)
    
    # Get model metrics if available
    model_metrics = {}
    if model_name and evaluation_results and model_name in evaluation_results:
        model_metrics = evaluation_results[model_name]
    elif evaluation_results:
        # Just use first model's metrics
        model_metrics = list(evaluation_results.values())[0]
    
    return dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("ChurnSense Overview", className="mt-3"),
                            html.P("Key metrics and insights from customer churn analysis."),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row([
                                            dbc.Col([
                                                html.H5("Total Customers", className="text-center"),
                                                html.H3(f"{len(df):,}", className="text-center"),
                                            ], width=4),
                                            dbc.Col([
                                                html.H5("Churn Rate", className="text-center"),
                                                html.H3(f"{churn_rate:.1f}%", className="text-center text-danger"),
                                            ], width=4),
                                            dbc.Col([
                                                html.H5("Monthly Revenue at Risk", className="text-center"),
                                                html.H3(f"${at_risk_revenue:,.2f}", className="text-center text-danger"),
                                            ], width=4),
                                        ]),
                                    ]
                                ),
                                className="mb-4"
                            ),
                        ],
                        width=12
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Churn Distribution", className="mt-3"),
                            dcc.Graph(
                                id="churn-distribution-pie",
                                figure=dashboard_plots.get("churn_distribution", {})
                            ),
                        ],
                        width=6
                    ),
                    dbc.Col(
                        [
                            html.H5("Model Performance", className="mt-3"),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row([
                                            dbc.Col([
                                                html.H5("Model", className="text-center"),
                                                html.P(model_name if model_name else "No model loaded", className="text-center"),
                                            ], width=4),
                                            dbc.Col([
                                                html.H5("Accuracy", className="text-center"),
                                                html.H3(f"{model_metrics.get('accuracy', 0):.3f}", className="text-center"),
                                            ], width=4),
                                            dbc.Col([
                                                html.H5("ROC AUC", className="text-center"),
                                                html.H3(f"{model_metrics.get('roc_auc', 0):.3f}", className="text-center"),
                                            ], width=4),
                                        ]),
                                        dbc.Row([
                                            dbc.Col([
                                                html.H5("Precision", className="text-center"),
                                                html.H3(f"{model_metrics.get('precision', 0):.3f}", className="text-center"),
                                            ], width=4),
                                            dbc.Col([
                                                html.H5("Recall", className="text-center"),
                                                html.H3(f"{model_metrics.get('recall', 0):.3f}", className="text-center"),
                                            ], width=4),
                                            dbc.Col([
                                                html.H5("F1 Score", className="text-center"),
                                                html.H3(f"{model_metrics.get('f1', 0):.3f}", className="text-center"),
                                            ], width=4),
                                        ]),
                                    ]
                                ),
                                className="mb-4"
                            ),
                        ],
                        width=6
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Churn by Contract Type", className="mt-3"),
                            dcc.Graph(
                                id="churn-by-contract",
                                figure=dashboard_plots.get("churn_by_contract", {})
                            ) if "churn_by_contract" in dashboard_plots else html.P("Contract data not available"),
                        ],
                        width=6
                    ),
                    dbc.Col(
                        [
                            html.H5("Monthly Charges vs. Tenure", className="mt-3"),
                            dcc.Graph(
                                id="monthly-charges-tenure",
                                figure=dashboard_plots.get("monthly_charges_vs_tenure", {})
                            ) if "monthly_charges_vs_tenure" in dashboard_plots else html.P("Monthly charges or tenure data not available"),
                        ],
                        width=6
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Feature Analysis", className="mt-3"),
                            dcc.Graph(
                                id="feature-analysis",
                                figure=dashboard_plots.get("feature_analysis", {})
                            ) if "feature_analysis" in dashboard_plots else html.P("Feature data not available"),
                        ],
                        width=12
                    ),
                ]
            ),
        ],
        label="Overview",
        tab_id="tab-overview",
    )


def create_feature_explorer_tab(df):
    """
    Create the Feature Explorer tab layout.
    
    Args:
        df: DataFrame with customer data
        
    Returns:
        dbc.Tab: The Feature Explorer tab component
    """
    # Get feature names
    categorical_cols, numerical_cols = get_feature_names(df)
    
    return dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Feature Explorer", className="mt-3"),
                            html.P("Explore how different features relate to customer churn."),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Feature Type"),
                                    dcc.RadioItems(
                                        id="feature-type",
                                        options=[
                                            {"label": "Categorical", "value": "categorical"},
                                            {"label": "Numerical", "value": "numerical"},
                                        ],
                                        value="categorical" if categorical_cols else "numerical",
                                        inline=True,
                                        className="mb-2",
                                    ),
                                ], width=12),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Feature"),
                                    dcc.Dropdown(
                                        id="feature-dropdown",
                                        options=[],  # Will be populated by callback
                                        value=None,
                                        clearable=False,
                                    ),
                                ], width=12),
                            ]),
                        ],
                        width=12
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="feature-explorer-output", className="mt-4"),
                        ],
                        width=12
                    ),
                ]
            ),
        ],
        label="Feature Explorer",
        tab_id="tab-feature-explorer",
    )


def create_model_performance_tab(model_name, evaluation_results):
    """
    Create the Model Performance tab layout.
    
    Args:
        model_name: Name of the loaded model
        evaluation_results: Model evaluation results
        
    Returns:
        dbc.Tab: The Model Performance tab component
    """
    # Create model comparison if multiple models available
    model_comparison = None
    if len(evaluation_results) > 1:
        # Extract key metrics from all models
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        comparison_data = []
        
        for model_id, results in evaluation_results.items():
            model_data = {"model_name": model_id}
            for metric in metrics:
                if metric in results:
                    model_data[metric] = results[metric]
            comparison_data.append(model_data)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            model_comparison = create_model_comparison_plot(comparison_df, metrics)
    
    # Get ROC curve data for current model
    roc_fig = None
    pr_fig = None
    threshold_fig = None
    feature_importance_fig = None
    
    if model_name and model_name in evaluation_results:
        results = evaluation_results[model_name]
        
        # Create ROC curve
        if "roc_fpr" in results and "roc_tpr" in results:
            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(
                    x=results["roc_fpr"],
                    y=results["roc_tpr"],
                    mode="lines",
                    name=f"ROC Curve (AUC = {results.get('roc_auc', 0):.3f})",
                    line=dict(color="#4e79a7", width=2),
                )
            )
            
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Classifier (AUC = 0.5)",
                    line=dict(color="gray", width=1, dash="dash"),
                )
            )
            
            roc_fig.update_layout(
                title="ROC Curve",
                title_font_size=16,
                xaxis=dict(title="False Positive Rate", title_font_size=14),
                yaxis=dict(title="True Positive Rate", title_font_size=14),
                legend=dict(font_size=12),
            )
        
        # Create PR curve
        if "pr_precision" in results and "pr_recall" in results:
            pr_fig = go.Figure()
            pr_fig.add_trace(
                go.Scatter(
                    x=results["pr_recall"],
                    y=results["pr_precision"],
                    mode="lines",
                    name=f"PR Curve (AP = {results.get('average_precision', 0):.3f})",
                    line=dict(color="#e15759", width=2),
                )
            )
            
            # Add no-skill line
            no_skill = results.get("tp", 0) + results.get("fn", 0)
            no_skill = no_skill / (no_skill + results.get("tn", 0) + results.get("fp", 0)) if no_skill > 0 else 0
            
            pr_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[no_skill, no_skill],
                    mode="lines",
                    name=f"No Skill Classifier ({no_skill:.3f})",
                    line=dict(color="gray", width=1, dash="dash"),
                )
            )
            
            pr_fig.update_layout(
                title="Precision-Recall Curve",
                title_font_size=16,
                xaxis=dict(title="Recall", title_font_size=14),
                yaxis=dict(title="Precision", title_font_size=14),
                legend=dict(font_size=12),
            )
    
    return dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Model Performance Analysis", className="mt-3"),
                            html.P("Evaluate model performance metrics and visualizations."),
                            dbc.Alert(
                                [
                                    html.H5("Current Model", className="alert-heading"),
                                    html.P(model_name if model_name else "No model loaded"),
                                ],
                                color="info",
                                className="mb-4",
                            ),
                        ],
                        width=12
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Model Comparison", className="mt-3"),
                            dcc.Graph(id="model-comparison", figure=model_comparison) if model_comparison else html.P("No multiple models available for comparison"),
                        ],
                        width=12
                    ),
                ]
            ) if model_comparison else None,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("ROC Curve", className="mt-3"),
                            dcc.Graph(id="roc-curve", figure=roc_fig) if roc_fig else html.P("ROC curve data not available"),
                        ],
                        width=6
                    ),
                    dbc.Col(
                        [
                            html.H5("Precision-Recall Curve", className="mt-3"),
                            dcc.Graph(id="pr-curve", figure=pr_fig) if pr_fig else html.P("Precision-Recall curve data not available"),
                        ],
                        width=6
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Threshold Analysis", className="mt-3"),
                            html.P("Analyze model performance at different probability thresholds."),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Threshold"),
                                    dcc.Slider(
                                        id="threshold-slider",
                                        min=0.1,
                                        max=0.9,
                                        step=0.05,
                                        value=0.5,
                                        marks={i/10: f"{i/10:.1f}" for i in range(1, 10)},
                                    ),
                                ], width=12),
                            ]),
                            html.Div(id="threshold-metrics-output", className="mt-3"),
                        ],
                        width=12
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Feature Importance", className="mt-3"),
                            html.Div(id="feature-importance-output"),
                        ],
                        width=12
                    ),
                ]
            ),
        ],
        label="Model Performance",
        tab_id="tab-model-performance",
    )


def create_prediction_tab(df, model):
    """
    Create the Prediction tab layout.
    
    Args:
        df: DataFrame with customer data
        model: Loaded model
        
    Returns:
        dbc.Tab: The Prediction tab component
    """
    return dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Customer Churn Prediction", className="mt-3"),
                            html.P("Identify customers at risk of churning and generate recommendations."),
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Risk Threshold"),
                                            dcc.Slider(
                                                id="risk-threshold",
                                                min=0.1,
                                                max=0.9,
                                                step=0.05,
                                                value=0.5,
                                                marks={i/10: f"{i/10:.1f}" for i in range(1, 10)},
                                            ),
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Top N Customers"),
                                            dbc.Input(id="top-n", type="number", value=50, min=1, max=1000),
                                        ], width=6),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button(
                                                "Identify At-Risk Customers",
                                                id="run-prediction",
                                                color="primary",
                                                className="mt-3",
                                                disabled=model is None,
                                            ),
                                        ], width=12),
                                    ]),
                                ]),
                                className="mb-4",
                            ),
                        ],
                        width=12
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="prediction-summary"),
                        ],
                        width=12
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([
                        html.Div(id="prediction-charts"),
                    ], width=12),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="prediction-table"),
                        ],
                        width=12
                    ),
                ]
            ),
        ],
        label="Prediction",
        tab_id="tab-prediction",
    )


def create_customer_explorer_tab(df, model):
    """
    Create the Customer Explorer tab layout.
    
    Args:
        df: DataFrame with customer data
        model: Loaded model
        
    Returns:
        dbc.Tab: The Customer Explorer tab component
    """
    return dbc.Tab(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Customer Explorer", className="mt-3"),
                            html.P("Explore individual customer profiles and churn risk factors."),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Filter by Risk Level"),
                                    dcc.Dropdown(
                                        id="risk-level-filter",
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "High Risk", "value": "high"},
                                            {"label": "Medium Risk", "value": "medium"},
                                            {"label": "Low Risk", "value": "low"},
                                        ],
                                        value="all",
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Customer ID"),
                                    dcc.Dropdown(
                                        id="customer-id-dropdown",
                                        options=[],  # Will be populated by callback
                                    ),
                                ], width=6),
                            ], className="mb-3"),
                        ],
                        width=12
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="customer-profile"),
                        ],
                        width=6
                    ),
                    dbc.Col(
                        [
                            html.Div(id="customer-recommendations"),
                        ],
                        width=6
                    ),
                ]
            ),
        ],
        label="Customer Explorer",
        tab_id="tab-customer-explorer",
    )


# ================ MAIN DASHBOARD LAYOUT ================

def create_dashboard(df, model, model_name, evaluation_results):
    """
    Create the main dashboard layout with all tabs.
    
    Args:
        df: DataFrame with customer data
        model: Loaded model
        model_name: Name of the loaded model
        evaluation_results: Model evaluation results
        
    Returns:
        dash.Dash: The Dash app instance
    """
    # Initialize the Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="ChurnSense Dashboard",
        suppress_callback_exceptions=True,
    )
    
    # Define the main layout
    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.H1("ChurnSense: Customer Churn Analysis & Prediction", className="text-center my-4"),
                        width=12,
                    )
                ]
            ),
            dbc.Tabs(
                [
                    create_overview_tab(df, model_name, evaluation_results),
                    create_feature_explorer_tab(df),
                    create_model_performance_tab(model_name, evaluation_results),
                    create_prediction_tab(df, model),
                    create_customer_explorer_tab(df, model),
                ],
                id="dashboard-tabs",
                active_tab="tab-overview",
            ),
            html.Footer(
                html.P("ChurnSense Dashboard | © 2025", className="text-center mt-5"),
            ),
        ],
        fluid=True,
    )
    
    # ================ CALLBACKS ================
    
    # Feature Explorer Callbacks
    @app.callback(
        Output("feature-dropdown", "options"),
        Input("feature-type", "value"),
    )
    def update_feature_dropdown(feature_type):
        categorical_cols, numerical_cols = get_feature_names(df)
        
        if feature_type == "categorical":
            return [{"label": col, "value": col} for col in categorical_cols]
        else:
            return [{"label": col, "value": col} for col in numerical_cols]
    
    @app.callback(
        Output("feature-dropdown", "value"),
        Input("feature-dropdown", "options"),
    )
    def set_feature_dropdown_value(available_options):
        if available_options and len(available_options) > 0:
            return available_options[0]["value"]
        return None
    
    @app.callback(
        Output("feature-explorer-output", "children"),
        [
            Input("feature-dropdown", "value"),
            Input("feature-type", "value"),
        ],
    )
    def update_feature_explorer(feature, feature_type):
        if not feature:
            return html.P("Select a feature to explore")
        
        try:
            # Create interactive feature explorer
            fig = create_feature_explorer(df, feature, feature_type)
            
            return dcc.Graph(figure=fig)
            
        except Exception as e:
            return html.Div([
                html.P(f"Error exploring feature: {str(e)}"),
                html.Pre(str(e)),
            ])
    
    # Model Performance Callbacks
    @app.callback(
        Output("threshold-metrics-output", "children"),
        Input("threshold-slider", "value"),
    )
    def update_threshold_metrics(threshold):
        if not model or model_name not in evaluation_results:
            return html.P("Model evaluation data not available")
        
        try:
            results = evaluation_results[model_name]
            
            # Need y_true and y_pred_proba to calculate metrics at different thresholds
            if "y_true" in results and "y_pred_proba" in results:
                y_true = np.array(results["y_true"])
                y_pred_proba = np.array(results["y_pred_proba"])
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                # Apply threshold
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                return dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Accuracy", className="text-center"),
                                html.H3(f"{accuracy:.3f}", className="text-center"),
                            ], width=2),
                            dbc.Col([
                                html.H5("Precision", className="text-center"),
                                html.H3(f"{precision:.3f}", className="text-center"),
                            ], width=2),
                            dbc.Col([
                                html.H5("Recall", className="text-center"),
                                html.H3(f"{recall:.3f}", className="text-center"),
                            ], width=2),
                            dbc.Col([
                                html.H5("F1 Score", className="text-center"),
                                html.H3(f"{f1:.3f}", className="text-center"),
                            ], width=2),
                            dbc.Col([
                                html.H5("Specificity", className="text-center"),
                                html.H3(f"{specificity:.3f}", className="text-center"),
                            ], width=2),
                            dbc.Col([
                                html.H5("Threshold", className="text-center"),
                                html.H3(f"{threshold:.2f}", className="text-center"),
                            ], width=2),
                        ]),
                    ]),
                    className="mb-4",
                )
            else:
                # Create simpler metrics display from available results
                return dbc.Card(
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Accuracy", className="text-center"),
                                html.H3(f"{results.get('accuracy', 0):.3f}", className="text-center"),
                            ], width=3),
                            dbc.Col([
                                html.H5("Precision", className="text-center"),
                                html.H3(f"{results.get('precision', 0):.3f}", className="text-center"),
                            ], width=3),
                            dbc.Col([
                                html.H5("Recall", className="text-center"),
                                html.H3(f"{results.get('recall', 0):.3f}", className="text-center"),
                            ], width=3),
                            dbc.Col([
                                html.H5("F1 Score", className="text-center"),
                                html.H3(f"{results.get('f1', 0):.3f}", className="text-center"),
                            ], width=3),
                        ]),
                    ]),
                    className="mb-4",
                )
                
        except Exception as e:
            return html.Div([
                html.P(f"Error calculating threshold metrics: {str(e)}"),
                html.Pre(str(e)),
            ])
    
    @app.callback(
        Output("feature-importance-output", "children"),
        Input("dashboard-tabs", "active_tab"),
    )
    def update_feature_importance(active_tab):
        if active_tab != "tab-model-performance" or not model:
            return html.P("")
        
        try:
            # Instead of using matplotlib, generate a Plotly figure
            # This avoids the threading warning
            try:
                # For sklearn pipeline models
                if hasattr(model, "named_steps") and hasattr(model.named_steps.get("classifier", None), "feature_importances_"):
                    importances = model.named_steps["classifier"].feature_importances_
                    
                # For direct tree-based models
                elif hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    
                # For linear models
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_[0])
                    
                # For sklearn pipelines with linear models
                elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("classifier", None), "coef_"):
                    importances = np.abs(model.named_steps["classifier"].coef_[0])
                    
                else:
                    return html.P("Feature importance not available for this model type")
                
                # Handle the feature names mismatch - use indices instead
                feature_names = [f"Feature {i+1}" for i in range(len(importances))]
                
                # Create a DataFrame and sort by importance
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                })
                importance_df = importance_df.sort_values("Importance", ascending=False).head(20)
                
                # Create Plotly figure
                fig = px.bar(
                    importance_df, 
                    x="Importance", 
                    y="Feature",
                    orientation='h',
                    title="Top 20 Feature Importance",
                    color="Importance",
                    color_continuous_scale="viridis"
                )
                
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    title_font_size=16,
                    xaxis_title="Importance Score",
                    xaxis_title_font_size=14,
                    yaxis_title="",
                    yaxis_title_font_size=14
                )
                
                return dcc.Graph(figure=fig)
                
            except Exception as e:
                return html.P(f"Error calculating feature importance: {str(e)}")
            
        except Exception as e:
            return html.Div([
                html.P(f"Error calculating feature importance: {str(e)}"),
                html.Pre(str(e)),
            ])
    
    # Prediction Callbacks
    @app.callback(
        [
            Output("prediction-summary", "children"),
            Output("prediction-charts", "children"),
            Output("prediction-table", "children"),
        ],
        Input("run-prediction", "n_clicks"),
        [
            State("risk-threshold", "value"),
            State("top-n", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_predictions(n_clicks, threshold, top_n):
        if n_clicks is None or model is None:
            return html.P("Run prediction to see results"), None, None
        
        if top_n is None or top_n < 1:
            top_n = 50
        
        # Identify at-risk customers
        try:
            # Ensure all required features are present
            prediction_df = df.copy()
            
            # Use try-except to catch and handle any feature engineering errors
            try:
                # Check for required engineered features
                required_features = ['CLV', 'ServiceCount', 'HasSecurityServices', 
                                   'TenureContractRatio', 'AvgSpendPerService', 'ContractDuration']
                missing_features = [f for f in required_features if f not in prediction_df.columns]
                
                if missing_features:
                    print(f"Creating missing engineered features: {missing_features}")
                    prediction_df = create_engineered_features(prediction_df)
            except Exception as e:
                print(f"Warning: Error during feature engineering: {e}")
                traceback.print_exc()
            
            at_risk_customers = identify_at_risk_customers(
                prediction_df,
                churn_probability_threshold=threshold,
                model=model,
                top_n=top_n,
            )
            
            # Generate recommendations if possible
            try:
                recommendations = generate_retention_recommendations(prediction_df, at_risk_customers)
                at_risk_customers = recommendations
            except Exception as e:
                print(f"Error generating recommendations: {e}")
                traceback.print_exc()
            
            # Create summary
            total_at_risk = len(at_risk_customers)
            risk_levels = at_risk_customers["RiskLevel"].value_counts().to_dict() if "RiskLevel" in at_risk_customers.columns else {}
            
            high_risk = risk_levels.get("High", 0)
            medium_risk = risk_levels.get("Medium", 0)
            low_risk = risk_levels.get("Low", 0)
            
            avg_prob = at_risk_customers["ChurnProbability"].mean() if "ChurnProbability" in at_risk_customers.columns else 0
            
            summary = dbc.Card(
                dbc.CardBody([
                    html.H4(f"Identified {total_at_risk} At-Risk Customers (Threshold: {threshold})", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.H5("High Risk", className="text-center text-danger"),
                            html.H3(f"{high_risk}", className="text-center"),
                        ], width=4),
                        dbc.Col([
                            html.H5("Medium Risk", className="text-center text-warning"),
                            html.H3(f"{medium_risk}", className="text-center"),
                        ], width=4),
                        dbc.Col([
                            html.H5("Low Risk", className="text-center text-success"),
                            html.H3(f"{low_risk}", className="text-center"),
                        ], width=4),
                    ]),
                    html.P(f"Average Churn Probability: {avg_prob:.2f}", className="mt-2"),
                ]),
                className="mb-4",
            )
            
            # Create charts
            risk_plots = create_risk_distribution_plot(at_risk_customers)
            
            charts = dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=risk_plots.get("probability_histogram", {})) 
                    if "probability_histogram" in risk_plots else html.P("Probability histogram not available"),
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=risk_plots.get("risk_level_distribution", {}))
                    if "risk_level_distribution" in risk_plots else html.P("Risk level distribution not available"),
                ], width=6),
            ])
            
            # Create table
            table_columns = ["CustomerID", "ChurnProbability", "RiskLevel"]
            if "Recommendations" in at_risk_customers.columns:
                table_columns.append("Recommendations")
            
            # Add some key features if they exist
            for feature in ["tenure", "MonthlyCharges", "Contract", "TotalCharges"]:
                if feature in at_risk_customers.columns:
                    table_columns.append(feature)
            
            table = dash_table.DataTable(
                id="risk-table",
                columns=[{"name": col, "id": col} for col in table_columns],
                data=at_risk_customers[table_columns].to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
                style_header={
                    "backgroundColor": "rgb(230, 230, 230)",
                    "fontWeight": "bold",
                },
                style_data_conditional=[
                    {
                        "if": {"column_id": "RiskLevel", "filter_query": "{RiskLevel} = 'High'"},
                        "backgroundColor": "rgba(255, 0, 0, 0.2)",
                    },
                    {
                        "if": {"column_id": "RiskLevel", "filter_query": "{RiskLevel} = 'Medium'"},
                        "backgroundColor": "rgba(255, 165, 0, 0.2)",
                    },
                    {
                        "if": {"column_id": "RiskLevel", "filter_query": "{RiskLevel} = 'Low'"},
                        "backgroundColor": "rgba(0, 128, 0, 0.2)",
                    },
                ],
                page_size=10,
                sort_action="native",
                filter_action="native",
            )
            
            return summary, charts, table
        
        except Exception as e:
            error_message = html.Div([
                html.P(f"Error running prediction: {str(e)}"),
                html.Pre(str(e)),
            ])
            return error_message, None, None
    
    # Customer Explorer Callbacks
    @app.callback(
        Output("customer-id-dropdown", "options"),
        [
            Input("risk-level-filter", "value"),
            Input("run-prediction", "n_clicks"),
        ],
    )
    def update_customer_dropdown(risk_level, prediction_clicks):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"] if ctx.triggered else None
        
        if CONFIG.id_column not in df.columns:
            return []
        
        customer_ids = []
        
        # If triggered by prediction, use prediction results
        if trigger_id == "run-prediction.n_clicks" and prediction_clicks:
            try:
                # Get at-risk customers based on latest prediction
                risk_table = dash.callback_context.outputs_list["prediction-table.children"]
                if risk_table and hasattr(risk_table, "data"):
                    customer_data = risk_table.data
                    customer_ids = [str(item["CustomerID"]) for item in customer_data]
            except:
                pass
        
        # If no customer IDs from prediction or if triggered by filter change
        if not customer_ids:
            # Use all available customer IDs
            customer_ids = df[CONFIG.id_column].astype(str).unique()
        
        options = [{"label": str(cid), "value": str(cid)} for cid in customer_ids[:100]]  # Limit to first 100
        return options
    
    @app.callback(
        [
            Output("customer-profile", "children"),
            Output("customer-recommendations", "children"),
        ],
        Input("customer-id-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_customer_profile(customer_id):
        if not customer_id or CONFIG.id_column not in df.columns:
            return html.P("Select a customer to view profile"), html.P("Select a customer to view recommendations")
        
        # Get customer data
        customer_data = df[df[CONFIG.id_column].astype(str) == customer_id]
        
        if len(customer_data) == 0:
            return html.P(f"Customer {customer_id} not found"), html.P("Customer not found")
        
        # Create profile card
        customer_info = customer_data.iloc[0].to_dict()
        
        # Predict churn probability if model exists
        churn_probability = None
        if model is not None:
            try:
                # Create a copy of customer data with all required features
                prediction_data = customer_data.copy()
                
                # Ensure all engineered features exist
                try:
                    required_features = ['CLV', 'ServiceCount', 'HasSecurityServices', 
                                       'TenureContractRatio', 'AvgSpendPerService', 'ContractDuration']
                    missing_features = [f for f in required_features if f not in prediction_data.columns]
                    
                    if missing_features:
                        print(f"Creating missing engineered features for customer profile: {missing_features}")
                        prediction_data = create_engineered_features(prediction_data)
                except Exception as e:
                    print(f"Warning: Error during feature engineering for customer profile: {e}")
                    traceback.print_exc()
                
                # Prepare data for prediction
                X = prediction_data.drop(columns=[CONFIG.id_column, CONFIG.target_column], errors="ignore")
                churn_probability = model.predict_proba(X)[0, 1]
            except Exception as e:
                print(f"Error predicting churn probability: {e}")
                traceback.print_exc()
        
        profile_card = dbc.Card(
            dbc.CardBody([
                html.H4(f"Customer Profile: {customer_id}", className="card-title"),
                
                html.H5("Churn Status", className="mt-3"),
                html.P(
                    customer_info.get(CONFIG.target_column, "Unknown"),
                    className="text-danger font-weight-bold" if customer_info.get(CONFIG.target_column) == CONFIG.positive_class else "",
                ),
                
                html.H5("Churn Probability", className="mt-3") if churn_probability is not None else None,
                dbc.Progress(
                    value=int(churn_probability * 100) if churn_probability is not None else 0,
                    color="danger" if churn_probability is not None and churn_probability >= 0.7 else
                          "warning" if churn_probability is not None and churn_probability >= 0.4 else "success",
                    className="mb-3",
                    label=f"{int(churn_probability * 100)}%" if churn_probability is not None else None,
                ) if churn_probability is not None else None,
                
                html.H5("Customer Details", className="mt-3"),
                html.Div([
                    dbc.Row([
                        dbc.Col(html.Strong(f"{key}:"), width=6),
                        dbc.Col(html.Span(f"{value}"), width=6),
                    ]) for key, value in customer_info.items() 
                    if key not in [CONFIG.id_column, CONFIG.target_column] and str(value) != "nan"
                ]),
            ]),
            className="mb-4",
        )
        
        # Generate recommendations
        if model is not None:
            try:
                # Create a DataFrame with just this customer
                single_customer_df = customer_data.copy()
                
                # Ensure all engineered features exist
                try:
                    required_features = ['CLV', 'ServiceCount', 'HasSecurityServices', 
                                       'TenureContractRatio', 'AvgSpendPerService', 'ContractDuration']
                    missing_features = [f for f in required_features if f not in single_customer_df.columns]
                    
                    if missing_features:
                        print(f"Creating missing engineered features for recommendations: {missing_features}")
                        single_customer_df = create_engineered_features(single_customer_df)
                except Exception as e:
                    print(f"Warning: Error during feature engineering for recommendations: {e}")
                    traceback.print_exc()
                
                # Identify risk and generate recommendations
                at_risk_df = identify_at_risk_customers(
                    single_customer_df,
                    churn_probability_threshold=0.0,  # Include all customers
                    model=model,
                )
                
                recommendations_df = generate_retention_recommendations(df, at_risk_df)
                
                if "Recommendations" in recommendations_df.columns:
                    recommendations = recommendations_df.iloc[0]["Recommendations"]
                    
                    recommendations_card = dbc.Card(
                        dbc.CardBody([
                            html.H4("Recommended Actions", className="card-title"),
                            html.P(recommendations),
                            
                            html.H5("Why This Customer Might Churn", className="mt-3"),
                            html.Ul([
                                html.Li(risk_factor) for risk_factor in [
                                    f"Contract Type: {customer_info.get('Contract', 'Unknown')}" if 'Contract' in customer_info else None,
                                    f"Tenure: {customer_info.get('tenure', 'Unknown')} months" if 'tenure' in customer_info else None,
                                    f"Monthly Charges: ${customer_info.get('MonthlyCharges', 'Unknown')}" if 'MonthlyCharges' in customer_info else None,
                                    f"Online Security: {customer_info.get('OnlineSecurity', 'Unknown')}" if 'OnlineSecurity' in customer_info else None,
                                    f"Tech Support: {customer_info.get('TechSupport', 'Unknown')}" if 'TechSupport' in customer_info else None,
                                ] if risk_factor is not None
                            ]),
                        ]),
                        className="mb-4",
                    )
                else:
                    recommendations_card = dbc.Card(
                        dbc.CardBody([
                            html.H4("Recommendations", className="card-title"),
                            html.P("No specific recommendations available."),
                        ]),
                        className="mb-4",
                    )
            
            except Exception as e:
                recommendations_card = dbc.Card(
                    dbc.CardBody([
                        html.H4("Recommendations", className="card-title"),
                        html.P(f"Error generating recommendations: {str(e)}"),
                    ]),
                    className="mb-4",
                )
        else:
            recommendations_card = dbc.Card(
                dbc.CardBody([
                    html.H4("Recommendations", className="card-title"),
                    html.P("No model available to generate recommendations."),
                ]),
                className="mb-4",
            )
        
        return profile_card, recommendations_card
    
    return app


# ================ MAIN EXECUTION ================

if __name__ == "__main__":
    # Load data and models
    df, model, model_name, evaluation_results = load_project_data()
    
    # Create and run the dashboard
    app = create_dashboard(df, model, model_name, evaluation_results)
    
    # Run the server
    app.run(debug=True)