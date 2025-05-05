# churnsense/dashboard/app.py
"""Interactive dashboard for ChurnSense."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate

from churnsense.config import config
from churnsense.data.loader import load_data, get_feature_types
from churnsense.data.features import identify_at_risk_customers
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_layout() -> html.Div:
    """
    Create the dashboard layout.
    """
    
    return html.Div(
        [
            html.H1("ChurnSense: Customer Churn Prediction", className="header"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Dataset Overview"),
                            html.Div(id="dataset-stats"),
                            html.Hr(),
                            html.H3("Load Data"),
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    ["Drag and Drop or ", html.A("Select a CSV File")]
                                ),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px",
                                },
                                multiple=False,
                            ),
                            html.Div(id="upload-status"),
                            html.Hr(),
                            html.H3("Load Model"),
                            dcc.Dropdown(
                                id="model-selector",
                                options=[],
                                placeholder="Select a model",
                            ),
                            html.Button(
                                "Load Model", id="load-model-button", n_clicks=0
                            ),
                            html.Div(id="model-status"),
                        ],
                        className="four columns sidebar",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H2("Churn Distribution"),
                                    dcc.Graph(id="churn-distribution"),
                                ],
                                className="row",
                            ),
                            html.Div(
                                [
                                    html.H2("Key Features"),
                                    dcc.Graph(id="feature-importance"),
                                ],
                                className="row",
                            ),
                            html.Div(
                                [
                                    html.H2("At-Risk Customers"),
                                    html.Button(
                                        "Identify At-Risk Customers",
                                        id="predict-button",
                                        n_clicks=0,
                                    ),
                                    html.Div(id="prediction-stats"),
                                    DataTable(
                                        id="prediction-table",
                                        page_size=10,
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "height": "auto",
                                            "minWidth": "80px",
                                            "width": "120px",
                                            "maxWidth": "200px",
                                            "whiteSpace": "normal",
                                            "textAlign": "left",
                                        },
                                        style_header={
                                            "backgroundColor": "rgb(230, 230, 230)",
                                            "fontWeight": "bold",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "rgb(248, 248, 248)",
                                            },
                                            {
                                                "if": {
                                                    "filter_query": '{RiskLevel} = "High"',
                                                },
                                                "backgroundColor": "rgba(255, 0, 0, 0.2)",
                                            },
                                            {
                                                "if": {
                                                    "filter_query": '{RiskLevel} = "Medium"',
                                                },
                                                "backgroundColor": "rgba(255, 165, 0, 0.2)",
                                            },
                                        ],
                                    ),
                                ],
                                className="row",
                            ),
                        ],
                        className="eight columns main-content",
                    ),
                ],
                className="row",
            ),
            html.Footer(
                [
                    html.P("ChurnSense - Customer Churn Prediction Platform"),
                    html.P("© 2025 Your Company"),
                ],
                className="footer",
            ),
            html.Div(id="hidden-data", style={"display": "none"}),
            html.Div(id="hidden-model", style={"display": "none"}),
        ],
        className="container",
    )


def parse_contents(contents: str, filename: str) -> Tuple[pd.DataFrame, str]:
    """
    Parse uploaded file contents.
    """

    import base64
    import io

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    try:
        if "csv" in filename:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            return df, "CSV file loaded successfully."

        elif "xls" in filename:
            df = pd.read_excel(io.BytesIO(decoded))
            return df, "Excel file loaded successfully."

        else:
            return pd.DataFrame(), f"Unsupported file type: {filename}"

    except Exception as e:
        logger.error(f"Error parsing file: {str(e)}")
        return pd.DataFrame(), f"Error parsing file: {str(e)}"


def create_churn_distribution_figure(df: pd.DataFrame) -> go.Figure:
    """
    Create the churn distribution plot.
    """

    if config.target_column not in df.columns:
        return go.Figure()

    target_counts = df[config.target_column].value_counts()

    fig = px.pie(
        names=target_counts.index,
        values=target_counts.values,
        title=f"Churn Distribution ({target_counts.get(config.positive_class, 0)} out of {len(df)} customers)",
        color=target_counts.index,
        color_discrete_map={config.positive_class: "#e74c3c", "No": "#3498db"},
        hole=0.3,
    )

    fig.update_layout(
        legend_title_text="Churn Status",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )

    return fig


def create_feature_importance_figure(model: Optional[object] = None) -> go.Figure:
    """
    Create the feature importance plot.
    """

    if model is None:
        return go.Figure()

    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_

        elif hasattr(model, "named_steps") and hasattr(
            model.named_steps["classifier"], "feature_importances_"
        ):
            importances = model.named_steps["classifier"].feature_importances_
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importances))]

        else:
            return go.Figure()

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        )

        importance_df = importance_df.sort_values("Importance", ascending=False).head(
            10
        )

        fig = px.bar(
            importance_df,
            y="Feature",
            x="Importance",
            orientation="h",
            title="Top 10 Feature Importance",
            color="Importance",
            color_continuous_scale="Viridis",
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        return go.Figure()


def create_dash_app() -> Dash:
    """
    Create and configure the Dash app.
    """

    app = Dash(
        __name__,
        title="ChurnSense Dashboard",
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        suppress_callback_exceptions=True,
    )

    app.layout = create_layout()
    return app


@callback(
    Output("hidden-data", "children"),
    Output("upload-status", "children"),
    Output("dataset-stats", "children"),
    Output("churn-distribution", "figure"),
    Input("upload-data", "contents"),
    Input("upload-data", "filename"),
)
def update_data(contents, filename):
    """
    Update data when file is uploaded.
    """

    if contents is None:
        try:
            df = load_data()
            message = "Loaded default dataset."

        except FileNotFoundError:
            raise PreventUpdate

    else:
        df, message = parse_contents(contents, filename)

    if df.empty:
        return "", message, "", go.Figure()

    stats = html.Div(
        [
            html.P(f"Total records: {len(df):,}"),
            html.P(f"Features: {len(df.columns):,}"),
            html.P(
                f"Target column: {config.target_column}"
                if config.target_column in df.columns
                else "No target column found"
            ),
        ]
    )

    churn_fig = create_churn_distribution_figure(df)
    return df.to_json(date_format="iso", orient="split"), message, stats, churn_fig


@callback(
    Output("model-selector", "options"),
    Input("model-selector", "value"),
)
def update_model_list(value):
    """
    Update the model dropdown list.
    """

    models_dir = Path(config.models_path)
    model_files = list(models_dir.glob("*.pkl"))
    options = [
        {"label": model_file.stem, "value": str(model_file)}
        for model_file in model_files
    ]

    return options


@callback(
    Output("hidden-model", "children"),
    Output("model-status", "children"),
    Output("feature-importance", "figure"),
    Input("load-model-button", "n_clicks"),
    Input("model-selector", "value"),
)
def load_model(n_clicks, model_path):
    """
    Load the selected model.
    """

    if n_clicks == 0 or not model_path:
        raise PreventUpdate

    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        feature_importance_fig = create_feature_importance_figure(model)
        return (
            model_path,
            html.Div(
                f"Model loaded: {Path(model_path).stem}", style={"color": "green"}
            ),
            feature_importance_fig,
        )

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return (
            "",
            html.Div(f"Error loading model: {str(e)}", style={"color": "red"}),
            go.Figure(),
        )


@callback(
    Output("prediction-table", "data"),
    Output("prediction-table", "columns"),
    Output("prediction-stats", "children"),
    Input("predict-button", "n_clicks"),
    Input("hidden-data", "children"),
    Input("hidden-model", "children"),
)
def predict_churn(n_clicks, json_data, model_path):
    """
    Make churn predictions on the loaded data.
    """

    if n_clicks == 0 or not json_data or not model_path:
        raise PreventUpdate

    try:
        df = pd.read_json(json_data, orient="split")
        model = joblib.load(model_path)

        at_risk_df = identify_at_risk_customers(df, model=model, top_n=100)
        if at_risk_df.empty:
            return [], [], html.Div("No at-risk customers identified.")

        columns = [{"name": col, "id": col} for col in at_risk_df.columns]
        stats = html.Div(
            [
                html.P(f"Identified {len(at_risk_df):,} at-risk customers."),
                html.P(
                    f"High risk: {len(at_risk_df[at_risk_df['RiskLevel'] == 'High']):,}"
                ),
                html.P(
                    f"Medium risk: {len(at_risk_df[at_risk_df['RiskLevel'] == 'Medium']):,}"
                ),
                html.P(
                    f"Low risk: {len(at_risk_df[at_risk_df['RiskLevel'] == 'Low']):,}"
                ),
            ]
        )

        return at_risk_df.to_dict("records"), columns, stats

    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return (
            [],
            [],
            html.Div(f"Error making predictions: {str(e)}", style={"color": "red"}),
        )


def run_dashboard(debug: bool = False, port: int = 8050) -> None:
    """
    Run the dashboard application.
    """

    app = create_dash_app()
    app.run(debug=debug, port=port)


if __name__ == "__main__":
    run_dashboard(debug=True)
