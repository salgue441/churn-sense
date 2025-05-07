"""Layout components for ChurnSense Dashboard"""

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path

from churnsense.config import config

COLORS = {
    "primary": "#4361EE",
    "secondary": "#3A0CA3",
    "accent": "#F72585",
    "success": "#4CC9F0",
    "warning": "#F8961E",
    "danger": "#F94144",
    "light": "#F8F9FA",
    "dark": "#212529",
    "muted": "#6C757D",
    "background": "#FFFFFF",
    "card": "#FFFFFF",
}

CHART_COLORS = [
    "#4361EE",
    "#3A0CA3",
    "#F72585",
    "#4CC9F0",
    "#F8961E",
    "#F94144",
    "#90BE6D",
    "#43AA8B",
]

CUSTOM_STYLES = {
    "card": {
        "border-radius": "10px",
        "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
        "margin-bottom": "20px",
        "border": "none",
        "background-color": COLORS["card"],
    },
    "card-header": {
        "background-color": COLORS["light"],
        "border-bottom": f'1px solid {COLORS["light"]}',
        "border-radius": "10px 10px 0 0",
        "font-weight": "bold",
        "color": COLORS["primary"],
    },
    "tab": {
        "border-radius": "5px 5px 0 0",
        "padding": "10px 15px",
        "font-weight": "500",
    },
    "button-primary": {
        "background-color": COLORS["primary"],
        "border-color": COLORS["primary"],
        "border-radius": "5px",
        "font-weight": "500",
    },
    "button-outline": {
        "border-color": COLORS["primary"],
        "color": COLORS["primary"],
        "border-radius": "5px",
        "font-weight": "500",
    },
    "div-spacing": {
        "margin-bottom": "20px",
    },
}


def create_header():
    """
    Create dashboard header
    """

    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.I(
                                    className="fas fa-chart-line me-2",
                                    style={
                                        "color": COLORS["accent"],
                                        "font-size": "24px",
                                    },
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                dbc.NavbarBrand(
                                    "ChurnSense",
                                    className="ms-2",
                                    style={
                                        "font-weight": "700",
                                        "letter-spacing": "0.5px",
                                    },
                                ),
                                width="auto",
                            ),
                        ],
                        align="center",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Dashboard", href="#", style={"font-weight": "500"}
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    "Documentation",
                                    href="#",
                                    style={"font-weight": "500"},
                                )
                            ),
                            dbc.NavItem(
                                dbc.NavLink(
                                    "About", href="#", style={"font-weight": "500"}
                                )
                            ),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]
        ),
        color=COLORS["light"],
        dark=False,
        className="mb-4",
        style={"box-shadow": "0 2px 4px rgba(0,0,0,0.05)"},
    )


def create_upload_tab():
    """
    Create data upload tab.
    """

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Upload Customer Data", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.P(
                                                "Upload your customer data file in CSV format to analyze churn risk.",
                                                className="text-muted",
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Upload(
                                                        id="upload-data",
                                                        children=html.Div(
                                                            [
                                                                html.I(
                                                                    className="fas fa-cloud-upload-alt me-2",
                                                                    style={
                                                                        "color": COLORS[
                                                                            "primary"
                                                                        ],
                                                                        "font-size": "32px",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    "Drag and Drop or"
                                                                ),
                                                                html.Button(
                                                                    "Select File",
                                                                    className="btn btn-outline-primary mt-2",
                                                                ),
                                                            ]
                                                        ),
                                                        style={
                                                            "width": "100%",
                                                            "height": "200px",
                                                            "lineHeight": "200px",
                                                            "borderWidth": "1px",
                                                            "borderStyle": "dashed",
                                                            "borderRadius": "10px",
                                                            "borderColor": COLORS[
                                                                "primary"
                                                            ],
                                                            "textAlign": "center",
                                                            "display": "flex",
                                                            "flexDirection": "column",
                                                            "justifyContent": "center",
                                                            "alignItems": "center",
                                                            "backgroundColor": COLORS[
                                                                "light"
                                                            ],
                                                            "margin": "20px 0",
                                                        },
                                                        multiple=False,
                                                    ),
                                                    html.Div(id="upload-status"),
                                                ],
                                                style=CUSTOM_STYLES["div-spacing"],
                                            ),
                                            html.H6(
                                                "Or use sample dataset",
                                                className="mt-4",
                                            ),
                                            dbc.Button(
                                                "Load Telco Sample Data",
                                                id="load-sample-data",
                                                color="primary",
                                                outline=True,
                                                className="mt-2",
                                                style=CUSTOM_STYLES["button-outline"],
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=8,
                        className="mx-auto",
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Collapse(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.H5(
                                                    "Data Preview", className="mb-0"
                                                )
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dash_table.DataTable(
                                                        id="data-preview-table",
                                                        page_size=10,
                                                        style_table={
                                                            "overflowX": "auto"
                                                        },
                                                        style_cell={
                                                            "textAlign": "left",
                                                            "padding": "10px",
                                                            "fontFamily": "system-ui",
                                                        },
                                                        style_header={
                                                            "backgroundColor": COLORS[
                                                                "light"
                                                            ],
                                                            "fontWeight": "bold",
                                                            "border": "none",
                                                        },
                                                        style_data_conditional=[
                                                            {
                                                                "if": {
                                                                    "row_index": "odd"
                                                                },
                                                                "backgroundColor": "rgb(248, 249, 250)",
                                                            }
                                                        ],
                                                        style_as_list_view=True,
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Button(
                                                                        "Process Data",
                                                                        id="process-data-button",
                                                                        color="primary",
                                                                        className="mt-3",
                                                                        style=CUSTOM_STYLES[
                                                                            "button-primary"
                                                                        ],
                                                                    ),
                                                                ],
                                                                width={"size": 3},
                                                                className="ml-auto",
                                                            ),
                                                        ],
                                                        justify="end",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style=CUSTOM_STYLES["card"],
                                    ),
                                ],
                                id="preview-collapse",
                                is_open=False,
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ]
    )


def create_overview_tab():
    """
    Create data overview tab.
    """

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Dataset Overview", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Total Customers",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="total-customers",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "primary"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Churn Rate",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="churn-rate",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "accent"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Avg. Monthly Charges",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="avg-monthly-charges",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "secondary"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Customer Demographics", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="tenure-distribution",
                                                                config={
                                                                    "displayModeBar": False
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        lg=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dcc.Graph(
                                                                id="contract-distribution",
                                                                config={
                                                                    "displayModeBar": False
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        lg=6,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Customer Segments", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="customer-segments",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Monthly Charges by Tenure",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="charges-by-tenure",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                ]
            ),
        ]
    )


def create_predictions_tab():
    """
    Create predictions tab.
    """

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Churn Prediction Summary", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Predicted to Churn",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="predicted-churn-count",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "danger"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Churn Probability",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="avg-churn-probability",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "warning"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Potential Revenue Loss",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="potential-revenue-loss",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "secondary"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Risk Distribution", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="risk-distribution",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Churn Probability Distribution",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="churn-probability-hist",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Customer Churn Predictions",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.InputGroup(
                                                                [
                                                                    dbc.InputGroupText(
                                                                        html.I(
                                                                            className="fas fa-search"
                                                                        )
                                                                    ),
                                                                    dbc.Input(
                                                                        id="prediction-search",
                                                                        placeholder="Search customers...",
                                                                        type="text",
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                        width=12,
                                                        lg=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.InputGroup(
                                                                [
                                                                    dbc.InputGroupText(
                                                                        "Risk Level"
                                                                    ),
                                                                    dbc.Select(
                                                                        id="risk-filter",
                                                                        options=[
                                                                            {
                                                                                "label": "All",
                                                                                "value": "all",
                                                                            },
                                                                            {
                                                                                "label": "High",
                                                                                "value": "High",
                                                                            },
                                                                            {
                                                                                "label": "Medium",
                                                                                "value": "Medium",
                                                                            },
                                                                            {
                                                                                "label": "Low",
                                                                                "value": "Low",
                                                                            },
                                                                        ],
                                                                        value="all",
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                        ],
                                                        width=12,
                                                        lg=6,
                                                    ),
                                                ]
                                            ),
                                            dash_table.DataTable(
                                                id="predictions-table",
                                                page_size=10,
                                                filter_action="native",
                                                sort_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "10px",
                                                    "fontFamily": "system-ui",
                                                },
                                                style_header={
                                                    "backgroundColor": COLORS["light"],
                                                    "fontWeight": "bold",
                                                    "border": "none",
                                                },
                                                style_data_conditional=[
                                                    {
                                                        "if": {"row_index": "odd"},
                                                        "backgroundColor": "rgb(248, 249, 250)",
                                                    },
                                                    {
                                                        "if": {
                                                            "filter_query": '{RiskLevel} = "High"'
                                                        },
                                                        "backgroundColor": "rgba(249, 65, 68, 0.15)",
                                                    },
                                                    {
                                                        "if": {
                                                            "filter_query": '{RiskLevel} = "Medium"'
                                                        },
                                                        "backgroundColor": "rgba(248, 150, 30, 0.15)",
                                                    },
                                                ],
                                                style_as_list_view=True,
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ]
    )


def create_feature_analysis_tab():
    """
    Create feature analysis tab.
    """

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Feature Importance", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="feature-importance-chart",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Feature Impact on Churn", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Select(
                                                id="feature-selector",
                                                placeholder="Select a feature to analyze",
                                                className="mb-3",
                                            ),
                                            dcc.Graph(
                                                id="feature-impact-chart",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Feature Correlation Matrix",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="correlation-heatmap",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ]
    )


def create_model_performance_tab():
    """
    Create model performance tab.
    """

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Model Performance Metrics",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Accuracy",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="model-accuracy",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "primary"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=6,
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Precision",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="model-precision",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "secondary"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=6,
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Recall",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="model-recall",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "accent"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=6,
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "F1 Score",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="model-f1",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "success"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=6,
                                                        md=3,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Confusion Matrix", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="confusion-matrix",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("ROC Curve", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="roc-curve",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Threshold Optimization", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="threshold-metrics",
                                                config={"displayModeBar": False},
                                            ),
                                            html.Div(
                                                [
                                                    html.P(
                                                        "Adjust threshold to optimize model performance:",
                                                        className="mt-3",
                                                    ),
                                                    dcc.Slider(
                                                        id="threshold-slider",
                                                        min=0.1,
                                                        max=0.9,
                                                        step=0.05,
                                                        value=0.5,
                                                        marks={
                                                            i / 10: str(i / 10)
                                                            for i in range(1, 10)
                                                        },
                                                        className="mt-3",
                                                    ),
                                                    html.Div(
                                                        id="threshold-output",
                                                        className="mt-3",
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ]
    )


def create_retention_tab():
    """
    Create retention strategies tab.
    """

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Retention Campaign Overview",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Customers to Target",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="customers-to-target",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "primary"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Campaign Cost",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="campaign-cost",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "warning"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.H6(
                                                                        "Expected ROI",
                                                                        className="text-muted",
                                                                    ),
                                                                    html.H3(
                                                                        id="expected-roi",
                                                                        className="mb-0",
                                                                        style={
                                                                            "color": COLORS[
                                                                                "success"
                                                                            ]
                                                                        },
                                                                    ),
                                                                ],
                                                                className="text-center p-3",
                                                                style={
                                                                    "border-radius": "8px",
                                                                    "background-color": COLORS[
                                                                        "light"
                                                                    ],
                                                                },
                                                            ),
                                                        ],
                                                        width=12,
                                                        md=4,
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Retention Recommendations",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Risk Level"),
                                                    dbc.Select(
                                                        id="retention-risk-filter",
                                                        options=[
                                                            {
                                                                "label": "All",
                                                                "value": "all",
                                                            },
                                                            {
                                                                "label": "High",
                                                                "value": "High",
                                                            },
                                                            {
                                                                "label": "Medium",
                                                                "value": "Medium",
                                                            },
                                                        ],
                                                        value="High",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dash_table.DataTable(
                                                id="retention-table",
                                                page_size=10,
                                                filter_action="native",
                                                sort_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "padding": "10px",
                                                    "fontFamily": "system-ui",
                                                },
                                                style_header={
                                                    "backgroundColor": COLORS["light"],
                                                    "fontWeight": "bold",
                                                    "border": "none",
                                                },
                                                style_data_conditional=[
                                                    {
                                                        "if": {"row_index": "odd"},
                                                        "backgroundColor": "rgb(248, 249, 250)",
                                                    },
                                                    {
                                                        "if": {
                                                            "filter_query": '{RiskLevel} = "High"'
                                                        },
                                                        "backgroundColor": "rgba(249, 65, 68, 0.15)",
                                                    },
                                                    {
                                                        "if": {
                                                            "filter_query": '{RiskLevel} = "Medium"'
                                                        },
                                                        "backgroundColor": "rgba(248, 150, 30, 0.15)",
                                                    },
                                                ],
                                                style_as_list_view=True,
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "Retention Strategy Effectiveness",
                                            className="mb-0",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="retention-effectiveness",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            "ROI by Customer Segment", className="mb-0"
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="segment-roi",
                                                config={"displayModeBar": False},
                                            ),
                                        ]
                                    ),
                                ],
                                style=CUSTOM_STYLES["card"],
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                ]
            ),
        ]
    )


def create_layout(app):
    """
    Create the main layout for the dashboard.

    Args:
        app: Dash application instance

    Returns:
        Layout component
    """

    return html.Div(
        [
            dcc.Store(id="stored-data", storage_type="memory"),
            dcc.Store(id="prediction-results", storage_type="memory"),
            dcc.Store(id="model-metrics", storage_type="memory"),
            create_header(),
            dbc.Container(
                [
                    # Hero section with summary statistics
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H2(
                                        "Customer Churn Analytics", className="mb-3"
                                    ),
                                    html.P(
                                        "Predict customer churn, analyze risk factors, and optimize retention strategies.",
                                        className="lead text-muted",
                                    ),
                                    html.Hr(),
                                ],
                                width=12,
                            )
                        ],
                        className="mb-4",
                    ),
                    # Status indicators
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Alert(
                                        "Upload data to begin analysis",
                                        id="status-message",
                                        color="primary",
                                        className="mb-4",
                                        style={"border-radius": "8px"},
                                    ),
                                ],
                                width=12,
                            )
                        ]
                    ),
                    # Main content tabs
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    dcc.Tabs(
                                                        id="main-tabs",
                                                        value="tab-upload",
                                                        children=[
                                                            dcc.Tab(
                                                                label="Upload Data",
                                                                value="tab-upload",
                                                                children=[
                                                                    create_upload_tab()
                                                                ],
                                                                style=CUSTOM_STYLES[
                                                                    "tab"
                                                                ],
                                                                selected_style={
                                                                    **CUSTOM_STYLES[
                                                                        "tab"
                                                                    ],
                                                                    "borderBottom": f'2px solid {COLORS["primary"]}',
                                                                },
                                                            ),
                                                            dcc.Tab(
                                                                label="Data Overview",
                                                                value="tab-overview",
                                                                children=[
                                                                    create_overview_tab()
                                                                ],
                                                                style=CUSTOM_STYLES[
                                                                    "tab"
                                                                ],
                                                                selected_style={
                                                                    **CUSTOM_STYLES[
                                                                        "tab"
                                                                    ],
                                                                    "borderBottom": f'2px solid {COLORS["primary"]}',
                                                                },
                                                            ),
                                                            dcc.Tab(
                                                                label="Churn Predictions",
                                                                value="tab-predictions",
                                                                children=[
                                                                    create_predictions_tab()
                                                                ],
                                                                style=CUSTOM_STYLES[
                                                                    "tab"
                                                                ],
                                                                selected_style={
                                                                    **CUSTOM_STYLES[
                                                                        "tab"
                                                                    ],
                                                                    "borderBottom": f'2px solid {COLORS["primary"]}',
                                                                },
                                                            ),
                                                            dcc.Tab(
                                                                label="Feature Analysis",
                                                                value="tab-features",
                                                                children=[
                                                                    create_feature_analysis_tab()
                                                                ],
                                                                style=CUSTOM_STYLES[
                                                                    "tab"
                                                                ],
                                                                selected_style={
                                                                    **CUSTOM_STYLES[
                                                                        "tab"
                                                                    ],
                                                                    "borderBottom": f'2px solid {COLORS["primary"]}',
                                                                },
                                                            ),
                                                            dcc.Tab(
                                                                label="Model Performance",
                                                                value="tab-performance",
                                                                children=[
                                                                    create_model_performance_tab()
                                                                ],
                                                                style=CUSTOM_STYLES[
                                                                    "tab"
                                                                ],
                                                                selected_style={
                                                                    **CUSTOM_STYLES[
                                                                        "tab"
                                                                    ],
                                                                    "borderBottom": f'2px solid {COLORS["primary"]}',
                                                                },
                                                            ),
                                                            dcc.Tab(
                                                                label="Retention Strategies",
                                                                value="tab-retention",
                                                                children=[
                                                                    create_retention_tab()
                                                                ],
                                                                style=CUSTOM_STYLES[
                                                                    "tab"
                                                                ],
                                                                selected_style={
                                                                    **CUSTOM_STYLES[
                                                                        "tab"
                                                                    ],
                                                                    "borderBottom": f'2px solid {COLORS["primary"]}',
                                                                },
                                                            ),
                                                        ],
                                                    ),
                                                ]
                                            ),
                                        ],
                                        style=CUSTOM_STYLES["card"],
                                    ),
                                ],
                                width=12,
                            )
                        ],
                        className="mt-4",
                    ),
                    # Footer
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Hr(),
                                    html.P(
                                        [
                                            "© 2025 ChurnSense • ",
                                            html.A(
                                                "Terms",
                                                href="#",
                                                className="text-decoration-none",
                                            ),
                                            " • ",
                                            html.A(
                                                "Privacy",
                                                href="#",
                                                className="text-decoration-none",
                                            ),
                                        ],
                                        className="text-center text-muted small",
                                    ),
                                ],
                                width=12,
                            )
                        ],
                        className="mt-5 mb-3",
                    ),
                ]
            ),
        ]
    )
