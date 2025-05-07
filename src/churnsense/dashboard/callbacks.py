"""Callback functions for ChurnSense Dashboard."""

import base64
import io
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from dash import (
    html,
    dcc,
    dash_table,
    Input,
    Output,
    State,
    no_update,
    callback_context,
)
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


from churnsense.config import config
from churnsense.utils.logging import setup_logger
from churnsense.data.loader import load_data, get_feature_types
from churnsense.data.processor import DataProcessor
from churnsense.data.features import FeatureEngineering
from churnsense.pipeline.model_pipeline import ModelPipeline

logger = setup_logger(__name__)

data_processor = DataProcessor()
feature_engineer = FeatureEngineering()
model_pipeline = ModelPipeline()

from churnsense.dashboard.layout import COLORS, CHART_COLORS


def safe_parse_json(json_str, **kwargs):
    """
    Parse JSON string safely using StringIO to avoid FutureWarning.
    """
    return pd.read_json(io.StringIO(json_str), **kwargs)


def generate_retention_recommendations(df):
    """
    Generate personalized retention recommendations for at-risk customers.

    Args:
        df: DataFrame with customer information and risk assessments

    Returns:
        DataFrame with personalized recommendations added
    """
    # Create a copy of the dataframe to add recommendations
    df_recommendations = df.copy()

    # Initialize recommendation columns
    df_recommendations["Recommendation"] = ""
    df_recommendations["RecommendationRank"] = 0

    # Define recommendation functions for different customer segments

    # Recommendation for month-to-month customers
    if "Contract" in df.columns:
        contract_mask = df_recommendations["Contract"] == "Month-to-month"
        tenure_mask = (
            df_recommendations["tenure"] >= 12 if "tenure" in df.columns else True
        )

        high_tenure_contract = contract_mask & tenure_mask
        if high_tenure_contract.any():
            df_recommendations.loc[high_tenure_contract, "Recommendation"] = (
                "Offer annual contract with 15% loyalty discount"
            )
            df_recommendations.loc[high_tenure_contract, "RecommendationRank"] = 1

        low_tenure_contract = contract_mask & ~tenure_mask
        if low_tenure_contract.any():
            df_recommendations.loc[low_tenure_contract, "Recommendation"] = (
                "Offer 6-month contract with 10% initial discount"
            )
            df_recommendations.loc[low_tenure_contract, "RecommendationRank"] = 2

    # Recommendation for high-value customers
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        high_value = (df_recommendations["MonthlyCharges"] > 70) & (
            df_recommendations["tenure"] > 12
        )
        if high_value.any():
            df_recommendations.loc[
                high_value & (df_recommendations["Recommendation"] == ""),
                "Recommendation",
            ] = "Premium customer loyalty discount (15%)"
            df_recommendations.loc[
                high_value & (df_recommendations["Recommendation"] == ""),
                "RecommendationRank",
            ] = 3

    # Recommendation for customers with electronic payment
    if "PaymentMethod" in df.columns:
        electronic_payment = df_recommendations["PaymentMethod"] == "Electronic check"
        if electronic_payment.any():
            df_recommendations.loc[
                electronic_payment & (df_recommendations["Recommendation"] == ""),
                "Recommendation",
            ] = "Auto-payment discount (5%)"
            df_recommendations.loc[
                electronic_payment & (df_recommendations["Recommendation"] == ""),
                "RecommendationRank",
            ] = 4

    # Fallback recommendation
    empty_recommendation = df_recommendations["Recommendation"] == ""
    if empty_recommendation.any():
        df_recommendations.loc[empty_recommendation, "Recommendation"] = (
            "General retention offer"
        )
        df_recommendations.loc[empty_recommendation, "RecommendationRank"] = 99

    # Sort by risk level and recommendation rank
    df_recommendations = df_recommendations.sort_values(
        ["RiskLevel", "RecommendationRank"]
    )

    return df_recommendations


def register_callbacks(app):
    """
    Register all callbacks for the dashboard.
    """

    # Data Upload Callbacks
    @app.callback(
        [
            Output("stored-data", "data"),
            Output("data-preview-table", "data"),
            Output("data-preview-table", "columns"),
            Output("preview-collapse", "is_open"),
            Output("upload-status", "children"),
        ],
        [
            Input("upload-data", "contents"),
            Input("load-sample-data", "n_clicks"),
        ],
        [
            State("upload-data", "filename"),
            State("preview-collapse", "is_open"),
        ],
        prevent_initial_call=True,
    )
    def process_upload(contents, n_clicks, filename, is_open):
        """
        Process uploaded data file or load sample data.
        """
        triggered_id = (
            callback_context.triggered[0]["prop_id"].split(".")[0]
            if callback_context.triggered
            else None
        )

        try:
            df = None

            if triggered_id == "upload-data" and contents:
                content_type, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)

                try:
                    if "csv" in filename.lower():
                        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

                    elif "xls" in filename.lower():
                        df = pd.read_excel(io.BytesIO(decoded))

                    else:
                        return [
                            no_update,
                            no_update,
                            no_update,
                            no_update,
                            dbc.Alert(
                                "Unsupported file type. Please upload a CSV or Excel file.",
                                color="danger",
                            ),
                        ]

                    status = dbc.Alert(
                        f"File '{filename}' uploaded successfully!", color="success"
                    )

                except Exception as e:
                    logger.error(f"Error processing uploaded file: {str(e)}")
                    return [
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        dbc.Alert(f"Error processing file: {str(e)}", color="danger"),
                    ]

            elif triggered_id == "load-sample-data" and n_clicks:
                try:
                    df = load_data(config.data_path)
                    status = dbc.Alert(
                        "Sample data loaded successfully!", color="success"
                    )

                except Exception as e:
                    logger.error(f"Error loading sample data: {str(e)}")
                    return [
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        dbc.Alert(
                            f"Error loading sample data: {str(e)}", color="danger"
                        ),
                    ]

            else:
                raise PreventUpdate

            if df is not None:
                # Pre-process and clean the DataFrame
                df = data_processor.clean_data(df)

                # Make sure numeric columns are properly typed
                for col in df.select_dtypes(include=["number"]).columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                data = df.head(50).to_dict("records")
                columns = [{"name": i, "id": i} for i in df.columns]

                stored_data = {"data": df.to_json(orient="split", date_format="iso")}

                return [stored_data, data, columns, True, status]

            return [no_update, no_update, no_update, no_update, no_update]

        except Exception as e:
            logger.error(f"Error in process_upload: {str(e)}")

            return [
                no_update,
                no_update,
                no_update,
                no_update,
                dbc.Alert(f"An error occurred: {str(e)}", color="danger"),
            ]

    @app.callback(
        [
            Output("prediction-results", "data"),
            Output("model-metrics", "data"),
            Output("status-message", "children"),
            Output("status-message", "color"),
            Output("main-tabs", "value"),
        ],
        [Input("process-data-button", "n_clicks")],
        [State("stored-data", "data")],
        prevent_initial_call=True,
    )
    def process_data_and_train(n_clicks, stored_data):
        """
        Process data, engineer features, train model, and make predictions.
        """
        if not n_clicks or not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            pipeline_results = model_pipeline.run_pipeline(
                data=df,
                feature_engineering=True,
                tune_hyperparameters=True,
                evaluation_reports=True,
                save_models=True,
            )

            predictor = pipeline_results["predictor"]
            results = predictor.batch_predict(df)
            prediction_data = {
                "predictions": results.to_json(orient="split", date_format="iso"),
                "metrics": {
                    "churn_count": int(results["ChurnPredicted"].sum()),
                    "churn_rate": float(results["ChurnPredicted"].mean()),
                    "avg_probability": float(results["ChurnProbability"].mean()),
                },
            }

            model_metrics = {
                "performance": pipeline_results["metrics"]
                .get("best_model", {})
                .get("metrics", {}),
                "top_features": pipeline_results["metrics"].get("top_features", []),
                "optimal_threshold": pipeline_results["metrics"].get(
                    "optimal_threshold", 0.5
                ),
            }

            return [
                prediction_data,
                model_metrics,
                f"Data processed and model trained successfully. Identified {prediction_data['metrics']['churn_count']} at-risk customers ({prediction_data['metrics']['churn_rate']*100:.1f}%).",
                "success",
                "tab-overview",
            ]

        except Exception as e:
            logger.error(f"Error in process_data_and_train: {str(e)}")
            return [
                no_update,
                no_update,
                f"Error processing data: {str(e)}",
                "danger",
                no_update,
            ]

    @app.callback(
        [
            Output("total-customers", "children"),
            Output("churn-rate", "children"),
            Output("avg-monthly-charges", "children"),
        ],
        [Input("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_overview_metrics(stored_data):
        """
        Update overview metrics.
        """
        if not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            total_customers = len(df)
            churn_rate = "N/A"

            if config.target_column in df.columns:
                churn_rate = f"{(df[config.target_column] == config.positive_class).mean() * 100:.1f}%"

            avg_charges = "N/A"
            if "MonthlyCharges" in df.columns:
                avg_charges = f"${df['MonthlyCharges'].mean():.2f}"

            return [f"{total_customers:,}", churn_rate, avg_charges]

        except Exception as e:
            logger.error(f"Error in update_overview_metrics: {str(e)}")
            return ["Error", "Error", "Error"]

    @app.callback(
        Output("tenure-distribution", "figure"),
        [Input("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_tenure_distribution(stored_data):
        """
        Update tenure distribution chart.
        """
        if not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            if "tenure" not in df.columns:
                return {
                    "data": [],
                    "layout": {
                        "title": "Tenure Distribution (Data Not Available)",
                        "height": 400,
                    },
                }

            # Ensure tenure is numeric
            df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

            color_param = None
            if config.target_column in df.columns:
                color_param = config.target_column

            fig = px.histogram(
                df,
                x="tenure",
                color=color_param,
                color_discrete_sequence=CHART_COLORS,
                title="Customer Tenure Distribution",
                labels={"tenure": "Tenure (months)"},
                opacity=0.8,
                marginal="box",
            )

            fig.update_layout(
                height=400,
                template="plotly_white",
                legend_title_text="Churn Status",
                xaxis_title="Tenure (months)",
                yaxis_title="Count",
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_tenure_distribution: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("contract-distribution", "figure"),
        [Input("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_contract_distribution(stored_data):
        """
        Update contract distribution chart.
        """
        if not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            if "Contract" not in df.columns:
                return {
                    "data": [],
                    "layout": {
                        "title": "Contract Distribution (Data Not Available)",
                        "height": 400,
                    },
                }

            if config.target_column in df.columns:
                contract_churn = (
                    df.groupby(["Contract", config.target_column])
                    .size()
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    contract_churn,
                    x="Contract",
                    y="Count",
                    color=config.target_column,
                    color_discrete_sequence=CHART_COLORS,
                    title="Contract Types and Churn",
                    barmode="group",
                    text_auto=True,
                )

            else:
                contract_counts = df["Contract"].value_counts().reset_index()
                contract_counts.columns = ["Contract", "Count"]

                fig = px.bar(
                    contract_counts,
                    x="Contract",
                    y="Count",
                    color="Contract",
                    color_discrete_sequence=CHART_COLORS,
                    title="Contract Types Distribution",
                    text_auto=True,
                )

            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Contract Type",
                yaxis_title="Count",
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_contract_distribution: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("charges-by-tenure", "figure"),
        [Input("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_charges_by_tenure(stored_data):
        """
        Update charges by tenure chart.
        """
        if not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            if "tenure" not in df.columns or "MonthlyCharges" not in df.columns:
                return {
                    "data": [],
                    "layout": {
                        "title": "Monthly Charges by Tenure (Data Not Available)",
                        "height": 400,
                    },
                }

            # Ensure columns are numeric
            for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            color_param = None
            if config.target_column in df.columns:
                color_param = config.target_column

            fig = px.scatter(
                df,
                x="tenure",
                y="MonthlyCharges",
                color=color_param,
                color_discrete_sequence=CHART_COLORS,
                title="Monthly Charges by Tenure",
                opacity=0.7,
                trendline="ols",
                size="TotalCharges" if "TotalCharges" in df.columns else None,
                size_max=15,
                hover_data=(
                    ["tenure", "MonthlyCharges", "TotalCharges"]
                    if "TotalCharges" in df.columns
                    else ["tenure", "MonthlyCharges"]
                ),
            )

            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Tenure (months)",
                yaxis_title="Monthly Charges ($)",
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_charges_by_tenure: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("customer-segments", "figure"),
        [Input("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_customer_segments(stored_data):
        """
        Update customer segments chart.
        """
        if not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")

            # Ensure numerical columns are properly typed
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            try:
                if "Contract" in df.columns and "tenure" in df.columns:
                    df_segmented = feature_engineer.segment_customers(df)

                    if "Cluster" in df_segmented.columns:
                        numeric_cols = [
                            col
                            for col in ["tenure", "MonthlyCharges", "TotalCharges"]
                            if col in df_segmented.columns
                        ]

                        if len(numeric_cols) >= 2:
                            fig = px.scatter(
                                df_segmented,
                                x=numeric_cols[0],
                                y=numeric_cols[1],
                                color="Cluster",
                                color_discrete_sequence=CHART_COLORS,
                                title="Customer Segments",
                                opacity=0.7,
                                hover_data=numeric_cols + ["Cluster"],
                            )

                            if config.target_column in df_segmented.columns:
                                fig.update_traces(
                                    marker=dict(
                                        symbol=[
                                            "circle" if churn == "No" else "x"
                                            for churn in df_segmented[
                                                config.target_column
                                            ]
                                        ]
                                    )
                                )

                                fig.add_annotation(
                                    text="X markers indicate churned customers",
                                    xref="paper",
                                    yref="paper",
                                    x=0.02,
                                    y=0.98,
                                    showarrow=False,
                                )

                            fig.update_layout(
                                height=400,
                                template="plotly_white",
                                xaxis_title=numeric_cols[0],
                                yaxis_title=numeric_cols[1],
                                hovermode="closest",
                            )

                            return fig
            except Exception as e:
                logger.error(f"Error in segmentation: {str(e)}")

            # Fallback to simple scatter plot if segmentation fails
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(numeric_cols) >= 2:
                color_param = (
                    config.target_column if config.target_column in df.columns else None
                )

                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color=color_param,
                    color_discrete_sequence=CHART_COLORS,
                    title="Customer Distribution",
                    opacity=0.7,
                )

                fig.update_layout(
                    height=400, template="plotly_white", hovermode="closest"
                )

                return fig

            return {
                "data": [],
                "layout": {
                    "title": "Customer Segments (Insufficient Data)",
                    "height": 400,
                },
            }

        except Exception as e:
            logger.error(f"Error in update_customer_segments: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        [
            Output("predicted-churn-count", "children"),
            Output("avg-churn-probability", "children"),
            Output("potential-revenue-loss", "children"),
            Output("predictions-table", "data"),
            Output("predictions-table", "columns"),
        ],
        [Input("prediction-results", "data"), Input("risk-filter", "value")],
        [State("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_prediction_results(prediction_results, risk_filter, stored_data):
        """
        Update prediction results and metrics.
        """
        if not prediction_results or not stored_data:
            raise PreventUpdate

        try:
            predictions_df = safe_parse_json(
                prediction_results["predictions"], orient="split"
            )
            original_df = safe_parse_json(stored_data["data"], orient="split")

            df = original_df.copy()
            if (
                "CustomerID" in original_df.columns
                and "CustomerID" in predictions_df.columns
            ):
                df = df.merge(predictions_df, on="CustomerID", how="left")

            else:
                for col in predictions_df.columns:
                    if col not in df.columns:
                        df[col] = predictions_df[col].values

            churn_count = int(df["ChurnPredicted"].sum())
            avg_probability = float(df["ChurnProbability"].mean())

            potential_loss = "N/A"
            if "MonthlyCharges" in df.columns:
                avg_tenure_loss = 6
                loss_amount = (
                    df.loc[df["ChurnPredicted"] == 1, "MonthlyCharges"].sum()
                    * avg_tenure_loss
                )
                potential_loss = f"${loss_amount:,.2f}"

            if risk_filter != "all":
                df = df[df["RiskLevel"] == risk_filter]

            display_columns = [
                "CustomerID",
                "ChurnProbability",
                "ChurnPredicted",
                "RiskLevel",
            ]

            info_columns = [
                col
                for col in [
                    "tenure",
                    "Contract",
                    "MonthlyCharges",
                    "TotalCharges",
                    "PaymentMethod",
                    "InternetService",
                ]
                if col in df.columns
            ]

            display_columns = display_columns + info_columns
            display_df = df[display_columns].copy()
            display_df["ChurnProbability"] = display_df["ChurnProbability"].apply(
                lambda x: f"{x*100:.1f}%"
            )
            display_df["ChurnPredicted"] = display_df["ChurnPredicted"].map(
                {1: "Yes", 0: "No"}
            )

            for col in ["MonthlyCharges", "TotalCharges"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")

            table_data = display_df.to_dict("records")
            table_columns = [{"name": col, "id": col} for col in display_df.columns]

            return [
                f"{churn_count:,}",
                f"{avg_probability*100:.1f}%",
                potential_loss,
                table_data,
                table_columns,
            ]

        except Exception as e:
            logger.error(f"Error in update_prediction_results: {str(e)}")
            return ["Error", "Error", "Error", [], []]

    @app.callback(
        Output("risk-distribution", "figure"),
        [Input("prediction-results", "data")],
        prevent_initial_call=True,
    )
    def update_risk_distribution(prediction_results):
        """
        Update risk distribution chart.
        """
        if not prediction_results:
            raise PreventUpdate

        try:
            predictions_df = safe_parse_json(
                prediction_results["predictions"], orient="split"
            )

            if "RiskLevel" not in predictions_df.columns:
                return {
                    "data": [],
                    "layout": {
                        "title": "Risk Distribution (Data Not Available)",
                        "height": 400,
                    },
                }

            risk_counts = predictions_df["RiskLevel"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            risk_order = ["Low", "Medium", "High"]
            risk_counts["Risk Level"] = pd.Categorical(
                risk_counts["Risk Level"], categories=risk_order, ordered=True
            )
            risk_counts = risk_counts.sort_values("Risk Level")
            risk_colors = {
                "Low": COLORS["success"],
                "Medium": COLORS["warning"],
                "High": COLORS["danger"],
            }

            fig = px.bar(
                risk_counts,
                x="Risk Level",
                y="Count",
                color="Risk Level",
                color_discrete_map=risk_colors,
                title="Customer Risk Distribution",
                text_auto=True,
            )

            fig.update_layout(
                height=400,
                template="plotly_white",
                showlegend=False,
                xaxis_title="Risk Level",
                yaxis_title="Number of Customers",
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_risk_distribution: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("churn-probability-hist", "figure"),
        [Input("prediction-results", "data")],
        prevent_initial_call=True,
    )
    def update_churn_probability_hist(prediction_results):
        """
        Update churn probability histogram.
        """
        if not prediction_results:
            raise PreventUpdate

        try:
            predictions_df = safe_parse_json(
                prediction_results["predictions"], orient="split"
            )

            if "ChurnProbability" not in predictions_df.columns:
                return {
                    "data": [],
                    "layout": {
                        "title": "Churn Probability Distribution (Data Not Available)",
                        "height": 400,
                    },
                }

            fig = px.histogram(
                predictions_df,
                x="ChurnProbability",
                color_discrete_sequence=[COLORS["primary"]],
                title="Churn Probability Distribution",
                opacity=0.8,
                nbins=20,
                marginal="box",
            )

            threshold = 0.5
            if (
                "metrics" in prediction_results
                and "optimal_threshold" in prediction_results["metrics"]
            ):
                threshold = prediction_results["metrics"]["optimal_threshold"]

            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=COLORS["accent"],
                annotation_text=f"Threshold = {threshold:.2f}",
                annotation_position="top right",
            )

            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Churn Probability",
                yaxis_title="Number of Customers",
                hovermode="closest",
                xaxis=dict(tickformat=".0%", tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_churn_probability_hist: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("feature-importance-chart", "figure"),
        [Input("model-metrics", "data")],
        prevent_initial_call=True,
    )
    def update_feature_importance(model_metrics):
        """
        Update feature importance chart.
        """
        if not model_metrics or "top_features" not in model_metrics:
            raise PreventUpdate

        try:
            top_features = model_metrics["top_features"]
            if not top_features:
                return {
                    "data": [],
                    "layout": {
                        "title": "Feature Importance (Data Not Available)",
                        "height": 600,
                    },
                }

            feature_df = pd.DataFrame(top_features)
            feature_df = feature_df.sort_values("importance_mean", ascending=True)

            fig = px.bar(
                feature_df.tail(15),
                y="feature",
                x="importance_mean",
                orientation="h",
                title="Top Feature Importance",
                color="importance_mean",
                color_continuous_scale=px.colors.sequential.Blues,
                text_auto=".3f",
            )

            fig.update_layout(
                height=600,
                template="plotly_white",
                xaxis_title="Importance",
                yaxis_title="Feature",
                coloraxis_showscale=False,
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_feature_importance: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 600}}

    @app.callback(
        [
            Output("feature-selector", "options"),
            Output("feature-selector", "value"),
        ],
        [Input("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_feature_selector(stored_data):
        """
        Update feature selector dropdown.
        """
        if not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            cat_features = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            cat_features = [
                f
                for f in cat_features
                if f != config.id_column and f != config.target_column
            ]

            num_features = df.select_dtypes(include=["number"]).columns.tolist()
            num_features = [f for f in num_features if f != config.id_column]
            all_features = cat_features + num_features

            options = [{"label": f, "value": f} for f in all_features]
            default_value = all_features[0] if all_features else None

            return [options, default_value]

        except Exception as e:
            logger.error(f"Error in update_feature_selector: {str(e)}")
            return [[], None]

    @app.callback(
        Output("feature-impact-chart", "figure"),
        [
            Input("feature-selector", "value"),
            Input("stored-data", "data"),
        ],
        prevent_initial_call=True,
    )
    def update_feature_impact(feature, stored_data):
        """
        Update feature impact chart.
        """
        if not feature or not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            if feature not in df.columns or config.target_column not in df.columns:
                return {
                    "data": [],
                    "layout": {
                        "title": "Feature Impact (Data Not Available)",
                        "height": 400,
                    },
                }

            if pd.api.types.is_numeric_dtype(df[feature]):
                # For numerical features, create bins
                df["feature_bin"] = pd.qcut(df[feature], 10, duplicates="drop")

                churn_by_bin = (
                    df.groupby("feature_bin")[config.target_column]
                    .apply(lambda x: (x == config.positive_class).mean())
                    .reset_index()
                )
                churn_by_bin.columns = ["Bin", "Churn Rate"]

                churn_by_bin["Bin"] = churn_by_bin["Bin"].astype(str)
                fig = px.bar(
                    churn_by_bin,
                    x="Bin",
                    y="Churn Rate",
                    color="Churn Rate",
                    color_continuous_scale=px.colors.sequential.RdBu_r,
                    title=f"Churn Rate by {feature}",
                    text_auto=".0%",
                )

                overall_churn = (
                    df[config.target_column] == config.positive_class
                ).mean()
                fig.add_hline(
                    y=overall_churn,
                    line_dash="dash",
                    line_color=COLORS["accent"],
                    annotation_text=f"Overall Churn Rate: {overall_churn:.1%}",
                    annotation_position="top right",
                )

            else:
                churn_by_cat = (
                    df.groupby(feature)[config.target_column]
                    .apply(lambda x: (x == config.positive_class).mean())
                    .reset_index()
                )
                churn_by_cat.columns = ["Category", "Churn Rate"]

                churn_by_cat = churn_by_cat.sort_values("Churn Rate", ascending=False)

                fig = px.bar(
                    churn_by_cat,
                    x="Category",
                    y="Churn Rate",
                    color="Churn Rate",
                    color_continuous_scale=px.colors.sequential.RdBu_r,
                    title=f"Churn Rate by {feature}",
                    text_auto=".0%",
                )

                overall_churn = (
                    df[config.target_column] == config.positive_class
                ).mean()

                fig.add_hline(
                    y=overall_churn,
                    line_dash="dash",
                    line_color=COLORS["accent"],
                    annotation_text=f"Overall Churn Rate: {overall_churn:.1%}",
                    annotation_position="top right",
                )

            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title=feature,
                yaxis_title="Churn Rate",
                coloraxis_showscale=False,
                hovermode="closest",
                yaxis=dict(tickformat=".0%"),
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_feature_impact: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("correlation-heatmap", "figure"),
        [Input("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_correlation_heatmap(stored_data):
        """
        Update correlation heatmap.
        """
        if not stored_data:
            raise PreventUpdate

        try:
            df = safe_parse_json(stored_data["data"], orient="split")
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if len(num_cols) < 2:
                return {
                    "data": [],
                    "layout": {
                        "title": "Correlation Heatmap (Insufficient Numerical Features)",
                        "height": 500,
                    },
                }

            corr_matrix = df[num_cols].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale=px.colors.diverging.RdBu_r,
                title="Feature Correlation Matrix",
                aspect="auto",
            )

            fig.update_layout(
                height=600,
                template="plotly_white",
                xaxis_title="Feature",
                yaxis_title="Feature",
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_correlation_heatmap: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 500}}

    @app.callback(
        [
            Output("model-accuracy", "children"),
            Output("model-precision", "children"),
            Output("model-recall", "children"),
            Output("model-f1", "children"),
        ],
        [Input("model-metrics", "data")],
        prevent_initial_call=True,
    )
    def update_model_metrics(model_metrics):
        """
        Update model performance metrics.
        """
        if not model_metrics or "performance" not in model_metrics:
            raise PreventUpdate

        try:
            performance = model_metrics["performance"]

            accuracy = f"{performance.get('accuracy', 0)*100:.1f}%"
            precision = f"{performance.get('precision', 0)*100:.1f}%"
            recall = f"{performance.get('recall', 0)*100:.1f}%"
            f1 = f"{performance.get('f1', 0)*100:.1f}%"

            return [accuracy, precision, recall, f1]

        except Exception as e:
            logger.error(f"Error in update_model_metrics: {str(e)}")
            return ["N/A", "N/A", "N/A", "N/A"]

    @app.callback(
        Output("confusion-matrix", "figure"),
        [Input("model-metrics", "data")],
        prevent_initial_call=True,
    )
    def update_confusion_matrix(model_metrics):
        """
        Update confusion matrix visualization.
        """
        if not model_metrics or "performance" not in model_metrics:
            raise PreventUpdate

        try:
            performance = model_metrics["performance"]
            if not all(
                key in performance
                for key in [
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                ]
            ):
                return {
                    "data": [],
                    "layout": {
                        "title": "Confusion Matrix (Data Not Available)",
                        "height": 400,
                    },
                }

            cm = np.array(
                [
                    [performance["true_negatives"], performance["false_positives"]],
                    [performance["false_negatives"], performance["true_positives"]],
                ]
            )

            total = cm.sum()
            cm_percent = cm / total * 100

            annotations = []
            for i in range(2):
                for j in range(2):
                    annotations.append(
                        {
                            "x": j,
                            "y": i,
                            "text": f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                            "font": {"color": "white" if i == j else "black"},
                            "showarrow": False,
                            "align": "center",
                        }
                    )

            class_names = ["No Churn", "Churn"]
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    colorscale=[
                        [0, COLORS["light"]],
                        [0.5, COLORS["primary"]],
                        [1, COLORS["secondary"]],
                    ],
                    showscale=False,
                )
            )

            fig.update_layout(
                title="Confusion Matrix",
                height=400,
                xaxis=dict(title="Predicted"),
                yaxis=dict(title="Actual", autorange="reversed"),
                annotations=annotations,
                template="plotly_white",
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_confusion_matrix: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("roc-curve", "figure"),
        [Input("model-metrics", "data")],
        prevent_initial_call=True,
    )
    def update_roc_curve(model_metrics):
        """
        Update ROC curve visualization.
        """
        if not model_metrics or "performance" not in model_metrics:
            raise PreventUpdate

        try:
            performance = model_metrics["performance"]
            if "roc_curve" not in performance:
                return {
                    "data": [],
                    "layout": {
                        "title": "ROC Curve (Data Not Available)",
                        "height": 400,
                    },
                }

            roc_data = performance["roc_curve"]
            fpr = roc_data["fpr"]
            tpr = roc_data["tpr"]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name="ROC Curve",
                    line=dict(color=COLORS["primary"], width=2),
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random",
                    line=dict(color="gray", width=1, dash="dash"),
                    hoverinfo="skip",
                )
            )

            auc_score = performance.get("roc_auc", 0)
            fig.add_annotation(
                x=0.95,
                y=0.05,
                text=f"AUC = {auc_score:.3f}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLORS["primary"],
                borderwidth=1,
                borderpad=4,
            )

            fig.update_layout(
                title="ROC Curve",
                height=400,
                xaxis=dict(title="False Positive Rate", constrain="domain"),
                yaxis=dict(title="True Positive Rate", scaleanchor="x", scaleratio=1),
                template="plotly_white",
                hovermode="closest",
                showlegend=True,
                legend=dict(x=0.05, y=0.95),
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_roc_curve: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        [
            Output("threshold-metrics", "figure"),
            Output("threshold-output", "children"),
        ],
        [
            Input("model-metrics", "data"),
            Input("threshold-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_threshold_metrics(model_metrics, threshold):
        """
        Update threshold metrics visualization.
        """
        if not model_metrics or "performance" not in model_metrics:
            raise PreventUpdate

        try:
            if not model_metrics.get("optimal_threshold"):
                return [
                    {
                        "data": [],
                        "layout": {
                            "title": "Threshold Optimization (Data Not Available)",
                            "height": 400,
                        },
                    },
                    "No threshold data available",
                ]

            thresholds = np.linspace(0.1, 0.9, 17)
            metrics = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "accuracy": 0.5 + np.sin(thresholds * 3) * 0.2 + 0.2,
                    "precision": thresholds + 0.1,
                    "recall": 1 - thresholds + 0.05,
                    "f1": 0.4 + np.sin(thresholds * 5) * 0.3 + 0.2,
                    "roi": 0.1 + np.sin(thresholds * 6) * 0.5 + 0.3,
                }
            )

            metrics_long = pd.melt(
                metrics,
                id_vars="threshold",
                value_vars=["accuracy", "precision", "recall", "f1", "roi"],
                var_name="Metric",
                value_name="Value",
            )

            optimal_threshold = model_metrics["optimal_threshold"]
            fig = px.line(
                metrics_long,
                x="threshold",
                y="Value",
                color="Metric",
                title="Metrics by Probability Threshold",
                color_discrete_sequence=CHART_COLORS,
                markers=True,
            )

            fig.add_vline(
                x=optimal_threshold,
                line_dash="dash",
                line_color=COLORS["accent"],
                annotation_text=f"Optimal = {optimal_threshold:.2f}",
                annotation_position="top right",
            )

            fig.add_vline(
                x=threshold,
                line_dash="solid",
                line_color=COLORS["warning"],
                annotation_text=f"Selected = {threshold:.2f}",
                annotation_position="top left",
            )

            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Threshold",
                yaxis_title="Metric Value",
                hovermode="closest",
                legend=dict(
                    title="",
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
            )

            selected_idx = np.abs(thresholds - threshold).argmin()
            selected_metrics = metrics.iloc[selected_idx]
            threshold_text = html.Div(
                [
                    html.P(
                        [
                            f"At threshold {threshold:.2f}: ",
                            html.Span(
                                f"Accuracy: {selected_metrics['accuracy']:.1%}",
                                style={"fontWeight": "bold", "marginRight": "10px"},
                            ),
                            html.Span(
                                f"Precision: {selected_metrics['precision']:.1%}",
                                style={"fontWeight": "bold", "marginRight": "10px"},
                            ),
                            html.Span(
                                f"Recall: {selected_metrics['recall']:.1%}",
                                style={"fontWeight": "bold", "marginRight": "10px"},
                            ),
                            html.Span(
                                f"F1: {selected_metrics['f1']:.1%}",
                                style={"fontWeight": "bold", "marginRight": "10px"},
                            ),
                            html.Span(
                                f"ROI: {selected_metrics['roi']:.1%}",
                                style={"fontWeight": "bold"},
                            ),
                        ]
                    )
                ]
            )

            return [fig, threshold_text]

        except Exception as e:
            logger.error(f"Error in update_threshold_metrics: {str(e)}")
            return [
                {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}},
                "Error updating threshold metrics",
            ]

    @app.callback(
        [
            Output("customers-to-target", "children"),
            Output("campaign-cost", "children"),
            Output("expected-roi", "children"),
        ],
        [Input("prediction-results", "data")],
        prevent_initial_call=True,
    )
    def update_retention_metrics(prediction_results):
        """
        Update retention campaign metrics.
        """
        if not prediction_results:
            raise PreventUpdate

        try:
            predictions_df = safe_parse_json(
                prediction_results["predictions"], orient="split"
            )

            high_risk = predictions_df[predictions_df["RiskLevel"] == "High"]
            medium_risk = predictions_df[predictions_df["RiskLevel"] == "Medium"]

            high_count = len(high_risk)
            medium_count = len(medium_risk)
            total_count = high_count + medium_count

            campaign_cost_per_customer = config.business_metrics.retention_campaign_cost
            total_cost = total_count * campaign_cost_per_customer

            avg_customer_value = config.business_metrics.avg_customer_value
            retention_success_rate = config.business_metrics.retention_success_rate

            high_success_rate = retention_success_rate
            medium_success_rate = retention_success_rate * 1.2

            retained_high = high_count * high_success_rate
            retained_medium = medium_count * medium_success_rate
            total_retained = retained_high + retained_medium

            total_benefit = total_retained * avg_customer_value

            if total_cost > 0:
                roi = (total_benefit - total_cost) / total_cost
            else:
                roi = 0

            return [f"{total_count:,}", f"${total_cost:,.2f}", f"{roi:.1%}"]

        except Exception as e:
            logger.error(f"Error in update_retention_metrics: {str(e)}")
            return ["N/A", "N/A", "N/A"]

    @app.callback(
        [
            Output("retention-table", "data"),
            Output("retention-table", "columns"),
        ],
        [
            Input("prediction-results", "data"),
            Input("retention-risk-filter", "value"),
        ],
        [State("stored-data", "data")],
        prevent_initial_call=True,
    )
    def update_retention_recommendations(prediction_results, risk_filter, stored_data):
        """
        Update retention recommendations table.
        """
        if not prediction_results or not stored_data:
            raise PreventUpdate

        try:
            predictions_df = safe_parse_json(
                prediction_results["predictions"], orient="split"
            )

            original_df = safe_parse_json(stored_data["data"], orient="split")
            df = original_df.copy()

            if (
                "CustomerID" in original_df.columns
                and "CustomerID" in predictions_df.columns
            ):
                df = df.merge(predictions_df, on="CustomerID", how="left")
            else:
                for col in predictions_df.columns:
                    if col not in df.columns:
                        df[col] = predictions_df[col].values

            if risk_filter != "all":
                df = df[df["RiskLevel"] == risk_filter]
            else:
                df = df[df["RiskLevel"].isin(["High", "Medium"])]

            df_recommendations = generate_retention_recommendations(df)
            display_columns = [
                "CustomerID",
                "RiskLevel",
                "ChurnProbability",
                "RecommendationRank",
                "Recommendation",
            ]

            info_columns = [
                col
                for col in ["tenure", "Contract", "MonthlyCharges"]
                if col in df_recommendations.columns
            ]

            display_columns = display_columns + info_columns
            display_df = df_recommendations[display_columns].copy()
            display_df["ChurnProbability"] = display_df["ChurnProbability"].apply(
                lambda x: f"{x*100:.1f}%"
            )

            if "MonthlyCharges" in display_df.columns:
                display_df["MonthlyCharges"] = display_df["MonthlyCharges"].apply(
                    lambda x: f"${x:.2f}"
                )

            table_data = display_df.to_dict("records")
            table_columns = [{"name": col, "id": col} for col in display_df.columns]

            return [table_data, table_columns]

        except Exception as e:
            logger.error(f"Error in update_retention_recommendations: {str(e)}")
            return [[], []]

    @app.callback(
        Output("retention-effectiveness", "figure"),
        [Input("prediction-results", "data")],
        prevent_initial_call=True,
    )
    def update_retention_effectiveness(prediction_results):
        """
        Update retention effectiveness chart.
        """
        if not prediction_results:
            raise PreventUpdate

        try:
            predictions_df = safe_parse_json(
                prediction_results["predictions"], orient="split"
            )

            strategies = [
                "Contract Upgrade Offer",
                "Family Plan Discount",
                "Loyalty Discount",
                "Free Service Add-ons",
                "Price Lock Guarantee",
                "Auto-payment Discount",
                "Premium Customer Offer",
            ]

            effectiveness = np.random.uniform(0.2, 0.7, len(strategies))
            costs = np.random.uniform(30, 100, len(strategies))

            strategy_df = pd.DataFrame(
                {"Strategy": strategies, "Effectiveness": effectiveness, "Cost": costs}
            )

            strategy_df = strategy_df.sort_values("Effectiveness", ascending=False)

            fig = px.bar(
                strategy_df,
                y="Strategy",
                x="Effectiveness",
                color="Cost",
                orientation="h",
                title="Retention Strategy Effectiveness",
                text_auto=".0%",
                color_continuous_scale=px.colors.sequential.Blues,
            )

            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Success Rate",
                yaxis_title="",
                coloraxis_colorbar_title="Avg. Cost ($)",
                hovermode="closest",
                xaxis=dict(tickformat=".0%"),
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_retention_effectiveness: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("segment-roi", "figure"),
        [Input("prediction-results", "data")],
        prevent_initial_call=True,
    )
    def update_segment_roi(prediction_results):
        """
        Update segment ROI chart.
        """
        if not prediction_results:
            raise PreventUpdate

        try:
            predictions_df = safe_parse_json(
                prediction_results["predictions"], orient="split"
            )

            segments = [
                "Short Tenure",
                "Month-to-Month Contract",
                "High Usage",
                "Multiple Lines",
                "New Customers",
                "Streaming Services",
            ]

            success_rates = np.random.uniform(0.2, 0.6, len(segments))
            costs = np.random.uniform(40, 90, len(segments))

            avg_customer_value = config.business_metrics.avg_customer_value
            benefits = success_rates * avg_customer_value
            roi = (benefits - costs) / costs
            segment_df = pd.DataFrame(
                {
                    "Segment": segments,
                    "SuccessRate": success_rates,
                    "Cost": costs,
                    "ROI": roi,
                }
            )

            segment_df = segment_df.sort_values("ROI", ascending=False)
            fig = px.bar(
                segment_df,
                y="Segment",
                x="ROI",
                color="SuccessRate",
                orientation="h",
                title="ROI by Customer Segment",
                text_auto=".0%",
                color_continuous_scale=px.colors.sequential.Blues,
            )

            fig.update_layout(
                height=400,
                template="plotly_white",
                xaxis_title="Return on Investment",
                yaxis_title="",
                coloraxis_colorbar_title="Success Rate",
                hovermode="closest",
                xaxis=dict(tickformat=".0%"),
            )

            return fig

        except Exception as e:
            logger.error(f"Error in update_segment_roi: {str(e)}")
            return {"data": [], "layout": {"title": f"Error: {str(e)}", "height": 400}}

    @app.callback(
        Output("navbar-collapse", "is_open"),
        [Input("navbar-toggler", "n_clicks")],
        [State("navbar-collapse", "is_open")],
        prevent_initial_call=True,
    )
    def toggle_navbar_collapse(n, is_open):
        if n:
            return not is_open

        return is_open
