# churnsense/data/features.py
"""Feature engineering for the ChurnSense project."""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from churnsense.config import config
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Class for feature engineering and transformation.
    """

    def __init__(self):
        """
        Initialize feature engineer.
        """

        self.transformers = {}
        self.feature_importance = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for the input DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with engineered features.
        """

        logger.info("Creating engineered features")
        df_featured = df.copy()

        created_features = []

        # Feature 1: Customer Lifetime Value (CLV)
        if "tenure" in df.columns and "MonthlyCharges" in df.columns:
            df_featured["CLV"] = df_featured["tenure"] * df_featured["MonthlyCharges"]
            created_features.append("CLV")

        # Feature 2: Service Count
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        if all(col in df.columns for col in service_cols):
            df_featured["ServiceCount"] = 0
            for col in service_cols:
                df_featured["ServiceCount"] += (
                    (df_featured[col] != "No")
                    & (df_featured[col] != "No internet service")
                    & (df_featured[col] != "No phone service")
                ).astype(int)
            created_features.append("ServiceCount")

        # Feature 3: Average Spend Per Service
        if "MonthlyCharges" in df.columns and "ServiceCount" in df_featured.columns:
            df_featured["AvgSpendPerService"] = df_featured.apply(
                lambda x: x["MonthlyCharges"] / max(x["ServiceCount"], 1), axis=1
            )
            created_features.append("AvgSpendPerService")

        # Feature 4: Tenure Groups
        if "tenure" in df.columns:
            bins = [0, 12, 24, 36, 48, 60, np.inf]
            labels = [
                "0-12 months",
                "13-24 months",
                "25-36 months",
                "37-48 months",
                "49-60 months",
                "60+ months",
            ]

            df_featured["TenureGroup"] = pd.cut(
                df_featured["tenure"], bins=bins, labels=labels
            )

            created_features.append("TenureGroup")

        # Feature 5: Security Services
        if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
            df_featured["HasSecurityServices"] = (
                (df_featured["OnlineSecurity"] == "Yes")
                & (df_featured["TechSupport"] == "Yes")
            ).astype(int)

            created_features.append("HasSecurityServices")

        # Feature 6: Contract Duration in Months
        if "Contract" in df.columns:
            contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
            df_featured["ContractDuration"] = df_featured["Contract"].map(contract_map)

            created_features.append("ContractDuration")

        # Feature 7: Tenure to Contract Ratio
        if "tenure" in df.columns and "ContractDuration" in df_featured.columns:
            df_featured["TenureContractRatio"] = (
                df_featured["tenure"] / df_featured["ContractDuration"]
            )

            created_features.append("TenureContractRatio")

        # Feature 8: Customer Payment Risk
        if "PaymentMethod" in df.columns:
            payment_risk = {
                "Electronic check": 3,  # Higher risk
                "Mailed check": 2,  # Medium risk
                "Bank transfer (automatic)": 1,  # Lower risk
                "Credit card (automatic)": 1,  # Lower risk
            }

            df_featured["PaymentRisk"] = df_featured["PaymentMethod"].map(payment_risk)
            created_features.append("PaymentRisk")

        # Feature 9: Services to Price Ratio
        if "ServiceCount" in df_featured.columns and "MonthlyCharges" in df.columns:
            df_featured["ServicePriceRatio"] = (
                df_featured["ServiceCount"] / df_featured["MonthlyCharges"]
            )

            df_featured["ServicePriceRatio"] = df_featured["ServicePriceRatio"].replace(
                [np.inf, -np.inf], np.nan
            )

            df_featured["ServicePriceRatio"] = df_featured["ServicePriceRatio"].fillna(
                0
            )
            created_features.append("ServicePriceRatio")

        logger.info(
            f"Created {len(created_features)} new features: {', '.join(created_features)}"
        )

        return df_featured

    def segment_customers(
        self,
        df: pd.DataFrame,
        n_clusters: int = 4,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Perform customer segmentation using K-means clustering.

        Args:
            df: Input DataFrame.
            n_clusters: Number of clusters to create.
            features: List of features to use for clustering.

        Returns:
            DataFrame with cluster assignments.
        """

        logger.info(f"Segmenting customers into {n_clusters} clusters")
        df_segmented = df.copy()

        if features is None:
            features = [
                "tenure",
                "MonthlyCharges",
                "CLV",
                "ServiceCount",
                "AvgSpendPerService",
            ]

        missing_features = [f for f in features if f not in df_segmented.columns]
        if missing_features:
            logger.warning(f"Missing features for clustering: {missing_features}")
            features = [f for f in features if f in df_segmented.columns]

        if not features:
            logger.error("No valid features for clustering")
            return df_segmented

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_segmented[features])

        kmeans = KMeans(
            n_clusters=n_clusters, random_state=config.random_seed, n_init=10
        )
        df_segmented["Cluster"] = kmeans.fit_predict(X_scaled)

        self.transformers["kmeans"] = kmeans
        self.transformers["cluster_scaler"] = scaler
        self.transformers["cluster_features"] = features

        cluster_analysis = self._analyze_clusters(df_segmented)
        logger.info(f"Cluster analysis complete")

        return df_segmented

    def _analyze_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the created customer segments.

        Args:
            df: DataFrame with cluster assignments.

        Returns:
            DataFrame with cluster statistics.
        """

        cluster_stats = df.groupby("Cluster").agg(
            {
                "tenure": "mean",
                "MonthlyCharges": "mean",
                "TotalCharges": "mean",
            }
        )

        for feature in ["CLV", "ServiceCount", "AvgSpendPerService"]:
            if feature in df.columns:
                cluster_stats[feature] = df.groupby("Cluster")[feature].mean()

        if config.target_column in df.columns:
            cluster_stats["ChurnRate"] = df.groupby("Cluster")[
                config.target_column
            ].apply(lambda x: (x == config.positive_class).mean() * 100)

        cluster_size = df["Cluster"].value_counts().sort_index()
        cluster_stats["Size"] = cluster_size.values
        cluster_stats["Percentage"] = (cluster_stats["Size"] / len(df) * 100).round(1)

        if "ChurnRate" in cluster_stats.columns:
            cluster_stats = cluster_stats.sort_values("ChurnRate", ascending=False)

        return cluster_stats

    def identify_at_risk_customers(
        self,
        df: pd.DataFrame,
        model: Optional[Any] = None,
        threshold: float = 0.5,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Identify customers at risk of churning.

        Args:
            df: Input DataFrame with customer data.
            model: Trained churn prediction model. If None, uses heuristics.
            threshold: Probability threshold for churn risk.
            top_n: Number of top-risk customers to return.

        Returns:
            DataFrame with at-risk customers and risk scores.
        """

        logger.info("Identifying at-risk customers")
        df_risk = df.copy()

        if model is not None:
            try:
                logger.info("Using model for churn prediction")
                X = df_risk.drop(
                    columns=[config.id_column, config.target_column], errors="ignore"
                )

                churn_proba = model.predict_proba(X)[:, 1]
                df_risk["ChurnProbability"] = churn_proba
                df_risk["AtRisk"] = (churn_proba >= threshold).astype(int)

                df_risk["RiskLevel"] = pd.cut(
                    churn_proba,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True,
                )

            except Exception as e:
                logger.error(f"Error using model for prediction: {str(e)}")
                logger.info("Falling back to rule-based risk assessment")
                model = None

        if model is None:
            logger.info("Using rule-based risk assessment")
            risk_score = np.zeros(len(df_risk))

            # Rule 1: Month-to-month contracts are higher risk
            if "Contract" in df_risk.columns:
                risk_score += (df_risk["Contract"] == "Month-to-month").astype(
                    int
                ) * 0.3

            # Rule 2: Low tenure customers are higher risk
            if "tenure" in df_risk.columns:
                risk_score += (df_risk["tenure"] <= 12).astype(int) * 0.2

            # Rule 3: Customers without security services are higher risk
            if "HasSecurityServices" in df_risk.columns:
                risk_score += (df_risk["HasSecurityServices"] == 0).astype(int) * 0.15

            elif (
                "OnlineSecurity" in df_risk.columns and "TechSupport" in df_risk.columns
            ):
                risk_score += (
                    (df_risk["OnlineSecurity"] != "Yes")
                    | (df_risk["TechSupport"] != "Yes")
                ).astype(int) * 0.15

            # Rule 4: High monthly charges are higher risk
            if "MonthlyCharges" in df_risk.columns:
                high_charges = (
                    df_risk["MonthlyCharges"] > df_risk["MonthlyCharges"].median()
                ).astype(int)
                risk_score += high_charges * 0.15

            # Rule 5: Customers with paperless billing are higher risk
            if "PaperlessBilling" in df_risk.columns:
                risk_score += (df_risk["PaperlessBilling"] == "Yes").astype(int) * 0.1

            # Rule 6: Customers with electronic payment methods are higher risk
            if "PaymentMethod" in df_risk.columns:
                electronic_payment = (
                    df_risk["PaymentMethod"].isin(["Electronic check", "Mailed check"])
                ).astype(int)
                risk_score += electronic_payment * 0.1

            df_risk["ChurnProbability"] = risk_score
            df_risk["AtRisk"] = (risk_score >= threshold).astype(int)
            df_risk["RiskLevel"] = pd.cut(
                risk_score,
                bins=[0, 0.3, 0.6, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )

        at_risk_customers = df_risk[df_risk["AtRisk"] == 1].copy()
        at_risk_customers = at_risk_customers.sort_values(
            "ChurnProbability", ascending=False
        )

        if top_n is not None and top_n < len(at_risk_customers):
            at_risk_customers = at_risk_customers.head(top_n)

        try:
            at_risk_customers = self.generate_retention_recommendations(
                df, at_risk_customers
            )

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")

        logger.info(f"Identified {len(at_risk_customers)} at-risk customers")
        return at_risk_customers

    def generate_retention_recommendations(
        self, df: pd.DataFrame, at_risk_customers: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate personalized retention recommendations for at-risk customers.

        Args:
            df: Original DataFrame with customer data.
            at_risk_customers: DataFrame with identified at-risk customers.

        Returns:
            DataFrame with retention recommendations.
        """

        logger.info("Generating retention recommendations")

        recommendations_df = at_risk_customers.copy()
        recommendations_df["Recommendations"] = ""

        for idx, customer in recommendations_df.iterrows():
            recommendations = []

            # Recommendation 1: Contract upgrade
            if "Contract" in customer and customer["Contract"] == "Month-to-month":
                recommendations.append("Offer contract upgrade with loyalty discount")

            # Recommendation 2: Add security services
            if "OnlineSecurity" in customer and customer["OnlineSecurity"] != "Yes":
                recommendations.append("Free online security services for 3 months")

            if "TechSupport" in customer and customer["TechSupport"] != "Yes":
                recommendations.append("Free technical support for 3 months")

            # Recommendation 3: Streaming services
            streaming_services = []
            if "StreamingTV" in customer and customer["StreamingTV"] != "Yes":
                streaming_services.append("TV")

            if "StreamingMovies" in customer and customer["StreamingMovies"] != "Yes":
                streaming_services.append("Movies")

            if streaming_services:
                service_str = " & ".join(streaming_services)
                recommendations.append(f"Discounted streaming bundle ({service_str})")

            # Recommendation 4: High-value customer discount
            if "tenure" in customer and "MonthlyCharges" in customer:
                if customer["tenure"] > 12 and customer["MonthlyCharges"] > 70:
                    recommendations.append("Premium customer loyalty discount (15%)")

            # Recommendation 5: Payment method change
            if (
                "PaymentMethod" in customer
                and customer["PaymentMethod"] == "Electronic check"
            ):
                recommendations.append("Auto-payment discount (5%)")

            # Combine recommendations
            if recommendations:
                recommendations_df.at[idx, "Recommendations"] = " | ".join(
                    recommendations
                )
                
            else:
                recommendations_df.at[idx, "Recommendations"] = (
                    "General retention offer"
                )

        logger.info(
            f"Generated recommendations for {len(recommendations_df)} customers"
        )
        return recommendations_df
