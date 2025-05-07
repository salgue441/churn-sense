"""Feature engineering model for the ChurnSense project"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from churnsense.config import config
from churnsense.utils.logging import setup_logger
from churnsense.utils.exceptions import FeatureEngineeringError

logger = setup_logger(__name__)


class FeatureEngineering:
    """
    Class for creating, transforming, and selecting features.
    """

    def __init__(self):
        """
        Initialize the feature engineer.
        """

        self.transformers = {}
        self.feature_importance = None
        self.created_features = []

    def create_features(
        self, df: pd.DataFrame, inplace: bool = False, advanced: bool = True
    ) -> pd.DataFrame:
        """
        Create engineered features for the input DataFrame.

        Args:
            df: Input DataFrame.
            inplace: Whether to modify the DataFrame in place.
            advanced: Whether to create advanced features (more computationally intensive).

        Returns:
            DataFrame with engineered features.
        """

        logger.info("Creating engineered features")
        df_featured = df if inplace else df.copy()
        self.created_features = []

        df_featured = self._create_basic_features(df_featured)
        if advanced:
            df_featured = self._create_advanced_features(df_featured)

        df_featured = self._create_feature_interactions(df_featured)
        logger.info(
            f"Created {len(self.created_features)} new features: {', '.join(self.created_features)}"
        )

        return df_featured

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic engineered features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with basic engineered features.
        """

        # Feature 1: Customer Lifetime Value (CLV)
        if "tenure" in df.columns and "MonthlyCharges" in df.columns:
            tenure = pd.to_numeric(df["tenure"], errors="coerce")
            monthly_charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

            df["CLV"] = tenure * monthly_charges
            self.created_features.append("CLV")

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
            df["ServiceCount"] = 0
            
            for col in service_cols:
                has_service = (
                    (df[col] != "No")
                    & (df[col] != "No internet service")
                    & (df[col] != "No phone service")
                ).astype(int)
                
                df["ServiceCount"] = pd.to_numeric(df["ServiceCount"]) + has_service

            self.created_features.append("ServiceCount")

        # Feature 3: Average Spend Per Service
        if "MonthlyCharges" in df.columns and "ServiceCount" in df.columns:
            # Avoid division by zero
            df["AvgSpendPerService"] = df.apply(
                lambda x: pd.to_numeric(x["MonthlyCharges"])
                / max(pd.to_numeric(x["ServiceCount"]), 1),
                axis=1,
            )

            self.created_features.append("AvgSpendPerService")

        # Feature 4: Tenure Groups
        if "tenure" in df.columns:
            tenure_numeric = pd.to_numeric(df["tenure"], errors="coerce")
            bins = [0, 12, 24, 36, 48, 60, np.inf]
            labels = [
                "0-12 months",
                "13-24 months",
                "25-36 months",
                "37-48 months",
                "49-60 months",
                "60+ months",
            ]

            df["TenureGroup"] = pd.cut(tenure_numeric, bins=bins, labels=labels)
            self.created_features.append("TenureGroup")

        # Feature 5: Has Security Services (both OnlineSecurity and TechSupport)
        if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
            df["HasSecurityServices"] = (
                (df["OnlineSecurity"] == "Yes") & (df["TechSupport"] == "Yes")
            ).astype(int)

            self.created_features.append("HasSecurityServices")

        # Feature 6: Contract Duration in Months
        if "Contract" in df.columns:
            contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}

            df["ContractDuration"] = df["Contract"].map(contract_map)
            self.created_features.append("ContractDuration")

        # Feature 7: Tenure to Contract Ratio (loyalty ratio)
        if "tenure" in df.columns and "ContractDuration" in df.columns:
            df["TenureContractRatio"] = df.apply(
                lambda x: pd.to_numeric(x["tenure"])
                / max(pd.to_numeric(x["ContractDuration"]), 1),
                axis=1,
            )

            self.created_features.append("TenureContractRatio")

        # Feature 8: Customer Payment Risk
        if "PaymentMethod" in df.columns:
            payment_risk = {
                "Electronic check": 3,  # Higher risk
                "Mailed check": 2,  # Medium risk
                "Bank transfer (automatic)": 1,  # Lower risk
                "Credit card (automatic)": 1,  # Lower risk
            }
            df["PaymentRisk"] = df["PaymentMethod"].map(payment_risk)
            self.created_features.append("PaymentRisk")

        return df

    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced engineered features that are more computationally intensive.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with advanced engineered features.
        """

        # Feature 9: Services to Price Ratio
        if "ServiceCount" in df.columns and "MonthlyCharges" in df.columns:
            df["ServicePriceRatio"] = df.apply(
                lambda x: x["ServiceCount"] / max(x["MonthlyCharges"], 0.01), axis=1
            )

            df["ServicePriceRatio"] = df["ServicePriceRatio"].replace(
                [np.inf, -np.inf], np.nan
            )

            df["ServicePriceRatio"] = df["ServicePriceRatio"].fillna(0)
            self.created_features.append("ServicePriceRatio")

        # Feature 10: Monthly to Total Charges Ratio
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            df["MonthlyToTotalRatio"] = df.apply(
                lambda x: x["MonthlyCharges"] / max(x["TotalCharges"], 0.01), axis=1
            )

            df["MonthlyToTotalRatio"] = df["MonthlyToTotalRatio"].replace(
                [np.inf, -np.inf], np.nan
            )
            df["MonthlyToTotalRatio"] = df["MonthlyToTotalRatio"].fillna(0)
            df["MonthlyToTotalRatio"] = df["MonthlyToTotalRatio"].clip(0, 1)

            self.created_features.append("MonthlyToTotalRatio")

        # Feature 11: Service Density (services per tenure month)
        if "ServiceCount" in df.columns and "tenure" in df.columns:
            df["ServiceDensity"] = df.apply(
                lambda x: x["ServiceCount"] / max(x["tenure"], 1), axis=1
            )

            self.created_features.append("ServiceDensity")

        # Feature 12: Monthly Cost Growth (TotalCharges / tenure / MonthlyCharges)
        if all(
            col in df.columns for col in ["TotalCharges", "tenure", "MonthlyCharges"]
        ):
            df["MonthlyCostGrowth"] = df.apply(
                lambda x: (
                    (x["TotalCharges"] / max(x["tenure"], 1))
                    / max(x["MonthlyCharges"], 0.01)
                    if x["tenure"] > 0
                    else 1
                ),
                axis=1,
            )

            df["MonthlyCostGrowth"] = df["MonthlyCostGrowth"].replace(
                [np.inf, -np.inf], np.nan
            )
            df["MonthlyCostGrowth"] = df["MonthlyCostGrowth"].fillna(1)
            df["MonthlyCostGrowth"] = df["MonthlyCostGrowth"].clip(0, 2)
            self.created_features.append("MonthlyCostGrowth")

        # Feature 13: IsNewCustomer (tenure <= 6 months)
        if "tenure" in df.columns:
            df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)
            self.created_features.append("IsNewCustomer")

        # Feature 14: IsLongTermCustomer (tenure >= 36 months)
        if "tenure" in df.columns:
            df["IsLongTermCustomer"] = (df["tenure"] >= 36).astype(int)
            self.created_features.append("IsLongTermCustomer")

        # Feature 15: Contract Commitment Level
        if "Contract" in df.columns and "PaymentMethod" in df.columns:
            commitment_level = 0
            contract_commitment = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            df["ContractCommitment"] = df["Contract"].map(contract_commitment)

            # Payment method contribution
            payment_commitment = {
                "Electronic check": 0,
                "Mailed check": 0,
                "Bank transfer (automatic)": 1,
                "Credit card (automatic)": 1,
            }

            df["PaymentCommitment"] = df["PaymentMethod"].map(payment_commitment)
            df["CommitmentLevel"] = df["ContractCommitment"] + df["PaymentCommitment"]

            df.drop(columns=["ContractCommitment", "PaymentCommitment"], inplace=True)

            self.created_features.append("CommitmentLevel")

        return df

    def _create_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with interaction features.
        """

        # Tenure * Contract interaction
        if "tenure" in df.columns and "ContractDuration" in df.columns:
            df["TenureByContract"] = df["tenure"] * df["ContractDuration"]
            self.created_features.append("TenureByContract")

        # Service density by commitment level
        if "ServiceDensity" in df.columns and "CommitmentLevel" in df.columns:
            df["ServiceDensityByCommitment"] = df["ServiceDensity"] * (
                df["CommitmentLevel"] + 1
            )

            self.created_features.append("ServiceDensityByCommitment")

        # New customer with high monthly charges
        if "IsNewCustomer" in df.columns and "MonthlyCharges" in df.columns:
            median_monthly = df["MonthlyCharges"].median()
            df["NewCustomerHighCharges"] = (
                (df["IsNewCustomer"] == 1) & (df["MonthlyCharges"] > median_monthly)
            ).astype(int)

            self.created_features.append("NewCustomerHighCharges")

        return df

    def segment_customers(
        self,
        df: pd.DataFrame,
        n_clusters: int = 4,
        features: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Perform customer segmentation using K-means clustering.

        Args:
            df: Input DataFrame.
            n_clusters: Number of clusters to create.
            features: List of features to use for clustering.
            random_state: Random seed for reproducibility.

        Returns:
            DataFrame with cluster assignments.
        """

        logger.info(f"Segmenting customers into {n_clusters} clusters")
        df_segmented = df.copy()

        if random_state is None:
            random_state = config.random_seed

        if features is None:
            features = [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "ServiceCount",
                "ContractDuration",
            ]

            additional_features = [
                "CLV",
                "AvgSpendPerService",
                "TenureContractRatio",
                "ServiceDensity",
                "CommitmentLevel",
            ]
            features.extend([f for f in additional_features if f in df.columns])

        missing_features = [f for f in features if f not in df_segmented.columns]
        if missing_features:
            logger.warning(f"Missing features for clustering: {missing_features}")
            features = [f for f in features if f in df_segmented.columns]

            if not features:
                error_msg = "No valid features for clustering"
                logger.error(error_msg)
                raise FeatureEngineeringError(error_msg)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_segmented[features])
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            init="k-means++",
            max_iter=300,
            tol=1e-4,
        )

        try:
            df_segmented["Cluster"] = kmeans.fit_predict(X_scaled)

        except Exception as e:
            error_msg = f"Error in customer segmentation: {str(e)}"
            logger.error(error_msg)
            raise FeatureEngineeringError(error_msg) from e

        self.transformers["kmeans"] = kmeans
        self.transformers["cluster_scaler"] = scaler
        self.transformers["cluster_features"] = features

        cluster_analysis = self._analyze_clusters(df_segmented)
        logger.info(f"Customer segmentation complete: {n_clusters} clusters created")
        return df_segmented

    def _analyze_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the created customer segments.

        Args:
            df: DataFrame with cluster assignments.

        Returns:
            DataFrame with cluster statistics.
        """

        metrics = ["tenure", "MonthlyCharges", "TotalCharges"]
        engineered_metrics = [
            "CLV",
            "ServiceCount",
            "AvgSpendPerService",
            "ContractDuration",
            "TenureContractRatio",
        ]

        metrics.extend([m for m in engineered_metrics if m in df.columns])
        cluster_stats = df.groupby("Cluster")[metrics].agg(
            ["mean", "median", "std", "min", "max"]
        )

        cluster_size = df["Cluster"].value_counts().sort_index()
        cluster_stats[("Size", "")] = cluster_size.values
        cluster_stats[("Percentage", "")] = (cluster_size / len(df) * 100).round(1)

        if config.target_column in df.columns:
            churn_rates = df.groupby("Cluster")[config.target_column].apply(
                lambda x: (x == config.positive_class).mean() * 100
            )

            cluster_stats[("ChurnRate", "%")] = churn_rates.values
            cluster_stats = cluster_stats.sort_values(
                ("ChurnRate", "%"), ascending=False
            )

        return cluster_stats

    def generate_cluster_profiles(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Generate descriptive profiles for each customer cluster.

        Args:
            df: DataFrame with cluster assignments.

        Returns:
            Dictionary mapping cluster IDs to profile descriptions.
        """
        if "Cluster" not in df.columns:
            logger.error("No cluster assignments found in DataFrame")
            return {}

        profiles = {}

        cat_features = ["Contract", "PaymentMethod", "InternetService"]
        cat_features = [f for f in cat_features if f in df.columns]

        num_features = ["tenure", "MonthlyCharges", "ServiceCount"]
        num_features = [f for f in num_features if f in df.columns]

        for cluster in sorted(df["Cluster"].unique()):
            cluster_df = df[df["Cluster"] == cluster]
            profile = {}

            profile["size"] = len(cluster_df)
            profile["percentage"] = len(cluster_df) / len(df) * 100

            if config.target_column in df.columns:
                profile["churn_rate"] = (
                    cluster_df[config.target_column] == config.positive_class
                ).mean() * 100

            for feature in cat_features:
                value_counts = cluster_df[feature].value_counts(normalize=True) * 100
                profile[f"{feature}_dominant"] = value_counts.index[0]
                profile[f"{feature}_dominant_pct"] = value_counts.iloc[0]

            for feature in num_features:
                profile[f"{feature}_mean"] = cluster_df[feature].mean()
                profile[f"{feature}_median"] = cluster_df[feature].median()

            profile["description"] = self._generate_cluster_description(
                cluster, profile
            )

            profiles[int(cluster)] = profile

        return profiles

    def _generate_cluster_description(
        self, cluster: int, profile: Dict[str, Any]
    ) -> str:
        """
        Generate a text description for a cluster based on its profile.

        Args:
            cluster: Cluster ID.
            profile: Cluster profile dictionary.

        Returns:
            Text description of the cluster.
        """

        description = f"Cluster {cluster}: "

        if profile["percentage"] > 30:
            description += "A major customer segment "

        elif profile["percentage"] > 15:
            description += "A significant customer segment "

        else:
            description += "A niche customer segment "

        description += f"representing {profile['percentage']:.1f}% of customers. "

        if "churn_rate" in profile:
            if profile["churn_rate"] > 30:
                description += "High churn risk group "

            elif profile["churn_rate"] > 15:
                description += "Moderate churn risk group "

            else:
                description += "Low churn risk group "

            description += f"with {profile['churn_rate']:.1f}% churn rate. "

        if "Contract_dominant" in profile:
            contract = profile["Contract_dominant"]
            description += f"Predominantly on {contract} contracts "

            if "PaymentMethod_dominant" in profile:
                payment = profile["PaymentMethod_dominant"]
                description += f"with {payment} payment method. "

            else:
                description += ". "

        if "tenure_mean" in profile:
            avg_tenure = profile["tenure_mean"]
            if avg_tenure < 12:
                description += "Newer customers "

            elif avg_tenure < 36:
                description += "Established customers "

            else:
                description += "Long-term customers "

            description += f"with average tenure of {avg_tenure:.1f} months. "

        if "MonthlyCharges_mean" in profile:
            avg_charges = profile["MonthlyCharges_mean"]
            if "ServiceCount_mean" in profile:
                avg_services = profile["ServiceCount_mean"]
                description += f"Use an average of {avg_services:.1f} services "
                description += f"with monthly charges of ${avg_charges:.2f}."

            else:
                description += f"With average monthly charges of ${avg_charges:.2f}."

        return description

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
                drop_cols = [
                    col
                    for col in [config.id_column, config.target_column]
                    if col in df_risk.columns
                ]
                X = df_risk.drop(columns=drop_cols, errors="ignore")

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

            # Rule 7: New customers with high charges are higher risk
            if "NewCustomerHighCharges" in df_risk.columns:
                risk_score += df_risk["NewCustomerHighCharges"] * 0.1

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
        recommendations_df["RecommendationScore"] = 0.0

        for idx, customer in recommendations_df.iterrows():
            recommendations = []
            recommendation_score = 0.0

            # Recommendation 1: Contract upgrade for month-to-month customers
            if "Contract" in customer and customer["Contract"] == "Month-to-month":
                if customer["tenure"] >= 12:
                    recommendations.append(
                        "Offer annual contract with loyalty discount (15%)"
                    )
                    recommendation_score += 0.9

                else:
                    recommendations.append(
                        "Offer 6-month contract with initial discount (10%)"
                    )
                    recommendation_score += 0.8

            # Recommendation 2: Add security services
            if (
                "OnlineSecurity" in customer and customer["OnlineSecurity"] != "Yes"
            ) or ("TechSupport" in customer and customer["TechSupport"] != "Yes"):

                missing_services = []
                if "OnlineSecurity" in customer and customer["OnlineSecurity"] != "Yes":
                    missing_services.append("online security")

                if "TechSupport" in customer and customer["TechSupport"] != "Yes":
                    missing_services.append("tech support")

                service_str = " and ".join(missing_services)
                recommendations.append(f"Free {service_str} for 3 months")
                recommendation_score += 0.75

            # Recommendation 3: Streaming services bundle
            streaming_services = []
            if "StreamingTV" in customer and customer["StreamingTV"] != "Yes":
                streaming_services.append("TV")

            if "StreamingMovies" in customer and customer["StreamingMovies"] != "Yes":
                streaming_services.append("Movies")

            if streaming_services:
                service_str = " & ".join(streaming_services)
                recommendations.append(f"Discounted streaming bundle ({service_str})")
                recommendation_score += 0.6

            # Recommendation 4: High-value customer discount
            if "tenure" in customer and "MonthlyCharges" in customer:
                if customer["tenure"] > 12 and customer["MonthlyCharges"] > 70:
                    recommendations.append("Premium customer loyalty discount (15%)")
                    recommendation_score += 0.85

            # Recommendation 5: Payment method change
            if (
                "PaymentMethod" in customer
                and customer["PaymentMethod"] == "Electronic check"
            ):
                recommendations.append("Auto-payment discount (5%)")
                recommendation_score += 0.5

            # Recommendation 6: Family plan for multiple services
            if "ServiceCount" in customer and customer["ServiceCount"] >= 3:
                recommendations.append(
                    "Family plan discount (10% off bundled services)"
                )
                recommendation_score += 0.7

            # Recommendation 7: Price lock guarantee
            if "MonthlyCharges" in customer and customer["MonthlyCharges"] > 80:
                recommendations.append("12-month price lock guarantee")
                recommendation_score += 0.65

            if recommendations:
                recommendations_df.at[idx, "Recommendations"] = " | ".join(
                    recommendations
                )

                recommendations_df.at[idx, "RecommendationScore"] = (
                    recommendation_score / len(recommendations)
                )

            else:
                recommendations_df.at[idx, "Recommendations"] = (
                    "General retention offer"
                )

                recommendations_df.at[idx, "RecommendationScore"] = 0.3

        logger.info(
            f"Generated recommendations for {len(recommendations_df)} customers"
        )
        return recommendations_df

    def get_feature_importance(
        self, model: Any, feature_names: List[str], top_n: int = 15
    ) -> pd.DataFrame:
        """
        Extract feature importance from a trained model.

        Args:
            model: Trained model with feature importance attribute.
            feature_names: List of feature names.
            top_n: Number of top features to return.

        Returns:
            DataFrame with feature importance scores.
        """

        try:
            importances = None

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])

            elif hasattr(model, "named_steps"):
                if hasattr(
                    model.named_steps.get("classifier", None), "feature_importances_"
                ):
                    importances = model.named_steps["classifier"].feature_importances_

                elif hasattr(model.named_steps.get("classifier", None), "coef_"):
                    importances = np.abs(model.named_steps["classifier"].coef_[0])

            if importances is None:
                logger.warning("Could not extract feature importance from model")
                return pd.DataFrame()

            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names[: len(importances)],
                    "Importance": importances,
                }
            )

            importance_df = importance_df.sort_values("Importance", ascending=False)
            if top_n is not None and top_n < len(importance_df):
                importance_df = importance_df.head(top_n)

            importance_df["CumulativeImportance"] = importance_df["Importance"].cumsum()

            self.feature_importance = importance_df
            return importance_df

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return pd.DataFrame()

    def select_features(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        n_features: Optional[int] = None,
        importance_threshold: float = 0.95,
        method: str = "model_based",
        model: Optional[Any] = None,
    ) -> List[str]:
        """
        Select important features based on various methods.

        Args:
            df: Feature DataFrame.
            target: Target series.
            n_features: Number of features to select. If None, uses importance_threshold.
            importance_threshold: Cumulative importance threshold (0-1).
            method: Method for feature selection ('model_based', 'statistical', 'correlation').
            model: Pre-trained model for model-based selection.

        Returns:
            List of selected feature names.
        """

        if method == "model_based":
            return self._model_based_selection(
                df, target, n_features, importance_threshold, model
            )

        elif method == "statistical":
            return self._statistical_selection(df, target, n_features)

        elif method == "correlation":
            return self._correlation_selection(df, target, n_features)

        else:
            logger.warning(
                f"Unknown feature selection method: {method}. Using model_based."
            )

            return self._model_based_selection(
                df, target, n_features, importance_threshold, model
            )

    def _model_based_selection(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        n_features: Optional[int] = None,
        importance_threshold: float = 0.95,
        model: Optional[Any] = None,
    ) -> List[str]:
        """
        Select features based on model feature importance.

        Args:
            df: Feature DataFrame.
            target: Target series.
            n_features: Number of features to select.
            importance_threshold: Cumulative importance threshold.
            model: Pre-trained model. If None, trains a new model.

        Returns:
            List of selected feature names.
        """

        if model is None:
            try:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=config.random_seed,
                    n_jobs=config.n_jobs,
                )
                model.fit(df, target)

            except Exception as e:
                logger.error(f"Error training model for feature selection: {str(e)}")
                return list(df.columns)

        importance_df = self.get_feature_importance(model, df.columns.tolist())
        if importance_df.empty:
            logger.warning("Could not get feature importance. Returning all features.")
            return list(df.columns)

        if n_features is not None:
            selected_features = importance_df.head(min(n_features, len(importance_df)))[
                "Feature"
            ].tolist()

        else:
            selected_features = importance_df[
                importance_df["CumulativeImportance"] <= importance_threshold
            ]["Feature"].tolist()

            if not selected_features:
                selected_features = [importance_df.iloc[0]["Feature"]]

        logger.info(
            f"Selected {len(selected_features)} features using model-based selection"
        )

        return selected_features

    def _statistical_selection(
        self, df: pd.DataFrame, target: pd.Series, n_features: Optional[int] = None
    ) -> List[str]:
        """
        Select features based on statistical tests.

        Args:
            df: Feature DataFrame.
            target: Target series.
            n_features: Number of features to select.

        Returns:
            List of selected feature names.
        """
        from sklearn.feature_selection import SelectKBest, f_classif

        if n_features is None:
            n_features = min(10, len(df.columns))

        try:
            selector = SelectKBest(f_classif, k=n_features)
            selector.fit(df, target)

            selected_mask = selector.get_support()
            selected_features = df.columns[selected_mask].tolist()

            logger.info(
                f"Selected {len(selected_features)} features using statistical selection"
            )

            return selected_features

        except Exception as e:
            logger.error(f"Error in statistical feature selection: {str(e)}")
            return list(df.columns)

    def _correlation_selection(
        self, df: pd.DataFrame, target: pd.Series, n_features: Optional[int] = None
    ) -> List[str]:
        """
        Select features based on correlation with target and remove collinear features.

        Args:
            df: Feature DataFrame.
            target: Target series.
            n_features: Number of features to select.

        Returns:
            List of selected feature names.
        """
        try:
            df_with_target = df.copy()
            df_with_target["target"] = target

            target_correlation = df_with_target.corr()["target"].drop("target")
            abs_correlation = target_correlation.abs().sort_values(ascending=False)

            if n_features is None:
                n_features = min(10, len(df.columns))

            ordered_features = abs_correlation.index.tolist()
            selected_features = []

            for feature in ordered_features:
                if len(selected_features) >= n_features:
                    break

                if not selected_features:
                    selected_features.append(feature)
                    continue

                corr_with_selected = (
                    df[selected_features + [feature]]
                    .corr()
                    .loc[feature, selected_features]
                    .abs()
                )

                if corr_with_selected.max() < 0.7:
                    selected_features.append(feature)

            logger.info(
                f"Selected {len(selected_features)} features using correlation selection"
            )

            return selected_features

        except Exception as e:
            logger.error(f"Error in correlation feature selection: {str(e)}")
            return list(df.columns)

    def reduce_dimensions(
        self, df: pd.DataFrame, n_components: int = 2, method: str = "pca"
    ) -> pd.DataFrame:
        """
        Reduce dimensionality of features for visualization or further analysis.

        Args:
            df: Input DataFrame.
            n_components: Number of components to keep.
            method: Dimensionality reduction method ('pca' or 'tsne').

        Returns:
            DataFrame with reduced dimensions.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)

        try:
            if method.lower() == "pca":
                from sklearn.decomposition import PCA

                reducer = PCA(
                    n_components=n_components, random_state=config.random_seed
                )

            elif method.lower() == "tsne":
                from sklearn.manifold import TSNE

                reducer = TSNE(
                    n_components=n_components,
                    random_state=config.random_seed,
                    n_jobs=config.n_jobs,
                )

            else:
                logger.warning(
                    f"Unknown dimensionality reduction method: {method}. Using PCA."
                )
                from sklearn.decomposition import PCA

                reducer = PCA(
                    n_components=n_components, random_state=config.random_seed
                )

            reduced_data = reducer.fit_transform(scaled_data)
            reduced_df = pd.DataFrame(
                reduced_data,
                columns=[f"{method.upper()}{i+1}" for i in range(n_components)],
                index=df.index,
            )

            self.transformers[f"{method.lower()}_reducer"] = reducer
            self.transformers[f"{method.lower()}_scaler"] = scaler

            if method.lower() == "pca":
                logger.info(
                    f"PCA explained variance ratio: {reducer.explained_variance_ratio_}"
                )

            return reduced_df

        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {str(e)}")
            return pd.DataFrame()

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        method: str = "one-hot",
        max_categories: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Encode categorical features using various methods.

        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical column names.
            method: Encoding method ('one-hot', 'label', 'target', 'count').
            max_categories: Maximum number of categories to encode.

        Returns:
            DataFrame with encoded features.
        """

        df_encoded = df.copy()
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        if not categorical_cols:
            logger.warning("No categorical columns found for encoding")
            return df_encoded

        try:
            if method == "one-hot":
                from sklearn.preprocessing import OneHotEncoder

                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                    drop="if_binary",
                    max_categories=max_categories,
                )

                encoded_data = encoder.fit_transform(df[categorical_cols])
                feature_names = encoder.get_feature_names_out(categorical_cols)
                encoded_df = pd.DataFrame(
                    encoded_data, columns=feature_names, index=df.index
                )

                df_encoded = df_encoded.drop(columns=categorical_cols)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

                self.transformers["one_hot_encoder"] = encoder

            elif method == "label":
                from sklearn.preprocessing import LabelEncoder

                for col in categorical_cols:
                    encoder = LabelEncoder()
                    df_encoded[f"{col}_encoded"] = encoder.fit_transform(df[col])

                    self.transformers[f"label_encoder_{col}"] = encoder

                df_encoded = df_encoded.drop(columns=categorical_cols)

            elif method == "target":
                if config.target_column not in df.columns:
                    logger.warning(
                        "Target column not found for target encoding. Using label encoding."
                    )

                    return self.encode_categorical_features(
                        df, categorical_cols, "label", max_categories
                    )

                target = df[config.target_column]

                for col in categorical_cols:
                    target_means = df.groupby(col)[config.target_column].mean()
                    df_encoded[f"{col}_target_encoded"] = df[col].map(target_means)

                    self.transformers[f"target_encoder_{col}"] = target_means.to_dict()

                df_encoded = df_encoded.drop(columns=categorical_cols)

            elif method == "count":
                for col in categorical_cols:
                    count_map = df[col].value_counts()
                    df_encoded[f"{col}_count_encoded"] = df[col].map(count_map)

                    self.transformers[f"count_encoder_{col}"] = count_map.to_dict()

                df_encoded = df_encoded.drop(columns=categorical_cols)

            else:
                logger.warning(
                    f"Unknown encoding method: {method}. Using one-hot encoding."
                )

                return self.encode_categorical_features(
                    df, categorical_cols, "one-hot", max_categories
                )

            logger.info(
                f"Encoded {len(categorical_cols)} categorical features using {method} encoding"
            )

            return df_encoded

        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
            return df

    def _ensure_numeric(self, df, column):
        """
        Ensure a column is of numeric type for arithmetic operations
        """

        if column in df.columns:
            if pd.api.types.is_categorical_dtype(
                df[column]
            ) or pd.api.types.is_object_dtype(df[column]):
                return pd.to_numeric(df[column], errors="coerce")

            return df[column]
        return None
