"""
Feature engineering module for the ChurnSense project.
This module handles creating new features and transforming existing ones.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from scipy import stats

from src.utils.config import CONFIG


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features based on domain knowledge and EDA insights.

    Args:
        df (pd.DataFrame): Input DataFrame with original features.

    Returns:
        pd.DataFrame: DataFrame with added engineered features.
    """

    df_featured = df.copy()

    # Feature 1: Customer Lifetime Value (CLV)
    if "tenure" in df.columns and "MonthlyCharges" in df.columns:
        df_featured["CLV"] = df_featured["tenure"] * df_featured["MonthlyCharges"]
        print("✓ Created CLV (Customer Lifetime Value) feature")

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
        print("✓ Created ServiceCount feature")

    # Feature 3: Average Spend Per Service
    if "MonthlyCharges" in df.columns and "ServiceCount" in df_featured.columns:
        df_featured["AvgSpendPerService"] = df_featured.apply(
            lambda x: x["MonthlyCharges"] / max(x["ServiceCount"], 1), axis=1
        )
        print("✓ Created AvgSpendPerService feature")

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
        print("✓ Created TenureGroup feature")

    # Feature 5: Has Security Services (both OnlineSecurity and TechSupport)
    if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
        df_featured["HasSecurityServices"] = (
            (df_featured["OnlineSecurity"] == "Yes")
            & (df_featured["TechSupport"] == "Yes")
        ).astype(int)
        print("✓ Created HasSecurityServices feature")

    # Feature 6: Contract Duration in Months
    if "Contract" in df.columns:
        contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
        df_featured["ContractDuration"] = df_featured["Contract"].map(contract_map)
        print("✓ Created ContractDuration feature")

    # Feature 7: Tenure to Contract Ratio (loyalty ratio)
    if "tenure" in df.columns and "ContractDuration" in df_featured.columns:
        df_featured["TenureContractRatio"] = (
            df_featured["tenure"] / df_featured["ContractDuration"]
        )

        print("✓ Created TenureContractRatio feature")

    print(f"Added {len(df_featured.columns) - len(df.columns)} new features")

    return df_featured


def analyze_feature_importance(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Analyze the importance of features based on statistical tests.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str, optional): Target column name. If None, uses CONFIG["target_column"].
        categorical_cols (List[str], optional): List of categorical feature names.
        numerical_cols (List[str], optional): List of numerical feature names.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames with importance metrics for:
            - 'categorical': Categorical features
            - 'numerical': Numerical features
    """

    if target_col is None:
        target_col = CONFIG["target_column"]

    target_mapper = {CONFIG["positive_class"]: 1, "No": 0}
    y = df[target_col].map(target_mapper)

    if categorical_cols is None or numerical_cols is None:
        from src.data.data_loader import get_feature_names

        auto_cat_cols, auto_num_cols = get_feature_names(df)

        if categorical_cols is None:
            categorical_cols = auto_cat_cols

        if numerical_cols is None:
            numerical_cols = auto_num_cols

    results = {}

    if categorical_cols:
        cat_results = []

        for col in categorical_cols:
            # Chi-square test
            contingency_table = pd.crosstab(df[col], df[target_col])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

            # Cramer's V for effect size
            n = contingency_table.sum().sum()
            cramer_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

            if cramer_v < 0.1:
                effect = "negligible"

            elif cramer_v < 0.2:
                effect = "weak"

            elif cramer_v < 0.3:
                effect = "moderate"

            else:
                effect = "strong"

            cat_results.append(
                {
                    "Feature": col,
                    "Chi2": chi2,
                    "P-value": p,
                    "Cramer_V": cramer_v,
                    "Effect_Size": effect,
                    "Significant": p < 0.05,
                }
            )

        cat_summary = pd.DataFrame(cat_results).sort_values("Cramer_V", ascending=False)
        results["categorical"] = cat_summary

    if numerical_cols:
        num_results = []

        for col in numerical_cols:
            churned = df[df[target_col] == CONFIG["positive_class"]][col]
            retained = df[df[target_col] != CONFIG["positive_class"]][col]

            if len(churned) < 2 or len(retained) < 2:
                continue

            # Mann-Whitney U test
            try:
                u_stat, p_value = stats.mannwhitneyu(churned, retained)

                mean_diff = churned.mean() - retained.mean()
                pooled_std = np.sqrt((churned.std() ** 2 + retained.std() ** 2) / 2)
                cohens_d = abs(mean_diff / pooled_std)

                if cohens_d < 0.2:
                    effect = "negligible"

                elif cohens_d < 0.5:
                    effect = "small"

                elif cohens_d < 0.8:
                    effect = "medium"

                else:
                    effect = "large"

                num_results.append(
                    {
                        "Feature": col,
                        "U_stat": u_stat,
                        "P-value": p_value,
                        "Cohens_d": cohens_d,
                        "Effect_Size": effect,
                        "Significant": p_value < 0.05,
                        "Mean_Difference": mean_diff,
                    }
                )
            except Exception as e:
                print(f"Error analyzing {col}: {str(e)}")

        if num_results:
            num_summary = pd.DataFrame(num_results).sort_values(
                "Cohens_d", ascending=False
            )

            results["numerical"] = num_summary

    return results


def perform_customer_segmentation(
    df: pd.DataFrame, n_clusters: int = 4
) -> pd.DataFrame:
    """
    Perform customer segmentation using K-means clustering.

    Args:
        df (pd.DataFrame): Input DataFrame.
        n_clusters (int, optional): Number of clusters. Default is 4.

    Returns:
        pd.DataFrame: DataFrame with added cluster labels.
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    cluster_features = [
        "tenure",
        "MonthlyCharges",
        "CLV",
        "ServiceCount",
        "AvgSpendPerService",
    ]

    df_segmented = df.copy()
    if not all(feature in df.columns for feature in cluster_features):
        print(
            "Warning: Not all cluster features exist. Creating engineered features..."
        )

        df_segmented = create_engineered_features(df)

    X_cluster = df_segmented[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(
        n_clusters=n_clusters, random_state=CONFIG["random_seed"], n_init=10
    )
    clusters = kmeans.fit_predict(X_scaled)

    df_segmented["Cluster"] = clusters
    cluster_analysis = (
        df_segmented.groupby("Cluster")
        .agg(
            {
                "tenure": "mean",
                "MonthlyCharges": "mean",
                "TotalCharges": "mean",
                "CLV": "mean",
                "ServiceCount": "mean",
                "AvgSpendPerService": "mean",
                CONFIG["target_column"]: lambda x: (
                    x == CONFIG["positive_class"]
                ).mean()
                * 100,
            }
        )
        .round(2)
    )
