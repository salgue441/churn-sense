"""
Data pipeline module for the ChurnSense project.
This module handles the entire data processing pipeline from loading to feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
from pathlib import Path

from src.utils.config import CONFIG
from src.utils.helpers import timer_decorator
from src.data.data_loader import load_data, get_feature_names
from src.data.data_preprocessing import (
    clean_data,
    save_processed_data,
    validate_categorical_features,
    validate_numerical_features,
)
from src.data.feature_engineering import (
    create_engineered_features,
    analyze_feature_importance,
    perform_customer_segmentation,
)


class DataPipeline:
    """
    Pipeline for loading, cleaning, and feature engineering of the customer churn data.
    """

    def __init__(self, config: CONFIG):
        """
        Initialize the data pipeline.

        Args:
            config (CONFIG): Configuration class. If None, uses default CONFIG.
        """

        self.config = config if config is not None else CONFIG()
        self.df_raw = None
        self.df_clean = None
        self.df_featured = None
        self.df_segmented = None
        self.feature_importance = None
        self.categorical_cols = None
        self.numerical_cols = None

    @timer_decorator
    def run_pipeline(
        self, data_path: Optional[str] = None, save_interim: bool = True
    ) -> pd.DataFrame:
        """
        Run the entire data pipeline from loading to feature engineering.

        Args:
            data_path (str, optional): Path to the raw data file. If None, uses CONFIG["data_path"].
            save_interim (bool, optional): Whether to save interim processed data. Default is True.

        Returns:
            pd.DataFrame: Fully processed and engineered DataFrame.
        """

        self.df_raw = self.load_data(data_path)
        self.df_clean = self.clean_data()
        self.validate_data()

        if save_interim:
            self.save_cleaned_data()

        self.df_featured = self.engineer_functions()
        self.feature_importance = self.analyze_features()
        self.df_segmented = self.segment_customer()

        return self.df_featured

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the data from file.

        Args:
            data_path (str, optional): Path to the data file. If None, uses CONFIG["data_path"].

        Returns:
            pd.DataFrame: Raw loaded DataFrame.
        """

        if data_path is None:
            data_path = self.config["data_path"]

        print(f"Loading data from: {data_path}")
        df = load_data(data_path)

        self.categorical_cols, self.numerical_cols = get_feature_names(df)
        return df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """

        if self.df_raw is None:
            raise ValueError("No data loaded. Call load_data() first.")

        print("Cleaning data")
        return clean_data(self.df_raw)

    def validate_data(self) -> Tuple[bool, list]:
        """
        Validate the cleaned data for issues.

        Returns:
            Tuple[bool, list]: Tuple containing:
                bool: Whether validation passed.
                list: List of validation issues.
        """

        if self.df_clean is None:
            raise ValueError("No cleaned data. Call clean_data() first")

        print("Validating data")
        cat_validation, cat_issues = validate_categorical_features(self.df_clean)
        num_validation, num_issues = validate_numerical_features(self.df_clean)

        issues = cat_issues + num_issues
        validation_passed = cat_validation and num_validation

        if validation_passed:
            print("✅ Data validation passed.")

        else:
            print(f"⚠️ Data validation found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")

        return validation_passed, issues

    def save_cleaned_data(self, output_path: Optional[str] = None) -> None:
        """
        Save the cleaned data to file.

        Args:
            output_path (str, optional): Path to save the file. If None, uses CONFIG["processed_data_path"].
        """

        if self.df_clean is None:
            raise ValueError("No cleaned data. Call clean_data() first.")

        if output_path is None:
            output_path = self.config["processed_data_path"]

        save_processed_data(self.df_clean, output_path)

    def engineer_features(self) -> pd.DataFrame:
        """
        Perform feature engineering on the cleaned data.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """

        if self.df_clean is None:
            raise ValueError("No cleaned data. Call clean_data() first.")

        print("Performing feature engineering")
        return create_engineered_features(self.df_clean)

    def analyze_features(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing DataFrames with importance metrics.
        """

        if self.df_featured is None:
            raise ValueError("No featured data. Call engineer_features() first.")

        print("Analyzing feature importance")
        return analyze_feature_importance(
            self.df_featured,
            self.config["target_column"],
            self.categorical_cols,
            self.numerical_cols,
        )

    def segment_customers(self, n_clusters: int = 4) -> pd.DataFrame:
        """
        Perform customer segmentation.

        Args:
            n_clusters (int, optional): Number of clusters. Default is 4.

        Returns:
            pd.DataFrame: DataFrame with cluster assignments.
        """

        if self.df_featured is None:
            raise ValueError("No featured data. Call engineer_features() first.")

        print(f"Performing customer segmentation with {n_clusters} clusters...")
        return perform_customer_segmentation(self.df_featured, n_clusters)

    def get_data(self, stage: str = "featured") -> pd.DataFrame:
        """
        Get data at a specific stage of the pipeline.

        Args:
            stage (str, optional): Pipeline stage. Options: "raw", "clean", "featured", "segmented".
                Default is "featured".

        Returns:
            pd.DataFrame: Data at the specified stage.
        """

        if stage == "raw":
            return self.df_raw

        elif stage == "clean":
            return self.df_clean

        elif stage == "featured":
            return self.df_featured

        elif stage == "segmented":
            return self.df_segmented

        else:
            raise ValueError(
                f"Unknown stage: {stage}. Valid stages: raw, clean, featured, segmented"
            )

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance analysis results.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing feature importance analysis.
        """

        if self.feature_importance is None:
            raise ValueError(
                "No feature importance analysis. Call analyze_features() first."
            )

        return self.feature_importance


# Function to run the pipeline
def run_data_pipeline(
    data_path: Optional[str] = None,
    save_interim: bool = True,
    config: CONFIG = CONFIG(),
) -> DataPipeline:
    """
    Run the complete data pipeline.

    Args:
        data_path (str, optional): Path to the raw data file. If None, uses CONFIG["data_path"].
        save_interim (bool, optional): Whether to save interim processed data. Default is True.
        config (Dict[str, Any], optional): Configuration dictionary. If None, uses default CONFIG.

    Returns:
        DataPipeline: Pipeline object with processed data.
    """

    pipeline = DataPipeline(config)
    pipeline.run_pipeline(data_path, save_interim)

    return pipeline
