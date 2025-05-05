# churnsense/pipelines/data_pipeline.py
"""Data pipeline for ChurnSense project."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd

from churnsense.config import config
from churnsense.data.loader import load_data
from churnsense.data.processor import DataProcessor
from churnsense.data.features import FeatureEngineer
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


class DataPipeline:
    """
    Pipeline for loading, cleaning, and feature engineering of customer data.
    """

    def __init__(self):
        """
        Initialize the data pipeline.
        """

        self.processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()

        self.df_raw = None
        self.df_clean = None
        self.df_featured = None
        self.df_segmented = None
        self.feature_types = None

    def run_pipeline(
        self,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        perform_segmentation: bool = True,
        n_clusters: int = 4,
    ) -> pd.DataFrame:
        """
        Run the complete data pipeline.

        Args:
            data_path: Path to input data file.
            output_path: Path to save processed data.
            perform_segmentation: Whether to perform customer segmentation.
            n_clusters: Number of clusters for segmentation.

        Returns:
            Processed DataFrame with engineered features.
        """

        start_time = time.time()
        logger.info("Starting data pipeline")

        self.df_raw = self._load_data(data_path)
        self.df_clean = self._clean_data()

        self._validate_data()
        self._extract_feature_types()

        self.df_featured = self._engineer_features()
        if perform_segmentation:
            self.df_segmented = self._segment_customers(n_clusters)
            result_df = self.df_segmented

        else:
            result_df = self.df_featured

        if output_path:
            self._save_processed_data(result_df, output_path)

        execution_time = time.time() - start_time
        logger.info(f"Data pipeline completed in {execution_time:.2f} seconds")

        return result_df

    def _load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            data_path: Path to input data file.

        Returns:
            Loaded DataFrame.
        """

        logger.info(f"Loading data from {data_path or config.data_path}")
        return load_data(data_path)

    def _clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data.

        Returns:
            Cleaned DataFrame.
        """

        if self.df_raw is None:
            logger.error("No data loaded. Call _load_data() first")
            raise ValueError("No data loaded")

        logger.info("Cleaning data")
        return self.processor.clean_data(self.df_raw)

    def _validate_data(self) -> None:
        """
        Validate the cleaned data.
        """

        if self.df_clean is None:
            logger.error("No cleaned data. Call _clean_data() first")
            raise ValueError("No cleaned data")

        logger.info("Validating data")
        self.processor.validate_data(self.df_clean)

    def _extract_feature_types(self) -> None:
        """
        Extract categorical and numerical feature types.
        """

        if self.df_clean is None:
            logger.error("No cleaned data. Call _clean_data() first")
            raise ValueError("No cleaned data")

        logger.info("Extracting feature types")

        cols_to_exclude = [config.id_column, config.target_column]
        categorical_cols = self.df_clean.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_cols = self.df_clean.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_cols = [
            col for col in categorical_cols if col not in cols_to_exclude
        ]
        numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]

        self.feature_types = {
            "categorical": categorical_cols,
            "numerical": numerical_cols,
        }

        logger.info(
            f"Identified {len(categorical_cols)} categorical features and {len(numerical_cols)} numerical features"
        )

    def _engineer_features(self) -> pd.DataFrame:
        """
        Perform feature engineering.

        Returns:
            DataFrame with engineered features.
        """

        if self.df_clean is None:
            logger.error("No cleaned data. Call _clean_data() first")
            raise ValueError("No cleaned data")

        logger.info("Engineering features")
        return self.feature_engineer.create_features(self.df_clean)

    def _segment_customers(self, n_clusters: int = 4) -> pd.DataFrame:
        """
        Perform customer segmentation.

        Args:
            n_clusters: Number of clusters for segmentation.

        Returns:
            DataFrame with cluster assignments.
        """

        if self.df_featured is None:
            logger.error("No featured data. Call _engineer_features() first")
            raise ValueError("No featured data")

        logger.info(f"Segmenting customers into {n_clusters} clusters")
        return self.feature_engineer.segment_customers(self.df_featured, n_clusters)

    def _save_processed_data(
        self, df: pd.DataFrame, output_path: Optional[str] = None
    ) -> None:
        """
        Save processed data to file.

        Args:
            df: DataFrame to save.
            output_path: Path to save file.
        """

        if output_path is None:
            output_path = config.processed_data_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")


def run_data_pipeline(
    data_path: Optional[str] = None,
    output_path: Optional[str] = None,
    perform_segmentation: bool = True,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """
    Run the complete data pipeline.

    Args:
        data_path: Path to input data file.
        output_path: Path to save processed data.
        perform_segmentation: Whether to perform customer segmentation.
        n_clusters: Number of clusters for segmentation.

    Returns:
        Processed DataFrame with engineered features.
    """

    pipeline = DataPipeline()
    return pipeline.run_pipeline(
        data_path, output_path, perform_segmentation, n_clusters
    )
