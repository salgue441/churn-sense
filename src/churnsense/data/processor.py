# churnsense/data/processor.py
"""Data preprocessing module for ChurnSense."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats


from churnsense.config import config
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    """
    Class for data cleaning and preprocessing.
    """

    def __init__(self):
        """
        Initialize data processor.
        """

        self.imputers = {}
        self.scalers = {}
        self.encoders = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean input DataFrame by handling missing values and data type conversions.

        Args:
            df: Input DataFrame.

        Returns:
            Cleaned DataFrame.
        """

        logger.info("Cleaning data")
        df_clean = df.copy()

        if "SeniorCitizen" in df_clean.columns:
            df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map(
                {0: "No", 1: "Yes"}
            )

            logger.info("✓ Converted 'SeniorCitizen' to categorical")

        if (
            "TotalCharges" in df_clean.columns
            and df_clean["TotalCharges"].dtype == "object"
        ):
            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"], errors="coerce"
            )

            missing_count = df_clean["TotalCharges"].isnull().sum()
            if missing_count > 0:
                logger.info(
                    f"✓ Converted 'TotalCharges' to numeric. Found {missing_count} missing values"
                )

                if (
                    "MonthlyCharges" in df_clean.columns
                    and "tenure" in df_clean.columns
                ):
                    mask = df_clean["TotalCharges"].isnull()
                    df_clean.loc[mask, "TotalCharges"] = (
                        df_clean.loc[mask, "MonthlyCharges"]
                        * df_clean.loc[mask, "tenure"]
                    )

                    logger.info(
                        f"✓ Filled {mask.sum()} missing values in 'TotalCharges'"
                    )
            else:
                logger.info("✓ Converted 'TotalCharges' to numeric")

        missing_counts = df_clean.isnull().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values after cleaning")

            for col in missing_counts[missing_counts > 0].index:
                logger.warning(f"- {col}: {missing_counts[col]} missing values")

        else:
            logger.info("✓ No missing values after cleaning")

        return df_clean

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality and integrity.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple containing validation result and list of issues.
        """

        logger.info("Validating data")
        issues = []

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")

        id_col = config.id_column
        if id_col in df.columns:
            id_duplicates = df[id_col].duplicated().sum()

            if id_duplicates > 0:
                issues.append(f"Found {id_duplicates} duplicate {id_col} values")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True) * 100
            rare_cats = value_counts[value_counts < 1]

            if len(rare_cats) > 0:
                categories = ", ".join(rare_cats.index)
                issues.append(
                    f"Column '{col}' has rare categories: {categories} (each <1%)"
                )

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in num_cols:
            if col == id_col:
                continue

            if df[col].nunique() < 10:
                continue

            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = (z_scores > 3).sum()

            if outliers > 0 and outliers / len(df) > 0.01:
                issues.append(
                    f"Column '{col}' has {outliers} extreme outliers (>3 std)"
                )

        if (
            "MonthlyCharges" in df.columns
            and "TotalCharges" in df.columns
            and "tenure" in df.columns
        ):
            inconsistent = df[
                (df["tenure"] > 0) & (df["TotalCharges"] < df["MonthlyCharges"])
            ]

            if len(inconsistent) > 0:
                issues.append(
                    f"Found {len(inconsistent)} customers with TotalCharges < MonthlyCharges"
                )

        if config.target_column in df.columns:
            target_dist = df[config.target_column].value_counts(normalize=True) * 100
            min_class_pct = target_dist.min()

            if min_class_pct < 10:
                issues.append(
                    f"Severe class imbalance: minority class is only {min_class_pct:.1f}%"
                )

        valid = len(issues) == 0

        if valid:
            logger.info("✓ Data validation passed with no issues")

        else:
            logger.warning(f"✗ Data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"- {issue}")

        return valid, issues

    def preprocess_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numerical_cols: List[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocess features by scaling numerical features and encoding categorical ones.

        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical feature names.
            numerical_cols: List of numerical feature names.
            fit: Whether to fit the transformers or use previously fitted ones.

        Returns:
            Preprocessed DataFrame.
        """

        logger.info("Preprocessing features")
        df_processed = df.copy()

        if numerical_cols:
            logger.info(f"Preprocessing {len(numerical_cols)} numerical features")
            df_processed = self._preprocess_numerical(df_processed, numerical_cols, fit)

        if categorical_cols:
            logger.info(f"Preprocessing {len(categorical_cols)} categorical features")
            df_processed = self._preprocess_categorical(
                df_processed, categorical_cols, fit
            )

        return df_processed

    def _preprocess_numerical(
        self, df: pd.DataFrame, numerical_cols: List[str], fit: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess numerical features.

        Args:
            df: Input DataFrame.
            numerical_cols: List of numerical feature names.
            fit: Whether to fit the transformers or use previously fitted ones.

        Returns:
            DataFrame with preprocessed numerical features.
        """

        df_processed = df.copy()

        if fit or "numerical" not in self.imputers:
            self.imputers["numerical"] = SimpleImputer(strategy="median")
            logger.info("Fitting numerical imputer")
            self.imputers["numerical"].fit(df[numerical_cols])

        imputed_values = self.imputers["numerical"].transform(df[numerical_cols])
        df_processed[numerical_cols] = imputed_values

        if fit or "numerical" not in self.scalers:
            self.scalers["numerical"] = StandardScaler()
            logger.info("Fitting numerical scaler")

            self.scalers["numerical"].fit(df_processed[numerical_cols])

        scaled_values = self.scalers["numerical"].transform(
            df_processed[numerical_cols]
        )

        df_processed[numerical_cols] = scaled_values

        return df_processed

    def _preprocess_categorical(
        self, df: pd.DataFrame, categorical_cols: List[str], fit: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess categorical features using one-hot encoding.

        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical feature names.
            fit: Whether to fit the transformers or use previously fitted ones.

        Returns:
            DataFrame with preprocessed categorical features.
        """

        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd

        df_processed = df.copy()
        if fit or "categorical" not in self.imputers:
            self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")
            logger.info("Fitting categorical imputer")
            self.imputers["categorical"].fit(df[categorical_cols])

        imputed_values = self.imputers["categorical"].transform(df[categorical_cols])
        df_processed[categorical_cols] = imputed_values

        if fit or "categorical" not in self.encoders:
            self.encoders["categorical"] = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore", drop="if_binary"
            )

            logger.info("Fitting one-hot encoder")
            self.encoders["categorical"].fit(df_processed[categorical_cols])

        encoded_values = self.encoders["categorical"].transform(
            df_processed[categorical_cols]
        )

        feature_names = self.encoders["categorical"].get_feature_names_out(
            categorical_cols
        )

        encoded_df = pd.DataFrame(
            encoded_values, columns=feature_names, index=df_processed.index
        )

        df_processed = df_processed.drop(columns=categorical_cols)
        df_processed = pd.concat([df_processed, encoded_df], axis=1)

        return df_processed
