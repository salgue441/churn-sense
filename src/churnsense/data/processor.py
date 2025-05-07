"""Data preprocessing module for ChurnSense"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

from churnsense.config import config
from churnsense.utils.logging import setup_logger
from churnsense.utils.exceptions import DataValidationError

logger = setup_logger(__name__)


class DataProcessor:
    """
    Class for data cleaning and preprocessing
    """

    def __init__(self):
        """
        Initialize the data processor with empty transformers.
        """

        self.imputers = {}
        self.scalers = {}
        self.encoders = {}
        self._fitted = False

    def clean_data(self, df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        """
        Clean input DataFrame by handling missing values, converting data
        types, etc.

        Args:
          df (pd.DataFrame): Input DataFrame
          inplace (bool): Whether to modify the DataFrame in place.

        Returns:
          pd.DataFrame: Cleaned DataFrame
        """

        logger.info("Cleaning data")
        df_clean = df if inplace else df.copy()

        if "SeniorCitizen" in df_clean.columns and pd.api.types.is_numeric_dtype(
            df_clean["SeniorCitizen"]
        ):
            df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map(
                {0: "No", 1: "Yes"}
            )

            logger.info("Converted 'SeniorCitizen' to categorical")

        if (
            "TotalCharges" in df_clean.columns
            and df_clean["TotalCharges"].dtype == "object"
        ):
            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"], errors="coerce"
            )

            total_missing = df_clean["TotalCharges"].isna().sum()
            if total_missing > 0:
                logger.info(
                    f"Converted 'TotalCharges' to numeric. Found {total_missing} missing values"
                )

                if (
                    "MonthlyCharges" in df_clean.columns
                    and "tenure" in df_clean.columns
                ):
                    mask = df_clean["TotalCharges"].isna()
                    df_clean.loc[mask, "TotalCharges"] = (
                        df_clean.loc[mask, "MonthlyCharges"]
                        * df_clean.loc[mask, "tenure"]
                    )

                    logger.info(f"Filled {mask.sum()} missing values in 'TotalCharges'")

            else:
                logger.info("Converted 'TotalCharges' to numeric")

        missing_counts = df_clean.isna().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            logger.warning(
                f"Found {total_missing} missing values after initial cleaning"
            )

            for col in missing_counts[missing_counts > 0].index:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_value = df_clean[col].median()
                    df_clean[col].fillna(median_value, inplace=True)

                    logger.info(
                        f"Filled {missing_counts[col]} missing values in '{col}' with median: {median_value}"
                    )

                else:
                    mode_value = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_value, inplace=True)

                    logger.info(
                        f"Filled {missing_counts[col]} missing values in '{col}' with mode: {mode_value}"
                    )

        self._optimize_dtypes(df_clean)
        return df_clean

    def _optimize_dtypes(self, df: pd.DataFrame) -> None:
        """
        Optimize data types for better memory usage.

        Args:
            df: DataFrame to optimize.
        """

        for col in df.select_dtypes(include=["int64"]).columns:
            col_min, col_max = df[col].min(), df[col].max()

            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)

                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)

                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)

            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)

                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)

                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)

        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() < len(df) * 0.5:
                df[col] = df[col].astype("category")

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality and integrity.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple containing validation result and list of issues.

        Raises:
            DataValidationError: If critical validation issues are found.
        """

        logger.info("Validating data")
        issues, critical_issues = [], []

        if df.empty:
            critical_issues.append("DataFrame is empty")
            return DataValidationError("DataFrame is empty")

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

        num_cols = df.select_dtypes(
            include=[
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "float16",
                "float32",
                "float64",
            ]
        ).columns

        for col in num_cols:
            if col == id_col or df[col].nunique() < 10:
                continue

        for col in num_cols:
            if col == id_col or df[col].nunique() < 10:
                continue

            median = df[col].median()
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            z_scores = 0.6745 * np.abs(df[col] - median) / iqr
            outliers = (z_scores > 3.5).sum()

            if outliers > 0 and outliers / len(df) > 0.01:
                issues.append(
                    f"Column '{col}' has {outliers} extreme outliers (>3.5 robust z-score)"
                )

        if all(
            col in df.columns for col in ["MonthlyCharges", "TotalCharges", "tenure"]
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
                    f"Class imbalance: minority class is only {min_class_pct:.1f}%"
                )

        # Log validation results
        if critical_issues:
            error_msg = "Critical validation issues found"
            logger.error(f"{error_msg}: {', '.join(critical_issues)}")
            raise DataValidationError(error_msg, {"issues": critical_issues})

        valid = len(issues) == 0
        if valid:
            logger.info("✓ Data validation passed with no issues")

        else:
            logger.warning(f"⚠ Data validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"- {issue}")

        return valid, issues

    def preprocess_features(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numerical_cols: List[str],
        fit: bool = True,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Preprocess features by scaling numerical features and encoding categorical ones.

        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical feature names.
            numerical_cols: List of numerical feature names.
            fit: Whether to fit the transformers or use previously fitted ones.
            n_jobs: Number of jobs for parallel processing. None uses config default.

        Returns:
            Preprocessed DataFrame.
        """

        logger.info("Preprocessing features")
        if n_jobs is None:
            n_jobs = config.n_jobs

        df_processed = df.copy()
        if n_jobs != 1 and (numerical_cols and categorical_cols):
            with ThreadPoolExecutor(max_workers=min(2, abs(n_jobs))) as executor:
                num_future = executor.submit(
                    self._preprocess_numerical, df, numerical_cols, fit
                )

                cat_future = executor.submit(
                    self._preprocess_categorical, df, categorical_cols, fit
                )

                num_result = num_future.result()
                cat_result = cat_future.result()

                df_processed = pd.concat(
                    [
                        num_result.drop(columns=categorical_cols, errors="ignore"),
                        cat_result.drop(columns=numerical_cols, errors="ignore"),
                    ],
                    axis=1,
                )

        else:
            if numerical_cols:
                logger.info(f"Preprocessing {len(numerical_cols)} numerical features")

                df_processed = self._preprocess_numerical(
                    df_processed, numerical_cols, fit
                )

            if categorical_cols:
                logger.info(
                    f"Preprocessing {len(categorical_cols)} categorical features"
                )

                df_processed = self._preprocess_categorical(
                    df_processed, categorical_cols, fit
                )

        self._fitted = True
        return df_processed

    def _preprocess_numerical(
        self, df: pd.DataFrame, numerical_cols: List[str], fit: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess numerical features with robust handling of outliers.

        Args:
            df: Input DataFrame.
            numerical_cols: List of numerical feature names.
            fit: Whether to fit the transformers or use previously fitted ones.

        Returns:
            DataFrame with preprocessed numerical features.
        """

        df_processed = df.copy()

        if not numerical_cols:
            return df_processed

        has_outliers = False
        for col in numerical_cols:
            if df[col].nunique() > 10:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1

                if iqr > 0:
                    outlier_count = (
                        (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
                    ).sum()

                    if outlier_count > len(df) * 0.01:
                        has_outliers = True
                        break

        if fit or "numerical" not in self.imputers:
            self.imputers["numerical"] = SimpleImputer(strategy="median")

            logger.info("Fitting numerical imputer")
            self.imputers["numerical"].fit(df[numerical_cols])

        imputed_values = self.imputers["numerical"].transform(df[numerical_cols])
        df_processed[numerical_cols] = imputed_values

        if fit or "numerical" not in self.scalers:
            if has_outliers:
                logger.info("Using RobustScaler due to detected outliers")
                self.scalers["numerical"] = RobustScaler()

            else:
                logger.info("Using StandardScaler")
                self.scalers["numerical"] = StandardScaler()

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
        Preprocess categorical features using advanced one-hot encoding techniques.

        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical feature names.
            fit: Whether to fit the transformers or use previously fitted ones.

        Returns:
            DataFrame with preprocessed categorical features.
        """

        df_processed = df.copy()

        if not categorical_cols:
            return df_processed

        if fit or "categorical" not in self.imputers:
            self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")

            logger.info("Fitting categorical imputer")
            self.imputers["categorical"].fit(df[categorical_cols])

        imputed_values = self.imputers["categorical"].transform(df[categorical_cols])
        df_processed[categorical_cols] = imputed_values

        if fit or "categorical" not in self.encoders:
            self.encoders["categorical"] = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                drop="if_binary",
                dtype=np.float32,
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

    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "winsorize",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.

        Args:
            df: Input DataFrame.
            columns: List of columns to check for outliers. If None, all numerical columns.
            method: Method to handle outliers ('winsorize', 'clip', 'remove').
            threshold: Z-score threshold to identify outliers.

        Returns:
            DataFrame with handled outliers.
        """

        df_result = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()

        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(
                    f"Column '{col}' is not numerical. Skipping outlier handling."
                )
                continue

            z_scores = np.abs(stats.zscore(df[col], nan_policy="omit"))
            outliers = z_scores > threshold

            if not np.any(outliers):
                logger.info(f"No outliers found in '{col}' using threshold {threshold}")
                continue

            if method == "winsorize":
                lower_limit = df[col].quantile(0.01)
                upper_limit = df[col].quantile(0.99)

                df_result[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
                logger.info(f"Winsorized {outliers.sum()} outliers in '{col}'")

            elif method == "clip":
                mean, std = df[col].mean(), df[col].std()
                lower_limit = mean - threshold * std
                upper_limit = mean + threshold * std

                df_result[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
                logger.info(f"Clipped {outliers.sum()} outliers in '{col}'")

            elif method == "remove":
                df_result.loc[outliers, col] = np.nan
                logger.info(f"Removed {outliers.sum()} outliers in '{col}'")

            else:
                logger.warning(f"Unknown outlier handling method: {method}")

        return df_result

    def save_preprocessor(self, path: str) -> None:
        """
        Save preprocessor components to disk for later use

        Args:
          path (str): Path to save the preprocessor
        """

        import joblib

        if not self._fitted:
            logger.warning("Attempting to save unfitted preprocessor")
            return

        preprocessor_data = {
            "imputers": self.imputers,
            "scalers": self.scalers,
            "encoders": self.encoders,
            "_fitted": self._fitted,
        }

        joblib.dump(preprocessor_data, path)
        logger.info(f"Preprocessor saved to {path}")

    def load_preprocessor(self, path: str) -> None:
        """
        Load preprocessor components from disk.

        Args:
            path: Path to load the preprocessor from.
        """
        import joblib

        try:
            preprocessor_data = joblib.load(path)
            self.imputers = preprocessor_data.get("imputers", {})
            self.scalers = preprocessor_data.get("scalers", {})
            self.encoders = preprocessor_data.get("encoders", {})
            self._fitted = preprocessor_data.get("_fitted", False)

            logger.info(f"Preprocessor loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise
