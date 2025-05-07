"""
Data loading module for ChurnSense
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from churnsense.config import config
from churnsense.utils.logging import setup_logger
from churnsense.utils.exceptions import DataLoadError, DataValidationError

logger = setup_logger(__name__)


async def load_data_async(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the customer churn dataset from CSV asynchronously.

    Args:
      data_path: Path to the CSV file. Uses the config path if none.

    Returns:
      Loaded DataFrame.

    Raises:
      FileNotFoundError: If the data file doesn't exist.
      pd.errors.EmptyDataError: If the file is empty.
      pd.errors.ParserError: If the file is not a valid CSV.
    """

    return await asyncio.to_thread(load_data, data_path)


def load_data(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the customer churn dataset from CSV

    Args:
      data_path: Path to the CSV file. Uses the config path if none.

    Returns:
      Loaded DataFrame.

    Raises:
      FileNotFoundError: If the data file doesn't exist.
      pd.errors.EmptyDataError: If the file is empty.
      pd.errors.ParserError: If the file is not a valid CSV.
    """

    if data_path is None:
        data_path = config.data_path

    path = Path(data_path)
    if not path.exists():
        error_msg = f"Data file not found: {path}"

        logger.error(error_msg)
        raise DataLoadError(error_msg, {"path": str(path)})

    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(
            path, low_memory=False, dtype_backend="numpy_nullable", on_bad_lines="warn"
        )

        logger.info(
            f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns"
        )

        if config.target_column not in df.columns:
            logger.warning(f"Target column '{config.target_column}' not found in data")

        if config.id_column not in df.columns:
            logger.warning(f"ID column '{config.id_column}' not found in data")

        return df

    except pd.errors.EmptyDataError as e:
        error_msg = f"Empty data file: {path}"

        logger.error(error_msg)
        raise DataLoadError(error_msg, {"path": str(path)}) from e

    except pd.errors.ParserError as e:
        error_msg = f"Invalid CSV file: {path}"

        logger.error(error_msg)
        raise DataLoadError(error_msg, {"path": str(path)}) from e

    except Exception as e:
        error_msg = f"Failed to load data: {str(e)}"

        logger.error(error_msg)
        raise DataLoadError(error_msg, {"path": str(path)}) from e


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        df: Input DataFrame.
        target_col: Target column name. Uses config default if None.
        test_size: Proportion for test split. Uses config default if None.
        random_state: Random seed. Uses config default if None.
        stratify: Whether to use stratified sampling based on target.

    Returns:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target values.
        y_test: Testing target values.

    Raises:
        DataValidationError: If the data cannot be split
    """

    if target_col is None:
        target_col = config.target_column

    if test_size is None:
        test_size = config.test_size

    if random_state is None:
        random_state = config.random_seed

    if target_col not in df.columns:
        error_msg = f"Target column '{target_col}' not found in data"

        logger.error(error_msg)
        raise DataValidationError(error_msg, {"available_columns": df.columns.tolist()})

    if df[target_col].nunique() != 2:
        logger.warning(
            f"Target column '{target_col}' does not have exactly 2 unique values"
        )

    target_mapper = {config.positive_class: 1, "No": 0}
    y = df[target_col].map(target_mapper)

    if y.isna().any():
        unmapped_values = df.loc[y.isna(), target_col].unique().tolist()
        error_msg = f"Target column '{target_col}' contains values that couldn't be mapped to binary"

        logger.error(f"{error_msg}: {unmapped_values}")
        raise DataValidationError(error_msg, {"unmapped_values": unmapped_values})

    drop_cols = [col for col in [config.id_column, target_col] if col in df.columns]
    X = df.drop(columns=drop_cols)
    stratify_param = y if stratify else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param,
            shuffle=True,
        )

    except Exception as e:
        error_msg = f"Error splitting data: {str(e)}"

        logger.error(error_msg)
        raise DataValidationError(error_msg) from e

    # Log split information
    logger.info(
        f"Data split - Training: {X_train.shape[0]} samples, Testing: {X_test.shape[0]} samples"
    )

    logger.info(
        f"Churn rate - Training: {y_train.mean()*100:.2f}%, Testing: {y_test.mean()*100:.2f}%"
    )

    return X_train, X_test, y_train, y_test


def get_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify categorical and numerical features in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with lists of categorical and numerical feature names.
    """

    cols_to_exclude = [
        col for col in [config.id_column, config.target_column] if col in df.columns
    ]

    numerical_dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
        "float64",
    ]

    numerical_cols = [
        col
        for col in df.columns
        if col not in cols_to_exclude
        and (
            pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype in numerical_dtypes
        )
    ]

    categorical_cols = [
        col
        for col in df.columns
        if col not in cols_to_exclude
        and col not in numerical_cols
        and (
            pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or df[col].nunique() < 10
        )
    ]

    for col in numerical_cols.copy():
        if df[col].nunique() <= 2:
            categorical_cols.append(col)
            numerical_cols.remove(col)
            logger.info(f"Treating binary numerical feature '{col}' as categorical")

    logger.info(
        f"Identified {len(categorical_cols)} categorical features and {len(numerical_cols)} numerical features"
    )

    return {
        "categorical": categorical_cols,
        "numerical": numerical_cols,
    }


def load_and_validate_data(
    data_path: Optional[Union[str, Path]] = None,
    validate: bool = True,
    auto_convert_types: bool = True,
) -> pd.DataFrame:
    """
    Load data and validate its structure.

    Args:
        data_path: Path to the data file.
        validate: Whether to validate the data.
        auto_convert_types: Whether to automatically convert data types.

    Returns:
        Validated DataFrame.

    Raises:
        DataLoadError: If the data cannot be loaded.
        DataValidationError: If the data validation fails.
    """

    df = load_data(data_path)

    if not validate:
        return df

    if df.empty:
        raise DataValidationError("Dataset is empty")

    missing_values = df.isnull().sum()
    if missing_values.any():
        cols_with_missing = missing_values[missing_values > 0]
        logger.warning(f"Found missing values in {len(cols_with_missing)} columns")

        for col, count in cols_with_missing.items():
            logger.warning(
                f"  {col}: {count} missing values ({count/len(df)*100:.2f}%)"
            )

    if auto_convert_types:
        df = auto_convert_data_types(df)

    return df


def auto_convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically convert data types for better memory usage and performance.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with optimized data types.
    """
    df_converted = df.copy()

    # Convert integer columns to the smallest possible integer type
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_min >= 0:
            if col_max < 2**8:
                df_converted[col] = df[col].astype(np.uint8)

            elif col_max < 2**16:
                df_converted[col] = df[col].astype(np.uint16)

            elif col_max < 2**32:
                df_converted[col] = df[col].astype(np.uint32)

        else:
            if col_min > -(2**7) and col_max < 2**7:
                df_converted[col] = df[col].astype(np.int8)

            elif col_min > -(2**15) and col_max < 2**15:
                df_converted[col] = df[col].astype(np.int16)

            elif col_min > -(2**31) and col_max < 2**31:
                df_converted[col] = df[col].astype(np.int32)

    for col in df.select_dtypes(include=["float64"]).columns:
        df_converted[col] = df[col].astype(np.float32)

    # Convert categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() < len(df) * 0.5:
            df_converted[col] = df[col].astype("category")

    original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
    converted_memory = df_converted.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = (1 - converted_memory / original_memory) * 100

    logger.info(
        f"Memory usage reduced from {original_memory:.2f} MB to {converted_memory:.2f} MB ({reduction:.2f}% reduction)"
    )

    return df_converted
