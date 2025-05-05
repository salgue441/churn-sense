# churnsense/data/loader.py
"""Data loading module for ChurnSense."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from churnsense.config import config
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


def load_data(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the customer churn dataset from CSV.

    Args:
        data_path: Path to the CSV file. Uses the config path if None.

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
        logger.error(f"Data file not found: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)

        logger.info(
            f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]:,} columns"
        )

        return df

    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {path}")
        raise

    except pd.errors.ParserError:
        logger.error(f"Invalid CSV file: {path}")
        raise


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
    """

    if target_col is None:
        target_col = config.target_column

    if test_size is None:
        test_size = config.test_size

    if random_state is None:
        random_state = config.random_seed

    target_mapper = {config.positive_class: 1, "No": 0}
    y = df[target_col].map(target_mapper)
    X = df.drop(columns=[config.id_column, target_col], errors="ignore")

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

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

    cols_to_exclude = [config.id_column, config.target_column]
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]
    numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]

    logger.info(
        f"Identified {len(categorical_cols)} categorical features and {len(numerical_cols)} numerical features"
    )

    return {
        "categorical": categorical_cols,
        "numerical": numerical_cols,
    }
