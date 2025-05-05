"""
Data loading module for the ChurnSense project.
This module handles loading data from different sources.
"""

import pandas as pd
import time
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from src.utils.config import CONFIG
from src.utils.helpers import timer_decorator


@timer_decorator
def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the customer churn dataset from CSV.

    Args:
        data_path (str, optional): Path to the CSV file. If None, uses the path from CONFIG.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """

    if data_path is None:
        data_path = CONFIG.data_path

    df = pd.read_csv(data_path)

    print(f"Dataset dimensions: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    return df


def prepare_data_splits(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str, optional): Target column name. If None, uses CONFIG["target_column"].
        test_size (float, optional): Proportion of the dataset to include in the test split.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        Tuple containing:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Testing features.
            y_train (pd.Series): Training target.
            y_test (pd.Series): Testing target.
    """

    if target_col is None:
        target_col = CONFIG.target_column

    if test_size is None:
        test_size = CONFIG.test_size

    if random_state is None:
        random_state = CONFIG.random_seed

    target_mapper = {CONFIG.positive_class: 1, "No": 0}
    y = df[target_col].map(target_mapper)
    X = df.drop(columns=[CONFIG["id_column"], target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Churn rate in training set: {y_train.mean()*100:.2f}%")
    print(f"Churn rate in test set: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def get_feature_names(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify categorical and numerical features in the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple containing:
            list: Categorical feature names.
            list: Numerical feature names.
    """

    cols_to_exclude = [CONFIG.id_column, CONFIG.target_column]
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]
    numerical_cols = [col for col in numerical_cols if col not in cols_to_exclude]

    return categorical_cols, numerical_cols
