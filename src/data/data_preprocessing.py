"""
Data preprocessing module for the ChurnSense project.
This module handles data cleaning, transformation, and validation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from src.utils.config import CONFIG
from src.utils.helpers import timer_decorator


@timer_decorator
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input dataset by handling missing values, converting data types, etc.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    df_clean = df.copy()
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].map({0: "No", 1: "Yes"})
        print("✓ Converted 'SeniorCitizen' to categorical.")

    if (
        "TotalCharges" in df_clean.columns
        and df_clean["TotalCharges"].dtype == "object"
    ):
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"], errors="coerce"
        )

        missing_count = df_clean["TotalCharges"].isnull().sum()
        if missing_count > 0:
            print(
                f"✓ Converted 'TotalCharges' to numeric. Found {missing_count} missing values."
            )

            if "MonthlyCharges" in df_clean.columns and "tenure" in df_clean.columns:
                mask = df_clean["TotalCharges"].isnull()
                df_clean.loc[mask, "TotalCharges"] = (
                    df_clean.loc[mask, "MonthlyCharges"] * df_clean.loc[mask, "tenure"]
                )

                print(
                    f"✓ Filled {mask.sum()} missing values in 'TotalCharges' based on MonthlyCharges × tenure"
                )

    remaining_missing = df_clean.isnull().sum().sum()
    print(f"\nRemaining missing values after cleaning: {remaining_missing}")

    if remaining_missing == 0:
        print("✅ All missing values have been successfully handled.")

    else:
        print("⚠️ There are still missing values that need to be addressed.")

        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        print(f"Columns with missing values: {missing_cols}")

    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """
    Save the processed DataFrame to CSV.

    Args:
        df (pd.DataFrame): Processed DataFrame to save.
        output_path (str, optional): Path to save the CSV file. If None, uses CONFIG["processed_data_path"].

    Returns:
        None
    """

    if output_path is None:
        output_path = CONFIG["processed_data_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Processed data saved to: {output_path}")


def validate_categorical_features(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate categorical features by checking for rare categories or unexpected values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple containing:
            bool: True if validation passed, False otherwise.
            list: List of validation issues if any.
    """

    issues = []
    validation_passed = True
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        if col == CONFIG["id_column"] or col == CONFIG["target_column"]:
            continue

        value_counts = df[col].value_counts(normalize=True) * 100
        rare_categories = value_counts[value_counts < 1].index.tolist()

        if rare_categories:
            issues.append(
                f"Column '{col}' has rare categories: {rare_categories} (each <1% of data)"
            )

            validation_passed = False

        if col == "SeniorCitizen":
            valid_values = ["Yes", "No"]
            invalid_values = [v for v in df[col].unique() if v not in valid_values]

            if invalid_values:
                issues.append(f"Column '{col}' has unexpected values: {invalid_values}")
                validation_passed = False

    return validation_passed, issues


def validate_numerical_features(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate numerical features by checking for outliers and unreasonable values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple containing:
            bool: True if validation passed, False otherwise.
            list: List of validation issues if any.
    """
    
    issues = []
    validation_passed = True
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numerical_cols:
        if col == CONFIG["id_column"]:
            continue

        non_negative_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        if col in non_negative_cols and (df[col] < 0).any():
            negative_count = (df[col] < 0).sum()
            issues.append(f"Column '{col}' has {negative_count} negative values")
            validation_passed = False

        mean = df[col].mean()
        std = df[col].std()
        outliers = ((df[col] - mean).abs() > 3 * std).sum()

        if outliers > 0 and outliers / len(df) > 0.01:  # If more than 1% are outliers
            issues.append(
                f"Column '{col}' has {outliers} extreme outliers (>3 std from mean)"
            )

            validation_passed = False

    return validation_passed, issues
