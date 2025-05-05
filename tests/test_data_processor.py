# tests/test_data_processor.py
"""Tests for the DataProcessor class."""

import pandas as pd
import numpy as np
import pytest

from churnsense.data.processor import DataProcessor


def test_clean_data(sample_data):
    """
    Test the clean_data method.
    """
    
    processor = DataProcessor()

    df = sample_data.copy()
    df.loc[0, "SeniorCitizen"] = 1
    df.loc[1:5, "TotalCharges"] = " "  

    cleaned_df = processor.clean_data(df)

    assert cleaned_df.loc[0, "SeniorCitizen"] == "Yes"
    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])
    assert cleaned_df["TotalCharges"].isna().sum() == 0

    for i in range(1, 6):
        expected = df.loc[i, "MonthlyCharges"] * df.loc[i, "tenure"]
        actual = cleaned_df.loc[i, "TotalCharges"]
        assert abs(actual - expected) < 1e-10


def test_validate_data(sample_data):
    """
    Test the validate_data method.
    """

    processor = DataProcessor()

    valid, issues = processor.validate_data(sample_data)
    assert valid
    assert len(issues) == 0

    df = sample_data.copy()

    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    df.loc[10, "TotalCharges"] = df.loc[10, "MonthlyCharges"] * 0.5
    valid, issues = processor.validate_data(df)

    assert not valid
    assert len(issues) >= 2


def test_preprocess_features(sample_data):
    """
    Test the preprocess_features method
    """

    processor = DataProcessor()
    categorical_cols = sample_data.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols.remove("customerID")
    categorical_cols.remove("Churn")

    numerical_cols = sample_data.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    processed_df = processor.preprocess_features(
        sample_data, categorical_cols, numerical_cols
    )

    for col in numerical_cols:
        assert col in processed_df.columns
        assert processed_df[col].mean() == pytest.approx(0, abs=1e-10)
        assert processed_df[col].std() == pytest.approx(1, abs=1e-10)

    for col in categorical_cols:
        assert col not in processed_df.columns

    assert processed_df.shape[1] > len(numerical_cols)
    assert "customerID" not in processed_df.columns
    assert "Churn" not in processed_df.columns
