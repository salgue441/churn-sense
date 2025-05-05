# tests/conftest.py
"""Test configuration for the ChurnSense project."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from churnsense.config import config


@pytest.fixture
def sample_data():
    """
    Generate a sample dataset for testing.
    """

    np.random.seed(42)
    n_samples = 100
    data = {
        "customerID": [f"CUST{i:05d}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], size=n_samples),
        "SeniorCitizen": np.random.choice([0, 1], size=n_samples),
        "Partner": np.random.choice(["Yes", "No"], size=n_samples),
        "Dependents": np.random.choice(["Yes", "No"], size=n_samples),
        "tenure": np.random.randint(0, 72, size=n_samples),
        "PhoneService": np.random.choice(["Yes", "No"], size=n_samples),
        "MultipleLines": np.random.choice(
            ["Yes", "No", "No phone service"], size=n_samples
        ),
        "InternetService": np.random.choice(
            ["DSL", "Fiber optic", "No"], size=n_samples
        ),
        "OnlineSecurity": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "OnlineBackup": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "DeviceProtection": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "TechSupport": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "StreamingTV": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "StreamingMovies": np.random.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ),
        "Contract": np.random.choice(
            ["Month-to-month", "One year", "Two year"], size=n_samples
        ),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
        "PaymentMethod": np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            size=n_samples,
        ),
        "MonthlyCharges": np.random.uniform(20, 120, size=n_samples),
    }

    data["TotalCharges"] = data["tenure"] * data["MonthlyCharges"]
    churn_prob = (
        (data["Contract"] == "Month-to-month") * 0.4
        + (data["MonthlyCharges"] > 70) * 0.3
        + (data["tenure"] < 12) * 0.3
    )

    churn_prob = churn_prob / churn_prob.max()
    data["Churn"] = ["Yes" if p > 0.5 else "No" for p in churn_prob]

    df = pd.DataFrame(data)

    return df


@pytest.fixture
def test_config():
    """
    Create a test configuration.
    """

    test_config = config.copy()
    test_config.data_path = "test_data.csv"
    test_config.models_path = "test_models"
    test_config.figures_path = "test_figures"
    test_config.results_path = "test_results"

    return test_config


@pytest.fixture
def trained_model_path(sample_data, tmp_path):
    """
    Create a simple trained model for testing.
    """

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import joblib

    X = sample_data.drop(columns=["customerID", "Churn"])
    y = (sample_data["Churn"] == "Yes").astype(int)

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )

    model.fit(X, y)

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    model_path = models_dir / "test_model.pkl"
    joblib.dump(model, model_path)

    return model_path
