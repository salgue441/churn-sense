"""
Model factory module for the ChurnSense project.
This module handles creating and configuring different types of models.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.config import CONFIG


def create_preprocessing_pipeline(
    categorical_cols: List[str], numerical_cols: List[str]
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for categorical and numerical features.

    Args:
        categorical_cols (List[str]): List of categorical feature column names.
        numerical_cols (List[str]): List of numerical feature column names.

    Returns:
        ColumnTransformer: A preprocessor that handles both categorical and numerical features.
    """

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def create_model(
    model_type: str,
    categorical_cols: List[str],
    numerical_cols: List[str],
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, str]:
    """
    Create a model pipeline with preprocessing for a specific model type.

    Args:
        model_type (str): Type of model to create ('logistic_regression', 'random_forest', etc.).
        categorical_cols (List[str]): List of categorical feature column names.
        numerical_cols (List[str]): List of numerical feature column names.
        model_params (Dict[str, Any], optional): Parameters for the model.

    Returns:
        Tuple containing:
            Pipeline: A scikit-learn Pipeline with preprocessing and the model.
            str: A descriptive name for the model.
    """

    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    base_params = {"random_state": CONFIG.random_seed}

    if model_params:
        base_params.update(model_params)

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000, class_weight="balanced", **base_params
        )

        name = "Logistic Regression"

    elif model_type == "random_forest":
        classifier = RandomForestClassifier(class_weight="balanced", **base_params)
        name = "Random Forest"

    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(**base_params)
        name = "Gradient Boosting"

    elif model_type == "xgboost":
        classifier = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", **base_params
        )

        name = "XGBoost"

    elif model_type == "lightgbm":
        classifier = LGBMClassifier(verbose=-1, **base_params)
        name = "LightGBM"

    elif model_type == "svc":
        classifier = SVC(probability=True, **base_params)
        name = "SVC"

    elif model_type == "knn":
        classifier = KNeighborsClassifier()
        name = "K-Nearest Neighbors"

    elif model_type == "naive_bayes":
        classifier = GaussianNB()
        name = "Naive Bayes"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    return pipeline, name


def get_model_hyperparameters(model_type: str) -> Dict[str, Any]:
    """
    Get hyperparameter grid for a specific model type.

    Args:
        model_type (str): Type of model.

    Returns:
        Dict[str, Any]: Dictionary containing hyperparameter grid for the model.
    """

    param_grids = {
        "logistic_regression": {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "classifier__penalty": ["l1", "l2", "elasticnet", None],
            "classifier__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7],
            "classifier__min_samples_split": [2, 5],
            "classifier__subsample": [0.8, 1.0],
        },
        "xgboost": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0],
            "classifier__gamma": [0, 0.1, 0.2],
        },
        "lightgbm": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7, -1],
            "classifier__num_leaves": [31, 50, 100],
            "classifier__subsample": [0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.8, 0.9, 1.0],
        },
        "svc": {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__gamma": ["scale", "auto", 0.1, 0.01],
            "classifier__kernel": ["rbf", "linear", "poly", "sigmoid"],
        },
        "knn": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
    }

    if model_type in param_grids:
        return param_grids[model_type]
    
    else:
        raise ValueError(f"No hyperparameter grid defined for model type: {model_type}")


def create_model_candidates(
    categorical_cols: List[str],
    numerical_cols: List[str],
    model_types: Optional[List[str]] = None,
) -> List[Tuple[Pipeline, str]]:
    """
    Create a list of model candidates for evaluation.

    Args:
        categorical_cols (List[str]): List of categorical feature column names.
        numerical_cols (List[str]): List of numerical feature column names.
        model_types (List[str], optional): List of model types to create. If None, creates a default set.

    Returns:
        List[Tuple[Pipeline, str]]: List of (model, name) tuples.
    """

    if model_types is None:
        model_types = [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
        ]

    model_candidates = []
    for model_type in model_types:
        try:
            model, name = create_model(model_type, categorical_cols, numerical_cols)
            model_candidates.append((model, name))
            print(f"Created model: {name}")
            
        except Exception as e:
            print(f"Error creating {model_type} model: {str(e)}")

    return model_candidates
