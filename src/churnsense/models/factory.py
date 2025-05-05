# churnsense/models/factory.py
"""Model factory for creating and configuring ML models."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Specialized models
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True

except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True

except ImportError:
    LIGHTGBM_AVAILABLE = False

from churnsense.config import config
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelFactory:
    """
    Factory for creating machine learning models with preprocessing pipelines."""

    @staticmethod
    def create_preprocessing_pipeline(
        categorical_cols: List[str],
        numerical_cols: List[str],
        categorical_strategy: str = "most_frequent",
        numerical_strategy: str = "median",
    ) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for categorical and numerical features.

        Args:
            categorical_cols: List of categorical feature column names.
            numerical_cols: List of numerical feature column names.
            categorical_strategy: Imputation strategy for categorical features.
            numerical_strategy: Imputation strategy for numerical features.

        Returns:
            Preprocessor pipeline.
        """

        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=numerical_strategy)),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_strategy)),
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

    @staticmethod
    def create_model(
        model_type: str,
        categorical_cols: List[str],
        numerical_cols: List[str],
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Pipeline, str]:
        """
        Create a model pipeline with preprocessing for a specific model type.

        Args:
            model_type: Type of model to create.
            categorical_cols: List of categorical feature column names.
            numerical_cols: List of numerical feature column names.
            model_params: Parameters for the model.

        Returns:
            Tuple of (pipeline, model_name).

        Raises:
            ValueError: If an unknown model type is specified.
            ImportError: If the requested model type is not available.
        """

        preprocessor = ModelFactory.create_preprocessing_pipeline(
            categorical_cols, numerical_cols
        )

        base_params = {"random_state": config.random_seed}
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
            if not XGBOOST_AVAILABLE:
                logger.error(
                    "XGBoost is not installed. Install with 'pip install xgboost'"
                )

                raise ImportError(
                    "XGBoost is not installed. Install with 'pip install xgboost'"
                )

            classifier = XGBClassifier(
                use_label_encoder=False, eval_metric="logloss", **base_params
            )
            name = "XGBoost"

        elif model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                logger.error(
                    "LightGBM is not installed. Install with 'pip install lightgbm'"
                )

                raise ImportError(
                    "LightGBM is not installed. Install with 'pip install lightgbm'"
                )

            classifier = LGBMClassifier(verbose=-1, **base_params)
            name = "LightGBM"

        elif model_type == "knn":
            classifier = KNeighborsClassifier()
            name = "K-Nearest Neighbors"

        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        logger.info(f"Created model pipeline: {name}")
        return pipeline, name

    @staticmethod
    def get_hyperparameter_grid(model_type: str) -> Dict[str, Any]:
        """
        Get hyperparameter grid for model tuning.

        Args:
            model_type: Type of model.

        Returns:
            Dictionary of hyperparameter grid.

        Raises:
            ValueError: If no hyperparameter grid is defined for the model type.
        """

        grids = {
            "logistic_regression": {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["l1", "l2", "elasticnet", None],
                "classifier__solver": [
                    "newton-cg",
                    "lbfgs",
                    "liblinear",
                    "sag",
                    "saga",
                ],
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
            "knn": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        }

        if model_type not in grids:
            logger.error(f"No hyperparameter grid defined for model type: {model_type}")

            raise ValueError(
                f"No hyperparameter grid defined for model type: {model_type}"
            )

        return grids[model_type]

    @staticmethod
    def create_model_candidates(
        categorical_cols: List[str],
        numerical_cols: List[str],
        model_types: Optional[List[str]] = None,
    ) -> List[Tuple[Pipeline, str]]:
        """
        Create multiple model candidates for evaluation.

        Args:
            categorical_cols: List of categorical feature column names.
            numerical_cols: List of numerical feature column names.
            model_types: List of model types to create.

        Returns:
            List of (model, name) tuples.
        """

        if model_types is None:
            model_types = [
                "logistic_regression",
                "random_forest",
                "gradient_boosting",
            ]

            if XGBOOST_AVAILABLE:
                model_types.append("xgboost")

            if LIGHTGBM_AVAILABLE:
                model_types.append("lightgbm")

        model_candidates = []
        for model_type in model_types:
            try:
                model, name = ModelFactory.create_model(
                    model_type, categorical_cols, numerical_cols
                )

                model_candidates.append((model, name))
                logger.info(f"Created model candidate: {name}")

            except (ValueError, ImportError) as e:
                logger.warning(f"Skipping model {model_type}: {str(e)}")

        return model_candidates
