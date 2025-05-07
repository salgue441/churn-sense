"""Model training module for ChurnSense"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

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
from churnsense.utils.exceptions import (
    ModelCreationError,
    ModelTrainingError,
    ModelSaveError,
    ModelLoadError,
)

logger = setup_logger(__name__)


class ModelTrainer:
    """Class for training and tuning machine learning models"""

    def __init__(
        self, random_state: Optional[int] = None, n_jobs: Optional[int] = None
    ):
        """
        Initialize the model trainer

        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs to run
        """

        self.random_state = random_state or config.random_seed
        self.n_jobs = n_jobs or config.n_jobs
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.performance_metrics = {}
        self.training_history = {}

    def register_models(
        self, custom_models: Optional[Dict[str, BaseEstimator]] = None
    ) -> None:
        """
        Register all models for training

        Args:
            custom_models: Optional dictionary of custom models to include
        """

        logger.info("Registering models")
        base_models = {
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced",
                n_jobs=self.n_jobs,
            ),
            "random_forest": RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight="balanced",
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "ada_boost": AdaBoostClassifier(random_state=self.random_state),
            "svc": SVC(random_state=self.random_state, probability=True),
            "knn": KNeighborsClassifier(n_jobs=self.n_jobs),
        }

        if XGBOOST_AVAILABLE:
            base_models["xgboost"] = XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=self.n_jobs,
            )

        else:
            logger.warning("XGBoost not available, skipping")

        if LIGHTGBM_AVAILABLE:
            base_models["lightgbm"] = LGBMClassifier(
                random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs
            )

        else:
            logger.warning("LightGBM not available, skipping")

        if custom_models:
            base_models.update(custom_models)

        self.models = base_models
        logger.info(
            f"Registered {len(self.models)} models: {', '.join(self.models.keys())}"
        )

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> BaseEstimator:
        """
        Train a single model with optional parameters

        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name of the model to train
            params: Optional parameters to override defaults

        Returns:
            Trained model

        Raises:
            ModelTrainingError: If training fails
        """

        if model_name not in self.models:
            raise ModelTrainingError(
                f"Model '{model_name}' not found",
                {"available_models": list(self.models.keys())},
            )

        start_time = time.time()
        logger.info(f"Training {model_name}")

        model = self.models[model_name]
        if params:
            model = self._get_model_with_params(model_name, params)

        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            self.feature_names = X_train.columns.tolist()
            logger.info(f"Model {model_name} trained in {training_time:.2f} seconds")
            self.trained_models[model_name] = model

            self.training_history[model_name] = {
                "training_time": training_time,
                "n_samples": X_train.shape[0],
                "n_features": X_train.shape[1],
                "parameters": model.get_params(),
            }

            return model

        except Exception as e:
            error_msg = f"Error training {model_name}: {str(e)}"

            logger.error(error_msg)
            raise ModelTrainingError(error_msg, {"model": model_name}) from e

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, BaseEstimator]:
        """
        Train all registered models or a subset

        Args:
            X_train: Training features
            y_train: Training target
            models_to_train: Optional list of model names to train

        Returns:
            Dictionary of trained models
        """

        if not self.models:
            self.register_models()

        if models_to_train is None:
            models_to_train = list(self.models.keys())

        logger.info(f"Training {len(models_to_train)} models")

        trained_models = {}
        for model_name in models_to_train:
            try:
                model = self.train_model(X_train, y_train, model_name)
                trained_models[model_name] = model

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")

        self.trained_models.update(trained_models)
        if trained_models:
            logger.info(f"Successfully trained {len(trained_models)} models")

        else:
            logger.warning("No models were successfully trained")

        return trained_models

    def evaluate_model(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
    ) -> Dict[str, float]:
        """
        Evaluate a model's performance on test data

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for reporting

        Returns:
            Dictionary of performance metrics
        """
        start_time = time.time()
        logger.info(f"Evaluating {model_name}")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "evaluation_time": time.time() - start_time,
        }

        logger.info(f"Model {model_name} evaluation results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = {}

        self.performance_metrics[model_name]["test"] = metrics
        return metrics

    def evaluate_all_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models_to_evaluate: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models

        Args:
            X_test: Test features
            y_test: Test target
            models_to_evaluate: Optional list of model names to evaluate

        Returns:
            Dictionary of evaluation metrics for each model
        """

        if not self.trained_models:
            logger.warning("No trained models to evaluate")
            return {}

        if models_to_evaluate is None:
            models_to_evaluate = list(self.trained_models.keys())

        logger.info(f"Evaluating {len(models_to_evaluate)} models")

        evaluation_results = {}
        for model_name in models_to_evaluate:
            if model_name not in self.trained_models:
                logger.warning(f"Model '{model_name}' not found in trained models")
                continue

            model = self.trained_models[model_name]
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            evaluation_results[model_name] = metrics

        if evaluation_results:
            best_model_name = max(
                evaluation_results, key=lambda x: evaluation_results[x]["roc_auc"]
            )
            self.best_model_name = best_model_name
            self.best_model = self.trained_models[best_model_name]

            logger.info(
                f"Best model: {best_model_name} with ROC AUC: {evaluation_results[best_model_name]['roc_auc']:.4f}"
            )

        return evaluation_results

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        n_folds: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation for a model

        Args:
            X: Feature data
            y: Target data
            model_name: Name of the model to validate
            n_folds: Number of cross-validation folds
            scoring: Scoring metric to use

        Returns:
            Dictionary with cross-validation results
        """

        if model_name not in self.models:
            raise ModelTrainingError(
                f"Model '{model_name}' not found",
                {"available_models": list(self.models.keys())},
            )

        logger.info(f"Performing {n_folds}-fold cross-validation for {model_name}")

        start_time = time.time()
        model = self.models[model_name]

        cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )

        try:
            scores = cross_val_score(
                model, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs
            )

            cv_time = time.time() - start_time
            cv_results = {
                "mean_score": scores.mean(),
                "std_score": scores.std(),
                "scores": scores.tolist(),
                "cv_time": cv_time,
                "n_folds": n_folds,
                "scoring": scoring,
            }

            logger.info(
                f"Cross-validation for {model_name}: "
                f"Mean {scoring}={cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}"
            )

            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = {}

            self.performance_metrics[model_name]["cv"] = cv_results

            return cv_results

        except Exception as e:
            error_msg = f"Error during cross-validation for {model_name}: {str(e)}"

            logger.error(error_msg)
            raise ModelTrainingError(error_msg, {"model": model_name}) from e

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_folds: int = 5,
        scoring: str = "roc_auc",
        n_iter: Optional[int] = None,
        search_method: str = "grid",
        refit: bool = True,
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using grid or randomized search

        Args:
            X: Feature data
            y: Target data
            model_name: Name of the model to tune
            param_grid: Parameter grid to search
            n_folds: Number of cross-validation folds
            scoring: Scoring metric to use
            n_iter: Number of iterations for randomized search
            search_method: Search method ("grid" or "random")
            refit: Whether to refit the model with the best parameters

        Returns:
            Dictionary with tuning results
        """

        if model_name not in self.models:
            raise ModelTrainingError(
                f"Model '{model_name}' not found",
                {"available_models": list(self.models.keys())},
            )

        if param_grid is None:
            param_grid = self._get_default_param_grid(model_name)

        if not param_grid:
            logger.warning(
                f"No parameter grid available for {model_name}, skipping tuning"
            )
            return {}

        logger.info(
            f"Tuning hyperparameters for {model_name} using {search_method} search"
        )

        logger.info(f"Parameter grid: {param_grid}")

        start_time = time.time()
        base_model = self.models[model_name]
        cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )

        try:
            if search_method.lower() == "random":
                if n_iter is None:
                    n_iter = max(10, len(param_grid) * 3)

                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    scoring=scoring,
                    cv=cv,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    verbose=1,
                    refit=refit,
                    return_train_score=True,
                )

            else:
                search = GridSearchCV(
                    base_model,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=cv,
                    n_jobs=self.n_jobs,
                    verbose=1,
                    refit=refit,
                    return_train_score=True,
                )

            search.fit(X, y)
            tuning_time = time.time() - start_time
            tuning_results = {
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "cv_results": {
                    "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
                    "std_test_score": search.cv_results_["std_test_score"].tolist(),
                    "mean_train_score": search.cv_results_["mean_train_score"].tolist(),
                    "std_train_score": search.cv_results_["std_train_score"].tolist(),
                    "params": [str(p) for p in search.cv_results_["params"]],
                },
                "tuning_time": tuning_time,
                "n_folds": n_folds,
                "scoring": scoring,
                "search_method": search_method,
            }

            logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
            logger.info(f"Best parameters: {tuning_results['best_params']}")
            logger.info(f"Best {scoring} score: {tuning_results['best_score']:.4f}")

            if refit:
                tuned_model = search.best_estimator_
                tuned_model_name = f"tuned_{model_name}"
                self.trained_models[tuned_model_name] = tuned_model

                logger.info(f"Tuned model '{tuned_model_name}' added to trained models")

                if self.best_model_name == model_name:
                    self.best_model_name = tuned_model_name
                    self.best_model = tuned_model
                    logger.info(f"Updated best model to '{tuned_model_name}'")

            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = {}

            self.performance_metrics[model_name]["tuning"] = tuning_results
            return tuning_results

        except Exception as e:
            error_msg = f"Error during hyperparameter tuning for {model_name}: {str(e)}"

            logger.error(error_msg)
            raise ModelTrainingError(error_msg, {"model": model_name}) from e

    def create_ensemble(
        self,
        models: Dict[str, BaseEstimator],
        ensemble_method: str = "soft",
        weights: Optional[List[float]] = None,
        ensemble_name: str = "ensemble",
    ) -> BaseEstimator:
        """
        Create an ensemble model from multiple trained models

        Args:
            models: Dictionary of trained models to include in ensemble
            ensemble_method: Voting method ("soft" or "hard")
            weights: Optional weights for models in the ensemble
            ensemble_name: Name for the ensemble model

        Returns:
            Trained ensemble model

        Raises:
            ModelCreationError: If ensemble creation fails
        """

        if not models:
            raise ModelCreationError("No models provided for ensemble creation")

        estimators = [(name, model) for name, model in models.items()]

        try:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=ensemble_method,
                weights=weights,
                n_jobs=self.n_jobs,
            )

            logger.info(
                f"Created {ensemble_method} voting ensemble with {len(estimators)} models: "
                f"{', '.join([name for name, _ in estimators])}"
            )

            self.models[ensemble_name] = ensemble
            return ensemble

        except Exception as e:
            error_msg = f"Error creating ensemble model: {str(e)}"

            logger.error(error_msg)
            raise ModelCreationError(error_msg) from e

    def save_model(
        self,
        model: BaseEstimator,
        model_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a trained model to disk

        Args:
            model: Model to save
            model_path: Path to save the model to
            model_name: Name of the model (used in filename if path not provided)
            metadata: Additional metadata to save with the model

        Returns:
            Path where the model was saved

        Raises:
            ModelSaveError: If saving the model fails
        """

        if model_name is None:
            for name, m in self.trained_models.items():
                if model is m:
                    model_name = name
                    break

            if model_name is None:
                model_name = "unknown_model"

        if model_path is None:
            models_dir = Path(config.models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir / f"{model_name}.pkl"

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if metadata is None:
                metadata = {}

            if model_name in self.performance_metrics:
                metadata["performance"] = self.performance_metrics[model_name]

            if model_name in self.training_history:
                metadata["training_history"] = self.training_history[model_name]

            metadata["feature_names"] = self.feature_names
            metadata["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            metadata["model_name"] = model_name
            model_data = {"model": model, "metadata": metadata}

            joblib.dump(model_data, model_path)
            logger.info(f"Model '{model_name}' saved to {model_path}")

            return model_path

        except Exception as e:
            error_msg = f"Error saving model '{model_name}': {str(e)}"

            logger.error(error_msg)
            raise ModelSaveError(
                error_msg, {"model_name": model_name, "path": str(model_path)}
            ) from e

    def load_model(
        self, model_path: Union[str, Path]
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Load a trained model from disk

        Args:
            model_path: Path to the saved model

        Returns:
            Tuple of (loaded model, metadata)

        Raises:
            ModelLoadError: If loading the model fails
        """

        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and "model" in model_data:
                model = model_data["model"]
                metadata = model_data.get("metadata", {})

            else:
                model = model_data
                metadata = {}

            model_name = metadata.get("model_name", model_path.stem)
            logger.info(f"Model '{model_name}' loaded from {model_path}")

            self.trained_models[model_name] = model
            if "performance" in metadata:
                self.performance_metrics[model_name] = metadata["performance"]

            if "training_history" in metadata:
                self.training_history[model_name] = metadata["training_history"]

            if "feature_names" in metadata:
                self.feature_names = metadata["feature_names"]

            return model, metadata

        except Exception as e:
            error_msg = f"Error loading model from {model_path}: {str(e)}"

            logger.error(error_msg)
            raise ModelLoadError(error_msg, {"path": str(model_path)}) from e

    def _get_model_with_params(
        self, model_name: str, params: Dict[str, Any]
    ) -> BaseEstimator:
        """
        Create a new model instance with the given parameters
        """

        base_model = self.models[model_name]
        model_type = type(base_model)

        model_params = base_model.get_params()
        model_params.update(params)

        return model_type(**model_params)

    def _get_default_param_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """
        Get default parameter grid for a model
        """

        param_grids = {
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2", "elasticnet", None],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            },
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "gradient_boosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5],
                "subsample": [0.8, 1.0],
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2],
            },
            "lightgbm": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7, -1],
                "num_leaves": [31, 50, 100],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
            "svc": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.1, 0.01],
                "kernel": ["rbf", "linear", "poly", "sigmoid"],
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "ada_boost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0],
                "algorithm": ["SAMME", "SAMME.R"],
            },
        }

        return param_grids.get(model_name, {})

    def get_best_model(self) -> Optional[BaseEstimator]:
        """
        Get the best performing model based on evaluation results
        """

        return self.best_model

    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get a summary of model performance metrics

        Returns:
            DataFrame with performance metrics for all evaluated models
        """

        if not self.performance_metrics:
            logger.warning("No performance metrics available")
            return pd.DataFrame()

        metrics_list = []
        for model_name, metrics in self.performance_metrics.items():
            model_metrics = {"model_name": model_name}

            if "test" in metrics:
                for metric, value in metrics["test"].items():
                    if metric != "evaluation_time":
                        model_metrics[f"test_{metric}"] = value

            if "cv" in metrics:
                model_metrics["cv_score"] = metrics["cv"]["mean_score"]
                model_metrics["cv_std"] = metrics["cv"]["std_score"]

            if "tuning" in metrics:
                model_metrics["tuned_score"] = metrics["tuning"]["best_score"]
                model_metrics["tuning_time"] = metrics["tuning"]["tuning_time"]

            metrics_list.append(model_metrics)

        if not metrics_list:
            return pd.DataFrame()

        summary_df = pd.DataFrame(metrics_list)
        if "test_roc_auc" in summary_df.columns:
            summary_df = summary_df.sort_values("test_roc_auc", ascending=False)

        return summary_df
