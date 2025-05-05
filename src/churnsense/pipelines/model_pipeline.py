# churnsense/pipelines/model_pipeline.py
"""Model pipeline for ChurnSense project."""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

# MLflow for experiment tracking (optional)
try:
    import mlflow
    from mlflow.sklearn import log_model

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from churnsense.config import config
from churnsense.data.loader import prepare_train_test_split
from churnsense.models.factory import ModelFactory
from churnsense.models.evaluation import ModelEvaluator
from churnsense.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelPipeline:
    """
    Pipeline for training, evaluating, and deploying customer churn prediction models.
    """

    def __init__(self, use_mlflow: bool = True):
        """
        Initialize the model pipeline.

        Args:
            use_mlflow: Whether to use MLflow for tracking experiments.
        """

        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_types = None
        self.models = {}
        self.model_performances = {}
        self.best_model = None
        self.best_model_name = None
        self.best_model_path = None
        self.evaluator = ModelEvaluator()

        if self.use_mlflow:
            experiment_name = "churnsense_experiments"

            try:
                mlflow.create_experiment(experiment_name)

            except:
                pass

            mlflow.set_experiment(experiment_name)

    def run_pipeline(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numerical_cols: List[str],
        target_col: Optional[str] = None,
        model_types: Optional[List[str]] = None,
        tune_best: bool = True,
        save_models: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete model pipeline.

        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical feature names.
            numerical_cols: List of numerical feature names.
            target_col: Target column name.
            model_types: Model types to train.
            tune_best: Whether to tune hyperparameters for the best model.
            save_models: Whether to save trained models.

        Returns:
            Dictionary with pipeline results.
        """

        pipeline_start = time.time()
        logger.info("Starting model pipeline")

        self.feature_types = {
            "categorical": categorical_cols,
            "numerical": numerical_cols,
        }

        self.X_train, self.X_test, self.y_train, self.y_test = prepare_train_test_split(
            df, target_col, stratify=True
        )

        self.train_baseline_models(model_types, save_models)
        if tune_best and self.best_model:
            self._tune_best_model(save_models)

        if save_models:
            self._prepare_production_model()

        execution_time = time.time() - pipeline_start
        logger.info(f"Model pipeline completed in {execution_time:.2f} seconds")

        return {
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "best_model_path": self.best_model_path,
            "model_performances": self.model_performances,
            "execution_time": execution_time,
        }

    def train_baseline_models(
        self,
        model_types: Optional[List[str]] = None,
        save_models: bool = True,
    ) -> None:
        """
        Train multiple baseline models and evaluate their performance.

        Args:
            model_types: List of model types to train.
            save_models: Whether to save trained models.
        """

        logger.info("Training baseline models")
        model_candidates = ModelFactory.create_model_candidates(
            self.feature_types["categorical"],
            self.feature_types["numerical"],
            model_types,
        )

        for pipeline, name in model_candidates:
            if self.use_mlflow:
                mlflow.start_run(run_name=f"baseline_{name}")

            try:
                logger.info(f"Training {name}")
                train_start = time.time()

                pipeline.fit(self.X_train, self.y_train)
                train_time = time.time() - train_start

                logger.info(f"Training completed in {train_time:.2f} seconds")
                evaluation_results = self.evaluator.evaluate_model(
                    pipeline, self.X_test, self.y_test, name
                )

                self.models[name] = pipeline
                self.model_performances[name] = evaluation_results

                if self.use_mlflow:
                    params = self._extract_params(pipeline)
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)

                    for metric_name, metric_value in evaluation_results.items():
                        if isinstance(metric_value, (int, float)):
                            mlflow.log_metric(metric_name, metric_value)

                    log_model(pipeline, f"model_{name.lower().replace(' ', '_')}")

                if save_models:
                    model_path = self._save_model(
                        pipeline, f"baseline_{name.lower().replace(' ', '_')}"
                    )

                    logger.info(f"Model saved to {model_path}")

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")

            finally:
                if self.use_mlflow:
                    mlflow.end_run()

        self._identify_best_model()

    def _tune_best_model(self, save_model: bool = True) -> None:
        """
        Tune hyperparameters for the best performing model.

        Args:
            save_model: Whether to save the tuned model.
        """

        if not self.best_model or not self.best_model_name:
            logger.warning("No best model available for tuning")
            return

        logger.info(f"Tuning hyperparameters for {self.best_model_name}")

        if self.use_mlflow:
            mlflow.start_run(run_name=f"tuned_{self.best_model_name}")

        try:
            model_type = self.best_model_name.lower().replace(" ", "_")
            param_grid = ModelFactory.get_hyperparameter_grid(model_type)
            cv = StratifiedKFold(
                n_splits=config.cv_folds,
                shuffle=True,
                random_state=config.random_seed,
            )

            if len(param_grid) > 20:
                logger.info("Using randomized search for hyperparameter tuning")
                search = RandomizedSearchCV(
                    estimator=self.best_model,
                    param_distributions=param_grid,
                    n_iter=20,
                    scoring="roc_auc",
                    cv=cv,
                    verbose=1,
                    random_state=config.random_seed,
                    n_jobs=config.n_jobs,
                    return_train_score=True,
                )

            else:
                logger.info("Using grid search for hyperparameter tuning")
                search = GridSearchCV(
                    estimator=self.best_model,
                    param_grid=param_grid,
                    scoring="roc_auc",
                    cv=cv,
                    verbose=1,
                    n_jobs=config.n_jobs,
                    return_train_score=True,
                )

            tuning_start = time.time()
            search.fit(self.X_train, self.y_train)
            tuning_time = time.time() - tuning_start

            logger.info(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best cross-validation score: {search.best_score_:.4f}")

            tuned_model = search.best_estimator_
            tuned_name = f"Tuned {self.best_model_name}"
            evaluation_results = self.evaluator.evaluate_model(
                tuned_model, self.X_test, self.y_test, tuned_name
            )

            self.models[tuned_name] = tuned_model
            self.model_performances[tuned_name] = evaluation_results

            best_auc = self.model_performances[self.best_model_name].get("roc_auc", 0)
            tuned_auc = evaluation_results.get("roc_auc", 0)

            if tuned_auc > best_auc:
                logger.info(
                    f"Tuned model improved ROC AUC from {best_auc:.4f} to {tuned_auc:.4f}"
                )
                self.best_model = tuned_model
                self.best_model_name = tuned_name

            else:
                logger.info(
                    f"Tuned model did not improve performance ({tuned_auc:.4f} vs {best_auc:.4f})"
                )

            if self.use_mlflow:
                for param_name, param_value in search.best_params_.items():
                    mlflow.log_param(param_name, param_value)

                mlflow.log_metric("cv_best_score", search.best_score_)
                mlflow.log_metric("tuning_time", tuning_time)

                for metric_name, metric_value in evaluation_results.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)

                log_model(tuned_model, f"model_{tuned_name.lower().replace(' ', '_')}")

            if save_model:
                model_path = self._save_model(tuned_model, f"tuned_{model_type}")

                logger.info(f"Tuned model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error tuning {self.best_model_name}: {str(e)}")

        finally:
            if self.use_mlflow:
                mlflow.end_run()

    def _prepare_production_model(self) -> None:
        """
        Prepare the best model for production use.
        """

        if not self.best_model or not self.best_model_name:
            logger.warning("No best model available for production")
            return

        logger.info(f"Preparing production model from {self.best_model_name}")

        model_type = (
            self.best_model_name.lower().replace(" ", "_").replace("tuned_", "")
        )
        production_name = f"production_{model_type}"

        try:
            model_path = self._save_model(self.best_model, production_name)
            self.best_model_path = model_path
            logger.info(f"Production model saved to {model_path}")

            self._create_model_card(production_name)

        except Exception as e:
            logger.error(f"Error preparing production model: {str(e)}")

    def _save_model(self, model, name: str) -> Path:
        """
        Save a model to disk.

        Args:
            model: Model to save.
            name: Name of the model file.

        Returns:
            Path to saved model.
        """

        models_dir = Path(config.models_path)
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f"{name}.pkl"
        joblib.dump(model, model_path)

        return model_path

    def _identify_best_model(self) -> None:
        """
        Identify the best performing model based on ROC AUC.
        """

        if not self.model_performances:
            logger.warning("No model performances available")
            return

        sorted_models = sorted(
            self.model_performances.items(),
            key=lambda x: x[1].get("roc_auc", 0),
            reverse=True,
        )

        if sorted_models:
            best_name, best_performance = sorted_models[0]
            logger.info(
                f"Best model: {best_name} (ROC AUC: {best_performance.get('roc_auc', 0):.4f})"
            )

            self.best_model_name = best_name
            self.best_model = self.models.get(best_name)

    def _extract_params(self, model) -> Dict[str, Any]:
        """
        Extract model parameters for logging.

        Args:
            model: Model to extract parameters from.

        Returns:
            Dictionary of parameters.
        """

        params = {}
        if hasattr(model, "get_params"):
            try:
                pipeline_params = model.get_params()

                for param_name, param_value in pipeline_params.items():
                    if "classifier__" in param_name and not isinstance(
                        param_value, object
                    ):
                        clean_name = param_name.replace("classifier__", "")
                        params[clean_name] = param_value
            except:
                pass

        return params

    def _create_model_card(self, model_name: str) -> None:
        """
        Create a model card with metadata for the model.

        Args:
            model_name: Name of the model.
        """

        if not self.best_model_name or model_name not in str(self.best_model_path):
            logger.warning("No best model available for model card creation")
            return

        model_card = {
            "model_name": model_name,
            "model_type": self.best_model_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework": "scikit-learn",
            "python_version": os.environ.get("PYTHON_VERSION", "3.x"),
            "performance_metrics": self.model_performances.get(
                self.best_model_name, {}
            ),
            "feature_types": self.feature_types,
        }

        models_dir = Path(config.models_path)
        model_card_path = models_dir / f"{model_name}_card.json"

        import json

        with open(model_card_path, "w") as f:
            json.dump(model_card, f, indent=4)

        logger.info(f"Model card saved to {model_card_path}")


def run_model_pipeline(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    target_col: Optional[str] = None,
    model_types: Optional[List[str]] = None,
    tune_best: bool = True,
    save_models: bool = True,
    use_mlflow: bool = False,
) -> ModelPipeline:
    """
    Run the complete model pipeline.

    Args:
        df: Input DataFrame.
        categorical_cols: List of categorical feature names.
        numerical_cols: List of numerical feature names.
        target_col: Target column name.
        model_types: Model types to train.
        tune_best: Whether to tune hyperparameters for the best model.
        save_models: Whether to save trained models.
        use_mlflow: Whether to use MLflow for experiment tracking.

    Returns:
        ModelPipeline object with results.
    """
    
    pipeline = ModelPipeline(use_mlflow=use_mlflow)
    pipeline.run_pipeline(
      df=df,
      categorical_cols=categorical_cols,
      numerical_cols=numerical_cols,
      target_col=target_col,
      model_types=model_types,
      tune_best=tune_best,
      save_models=save_models
    )

    return pipeline
