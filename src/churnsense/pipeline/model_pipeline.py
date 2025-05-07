"""Pipeline module for ChurnSense modeling workflow"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import time
from pathlib import Path
import joblib
import datetime

from churnsense.config import config
from churnsense.utils.logging import setup_logger, JsonLogger
from churnsense.utils.exceptions import (
    ModelCreationError,
    ModelTrainingError,
    ModelEvaluationError,
)
from churnsense.model.trainer import ModelTrainer
from churnsense.model.evaluator import ModelEvaluator
from churnsense.model.predictor import ChurnPredictor
from churnsense.data.features import FeatureEngineering
from churnsense.data.processor import DataProcessor

logger = setup_logger(__name__)
pipeline_logger = JsonLogger("pipeline")


class ModelPipeline:
    """
    End-to-end pipeline for churn prediction modeling
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        models_path: Optional[Union[str, Path]] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize the model pipeline

        Args:
            data_path: Path to the data file
            models_path: Path to save models
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs to run
        """

        self.data_path = (
            Path(data_path) if data_path is not None else Path(config.data_path)
        )
        self.models_path = (
            Path(models_path) if models_path is not None else Path(config.models_dir)
        )
        self.random_state = random_state or config.random_seed
        self.n_jobs = n_jobs or config.n_jobs

        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineering()
        self.model_trainer = ModelTrainer(
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        self.model_evaluator = ModelEvaluator()
        self.predictor = None

        self.pipeline_start_time = None
        self.pipeline_end_time = None
        self.pipeline_metrics = {}
        self.best_model = None
        self.best_model_name = None

        self.models_path.mkdir(parents=True, exist_ok=True)

    def run_pipeline(
        self,
        data: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        test_size: Optional[float] = None,
        stratify: bool = True,
        feature_engineering: bool = True,
        models_to_train: Optional[List[str]] = None,
        tune_hyperparameters: bool = True,
        evaluation_reports: bool = True,
        save_models: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full model development pipeline

        Args:
            data: Input DataFrame (loaded from data_path if None)
            target_col: Target column name
            test_size: Test split size
            stratify: Whether to use stratified sampling
            feature_engineering: Whether to perform feature engineering
            models_to_train: List of models to train
            tune_hyperparameters: Whether to tune hyperparameters
            evaluation_reports: Whether to generate evaluation reports
            save_models: Whether to save trained models

        Returns:
            Dictionary with pipeline results

        Raises:
            Various exceptions from component classes
        """

        self.pipeline_start_time = time.time()
        pipeline_logger.log_event("pipeline_start", "Starting model pipeline")

        if target_col is None:
            target_col = config.target_column

        if test_size is None:
            test_size = config.test_size

        logger.info("Step 1: Loading and preparing data")
        X_train, X_test, y_train, y_test, feature_info = self._prepare_data(
            data, target_col, test_size, stratify, feature_engineering
        )

        logger.info("Step 2: Training models")
        trained_models = self._train_models(X_train, y_train, models_to_train)

        logger.info("Step 3: Evaluating models")
        evaluation_results = self._evaluate_models(trained_models, X_test, y_test)

        if tune_hyperparameters and self.best_model is not None:
            logger.info("Step 4: Tuning hyperparameters for best model")
            self._tune_best_model(X_train, y_train, X_test, y_test)

        else:
            logger.info("Step 4: Skipping hyperparameter tuning")

        logger.info("Step 5: Creating predictor with best model")
        if self.best_model is not None:
            self.predictor = ChurnPredictor(model=self.best_model)

            threshold_metrics = self.predictor.get_threshold_metrics(X_test, y_test)
            optimal_threshold = self.predictor.find_optimal_threshold(
                X_test, y_test, metric="roi"
            )

            self.pipeline_metrics["optimal_threshold"] = optimal_threshold

        if evaluation_reports and self.best_model is not None:
            logger.info("Step 6: Generating evaluation reports")
            self._generate_evaluation_reports(X_test, y_test)

        else:
            logger.info("Step 6: Skipping evaluation reports")

        if save_models:
            logger.info("Step 7: Saving models")
            self._save_models()

        else:
            logger.info("Step 7: Skipping model saving")

        self.pipeline_end_time = time.time()
        runtime = self.pipeline_end_time - self.pipeline_start_time

        self.pipeline_metrics["runtime"] = runtime
        self.pipeline_metrics["completed_at"] = datetime.datetime.now().isoformat()

        logger.info(f"Pipeline completed in {runtime:.2f} seconds")
        pipeline_logger.log_event(
            "pipeline_complete",
            "Model pipeline completed",
            {
                "runtime": runtime,
                "best_model": self.best_model_name,
                "metrics": {
                    k: v
                    for k, v in self.pipeline_metrics.items()
                    if not isinstance(v, (dict, list, np.ndarray))
                },
            },
        )

        return {
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "predictor": self.predictor,
            "metrics": self.pipeline_metrics,
            "runtime": runtime,
        }

    def _prepare_data(
        self,
        data: Optional[pd.DataFrame],
        target_col: str,
        test_size: float,
        stratify: bool,
        feature_engineering: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Prepare data for modeling
        """

        data_prep_start = time.time()

        if data is None:
            logger.info(f"Loading data from {self.data_path}")
            from churnsense.data.loader import load_data

            data = load_data(self.data_path)

        logger.info("Cleaning data")
        df_clean = self.data_processor.clean_data(data)

        self.data_processor.validate_data(df_clean)
        if feature_engineering:
            logger.info("Engineering features")
            df_featured = self.feature_engineer.create_features(df_clean)

        else:
            df_featured = df_clean

        from churnsense.data.loader import get_feature_types

        feature_types = get_feature_types(df_featured)
        logger.info(f"Splitting data with test_size={test_size}, stratify={stratify}")

        from churnsense.data.loader import prepare_train_test_split

        X_train, X_test, y_train, y_test = prepare_train_test_split(
            df_featured, target_col, test_size, self.random_state, stratify
        )

        logger.info("Preprocessing features")
        X_train = self.data_processor.preprocess_features(
            X_train, feature_types["categorical"], feature_types["numerical"], fit=True
        )

        X_test = self.data_processor.preprocess_features(
            X_test, feature_types["categorical"], feature_types["numerical"], fit=False
        )

        preprocessor_path = self.models_path / "preprocessor.pkl"
        self.data_processor.save_preprocessor(preprocessor_path)
        logger.info(f"Preprocessor saved to {preprocessor_path}")

        data_prep_time = time.time() - data_prep_start
        logger.info(f"Data preparation completed in {data_prep_time:.2f} seconds")

        self.pipeline_metrics["data_preparation_time"] = data_prep_time
        self.pipeline_metrics["dataset_size"] = len(data)
        self.pipeline_metrics["train_size"] = len(X_train)
        self.pipeline_metrics["test_size"] = len(X_test)
        self.pipeline_metrics["features"] = {
            "total": len(X_train.columns),
            "categorical": len(feature_types["categorical"]),
            "numerical": len(feature_types["numerical"]),
            "engineered": (
                len(self.feature_engineer.created_features)
                if feature_engineering
                else 0
            ),
        }

        feature_info = {
            "feature_types": feature_types,
            "feature_names": X_train.columns.tolist(),
            "engineered_features": (
                self.feature_engineer.created_features if feature_engineering else []
            ),
        }

        return X_train, X_test, y_train, y_test, feature_info

    def _train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train models
        """

        train_start = time.time()

        self.model_trainer.register_models()
        trained_models = self.model_trainer.train_all_models(
            X_train, y_train, models_to_train
        )

        cv_results = {}
        for model_name in trained_models.keys():
            cv_result = self.model_trainer.cross_validate(
                X_train, y_train, model_name, n_folds=config.cv_folds
            )

            cv_results[model_name] = cv_result

        train_time = time.time() - train_start
        logger.info(f"Model training completed in {train_time:.2f} seconds")

        self.pipeline_metrics["training_time"] = train_time
        self.pipeline_metrics["models_trained"] = len(trained_models)
        self.pipeline_metrics["cv_results"] = {
            name: {"mean_score": result["mean_score"], "std_score": result["std_score"]}
            for name, result in cv_results.items()
        }

        return trained_models

    def _evaluate_models(
        self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate models
        """

        eval_start = time.time()
        evaluation_results = self.model_trainer.evaluate_all_models(X_test, y_test)

        self.best_model = self.model_trainer.get_best_model()
        self.best_model_name = self.model_trainer.best_model_name

        eval_summary = self.model_trainer.get_performance_summary()
        if (
            self.best_model is not None
            and hasattr(self.best_model, "feature_importances_")
            or hasattr(self.best_model, "coef_")
        ):
            feature_names = X_test.columns.tolist()
            importance_df = self.model_evaluator.evaluate_feature_importance(
                self.best_model, X_test, y_test, feature_names
            )

            self.pipeline_metrics["top_features"] = importance_df.head(10)[
                ["feature", "importance_mean"]
            ].to_dict(orient="records")

        eval_time = time.time() - eval_start
        logger.info(f"Model evaluation completed in {eval_time:.2f} seconds")

        self.pipeline_metrics["evaluation_time"] = eval_time
        if self.best_model_name:
            self.pipeline_metrics["best_model"] = {
                "name": self.best_model_name,
                "metrics": evaluation_results.get(self.best_model_name, {}),
            }

        return evaluation_results

    def _tune_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """
        Tune hyperparameters for the best model
        """

        if self.best_model_name is None:
            logger.warning("No best model found for hyperparameter tuning")
            return

        tune_start = time.time()
        base_model_name = self.best_model_name
        if base_model_name.startswith("tuned_"):
            base_model_name = base_model_name[6:]

        tuning_results = self.model_trainer.tune_hyperparameters(
            X_train, y_train, base_model_name, n_folds=config.cv_folds
        )

        tuned_model_name = f"tuned_{base_model_name}"
        if tuned_model_name in self.model_trainer.trained_models:
            tuned_metrics = self.model_trainer.evaluate_model(
                self.model_trainer.trained_models[tuned_model_name],
                X_test,
                y_test,
                tuned_model_name,
            )

            if (
                tuned_metrics["roc_auc"]
                > self.pipeline_metrics["best_model"]["metrics"]["roc_auc"]
            ):

                self.best_model = self.model_trainer.trained_models[tuned_model_name]
                self.best_model_name = tuned_model_name

                self.pipeline_metrics["best_model"] = {
                    "name": tuned_model_name,
                    "metrics": tuned_metrics,
                }

                logger.info(f"Tuned model '{tuned_model_name}' is now the best model")

        tune_time = time.time() - tune_start
        logger.info(f"Hyperparameter tuning completed in {tune_time:.2f} seconds")

        self.pipeline_metrics["tuning_time"] = tune_time
        self.pipeline_metrics["tuning_results"] = {
            "best_params": tuning_results.get("best_params", {}),
            "best_score": tuning_results.get("best_score", None),
        }

    def _generate_evaluation_reports(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """
        Generate evaluation reports for the best model
        """

        if self.best_model is None:
            logger.warning("No best model found for generating evaluation reports")
            return

        reports_start = time.time()
        best_model_results = self.model_evaluator.evaluate_model(
            self.best_model, X_test, y_test, self.best_model_name
        )

        self.model_evaluator.plot_confusion_matrix(
            self.best_model,
            X_test,
            y_test,
            self.best_model_name,
            save_path=self.model_evaluator.results_path,
        )

        business_df = self.model_evaluator.evaluate_business_impact(
            self.best_model, X_test, y_test
        )

        self.model_evaluator.plot_business_impact(
            business_df, save_path=self.model_evaluator.results_path
        )

        self.model_evaluator.plot_metric_at_thresholds(
            self.best_model, X_test, y_test, save_path=self.model_evaluator.results_path
        )

        report = self.model_evaluator.generate_evaluation_report(self.best_model_name)
        reports_time = time.time() - reports_start
        logger.info(f"Evaluation reports generated in {reports_time:.2f} seconds")

        self.pipeline_metrics["reports_generation_time"] = reports_time

    def _save_models(self) -> None:
        """
        Save trained models to disk
        """

        save_start = time.time()

        if self.best_model is not None:
            production_model_name = f"production_{self.best_model_name}"
            metadata = {
                "pipeline_metrics": self.pipeline_metrics,
                "production_ready": True,
                "created_at": datetime.datetime.now().isoformat(),
            }

            best_model_path = self.model_trainer.save_model(
                self.best_model, model_name=production_model_name, metadata=metadata
            )

            logger.info(
                f"Best model saved as production model: {production_model_name}"
            )

            if config.save_all_models:
                for model_name, model in self.model_trainer.trained_models.items():
                    if model_name != production_model_name:
                        self.model_trainer.save_model(model, model_name=model_name)

                logger.info(
                    f"Saved {len(self.model_trainer.trained_models)} trained models"
                )

        save_time = time.time() - save_start
        logger.info(f"Model saving completed in {save_time:.2f} seconds")

        self.pipeline_metrics["model_saving_time"] = save_time

    def load_production_model(self) -> ChurnPredictor:
        """
        Load the latest production model

        Returns:
            Predictor with loaded model
        """

        production_models = list(self.models_path.glob("production_*.pkl"))
        if not production_models:
            logger.warning("No production models found")
            return None

        latest_model_path = sorted(
            production_models, key=lambda p: p.stat().st_mtime, reverse=True
        )[0]

        logger.info(f"Loading production model from {latest_model_path}")

        model, metadata = self.model_trainer.load_model(latest_model_path)
        predictor = ChurnPredictor(model=model)

        if metadata and "pipeline_metrics" in metadata:
            if "optimal_threshold" in metadata["pipeline_metrics"]:
                predictor.threshold = metadata["pipeline_metrics"]["optimal_threshold"]
                logger.info(f"Using optimal threshold: {predictor.threshold}")

        self.predictor = predictor
        self.best_model = model
        self.best_model_name = metadata.get("model_name", latest_model_path.stem)

        return predictor

    def predict(
        self,
        data: pd.DataFrame,
        return_proba: bool = True,
        threshold: Optional[float] = None,
        explain: bool = False,
    ) -> Dict[str, Any]:
        """
        Make predictions using the best model

        Args:
            data: Data to predict on
            return_proba: Whether to include probabilities
            threshold: Custom threshold (overrides optimal threshold)
            explain: Whether to include feature contribution explanations

        Returns:
            Dictionary with predictions
        """

        if self.predictor is None:
            if self.best_model is not None:
                self.predictor = ChurnPredictor(model=self.best_model)

            else:
                self.predictor = self.load_production_model()

        if self.predictor is None:
            logger.error("No model available for prediction")
            raise ModelCreationError("No model available for prediction")

        if explain and len(data) == 1:
            result = self.predictor.explain_prediction(data)

        elif len(data) == 1:
            result = self.predictor.predict_customer(
                data.iloc[0], return_proba, threshold
            )

        else:
            result = self.predictor.predict(data, return_proba, threshold)

        return result
