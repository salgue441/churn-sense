"""
Model pipeline module for the ChurnSense project.
This module handles the entire model training, evaluation, and prediction pipeline.
"""

import pandas as pd
import numpy as np
import joblib
import time
from typing import Dict, Any, Tuple, List, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils.config import CONFIG
from src.utils.helpers import timer_decorator, save_model_results
from src.models.model_factory import (
    create_model,
    create_model_candidates,
    get_model_hyperparameters,
)
from src.models.training import (
    train_model,
    train_multiple_models,
    tune_hyperparameters,
    prepare_production_model,
)
from src.models.evaluation import (
    evaluate_model,
    compute_permutation_importance,
    evaluate_business_impact,
)
from src.data.feature_engineering import (
    identify_at_risk_customers,
    generate_retention_recommendations,
)


class ModelPipeline:
    """
    Pipeline for training, evaluating, and deploying customer churn prediction models.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model pipeline.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. If None, uses default CONFIG.
        """
        
        self.config = config if config is not None else CONFIG
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.models = {}
        self.baseline_results = None
        self.tuned_models = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}
        self.feature_importance = {}

    @timer_decorator
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
        Run the entire model pipeline from data splitting to model evaluation.

        Args:
            df (pd.DataFrame): Processed DataFrame with features.
            categorical_cols (List[str]): List of categorical feature column names.
            numerical_cols (List[str]): List of numerical feature column names.
            target_col (str, optional): Target column name. If None, uses CONFIG["target_column"].
            model_types (List[str], optional): List of model types to train.
                If None, trains default set of models.
            tune_best (bool, optional): Whether to tune hyperparameters for the best model.
                Default is True.
            save_models (bool, optional): Whether to save trained models. Default is True.

        Returns:
            Dict[str, Any]: Dictionary with pipeline results.
        """
        start_time = time.time()

        self.split_data(df, target_col)

        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

        self.train_models(model_types, save_models)

        if tune_best and self.best_model is not None:
            print(f"\nTuning hyperparameters for best model: {self.best_model_name}")
            self.tune_best_model(save_models)

        self.prepare_production_model(save_models)
        self.evaluate_models()
        self.calculate_feature_importance()
        self.assess_business_impact()

        pipeline_duration = time.time() - start_time
        results = {
            "models": self.models,
            "tuned_models": self.tuned_models,
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "baseline_results": self.baseline_results,
            "evaluation_results": self.evaluation_results,
            "feature_importance": self.feature_importance,
            "duration": pipeline_duration,
        }

        print(f"\nModel pipeline completed in {pipeline_duration:.2f} seconds")
        print(f"Best model: {self.best_model_name}")

        return results

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Split data into training and testing sets.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str, optional): Target column name. If None, uses CONFIG["target_column"].
            test_size (float, optional): Proportion of the dataset to include in the test split.
                If None, uses CONFIG["test_size"].
            random_state (int, optional): Random seed for reproducibility.
                If None, uses CONFIG["random_seed"].
        """

        if target_col is None:
            target_col = self.config["target_column"]

        if test_size is None:
            test_size = self.config["test_size"]

        if random_state is None:
            random_state = self.config["random_seed"]

        target_mapper = {self.config["positive_class"]: 1, "No": 0}
        y = df[target_col].map(target_mapper)

        X = df.drop(columns=[self.config["id_column"], target_col], errors="ignore")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(
            f"Data split - Training: {len(self.X_train)} samples, Testing: {len(self.X_test)} samples"
        )

        print(
            f"Churn rate - Training: {self.y_train.mean()*100:.2f}%, Testing: {self.y_test.mean()*100:.2f}%"
        )

    def train_models(
        self, model_types: Optional[List[str]] = None, save_models: bool = True
    ) -> None:
        """
        Create and train multiple model candidates.

        Args:
            model_types (List[str], optional): List of model types to train.
                If None, trains default set of models.
            save_models (bool, optional): Whether to save trained models. Default is True.
        """

        if self.X_train is None or self.y_train is None:
            raise ValueError("No data available. Call split_data() first.")

        if self.categorical_cols is None or self.numerical_cols is None:
            raise ValueError(
                "Feature types not defined. Set categorical_cols and numerical_cols first."
            )

        model_candidates = create_model_candidates(
            self.categorical_cols, self.numerical_cols, model_types
        )

        self.models, self.baseline_results = train_multiple_models(
            model_candidates,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            save=save_models,
        )

        self.best_model_name = self.baseline_results.iloc[0]["model_name"]
        self.best_model = self.models.get(self.best_model_name)

        print(f"\nBest baseline model: {self.best_model_name}")

    def tune_best_model(self, save_model: bool = True) -> None:
        """
        Tune hyperparameters for the best model.

        Args:
            save_model (bool, optional): Whether to save the tuned model. Default is True.
        """

        if self.best_model is None or self.best_model_name is None:
            raise ValueError("No best model available. Call train_models() first.")

        model_type = self.best_model_name.lower().replace(" ", "_")
        tuned_model, best_params = tune_hyperparameters(
            self.best_model,
            self.X_train,
            self.y_train,
            model_type,
            self.best_model_name,
        )

        tuned_model_name = f"Tuned {self.best_model_name}"
        self.tuned_models[tuned_model_name] = tuned_model

        y_pred_proba = tuned_model.predict_proba(self.X_test)[:, 1]
        from sklearn.metrics import roc_auc_score

        tuned_auc = roc_auc_score(self.y_test, y_pred_proba)
        best_auc = self.baseline_results.loc[
            self.baseline_results["model_name"] == self.best_model_name, "roc_auc"
        ].values[0]

        if tuned_auc > best_auc:
            print(
                f"Tuned model is better: AUC improved from {best_auc:.4f} to {tuned_auc:.4f}"
            )
            self.best_model = tuned_model
            self.best_model_name = tuned_model_name

        else:
            print(
                f"Tuned model did not improve: AUC {tuned_auc:.4f} vs. baseline {best_auc:.4f}"
            )

    def prepare_production_model(self, save_model: bool = True) -> None:
        """
        Prepare the best model for production use.

        Args:
            save_model (bool, optional): Whether to save the production model. Default is True.
        """

        if self.best_model is None or self.best_model_name is None:
            raise ValueError("No best model available. Call train_models() first.")

        production_name = prepare_production_model(
            self.best_model, self.best_model_name
        )

        self.production_model_name = production_name

    def evaluate_models(self) -> None:
        """
        Evaluate all trained models.
        """

        if not self.models:
            raise ValueError("No models available. Call train_models() first.")

        print("\nEvaluating models")
        for name, model in self.models.items():
            print(f"Evaluating {name}")
            self.evaluation_results[name] = evaluate_model(
                model, self.X_test, self.y_test, name
            )

        for name, model in self.tuned_models.items():
            print(f"Evaluating {name}")
            self.evaluation_results[name] = evaluate_model(
                model, self.X_test, self.y_test, name
            )

    def calculate_feature_importance(self) -> None:
        """
        Calculate feature importance for the best model.
        """

        if self.best_model is None or self.best_model_name is None:
            raise ValueError("No best model available. Call train_models() first.")

        print(f"\nCalculating feature importance for {self.best_model_name}")
        perm_importance = compute_permutation_importance(
            self.best_model, self.X_test, self.y_test
        )

        self.feature_importance["permutation"] = perm_importance

    def assess_business_impact(self) -> None:
        """
        Assess the business impact of the best model.
        """

        if self.best_model is None or self.best_model_name is None:
            raise ValueError("No best model available. Call train_models() first.")

        print(f"\nAssessing business impact of {self.best_model_name}")

        business_results = evaluate_business_impact(
            self.best_model, self.X_test, self.y_test
        )

        self.evaluation_results["business_impact"] = business_results

        print(f"Retained customers: {business_results['retained_customers']:.0f}")
        print(f"Saved revenue: ${business_results['saved_revenue']:,.2f}")
        print(f"Campaign cost: ${business_results['total_campaign_cost']:,.2f}")
        print(f"Net benefit: ${business_results['net_benefit']:,.2f}")
        print(f"ROI: {business_results['roi']*100:.2f}%")

    def identify_at_risk_customers(
        self, df: pd.DataFrame, threshold: float = 0.5, top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Identify customers at risk of churning.

        Args:
            df (pd.DataFrame): Customer data.
            threshold (float, optional): Probability threshold for churn risk. Default is 0.5.
            top_n (int, optional): If provided, returns only the top N highest risk customers.

        Returns:
            pd.DataFrame: DataFrame with at-risk customers and risk scores.
        """

        if self.best_model is None:
            raise ValueError("No best model available. Call train_models() first.")

        print(f"Identifying at-risk customers with threshold {threshold}")

        at_risk = identify_at_risk_customers(
            df,
            churn_probability_threshold=threshold,
            model=self.best_model,
            top_n=top_n,
        )

        recommendations = generate_retention_recommendations(df, at_risk)
        return recommendations

    def save_model(self, model, model_name: str) -> None:
        """
        Save a model to disk.

        Args:
            model: Model to save.
            model_name (str): Name of the model for the saved file.
        """

        model_path = Path(self.config["models_path"]) / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

    def load_model(self, model_name: str):
        """
        Load a model from disk.

        Args:
            model_name (str): Name of the model to load.

        Returns:
            Loaded model.
        """

        model_path = Path(self.config["models_path"]) / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        print(f"Model loaded: {model_path}")

        return model


# Function to run the pipeline
def run_model_pipeline(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    target_col: Optional[str] = None,
    model_types: Optional[List[str]] = None,
    tune_best: bool = True,
    save_models: bool = True,
    config: Dict[str, Any] = None,
) -> ModelPipeline:
    """
    Run the complete model pipeline.

    Args:
        df (pd.DataFrame): Processed DataFrame with features.
        categorical_cols (List[str]): List of categorical feature column names.
        numerical_cols (List[str]): List of numerical feature column names.
        target_col (str, optional): Target column name. If None, uses CONFIG["target_column"].
        model_types (List[str], optional): List of model types to train.
            If None, trains default set of models.
        tune_best (bool, optional): Whether to tune hyperparameters for the best model.
            Default is True.
        save_models (bool, optional): Whether to save trained models. Default is True.
        config (Dict[str, Any], optional): Configuration dictionary. If None, uses default CONFIG.

    Returns:
        ModelPipeline: Pipeline object with trained models.
    """

    pipeline = ModelPipeline(config)
    pipeline.run_pipeline(
        df,
        categorical_cols,
        numerical_cols,
        target_col,
        model_types,
        tune_best,
        save_models,
    )

    return pipeline
