"""
Model training module for the ChurnSense project.
This module handles training and optimizing models.
"""

import time
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from pathlib import Path
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils.config import CONFIG
from src.utils.helpers import timer_decorator, format_runtime, save_model_results
from src.models.model_factory import create_model, get_model_hyperparameters


@timer_decorator
def train_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    save: bool = True,
) -> Any:
    """
    Train a model on the training data.

    Args:
        model (Any): Model to train.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_name (str): Name of the model for saving.
        save (bool, optional): Whether to save the trained model. Default is True.

    Returns:
        Any: Trained model.
    """

    print(f"Training {model_name}...")
    model_start_time = time.time()
    model.fit(X_train, y_train)

    train_time = time.time() - model_start_time
    print(f"{model_name} trained in {format_runtime(train_time)}")

    if save:
        save_trained_model(model, f"baseline_{model_name.lower().replace(' ', '_')}")

    return model


@timer_decorator
def train_multiple_models(
    model_candidates: List[Tuple[Any, str]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train multiple models and compare their performance.

    Args:
        model_candidates (List[Tuple[Any, str]]): List of (model, name) tuples.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        save (bool, optional): Whether to save the trained models. Default is True.

    Returns:
        Tuple containing:
            Dict[str, Any]: Dictionary of trained models.
            pd.DataFrame: Performance comparison DataFrame.
    """

    trained_models = {}
    baseline_results = []

    for model, name in model_candidates:
        model = train_model(model, X_train, y_train, name, save=save)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model_name": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        baseline_results.append(metrics)

        print(
            f"{name} - Accuracy: {metrics['accuracy']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}"
        )

    baseline_df = pd.DataFrame(baseline_results)
    baseline_df = baseline_df[
        ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    baseline_df = baseline_df.sort_values("roc_auc", ascending=False)

    if save:
        save_model_results(baseline_df, "baseline_model_comparison.csv")

    print("\nModel Comparison:")
    print(baseline_df)

    return trained_models, baseline_df


@timer_decorator
def tune_hyperparameters(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    model_name: str,
    n_iter: Optional[int] = None,
    cv: Optional[int] = None,
    random: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune model hyperparameters using grid search or randomized search.

    Args:
        model (Any): Base model to tune.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_type (str): Type of model ('logistic_regression', 'random_forest', etc.).
        model_name (str): Descriptive name of the model.
        n_iter (int, optional): Number of iterations for randomized search. Default is None.
        cv (int, optional): Number of cross-validation folds. Default uses CONFIG['cv_folds'].
        random (bool, optional): Whether to use randomized search. Default is False (grid search).

    Returns:
        Tuple containing:
            Any: Tuned model (best estimator).
            Dict[str, Any]: Best parameters found.
    """

    param_grid = get_model_hyperparameters(model_type)

    if cv is None:
        cv = CONFIG["cv_folds"]

    cv_method = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=CONFIG["random_seed"]
    )

    print(f"Tuning hyperparameters for {model_name}...")

    if random:
        if n_iter is None:
            n_iter = 20

        print(f"Using randomized search with {n_iter} iterations")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring="roc_auc",
            cv=cv_method,
            n_iter=n_iter,
            n_jobs=CONFIG["n_jobs"],
            verbose=1,
            random_state=CONFIG["random_seed"],
            return_train_score=True,
        )

    else:
        print("Using grid search")

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv_method,
            n_jobs=CONFIG["n_jobs"],
            verbose=1,
            return_train_score=True,
        )

    start_time = time.time()
    search.fit(X_train, y_train)

    tuning_time = time.time() - start_time
    print(f"Hyperparameter tuning completed in {format_runtime(tuning_time)}")
    print(f"Best cross-validation score: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    tuned_model = search.best_estimator_
    save_trained_model(tuned_model, f"tuned_{model_name.lower().replace(' ', '_')}")

    return tuned_model, search.best_params_


def save_trained_model(model: Any, model_name: str) -> None:
    """
    Save a trained model to disk.

    Args:
        model (Any): Trained model to save.
        model_name (str): Name to use when saving the model.

    Returns:
        None
    """

    model_dir = Path(CONFIG["models_path"])
    model_dir.mkdir(parents=True, exist_ok=True)

    full_path = model_dir / f"{model_name}.pkl"
    joblib.dump(model, full_path)

    print(f"Model saved: {full_path}")


def prepare_production_model(model: Any, model_name: str) -> None:
    """
    Prepare a model for production use (save with production_ prefix).

    Args:
        model (Any): Trained model to prepare for production.
        model_name (str): Name of the model.

    Returns:
        None
    """
    
    production_name = f"production_{model_name.lower().replace(' ', '_')}"
    save_trained_model(model, production_name)

    print(f"Production model prepared: {production_name}")

    return production_name
