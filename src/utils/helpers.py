"""
Helper functions for the ChurnSense project
"""

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple

from src.utils.config import CONFIG


def save_fig(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure with a standardized format.

    Args:
      fig (plt.Figure): The matplotlib figure to save.
      filename (str): The name of the file to save the figure as.
      dpi (int, optional): The resolution of the saved figure in dots per inch.
                           Default is 300.

    Returns:
      None: This function does not return any value. It saves the figure to
            specified path.
    """

    full_path = Path(CONFIG["figures_path"]) / filename
    fig.savefig(full_path, bbox_inches="tight", dpi=dpi)

    print(f"Figure saved: {full_path}")


def format_runtime(seconds: float) -> str:
    """
    Format runtime in human-readable format.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Human-readable time string.
    """

    if seconds < 60:
        return f"{seconds:.2f} seconds"

    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"

    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def save_evaluation_result(results: Dict, model_name: str) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results (Dict): Dictionary containing evaluation metrics.
        model_name (str): Name of the model.

    Returns:
        None
    """

    results_copy = results.copy()
    for key, value in results_copy.items():
        if isinstance(value, np.ndarray):
            results_copy[key] = value.tolist()

        elif isinstance(value, np.integer):
            results_copy[key] = int(value)

        elif isinstance(value, np.floating):
            results_copy[key] = float(value)

    full_path = Path(CONFIG.evaluation_path) / f"{model_name}_evaluation.json"
    with open(full_path, "w") as f:
        json.dump(results_copy, f, indent=2)

    print(f"Evaluation results saved: {full_path}")


def timer_decorator(func):
    """
    Decorator to measure and print execution time of functions.

    Args:
        func: Function to time.

    Returns:
        Wrapped function with timing.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"{func.__name__} executed in {format_runtime(execution_time)}")
        return result

    return wrapper


def save_model_results(df: pd.DataFrame, filename: str) -> None:
    """
    Save model comparison results to CSV.

    Args:
        df (pd.DataFrame): DataFrame containing model comparison results.
        filename (str): Name of the file to save results as.

    Returns:
        None
    """

    full_path = Path(CONFIG["results_path"]) / filename
    df.to_csv(full_path, index=False)

    print(f"Results saved: {full_path}")
