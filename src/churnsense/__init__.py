# src/churnsense/__init__.py
"""ChurnSense - Customer Churn Prediction Platform."""

import logging

from churnsense.config import config
from churnsense.data.loader import load_data
from churnsense.pipelines.data_pipeline import run_data_pipeline
from churnsense.pipelines.model_pipeline import run_model_pipeline
from churnsense.utils.logging import setup_logger

__version__ = "0.1.0"

# Setup root logger
logger = setup_logger("churnsense")
logger.info(f"ChurnSense v{__version__} initialized")

__all__ = [
    "config",
    "load_data",
    "run_data_pipeline",
    "run_model_pipeline",
]
