"""
Utilities module for ChurnSense.

This module provides utility functions and classes for the ChurnSense application,
including logging and exception handling.
"""

from churnsense.utils.logging import setup_logger, JsonLogger
from churnsense.utils.exceptions import (
    ChurnSenseError,
    DataLoadError,
    DataValidationError,
    FeatureEngineeringError,
    ModelCreationError,
    ModelTrainingError,
    ModelEvaluationError,
    ModelSaveError,
    ModelLoadError,
    PredictionError,
    VisualizationError,
    ConfigurationError,
    DashboardError,
    APIError,
    ResourceNotFoundError,
    UnauthorizedAccessError,
    ValidationError,
)

__all__ = [
    "setup_logger",
    "JsonLogger",
    "ChurnSenseError",
    "DataLoadError",
    "DataValidationError",
    "FeatureEngineeringError",
    "ModelCreationError",
    "ModelTrainingError",
    "ModelEvaluationError",
    "ModelSaveError",
    "ModelLoadError",
    "PredictionError",
    "VisualizationError",
    "ConfigurationError",
    "DashboardError",
    "APIError",
    "ResourceNotFoundError",
    "UnauthorizedAccessError",
    "ValidationError",
]
