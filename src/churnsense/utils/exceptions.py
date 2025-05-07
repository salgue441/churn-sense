"""Custom exceptions for ChurnSense."""

from typing import Any, Dict, Optional


class ChurnSenseError(Exception):
    """Base exception for all ChurnSense errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            message: Error message.
            details: Additional error details.
        """

        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """
        String representation of the error.
        """

        if self.details:
            return f"{self.message} - Details: {self.details}"

        return self.message


class DataLoadError(ChurnSenseError):
    """Exception raised when data cannot be loaded."""

    pass


class DataValidationError(ChurnSenseError):
    """Exception raised when data validation fails."""

    pass


class FeatureEngineeringError(ChurnSenseError):
    """Exception raised when feature engineering fails."""

    pass


class ModelCreationError(ChurnSenseError):
    """Exception raised when a model cannot be created."""

    pass


class ModelTrainingError(ChurnSenseError):
    """Exception raised when a model cannot be trained."""

    pass


class ModelEvaluationError(ChurnSenseError):
    """Exception raised when a model cannot be evaluated."""

    pass


class ModelSaveError(ChurnSenseError):
    """Exception raised when a model cannot be saved."""

    pass


class ModelLoadError(ChurnSenseError):
    """Exception raised when a model cannot be loaded."""

    pass


class PredictionError(ChurnSenseError):
    """Exception raised when predictions cannot be made."""

    pass


class VisualizationError(ChurnSenseError):
    """Exception raised when visualizations cannot be created."""

    pass


class ConfigurationError(ChurnSenseError):
    """Exception raised when there is a configuration error."""

    pass


class DashboardError(ChurnSenseError):
    """Exception raised when there is a dashboard error."""

    pass


class APIError(ChurnSenseError):
    """Exception raised when there is an API error."""

    pass


class ResourceNotFoundError(ChurnSenseError):
    """Exception raised when a resource is not found."""

    pass


class UnauthorizedAccessError(ChurnSenseError):
    """Exception raised when access is unauthorized."""

    pass


class ValidationError(ChurnSenseError):
    """Exception raised when validation fails."""

    pass
