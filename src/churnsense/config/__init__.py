"""
Configuration module for ChurnSense
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import cached_property
from dataclasses import field

from pydantic import Field, BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BusinessMetrics(BaseModel):
    """
    Business metrics for ROI calculations
    """

    avg_customer_value: float = 1000.0
    retention_campaign_cost: float = 50.0
    retention_success_rate: float = 0.30


class ChurnSenseConfig(BaseSettings):
    """
    Configuration settings for ChurnSense
    """

    # Paths
    base_dir: Path = Path.cwd()
    data_dir: Path = Field(default="data")
    models_dir: Path = Field(default="models")
    reports_dir: Path = Field(default="reports")
    logs_dir: Path = Field(default="logs")

    data_path: Path = Field(default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    processed_data_path: Path = Field(default="data/processed/cleaned_churn_data.csv")

    # Model settings
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.15
    cv_folds: int = 5
    n_jobs: int = -1

    # Target settings
    target_column: str = "Churn"
    positive_class: str = "Yes"
    id_column: str = "customerID"

    # Business metrics
    business_metrics: BusinessMetrics = Field(default_factory=BusinessMetrics)

    # Web service settings
    host: str = "0.0.0.0"
    port: int = 8050
    debug: bool = False

    # Feature selection
    default_features: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "categorical": [],
            "numerical": ["tenure", "MonthlyCharges", "TotalCharges"],
        }
    )

    model_config = SettingsConfigDict(
        env_prefix="CHURNSENSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    @field_validator("test_size", "validation_size")
    @classmethod
    def validate_sizes(cls, v: float) -> float:
        """
        Validate size parameters are between 0 and 1
        """

        if not 0 < v < 1:
            raise ValueError("Size must be between 0 and 1")

        return v

    @model_validator(mode="after")
    def validate_paths(self) -> "ChurnSenseConfig":
        """
        Validate and create paths
        """

        self.create_directories()
        return self

    def create_directories(self) -> None:
        """
        Create necessary directories
        """

        for directory in [
            self.data_dir,
            self.models_dir,
            self.reports_dir,
            self.logs_dir,
        ]:
            full_path = self.base_dir / directory
            full_path.mkdir(parents=True, exist_ok=True)

        for subdir in ["raw", "processed"]:
            (self.base_dir / self.data_dir / subdir).mkdir(parents=True, exist_ok=True)

        (self.base_dir / self.models_dir / "evaluation").mkdir(
            parents=True, exist_ok=True
        )

        (self.base_dir / self.reports_dir / "figures").mkdir(
            parents=True, exist_ok=True
        )

        (self.base_dir / self.reports_dir / "results").mkdir(
            parents=True, exist_ok=True
        )

    @cached_property
    def evaluation_path(self) -> Path:
        """
        Get the path to the evaluation directory
        """

        return self.base_dir / self.models_dir / "evaluation"

    @cached_property
    def figures_path(self) -> Path:
        """
        Get the path to the figures directory
        """

        return self.base_dir / self.reports_dir / "figures"

    @cached_property
    def results_path(self) -> Path:
        """
        Get the path to the results directory
        """

        return self.base_dir / self.reports_dir / "results"

    @cached_property
    def log_path(self) -> Path:
        """
        Get the path to the log file
        """

        return self.base_dir / self.logs_dir / "churnsense.log"


# Create a global config instance
config = ChurnSenseConfig()
