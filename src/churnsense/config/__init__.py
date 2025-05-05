"""Configuration module for ChurnSense."""

from pathlib import Path
from typing import Optional

# Update these imports for Pydantic v2
from pydantic import Field
from pydantic_settings import BaseSettings


class ChurnSenseConfig(BaseSettings):
    """
    Configuration settings for ChurnSense.
    """

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent.parent
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    reports_dir: Path = Field(default_factory=lambda: Path("reports"))
    evaluation_path: Path = Field(default_factory=lambda: Path("models/evaluation"))

    # Data settings
    data_path: Path = Field(
        default_factory=lambda: Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    )
    processed_data_path: Path = Field(
        default_factory=lambda: Path("data/processed/cleaned_churn_data.csv")
    )

    # Model settings
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.25
    cv_folds: int = 5
    n_jobs: int = -1

    # Target settings
    target_column: str = "Churn"
    positive_class: str = "Yes"
    id_column: str = "customerID"

    # Business metrics
    avg_customer_value: float = 1000.0
    retention_campaign_cost: float = 50.0
    retention_success_rate: float = 0.30

    model_config = {
        "env_prefix": "CHURNSENSE_",
        "env_file": ".env",
    }

    def setup_directories(self) -> None:
        """Create necessary directories."""
        for directory in [self.data_dir, self.models_dir, self.reports_dir]:
            self.base_dir.joinpath(directory).mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.base_dir.joinpath(self.data_dir, "raw").mkdir(parents=True, exist_ok=True)
        self.base_dir.joinpath(self.data_dir, "processed").mkdir(
            parents=True, exist_ok=True
        )
        self.base_dir.joinpath(self.models_dir, "evaluation").mkdir(
            parents=True, exist_ok=True
        )
        self.base_dir.joinpath(self.reports_dir, "figures").mkdir(
            parents=True, exist_ok=True
        )
        self.base_dir.joinpath(self.reports_dir, "results").mkdir(
            parents=True, exist_ok=True
        )


# Create a global config instance
config = ChurnSenseConfig()
