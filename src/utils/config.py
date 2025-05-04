"""
Configuration module for the ChurnSense project.
This module centralizes all configuration parameters used across the project.
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Base Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data Paths
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_churn_data.csv"

# Model paths
MODEL_PATH = MODELS_DIR
EVALUATION_PATH = MODELS_DIR / "evaluation"
EVALUATION_PATH.mkdir(parents=True, exist_ok=True)

# Report paths
FIGURES_PATH = REPORTS_DIR / "figures"
RESULTS_PATH = REPORTS_DIR / "results"
for path in [FIGURES_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


# Project configuration
@dataclass
class CONFIG:
    # General settings
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.25
    cv_folds: int = 5
    n_jobs: int = -1

    # Data settings
    data_path: str = str(RAW_DATA_PATH)
    processed_data_path: str = str(PROCESSED_DATA_PATH)

    # Model settings
    models_path: str = str(MODEL_PATH)
    evaluation_path: str = str(EVALUATION_PATH)

    # Report settings
    figures_path: str = str(FIGURES_PATH)
    results_path: str = str(RESULTS_PATH)

    # Target settings
    target_column: str = "Churn"
    positive_class: str = "Yes"
    id_column: str = "customerID"

    # Business metrics
    avg_customer_value: float = 1000.0
    retention_campaign_cost: float = 50.0
    retention_success_rate: float = 0.30
