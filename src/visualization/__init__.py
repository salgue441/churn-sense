"""
Visualization package for the ChurnSense project.
This package includes modules for data visualization, model visualization, and interactive plotting.
"""

from src.visualization.data_viz import (
    plot_churn_distribution,
    plot_categorical_feature,
    plot_numerical_feature,
    plot_correlation_matrix,
    plot_customer_segments,
)
from src.visualization.model_viz import (
    plot_model_performance,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_calibration_curve,
)
from src.visualization.interactive_viz import (
    create_churn_dashboard,
    create_feature_explorer,
    create_model_comparison_plot,
    create_risk_distribution_plot,
)

__all__ = [
    # Data visualization
    "plot_churn_distribution",
    "plot_categorical_feature",
    "plot_numerical_feature",
    "plot_correlation_matrix",
    "plot_customer_segments",
    # Model visualization
    "plot_model_performance",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    # Interactive visualization
    "create_churn_dashboard",
    "create_feature_explorer",
    "create_model_comparison_plot",
    "create_risk_distribution_plot",
]
