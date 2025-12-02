"""Visualization utilities for MFM project."""

from .visualize_cl_metrics import (
    plot_performance_matrix,
    plot_forgetting_evolution,
    compare_cl_metrics_across_experiments,
    create_all_cl_visualizations
)

from .visualize_predictions import (
    plot_prediction_vs_actual,
    create_all_prediction_visualizations
)

from .compare_experiments import (
    compare_domain_mse_across_experiments,
    compare_forgetting_across_experiments,
    create_all_experiment_comparisons
)

__all__ = [
    # CL metrics visualization
    'plot_performance_matrix',
    'plot_forgetting_evolution',
    'compare_cl_metrics_across_experiments',
    'create_all_cl_visualizations',
    # Prediction visualization
    'plot_prediction_vs_actual',
    'create_all_prediction_visualizations',
    # Experiment comparison
    'compare_domain_mse_across_experiments',
    'compare_forgetting_across_experiments',
    'create_all_experiment_comparisons',
]
