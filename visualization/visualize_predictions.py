"""
Visualization functions for prediction vs actual comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_prediction_vs_actual(all_preds, all_trues, save_path, domain_name,
                               n_samples=5, seed=42):
    """Plot prediction vs actual time series for randomly sampled examples.

    Args:
        all_preds: Predictions array [batch, channels, time_steps]
        all_trues: Ground truth array [batch, channels, time_steps]
        save_path: Path to save the visualization
        domain_name: Name of the domain for the title
        n_samples: Number of random samples to plot
        seed: Random seed for reproducibility
    """
    if all_preds is None or all_trues is None:
        print(f"Cannot create prediction plot for {domain_name}: predictions or ground truth is None")
        return

    # Debug: Print shapes
    print(f"  all_preds shape: {all_preds.shape}")
    print(f"  all_trues shape: {all_trues.shape}")

    # Handle different possible shapes
    # Expected: [batch, channels, time_steps] or [batch, time_steps, channels]
    if all_preds.shape[1] < all_preds.shape[2]:
        # Shape is [batch, channels, time_steps]
        pass
    else:
        # Shape is [batch, time_steps, channels] - need to transpose
        all_preds = np.transpose(all_preds, (0, 2, 1))
        all_trues = np.transpose(all_trues, (0, 2, 1))
        print(f"  Transposed to: {all_preds.shape}")

    # Set random seed
    np.random.seed(seed)

    batch_size = all_preds.shape[0]
    n_channels = all_preds.shape[1]
    forecast_horizon = all_preds.shape[2]

    n_samples = min(n_samples, batch_size)

    # Randomly select samples
    random_indices = np.random.choice(batch_size, size=n_samples, replace=False)

    # Create subplots
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    time_steps = np.arange(forecast_horizon)

    for i, idx in enumerate(random_indices):
        # Extract single channel (use channel 0)
        pred = all_preds[idx, 0, :]  # [forecast_horizon]
        true = all_trues[idx, 0, :]  # [forecast_horizon]

        axes[i].plot(time_steps, true, label='Actual', color='blue', linewidth=2, alpha=0.7)
        axes[i].plot(time_steps, pred, label='Prediction', color='red', linewidth=2,
                    linestyle='--', alpha=0.7)

        # Calculate error for this sample
        mse = np.mean((true - pred) ** 2)
        mae = np.mean(np.abs(true - pred))

        axes[i].set_title(f'Sample {idx} - MSE: {mse:.4f}, MAE: {mae:.4f}',
                         fontweight='bold', fontsize=11)
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'Prediction vs Actual - {domain_name}',
                fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction vs actual plot to {save_path}")


def plot_scatter_prediction_vs_actual(all_preds, all_trues, save_path, domain_name,
                                     max_points=5000, seed=42):
    """Create scatter plot of predictions vs actual values.

    Args:
        all_preds: Predictions array [batch, channels, time_steps] or [batch, time_steps, channels]
        all_trues: Ground truth array [batch, channels, time_steps] or [batch, time_steps, channels]
        save_path: Path to save the visualization
        domain_name: Name of the domain for the title
        max_points: Maximum number of points to plot (for performance)
        seed: Random seed for reproducibility
    """
    if all_preds is None or all_trues is None:
        print(f"Cannot create scatter plot for {domain_name}: predictions or ground truth is None")
        return

    # Handle different possible shapes
    if all_preds.shape[1] > all_preds.shape[2]:
        # Shape is [batch, time_steps, channels] - need to transpose
        all_preds = np.transpose(all_preds, (0, 2, 1))
        all_trues = np.transpose(all_trues, (0, 2, 1))

    # Flatten arrays
    preds_flat = all_preds.flatten()
    trues_flat = all_trues.flatten()

    # Subsample if too many points
    n_points = len(preds_flat)
    if n_points > max_points:
        np.random.seed(seed)
        indices = np.random.choice(n_points, size=max_points, replace=False)
        preds_flat = preds_flat[indices]
        trues_flat = trues_flat[indices]

    # Calculate metrics
    mse = np.mean((trues_flat - preds_flat) ** 2)
    mae = np.mean(np.abs(trues_flat - preds_flat))
    rmse = np.sqrt(mse)

    # Calculate R-squared
    ss_res = np.sum((trues_flat - preds_flat) ** 2)
    ss_tot = np.sum((trues_flat - np.mean(trues_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot scatter with transparency
    ax.scatter(trues_flat, preds_flat, alpha=0.3, s=10, color='steelblue',
              edgecolors='none', label='Predictions')

    # Plot perfect prediction line
    min_val = min(trues_flat.min(), preds_flat.min())
    max_val = max(trues_flat.max(), preds_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
           label='Perfect Prediction', alpha=0.7)

    # Add metrics text box
    textstr = f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, family='monospace')

    ax.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
    ax.set_title(f'Prediction vs Actual Scatter - {domain_name}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {save_path}")


def plot_error_distribution(all_preds, all_trues, save_path, domain_name):
    """Plot distribution of prediction errors.

    Args:
        all_preds: Predictions array [batch, channels, time_steps] or [batch, time_steps, channels]
        all_trues: Ground truth array [batch, channels, time_steps] or [batch, time_steps, channels]
        save_path: Path to save the visualization
        domain_name: Name of the domain for the title
    """
    if all_preds is None or all_trues is None:
        print(f"Cannot create error distribution for {domain_name}: predictions or ground truth is None")
        return

    # Handle different possible shapes
    if all_preds.shape[1] > all_preds.shape[2]:
        # Shape is [batch, time_steps, channels] - need to transpose
        all_preds = np.transpose(all_preds, (0, 2, 1))
        all_trues = np.transpose(all_trues, (0, 2, 1))

    # Calculate errors
    errors = (all_trues - all_preds).flatten()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of errors
    axes[0].hist(errors, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (Actual - Predicted)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    textstr = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nMedian: {median_error:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    axes[0].text(0.72, 0.97, textstr, transform=axes[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')

    # Box plot of absolute errors
    abs_errors = np.abs(errors)
    axes[1].boxplot(abs_errors, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
    axes[1].set_ylabel('Absolute Prediction Error', fontsize=11, fontweight='bold')
    axes[1].set_title('Absolute Error Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticklabels(['All Predictions'])

    # Add statistics for absolute errors
    q1, q2, q3 = np.percentile(abs_errors, [25, 50, 75])
    iqr = q3 - q1
    textstr = f'Q1: {q1:.4f}\nMedian: {q2:.4f}\nQ3: {q3:.4f}\nIQR: {iqr:.4f}'
    axes[1].text(0.65, 0.97, textstr, transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')

    plt.suptitle(f'Prediction Error Analysis - {domain_name}',
                fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error distribution plot to {save_path}")


def create_all_prediction_visualizations(all_preds, all_trues, output_dir, domain_name):
    """Create all prediction-related visualizations.

    Args:
        all_preds: Predictions array [batch, channels, forecast_horizon]
        all_trues: Ground truth array [batch, channels, forecast_horizon]
        output_dir: Directory to save visualizations
        domain_name: Name of the domain
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating prediction visualizations for {domain_name}...")

    # 1. Time series comparison plot
    plot_prediction_vs_actual(
        all_preds, all_trues,
        output_path / f'pred_vs_actual_timeseries_{domain_name}.png',
        domain_name,
        n_samples=5
    )

    # 2. Scatter plot
    plot_scatter_prediction_vs_actual(
        all_preds, all_trues,
        output_path / f'pred_vs_actual_scatter_{domain_name}.png',
        domain_name
    )

    # 3. Error distribution
    plot_error_distribution(
        all_preds, all_trues,
        output_path / f'error_distribution_{domain_name}.png',
        domain_name
    )

    print(f"All prediction visualizations saved to {output_path}")
