"""
Visualization utilities for continual learning metrics.

Creates comprehensive visualizations of:
- Performance matrix (forgetting across domains)
- BWT and Forgetting comparison across experiments
- Domain-wise performance evolution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def plot_performance_matrix(matrix_csv_path: str, save_path: str = None):
    """Plot performance matrix as heatmap showing forgetting patterns.

    Args:
        matrix_csv_path: Path to performance matrix CSV
        save_path: Path to save the plot (optional)
    """
    # Load matrix
    df = pd.read_csv(matrix_csv_path, index_col=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mask for upper triangle (future domains not yet trained)
    mask = np.triu(np.ones_like(df.values, dtype=bool), k=1)

    # Plot heatmap
    sns.heatmap(
        df.values,
        mask=mask,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn_r',  # Red=high loss (bad), Green=low loss (good)
        cbar_kws={'label': 'MSE Loss'},
        xticklabels=df.columns,
        yticklabels=df.index,
        ax=ax,
        linewidths=1,
        linecolor='gray'
    )

    ax.set_title('Performance Matrix: MSE Loss Across Domains\n(Row = After training on, Column = Evaluated on)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Evaluated on Domain', fontsize=12, fontweight='bold')
    ax.set_ylabel('After Training on Domain', fontsize=12, fontweight='bold')

    # Add annotations for diagonal and forgetting
    n_domains = len(df.columns)
    for i in range(n_domains):
        # Highlight diagonal (immediate performance)
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

        # Show forgetting for previous domains
        if i > 0:
            for j in range(i):
                initial_perf = df.iloc[j, j]  # Performance when first trained
                current_perf = df.iloc[i, j]  # Performance after training on more domains
                forgetting = current_perf - initial_perf

                if forgetting > 0.01:  # Significant forgetting
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2, linestyle='--'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Performance matrix heatmap saved to {save_path}")

    return fig


def plot_forgetting_evolution(matrix_csv_path: str, save_path: str = None):
    """Plot how performance on each domain evolves over training stages.

    Args:
        matrix_csv_path: Path to performance matrix CSV
        save_path: Path to save the plot (optional)
    """
    # Load matrix
    df = pd.read_csv(matrix_csv_path, index_col=0)

    n_domains = len(df.columns)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Performance evolution per domain
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, n_domains))

    for col_idx, domain in enumerate(df.columns):
        performances = []
        training_stages = []

        for row_idx in range(col_idx, n_domains):
            perf = df.iloc[row_idx, col_idx]
            if not np.isnan(perf):
                performances.append(perf)
                training_stages.append(row_idx)

        ax1.plot(training_stages, performances,
                marker='o', linewidth=2, markersize=8,
                label=domain, color=colors[col_idx])

        # Mark initial performance
        if len(performances) > 0:
            ax1.scatter(training_stages[0], performances[0],
                       s=200, marker='*', color=colors[col_idx],
                       edgecolors='black', linewidths=2, zorder=5)

    ax1.set_xlabel('Training Stage (after domain)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Evolution per Domain\n(★ = Initial performance)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(n_domains))
    ax1.set_xticklabels([f"After\n{df.columns[i]}" for i in range(n_domains)], rotation=0)

    # Right plot: Forgetting per domain
    ax2 = axes[1]

    forgetting_per_domain = []
    domain_names = []

    for col_idx in range(n_domains - 1):  # Exclude last domain (can't forget yet)
        domain = df.columns[col_idx]
        initial_perf = df.iloc[col_idx, col_idx]  # Performance when first trained
        final_perf = df.iloc[n_domains - 1, col_idx]  # Final performance

        if not np.isnan(initial_perf) and not np.isnan(final_perf):
            forgetting = final_perf - initial_perf
            forgetting_per_domain.append(forgetting)
            domain_names.append(domain)

    bars = ax2.bar(range(len(domain_names)), forgetting_per_domain,
                   color=['red' if f > 0 else 'green' for f in forgetting_per_domain],
                   alpha=0.7, edgecolor='black', linewidth=2)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Forgetting (Final MSE - Initial MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Forgetting per Domain\n(Positive = Forgetting, Negative = Improvement)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(domain_names)))
    ax2.set_xticklabels(domain_names, rotation=15, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for idx, (bar, val) in enumerate(zip(bars, forgetting_per_domain)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Forgetting evolution plot saved to {save_path}")

    return fig


def compare_cl_metrics_across_experiments(experiment_dirs: Dict[str, str],
                                          save_path: str = None):
    """Compare continual learning metrics across different experiments.

    Args:
        experiment_dirs: Dict mapping experiment name to directory path
        save_path: Path to save the plot (optional)
    """
    # Collect metrics from all experiments
    all_metrics = {}

    for exp_name, exp_dir in experiment_dirs.items():
        metrics_path = Path(exp_dir) / "metrics" / "continual_learning_metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            all_metrics[exp_name] = df.iloc[0].to_dict()

    if not all_metrics:
        print("No continual learning metrics found in any experiment directory")
        return None

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    exp_names = list(all_metrics.keys())
    colors = plt.cm.Set1(np.linspace(0, 1, len(exp_names)))

    # Plot 1: Backward Transfer (BWT)
    ax1 = axes[0, 0]
    bwt_values = [all_metrics[exp]['backward_transfer'] for exp in exp_names]
    bars = ax1.bar(range(len(exp_names)), bwt_values, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('BWT (higher is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Backward Transfer (BWT)\nNegative = Catastrophic Forgetting',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels(exp_names, rotation=15, ha='right')
    ax1.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, bwt_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Plot 2: Forgetting
    ax2 = axes[0, 1]
    forgetting_values = [all_metrics[exp]['forgetting'] for exp in exp_names]
    bars = ax2.bar(range(len(exp_names)), forgetting_values, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Forgetting (lower is better)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Forgetting\nPerformance Degradation on Previous Domains',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(exp_names)))
    ax2.set_xticklabels(exp_names, rotation=15, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, forgetting_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 3: Average Performance
    ax3 = axes[1, 0]
    avg_perf_values = [all_metrics[exp]['average_performance'] for exp in exp_names]
    bars = ax3.bar(range(len(exp_names)), avg_perf_values, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Average MSE (lower is better)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Performance Across All Domains\nOverall Continual Learning Performance',
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(exp_names)))
    ax3.set_xticklabels(exp_names, rotation=15, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, avg_perf_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 4: Summary comparison (normalized scores)
    ax4 = axes[1, 1]

    # Normalize metrics to [0, 1] for comparison
    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    # For BWT: higher is better, so normalize as-is
    bwt_norm = normalize(bwt_values)
    # For forgetting: lower is better, so invert
    forgetting_norm = [1 - v for v in normalize(forgetting_values)]
    # For avg_perf: lower is better, so invert
    avg_perf_norm = [1 - v for v in normalize(avg_perf_values)]

    x = np.arange(len(exp_names))
    width = 0.25

    ax4.bar(x - width, bwt_norm, width, label='BWT (↑)', color='steelblue', alpha=0.7, edgecolor='black')
    ax4.bar(x, forgetting_norm, width, label='Anti-Forgetting (↑)', color='coral', alpha=0.7, edgecolor='black')
    ax4.bar(x + width, avg_perf_norm, width, label='Performance (↑)', color='green', alpha=0.7, edgecolor='black')

    ax4.set_ylabel('Normalized Score (higher is better)', fontsize=12, fontweight='bold')
    ax4.set_title('Normalized Metrics Comparison\n(All metrics scaled to 0-1, higher is better)',
                  fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(exp_names, rotation=15, ha='right')
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.set_ylim([0, 1.1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"CL metrics comparison saved to {save_path}")

    return fig


def create_all_cl_visualizations(output_dir: str):
    """Generate all continual learning visualizations for a single experiment.

    Args:
        output_dir: Path to experiment output directory
    """
    output_path = Path(output_dir)
    matrix_path = output_path / "metrics" / "continual_learning_metrics_matrix.csv"

    if not matrix_path.exists():
        print(f"No continual learning metrics found in {output_dir}/metrics")
        print("Make sure compute_cl_metrics is enabled in config.")
        return

    print(f"\nGenerating continual learning visualizations for {output_dir}...")

    # Create visualizations directory
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    plot_performance_matrix(
        str(matrix_path),
        str(vis_dir / "cl_performance_matrix.png")
    )

    plot_forgetting_evolution(
        str(matrix_path),
        str(vis_dir / "cl_forgetting_evolution.png")
    )

    print(f"\nAll visualizations saved to {vis_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_cl_metrics.py <output_dir>")
        print("   or: python visualize_cl_metrics.py --compare <exp1_dir> <exp2_dir> ...")
        sys.exit(1)

    if sys.argv[1] == "--compare":
        # Compare multiple experiments
        experiment_dirs = {}
        for exp_dir in sys.argv[2:]:
            exp_name = Path(exp_dir).name
            experiment_dirs[exp_name] = exp_dir

        compare_cl_metrics_across_experiments(
            experiment_dirs,
            save_path="cl_metrics_comparison.png"
        )
    else:
        # Single experiment visualization
        create_all_cl_visualizations(sys.argv[1])
