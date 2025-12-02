"""
Utility to compare metrics across different experimental approaches.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def compare_domain_mse_across_experiments(experiment_dirs, save_path):
    """Compare MSE evolution across domains for different experiments.

    Args:
        experiment_dirs: Dict mapping experiment name to directory path
        save_path: Path to save the comparison visualization
    """
    all_matrices = {}

    # Load performance matrices from each experiment
    for exp_name, exp_dir in experiment_dirs.items():
        matrix_path = Path(exp_dir) / "metrics" / "continual_learning_metrics_matrix.csv"
        if matrix_path.exists():
            df = pd.read_csv(matrix_path, index_col=0)
            all_matrices[exp_name] = df

    if len(all_matrices) == 0:
        print("No performance matrix data found")
        return

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: MSE on each domain after learning that domain (diagonal)
    ax1 = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_matrices)))

    for idx, (exp_name, matrix) in enumerate(all_matrices.items()):
        n_domains = len(matrix.columns)
        # Extract diagonal (performance on domain i after learning domain i)
        diagonal_mse = [matrix.iloc[i, i] for i in range(n_domains)]
        domains = [f"D{i+1}" for i in range(n_domains)]

        ax1.plot(domains, diagonal_mse, marker='o', linewidth=2.5, markersize=10,
                label=exp_name, color=colors[idx], alpha=0.8)

    ax1.set_xlabel('Domain', fontsize=13, fontweight='bold')
    ax1.set_ylabel('MSE (Test Loss)', fontsize=13, fontweight='bold')
    ax1.set_title('MSE After Learning Each Domain\n(Lower is Better)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average MSE on all previous domains
    ax2 = axes[1]

    for idx, (exp_name, matrix) in enumerate(all_matrices.items()):
        n_domains = len(matrix.columns)
        avg_mse_list = []
        domains = []

        for train_idx in range(n_domains):
            # Average MSE on all domains seen so far
            mse_values = [matrix.iloc[train_idx, eval_idx]
                         for eval_idx in range(train_idx + 1)]
            avg_mse = np.mean(mse_values)
            avg_mse_list.append(avg_mse)
            domains.append(f"D{train_idx+1}")

        ax2.plot(domains, avg_mse_list, marker='s', linewidth=2.5, markersize=10,
                label=exp_name, color=colors[idx], alpha=0.8)

    ax2.set_xlabel('Training Stage', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average MSE', fontsize=13, fontweight='bold')
    ax2.set_title('Average MSE on All Learned Domains\n(Lower is Better)',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Domain Learning Performance Comparison',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved MSE comparison to {save_path}")


def compare_forgetting_across_experiments(experiment_dirs, save_path):
    """Compare forgetting metrics across different experiments.

    Args:
        experiment_dirs: Dict mapping experiment name to directory path
        save_path: Path to save the comparison visualization
    """
    all_metrics = {}
    all_matrices = {}

    # Load metrics from each experiment
    for exp_name, exp_dir in experiment_dirs.items():
        metrics_path = Path(exp_dir) / "metrics" / "continual_learning_metrics.csv"
        matrix_path = Path(exp_dir) / "metrics" / "continual_learning_metrics_matrix.csv"

        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            all_metrics[exp_name] = df.iloc[0].to_dict()

        if matrix_path.exists():
            matrix_df = pd.read_csv(matrix_path, index_col=0)
            all_matrices[exp_name] = matrix_df

    if len(all_metrics) == 0:
        print("No metrics data found")
        return

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Backward Transfer (BWT)
    ax1 = axes[0, 0]
    exp_names = list(all_metrics.keys())
    bwt_values = [all_metrics[exp]['backward_transfer'] for exp in exp_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(exp_names)))

    bars = ax1.bar(range(len(exp_names)), bwt_values, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No Forgetting')
    ax1.set_ylabel('BWT (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Backward Transfer (BWT)\nNegative = Catastrophic Forgetting',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels(exp_names, rotation=15, ha='right')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend()

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
    ax2.set_ylabel('Forgetting (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Forgetting', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(exp_names)))
    ax2.set_xticklabels(exp_names, rotation=15, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, forgetting_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 3: Forgetting per domain
    ax3 = axes[1, 0]

    for idx, (exp_name, matrix) in enumerate(all_matrices.items()):
        n_domains = len(matrix.columns)
        forgetting_per_domain = []

        for domain_idx in range(n_domains - 1):  # Exclude last domain (no forgetting yet)
            # Max performance on this domain
            max_perf = matrix.iloc[domain_idx, domain_idx]
            # Final performance on this domain
            final_perf = matrix.iloc[-1, domain_idx]
            # Forgetting = max - final
            forgetting = max_perf - final_perf
            forgetting_per_domain.append(forgetting)

        domains = [f"D{i+1}" for i in range(n_domains - 1)]
        ax3.plot(domains, forgetting_per_domain, marker='o', linewidth=2.5,
                markersize=10, label=exp_name, color=colors[idx], alpha=0.8)

    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Forgetting (MSE Increase)', fontsize=12, fontweight='bold')
    ax3.set_title('Forgetting Per Domain\n(Lower is Better)',
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance retention (%)
    ax4 = axes[1, 1]

    for idx, (exp_name, matrix) in enumerate(all_matrices.items()):
        n_domains = len(matrix.columns)
        retention_rates = []

        for domain_idx in range(n_domains - 1):
            max_perf = matrix.iloc[domain_idx, domain_idx]
            final_perf = matrix.iloc[-1, domain_idx]
            # Retention rate (higher is better)
            retention = (1 - (max_perf - final_perf) / max_perf) * 100
            retention_rates.append(retention)

        domains = [f"D{i+1}" for i in range(n_domains - 1)]
        ax4.plot(domains, retention_rates, marker='s', linewidth=2.5,
                markersize=10, label=exp_name, color=colors[idx], alpha=0.8)

    ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label='Perfect Retention')
    ax4.set_xlabel('Domain', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Performance Retention (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Retention Rate\n(Higher is Better)',
                  fontsize=13, fontweight='bold')
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    plt.suptitle('Forgetting Analysis Across Experiments',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved forgetting comparison to {save_path}")


def create_all_experiment_comparisons(output_dir):
    """Create all comparison visualizations for experiments.

    Args:
        output_dir: Base output directory (simplified structure)
    """
    output_path = Path(output_dir)
    metrics_dir = output_path / "metrics"
    vis_dir = output_path / "visualizations"

    # Check if CL metrics exist
    cl_matrix_path = metrics_dir / "continual_learning_metrics_matrix.csv"
    if not cl_matrix_path.exists():
        print("No continual learning metrics found for comparison")
        return

    vis_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("CREATING EXPERIMENT COMPARISONS")
    print("=" * 80)

    # For simplified structure, use single experiment dir
    experiment_dirs = {'Experiment': str(output_path)}

    # Generate MSE comparison
    print("\nGenerating domain MSE comparison...")
    compare_domain_mse_across_experiments(
        experiment_dirs,
        str(vis_dir / "domain_mse_comparison.png")
    )

    # Generate forgetting comparison
    print("\nGenerating forgetting comparison...")
    compare_forgetting_across_experiments(
        experiment_dirs,
        str(vis_dir / "forgetting_comparison.png")
    )

    print("\n" + "-" * 80)
    print(f"All comparison visualizations saved to {vis_dir}")
    print("-" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compare_experiments.py <output_dir>")
        print("Example: python compare_experiments.py /path/to/results/continual_pretrain_results")
        sys.exit(1)

    output_dir = sys.argv[1]
    create_all_experiment_comparisons(output_dir)
