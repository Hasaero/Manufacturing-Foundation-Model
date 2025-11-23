"""
Main entry point for continual pretraining experiments.

This script compares the performance of:
1. Baseline: MOMENT fine-tuned on SAMYANG_dataset only (no continual pretraining)
2. Continual Pretrained: MOMENT continual pretrained on manufacturing datasets,
   then fine-tuned on SAMYANG_dataset
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness

from config import parse_args, load_config
from datasets import load_manufacturing_data, load_samyang_data, create_moment_dataloader
from trainer import continual_pretrain, train_forecasting
from evaluator import evaluate_forecasting
from utils import print_memory_stats, safe_save_model, clear_memory


def main():
    args = parse_args()
    config = load_config(args.config)

    # Set seed
    control_randomness(seed=config['seed'])

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print initial memory stats
    if torch.cuda.is_available():
        print_memory_stats("System ")

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load manufacturing datasets for continual pretraining
    manufacturing_datasets = load_manufacturing_data(
        config['data_dir'],
        config['pretrain_files']
    )

    # Load SAMYANG dataset (returns DataFrame)
    samyang_df, _ = load_samyang_data(
        config['data_dir'],
        config['samyang_file'],
        config['target_column']
    )

    # Create SAMYANG datasets using Dataset_Custom (with continuous sequence validation)
    print("\nCreating datasets with continuous sequence validation (using Dataset_Custom)...")

    train_loader, train_dataset, target_idx = create_moment_dataloader(
        samyang_df,
        flag='train',
        config=config,
        shuffle=True,
        drop_last=True
    )

    val_loader, val_dataset, _ = create_moment_dataloader(
        samyang_df,
        flag='val',
        config=config,
        shuffle=False,
        drop_last=False
    )

    test_loader, test_dataset, _ = create_moment_dataloader(
        samyang_df,
        flag='test',
        config=config,
        shuffle=False,
        drop_last=False
    )

    # ========================================================================
    # Experiment 1: Baseline (SAMYANG only, no continual pretraining)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: BASELINE (No Continual Pretraining)")
    print("=" * 80)

    # Load fresh MOMENT model for forecasting
    baseline_model = MOMENTPipeline.from_pretrained(
        config['model_name'],
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': config['forecast_horizon'],
            'head_dropout': config['head_dropout'],
            'weight_decay': config['weight_decay'],
            'freeze_encoder': config['freeze_encoder'],
            'freeze_embedder': config['freeze_embedder'],
            'freeze_head': config['freeze_head'],
        }
    )
    baseline_model.init()
    baseline_model = baseline_model.to(device)

    print(f"Baseline model loaded on {device}")

    # Fine-tune on SAMYANG
    baseline_model = train_forecasting(
        baseline_model,
        train_loader,
        val_loader,
        config,
        device,
        target_idx,
        output_dir,
        model_name="baseline"
    )

    # Evaluate baseline
    baseline_metrics, baseline_preds, baseline_trues = evaluate_forecasting(
        baseline_model,
        test_loader,
        device,
        target_idx,
        y_scaler=test_dataset.y_scaler
    )

    print("\n" + "-" * 80)
    print("BASELINE TEST RESULTS:")
    print("-" * 80)
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Save baseline model
    safe_save_model(baseline_model, output_dir / "baseline_model.pt", "Baseline model")

    # Clean up
    del baseline_model
    torch.cuda.empty_cache()

    # ========================================================================
    # Experiment 2: Continual Pretrained Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: CONTINUAL PRETRAINED MODEL")
    print("=" * 80)

    # Load fresh MOMENT model for continual pretraining (reconstruction mode)
    pretrain_model = MOMENTPipeline.from_pretrained(
        config['model_name'],
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False,
            'freeze_embedder': False,
        }
    )
    pretrain_model.init()
    pretrain_model = pretrain_model.to(device)

    print(f"Model loaded for continual pretraining on {device}")

    # Continual pretrain on manufacturing datasets
    pretrain_model = continual_pretrain(
        pretrain_model,
        manufacturing_datasets,
        config,
        device,
        output_dir
    )

    # Save continual pretrained weights
    safe_save_model(pretrain_model, output_dir / "continual_pretrained_weights.pt", "Continual pretrained model")

    # Convert to forecasting mode
    continual_model = MOMENTPipeline.from_pretrained(
        config['model_name'],
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': config['forecast_horizon'],
            'head_dropout': config['head_dropout'],
            'weight_decay': config['weight_decay'],
            'freeze_encoder': config['freeze_encoder'],
            'freeze_embedder': config['freeze_embedder'],
            'freeze_head': config['freeze_head'],
        }
    )
    continual_model.init()

    # Transfer pretrained weights
    pretrained_state = pretrain_model.state_dict()
    continual_state = continual_model.state_dict()

    # Transfer encoder and embedder weights
    for key in continual_state.keys():
        if 'encoder' in key or 'embed' in key:
            if key in pretrained_state:
                continual_state[key] = pretrained_state[key]

    continual_model.load_state_dict(continual_state)
    continual_model = continual_model.to(device)

    # Clean up pretrain model
    del pretrain_model
    torch.cuda.empty_cache()

    print("Continual pretrained weights transferred to forecasting model")

    # Fine-tune on SAMYANG
    continual_model = train_forecasting(
        continual_model,
        train_loader,
        val_loader,
        config,
        device,
        target_idx,
        output_dir,
        model_name="continual"
    )

    # Evaluate continual model
    continual_metrics, continual_preds, continual_trues = evaluate_forecasting(
        continual_model,
        test_loader,
        device,
        target_idx,
        y_scaler=test_dataset.y_scaler
    )

    print("\n" + "-" * 80)
    print("CONTINUAL PRETRAINED TEST RESULTS:")
    print("-" * 80)
    for metric, value in continual_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Save continual model
    safe_save_model(continual_model, output_dir / "continual_model.pt", "Continual model")

    # ========================================================================
    # Comparison and Visualization
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: BASELINE vs CONTINUAL PRETRAINED")
    print("=" * 80)

    comparison = {
        'Baseline': baseline_metrics,
        'Continual': continual_metrics,
        'Improvement (%)': {
            k: (baseline_metrics[k] - continual_metrics[k]) / baseline_metrics[k] * 100
            for k in baseline_metrics.keys()
        }
    }

    for model_name, metrics in comparison.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if model_name == 'Improvement (%)':
                print(f"  {metric}: {value:+.2f}%")
            else:
                print(f"  {metric}: {value:.6f}")

    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    # Visualizations
    if baseline_preds is not None and continual_preds is not None:
        # Metric comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, metric in enumerate(['MSE', 'MAE', 'RMSE']):
            ax = axes[idx]
            values = [baseline_metrics[metric], continual_metrics[metric]]
            bars = ax.bar(['Baseline', 'Continual\nPretrained'], values, color=['steelblue', 'coral'])
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
        print(f"\nMetrics comparison saved to {output_dir / 'metrics_comparison.png'}")

        # Sample predictions plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        n_samples = min(6, len(baseline_trues))
        sample_indices = np.random.choice(len(baseline_trues), size=n_samples, replace=False)

        for plot_idx, idx in enumerate(sample_indices):
            ax = axes[plot_idx]

            true_values = baseline_trues[idx, 0, :]
            baseline_pred = baseline_preds[idx, 0, :]
            continual_pred = continual_preds[idx, 0, :]

            timesteps = np.arange(len(true_values))

            ax.plot(timesteps, true_values, color='black', linewidth=2,
                   marker='o', markersize=4, label='Actual', alpha=0.8)
            ax.plot(timesteps, baseline_pred, color='steelblue', linewidth=2,
                   marker='s', markersize=3, label='Baseline', alpha=0.7, linestyle='--')
            ax.plot(timesteps, continual_pred, color='coral', linewidth=2,
                   marker='^', markersize=3, label='Continual Pretrained', alpha=0.7, linestyle='--')

            baseline_mae = np.mean(np.abs(true_values - baseline_pred))
            continual_mae = np.mean(np.abs(true_values - continual_pred))

            ax.set_title(f'Sample {idx}\nBaseline MAE: {baseline_mae:.4f}, Continual MAE: {continual_mae:.4f}',
                        fontsize=10)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "sample_predictions.png", dpi=150)
        print(f"Sample predictions saved to {output_dir / 'sample_predictions.png'}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
