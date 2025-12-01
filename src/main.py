"""
Main entry point for continual pretraining experiments.

This script compares the performance of:
1. Baseline: MOMENT fine-tuned on SAMYANG_dataset only (no continual pretraining)
2. Continual Pretrained: MOMENT continual pretrained on manufacturing datasets,
   then fine-tuned on SAMYANG_dataset
"""

import json
import sys
from pathlib import Path

# Add project root to path for visualization imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch

from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness

from config import parse_args, load_config
from data import load_manufacturing_data, load_samyang_data, create_moment_dataloader
from training import continual_pretrain, train_forecasting, evaluate_forecasting
from utils import print_memory_stats, safe_save_model, clear_memory
from visualization import (
    create_all_cl_visualizations,
    compare_cl_metrics_across_experiments,
    create_all_experiment_comparisons,
    create_all_prediction_visualizations
)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Set seed
    control_randomness(seed=config['seed'])

    # Create output directory structure
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create organized subdirectories for each experiment
    for exp_name in ['soft_masking', 'sequential', 'baseline', 'comparison']:
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        if exp_name != 'comparison':
            (exp_dir / 'metrics').mkdir(exist_ok=True)
            (exp_dir / 'models').mkdir(exist_ok=True)
            (exp_dir / 'visualizations').mkdir(exist_ok=True)
        else:
            (exp_dir / 'metrics').mkdir(exist_ok=True)
            (exp_dir / 'visualizations').mkdir(exist_ok=True)

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

    # Store results for comparison
    all_results = {}

    # ========================================================================
    # Experiment 1: Soft-Masking CL (WITH Soft-Masking)
    # ========================================================================
    if config.get('run_soft_masking', True):
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: SOFT-MASKING CL (WITH Soft-Masking)")
        print("=" * 80)

        # Ensure soft-masking is enabled
        config['use_soft_masking'] = True

        # Load fresh MOMENT model for continual pretraining (reconstruction mode)
        softmask_pretrain_model = MOMENTPipeline.from_pretrained(
            config['model_name'],
            model_kwargs={
                'task_name': 'reconstruction',
                'freeze_encoder': False,
                'freeze_embedder': False,
            }
        )
        softmask_pretrain_model.init()
        softmask_pretrain_model = softmask_pretrain_model.to(device)

        print(f"Model loaded for soft-masking continual pretraining on {device}")

        # Continual pretrain on manufacturing datasets WITH soft-masking
        softmask_pretrain_model = continual_pretrain(
            softmask_pretrain_model,
            manufacturing_datasets,
            config,
            device,
            output_dir / "soft_masking"
        )

        # Save soft-masking pretrain model weights
        safe_save_model(
            softmask_pretrain_model,
            output_dir / "soft_masking" / "models" / "pretrained_weights.pt",
            "Soft-masking pretrained model"
        )

        softmask_model = MOMENTPipeline.from_pretrained(
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
        softmask_model.init()

        # Transfer pretrained weights
        pretrained_state = softmask_pretrain_model.state_dict()
        softmask_state = softmask_model.state_dict()

        # Transfer encoder and embedder weights
        for key in softmask_state.keys():
            if 'encoder' in key or 'embed' in key:
                if key in pretrained_state:
                    softmask_state[key] = pretrained_state[key]

        softmask_model.load_state_dict(softmask_state)
        softmask_model = softmask_model.to(device)

        # Clean up pretrain model
        del softmask_pretrain_model
        torch.cuda.empty_cache()

        print("Soft-masking pretrained weights transferred to forecasting model")

        # Fine-tune on SAMYANG
        softmask_model = train_forecasting(
            softmask_model,
            train_loader,
            val_loader,
            config,
            device,
            target_idx,
            output_dir,
            model_name="softmask"
        )

        # Evaluate soft-masking model
        softmask_metrics, softmask_preds, softmask_trues = evaluate_forecasting(
            softmask_model,
            test_loader,
            device,
            target_idx,
            y_scaler=test_dataset.y_scaler
        )

        print("\n" + "-" * 80)
        print("SOFT-MASKING CL TEST RESULTS:")
        print("-" * 80)
        for metric, value in softmask_metrics.items():
            print(f"  {metric}: {value:.6f}")

        # Save soft-masking model and results
        safe_save_model(
            softmask_model,
            output_dir / "soft_masking" / "models" / "finetuned_model.pt",
            "Soft-masking model"
        )
        all_results['soft_masking'] = softmask_metrics

        # Save metrics to JSON (convert numpy types to native Python types)
        softmask_metrics_native = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                   for k, v in softmask_metrics.items()}
        with open(output_dir / "soft_masking" / "metrics" / "test_metrics.json", 'w') as f:
            json.dump(softmask_metrics_native, f, indent=2)

        # Create prediction visualizations
        create_all_prediction_visualizations(
            softmask_preds,
            softmask_trues,
            output_dir / "soft_masking" / "visualizations",
            "Soft-Masking CL"
        )

        # Clean up
        del softmask_model
        torch.cuda.empty_cache()
    else:
        print("\nSkipping Soft-Masking CL experiment (run_soft_masking=False)")

    # ========================================================================
    # Experiment 2: Sequential CL (WITHOUT Soft-Masking)
    # ========================================================================
    if config.get('run_sequential', True):
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: SEQUENTIAL CL (WITHOUT Soft-Masking)")
        print("=" * 80)

        # Temporarily disable soft-masking
        original_use_soft_masking = config.get('use_soft_masking', True)
        config['use_soft_masking'] = False

        # Load fresh MOMENT model for continual pretraining (reconstruction mode)
        sequential_pretrain_model = MOMENTPipeline.from_pretrained(
            config['model_name'],
            model_kwargs={
                'task_name': 'reconstruction',
                'freeze_encoder': False,
                'freeze_embedder': False,
            }
        )
        sequential_pretrain_model.init()
        sequential_pretrain_model = sequential_pretrain_model.to(device)

        print(f"Model loaded for sequential continual pretraining on {device}")

        # Continual pretrain on manufacturing datasets WITHOUT soft-masking
        sequential_pretrain_model = continual_pretrain(
            sequential_pretrain_model,
            manufacturing_datasets,
            config,
            device,
            output_dir / "sequential"
        )

        # Save sequential pretrain model weights
        safe_save_model(
            sequential_pretrain_model,
            output_dir / "sequential" / "models" / "pretrained_weights.pt",
            "Sequential pretrained model"
        )

        sequential_model = MOMENTPipeline.from_pretrained(
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
        sequential_model.init()

        # Transfer pretrained weights
        pretrained_state = sequential_pretrain_model.state_dict()
        sequential_state = sequential_model.state_dict()

        # Transfer encoder and embedder weights
        for key in sequential_state.keys():
            if 'encoder' in key or 'embed' in key:
                if key in pretrained_state:
                    sequential_state[key] = pretrained_state[key]

        sequential_model.load_state_dict(sequential_state)
        sequential_model = sequential_model.to(device)

        # Clean up pretrain model
        del sequential_pretrain_model
        torch.cuda.empty_cache()

        print("Sequential pretrained weights transferred to forecasting model")

        # Fine-tune on SAMYANG
        sequential_model = train_forecasting(
            sequential_model,
            train_loader,
            val_loader,
            config,
            device,
            target_idx,
            output_dir,
            model_name="sequential"
        )

        # Evaluate sequential model
        sequential_metrics, sequential_preds, sequential_trues = evaluate_forecasting(
            sequential_model,
            test_loader,
            device,
            target_idx,
            y_scaler=test_dataset.y_scaler
        )

        print("\n" + "-" * 80)
        print("SEQUENTIAL CL TEST RESULTS:")
        print("-" * 80)
        for metric, value in sequential_metrics.items():
            print(f"  {metric}: {value:.6f}")

        # Save sequential model and results
        safe_save_model(
            sequential_model,
            output_dir / "sequential" / "models" / "finetuned_model.pt",
            "Sequential model"
        )
        all_results['sequential'] = sequential_metrics

        # Save metrics to JSON (convert numpy types to native Python types)
        sequential_metrics_native = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                     for k, v in sequential_metrics.items()}
        with open(output_dir / "sequential" / "metrics" / "test_metrics.json", 'w') as f:
            json.dump(sequential_metrics_native, f, indent=2)

        # Create prediction visualizations
        create_all_prediction_visualizations(
            sequential_preds,
            sequential_trues,
            output_dir / "sequential" / "visualizations",
            "Sequential CL"
        )

        # Clean up
        del sequential_model
        torch.cuda.empty_cache()

        # Restore original soft-masking setting
        config['use_soft_masking'] = original_use_soft_masking
    else:
        print("\nSkipping Sequential CL experiment (run_sequential=False)")

    # ========================================================================
    # Experiment 3: Baseline (No Continual Learning - Mixed Training)
    # ========================================================================
    if config.get('run_baseline', True):
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: BASELINE (No Continual Learning - Mixed Training)")
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

        # Save baseline model and results
        safe_save_model(
            baseline_model,
            output_dir / "baseline" / "models" / "finetuned_model.pt",
            "Baseline model"
        )
        all_results['baseline'] = baseline_metrics

        # Save metrics to JSON (convert numpy types to native Python types)
        baseline_metrics_native = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                   for k, v in baseline_metrics.items()}
        with open(output_dir / "baseline" / "metrics" / "test_metrics.json", 'w') as f:
            json.dump(baseline_metrics_native, f, indent=2)

        # Create prediction visualizations
        create_all_prediction_visualizations(
            baseline_preds,
            baseline_trues,
            output_dir / "baseline" / "visualizations",
            "Baseline"
        )

        # Clean up
        del baseline_model
        torch.cuda.empty_cache()
    else:
        print("\nSkipping Baseline experiment (run_baseline=False)")

    # ========================================================================
    # Comparison and Visualization
    # ========================================================================
    if len(all_results) > 0:
        print("\n" + "=" * 80)
        print("COMPARISON ACROSS ALL EXPERIMENTS")
        print("=" * 80)

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        # Print comparison table
        print("\n" + "-" * 80)
        print(f"{'Experiment':<25} {'MSE':<15} {'MAE':<15} {'RMSE':<15}")
        print("-" * 80)

        for exp_name, metrics in all_results.items():
            print(f"{exp_name.upper():<25} {metrics['MSE']:<15.6f} {metrics['MAE']:<15.6f} {metrics['RMSE']:<15.6f}")

        # Calculate improvements if baseline exists
        if 'baseline' in all_results:
            baseline_metrics = all_results['baseline']
            print("\n" + "-" * 80)
            print("IMPROVEMENT OVER BASELINE (%)")
            print("-" * 80)
            print(f"{'Experiment':<25} {'MSE':<15} {'MAE':<15} {'RMSE':<15}")
            print("-" * 80)

            for exp_name, metrics in all_results.items():
                if exp_name != 'baseline':
                    improvements = {
                        k: (baseline_metrics[k] - metrics[k]) / baseline_metrics[k] * 100
                        for k in ['MSE', 'MAE', 'RMSE']
                    }
                    print(f"{exp_name.upper():<25} {improvements['MSE']:<15.2f} {improvements['MAE']:<15.2f} {improvements['RMSE']:<15.2f}")

        # Save results to JSON
        comparison_native = convert_to_native(all_results)
        with open(output_dir / 'comparison' / 'metrics' / 'comparison_results.json', 'w') as f:
            json.dump(comparison_native, f, indent=2)

        print(f"\nComparison results saved to {output_dir / 'comparison' / 'metrics' / 'comparison_results.json'}")
    else:
        print("\nNo experiments were run. Enable experiments in config.")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)

    # Visualizations (only if we have results)
    if len(all_results) >= 2:
        # Metric comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        exp_names = list(all_results.keys())
        colors = ['steelblue', 'coral', 'green', 'orange'][:len(exp_names)]

        for idx, metric in enumerate(['MSE', 'MAE', 'RMSE']):
            ax = axes[idx]
            values = [all_results[name][metric] for name in exp_names]
            bars = ax.bar(exp_names, values, color=colors)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(exp_names, rotation=15, ha='right')

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "comparison" / "visualizations" / "metrics_comparison.png", dpi=150)
        print(f"\nMetrics comparison saved to {output_dir / 'comparison' / 'visualizations' / 'metrics_comparison.png'}")

    # ========================================================================
    # Generate Continual Learning Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING CONTINUAL LEARNING VISUALIZATIONS")
    print("=" * 80)

    # Generate CL visualizations for each experiment that used continual learning
    cl_experiment_dirs = {}

    if config.get('run_sequential', False):
        seq_dir = output_dir / "sequential"
        if (seq_dir / "metrics" / "continual_learning_metrics_matrix.csv").exists():
            cl_experiment_dirs['Sequential'] = str(seq_dir)
            print(f"\nGenerating visualizations for Sequential CL experiment...")
            create_all_cl_visualizations(str(seq_dir))

    if config.get('run_soft_masking', False):
        softmask_dir = output_dir / "soft_masking"
        if (softmask_dir / "metrics" / "continual_learning_metrics_matrix.csv").exists():
            cl_experiment_dirs['Soft-Masking'] = str(softmask_dir)
            print(f"\nGenerating visualizations for Soft-Masking CL experiment...")
            create_all_cl_visualizations(str(softmask_dir))

    # Compare CL metrics across experiments
    if len(cl_experiment_dirs) >= 2:
        print(f"\nComparing continual learning metrics across experiments...")
        compare_cl_metrics_across_experiments(
            cl_experiment_dirs,
            save_path=str(output_dir / "comparison" / "visualizations" / "cl_metrics_comparison.png")
        )

    if len(cl_experiment_dirs) > 0:
        print("\n" + "-" * 80)
        print("CONTINUAL LEARNING VISUALIZATIONS GENERATED:")
        print("-" * 80)
        print("For each CL experiment:")
        print("  - cl_performance_matrix.png: Heatmap showing forgetting patterns")
        print("  - cl_forgetting_evolution.png: Performance evolution and forgetting per domain")
        if len(cl_experiment_dirs) >= 2:
            print("\nCross-experiment comparison:")
            print("  - cl_metrics_comparison.png: Compare BWT, Forgetting, and Performance")
        print("-" * 80)
    else:
        print("\nNo continual learning experiments were run.")
        print("Enable run_sequential or run_soft_masking to see CL visualizations.")

    # ========================================================================
    # Generate Experiment Comparisons (MSE, Forgetting across models)
    # ========================================================================
    if len(cl_experiment_dirs) >= 2:
        print("\n" + "=" * 80)
        print("GENERATING CROSS-EXPERIMENT COMPARISONS")
        print("=" * 80)
        create_all_experiment_comparisons(str(output_dir))


if __name__ == "__main__":
    main()
