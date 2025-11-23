# Continual Pretraining Experiments

Modular implementation of continual pretraining experiments for time series forecasting.

## Module Overview

### `config.py`
- `DEFAULT_CONFIG`: Default experiment configuration
- `parse_args()`: Command line argument parsing
- `load_config()`: Configuration file loading

### `utils.py`
- `clear_memory()`: GPU/CPU memory cleanup
- `print_memory_stats()`: Memory usage monitoring
- `save_checkpoint()`: Training checkpoint saving
- `safe_save_model()`: Safe model persistence
- `load_checkpoint()`: Checkpoint loading

### `datasets.py`
- `Dataset_Custom`: Custom dataset with temporal continuity validation
- `PretrainDataset`: Dataset for continual pretraining
- `MOMENTDatasetWrapper`: Adapter for MOMENT model format
- `load_manufacturing_data()`: Manufacturing data loading
- `create_moment_dataloader()`: DataLoader creation for MOMENT

### `trainer.py`
- `continual_pretrain()`: Continual pretraining with masking
- `train_forecasting()`: Forecasting head fine-tuning

### `evaluator.py`
- `evaluate_forecasting()`: Model evaluation with metrics

### `main.py`
- Main experiment execution
- Baseline vs Continual pretrained model comparison

## Usage

### Basic Execution
```bash
cd src/experiments
python main.py
```

### Custom Configuration
```bash
python main.py --config custom_config.json
```

## Configuration Example

```json
{
  "seed": 13,
  "data_dir": "path/to/data",
  "model_name": "AutonLab/MOMENT-1-base",
  "context_length": 512,
  "forecast_horizon": 6,
  "pretrain_epochs": 3,
  "pretrain_batch_size": 32,
  "pretrain_lr": 1e-4,
  "finetune_epochs": 3,
  "finetune_batch_size": 32,
  "finetune_lr": 1e-4,
  "freeze_encoder": true,
  "freeze_embedder": true,
  "output_dir": "results/experiments"
}
```

## Output

Results are saved to the configured output directory:
- `config.json`: Experiment configuration
- `metrics.json`: Evaluation metrics (MSE, MAE, RMSE)
- `metrics_comparison.png`: Metric comparison chart
- `sample_predictions.png`: Sample prediction visualizations
- Model checkpoints (*.pt)

## Architecture

The codebase follows a modular design:
1. **Configuration**: Centralized experiment parameters
2. **Data**: Custom datasets with validation
3. **Training**: Separate pretraining and fine-tuning pipelines
4. **Evaluation**: Comprehensive metrics and visualization
5. **Utilities**: Memory management and checkpointing

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
pip install momentfm
```

## Notes

- All modules use absolute imports for clarity
- GPU memory is actively managed to prevent OOM
- Checkpointing includes atomic writes for safety
- Data loading validates temporal continuity
