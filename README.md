# MFM - Manufacturing Foundation Model

Time series forecasting with continual learning for manufacturing data.

## Project Structure

```
MFM/
├── data/                           # Datasets (not included in repo)
│
├── src/                            # Source code
│   ├── config/                     # Configuration management
│   ├── data/                       # Data loading utilities
│   ├── training/                   # Training & evaluation
│   ├── utils/                      # Utility functions
│   └── main.py                     # Main entry point
│
├── visualization/                  # Visualization scripts
│   ├── visualize_cl_metrics.py     # Continual learning metrics
│   ├── visualize_predictions.py    # Prediction plots
│   ├── visualize_soft_masking.py   # Soft-masking analysis
│   └── compare_experiments.py      # Cross-experiment comparison
│
└── results/                        # Experiment results (not included)
```

## Experiments

### Continual Pretraining

Foundation model continual pretraining on manufacturing datasets for improved forecasting performance.

**Run**:
```bash
cd src
python main.py
```

**Output**: Results saved to `results/continual_pretrain_results/`

## Features

- **Modular Architecture**: Clean separation of config, data, training, and evaluation
- **Continual Learning**: Soft-masking based approach to prevent catastrophic forgetting
- **Time Series**: Custom dataset with temporal continuity validation
- **Foundation Model**: Based on MOMENT architecture
- **Visualization**: CL metrics, predictions, and experiment comparison tools

## Technical Stack

- **Framework**: PyTorch
- **Model**: MOMENT (AutonLab/MOMENT-1-base)
- **Task**: Time Series Forecasting
- **Method**: Continual Pretraining + Fine-tuning

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
pip install momentfm
```

## License

Research and Educational Use
