# MFM - Manufacturing Foundation Model

Time series forecasting with continual learning for manufacturing data.

## Project Structure

```
MFM/
├── data/                           # Datasets (not included in repo)
│
├── src/                           # Source code
│   └── experiments/               # Experiment modules
│       ├── __init__.py           # Package initialization
│       ├── config.py             # Configuration management
│       ├── utils.py              # Utility functions
│       ├── datasets.py           # Dataset loaders
│       ├── trainer.py            # Training functions
│       ├── evaluator.py          # Evaluation functions
│       ├── main.py               # Main execution script
│       └── README.md             # Experiment guide
│
└── results/                       # Experiment results (not included)
```

## Experiments

### Continual Pretraining

Foundation model continual pretraining on manufacturing datasets for improved forecasting performance.

**Run**:
```bash
cd src/experiments
python main.py
```

**Output**: Results saved to `results/continual_pretrain_results/`

## Features

- **Modular Architecture**: Clean separation of config, data, training, and evaluation
- **Continual Learning**: Domain sequential pretraining approach
- **Time Series**: Custom dataset with temporal continuity validation
- **Foundation Model**: Based on MOMENT architecture

## Technical Stack

- **Framework**: PyTorch
- **Model**: MOMENT (AutonLab/MOMENT-1-base)
- **Task**: Time Series Forecasting
- **Method**: Continual Pretraining + Fine-tuning

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
pip install momentfm
```

## Documentation

See [Experiment Guide](src/experiments/README.md) for detailed usage.

## License

Research and Educational Use
