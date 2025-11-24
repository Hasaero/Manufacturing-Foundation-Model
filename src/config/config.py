"""
Configuration module for continual pretraining experiments.
"""

import argparse
import json
from typing import Dict, Any


DEFAULT_CONFIG = {
    "seed": 13,
    "data_dir": "/home/juyoung_ha/MFM/data",
    "samyang_file": "SAMYANG_dataset.csv",
    "pretrain_files": ["ai4i2020.csv", "IoT.csv", "Steel_industry_downsampled.csv"],
    "target_column": "SATURATOR_ML_SUPPLY_F_PV.Value",

    # Model settings
    "model_name": "AutonLab/MOMENT-1-large",
    "context_length": 512,
    "forecast_horizon": 6,

    # Continual pretraining settings (MOMENT official settings)
    "pretrain_epochs": 50,
    "pretrain_batch_size": 64,
    "pretrain_lr": 1e-4,  # init_lr
    "pretrain_min_lr": 1e-5,  # min_lr
    "pretrain_warmup_lr": 1e-5,  # warmup_lr
    "pretrain_weight_decay": 0.05,
    "pretrain_warmup_steps": 1000,
    "pretrain_lr_decay_rate": 0.9,
    "pretrain_lr_scheduler": "linearwarmupcosinelr",
    "pretrain_grad_clip": 0.5,
    "pretrain_use_amp": True,  # Automatic Mixed Precision
    "pretrain_max_opt_steps": 5000000,
    "mask_ratio": 0.3,

    # Forecasting fine-tuning settings (MOMENT official settings)
    "finetune_epochs": 5,
    "finetune_batch_size": 32,
    "finetune_lr": 5e-5,  # init_lr
    "finetune_lr_scheduler": "onecyclelr",
    "finetune_pct_start": 0.3,  # Percentage of cycle spent increasing LR
    "head_dropout": 0.1,
    "weight_decay": 0.01,
    "freeze_encoder": True,
    "freeze_embedder": True,
    "freeze_head": False,

    # Data split (for SAMYANG)
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    # Memory management
    "gradient_accumulation_steps": 1,
    "min_batch_size": 1,
    "auto_reduce_batch_on_oom": True,
    "save_checkpoints": False,

    "output_dir": "/home/juyoung_ha/MFM/results/continual_pretrain_results",
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Continual Pretraining Experiment")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    return parser.parse_args()


def load_config(config_path=None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults

    Args:
        config_path: Path to JSON config file (optional)

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if config_path:
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    return config
