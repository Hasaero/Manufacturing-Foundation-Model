"""
Utility functions for memory management and model saving.
"""

import gc
import shutil
from pathlib import Path
import torch


def clear_memory():
    """Clear GPU and CPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_memory_stats(prefix=""):
    """Print current GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated

        print(f"{prefix}GPU Memory: Allocated={allocated:.2f}GB, "
              f"Reserved={reserved:.2f}GB, "
              f"Free={free:.2f}GB, "
              f"Total={total:.2f}GB")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint with error handling and atomic write"""
    filepath = Path(filepath)
    temp_filepath = None

    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check disk space
        stat = shutil.disk_usage(filepath.parent)
        free_gb = stat.free / (1024**3)
        if free_gb < 2:
            print(f"WARNING: Low disk space ({free_gb:.2f}GB free). Skipping checkpoint save.")
            return False

        # Save to temporary file first
        temp_filepath = filepath.with_suffix('.tmp')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        torch.save(checkpoint, temp_filepath)
        temp_filepath.replace(filepath)

        print(f"Checkpoint saved: {filepath}")
        return True

    except Exception as e:
        print(f"WARNING: Failed to save checkpoint: {e}")
        if temp_filepath and temp_filepath.exists():
            try:
                temp_filepath.unlink()
            except:
                pass
        return False


def safe_save_model(model, filepath, model_name="model"):
    """Safely save model state dict with error handling and atomic write"""
    filepath = Path(filepath)
    temp_filepath = None

    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check disk space
        stat = shutil.disk_usage(filepath.parent)
        free_gb = stat.free / (1024**3)
        if free_gb < 2:
            print(f"WARNING: Low disk space ({free_gb:.2f}GB free). Skipping {model_name} save.")
            return False

        # Save to temporary file first for atomic write
        temp_filepath = filepath.with_suffix('.tmp')
        torch.save(model.state_dict(), temp_filepath)
        temp_filepath.replace(filepath)

        print(f"{model_name} saved: {filepath}")
        return True

    except Exception as e:
        print(f"WARNING: Failed to save {model_name}: {e}")
        if temp_filepath and temp_filepath.exists():
            try:
                temp_filepath.unlink()
            except:
                pass
        return False
