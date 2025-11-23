"""
Utility functions for memory management, checkpointing, and system operations.
"""

import gc
import shutil
from pathlib import Path
import torch


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    return None


def clear_memory():
    """Clear GPU and CPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_disk_space(path, min_gb=5):
    """Check if there's enough disk space"""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        return free_gb >= min_gb, free_gb
    except:
        return True, 0  # Assume OK if we can't check


def print_memory_stats(prefix=""):
    """Print current memory statistics"""
    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info()
        print(f"{prefix}GPU Memory: Allocated={mem_info['allocated']:.2f}GB, "
              f"Reserved={mem_info['reserved']:.2f}GB, "
              f"Free={mem_info['free']:.2f}GB, "
              f"Total={mem_info['total']:.2f}GB")


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint with error handling"""
    filepath = Path(filepath)
    temp_filepath = None

    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check disk space (need at least 2GB for MOMENT checkpoint)
        has_space, free_gb = check_disk_space(filepath.parent, min_gb=2)
        if not has_space:
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

        # Rename to final path (atomic operation)
        temp_filepath.replace(filepath)

        print(f"Checkpoint saved: {filepath}")
        return True

    except Exception as e:
        print(f"WARNING: Failed to save checkpoint: {e}")
        print(f"  Filepath: {filepath}")
        print(f"  Consider disabling checkpoints by setting 'save_checkpoints': false in config")

        # Try to clean up temp file
        try:
            if temp_filepath and temp_filepath.exists():
                temp_filepath.unlink()
        except:
            pass

        return False


def safe_save_model(model, filepath, model_name="model"):
    """Safely save model state dict with error handling"""
    filepath = Path(filepath)
    temp_filepath = None

    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Check disk space
        has_space, free_gb = check_disk_space(filepath.parent, min_gb=2)
        if not has_space:
            print(f"WARNING: Low disk space ({free_gb:.2f}GB free). Skipping {model_name} save.")
            print(f"  Model will remain in memory but won't be saved to disk.")
            return False

        # Save to temporary file first
        temp_filepath = filepath.with_suffix('.tmp')
        torch.save(model.state_dict(), temp_filepath)

        # Rename to final path (atomic operation)
        temp_filepath.replace(filepath)

        print(f"{model_name} saved: {filepath}")
        return True

    except Exception as e:
        print(f"WARNING: Failed to save {model_name}: {e}")
        print(f"  Filepath: {filepath}")
        print(f"  Model will remain in memory for evaluation.")

        # Try to clean up temp file
        try:
            if temp_filepath and temp_filepath.exists():
                temp_filepath.unlink()
        except:
            pass

        return False


def load_checkpoint(model, optimizer, filepath):
    """Load training checkpoint"""
    if Path(filepath).exists():
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: {filepath} (epoch {epoch}, loss {loss:.6f})")
        return epoch, loss
    return 0, None
