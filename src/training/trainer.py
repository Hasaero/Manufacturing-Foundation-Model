"""
Training functions for continual pretraining and forecasting fine-tuning.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from pathlib import Path
import math

from momentfm.utils.masking import Masking

from data import PretrainDataset
from utils import (
    clear_memory,
    print_memory_stats,
    save_checkpoint
)


class LinearWarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup followed by cosine annealing learning rate scheduler.

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of steps for linear warmup
        max_steps: Total number of training steps
        warmup_start_lr: Initial learning rate at start of warmup
        eta_min: Minimum learning rate
    """
    def __init__(self, optimizer, warmup_steps, max_steps, warmup_start_lr=1e-5, eta_min=1e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]


def continual_pretrain(model, datasets, config, device, output_dir):
    """Continual pretraining with masking and reconstruction"""
    print("\n" + "=" * 80)
    print("CONTINUAL PRETRAINING (Masking & Reconstruction)")
    print("=" * 80)

    # Print initial memory stats
    print_memory_stats("Initial ")

    # Set model to reconstruction mode if not already
    model.train()

    # Setup optimizer with MOMENT official settings
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['pretrain_lr'],
        weight_decay=config.get('pretrain_weight_decay', 0.05)
    )
    criterion = torch.nn.MSELoss(reduction='none')

    # Setup masking
    mask_generator = Masking(mask_ratio=config['mask_ratio'])

    # Calculate total training steps for scheduler
    total_samples = sum(len(d) for d in datasets)
    steps_per_epoch = total_samples // config['pretrain_batch_size']
    total_steps = steps_per_epoch * config['pretrain_epochs']
    max_steps = min(total_steps, config.get('pretrain_max_opt_steps', 5000000))

    # Setup learning rate scheduler: Linear Warmup + Cosine Annealing
    warmup_steps = config.get('pretrain_warmup_steps', 1000)
    warmup_lr = config.get('pretrain_warmup_lr', 1e-5)
    min_lr = config.get('pretrain_min_lr', 1e-5)

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        warmup_start_lr=warmup_lr,
        eta_min=min_lr
    )

    global_step = 0  # Track global training steps

    # Mixed precision (AMP)
    use_amp = config.get('pretrain_use_amp', True)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Gradient accumulation
    accumulation_steps = config.get('gradient_accumulation_steps', 1)

    # Get batch size (will be adjusted if OOM occurs)
    current_batch_size = config['pretrain_batch_size']
    min_batch_size = config.get('min_batch_size', 1)

    # Get gradient clipping value
    grad_clip = config.get('pretrain_grad_clip', 0.5)

    print(f"Continual Pretraining Configuration (MOMENT Official Settings):")
    print(f"  Initial LR: {config['pretrain_lr']}")
    print(f"  Min LR: {min_lr}")
    print(f"  Warmup LR: {warmup_lr}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Total Steps: {total_steps:,}")
    print(f"  Max Steps: {max_steps:,}")
    print(f"  Weight Decay: {config.get('pretrain_weight_decay', 0.05)}")
    print(f"  Gradient Clipping: {grad_clip}")
    print(f"  LR Scheduler: {config.get('pretrain_lr_scheduler', 'linearwarmupcosinelr')}")
    print(f"  Use AMP: {use_amp}")
    print(f"  Epochs per Dataset: {config['pretrain_epochs']}")
    print(f"  Total Datasets: {len(datasets)}")
    print(f"\n  Training Order: Domain Sequential (Dataset 1 → Dataset 2 → Dataset 3)")

    # Domain sequential training: Train completely on Dataset 1, then Dataset 2, then Dataset 3
    for dataset_idx, dataset_data in enumerate(datasets):
        print(f"\n{'='*80}")
        print(f"DATASET {dataset_idx + 1}/{len(datasets)}")
        print(f"{'='*80}")

        # Train on this dataset for all epochs
        for epoch in range(config['pretrain_epochs']):
            print(f"\n  Epoch {epoch + 1}/{config['pretrain_epochs']} (Dataset {dataset_idx + 1})")

            epoch_losses = []
            oom_count = 0

            # Create dataset and dataloader with current batch size
            pretrain_dataset = PretrainDataset(dataset_data, config['context_length'])

            # Retry loop for handling OOM
            success = False
            retry_batch_size = current_batch_size

            while not success and retry_batch_size >= min_batch_size:
                try:
                    pretrain_loader = DataLoader(
                        pretrain_dataset,
                        batch_size=retry_batch_size,
                        shuffle=True
                    )

                    batch_losses = []
                    optimizer.zero_grad(set_to_none=True)

                    for batch_idx, (batch_x, input_mask) in enumerate(tqdm(
                        pretrain_loader,
                        desc=f"    Dataset {dataset_idx+1} (BS={retry_batch_size})"
                    )):
                        try:
                            batch_x = batch_x.float().to(device)
                            input_mask = input_mask.to(device)

                            # Clip input values to prevent extreme values BEFORE any processing
                            batch_x = torch.clamp(batch_x, -10.0, 10.0)

                            # Save clipped version for loss calculation
                            batch_x_orig = batch_x.clone()

                            # Reshape: [batch, n_channels, length] -> [batch*n_channels, 1, length]
                            batch_size, n_channels, seq_len = batch_x.shape
                            batch_x_reshaped = batch_x.reshape(-1, 1, seq_len)
                            input_mask_reshaped = input_mask.repeat_interleave(n_channels, dim=0)

                            # Generate mask
                            mask = mask_generator.generate_mask(
                                x=batch_x_reshaped,
                                input_mask=input_mask_reshaped
                            ).to(device).long()

                            # DISABLE mixed precision to prevent NaN
                            # Mixed precision (FP16) can cause numerical instability
                            # Forward pass in FP32
                            output = model(
                                x_enc=batch_x_reshaped,
                                input_mask=input_mask_reshaped,
                                mask=mask
                            )

                            # Reshape reconstruction back
                            reconstruction = output.reconstruction.reshape(batch_size, n_channels, seq_len)

                            # Check for NaN/Inf in output
                            if torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any():
                                print(f"\n    WARNING: NaN/Inf detected in reconstruction output!")
                                print(f"    Input stats - min: {batch_x_orig.min():.4f}, max: {batch_x_orig.max():.4f}, mean: {batch_x_orig.mean():.4f}")
                                print(f"    Output stats - min: {reconstruction.min():.4f}, max: {reconstruction.max():.4f}, mean: {reconstruction.mean():.4f}")
                                print(f"    Skipping this batch...")
                                optimizer.zero_grad(set_to_none=True)
                                continue

                            # Calculate loss
                            loss_per_element = criterion(reconstruction, batch_x_orig)
                            loss = loss_per_element.mean()

                            # Check for NaN in loss
                            if torch.isnan(loss) or torch.isinf(loss):
                                print(f"\n    WARNING: NaN/Inf loss detected! Skipping batch...")
                                optimizer.zero_grad(set_to_none=True)
                                continue

                            # Scale loss for gradient accumulation
                            loss = loss / accumulation_steps

                            # Backward pass with optional mixed precision
                            if use_amp and scaler is not None:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()

                            # Update weights every accumulation_steps
                            if (batch_idx + 1) % accumulation_steps == 0:
                                # Get current learning rate
                                current_lr = optimizer.param_groups[0]['lr']

                                # Gradient clipping
                                if use_amp and scaler is not None:
                                    scaler.unscale_(optimizer)
                                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                                # Check for exploding gradients or NaN gradients
                                if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                                    print(f"\n    WARNING: NaN/Inf gradient detected! Skipping update...")
                                    optimizer.zero_grad(set_to_none=True)
                                    continue

                                if grad_norm > 5.0:
                                    print(f"\n    WARNING: Large gradient norm detected: {grad_norm:.2f} (LR: {current_lr:.6f})")

                                # Optimizer step with optional AMP scaling
                                if use_amp and scaler is not None:
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    optimizer.step()

                                # Update learning rate scheduler
                                scheduler.step()
                                global_step += 1  # Increment global step

                                # Check if model parameters became NaN after update
                                params_nan = any(torch.isnan(p).any() for p in model.parameters() if p.requires_grad)
                                if params_nan:
                                    print(f"\n    ERROR: Model parameters became NaN! This should not happen.")
                                    print(f"    Last grad norm: {grad_norm:.4f}, Last loss: {loss.item():.6f}, LR: {current_lr:.6f}")
                                    raise RuntimeError("Model parameters became NaN")

                                optimizer.zero_grad(set_to_none=True)

                            batch_losses.append(loss.item() * accumulation_steps)

                            # Clean up batch tensors
                            del batch_x, input_mask, batch_x_orig, mask, output, reconstruction, loss

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print(f"\n    WARNING: OOM in batch {batch_idx}. Clearing cache...")
                                clear_memory()
                                oom_count += 1

                                if config.get('auto_reduce_batch_on_oom', True):
                                    # Reduce batch size and retry this dataset
                                    raise e  # Re-raise to trigger outer retry
                                else:
                                    # Skip this batch and continue
                                    print(f"    Skipping batch {batch_idx}")
                                    continue
                            else:
                                raise e

                    # Successful completion
                    dataset_loss = np.mean(batch_losses) if batch_losses else 0.0
                    epoch_losses.append(dataset_loss)
                    print(f"    Dataset {dataset_idx + 1} Loss: {dataset_loss:.6f}")

                    # Update current batch size if successful
                    current_batch_size = retry_batch_size
                    success = True

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n    ERROR: Out of memory with batch size {retry_batch_size}")
                        clear_memory()

                        # Reduce batch size
                        retry_batch_size = max(retry_batch_size // 2, min_batch_size)

                        if retry_batch_size >= min_batch_size:
                            print(f"    Retrying with batch size {retry_batch_size}...")
                        else:
                            print(f"    Cannot reduce batch size below {min_batch_size}. Skipping this dataset.")
                            success = True  # Exit retry loop
                    else:
                        raise e

            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            # Get current LR from optimizer
            current_epoch_lr = optimizer.param_groups[0]['lr']

            print(f"  Epoch {epoch + 1}/{config['pretrain_epochs']} (Dataset {dataset_idx + 1}) Average Loss: {avg_epoch_loss:.6f}")
            print(f"  Current LR: {current_epoch_lr:.6f} (Step: {global_step})")

            if oom_count > 0:
                print(f"  OOM events this epoch: {oom_count}")

            print_memory_stats(f"  After Epoch {epoch + 1} (Dataset {dataset_idx + 1}) ")

            # Save checkpoint
            if config.get('save_checkpoints', True):
                checkpoint_path = output_dir / f"pretrain_checkpoint_dataset{dataset_idx+1}_epoch{epoch+1}.pt"
                save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)

            clear_memory()

        # Dataset completion summary
        print(f"\n{'='*80}")
        print(f"DATASET {dataset_idx + 1}/{len(datasets)} COMPLETED")
        print(f"{'='*80}\n")

    print("\nContinual pretraining completed!")
    return model


def train_forecasting(model, train_loader, val_loader, config, device, target_idx, output_dir, model_name="forecasting"):
    """Fine-tune forecasting head"""
    print("\n" + "=" * 80)
    print("FORECASTING FINE-TUNING")
    print("=" * 80)

    print_memory_stats("Initial ")

    model.train()

    # Setup optimizer and scheduler (MOMENT official settings)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['finetune_lr'])

    total_steps = len(train_loader) * config['finetune_epochs']
    pct_start = config.get('finetune_pct_start', 0.3)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['finetune_lr'],
        total_steps=total_steps,
        pct_start=pct_start
    )

    print(f"\nFine-tuning Configuration (MOMENT Official Settings):")
    print(f"  Initial LR: {config['finetune_lr']}")
    print(f"  LR Scheduler: {config.get('finetune_lr_scheduler', 'onecyclelr')}")
    print(f"  PCT Start: {pct_start}")
    print(f"  Epochs: {config['finetune_epochs']}")
    print(f"  Total Steps: {total_steps:,}")
    print(f"  Weight Decay: {config.get('weight_decay', 0.01)}")
    print(f"  Head Dropout: {config.get('head_dropout', 0.1)}")

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Gradient accumulation
    accumulation_steps = config.get('gradient_accumulation_steps', 1)

    best_val_loss = float('inf')
    best_model_state = None
    oom_count = 0

    for epoch in range(config['finetune_epochs']):
        # Training
        model.train()
        train_losses = []

        for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['finetune_epochs']} [Train]"
        )):
            try:
                timeseries = timeseries.float().to(device)
                input_mask = input_mask.to(device)
                forecast = forecast.float().to(device)

                with torch.cuda.amp.autocast():
                    output = model(x_enc=timeseries, input_mask=input_mask)

                # Calculate loss only on target variable
                loss = criterion(
                    output.forecast[:, target_idx:target_idx+1, :],
                    forecast[:, target_idx:target_idx+1, :]
                )

                # Scale for gradient accumulation
                loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                train_losses.append(loss.item() * accumulation_steps)

                # Clean up
                del timeseries, input_mask, forecast, output, loss

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  WARNING: OOM in training batch {batch_idx}. Skipping batch...")
                    clear_memory()
                    oom_count += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e

        train_loss = np.mean(train_losses) if train_losses else 0.0

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_idx, (timeseries, forecast, input_mask) in enumerate(tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{config['finetune_epochs']} [Val]"
            )):
                try:
                    timeseries = timeseries.float().to(device)
                    input_mask = input_mask.to(device)
                    forecast = forecast.float().to(device)

                    with torch.cuda.amp.autocast():
                        output = model(x_enc=timeseries, input_mask=input_mask)

                    loss = criterion(
                        output.forecast[:, target_idx:target_idx+1, :],
                        forecast[:, target_idx:target_idx+1, :]
                    )

                    val_losses.append(loss.item())

                    # Clean up
                    del timeseries, input_mask, forecast, output, loss

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n  WARNING: OOM in validation batch {batch_idx}. Skipping batch...")
                        clear_memory()
                        oom_count += 1
                        continue
                    else:
                        raise e

        val_loss = np.mean(val_losses) if val_losses else float('inf')

        print(f"\nEpoch {epoch+1}/{config['finetune_epochs']}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")

        if oom_count > 0:
            print(f"  OOM events this epoch: {oom_count}")
            oom_count = 0  # Reset for next epoch

        print_memory_stats(f"  After Epoch {epoch + 1} ")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best validation loss: {best_val_loss:.6f}")

        # Save checkpoint
        if config.get('save_checkpoints', True):
            checkpoint_path = output_dir / f"{model_name}_checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)

        clear_memory()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nFine-tuning completed! Best Val Loss: {best_val_loss:.6f}")
    return model
