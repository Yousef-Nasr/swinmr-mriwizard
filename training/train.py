"""
Main Training Script for SwinMR + MriWizard

Supports:
- Hybrid data loading (NPY + DICOM)
- On-the-fly degradation
- Multi-component loss (Charbonnier + FFT + Perceptual)
- TensorBoard logging
- Checkpoint management
- Early stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from pathlib import Path
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_swinmr import ModelSwinMR
from data.dataloader import HybridMRIDataset, collate_fn_with_metadata
from utils.config_loader import load_config
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils.metrics import calculate_psnr, calculate_ssim
from utils.utils_early_stopping import EarlyStopping


def train_epoch(model, loader, optimizer, epoch, config, writer, logger, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(loader)

    for batch_idx, (inputs, targets, metadata) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Compute loss
        loss = model.compute_loss(outputs, targets, inputs)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['training']['gradient'].get('clip_enabled', False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient']['clip_value']
            )

        optimizer.step()

        total_loss += loss.item()

        # Logging
        if batch_idx % config['training']['print_freq'] == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                f"Loss: {loss.item():.6f}"
            )

            # TensorBoard
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

            # Log sample images periodically
            if config['logging'].get('log_images', False) and batch_idx % (config['training']['print_freq'] * 10) == 0:
                with torch.no_grad():
                    # Log first image in batch
                    writer.add_image('train/input', inputs[0], global_step)
                    writer.add_image('train/target', targets[0], global_step)
                    writer.add_image('train/output', outputs[0].clamp(0, 1), global_step)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, loader, epoch, config, writer, logger, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = model.compute_loss(outputs, targets, inputs)
            total_loss += loss.item()

            # Compute metrics
            for i in range(outputs.shape[0]):
                psnr = calculate_psnr(outputs[i:i+1], targets[i:i+1])
                ssim = calculate_ssim(outputs[i:i+1], targets[i:i+1])

                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    logger.info(
        f"Validation Epoch {epoch} - "
        f"Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}"
    )

    # TensorBoard logging
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/psnr', avg_psnr, epoch)
    writer.add_scalar('val/ssim', avg_ssim, epoch)

    # Log validation images
    if config['logging'].get('log_images', False):
        with torch.no_grad():
            inputs_sample, targets_sample, _ = next(iter(loader))
            inputs_sample = inputs_sample.to(device)
            targets_sample = targets_sample.to(device)
            outputs_sample = model(inputs_sample)

            # Log first few images
            num_to_log = min(4, inputs_sample.shape[0])
            for i in range(num_to_log):
                writer.add_image(f'val/input_{i}', inputs_sample[i], epoch)
                writer.add_image(f'val/target_{i}', targets_sample[i], epoch)
                writer.add_image(f'val/output_{i}', outputs_sample[i].clamp(0, 1), epoch)

    return avg_loss, avg_psnr, avg_ssim


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train SwinMR with MriWizard')
    parser.add_argument('--config', type=str, required=True, help='Path to training config')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup directories
    exp_dir = Path(config['paths']['root']) / config['experiment_name']
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Setup logger
    logger = setup_logger('train', str(log_dir / 'train.log'))
    logger.info("="*60)
    logger.info(f"Training {config['experiment_name']}")
    logger.info("="*60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")

    # Setup TensorBoard
    if config['logging'].get('tensorboard', True):
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # Get degradation config path
    degradation_config_path = config['data']['degradation_config']
    logger.info(f"Degradation config: {degradation_config_path}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = HybridMRIDataset(
        data_dir=config['data']['train_dir'],
        degradation_config=degradation_config_path,
        patch_size=config['data'].get('patch_size'),
        cache_dir=exp_dir / 'cache' / 'train',
        use_augmentation=config['data'].get('augmentation', {}).get('random_flip', True),
        return_metadata=True
    )

    val_dataset = HybridMRIDataset(
        data_dir=config['data']['val_dir'],
        degradation_config=degradation_config_path,
        patch_size=None,  # Full images for validation
        cache_dir=exp_dir / 'cache' / 'val',
        use_augmentation=False,
        return_metadata=True
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn_with_metadata
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_with_metadata
    )

    # Create model
    logger.info("Creating model...")

    # Prepare model options (ModelBase expects full config structure)
    model_opt = {
        'netG': config['model'].copy(),  # Network architecture config
        'path': {
            'models': config['paths']['checkpoints'],
            'log': config['paths']['logs'],
            'samples': config['paths']['samples']
        },
        'gpu_ids': [args.gpu] if args.gpu is not None else config['system']['gpu_ids'],
        'is_train': True,  # Training mode
        'dist': config['system'].get('distributed', False),
        'train': {
            'freeze_patch_embedding': config.get('training', {}).get('freeze_patch_embedding', False),
            'E_decay': 0,  # No EMA for now
            'G_lossfn_type': 'charbonnier',  # From training config
            'G_lossfn_weight': 1.0,
            'G_charbonnier_eps': config['training']['loss']['components']['spatial'].get('eps', 1e-3),
            'alpha': config['training']['loss']['components']['spatial']['weight'],  # Spatial loss
            'beta': config['training']['loss']['components']['frequency']['weight'],  # Frequency loss
            'gamma': config['training']['loss']['components']['perceptual']['weight'],  # Perceptual loss
            'G_optimizer_type': config['training']['optimizer']['type'],
            'G_optimizer_lr': config['training']['optimizer']['lr'],
            'G_optimizer_wd': config['training']['optimizer'].get('weight_decay', 0),
            'G_optimizer_clipgrad': config['training']['gradient'].get('clip_value', 0) if config['training']['gradient'].get('clip_enabled', False) else 0,
            'G_scheduler_milestones': config['training']['optimizer']['scheduler']['milestones'],
            'G_scheduler_gamma': config['training']['optimizer']['scheduler']['gamma']
        },
        'datasets': {
            'train': {
                'batch_size': config['data']['batch_size']
            }
        }
    }

    model = ModelSwinMR(model_opt)

    # Count parameters from netG
    num_params = sum(p.numel() for p in model.netG.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Initialize model's loss, optimizer, and scheduler (model handles this internally)
    model.define_loss()
    model.define_optimizer()
    model.define_scheduler()

    # Get references to optimizer and scheduler for checkpoint saving
    optimizer = model.G_optimizer
    scheduler = model.schedulers[0] if model.schedulers else None

    logger.info(f"Loss function: {model_opt['train']['G_lossfn_type']}")
    logger.info(f"Optimizer: {config['training']['optimizer']['type']}")
    logger.info(f"Learning rate: {config['training']['optimizer']['lr']}")
    logger.info(f"Scheduler: {config['training']['optimizer']['scheduler']['type']}")

    # Setup early stopping
    early_stopping = None
    if config['training']['early_stopping'].get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            delta=config['training']['early_stopping'].get('min_delta', 0)
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_ssim = 0.0

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_ssim = checkpoint.get('best_val_ssim', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best SSIM: {best_val_ssim:.4f}")

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Total epochs: {config['training']['epochs']}")

    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()

        # Train one epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, epoch,
            config, writer, logger, device
        )

        # Validate
        if (epoch + 1) % config['training']['val_freq'] == 0:
            val_loss, val_psnr, val_ssim = validate(
                model, val_loader, epoch, config, writer, logger, device
            )

            # Save best checkpoint
            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                save_checkpoint(
                    str(ckpt_dir / 'best.pth'),
                    model.netG,  # Save the network, not the wrapper
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    step=0,
                    metrics={'val_loss': val_loss, 'val_psnr': val_psnr, 'val_ssim': val_ssim}
                )
                logger.info(f"✓ Saved best checkpoint (SSIM: {val_ssim:.4f})")

            # Early stopping check
            if early_stopping is not None:
                early_stopping(val_loss if config['training']['early_stopping']['metric'] == 'val_loss' else -val_ssim)
                if early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Step scheduler
        scheduler.step()

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                str(ckpt_dir / f'epoch_{epoch+1}.pth'),
                model.netG,  # Save the network, not the wrapper
                optimizer,
                scheduler,
                epoch=epoch,
                step=0,
                metrics={'val_ssim': best_val_ssim}
            )

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

    # Save final checkpoint
    save_checkpoint(
        str(ckpt_dir / 'final.pth'),
        model.netG,  # Save the network, not the wrapper
        optimizer,
        scheduler,
        epoch=config['training']['epochs'] - 1,
        step=0,
        metrics={'best_val_ssim': best_val_ssim}
    )

    if writer is not None:
        writer.close()

    logger.info("="*60)
    logger.info(f"Training completed! Best SSIM: {best_val_ssim:.4f}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
