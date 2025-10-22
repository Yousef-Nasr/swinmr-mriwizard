"""
Evaluation Script for SwinMR + MriWizard

Tests model on test dataset and computes metrics.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_swinmr import ModelSwinMR
from data.dataloader import HybridMRIDataset, collate_fn_with_metadata
from utils.config_loader import load_config
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_psnr, calculate_ssim


def evaluate_model(model, loader, device, save_dir=None):
    """
    Evaluate model on dataset

    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to use
        save_dir: Optional directory to save output images

    Returns:
        Dictionary with metrics
    """
    model.eval()

    all_psnr = []
    all_ssim = []
    all_metadata = []

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(tqdm(loader, desc="Evaluating")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute metrics for each image
            for i in range(outputs.shape[0]):
                psnr = calculate_psnr(outputs[i:i+1], targets[i:i+1])
                ssim = calculate_ssim(outputs[i:i+1], targets[i:i+1])

                all_psnr.append(psnr)
                all_ssim.append(ssim)
                all_metadata.append(metadata[i])

                # Save images if requested
                if save_dir:
                    sample_idx = batch_idx * loader.batch_size + i
                    np.save(save_dir / f'input_{sample_idx:04d}.npy', inputs[i].cpu().numpy())
                    np.save(save_dir / f'target_{sample_idx:04d}.npy', targets[i].cpu().numpy())
                    np.save(save_dir / f'output_{sample_idx:04d}.npy', outputs[i].cpu().numpy())

    # Compute statistics
    results = {
        'mean_psnr': np.mean(all_psnr),
        'std_psnr': np.std(all_psnr),
        'mean_ssim': np.mean(all_ssim),
        'std_ssim': np.std(all_ssim),
        'num_samples': len(all_psnr),
        'all_psnr': all_psnr,
        'all_ssim': all_ssim
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SwinMR model')
    parser.add_argument('--config', type=str, required=True, help='Path to training config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, default=None, help='Test data directory (overrides config)')
    parser.add_argument('--save-images', action='store_true', help='Save output images')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine test directory
    test_dir = args.test_dir if args.test_dir else config['data'].get('test_dir')
    if not test_dir:
        raise ValueError("Test directory not specified. Use --test-dir or add to config.")

    print(f"Test directory: {test_dir}")

    # Get degradation config
    degradation_config_path = config['data']['degradation_config']

    # Create dataset
    print("Creating test dataset...")
    test_dataset = HybridMRIDataset(
        data_dir=test_dir,
        degradation_config=degradation_config_path,
        patch_size=None,  # Full images for testing
        cache_dir=Path(config['paths']['root']) / config['experiment_name'] / 'cache' / 'test',
        use_augmentation=False,
        return_metadata=True
    )

    print(f"Test dataset: {len(test_dataset)} samples")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_with_metadata
    )

    # Create model
    print("Creating model...")
    model = ModelSwinMR(config['model'])
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, None, None)

    # Setup save directory
    save_dir = None
    if args.save_images:
        save_dir = Path(config['paths']['root']) / config['experiment_name'] / 'test_results'
        print(f"Saving images to: {save_dir}")

    # Evaluate
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)

    results = evaluate_model(model, test_loader, device, save_dir)

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"PSNR: {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f} dB")
    print(f"SSIM: {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f}")
    print("="*60)

    # Save results to JSON
    results_path = Path(config['paths']['root']) / config['experiment_name'] / 'test_results.json'
    results_to_save = {
        'mean_psnr': float(results['mean_psnr']),
        'std_psnr': float(results['std_psnr']),
        'mean_ssim': float(results['mean_ssim']),
        'std_ssim': float(results['std_ssim']),
        'num_samples': results['num_samples'],
        'checkpoint': args.checkpoint
    }

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
