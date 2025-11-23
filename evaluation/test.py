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
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_swinmr import ModelSwinMR
from data.dataloader import HybridMRIDataset, collate_fn_with_metadata
from utils.config_loader import load_config
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_psnr, calculate_ssim


def save_test_sample(input_img, target_img, output_img, psnr_out, ssim_out, save_path, metadata=None, sample_idx=0):
    """
    Save test sample with GT and predicted images side by side.

    Args:
        input_img: Input degraded image (C, H, W)
        target_img: Ground truth image (C, H, W)
        output_img: Model output image (C, H, W)
        psnr_out: PSNR value for output
        ssim_out: SSIM value for output
        save_path: Path to save the image
        metadata: Optional metadata dict
        sample_idx: Sample index for title
    """
    # Convert to numpy and squeeze channel dimension
    input_np = input_img.detach().cpu().numpy().squeeze()
    target_np = target_img.detach().cpu().numpy().squeeze()
    output_np = output_img.detach().cpu().numpy().squeeze()

    # Clip to valid range
    input_np = np.clip(input_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    output_np = np.clip(output_np, 0, 1)

    # Calculate metrics for degraded input
    psnr_input = calculate_psnr(input_img.unsqueeze(0), target_img.unsqueeze(0))
    ssim_input = calculate_ssim(input_img.unsqueeze(0), target_img.unsqueeze(0))

    # Calculate improvement
    psnr_improvement = psnr_out - psnr_input
    ssim_improvement = ssim_out - ssim_input

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input (degraded)
    axes[0].imshow(input_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(
        f'Input (Degraded)\nPSNR: {psnr_input:.2f} dB | SSIM: {ssim_input:.4f}',
        fontsize=11
    )
    axes[0].axis('off')

    # Ground Truth
    axes[1].imshow(target_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth', fontsize=11)
    axes[1].axis('off')

    # Output (predicted)
    axes[2].imshow(output_np, cmap='gray', vmin=0, vmax=1)

    # Color code improvement: green if positive, red if negative
    psnr_color = 'green' if psnr_improvement > 0 else 'red'
    ssim_color = 'green' if ssim_improvement > 0 else 'red'

    axes[2].set_title(
        f'Output (Reconstructed)\n'
        f'PSNR: {psnr_out:.2f} dB (${psnr_improvement:+.2f}$) | '
        f'SSIM: {ssim_out:.4f} (${ssim_improvement:+.4f}$)',
        fontsize=11,
        color='black'
    )
    axes[2].axis('off')

    # Add metadata and improvement summary
    title_parts = [f'Test Sample #{sample_idx}']
    if metadata:
        source = metadata.get('source', 'Unknown')
        title_parts.append(f'Source: {Path(source).name}')

    title_parts.append(
        f'Improvement: PSNR {psnr_improvement:+.2f} dB, SSIM {ssim_improvement:+.4f}'
    )

    fig.suptitle(' | '.join(title_parts), fontsize=10, y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, loader, device, save_dir=None):
    """
    Evaluate model on dataset with batch processing for efficiency.
    
    The function processes images in batches (forward pass on GPU in parallel),
    then computes metrics and saves images individually. This provides a good
    balance between speed and functionality.

    Args:
        model: Model to evaluate
        loader: DataLoader (with batch_size > 1 for efficiency)
        device: Device to use
        save_dir: Optional directory to save output images

    Returns:
        Dictionary with metrics (mean/std PSNR, SSIM, all individual scores)
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
                    
                    # Save visual comparison (like validation images)
                    save_path = save_dir / f'test_sample_{sample_idx:04d}.png'
                    save_test_sample(
                        inputs[i],
                        targets[i],
                        outputs[i],
                        psnr,
                        ssim,
                        save_path,
                        metadata=metadata[i],
                        sample_idx=sample_idx
                    )
                    
                    # Also save raw numpy arrays for further analysis
                    npy_dir = save_dir / 'npy'
                    npy_dir.mkdir(exist_ok=True)
                    np.save(npy_dir / f'input_{sample_idx:04d}.npy', inputs[i].cpu().numpy())
                    np.save(npy_dir / f'target_{sample_idx:04d}.npy', targets[i].cpu().numpy())
                    np.save(npy_dir / f'output_{sample_idx:04d}.npy', outputs[i].cpu().numpy())

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
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers (default: 4)')
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

    # Create dataloader with configurable batch size
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_with_metadata
    )
    
    print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")

    # Create model with proper config structure
    print("Creating model...")
    
    # Build model config in the expected format for SwinMR
    model_opt = {
        **config['model'],  # Include all model architecture params
        'path': {
            'root': config['paths']['root'],
            'models': str(Path(config['paths']['root']) / config['experiment_name'] / 'checkpoints'),
            'pretrained_netG': None,
            'pretrained_netE': None,
            'pretrained_optimizerG': None
        },
        'gpu_ids': [args.gpu] if args.gpu is not None else config['system']['gpu_ids'],
        'is_train': False,  # Evaluation mode
        'dist': False,
        'find_unused_parameters': False,
        'train': {
            'freeze_patch_embedding': False,
            'E_decay': 0,
            'G_lossfn_type': 'charbonnier',
            'G_lossfn_weight': 1.0,
            'G_charbonnier_eps': 1e-3,
            'alpha': 1.0,
            'beta': 0.1,
            'gamma': 0.0025,
            'G_param_strict': True,
            'E_param_strict': True
        },
        'datasets': {},
        'netG': config['model']  # netG config
    }
    
    model = ModelSwinMR(model_opt)
    model = model.netG.to(device)

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
