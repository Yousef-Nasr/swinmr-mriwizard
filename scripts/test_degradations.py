"""
Interactive Degradation Configuration Tester

Test and optimize degradation configurations before training.
Generates visual samples with different degradation settings to help you
choose the best configuration for your use case.

Usage:
    # Interactive mode (asks questions)
    python scripts/test_degradations.py --image path/to/image.dcm

    # Test existing config
    python scripts/test_degradations.py \
        --image path/to/image.dcm \
        --config configs/degradation_all.json \
        --num-samples 10

    # Generate comparison grid
    python scripts/test_degradations.py \
        --image path/to/image.dcm \
        --compare-configs configs/degradation_minimal.json configs/degradation_all.json

    # Test specific degradations only
    python scripts/test_degradations.py \
        --image path/to/image.dcm \
        --test-noise --test-undersampling --test-artifacts
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MriWizard.io.dicom_loader import LoadDICOM
from MriWizard.io.image_loader import LoadImage
from MriWizard.core.utils import fft2c, normalize
from data.pipeline_builder import build_degradation_pipeline
from utils.metrics import calculate_psnr, calculate_ssim


def load_image(image_path):
    """Load image from file (DICOM, NPY, or PNG/JPG)."""
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()

    if suffix == '.npy':
        image = np.load(image_path).astype(np.float32)
        if image.ndim == 3:
            image = image[0] if image.shape[0] == 1 else image[image.shape[0] // 2]
        if image.max() > 1.0:
            image = normalize(image)
        kspace = fft2c(image)

    elif suffix == '.dcm':
        loader = LoadDICOM(convert_to_kspace=True)
        record = loader.load(str(image_path))
        image = record['image']
        kspace = record['kspace']

    elif suffix in ['.png', '.jpg', '.jpeg']:
        loader = LoadImage(convert_to_kspace=True, grayscale=True)
        record = loader.load(str(image_path))
        image = record['image']
        kspace = record['kspace']

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return {
        'image': image,
        'kspace': kspace,
        'mask': None,
        'metadata': {'source': str(image_path), 'applied': []}
    }


def apply_degradation(record, config):
    """Apply degradation pipeline to record."""
    pipeline = build_degradation_pipeline(config)
    degraded = pipeline(record.copy())
    return degraded


def create_comparison_figure(original, degraded_samples, save_path=None):
    """
    Create comparison figure showing original and multiple degraded versions.

    Args:
        original: Original image
        degraded_samples: List of (degraded_image, title, metrics) tuples
        save_path: Optional path to save figure
    """
    n_samples = len(degraded_samples)
    n_cols = min(4, n_samples + 1)  # +1 for original
    n_rows = (n_samples + 1 + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)

    # Plot original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Original (Ground Truth)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot degraded samples
    for idx, (degraded, title, metrics) in enumerate(degraded_samples, 1):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        ax.imshow(degraded, cmap='gray', vmin=0, vmax=1)

        # Build title with metrics
        title_str = f'{title}\n'
        if metrics:
            title_str += f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}"

        ax.set_title(title_str, fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_test_configs():
    """Generate test configurations for common scenarios."""
    configs = {
        'minimal': {
            'name': 'Minimal Degradation',
            'noise': {
                'enabled': True,
                'type': 'gaussian_kspace',
                'mean': 0.0,
                'std_range': [0.01, 0.02],
                'relative': True,
                'reference': 'std'
            },
            'undersampling': {
                'strategy': 'one_of',
                'patterns': [{
                    'name': 'random',
                    'enabled': True,
                    'weight': 1.0,
                    'params': {
                        'prob_range': [0.6, 0.7],
                        'center_fraction': 0.08,
                        'axis': -2
                    }
                }]
            },
            'artifacts': {},
            'augmentation': {'enabled': False}
        },
        'moderate': {
            'name': 'Moderate Degradation',
            'noise': {
                'enabled': True,
                'type': 'gaussian_kspace',
                'mean': 0.0,
                'std_range': [0.02, 0.04],
                'relative': True,
                'reference': 'std'
            },
            'undersampling': {
                'strategy': 'one_of',
                'patterns': [{
                    'name': 'random',
                    'enabled': True,
                    'weight': 1.0,
                    'params': {
                        'prob_range': [0.4, 0.5],
                        'center_fraction': 0.08,
                        'axis': -2
                    }
                }]
            },
            'artifacts': {
                'gibbs': {
                    'enabled': True,
                    'probability': 0.3,
                    'params': {
                        'fraction_range': [0.7, 0.9],
                        'axes': [-1, -2]
                    }
                }
            },
            'augmentation': {'enabled': False}
        },
        'heavy': {
            'name': 'Heavy Degradation',
            'noise': {
                'enabled': True,
                'type': 'gaussian_kspace',
                'mean': 0.0,
                'std_range': [0.05, 0.08],
                'relative': True,
                'reference': 'std'
            },
            'undersampling': {
                'strategy': 'one_of',
                'patterns': [{
                    'name': 'random',
                    'enabled': True,
                    'weight': 1.0,
                    'params': {
                        'prob_range': [0.2, 0.3],
                        'center_fraction': 0.08,
                        'axis': -2
                    }
                }]
            },
            'artifacts': {
                'gibbs': {
                    'enabled': True,
                    'probability': 0.5,
                    'params': {
                        'fraction_range': [0.5, 0.7],
                        'axes': [-1, -2]
                    }
                },
                'motion': {
                    'enabled': True,
                    'probability': 0.3,
                    'params': {
                        'num_transforms': [2, 5],
                        'max_rotation': 5.0,
                        'max_translation': 10.0
                    }
                }
            },
            'augmentation': {'enabled': False}
        }
    }
    return configs


def interactive_mode(image_path, output_dir):
    """Interactive mode - asks user questions to build config."""
    print("\n" + "="*70)
    print(" "*20 + "INTERACTIVE DEGRADATION TESTER")
    print("="*70)

    print(f"\nLoading image: {image_path}")
    original_record = load_image(image_path)
    original_image = original_record['image']

    print("\nLet's build your degradation configuration!\n")

    # 1. Choose base level
    print("1. Choose base degradation level:")
    print("   [1] Minimal  - Light degradation for easy reconstruction")
    print("   [2] Moderate - Balanced degradation (recommended)")
    print("   [3] Heavy    - Strong degradation for challenging scenarios")
    print("   [4] Custom   - Build from scratch")

    choice = input("\nYour choice [1-4]: ").strip()

    test_configs = generate_test_configs()

    if choice == '1':
        config = test_configs['minimal']
    elif choice == '2':
        config = test_configs['moderate']
    elif choice == '3':
        config = test_configs['heavy']
    elif choice == '4':
        config = build_custom_config()
    else:
        print("Invalid choice, using moderate.")
        config = test_configs['moderate']

    # 2. Augmentation
    print("\n2. Enable spatial augmentation?")
    print("   Augmentation includes: horizontal flip, vertical flip, 90° rotations")
    aug_choice = input("   Enable? [y/n]: ").strip().lower()

    if aug_choice == 'y':
        config['augmentation'] = {
            'enabled': True,
            'horizontal_flip': {'enabled': True, 'probability': 0.5},
            'vertical_flip': {'enabled': True, 'probability': 0.5},
            'rotation_90': {'enabled': True, 'probability': 0.5, 'angles': [90, 180, 270]}
        }

    # 3. Generate samples
    print("\n3. How many samples to generate?")
    num_samples = int(input("   Number of samples [5-20]: ").strip() or "10")

    print(f"\n{'='*70}")
    print("Generating samples...")
    print(f"{'='*70}\n")

    # Generate samples
    degraded_samples = []
    for i in range(num_samples):
        degraded_record = apply_degradation(original_record, config)
        degraded_image = degraded_record['image']

        # Compute metrics
        psnr = calculate_psnr(degraded_image, original_image)
        ssim = calculate_ssim(degraded_image, original_image)

        degraded_samples.append((
            degraded_image,
            f"Sample {i+1}",
            {'psnr': psnr, 'ssim': ssim}
        ))

        print(f"  Sample {i+1}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

    # Compute statistics
    psnrs = [m['psnr'] for _, _, m in degraded_samples]
    ssims = [m['ssim'] for _, _, m in degraded_samples]

    print(f"\n{'='*70}")
    print("Statistics:")
    print(f"  PSNR: {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB")
    print(f"  SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"{'='*70}\n")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / 'degradation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved configuration to {config_path}")

    # Save comparison figure
    comparison_path = output_dir / 'degradation_samples.png'
    create_comparison_figure(original_image, degraded_samples[:12], comparison_path)

    # Save individual samples
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    for i, (degraded, _, metrics) in enumerate(degraded_samples):
        save_sample_comparison(
            original_image, degraded, metrics,
            samples_dir / f'sample_{i:03d}.png'
        )

    print(f"✓ Saved {len(degraded_samples)} individual samples to {samples_dir}")

    print(f"\n{'='*70}")
    print("✓ Testing complete!")
    print(f"{'='*70}\n")

    return config


def build_custom_config():
    """Build custom configuration interactively."""
    print("\n--- Building Custom Configuration ---\n")

    config = {
        'name': 'custom',
        'noise': {'enabled': False},
        'undersampling': {'strategy': 'none', 'patterns': []},
        'artifacts': {},
        'augmentation': {'enabled': False}
    }

    # Noise
    if input("Enable noise? [y/n]: ").strip().lower() == 'y':
        std = float(input("  Noise level (0.01-0.1) [0.03]: ").strip() or "0.03")
        config['noise'] = {
            'enabled': True,
            'type': 'gaussian_kspace',
            'mean': 0.0,
            'std_range': [std * 0.8, std * 1.2],
            'relative': True,
            'reference': 'std'
        }

    # Undersampling
    if input("Enable undersampling? [y/n]: ").strip().lower() == 'y':
        prob = float(input("  Sampling probability (0.2-0.8) [0.5]: ").strip() or "0.5")
        config['undersampling'] = {
            'strategy': 'one_of',
            'patterns': [{
                'name': 'random',
                'enabled': True,
                'weight': 1.0,
                'params': {
                    'prob_range': [prob * 0.9, prob * 1.1],
                    'center_fraction': 0.08,
                    'axis': -2
                }
            }]
        }

    # Gibbs artifact
    if input("Enable Gibbs ringing? [y/n]: ").strip().lower() == 'y':
        config['artifacts']['gibbs'] = {
            'enabled': True,
            'probability': 0.5,
            'params': {
                'fraction_range': [0.6, 0.9],
                'axes': [-1, -2]
            }
        }

    return config


def save_sample_comparison(original, degraded, metrics, save_path):
    """Save side-by-side comparison of original and degraded."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(degraded, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(
        f"Degraded\nPSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}",
        fontsize=12
    )
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_configs_mode(image_path, config_paths, output_dir):
    """Compare multiple configurations side-by-side."""
    print("\n" + "="*70)
    print(" "*20 + "CONFIGURATION COMPARISON")
    print("="*70)

    print(f"\nLoading image: {image_path}")
    original_record = load_image(image_path)
    original_image = original_record['image']

    print(f"\nComparing {len(config_paths)} configurations...\n")

    degraded_samples = []

    for config_path in config_paths:
        print(f"Testing: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Generate samples
        sample_psnrs = []
        sample_ssims = []

        for i in range(5):  # 5 samples per config
            degraded_record = apply_degradation(original_record, config)
            degraded_image = degraded_record['image']

            psnr = calculate_psnr(degraded_image, original_image)
            ssim = calculate_ssim(degraded_image, original_image)

            sample_psnrs.append(psnr)
            sample_ssims.append(ssim)

            if i == 0:  # Save first sample
                degraded_samples.append((
                    degraded_image,
                    Path(config_path).stem,
                    {'psnr': psnr, 'ssim': ssim}
                ))

        print(f"  PSNR: {np.mean(sample_psnrs):.2f} ± {np.std(sample_psnrs):.2f} dB")
        print(f"  SSIM: {np.mean(sample_ssims):.4f} ± {np.std(sample_ssims):.4f}\n")

    # Save comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / 'config_comparison.png'
    create_comparison_figure(original_image, degraded_samples, comparison_path)

    print(f"✓ Saved comparison to {comparison_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive degradation configuration tester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image (DICOM, NPY, PNG, JPG)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to degradation config JSON')
    parser.add_argument('--compare-configs', nargs='+', default=None,
                       help='Compare multiple configs')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='./degradation_tests',
                       help='Output directory for results')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Mode 1: Compare multiple configs
    if args.compare_configs:
        compare_configs_mode(args.image, args.compare_configs, output_dir)

    # Mode 2: Test single config
    elif args.config:
        print(f"\nTesting configuration: {args.config}")
        with open(args.config) as f:
            config = json.load(f)

        original_record = load_image(args.image)
        original_image = original_record['image']

        degraded_samples = []
        for i in range(args.num_samples):
            degraded_record = apply_degradation(original_record, config)
            degraded_image = degraded_record['image']

            psnr = calculate_psnr(degraded_image, original_image)
            ssim = calculate_ssim(degraded_image, original_image)

            degraded_samples.append((
                degraded_image,
                f"Sample {i+1}",
                {'psnr': psnr, 'ssim': ssim}
            ))

        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = output_dir / f'test_{Path(args.config).stem}.png'
        create_comparison_figure(original_image, degraded_samples, comparison_path)

    # Mode 3: Interactive mode
    else:
        interactive_mode(args.image, output_dir)


if __name__ == '__main__':
    main()
