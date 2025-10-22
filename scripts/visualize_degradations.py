#!/usr/bin/env python3
"""
Visualize Degradation Patterns

Shows what each degradation pattern does to an example image.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'MriWizard'))

from data.pipeline_builder import build_degradation_pipeline
from MriWizard.io.dicom_loader import LoadDICOM
from MriWizard.io.image_loader import LoadImage


def visualize():
    parser = argparse.ArgumentParser(description='Visualize degradation patterns')
    parser.add_argument('--degradation-config', type=str, required=True, help='Path to degradation config')
    parser.add_argument('--image', type=str, required=True, help='Path to input image (DICOM/NPY/PNG/JPG)')
    parser.add_argument('--output', type=str, default='degradation_visualization.png', help='Output visualization path')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of degradation samples to generate')
    args = parser.parse_args()

    print("="*60)
    print("Degradation Visualization")
    print("="*60)

    # Load image
    image_path = Path(args.image)
    suffix = image_path.suffix.lower()

    print(f"\nLoading image: {args.image}")

    if suffix == '.dcm':
        loader = LoadDICOM(convert_to_kspace=True)
        record = loader.load(str(image_path))
    elif suffix == '.npy':
        image = np.load(image_path).astype(np.float32)
        if image.ndim == 3:
            image = image[image.shape[0] // 2]  # Take middle slice
        if image.max() > 1.0:
            image = image / image.max()
        from MriWizard.core.utils import fft2c
        record = {
            'kspace': fft2c(image),
            'image': image,
            'mask': None,
            'metadata': {'source': str(image_path)}
        }
    else:
        loader = LoadImage(convert_to_kspace=True, grayscale=True)
        record = loader.load(str(image_path))

    print(f"  ✓ Image loaded, shape: {record['image'].shape}")

    # Load degradation config
    print(f"\nLoading degradation config: {args.degradation_config}")
    with open(args.degradation_config) as f:
        deg_config = json.load(f)

    print(f"  ✓ Config loaded: {deg_config.get('name', 'unnamed')}")

    # Build pipeline
    print("\nBuilding degradation pipeline...")
    pipeline = build_degradation_pipeline(deg_config)
    print(f"  ✓ Pipeline built with transforms")

    # Generate multiple degraded versions
    print(f"\nGenerating {args.num_samples} degraded samples...")
    degraded_samples = []

    for i in range(args.num_samples):
        # Apply pipeline (creates new degradation each time due to randomness)
        degraded_record = pipeline(record.copy())
        degraded_samples.append(degraded_record)
        print(f"  ✓ Sample {i+1} generated")

    # Create visualization
    print(f"\nCreating visualization...")

    num_cols = args.num_samples + 2  # Original + k-space + degraded samples
    fig, axes = plt.subplots(2, num_cols, figsize=(4*num_cols, 8))

    # Plot original image
    axes[0, 0].imshow(record['image'], cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    # Plot original k-space
    kspace_mag = np.log(np.abs(record['kspace']) + 1e-10)
    axes[0, 1].imshow(kspace_mag, cmap='jet')
    axes[0, 1].set_title('Original K-space\n(log magnitude)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    axes[1, 1].axis('off')

    # Plot degraded samples
    for i, degraded_record in enumerate(degraded_samples):
        col = i + 2

        # Degraded image
        axes[0, col].imshow(degraded_record['image'], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'Degraded Sample {i+1}', fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        # Degraded k-space
        if degraded_record.get('kspace') is not None:
            kspace_deg_mag = np.log(np.abs(degraded_record['kspace']) + 1e-10)
            axes[1, col].imshow(kspace_deg_mag, cmap='jet')
            axes[1, col].set_title(f'K-space (degraded)', fontsize=10)
            axes[1, col].axis('off')
        else:
            axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: {args.output}")

    # Print applied degradations
    print("\nApplied degradations (Sample 1):")
    for deg in degraded_samples[0]['metadata'].get('applied', []):
        print(f"  - {deg['transform']}")
        # Print some parameters
        for key, value in deg.items():
            if key != 'transform':
                if isinstance(value, float):
                    print(f"      {key}: {value:.4f}")
                else:
                    print(f"      {key}: {value}")

    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    visualize()
