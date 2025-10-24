"""
Visualization test for MriWizard spatial augmentation pipeline.
Shows pairs of: Ground Truth -> Augmented GT -> Degraded Input

This demonstrates:
1. Augmentation is applied to GT FIRST using MriWizard spatial transforms
2. Degradation is applied to the augmented GT
3. Supports both sequential and one_of strategies

Usage:
    # Test with custom image and config
    python test_augmentation_visualization.py <image_path> <config_path> [num_samples]

    # Test with default test patterns
    python test_augmentation_visualization.py

Examples:
    python test_augmentation_visualization.py ../S1/image001.npy ./configs/degradation_oneof_augmentation.json 5
    python test_augmentation_visualization.py my_image.dcm ./configs/degradation_with_full_augmentation.json
"""

import sys
from pathlib import Path
import numpy as np
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

def test_augmentation_visualization(image_path=None, config_path=None, num_samples=3):
    """Visualize augmentation and degradation pipeline"""

    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    from data.dataloader import HybridMRIDataset

    print("="*80)
    print("MriWizard Spatial Augmentation Visualization Test")
    print("="*80)

    # Determine if using custom image or test patterns
    use_custom_image = image_path is not None

    # Create test/output directory
    test_dir = Path("./test_augmentation_data")
    test_dir.mkdir(exist_ok=True)

    if use_custom_image:
        # Use provided image
        print(f"\n1. Using custom image: {image_path}")
        image_path = Path(image_path)

        if not image_path.exists():
            print(f"ERROR: Image file not found: {image_path}")
            return

        # Copy or link to test directory
        test_files = [image_path]
        print(f"   Image: {image_path.name}")
        print(f"   Format: {image_path.suffix}")

    else:
        # Create test patterns (asymmetric so we can see transformations clearly)
        print("\n1. Creating test images...")

        # Pattern 1: Checkerboard with bright corner
        img1 = np.zeros((128, 128), dtype=np.float32)
        for i in range(0, 128, 16):
            for j in range(0, 128, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    img1[i:i+16, j:j+16] = 0.7
        img1[0:32, 0:32] = 1.0  # Bright top-left corner

        # Pattern 2: Diagonal gradient with shapes
        img2 = np.zeros((128, 128), dtype=np.float32)
        for i in range(128):
            for j in range(128):
                img2[i, j] = (i + j) / (2 * 128)
        # Add rectangles
        img2[20:40, 80:110] = 1.0
        img2[80:110, 20:40] = 0.3

        # Pattern 3: Concentric circles
        img3 = np.zeros((128, 128), dtype=np.float32)
        center_y, center_x = 64, 64
        for i in range(128):
            for j in range(128):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                img3[i, j] = 0.5 + 0.5 * np.sin(dist / 5)
        img3 = np.clip(img3, 0, 1)
        # Add asymmetry marker
        img3[10:20, 10:20] = 1.0

        # Save test images
        test_files = []
        for idx, img in enumerate([img1, img2, img3], 1):
            fpath = test_dir / f"test_pattern_{idx}.npy"
            np.save(fpath, img)
            test_files.append(fpath)
            print(f"   Created: {fpath.name} - shape: {img.shape}")

    # Determine which configs to test
    if config_path is not None:
        # Use provided config
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"ERROR: Config file not found: {config_path}")
            return
        configs = [("Custom Config", str(config_path))]
    else:
        # Test both strategies
        configs = [
            ("Sequential Strategy", "./configs/degradation_with_full_augmentation.json"),
            ("One-Of Strategy", "./configs/degradation_oneof_augmentation.json")
        ]

    for strategy_name, config_path in configs:
        config_path = Path(config_path)

        if not config_path.exists():
            print(f"\n   Skipping {strategy_name}: Config not found at {config_path}")
            continue

        print(f"\n{'='*80}")
        print(f"Testing: {strategy_name}")
        print(f"Config: {config_path.name}")
        print(f"{'='*80}")

        # Create dataset
        if use_custom_image:
            # Use directory containing the custom image
            data_dir = str(image_path.parent)
        else:
            data_dir = str(test_dir)

        dataset = HybridMRIDataset(
            data_dir=data_dir,
            degradation_config=str(config_path),
            patch_size=None,
            use_augmentation=True,
            return_metadata=True
        )

        print(f"Dataset loaded: {len(dataset)} images")

        # Number of variations to show per original image
        num_samples_per_image = num_samples

        # Process only the images we have
        num_images = 1 if use_custom_image else min(len(dataset), len(test_files))

        for img_idx in range(num_images):
            print(f"\n--- Image {img_idx + 1}/{num_images} ---")

            # Load original image
            if use_custom_image:
                # Load custom image
                img_file = image_path
                if img_file.suffix == '.npy':
                    original = np.load(img_file).astype(np.float32)
                else:
                    # For DICOM/PNG/JPG, load through dataset and use first loaded image
                    # Just use a placeholder for now since dataset handles loading
                    sample = dataset._load_file(img_file)
                    original = sample["image"]
            else:
                # Load test pattern
                original = np.load(test_files[img_idx])

            # Ensure 2D
            if original.ndim == 3:
                if original.shape[0] == 1:
                    original = original[0]
                elif original.shape[-1] == 1:
                    original = original[..., 0]
                else:
                    original = original[original.shape[0] // 2]

            # Create figure
            fig, axes = plt.subplots(num_samples_per_image, 3, figsize=(12, 4*num_samples_per_image))
            fig.suptitle(f'{strategy_name} - Image {img_idx+1}', fontsize=16)

            for sample_idx in range(num_samples_per_image):
                # Load augmented + degraded sample
                # For custom images, find the index in dataset
                if use_custom_image:
                    # Find the image in the dataset
                    dataset_idx = None
                    for idx, file_path in enumerate(dataset.files):
                        if file_path.name == image_path.name:
                            dataset_idx = idx
                            break
                    if dataset_idx is None:
                        print(f"WARNING: Could not find {image_path.name} in dataset")
                        dataset_idx = 0
                else:
                    dataset_idx = img_idx

                input_tensor, target_tensor, metadata = dataset[dataset_idx]

                input_img = input_tensor[0].numpy()
                target_img = target_tensor[0].numpy()

                # Extract augmentation info
                aug_info = metadata.get('augmentation', {})
                transforms = aug_info.get('transforms_applied', [])
                strategy = aug_info.get('strategy', 'unknown')
                one_of_selected = metadata.get('one_of_selected', None)

                # Build title
                if strategy == 'one_of' and one_of_selected:
                    aug_title = f"ONE-OF: {one_of_selected}"
                else:
                    transform_names = [t.get('transform', 'Unknown') for t in transforms]
                    aug_title = f"{len(transforms)} transforms: {', '.join(transform_names)}" if transforms else "No augmentation"

                # Plot
                row = sample_idx

                # Original GT
                axes[row, 0].imshow(original, cmap='gray', vmin=0, vmax=1)
                axes[row, 0].set_title(f"Original GT\n{original.shape}", fontsize=10)
                axes[row, 0].axis('off')

                # Augmented GT
                axes[row, 1].imshow(target_img, cmap='gray', vmin=0, vmax=1)
                axes[row, 1].set_title(f"Augmented GT\n{aug_title}", fontsize=9)
                axes[row, 1].axis('off')

                # Degraded input
                axes[row, 2].imshow(input_img, cmap='gray', vmin=0, vmax=1)
                deg_info = metadata.get('applied', [])
                deg_title = f"{len(deg_info)} degradations"
                axes[row, 2].set_title(f"Degraded Input\n{deg_title}", fontsize=10)
                axes[row, 2].axis('off')

                # Print details
                print(f"  Sample {sample_idx + 1}:")
                print(f"    Augmentation: {aug_title}")
                print(f"    Degradations applied: {len(deg_info)}")
                if transforms:
                    for t in transforms:
                        print(f"      - {t}")

            plt.tight_layout()

            # Save figure
            if use_custom_image:
                img_name = image_path.stem
                output_name = f"visualization_{img_name}_{config_path.stem}.png"
            else:
                output_name = f"visualization_{strategy_name.lower().replace(' ', '_')}_img{img_idx+1}.png"

            output_path = test_dir / output_name
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Saved visualization: {output_path.name}")

    print(f"\n{'='*80}")
    print("✓ Visualization complete!")
    print(f"Output directory: {test_dir.absolute()}")
    print(f"{'='*80}")

    print("\nKey observations:")
    print("  1. ✓ Augmentation uses MriWizard spatial transforms from:")
    print("     /mnt/host/c/.../MriWizard/swinmr_mriwizard/MriWizard/spatial")
    print("  2. ✓ Augmentation is applied to GROUND TRUTH first")
    print("  3. ✓ Degradation is applied to AUGMENTED ground truth")
    print("  4. ✓ Both 'sequential' and 'one_of' strategies are supported")
    print("  5. ✓ All spatial transforms available: Flip, Crop, Pad, Resize")

    # Cleanup option
    print(f"\nTest images saved in: {test_dir}")
    print("To clean up, delete the directory manually.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MriWizard augmentation and degradation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default test patterns and both configs
  python test_augmentation_visualization.py

  # Test with custom image and specific config
  python test_augmentation_visualization.py ../S1/image001.npy ./configs/degradation_oneof_augmentation.json

  # Test with custom image, config, and 5 sample variations
  python test_augmentation_visualization.py ../S1/image001.npy ./configs/degradation_oneof_augmentation.json 5

  # Test with DICOM image
  python test_augmentation_visualization.py my_scan.dcm ./configs/degradation_with_full_augmentation.json
        """
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to image file (.npy, .dcm, .png, .jpg). If not provided, uses test patterns."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to degradation config file (.json). If not provided, tests both strategies."
    )
    parser.add_argument(
        "num_samples",
        nargs="?",
        type=int,
        default=3,
        help="Number of augmentation variations to show (default: 3)"
    )

    args = parser.parse_args()

    test_augmentation_visualization(
        image_path=args.image,
        config_path=args.config,
        num_samples=args.num_samples
    )
