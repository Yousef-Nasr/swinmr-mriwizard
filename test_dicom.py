#!/usr/bin/env python3
"""
Test Script for DICOM Images

Load and test DICOM images with the trained SwinMR model.
"""

import torch
import argparse
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from MriWizard.io.dicom_loader import LoadDICOM
from models.model_swinmr import ModelSwinMR
from utils.metrics import calculate_psnr, calculate_ssim
from utils.config_loader import load_config


def load_image(path, convert_to_kspace=False):
    """
    Load an image (DICOM, PNG, JPG, NPY).

    Args:
        path: Path to image file
        convert_to_kspace: Whether to convert to k-space (DICOM only)

    Returns:
        image: Numpy array [H, W]
        metadata: Dictionary with metadata
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        loader = LoadDICOM(convert_to_kspace=convert_to_kspace)
        record = loader.load(str(path))
        return record["image"], record["metadata"]

    elif suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        # Read image
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

        # Handle channels
        if img.ndim == 3:
            # Convert to grayscale if RGB/BGR
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Normalize to [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)
            # Min-max normalization if not already in [0,1]
            if img.max() > 1.0:
                img = (img - img.min()) / (img.max() - img.min())

        metadata = {
            "source": str(path),
            "modality": "Photo",
            "format": suffix[1:].upper(),
            "original_shape": img.shape,
        }

        return img, metadata

    elif suffix == ".npy":
        img = np.load(str(path))
        # Ensure float32
        img = img.astype(np.float32)
        # Normalize if needed
        if img.max() > 1.0:
            img = (img - img.min()) / (img.max() - img.min())

        metadata = {
            "source": str(path),
            "modality": "Numpy",
            "format": "NPY",
            "original_shape": img.shape,
        }
        return img, metadata

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def prepare_image_for_model(image):
    """
    Prepare image for model input.

    Args:
        image: Numpy array [H, W]

    Returns:
        Tensor [1, 1, H, W]
    """
    # Ensure float32
    image = image.astype(np.float32)

    # Add batch and channel dimensions
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    return image_tensor


def visualize_result(original, processed, metadata, save_path=None):
    """
    Visualize original and processed images side by side.

    Args:
        original: Original image tensor [1, 1, H, W]
        processed: Processed image tensor [1, 1, H, W]
        metadata: DICOM metadata dictionary
        save_path: Optional path to save figure
    """
    # Convert to numpy
    orig_np = original.squeeze().cpu().numpy()
    proc_np = processed.squeeze().cpu().numpy()

    # Clip to [0, 1]
    orig_np = np.clip(orig_np, 0, 1)
    proc_np = np.clip(proc_np, 0, 1)

    # Calculate metrics
    psnr_val = calculate_psnr(processed, original)
    ssim_val = calculate_ssim(processed, original)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original
    axes[0].imshow(orig_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original DICOM Image", fontsize=12)
    axes[0].axis("off")

    # Processed
    axes[1].imshow(proc_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Processed Image\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}", fontsize=12
    )
    axes[1].axis("off")

    # Add metadata info
    modality = metadata.get("modality", "Unknown")
    source = Path(metadata.get("source", "")).name

    plt.suptitle(
        f"DICOM Test: {source} | Modality: {modality}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved visualization to {save_path}")

    plt.show()


def test_single_image(image_path, model, device, output_dir=None):
    """
    Test a single image file (DICOM, PNG, JPG, NPY).

    Args:
        image_path: Path to image file
        model: SwinMR model
        device: torch device
        output_dir: Optional output directory for results
    """
    print(f"\n{'=' * 60}")
    print(f"Testing Image: {Path(image_path).name}")
    print(f"{'=' * 60}")

    # Load Image
    print("Loading image...")
    try:
        image, metadata = load_image(image_path, convert_to_kspace=False)
        print(f"✓ Loaded image: {image.shape} | dtype: {image.dtype}")
        print(f"  Modality: {metadata.get('modality', 'Unknown')}")
        print(f"  Value range: [{image.min():.4f}, {image.max():.4f}]")

        # Print MR parameters if available
        for key in ["TE", "TR", "flip_angle", "field_strength", "slice_thickness"]:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")

    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return

    # Prepare for model
    print("\nPreparing image for model...")
    image_tensor = prepare_image_for_model(image).to(device)
    print(f"✓ Input tensor shape: {image_tensor.shape}")

    # Run inference
    print("\nRunning model inference...")
    model.eval()
    with torch.no_grad():
        try:
            output = model.netG(image_tensor)
            print(f"✓ Output tensor shape: {output.shape}")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
        except Exception as e:
            print(f"✗ Error during inference: {e}")
            return

    # Calculate metrics
    psnr_val = calculate_psnr(output, image_tensor)
    ssim_val = calculate_ssim(output, image_tensor)

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"{'=' * 60}")

    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save processed image
        output_npy_path = output_dir / f"{Path(image_path).stem}_processed.npy"
        np.save(output_npy_path, output.squeeze().cpu().numpy())
        print(f"\n✓ Saved processed image to {output_npy_path}")

        # Save visualization
        viz_path = output_dir / f"{Path(image_path).stem}_comparison.png"
        visualize_result(image_tensor, output, metadata, save_path=viz_path)
    else:
        visualize_result(image_tensor, output, metadata)


def test_image_directory(image_dir, model, device, output_dir=None, recursive=False):
    """
    Test all image files in a directory.

    Args:
        image_dir: Directory containing image files
        model: SwinMR model
        device: torch device
        output_dir: Optional output directory for results
        recursive: Whether to search recursively
    """
    image_dir = Path(image_dir)

    # Find files
    extensions = ["*.dcm", "*.png", "*.jpg", "*.jpeg", "*.npy"]
    image_files = []

    for ext in extensions:
        if recursive:
            image_files.extend(list(image_dir.rglob(ext)))
        else:
            image_files.extend(list(image_dir.glob(ext)))

    # Also try uppercase extensions
    for ext in extensions:
        if recursive:
            image_files.extend(list(image_dir.rglob(ext.upper())))
        else:
            image_files.extend(list(image_dir.glob(ext.upper())))

    # Remove duplicates
    image_files = sorted(list(set(image_files)))

    if len(image_files) == 0:
        print(f"✗ No image files found in {image_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"Found {len(image_files)} image file(s)")
    print(f"{'=' * 60}\n")

    # Process each file
    results = []
    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            # Create output subdirectory for this file
            if output_dir:
                file_output_dir = Path(output_dir) / image_path.stem
            else:
                file_output_dir = None

            test_single_image(image_path, model, device, file_output_dir)
            results.append({"file": image_path.name, "status": "success"})

        except Exception as e:
            print(f"\n✗ Error processing {image_path.name}: {e}")
            results.append(
                {"file": image_path.name, "status": "failed", "error": str(e)}
            )

    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Test images (DICOM/PNG/JPG) with SwinMR model"
    )
    parser.add_argument("--image", type=str, help="Path to single image file")
    parser.add_argument("--dicom", type=str, help="Alias for --image")
    parser.add_argument(
        "--image-dir", type=str, help="Directory containing image files"
    )
    parser.add_argument("--dicom-dir", type=str, help="Alias for --image-dir")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.json",
        help="Path to config file",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument(
        "--recursive", action="store_true", help="Search recursively for files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Handle aliases
    if args.dicom and not args.image:
        args.image = args.dicom
    if args.dicom_dir and not args.image_dir:
        args.image_dir = args.dicom_dir

    # Check inputs
    if not args.image and not args.image_dir:
        parser.error(
            "Must provide either --image (or --dicom) or --image-dir (or --dicom-dir)"
        )

    # Setup device
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Load config
    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)

    # Initialize model
    print("Initializing model...")

    # Prepare model options (ModelBase expects full config structure)
    model_opt = {
        "netG": config["model"].copy(),  # Network architecture config
        "path": {
            "models": config["paths"]["checkpoints"],
            "log": config["paths"]["logs"],
            "samples": config["paths"]["samples"],
        },
        "gpu_ids": [0] if device.type == "cuda" else None,
        "is_train": False,  # Inference mode
        "dist": False,
        "train": {
            "G_optimizer_type": "adam",
            "G_optimizer_lr": 0.0002,
            "E_decay": 0,
            "freeze_patch_embedding": False,
        },
        "datasets": {"train": {"batch_size": 1}},
    }

    model = ModelSwinMR(model_opt)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract model weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        epoch = checkpoint.get("epoch", "unknown")
    else:
        state_dict = checkpoint
        epoch = "unknown"

    # Load into netG (the actual network)
    model.netG.load_state_dict(state_dict, strict=True)
    print(f"✓ Loaded checkpoint (epoch {epoch})")

    # Set model to evaluation mode
    model.netG.eval()

    # Test Image(s)
    if args.image:
        # Single file
        test_single_image(args.image, model, device, args.output_dir)
    else:
        # Directory of files
        test_image_directory(
            args.image_dir, model, device, args.output_dir, args.recursive
        )

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
