"""
Hybrid MRI Dataset Loader

Supports:
- NPY files (fast path)
- DICOM files (with caching)
- PNG/JPG images
- On-the-fly degradation during training
- Patch extraction
- Spatial augmentation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import sys

# Add MriWizard to path
mriwizard_path = '/mnt/host/c/Users/Yousef Nasr/Desktop/Ai Projects/mri/swinMR_final/MriWizard/swinmr_mriwizard/MriWizard'
if mriwizard_path not in sys.path:
    sys.path.insert(0, mriwizard_path)

from MriWizard.io.dicom_loader import LoadDICOM
from MriWizard.io.image_loader import LoadImage
from MriWizard.core.utils import fft2c, normalize
from MriWizard.spatial import RandomFlip, Crop, Pad, CropOrPad, Resize, RandomCrop, RandomResize
from .pipeline_builder import build_degradation_pipeline


class HybridMRIDataset(Dataset):
    """
    Hybrid dataloader supporting NPY/DICOM with on-the-fly degradation

    Args:
        data_dir: Directory containing NPY or DICOM files
        degradation_config: Either path to JSON config or loaded config dict
        patch_size: Extract random patches (None = full image)
        cache_dir: Directory for cached DICOM→NPY conversions
        use_augmentation: Apply spatial augmentation
        return_metadata: Return metadata dict with samples

    Returns:
        input_tensor: Degraded image (1, H, W)
        target_tensor: Reference image (1, H, W)
        metadata: Dictionary with degradation info (if return_metadata=True)
    """

    def __init__(
        self,
        data_dir: str,
        degradation_config,  # Path or dict
        patch_size: int = None,
        cache_dir: str = "./cache",
        use_augmentation: bool = True,  # Deprecated: use config instead
        return_metadata: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir)
        self.return_metadata = return_metadata

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load degradation config if path provided
        if isinstance(degradation_config, (str, Path)):
            with open(degradation_config) as f:
                degradation_config = json.load(f)

        # Store full degradation config
        self.degradation_config = degradation_config

        # Parse augmentation config from degradation config
        aug_config = degradation_config.get("augmentation", {})
        self.use_augmentation = aug_config.get("enabled", use_augmentation)
        self.aug_config = aug_config

        # Check if deterministic mode is enabled
        execution_config = degradation_config.get("execution", {})
        self.deterministic = execution_config.get("deterministic", False)
        self.base_seed = execution_config.get("seed", None)

        # Build MriWizard pipeline
        self.pipeline = build_degradation_pipeline(degradation_config)

        # Scan for files
        self.files = self._scan_files()

        if len(self.files) == 0:
            raise ValueError(f"No valid files found in {self.data_dir}")

        # Setup loaders
        self.dicom_loader = LoadDICOM(convert_to_kspace=True)
        self.image_loader = LoadImage(convert_to_kspace=True, grayscale=True)

        print(f"Hybrid MRI Dataset: Found {len(self.files)} files in {self.data_dir}")

    def _scan_files(self):
        """Scan directory for NPY/DICOM/PNG/JPG files"""
        files = []

        # Scan for different file types
        for ext in ['*.npy', '*.dcm', '*.png', '*.jpg', '*.jpeg']:
            files.extend(list(self.data_dir.glob(f"**/{ext}")))

        # Sort for reproducibility
        files.sort()

        return files

    def _load_file(self, file_path: Path):
        """Load file with appropriate loader"""
        suffix = file_path.suffix.lower()

        if suffix == ".npy":
            # Fast path: direct NPY load
            image = np.load(file_path).astype(np.float32)

            # Ensure 2D
            if image.ndim == 3:
                if image.shape[0] == 1:
                    image = image[0]
                elif image.shape[-1] == 1:
                    image = image[..., 0]
                else:
                    # Take middle slice if 3D volume
                    image = image[image.shape[0] // 2]

            # Normalize if needed
            if image.max() > 1.0:
                image = normalize(image)

            # Create record
            record = {
                "kspace": None,
                "image": image,
                "mask": None,
                "metadata": {
                    "source": str(file_path),
                    "format": "npy",
                    "applied": []  # Track applied degradations
                }
            }

            # Convert to k-space
            record["kspace"] = fft2c(image)

        elif suffix == ".dcm":
            # DICOM: check cache first
            cache_path = self.cache_dir / file_path.relative_to(self.data_dir).with_suffix(".npy")

            if cache_path.exists():
                # Load from cache
                image = np.load(cache_path).astype(np.float32)
                record = {
                    "kspace": fft2c(image),
                    "image": image,
                    "mask": None,
                    "metadata": {
                        "source": str(file_path),
                        "cached": True,
                        "format": "dicom",
                        "applied": []  # Track applied degradations
                    }
                }
            else:
                # Load DICOM and cache
                record = self.dicom_loader.load(str(file_path))

                # Ensure metadata has required fields
                if "metadata" not in record:
                    record["metadata"] = {}
                record["metadata"]["cached"] = False
                record["metadata"]["format"] = "dicom"
                record["metadata"]["applied"] = []  # Track applied degradations

                # Save to cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, record["image"])

        elif suffix in [".png", ".jpg", ".jpeg"]:
            # Image file
            cache_path = self.cache_dir / file_path.relative_to(self.data_dir).with_suffix(".npy")

            if cache_path.exists():
                # Load from cache
                image = np.load(cache_path).astype(np.float32)
                record = {
                    "kspace": fft2c(image),
                    "image": image,
                    "mask": None,
                    "metadata": {
                        "source": str(file_path),
                        "cached": True,
                        "format": "image",
                        "applied": []  # Track applied degradations
                    }
                }
            else:
                # Load image and cache
                record = self.image_loader.load(str(file_path))

                # Ensure metadata has required fields
                if "metadata" not in record:
                    record["metadata"] = {}
                record["metadata"]["cached"] = False
                record["metadata"]["format"] = "image"
                record["metadata"]["applied"] = []  # Track applied degradations

                # Save to cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, record["image"])

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return record

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            If return_metadata=True:
                (input_tensor, target_tensor, metadata)
            Else:
                (input_tensor, target_tensor)

        Processing order:
            1. Load image
            2. Extract patch (if specified)
            3. Apply spatial augmentation (BEFORE degradation) - same transform to GT
            4. Apply degradation pipeline (creates input from augmented GT)
        """
        # Load file
        record = self._load_file(self.files[idx])

        # Get clean image (this will be our target)
        target_image = record["image"].copy()

        # Ensure image is valid
        if target_image is None:
            raise ValueError(f"Invalid image data for file: {self.files[idx]}")

        # Extract patch from GT image if specified
        if self.patch_size is not None:
            target_image = self._extract_patch_single(target_image)

        # Apply spatial augmentation to GT image BEFORE degradation
        # This ensures both input and target have the same augmentation
        if self.use_augmentation:
            target_image, aug_params = self._augment_single(target_image)
        else:
            aug_params = None

        # Update record with augmented image for degradation
        record["image"] = target_image.copy()

        # Recompute k-space from augmented image
        record["kspace"] = fft2c(target_image)

        # Set deterministic seed if enabled (for reproducible validation)
        if self.deterministic and self.base_seed is not None:
            # Use idx to ensure each sample gets consistent but different degradation
            sample_seed = self.base_seed + idx
            np.random.seed(sample_seed)
            import random
            random.seed(sample_seed)
            try:
                import torch
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
            except ImportError:
                pass

        # Apply degradation pipeline to augmented image
        degraded_record = self.pipeline(record)
        input_image = degraded_record["image"]

        # Ensure degraded image is valid
        if input_image is None:
            raise ValueError(f"Invalid degraded image for file: {self.files[idx]}")

        # Ensure 2D
        if input_image.ndim != 2 or target_image.ndim != 2:
            raise ValueError(f"Expected 2D images, got input: {input_image.shape}, target: {target_image.shape}")

        # CRITICAL: Ensure both input and target use the SAME normalization
        # The degradation pipeline normalizes the input via IFFTReconstruct(normalize=True)
        # which uses 99th percentile normalization.
        # 
        # IMPORTANT: We must normalize BOTH images using the SAME reference (the original GT)
        # to maintain consistent intensity scaling between input and target.
        # 
        # Strategy: Use the target's 99th percentile to normalize BOTH images
        # This ensures the target is properly normalized AND the input maintains
        # the correct relative intensity relationship to the target.
        
        # Calculate normalization factor from the ground truth (target)
        target_max = np.percentile(target_image, 99)
        
        if target_max > 0:
            # Normalize both using the same reference (target's 99th percentile)
            target_image = np.clip(target_image / target_max, 0, 1)
            input_image = np.clip(input_image / target_max, 0, 1)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_image).unsqueeze(0).float()  # (1, H, W)
        target_tensor = torch.from_numpy(target_image).unsqueeze(0).float()  # (1, H, W)

        if self.return_metadata:
            # Add augmentation info to metadata
            if aug_params:
                degraded_record["metadata"]["augmentation"] = aug_params
            return input_tensor, target_tensor, degraded_record["metadata"]
        else:
            return input_tensor, target_tensor

    def _extract_patch_single(self, img):
        """Extract random patch from single image"""
        h, w = img.shape
        ps = self.patch_size

        if h < ps or w < ps:
            # Pad if image is smaller than patch size
            pad_h = max(ps - h, 0)
            pad_w = max(ps - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape

        # Random top-left corner
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)

        patch = img[top:top+ps, left:left+ps]

        return patch

    def _augment_single(self, img):
        """
        Apply spatial augmentation using MriWizard spatial transforms
        IMPORTANT: This is applied to the GROUND TRUTH first, before degradation

        Supports:
        - Sequential transforms (apply all enabled transforms)
        - One-of strategy (randomly select one transform from a group)
        - All MriWizard spatial transforms: Flip, Crop, Pad, Resize, etc.

        Args:
            img: Ground truth image (2D numpy array)

        Returns:
            augmented_img: Augmented ground truth image
            aug_params: Dictionary with augmentation parameters applied
        """
        # Create a MriWizard record for the ground truth image
        record = {
            "image": img.copy(),
            "kspace": None,  # Will be computed later
            "mask": None,
            "metadata": {
                "applied": []
            }
        }

        # Get augmentation strategy (default: sequential)
        strategy = self.aug_config.get("strategy", "sequential")

        if strategy == "one_of":
            # ONE-OF STRATEGY: Pick one transform randomly
            record = self._apply_one_of_augmentation(record)
        else:
            # SEQUENTIAL STRATEGY: Apply all enabled transforms in sequence
            record = self._apply_sequential_augmentation(record)

        # Extract augmented image
        augmented_img = record["image"]

        # Build summary of applied augmentations for metadata
        aug_params = {
            "transforms_applied": record["metadata"]["applied"],
            "num_transforms": len(record["metadata"]["applied"]),
            "strategy": strategy
        }

        return augmented_img, aug_params

    def _apply_sequential_augmentation(self, record):
        """Apply all enabled augmentation transforms sequentially"""

        # 1. Horizontal Flip
        h_flip_cfg = self.aug_config.get("horizontal_flip", {})
        if h_flip_cfg.get("enabled", False):
            prob = h_flip_cfg.get("probability", 0.5)
            transform = RandomFlip(axes=1, flip_probability=prob, apply_to="image")
            record = transform(record)

        # 2. Vertical Flip
        v_flip_cfg = self.aug_config.get("vertical_flip", {})
        if v_flip_cfg.get("enabled", False):
            prob = v_flip_cfg.get("probability", 0.5)
            transform = RandomFlip(axes=0, flip_probability=prob, apply_to="image")
            record = transform(record)

        # 3. Rotation 90 degrees
        rot_cfg = self.aug_config.get("rotation_90", {})
        if rot_cfg.get("enabled", False):
            prob = rot_cfg.get("probability", 0.5)
            if np.random.rand() < prob:
                angles = rot_cfg.get("angles", [90, 180, 270])
                k_values = [angle // 90 for angle in angles]
                k = np.random.choice(k_values)
                if k > 0:
                    record["image"] = np.rot90(record["image"], k).copy()
                    record["metadata"]["applied"].append({
                        "transform": "Rotation90",
                        "rotation_k": int(k),
                        "rotation_angle": int(k * 90)
                    })

        # 4. Random Crop
        crop_cfg = self.aug_config.get("random_crop", {})
        if crop_cfg.get("enabled", False):
            prob = crop_cfg.get("probability", 0.5)
            if np.random.rand() < prob:
                # Support both old and new config formats
                if "crop_range" in crop_cfg:
                    # New range-based format: (min, max) or ((min_h, max_h), (min_w, max_w))
                    crop_range = crop_cfg["crop_range"]
                    transform = RandomCrop(cropping_range=crop_range, apply_to="image")
                else:
                    # Legacy format: single crop_size value
                    crop_size = crop_cfg.get("crop_size", 10)
                    h, w = record["image"].shape
                    crop_vals = np.random.randint(0, crop_size, size=4)
                    transform = Crop(cropping=tuple(crop_vals), apply_to="image")
                record = transform(record)

        # 5. Padding
        pad_cfg = self.aug_config.get("padding", {})
        if pad_cfg.get("enabled", False):
            prob = pad_cfg.get("probability", 0.5)
            if np.random.rand() < prob:
                pad_size = pad_cfg.get("pad_size", 10)
                padding_mode = pad_cfg.get("padding_mode", "reflect")
                transform = Pad(padding=pad_size, padding_mode=padding_mode, apply_to="image")
                record = transform(record)

        # 6. Resize
        resize_cfg = self.aug_config.get("resize", {})
        if resize_cfg.get("enabled", False):
            prob = resize_cfg.get("probability", 0.5)
            if np.random.rand() < prob:
                interpolation = resize_cfg.get("interpolation", 1)
                # Support both old and new config formats
                if "size_range" in resize_cfg:
                    # New range-based format: (min, max) or ((min_h, max_h), (min_w, max_w))
                    size_range = resize_cfg["size_range"]
                    transform = RandomResize(
                        size_range=size_range,
                        interpolation=interpolation,
                        apply_to="image"
                    )
                else:
                    # Legacy format: fixed target_shape
                    target_shape = resize_cfg.get("target_shape", (256, 256))
                    transform = Resize(
                        target_shape=target_shape,
                        interpolation=interpolation,
                        apply_to="image"
                    )
                record = transform(record)

        return record

    def _apply_one_of_augmentation(self, record):
        """Apply ONE randomly selected transform from available options"""

        # Build list of available transforms with their weights
        available_transforms = []

        # 0. Identity (no transform)
        identity_cfg = self.aug_config.get("identity", {})
        if identity_cfg.get("enabled", False):
            weight = identity_cfg.get("weight", 0.1)
            available_transforms.append({
                "name": "identity",
                "weight": weight,
                "transform": lambda r: r  # No transformation
            })

        # 1. Horizontal Flip
        h_flip_cfg = self.aug_config.get("horizontal_flip", {})
        if h_flip_cfg.get("enabled", False):
            weight = h_flip_cfg.get("weight", 1.0)
            prob = h_flip_cfg.get("probability", 0.5)
            available_transforms.append({
                "name": "horizontal_flip",
                "weight": weight,
                "transform": lambda r: RandomFlip(axes=1, flip_probability=prob, apply_to="image")(r)
            })

        # 2. Vertical Flip
        v_flip_cfg = self.aug_config.get("vertical_flip", {})
        if v_flip_cfg.get("enabled", False):
            weight = v_flip_cfg.get("weight", 1.0)
            prob = v_flip_cfg.get("probability", 0.5)
            available_transforms.append({
                "name": "vertical_flip",
                "weight": weight,
                "transform": lambda r: RandomFlip(axes=0, flip_probability=prob, apply_to="image")(r)
            })

        # 3. Rotation 90
        rot_cfg = self.aug_config.get("rotation_90", {})
        if rot_cfg.get("enabled", False):
            weight = rot_cfg.get("weight", 1.0)
            def apply_rotation(r):
                angles = rot_cfg.get("angles", [90, 180, 270])
                k_values = [angle // 90 for angle in angles]
                k = np.random.choice(k_values)
                if k > 0:
                    r["image"] = np.rot90(r["image"], k).copy()
                    r["metadata"]["applied"].append({
                        "transform": "Rotation90",
                        "rotation_k": int(k),
                        "rotation_angle": int(k * 90)
                    })
                return r
            available_transforms.append({
                "name": "rotation_90",
                "weight": weight,
                "transform": apply_rotation
            })

        # 4. Random Crop
        crop_cfg = self.aug_config.get("random_crop", {})
        if crop_cfg.get("enabled", False):
            weight = crop_cfg.get("weight", 1.0)
            def apply_crop(r):
                # Support both old and new config formats
                if "crop_range" in crop_cfg:
                    # New range-based format
                    crop_range = crop_cfg["crop_range"]
                    transform = RandomCrop(cropping_range=crop_range, apply_to="image")
                else:
                    # Legacy format
                    crop_size = crop_cfg.get("crop_size", 10)
                    h, w = r["image"].shape
                    crop_vals = np.random.randint(0, crop_size, size=4)
                    transform = Crop(cropping=tuple(crop_vals), apply_to="image")
                return transform(r)
            available_transforms.append({
                "name": "random_crop",
                "weight": weight,
                "transform": apply_crop
            })

        # 5. Padding
        pad_cfg = self.aug_config.get("padding", {})
        if pad_cfg.get("enabled", False):
            weight = pad_cfg.get("weight", 1.0)
            def apply_pad(r):
                pad_size = pad_cfg.get("pad_size", 10)
                padding_mode = pad_cfg.get("padding_mode", "reflect")
                transform = Pad(padding=pad_size, padding_mode=padding_mode, apply_to="image")
                return transform(r)
            available_transforms.append({
                "name": "padding",
                "weight": weight,
                "transform": apply_pad
            })

        # 6. Resize
        resize_cfg = self.aug_config.get("resize", {})
        if resize_cfg.get("enabled", False):
            weight = resize_cfg.get("weight", 1.0)
            def apply_resize(r):
                interpolation = resize_cfg.get("interpolation", 1)
                # Support both old and new config formats
                if "size_range" in resize_cfg:
                    # New range-based format
                    size_range = resize_cfg["size_range"]
                    transform = RandomResize(size_range=size_range, interpolation=interpolation, apply_to="image")
                else:
                    # Legacy format
                    target_shape = resize_cfg.get("target_shape", (256, 256))
                    transform = Resize(target_shape=target_shape, interpolation=interpolation, apply_to="image")
                return transform(r)
            available_transforms.append({
                "name": "resize",
                "weight": weight,
                "transform": apply_resize
            })

        # If no transforms available, return unchanged
        if not available_transforms:
            return record

        # Select one transform based on weights
        weights = [t["weight"] for t in available_transforms]
        weights_sum = sum(weights)
        probabilities = [w / weights_sum for w in weights]

        selected_idx = np.random.choice(len(available_transforms), p=probabilities)
        selected = available_transforms[selected_idx]

        # Apply selected transform
        record = selected["transform"](record)

        # Add info about one_of selection
        record["metadata"]["one_of_selected"] = selected["name"]

        return record

    # Keep old methods for backward compatibility (not used anymore)
    def _extract_patch(self, input_img, target_img):
        """DEPRECATED: Extract random patch from both images"""
        h, w = input_img.shape
        ps = self.patch_size

        if h < ps or w < ps:
            # Pad if image is smaller than patch size
            pad_h = max(ps - h, 0)
            pad_w = max(ps - w, 0)
            input_img = np.pad(input_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            target_img = np.pad(target_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = input_img.shape

        # Random top-left corner
        top = np.random.randint(0, h - ps + 1)
        left = np.random.randint(0, w - ps + 1)

        input_patch = input_img[top:top+ps, left:left+ps]
        target_patch = target_img[top:top+ps, left:left+ps]

        return input_patch, target_patch

    def _augment(self, input_img, target_img):
        """DEPRECATED: Apply spatial augmentation consistently to both images"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img).copy()
            target_img = np.fliplr(target_img).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img).copy()
            target_img = np.flipud(target_img).copy()

        # Random 90-degree rotations
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            input_img = np.rot90(input_img, k).copy()
            target_img = np.rot90(target_img, k).copy()

        return input_img, target_img


def collate_fn_with_metadata(batch):
    """
    Custom collate function to handle metadata

    Args:
        batch: List of (input, target, metadata) tuples

    Returns:
        inputs: Batched input tensors (B, 1, H, W)
        targets: Batched target tensors (B, 1, H, W)
        metadata: List of metadata dicts
    """
    inputs, targets, metadata = zip(*batch)

    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)

    return inputs, targets, list(metadata)


if __name__ == '__main__':
    # Test the dataloader
    import sys
    from torch.utils.data import DataLoader

    if len(sys.argv) < 3:
        print("Usage: python dataloader.py <data_dir> <degradation_config>")
        sys.exit(1)

    data_dir = sys.argv[1]
    deg_config = sys.argv[2]

    print("="*60)
    print("Testing Hybrid MRI Dataloader")
    print("="*60)

    # Create dataset
    dataset = HybridMRIDataset(
        data_dir=data_dir,
        degradation_config=deg_config,
        patch_size=256,
        use_augmentation=True,
        return_metadata=True
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    print("\nLoading single sample...")
    input_img, target_img, metadata = dataset[0]
    print(f"Input shape: {input_img.shape}")
    print(f"Target shape: {target_img.shape}")
    print(f"Metadata: {metadata['source']}")
    print(f"Applied degradations: {len(metadata.get('applied', []))}")

    # Test dataloader
    print("\nTesting DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        collate_fn=collate_fn_with_metadata
    )

    inputs, targets, metadata_list = next(iter(loader))
    print(f"Batch inputs shape: {inputs.shape}")
    print(f"Batch targets shape: {targets.shape}")
    print(f"Batch metadata length: {len(metadata_list)}")

    print("\n✓ Dataloader test passed!")
