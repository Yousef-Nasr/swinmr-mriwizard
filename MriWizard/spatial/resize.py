"""Resize transform for MRI images.

Adapted from TorchIO's Resize transform for compatibility with MriWizard's record-based architecture.
This uses interpolation to change the image size.
"""

import numpy as np
from typing import Union, Tuple
from scipy.ndimage import zoom
from MriWizard.core.base import Record


class Resize:
    """Resize an image to a target shape using interpolation.
    
    This transform resizes the image by resampling with interpolation. Both k-space
    and image domain data can be resized. Note that resizing in k-space is generally
    not recommended as it can introduce artifacts.
    
    Args:
        target_shape: Target shape for the image.
              - If int n: target shape is (n, n) for 2D or (n, n, n) for 3D
              - If tuple: target shape for each dimension
              For 2D: (height, width)
              For 3D: (depth, height, width)
              - Use -1 to keep the original size for that dimension
        
        interpolation: Interpolation order for scipy.ndimage.zoom:
              - 0: Nearest-neighbor (fastest, no smoothing)
              - 1: Linear/bilinear (good balance)
              - 2: Quadratic
              - 3: Cubic (smoothest, slowest)
              Default: 1 (linear)
        
        apply_to: What to resize - "image" or "kspace" (not "both" as it's unusual to resize k-space).
              Default: "image"
        
        anti_aliasing: If True, apply Gaussian smoothing before downsampling to reduce aliasing.
              Default: True
    
    Example:
        >>> # Resize to 256x256
        >>> resize = Resize(target_shape=(256, 256))
        >>> 
        >>> # Resize to 128x128x128 (3D)
        >>> resize = Resize(target_shape=(128, 128, 128))
        >>>
        >>> # Resize height to 256, keep width unchanged
        >>> resize = Resize(target_shape=(256, -1))
        >>>
        >>> # Use cubic interpolation
        >>> resize = Resize(target_shape=(512, 512), interpolation=3)
    
    Warning:
        - Resizing changes the pixel spacing and may introduce interpolation artifacts
        - For medical imaging, consider using CropOrPad instead to preserve physical dimensions
        - Resizing k-space directly is generally not recommended
        - Anti-aliasing is important when downsampling to prevent aliasing artifacts
    """
    
    def __init__(
        self,
        target_shape: Union[int, Tuple[int, ...]],
        interpolation: int = 1,
        apply_to: str = "image",
        anti_aliasing: bool = True
    ):
        """Initialize the Resize transform."""
        self.target_shape = self._parse_target_shape(target_shape)
        
        if not isinstance(interpolation, int) or not 0 <= interpolation <= 5:
            raise ValueError(f"interpolation must be an integer between 0 and 5, got {interpolation}")
        self.interpolation = interpolation
        
        if apply_to not in ("image", "kspace"):
            raise ValueError(f"apply_to must be 'image' or 'kspace', got {apply_to}")
        self.apply_to = apply_to
        
        self.anti_aliasing = anti_aliasing
    
    def _parse_target_shape(self, target_shape: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """Parse target shape parameter.
        
        Args:
            target_shape: Target shape specification.
            
        Returns:
            Tuple of target dimensions (can contain -1 for unchanged dimensions).
        """
        if isinstance(target_shape, int):
            return (target_shape,)
        
        if isinstance(target_shape, (tuple, list)):
            return tuple(int(s) for s in target_shape)
        
        raise ValueError(f"target_shape must be int or tuple, got {type(target_shape)}")
    
    def _expand_target_shape(
        self,
        target_shape: Tuple[int, ...],
        current_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Expand target shape to match data dimensionality and handle -1 values.
        
        Args:
            target_shape: Parsed target shape (can contain -1).
            current_shape: Current shape of the data.
            
        Returns:
            Tuple of target dimensions matching the data dimensionality.
        """
        ndim = len(current_shape)
        
        if len(target_shape) == 1:
            # Single value: replicate for all dimensions (unless it's -1)
            n = target_shape[0]
            if n == -1:
                return current_shape
            return tuple([n] * ndim)
        
        if len(target_shape) == ndim:
            # Replace -1 with current dimension size
            return tuple(
                current if target == -1 else target
                for target, current in zip(target_shape, current_shape)
            )
        
        raise ValueError(
            f"target_shape length ({len(target_shape)}) doesn't match data dimensionality ({ndim})"
        )
    
    def _compute_zoom_factors(
        self,
        current_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...]
    ) -> Tuple[float, ...]:
        """Compute zoom factors for each dimension.
        
        Args:
            current_shape: Current shape of the data.
            target_shape: Target shape.
            
        Returns:
            Tuple of zoom factors for each dimension.
        """
        return tuple(
            target / current
            for target, current in zip(target_shape, current_shape)
        )
    
    def _apply_anti_aliasing(self, data: np.ndarray, zoom_factors: Tuple[float, ...]) -> np.ndarray:
        """Apply Gaussian smoothing before downsampling to reduce aliasing.
        
        Args:
            data: Input array.
            zoom_factors: Zoom factors for each dimension.
            
        Returns:
            Smoothed array.
        """
        from scipy.ndimage import gaussian_filter
        
        # Only apply smoothing for dimensions that are being downsampled
        sigmas = []
        for zoom_factor in zoom_factors:
            if zoom_factor < 1.0:
                # Downsample: apply smoothing
                # Sigma is proportional to the downsampling factor
                sigma = (1.0 / zoom_factor - 1.0) / 2.0
            else:
                # Upsample or no change: no smoothing needed
                sigma = 0.0
            sigmas.append(sigma)
        
        if any(s > 0 for s in sigmas):
            # Apply Gaussian smoothing
            if np.iscomplexobj(data):
                # For complex data, smooth real and imaginary parts separately
                real_smoothed = gaussian_filter(data.real, sigma=sigmas)
                imag_smoothed = gaussian_filter(data.imag, sigma=sigmas)
                return real_smoothed + 1j * imag_smoothed
            else:
                return gaussian_filter(data, sigma=sigmas)
        
        return data
    
    def _resize_array(
        self,
        data: np.ndarray,
        target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Resize array using interpolation.
        
        Args:
            data: Input array to resize.
            target_shape: Target shape.
            
        Returns:
            Resized array.
        """
        current_shape = data.shape
        
        if current_shape == target_shape:
            return data
        
        # Compute zoom factors
        zoom_factors = self._compute_zoom_factors(current_shape, target_shape)
        
        # Apply anti-aliasing if requested
        if self.anti_aliasing:
            data = self._apply_anti_aliasing(data, zoom_factors)
        
        # Apply zoom
        if np.iscomplexobj(data):
            # For complex data, zoom real and imaginary parts separately
            real_resized = zoom(data.real, zoom_factors, order=self.interpolation)
            imag_resized = zoom(data.imag, zoom_factors, order=self.interpolation)
            resized = real_resized + 1j * imag_resized
        else:
            resized = zoom(data, zoom_factors, order=self.interpolation)
        
        return resized
    
    def __call__(self, record: Record) -> Record:
        """Apply resize to the record.
        
        Args:
            record: Input record with image and/or kspace data.
            
        Returns:
            Record with resized data to match target shape.
        """
        # Determine what to resize
        if self.apply_to == "image":
            if record["image"] is None:
                raise ValueError("Cannot resize image: image data is None")
            ref_data = record["image"]
        else:  # apply_to == "kspace"
            if record["kspace"] is None:
                raise ValueError("Cannot resize kspace: kspace data is None")
            ref_data = record["kspace"]
        
        current_shape = ref_data.shape
        
        # Expand target shape to match dimensionality and handle -1 values
        target_shape = self._expand_target_shape(self.target_shape, current_shape)
        
        # Resize the data
        if self.apply_to == "image":
            record["image"] = self._resize_array(record["image"], target_shape)
            
            # Also resize mask if present
            if record["mask"] is not None:
                # Use nearest-neighbor for mask (order=0)
                old_order = self.interpolation
                self.interpolation = 0
                record["mask"] = self._resize_array(record["mask"], target_shape)
                self.interpolation = old_order
        
        else:  # apply_to == "kspace"
            record["kspace"] = self._resize_array(record["kspace"], target_shape)
            
            # Also resize mask if present
            if record["mask"] is not None:
                old_order = self.interpolation
                self.interpolation = 0
                record["mask"] = self._resize_array(record["mask"], target_shape)
                self.interpolation = old_order
        
        # Update metadata
        record["metadata"]["applied"].append({
            "transform": "Resize",
            "target_shape": list(target_shape),
            "original_shape": list(current_shape),
            "interpolation": int(self.interpolation),
            "apply_to": self.apply_to,
            "anti_aliasing": bool(self.anti_aliasing)
        })
        
        return record

