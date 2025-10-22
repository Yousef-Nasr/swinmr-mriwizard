"""CropOrPad transform for MRI images.

Adapted from TorchIO's CropOrPad transform for compatibility with MriWizard's record-based architecture.
"""

import numpy as np
from typing import Union, Tuple, Optional
from MriWizard.core.base import Record
from MriWizard.spatial.crop import Crop
from MriWizard.spatial.pad import Pad


class CropOrPad:
    """Crop and/or pad an image to match a target shape.
    
    This transform modifies the image size by cropping or padding (or both) to achieve
    a target shape. The cropping/padding is centered by default.
    
    Args:
        target_shape: Target shape for the image.
              - If int n: target shape is (n, n) for 2D or (n, n, n) for 3D
              - If tuple: target shape for each dimension
              For 2D: (height, width)
              For 3D: (depth, height, width)
        
        padding_mode: Padding mode (see Pad transform for details). Default: 0 (zero padding).
        
        apply_to: What to crop/pad - "image", "kspace", or "both" (default: "both").
        
        only_crop: If True, only crop (no padding). Cannot be True if only_pad is True.
        
        only_pad: If True, only pad (no crop). Cannot be True if only_crop is True.
    
    Example:
        >>> # Resize to 256x256
        >>> crop_pad = CropOrPad(target_shape=(256, 256))
        >>> 
        >>> # Resize to 128x128x128 (3D)
        >>> crop_pad = CropOrPad(target_shape=(128, 128, 128))
        >>>
        >>> # Only crop (no padding)
        >>> crop_pad = CropOrPad(target_shape=(224, 224), only_crop=True)
        >>>
        >>> # Only pad (no cropping)
        >>> crop_pad = CropOrPad(target_shape=(512, 512), only_pad=True)
    
    Note:
        - Cropping/padding is centered (equal amounts removed/added from each side)
        - If odd amounts are needed, the extra voxel goes to the end
        - Both k-space and image domain data are modified if present
        - The mask is also cropped/padded if present
    """
    
    def __init__(
        self,
        target_shape: Union[int, Tuple[int, ...]],
        padding_mode: Union[str, float] = 0,
        apply_to: str = "both",
        only_crop: bool = False,
        only_pad: bool = False
    ):
        """Initialize the CropOrPad transform."""
        self.target_shape = self._parse_target_shape(target_shape)
        self.padding_mode = padding_mode
        
        if apply_to not in ("image", "kspace", "both"):
            raise ValueError(f"apply_to must be 'image', 'kspace', or 'both', got {apply_to}")
        self.apply_to = apply_to
        
        if only_crop and only_pad:
            raise ValueError("only_crop and only_pad cannot both be True")
        self.only_crop = only_crop
        self.only_pad = only_pad
    
    def _parse_target_shape(self, target_shape: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """Parse target shape parameter.
        
        Args:
            target_shape: Target shape specification.
            
        Returns:
            Tuple of target dimensions.
        """
        if isinstance(target_shape, int):
            # Single value: will be expanded based on data dimensionality
            return (target_shape,)
        
        if isinstance(target_shape, (tuple, list)):
            return tuple(int(s) for s in target_shape)
        
        raise ValueError(f"target_shape must be int or tuple, got {type(target_shape)}")
    
    def _expand_target_shape(self, target_shape: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
        """Expand target shape to match data dimensionality.
        
        Args:
            target_shape: Parsed target shape.
            ndim: Number of spatial dimensions (2 or 3).
            
        Returns:
            Tuple of target dimensions matching the data dimensionality.
        """
        if len(target_shape) == 1:
            # Single value: replicate for all dimensions
            n = target_shape[0]
            if ndim == 2:
                return (n, n)
            else:  # ndim == 3
                return (n, n, n)
        
        if len(target_shape) == ndim:
            return target_shape
        
        raise ValueError(
            f"target_shape length ({len(target_shape)}) doesn't match data dimensionality ({ndim})"
        )
    
    def _compute_crop_pad_params(
        self,
        current_shape: Tuple[int, ...],
        target_shape: Tuple[int, ...]
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
        """Compute cropping and padding parameters.
        
        Args:
            current_shape: Current shape of the data.
            target_shape: Target shape.
            
        Returns:
            Tuple of (padding_params, cropping_params).
            Each can be None if not needed.
        """
        diff = np.array(target_shape) - np.array(current_shape)
        
        # Compute padding (where target > current)
        padding_amounts = np.maximum(diff, 0)
        if padding_amounts.any() and not self.only_crop:
            padding_params = self._get_bounds_parameters(padding_amounts)
        else:
            padding_params = None
        
        # Compute cropping (where current > target)
        cropping_amounts = np.maximum(-diff, 0)
        if cropping_amounts.any() and not self.only_pad:
            cropping_params = self._get_bounds_parameters(cropping_amounts)
        else:
            cropping_params = None
        
        return padding_params, cropping_params
    
    @staticmethod
    def _get_bounds_parameters(amounts: np.ndarray) -> Tuple[int, ...]:
        """Convert amounts to bounds parameters (split evenly between start and end).
        
        Args:
            amounts: Array of amounts to pad/crop for each dimension.
            
        Returns:
            Tuple of bounds (ini, fin, ini, fin, ...) for each dimension.
        """
        bounds = []
        for amount in amounts:
            # Split evenly, with extra going to the end if odd
            ini = int(np.ceil(amount / 2))
            fin = int(np.floor(amount / 2))
            bounds.extend([ini, fin])
        
        return tuple(bounds)
    
    def __call__(self, record: Record) -> Record:
        """Apply crop/pad to the record.
        
        Args:
            record: Input record with image and/or kspace data.
            
        Returns:
            Record with cropped/padded data to match target shape.
        """
        # Determine current shape and dimensionality
        if record["image"] is not None:
            ref_data = record["image"]
        elif record["kspace"] is not None:
            ref_data = record["kspace"]
        else:
            raise ValueError("Record must have either image or kspace data")
        
        current_shape = ref_data.shape
        ndim = ref_data.ndim
        
        # Expand target shape to match dimensionality
        target_shape = self._expand_target_shape(self.target_shape, ndim)
        
        # Compute crop/pad parameters
        padding_params, cropping_params = self._compute_crop_pad_params(
            current_shape, target_shape
        )
        
        # Apply padding first, then cropping
        if padding_params is not None:
            pad_transform = Pad(
                padding=padding_params,
                padding_mode=self.padding_mode,
                apply_to=self.apply_to
            )
            record = pad_transform(record)
        
        if cropping_params is not None:
            crop_transform = Crop(
                cropping=cropping_params,
                apply_to=self.apply_to
            )
            record = crop_transform(record)
        
        # Update metadata with overall transform info
        record["metadata"]["applied"].append({
            "transform": "CropOrPad",
            "target_shape": list(target_shape),
            "original_shape": list(current_shape),
            "padding_applied": padding_params is not None,
            "cropping_applied": cropping_params is not None,
            "apply_to": self.apply_to
        })
        
        return record

