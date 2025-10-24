"""Crop transform for MRI images.

Adapted from TorchIO's Crop transform for compatibility with MriWizard's record-based architecture.
"""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record


TypeBounds = Union[int, Tuple[int, ...]]


class Crop:
    """Crop an image by removing voxels from the borders.
    
    This transform crops the image by removing a specified number of voxels from each
    border of each spatial dimension. Both k-space and image domain data are cropped if present.
    
    Args:
        cropping: Number of voxels to crop from each border.
              - If int n: crop n voxels from all borders (6 values: start and end of each axis)
              - If tuple of 3 ints (w, h, d): crop w from width borders, h from height, d from depth
              - If tuple of 6 ints (w_i, w_f, h_i, h_f, d_i, d_f): crop specific amounts from each border
              
              For 2D images with shape (H, W):
                - (h_start, h_end, w_start, w_end)
                
              For 3D images with shape (D, H, W):
                - (d_start, d_end, h_start, h_end, w_start, w_end)
        
        apply_to: What to crop - "image", "kspace", or "both" (default: "both").
    
    Example:
        >>> # Crop 10 pixels from all borders
        >>> crop = Crop(cropping=10)
        >>> 
        >>> # Crop specific amounts: 10 from top/bottom, 20 from left/right
        >>> crop = Crop(cropping=(10, 20))
        >>>
        >>> # Crop different amounts from each border
        >>> crop = Crop(cropping=(5, 10, 15, 20))  # 2D: (h_start, h_end, w_start, w_end)
    
    Note:
        - The final shape will be: original_shape - cropping_start - cropping_end
        - Cropping in k-space affects the field of view in image space
        - The mask is also cropped if present
    """
    
    def __init__(
        self,
        cropping: TypeBounds,
        apply_to: str = "both"
    ):
        """Initialize the Crop transform."""
        self.cropping = cropping
        self.bounds_parameters = self._parse_bounds(cropping)
        
        if apply_to not in ("image", "kspace", "both"):
            raise ValueError(f"apply_to must be 'image', 'kspace', or 'both', got {apply_to}")
        self.apply_to = apply_to
    
    def _parse_bounds(self, bounds: TypeBounds) -> Tuple[int, ...]:
        """Parse bounds parameter into a tuple of 4 or 6 values.
        
        Args:
            bounds: Cropping specification.
            
        Returns:
            Tuple of bounds (h_i, h_f, w_i, w_f) for 2D or (d_i, d_f, h_i, h_f, w_i, w_f) for 3D.
        """
        if isinstance(bounds, int):
            # Single value: apply to all borders equally
            # Will be expanded to match data dimensionality in __call__
            return (bounds,)
        
        if isinstance(bounds, (tuple, list)):
            bounds_tuple = tuple(int(b) for b in bounds)
            
            # Validate lengths (1, 2, 3, 4, or 6)
            if len(bounds_tuple) not in (1, 2, 3, 4, 6):
                raise ValueError(
                    f"Bounds must have 1, 2, 3, 4, or 6 elements, got {len(bounds_tuple)}"
                )
            
            return bounds_tuple
        
        raise ValueError(f"Bounds must be int or tuple, got {type(bounds)}")
    
    def _expand_bounds(self, bounds: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
        """Expand bounds to match data dimensionality.
        
        Args:
            bounds: Parsed bounds tuple.
            ndim: Number of spatial dimensions (2 or 3).
            
        Returns:
            Tuple of bounds matching the data dimensionality.
        """
        if len(bounds) == 1:
            # Single value: apply to all borders
            n = bounds[0]
            if ndim == 2:
                return (n, n, n, n)
            else:  # ndim == 3
                return (n, n, n, n, n, n)
        
        if len(bounds) == 2:
            # Two values: (h, w) for 2D
            if ndim == 2:
                h, w = bounds
                return (h, h, w, w)
            else:
                raise ValueError(f"Cannot use 2 bounds values for {ndim}D data")
        
        if len(bounds) == 3:
            # Three values: (d, h, w) for 3D
            if ndim == 3:
                d, h, w = bounds
                return (d, d, h, h, w, w)
            else:
                raise ValueError(f"Cannot use 3 bounds values for {ndim}D data")
        
        if len(bounds) == 4:
            # Four values: (h_i, h_f, w_i, w_f) for 2D
            if ndim == 2:
                return bounds
            else:
                raise ValueError(f"Cannot use 4 bounds values for {ndim}D data")
        
        if len(bounds) == 6:
            # Six values: (d_i, d_f, h_i, h_f, w_i, w_f) for 3D
            if ndim == 3:
                return bounds
            else:
                raise ValueError(f"Cannot use 6 bounds values for {ndim}D data")
        
        raise ValueError(f"Unexpected bounds length: {len(bounds)}")
    
    def __call__(self, record: Record) -> Record:
        """Apply crop to the record.
        
        Args:
            record: Input record with image and/or kspace data.
            
        Returns:
            Record with cropped data.
        """
        # Determine dimensionality from available data
        if record["image"] is not None:
            ref_data = record["image"]
        elif record["kspace"] is not None:
            ref_data = record["kspace"]
        else:
            raise ValueError("Record must have either image or kspace data")
        
        ndim = ref_data.ndim
        bounds = self._expand_bounds(self.bounds_parameters, ndim)
        
        # Crop image if present and requested
        if record["image"] is not None and self.apply_to in ("image", "both"):
            record["image"] = _crop_array(record["image"], bounds)
        
        # Crop k-space if present and requested
        if record["kspace"] is not None and self.apply_to in ("kspace", "both"):
            record["kspace"] = _crop_array(record["kspace"], bounds)
        
        # Crop mask if present
        if record["mask"] is not None:
            record["mask"] = _crop_array(record["mask"], bounds)
        
        # Update metadata
        record["metadata"]["applied"].append({
            "transform": "Crop",
            "cropping": list(bounds),
            "apply_to": self.apply_to
        })
        
        return record


def _crop_array(data: np.ndarray, bounds: Tuple[int, ...]) -> np.ndarray:
    """Crop array according to bounds.

    Args:
        data: Input array to crop.
        bounds: Cropping bounds (h_i, h_f, w_i, w_f) or (d_i, d_f, h_i, h_f, w_i, w_f).

    Returns:
        Cropped array.
    """
    if len(bounds) == 4:
        # 2D crop
        h_i, h_f, w_i, w_f = bounds
        h_end = data.shape[0] - h_f if h_f > 0 else data.shape[0]
        w_end = data.shape[1] - w_f if w_f > 0 else data.shape[1]
        return data[h_i:h_end, w_i:w_end]

    elif len(bounds) == 6:
        # 3D crop
        d_i, d_f, h_i, h_f, w_i, w_f = bounds
        d_end = data.shape[0] - d_f if d_f > 0 else data.shape[0]
        h_end = data.shape[1] - h_f if h_f > 0 else data.shape[1]
        w_end = data.shape[2] - w_f if w_f > 0 else data.shape[2]
        return data[d_i:d_end, h_i:h_end, w_i:w_end]

    else:
        raise ValueError(f"Unexpected bounds length: {len(bounds)}")


class RandomCrop:
    """Randomly crop an image with range-based cropping amounts.

    This transform randomly crops the image by removing a random number of voxels from each
    border, where the amount is sampled from specified ranges. Both k-space and image domain
    data are cropped if present.

    Args:
        cropping_range: Range(s) for random cropping amounts.
              - If tuple of 2 ints (min, max): uniform range for all borders
              - If tuple of 2 tuples ((min_h, max_h), (min_w, max_w)): separate ranges for each dimension (2D)
              - If tuple of 3 tuples ((min_d, max_d), (min_h, max_h), (min_w, max_w)): ranges for 3D

              The cropping amount for each border is randomly sampled from the range independently.

        apply_to: What to crop - "image", "kspace", or "both" (default: "both").

        seed: Random seed for reproducibility (optional).

    Example:
        >>> # Random crop 0-20 pixels from all borders
        >>> crop = RandomCrop(cropping_range=(0, 20))
        >>>
        >>> # Different ranges per dimension: height [0,10], width [0,20]
        >>> crop = RandomCrop(cropping_range=((0, 10), (0, 20)))
        >>>
        >>> # 3D with specific ranges
        >>> crop = RandomCrop(cropping_range=((0, 5), (0, 10), (0, 15)))

    Note:
        - Each border gets an independent random value from the specified range
        - The final shape depends on the randomly sampled cropping amounts
        - For reproducible results, set the seed parameter
    """

    def __init__(
        self,
        cropping_range: Union[Tuple[int, int], Tuple[Tuple[int, int], ...]],
        apply_to: str = "both",
        seed: int = None
    ):
        """Initialize the RandomCrop transform."""
        self.cropping_range = cropping_range
        self.apply_to = apply_to
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        if apply_to not in ("image", "kspace", "both"):
            raise ValueError(f"apply_to must be 'image', 'kspace', or 'both', got {apply_to}")

    def _parse_ranges(
        self,
        cropping_range: Union[Tuple[int, int], Tuple[Tuple[int, int], ...]],
        ndim: int
    ) -> Tuple[Tuple[int, int], ...]:
        """Parse cropping ranges based on data dimensionality.

        Args:
            cropping_range: Range specification.
            ndim: Number of spatial dimensions (2 or 3).

        Returns:
            Tuple of (min, max) ranges for each dimension.
        """
        # Check if it's a single range (min, max)
        if len(cropping_range) == 2 and isinstance(cropping_range[0], int):
            # Single range for all dimensions
            return tuple([cropping_range] * ndim)

        # Multiple ranges - validate length matches ndim
        if len(cropping_range) != ndim:
            raise ValueError(
                f"Number of ranges ({len(cropping_range)}) doesn't match data dimensionality ({ndim})"
            )

        # Validate each range
        for i, r in enumerate(cropping_range):
            if not isinstance(r, (tuple, list)) or len(r) != 2:
                raise ValueError(f"Range {i} must be a tuple of 2 ints, got {r}")
            if r[0] > r[1]:
                raise ValueError(f"Range {i} min ({r[0]}) cannot be greater than max ({r[1]})")

        return tuple(cropping_range)

    def _sample_cropping(
        self,
        ranges: Tuple[Tuple[int, int], ...],
        ndim: int
    ) -> Tuple[int, ...]:
        """Sample random cropping amounts from ranges.

        Args:
            ranges: Tuple of (min, max) ranges for each dimension.
            ndim: Number of spatial dimensions.

        Returns:
            Tuple of cropping bounds for Crop transform.
        """
        bounds = []
        for min_val, max_val in ranges:
            # Sample start and end independently
            start = self.rng.randint(min_val, max_val + 1)
            end = self.rng.randint(min_val, max_val + 1)
            bounds.extend([start, end])

        return tuple(bounds)

    def __call__(self, record: Record) -> Record:
        """Apply random crop to the record.

        Args:
            record: Input record with image and/or kspace data.

        Returns:
            Record with randomly cropped data.
        """
        # Determine dimensionality from available data
        if record["image"] is not None:
            ref_data = record["image"]
        elif record["kspace"] is not None:
            ref_data = record["kspace"]
        else:
            raise ValueError("Record must have either image or kspace data")

        ndim = ref_data.ndim

        # Parse ranges based on dimensionality
        ranges = self._parse_ranges(self.cropping_range, ndim)

        # Sample random cropping amounts
        cropping_bounds = self._sample_cropping(ranges, ndim)

        # Apply crop using base Crop transform
        crop_transform = Crop(cropping=cropping_bounds, apply_to=self.apply_to)
        record = crop_transform(record)

        # Update metadata to indicate this was random
        record["metadata"]["applied"][-1]["transform"] = "RandomCrop"
        record["metadata"]["applied"][-1]["cropping_range"] = [list(r) for r in ranges]
        record["metadata"]["applied"][-1]["sampled_cropping"] = list(cropping_bounds)

        return record

