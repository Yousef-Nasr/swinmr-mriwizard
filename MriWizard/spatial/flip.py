"""Flip transforms for MRI images.

Adapted from TorchIO's RandomFlip transform for compatibility with MriWizard's record-based architecture.
"""

import numpy as np
from typing import Union, Tuple, List
from MriWizard.core.base import Record
from MriWizard.core.utils import to_float32


class RandomFlip:
    """Randomly flip an image along the given axes.
    
    This transform randomly flips the image data along specified axes with a given probability.
    Both k-space and image domain data are flipped if present.
    
    Args:
        axes: Axis or tuple of axes along which to flip.
              For 2D images: 0 (vertical/height), 1 (horizontal/width)
              For 3D images: 0 (depth), 1 (height), 2 (width)
              Can also use negative indexing: -1 (last axis), -2 (second-to-last), etc.
        flip_probability: Probability of flipping along each axis (default: 0.5).
        apply_to: What to flip - "image", "kspace", or "both" (default: "both").
    
    Example:
        >>> # Randomly flip horizontally with 50% probability
        >>> flip = RandomFlip(axes=1, flip_probability=0.5)
        >>> 
        >>> # Flip along both axes
        >>> flip = RandomFlip(axes=(0, 1), flip_probability=0.5)
        >>>
        >>> # Higher probability
        >>> flip = RandomFlip(axes=(0, 1), flip_probability=0.8)
    
    Note:
        - The flip is applied independently to each axis with the specified probability
        - Flipping in k-space is equivalent to flipping in image space (up to phase)
        - The mask is also flipped if present
    """
    
    def __init__(
        self,
        axes: Union[int, Tuple[int, ...]] = (0, 1),
        flip_probability: float = 0.5,
        apply_to: str = "both"
    ):
        """Initialize the RandomFlip transform."""
        self.axes = _parse_axes(axes)
        
        if not 0 <= flip_probability <= 1:
            raise ValueError(f"flip_probability must be in [0, 1], got {flip_probability}")
        self.flip_probability = flip_probability
        
        if apply_to not in ("image", "kspace", "both"):
            raise ValueError(f"apply_to must be 'image', 'kspace', or 'both', got {apply_to}")
        self.apply_to = apply_to
    
    def __call__(self, record: Record) -> Record:
        """Apply random flip to the record.
        
        Args:
            record: Input record with image and/or kspace data.
            
        Returns:
            Record with randomly flipped data.
        """
        # Determine which axes to flip based on probability
        axes_to_flip = []
        for axis in self.axes:
            if np.random.rand() < self.flip_probability:
                axes_to_flip.append(axis)
        
        if not axes_to_flip:
            # No flipping needed
            record["metadata"]["applied"].append({
                "transform": "RandomFlip",
                "axes_flipped": [],
                "flip_probability": float(self.flip_probability)
            })
            return record
        
        # Apply the flip using the deterministic Flip transform
        flip_transform = Flip(axes=tuple(axes_to_flip), apply_to=self.apply_to)
        record = flip_transform(record)
        
        # Update metadata
        record["metadata"]["applied"][-1].update({
            "transform": "RandomFlip",
            "flip_probability": float(self.flip_probability)
        })
        
        return record


class Flip:
    """Deterministically flip an image along the given axes.
    
    This transform flips the image data along specified axes. Both k-space and 
    image domain data are flipped if present.
    
    Args:
        axes: Axis or tuple of axes along which to flip.
              For 2D images: 0 (vertical/height), 1 (horizontal/width)
              For 3D images: 0 (depth), 1 (height), 2 (width)
              Can also use negative indexing: -1 (last axis), -2 (second-to-last), etc.
        apply_to: What to flip - "image", "kspace", or "both" (default: "both").
    
    Example:
        >>> # Flip horizontally
        >>> flip = Flip(axes=1)
        >>> 
        >>> # Flip along both axes
        >>> flip = Flip(axes=(0, 1))
    
    Note:
        - Flipping in k-space is equivalent to flipping in image space (up to phase)
        - The mask is also flipped if present
        - This transform is invertible (flipping twice returns the original)
    """
    
    def __init__(
        self,
        axes: Union[int, Tuple[int, ...]],
        apply_to: str = "both"
    ):
        """Initialize the Flip transform."""
        self.axes = _parse_axes(axes)
        
        if apply_to not in ("image", "kspace", "both"):
            raise ValueError(f"apply_to must be 'image', 'kspace', or 'both', got {apply_to}")
        self.apply_to = apply_to
    
    def __call__(self, record: Record) -> Record:
        """Apply flip to the record.
        
        Args:
            record: Input record with image and/or kspace data.
            
        Returns:
            Record with flipped data.
        """
        axes_list = list(self.axes)
        
        # Flip image if present and requested
        if record["image"] is not None and self.apply_to in ("image", "both"):
            record["image"] = _flip_array(record["image"], axes_list)
        
        # Flip k-space if present and requested
        if record["kspace"] is not None and self.apply_to in ("kspace", "both"):
            record["kspace"] = _flip_array(record["kspace"], axes_list)
        
        # Flip mask if present
        if record["mask"] is not None:
            record["mask"] = _flip_array(record["mask"], axes_list)
        
        # Update metadata
        record["metadata"]["applied"].append({
            "transform": "Flip",
            "axes": axes_list,
            "apply_to": self.apply_to
        })
        
        return record


def _parse_axes(axes: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Parse and validate axes parameter.
    
    Args:
        axes: Single axis index or tuple of axis indices.
        
    Returns:
        Tuple of axis indices.
        
    Raises:
        ValueError: If axes are invalid.
    """
    if isinstance(axes, int):
        axes_tuple = (axes,)
    elif isinstance(axes, (tuple, list)):
        axes_tuple = tuple(axes)
    else:
        raise ValueError(f"axes must be int or tuple of ints, got {type(axes)}")
    
    # Validate all axes are integers
    for axis in axes_tuple:
        if not isinstance(axis, (int, np.integer)):
            raise ValueError(f"All axes must be integers, found {type(axis)}")
    
    return axes_tuple


def _flip_array(data: np.ndarray, axes: List[int]) -> np.ndarray:
    """Flip array along specified axes.
    
    Args:
        data: Input array to flip.
        axes: List of axes to flip along.
        
    Returns:
        Flipped array.
    """
    if not axes:
        return data
    
    # Numpy's flip can handle negative indices
    flipped = np.flip(data, axis=axes)
    
    # Ensure contiguous array (flip can create views with negative strides)
    return np.ascontiguousarray(flipped)

