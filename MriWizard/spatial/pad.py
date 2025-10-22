"""Pad transform for MRI images.

Adapted from TorchIO's Pad transform for compatibility with MriWizard's record-based architecture.
"""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record


TypeBounds = Union[int, Tuple[int, ...]]


class Pad:
    """Pad an image by adding voxels to the borders.
    
    This transform pads the image by adding a specified number of voxels to each
    border of each spatial dimension. Both k-space and image domain data are padded if present.
    
    Args:
        padding: Number of voxels to pad at each border.
              - If int n: pad n voxels at all borders
              - If tuple of 2 ints (h, w): pad h voxels at height borders, w at width borders
              - If tuple of 3 ints (d, h, w): pad d at depth, h at height, w at width borders
              - If tuple of 4 ints (h_i, h_f, w_i, w_f): pad specific amounts at each border (2D)
              - If tuple of 6 ints (d_i, d_f, h_i, h_f, w_i, w_f): pad specific amounts (3D)
        
        padding_mode: Padding mode, same as numpy.pad:
              - 'constant': Pads with a constant value (default: 0)
              - 'edge': Pads with the edge values of the array
              - 'reflect': Pads with reflection of the array
              - 'symmetric': Pads with symmetric reflection
              - 'wrap': Pads with wrap-around values
              - 'mean': Pads with the mean value of the array
              - 'median': Pads with the median value of the array
              - 'minimum': Pads with the minimum value of the array
              - 'maximum': Pads with the maximum value of the array
              - float/int: Pads with the specified constant value
        
        apply_to: What to pad - "image", "kspace", or "both" (default: "both").
    
    Example:
        >>> # Pad 10 pixels on all borders with zeros
        >>> pad = Pad(padding=10)
        >>> 
        >>> # Pad specific amounts: 10 on top/bottom, 20 on left/right
        >>> pad = Pad(padding=(10, 20))
        >>>
        >>> # Pad with edge values
        >>> pad = Pad(padding=10, padding_mode='edge')
        >>>
        >>> # Pad with constant value
        >>> pad = Pad(padding=10, padding_mode=0.5)
    
    Note:
        - The final shape will be: original_shape + padding_start + padding_end
        - Padding in k-space affects the field of view in image space
        - The mask is also padded if present
        - For complex arrays (k-space), padding is applied separately to real and imaginary parts
    """
    
    def __init__(
        self,
        padding: TypeBounds,
        padding_mode: Union[str, float] = 0,
        apply_to: str = "both"
    ):
        """Initialize the Pad transform."""
        self.padding = padding
        self.bounds_parameters = self._parse_bounds(padding)
        self.padding_mode = padding_mode
        self._check_padding_mode(padding_mode)
        
        if apply_to not in ("image", "kspace", "both"):
            raise ValueError(f"apply_to must be 'image', 'kspace', or 'both', got {apply_to}")
        self.apply_to = apply_to
    
    @staticmethod
    def _check_padding_mode(padding_mode):
        """Validate padding mode."""
        valid_modes = (
            'constant', 'edge', 'reflect', 'symmetric', 'wrap',
            'mean', 'median', 'minimum', 'maximum'
        )
        if isinstance(padding_mode, (int, float)):
            return  # Numeric value is valid
        if padding_mode not in valid_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_modes} or a number, got {padding_mode}"
            )
    
    def _parse_bounds(self, bounds: TypeBounds) -> Tuple[int, ...]:
        """Parse bounds parameter into a tuple.
        
        Args:
            bounds: Padding specification.
            
        Returns:
            Tuple of bounds.
        """
        if isinstance(bounds, int):
            return (bounds,)
        
        if isinstance(bounds, (tuple, list)):
            bounds_tuple = tuple(int(b) for b in bounds)
            
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
            n = bounds[0]
            if ndim == 2:
                return (n, n, n, n)
            else:  # ndim == 3
                return (n, n, n, n, n, n)
        
        if len(bounds) == 2:
            if ndim == 2:
                h, w = bounds
                return (h, h, w, w)
            else:
                raise ValueError(f"Cannot use 2 bounds values for {ndim}D data")
        
        if len(bounds) == 3:
            if ndim == 3:
                d, h, w = bounds
                return (d, d, h, h, w, w)
            else:
                raise ValueError(f"Cannot use 3 bounds values for {ndim}D data")
        
        if len(bounds) == 4:
            if ndim == 2:
                return bounds
            else:
                raise ValueError(f"Cannot use 4 bounds values for {ndim}D data")
        
        if len(bounds) == 6:
            if ndim == 3:
                return bounds
            else:
                raise ValueError(f"Cannot use 6 bounds values for {ndim}D data")
        
        raise ValueError(f"Unexpected bounds length: {len(bounds)}")
    
    def __call__(self, record: Record) -> Record:
        """Apply padding to the record.
        
        Args:
            record: Input record with image and/or kspace data.
            
        Returns:
            Record with padded data.
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
        
        # Pad image if present and requested
        if record["image"] is not None and self.apply_to in ("image", "both"):
            record["image"] = _pad_array(
                record["image"], bounds, self.padding_mode
            )
        
        # Pad k-space if present and requested
        if record["kspace"] is not None and self.apply_to in ("kspace", "both"):
            record["kspace"] = _pad_array(
                record["kspace"], bounds, self.padding_mode
            )
        
        # Pad mask if present (always use zeros for mask)
        if record["mask"] is not None:
            record["mask"] = _pad_array(record["mask"], bounds, padding_mode=0)
        
        # Update metadata
        record["metadata"]["applied"].append({
            "transform": "Pad",
            "padding": list(bounds),
            "padding_mode": self.padding_mode if isinstance(self.padding_mode, str) else float(self.padding_mode),
            "apply_to": self.apply_to
        })
        
        return record


def _pad_array(
    data: np.ndarray,
    bounds: Tuple[int, ...],
    padding_mode: Union[str, float] = 0
) -> np.ndarray:
    """Pad array according to bounds.
    
    Args:
        data: Input array to pad.
        bounds: Padding bounds (h_i, h_f, w_i, w_f) or (d_i, d_f, h_i, h_f, w_i, w_f).
        padding_mode: Padding mode or constant value.
        
    Returns:
        Padded array.
    """
    # Prepare padding specification for numpy
    if len(bounds) == 4:
        # 2D padding
        h_i, h_f, w_i, w_f = bounds
        pad_width = ((h_i, h_f), (w_i, w_f))
    elif len(bounds) == 6:
        # 3D padding
        d_i, d_f, h_i, h_f, w_i, w_f = bounds
        pad_width = ((d_i, d_f), (h_i, h_f), (w_i, w_f))
    else:
        raise ValueError(f"Unexpected bounds length: {len(bounds)}")
    
    # Handle different padding modes
    if isinstance(padding_mode, (int, float)):
        # Constant padding with specific value
        mode = 'constant'
        kwargs = {'constant_values': padding_mode}
    elif padding_mode in ('mean', 'median', 'minimum', 'maximum'):
        # Statistical padding modes
        mode = 'constant'
        if np.iscomplexobj(data):
            # For complex data, compute statistic on magnitude
            mag = np.abs(data)
            if padding_mode == 'mean':
                stat_val = float(np.mean(mag))
            elif padding_mode == 'median':
                stat_val = float(np.median(mag))
            elif padding_mode == 'minimum':
                stat_val = float(np.min(mag))
            else:  # maximum
                stat_val = float(np.max(mag))
            # Convert back to complex (zero phase)
            kwargs = {'constant_values': stat_val}
        else:
            if padding_mode == 'mean':
                stat_val = float(np.mean(data))
            elif padding_mode == 'median':
                stat_val = float(np.median(data))
            elif padding_mode == 'minimum':
                stat_val = float(np.min(data))
            else:  # maximum
                stat_val = float(np.max(data))
            kwargs = {'constant_values': stat_val}
    else:
        # Standard numpy padding mode
        mode = padding_mode
        kwargs = {}
    
    # Apply padding
    padded = np.pad(data, pad_width, mode=mode, **kwargs)
    
    return padded

