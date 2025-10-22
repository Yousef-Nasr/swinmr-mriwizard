"""Gibbs ringing artifact via high-frequency k-space truncation."""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, to_complex64

class RandomGibbsRinging:
    """Simulate Gibbs ringing by truncating high-frequency k-space.
    
    Notes:
    - Ringing is most visible when truncation happens along a single axis.
      Provide `axes` to control along which spatial axes truncation occurs.
    """
    
    def __init__(self, fraction_range: Union[Tuple[float, float], None] = None,
                 axes: Union[Tuple[int, ...], None] = None):
        """
        Initialize Gibbs ringing transform.
        
        Args:
            fraction_range: Range for fraction of k-space to keep (min, max)
                          Lower values = more truncation = stronger ringing
        
        Gibbs ringing occurs when high frequencies are cut off in k-space.
        """
        if fraction_range is None:
            fraction_range = (0.6, 0.9)
        
        self.fraction_range = fraction_range
        # Default: last axis only for clearer ringing (phase-encode-like)
        self.axes = axes if axes is not None else (-1,)
    
    def __call__(self, record: Record) -> Record:
        """
        Apply Gibbs ringing by truncating k-space.
        
        Args:
            record: Input record
            
        Returns:
            Record with Gibbs-ringing-affected k-space
        """
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"].astype(np.complex64)
        
        # Sample fraction
        fraction = np.random.uniform(
            self.fraction_range[0],
            self.fraction_range[1]
        )
        
        # Create centered truncation mask
        mask = np.ones(kspace.shape, dtype=np.float32)
        
        # Apply truncation along selected axes (default: last axis)
        for axis in self.axes:
            axis_size = kspace.shape[axis]
            keep_size = int(axis_size * fraction)
            
            # Center the kept region
            start = (axis_size - keep_size) // 2
            end = start + keep_size
            
            # Zero out edges
            idx = [slice(None)] * kspace.ndim
            idx[axis] = slice(0, start)
            mask[tuple(idx)] = 0.0
            
            idx[axis] = slice(end, None)
            mask[tuple(idx)] = 0.0
        
        # Apply truncation
        truncated_kspace = kspace * mask
        
        # Update or combine with existing mask
        if record["mask"] is not None:
            record["mask"] = record["mask"] * mask
        else:
            record["mask"] = mask
        
        record["kspace"] = truncated_kspace
        record["metadata"]["applied"].append({
            "transform": "RandomGibbsRinging",
            "fraction": float(fraction),
            "axes": tuple(int(a) for a in self.axes)
        })
        
        return record

