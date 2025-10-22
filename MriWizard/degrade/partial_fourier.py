"""Partial Fourier undersampling across encoding directions."""

import numpy as np
from typing import Union, Tuple, List
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, to_complex64

class PartialFourier:
    """Apply partial Fourier undersampling."""
    
    def __init__(self, fractions: Union[Tuple[float, ...], None] = None,
                 fraction_ranges: Union[Tuple[Tuple[float, float], ...], None] = None,
                 directions: Union[Tuple[Union[str, None], ...], None] = None):
        """
        Initialize partial Fourier undersampling.
        
        Args:
            fractions: Fractions to keep per axis (e.g., (0.75, 1.0, 1.0))
                      1.0 means full k-space on that axis
            fraction_ranges: Ranges for random fraction sampling per axis
            directions: Direction to keep per axis ("+" for positive half, "-" for negative,
                       None for no partial Fourier on that axis)
        """
        if fractions is None and fraction_ranges is None:
            raise ValueError("Either fractions or fraction_ranges must be provided")
        
        self.fractions = fractions
        self.fraction_ranges = fraction_ranges
        self.directions = directions
    
    def __call__(self, record: Record) -> Record:
        """
        Apply partial Fourier undersampling.
        
        Args:
            record: Input record
            
        Returns:
            Record with partial Fourier k-space
        """
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"]
        
        # Determine fractions for each spatial dimension
        n_spatial = min(3, kspace.ndim)
        
        if self.fractions is not None:
            fracs = self.fractions[:n_spatial]
        else:
            fracs = []
            for i in range(n_spatial):
                if i < len(self.fraction_ranges):
                    f_min, f_max = self.fraction_ranges[i]
                    fracs.append(np.random.uniform(f_min, f_max))
                else:
                    fracs.append(1.0)
        
        # Determine directions
        if self.directions is not None:
            dirs = self.directions[:n_spatial]
        else:
            # Default: random choice between +, -, or None
            dirs = []
            for frac in fracs:
                if frac >= 1.0:
                    dirs.append(None)
                else:
                    dirs.append(np.random.choice(["+", "-"]))
        
        # Create mask
        mask = np.ones(kspace.shape, dtype=np.float32)
        
        applied_fractions = []
        applied_directions = []
        
        for i, (frac, direction) in enumerate(zip(fracs, dirs)):
            if frac >= 1.0 or direction is None:
                applied_fractions.append(1.0)
                applied_directions.append(None)
                continue
            
            axis = -(n_spatial - i)  # -1, -2, or -3
            axis_size = kspace.shape[axis]
            keep_size = int(axis_size * frac)
            
            # Decide which half to keep
            if direction == "+":
                # Keep from center to positive end
                center = axis_size // 2
                start = center
                end = center + keep_size - (axis_size // 2)
                
                # Zero out negative half
                idx = [slice(None)] * kspace.ndim
                idx[axis] = slice(0, start - (keep_size - (axis_size - center)))
                mask[tuple(idx)] = 0.0
            else:  # direction == "-"
                # Keep from negative end to center
                center = axis_size // 2
                end = center
                start = center - keep_size + (axis_size - center)
                
                # Zero out positive half
                idx = [slice(None)] * kspace.ndim
                idx[axis] = slice(end + (keep_size - center), None)
                mask[tuple(idx)] = 0.0
            
            applied_fractions.append(float(frac))
            applied_directions.append(direction)
        
        # Apply mask
        partial_kspace = kspace * mask
        
        # Update or combine with existing mask
        if record["mask"] is not None:
            record["mask"] = record["mask"] * mask
        else:
            record["mask"] = mask
        
        record["kspace"] = partial_kspace
        record["metadata"]["applied"].append({
            "transform": "PartialFourier",
            "fractions": applied_fractions,
            "directions": applied_directions
        })
        
        return record

