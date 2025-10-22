"""Kmax undersampling across all encoding directions."""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, to_complex64

class KmaxUndersample:
    """Crop k-space to central region (kmax undersampling)."""
    
    def __init__(self, fractions: Union[Tuple[float, ...], None] = None,
                 fraction_ranges: Union[Tuple[Tuple[float, float], ...], None] = None):
        """
        Initialize kmax undersampling.
        
        Args:
            fractions: Fixed fractions to keep per axis (e.g., (0.8, 0.7, 1.0) for 3D)
                      1.0 means no cropping on that axis
            fraction_ranges: Ranges for random fraction sampling per axis
        """
        if fractions is None and fraction_ranges is None:
            raise ValueError("Either fractions or fraction_ranges must be provided")
        
        self.fractions = fractions
        self.fraction_ranges = fraction_ranges
    
    def __call__(self, record: Record) -> Record:
        """
        Apply kmax undersampling.
        
        Args:
            record: Input record
            
        Returns:
            Record with cropped k-space
        """
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"]
        
        # Determine fractions for each spatial dimension
        # Assume last 2 or 3 dimensions are spatial
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
        
        # Create centered box mask
        mask = np.ones(kspace.shape, dtype=np.float32)
        
        applied_fractions = []
        for i, frac in enumerate(fracs):
            if frac >= 1.0:
                applied_fractions.append(1.0)
                continue
            
            axis = -(n_spatial - i)  # -1, -2, or -3
            axis_size = kspace.shape[axis]
            keep_size = int(axis_size * frac)
            
            # Center the kept region
            start = (axis_size - keep_size) // 2
            end = start + keep_size
            
            # Zero out edges
            idx = [slice(None)] * kspace.ndim
            idx[axis] = slice(0, start)
            mask[tuple(idx)] = 0.0
            
            idx[axis] = slice(end, None)
            mask[tuple(idx)] = 0.0
            
            applied_fractions.append(float(frac))
        
        # Apply mask
        cropped_kspace = kspace * mask
        
        # Update or combine with existing mask
        if record["mask"] is not None:
            record["mask"] = record["mask"] * mask
        else:
            record["mask"] = mask
        
        record["kspace"] = cropped_kspace
        record["metadata"]["applied"].append({
            "transform": "KmaxUndersample",
            "fractions": applied_fractions
        })
        
        return record

