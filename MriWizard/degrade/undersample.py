"""Uniform and random undersampling transforms."""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, to_complex64

class UniformUndersample:
    """Uniform pattern undersampling (every R-th line) with center preservation."""
    
    def __init__(self, R: Union[int, None] = None, 
                 R_range: Union[Tuple[int, int], None] = None,
                 center_fraction: float = 0.08,
                 axis: int = -2):
        """
        Initialize uniform undersampling.
        
        Args:
            R: Acceleration factor (if None, use R_range)
            R_range: Range (min, max) for random R sampling
            center_fraction: Fraction of k-space center to preserve (low frequencies)
            axis: Axis along which to undersample (default: -2, phase-encode)
        """
        if R is None and R_range is None:
            raise ValueError("Either R or R_range must be provided")
        
        self.R = R
        self.R_range = R_range
        self.center_fraction = center_fraction
        self.axis = axis
    
    def __call__(self, record: Record) -> Record:
        """
        Apply uniform undersampling with center preservation.
        
        Args:
            record: Input record
            
        Returns:
            Record with undersampled k-space
        """
        # Get or sample R
        if self.R is not None:
            R = self.R
        else:
            R = np.random.randint(self.R_range[0], self.R_range[1] + 1)
        
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"]
        axis_size = kspace.shape[self.axis]
        
        # Create mask
        mask = np.zeros(kspace.shape, dtype=np.float32)
        
        # Step 1: Keep center (low frequencies)
        num_low_frequencies = round(axis_size * self.center_fraction)
        center_start = (axis_size - num_low_frequencies) // 2
        center_end = center_start + num_low_frequencies
        
        # Build index tuple for center
        idx_center = [slice(None)] * kspace.ndim
        idx_center[self.axis] = slice(center_start, center_end)
        mask[tuple(idx_center)] = 1.0
        
        # Step 2: Uniform undersampling in outer regions
        # Exclude center region
        outer_indices = np.concatenate([
            np.arange(0, center_start, R),
            np.arange(center_end + (R - (center_end % R)) % R, axis_size, R)
        ])
        
        idx_outer = [slice(None)] * kspace.ndim
        idx_outer[self.axis] = outer_indices
        mask[tuple(idx_outer)] = 1.0
        
        # Apply mask
        undersampled_kspace = kspace * mask
        
        # Update or combine with existing mask
        if record["mask"] is not None:
            record["mask"] = record["mask"] * mask
        else:
            record["mask"] = mask
        
        record["kspace"] = undersampled_kspace
        record["metadata"]["applied"].append({
            "transform": "UniformUndersample",
            "R": int(R),
            "center_fraction": float(self.center_fraction),
            "axis": int(self.axis)
        })
        
        return record

class RandomUndersample:
    """Random pattern undersampling with center preservation."""
    
    def __init__(self, prob: Union[float, None] = None,
                 prob_range: Union[Tuple[float, float], None] = None,
                 center_fraction: float = 0.08,
                 axis: int = -2):
        """
        Initialize random undersampling.
        
        Args:
            prob: Probability of keeping a line (if None, use prob_range)
            prob_range: Range (min, max) for random prob sampling
            center_fraction: Fraction of k-space center to preserve (low frequencies)
            axis: Axis along which to undersample (default: -2, phase-encode)
        """
        if prob is None and prob_range is None:
            raise ValueError("Either prob or prob_range must be provided")
        
        self.prob = prob
        self.prob_range = prob_range
        self.center_fraction = center_fraction
        self.axis = axis
    
    def __call__(self, record: Record) -> Record:
        """
        Apply random undersampling with center preservation.
        
        Args:
            record: Input record
            
        Returns:
            Record with undersampled k-space
        """
        # Get or sample prob
        if self.prob is not None:
            prob = self.prob
        else:
            prob = np.random.uniform(self.prob_range[0], self.prob_range[1])
        
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"]
        axis_size = kspace.shape[self.axis]
        
        # Create mask
        mask = np.zeros(kspace.shape, dtype=np.float32)
        
        # Step 1: Keep center (low frequencies)
        num_low_frequencies = round(axis_size * self.center_fraction)
        center_start = (axis_size - num_low_frequencies) // 2
        center_end = center_start + num_low_frequencies
        
        # Build index tuple for center
        idx_center = [slice(None)] * kspace.ndim
        idx_center[self.axis] = slice(center_start, center_end)
        mask[tuple(idx_center)] = 1.0
        
        # Step 2: Random undersampling in outer regions
        line_mask = np.zeros(axis_size, dtype=bool)
        # Center is always kept
        line_mask[center_start:center_end] = True
        # Randomly keep outer lines
        outer_mask = np.random.rand(axis_size) < prob
        line_mask = np.logical_or(line_mask, outer_mask)
        
        idx_outer = [slice(None)] * kspace.ndim
        idx_outer[self.axis] = line_mask
        mask[tuple(idx_outer)] = 1.0
        
        # Apply mask
        undersampled_kspace = kspace * mask
        
        # Update or combine with existing mask
        if record["mask"] is not None:
            record["mask"] = record["mask"] * mask
        else:
            record["mask"] = mask
        
        record["kspace"] = undersampled_kspace
        record["metadata"]["applied"].append({
            "transform": "RandomUndersample",
            "prob": float(prob),
            "center_fraction": float(self.center_fraction),
            "axis": int(self.axis)
        })
        
        return record

