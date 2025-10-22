"""Elliptical undersampling for 2D/3D k-space."""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, to_complex64

class EllipticalUndersample:
    """Apply elliptical mask to k-space."""
    
    def __init__(self, radii_fractions: Union[Tuple[float, ...], None] = None,
                 radii_ranges: Union[Tuple[Tuple[float, float], ...], None] = None):
        """
        Initialize elliptical undersampling.
        
        Args:
            radii_fractions: Fractions of half-width to use as ellipse radii per axis.
                For example, (0.8, 0.7, 1.0) for a 3D k-space will preserve an ellipse
                occupying 80% of half-width in the first dimension, 70% in the second, and full in the third.
            radii_ranges: Ranges for random radii sampling per axis. Each tuple specifies (min_frac, max_frac)
                for that axis, and a random fraction will be sampled in this range each call. E.g.,
                ((0.6, 0.8), (0.5, 0.7)) for 2D will randomly select the first axis ellipse radius
                between 60% and 80% of half-width, and second axis between 50% and 70%.
        """
        if radii_fractions is None and radii_ranges is None:
            raise ValueError("Either radii_fractions or radii_ranges must be provided")
        
        self.radii_fractions = radii_fractions
        self.radii_ranges = radii_ranges
    
    def __call__(self, record: Record) -> Record:
        """
        Apply elliptical mask.
        
        Args:
            record: Input record
            
        Returns:
            Record with elliptical-masked k-space
        """
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"]
        
        # Determine radii for each spatial dimension
        n_spatial = min(3, kspace.ndim)
        
        if self.radii_fractions is not None:
            radii_fracs = self.radii_fractions[:n_spatial]
        else:
            radii_fracs = []
            for i in range(n_spatial):
                if i < len(self.radii_ranges):
                    r_min, r_max = self.radii_ranges[i]
                    radii_fracs.append(np.random.uniform(r_min, r_max))
                else:
                    radii_fracs.append(1.0)
        
        # Create elliptical mask
        mask = np.zeros(kspace.shape, dtype=np.float32)
        
        # Get spatial dimensions
        spatial_shape = kspace.shape[-n_spatial:]
        
        # Create coordinate grids centered at 0
        grids = []
        radii = []
        for i, (size, frac) in enumerate(zip(spatial_shape, radii_fracs)):
            center = size / 2.0
            radius = (size / 2.0) * frac
            coords = (np.arange(size) - center) / radius
            grids.append(coords)
            radii.append(float(frac))
        
        # Compute elliptical distance
        if n_spatial == 2:
            Y, X = np.meshgrid(grids[0], grids[1], indexing='ij')
            dist = X**2 + Y**2
        elif n_spatial == 3:
            Z, Y, X = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
            dist = X**2 + Y**2 + Z**2
        else:  # n_spatial == 1
            dist = grids[0]**2
        
        # Apply elliptical mask
        ellipse_mask = (dist <= 1.0).astype(np.float32)
        
        # Broadcast to full shape if needed
        if kspace.ndim > n_spatial:
            # Add dimensions for channels/coils
            for _ in range(kspace.ndim - n_spatial):
                ellipse_mask = np.expand_dims(ellipse_mask, axis=0)
            ellipse_mask = np.broadcast_to(ellipse_mask, kspace.shape)
        
        mask = ellipse_mask
        
        # Apply mask
        masked_kspace = kspace * mask
        
        # Update or combine with existing mask
        if record["mask"] is not None:
            record["mask"] = record["mask"] * mask
        else:
            record["mask"] = mask
        
        record["kspace"] = masked_kspace
        record["metadata"]["applied"].append({
            "transform": "EllipticalUndersample",
            "radii_fractions": radii
        })
        
        return record

