"""FFT-based reconstruction from k-space."""

import numpy as np
from MriWizard.core.base import Record
from MriWizard.core.utils import ifft2c, ifftnc, to_float32

class IFFTReconstruct:
    """Reconstruct image from k-space using inverse FFT."""
    
    def __init__(self, axes: tuple = None, normalize: bool = False):
        """
        Initialize IFFT reconstruction.
        
        Args:
            axes: Axes to reconstruct (None = auto-detect based on k-space dims)
            normalize: Whether to normalize output to [0, 1]
        """
        self.axes = axes
        self.normalize = normalize
    
    def __call__(self, record: Record) -> Record:
        """
        Reconstruct image from k-space.
        
        Args:
            record: Input record with k-space
            
        Returns:
            Record with reconstructed image
        """
        if record["kspace"] is None:
            raise ValueError("Cannot reconstruct: no k-space data in record")
        
        kspace = record["kspace"]
        
        # Determine axes for reconstruction
        if self.axes is not None:
            axes = self.axes
        else:
            # Auto-detect: use last 2 or 3 spatial dimensions
            n_spatial = min(3, kspace.ndim)
            axes = tuple(range(-n_spatial, 0))
        
        # Perform inverse FFT
        if len(axes) == 2:
            image_complex = ifft2c(kspace)
        else:
            image_complex = ifftnc(kspace, axes=axes)
        
        # Take magnitude
        image = to_float32(image_complex)
        
        # Optional normalization
        if self.normalize:
            max_val = np.percentile(image, 99)
            if max_val > 0:
                image = np.clip(image / max_val, 0, 1)
        
        # Update record
        record["image"] = image
        record["metadata"]["applied"].append({
            "transform": "IFFTReconstruct",
            "axes": list(axes) if axes else None
        })
        
        return record

