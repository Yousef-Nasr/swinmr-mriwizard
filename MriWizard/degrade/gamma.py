"""Gamma correction for contrast adjustment in image domain."""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, ifft2c, to_complex64, to_float32

class RandomGamma:
    """Randomly change contrast by raising image values to power gamma.
    
    Gamma correction is commonly used in medical imaging to adjust contrast.
    - Gamma > 1: Gamma expansion (brightens mid-tones)
    - Gamma < 1: Gamma compression (darkens mid-tones)
    
    TorchIO semantics:
    - Uses log-gamma parameterization: gamma = exp(log_gamma)
    - Handles negative values: sign(I) * |I|^gamma
    
    Reference: https://en.wikipedia.org/wiki/Gamma_correction
    """
    
    def __init__(self, log_gamma_range: Union[Tuple[float, float], None] = None):
        """
        Initialize gamma correction transform.
        
        Args:
            log_gamma_range: Range for log(gamma) values (min, max)
                           gamma = exp(log_gamma)
                           Default: (-0.3, 0.3) gives gamma in ~(0.74, 1.35)
                           Negative log_gamma = compression (darker mid-tones)
                           Positive log_gamma = expansion (brighter mid-tones)
        
        Example gamma values from log_gamma:
            log_gamma = -0.3 -> gamma ≈ 0.74 (compression)
            log_gamma =  0.0 -> gamma = 1.00 (no change)
            log_gamma =  0.3 -> gamma ≈ 1.35 (expansion)
        """
        if log_gamma_range is None:
            log_gamma_range = (-0.3, 0.3)
        
        self.log_gamma_range = log_gamma_range
    
    def _apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction handling negative values.
        
        For images with negative values, use: sign(I) * |I|^gamma
        This ensures the operation is well-defined.
        """
        if np.min(image) < 0:
            # Handle negative values: preserve sign, apply gamma to absolute value
            output = np.sign(image) * np.abs(image) ** gamma
        else:
            # Standard gamma correction for non-negative images
            output = image ** gamma
        
        return output.astype(np.float32)
    
    def __call__(self, record: Record) -> Record:
        """
        Apply random gamma correction to image.
        
        Args:
            record: Input record
            
        Returns:
            Record with gamma-corrected image/kspace
        """
        # Get image (convert from k-space if needed)
        had_kspace_only = False
        if record["image"] is not None:
            image = to_float32(record["image"])
        elif record["kspace"] is not None:
            image = to_float32(ifft2c(record["kspace"]))
            had_kspace_only = True
        else:
            raise ValueError("Record must have either kspace or image")
        
        # Sample log_gamma and compute gamma
        log_gamma = float(np.random.uniform(
            self.log_gamma_range[0],
            self.log_gamma_range[1]
        ))
        gamma = float(np.exp(log_gamma))
        
        # Apply gamma correction
        corrected_image = self._apply_gamma(image, gamma)
        
        # Update record
        if had_kspace_only:
            # Convert back to k-space
            record["kspace"] = fft2c(to_complex64(corrected_image))
        else:
            record["image"] = corrected_image
            # Also update k-space if it exists
            if record["kspace"] is not None:
                record["kspace"] = fft2c(to_complex64(corrected_image))
        
        record["metadata"]["applied"].append({
            "transform": "RandomGamma",
            "log_gamma": float(log_gamma),
            "gamma": float(gamma)
        })
        
        return record


