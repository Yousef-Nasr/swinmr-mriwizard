"""Gaussian noise degradation for complex k-space.

Mimics TorchIO's RandomNoise behavior but adapted for k-space data.
TorchIO adds noise: image_out = image_in + N(mean, std)
"""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, to_complex64

class AddGaussianNoiseKspace:
    """Add Gaussian noise to complex k-space data.
    
    Mimics TorchIO's RandomNoise transform:
    - TorchIO: mean default=0, std default=(0, 0.25)
    - Samples mean from uniform distribution if range provided
    - Samples std from uniform distribution if range provided
    - Adds noise with sampled parameters: data_out = data_in + N(mean, std)
    """
    
    def __init__(self,
                 mean: Union[float, Tuple[float, float]] = 0.0,
                 std: Union[float, Tuple[float, float]] = (0, 0.25),
                 relative: bool = True,
                 reference: str = "std"):
        """
        Initialize noise transform (TorchIO-compatible semantics).
        
        Args:
            mean: Mean of the Gaussian noise distribution.
                  If float: use fixed mean value.
                  If tuple (a, b): sample mean ~ Uniform(a, b).
                  If single value d: sample mean ~ Uniform(-d, d).
                  Default: 0 (zero mean noise).
                  
            std: Standard deviation of the Gaussian noise distribution.
                 If float: use fixed std value.
                 If tuple (a, b): sample std ~ Uniform(a, b).
                 If single value d: sample std ~ Uniform(0, d).
                 Default: (0, 0.25) matches TorchIO.
                 
            relative: If True, interpret mean/std as fractions of k-space amplitude.
                     If False, interpret as absolute values.
                     Default: True (relative to data scale).
                     
            reference: Amplitude reference when relative=True:
                      "std" (default, matches TorchIO behavior),
                      "rms", "p99", "max".
        
        Example (TorchIO equivalent):
            # TorchIO: RandomNoise(mean=0, std=(0, 0.25))
            AddGaussianNoiseKspace(mean=0, std=(0, 0.25), relative=True)
            
            # TorchIO: RandomNoise(mean=(-0.1, 0.1), std=(0.05, 0.15))
            AddGaussianNoiseKspace(mean=(-0.1, 0.1), std=(0.05, 0.15), relative=True)
        """
        # Parse mean range (TorchIO semantics)
        self.mean_range = self._parse_range(mean, 'mean')
        
        # Parse std range (TorchIO semantics)
        self.std_range = self._parse_range(std, 'std', min_constraint=0)
        
        self.relative = relative
        self.reference = reference
    
    def _parse_range(
        self,
        value: Union[float, Tuple[float, float]],
        name: str,
        min_constraint: Union[float, None] = None
    ) -> Tuple[float, float]:
        """Parse parameter range following TorchIO semantics.
        
        Args:
            value: Either a single value or a tuple (min, max).
            name: Parameter name for error messages.
            min_constraint: If provided, enforce minimum value.
            
        Returns:
            Tuple (min, max) for uniform sampling.
            
        TorchIO behavior:
            - If tuple (a, b): return (a, b)
            - If single value d for 'mean': return (-d, d)
            - If single value d for 'std': return (0, d)
        """
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"{name} must be a float or a tuple of two floats")
            range_min, range_max = value
        else:
            # Single value: apply TorchIO semantics
            if name == 'mean':
                # For mean: single value d -> range (-d, d)
                range_min, range_max = -abs(value), abs(value)
            else:
                # For std: single value d -> range (0, d)
                range_min, range_max = 0, abs(value)
        
        # Apply minimum constraint if provided
        if min_constraint is not None:
            if range_min < min_constraint:
                raise ValueError(
                    f"{name} minimum ({range_min}) is below constraint ({min_constraint})"
                )
        
        if range_min > range_max:
            raise ValueError(
                f"{name} range ({range_min}, {range_max}) is invalid: min > max"
            )
        
        return float(range_min), float(range_max)
    
    def _compute_reference_amplitude(self, kspace: np.ndarray) -> float:
        mag = np.abs(kspace).astype(np.float32)
        ref = self.reference.lower()
        if ref == "rms":
            return float(np.sqrt(np.mean(mag * mag)) + 1e-12)
        if ref == "std":
            return float(np.std(mag) + 1e-12)
        if ref == "p99":
            return float(np.percentile(mag, 99.0) + 1e-12)
        if ref == "max":
            return float(np.max(mag) + 1e-12)
        # Fallback to RMS
        return float(np.sqrt(np.mean(mag * mag)) + 1e-12)
    
    def __call__(self, record: Record) -> Record:
        """
        Add Gaussian noise to k-space (TorchIO semantics).
        
        Args:
            record: Input record
            
        Returns:
            Record with noisy k-space
            
        Noise generation follows TorchIO:
            noise = randn() * std + mean
            output = input + noise
        """
        # Sample mean and std from their ranges (TorchIO semantics)
        mean_sampled = float(np.random.uniform(self.mean_range[0], self.mean_range[1]))
        std_sampled = float(np.random.uniform(self.std_range[0], self.std_range[1]))
        
        # Ensure we have k-space
        if record["kspace"] is None:
            if record["image"] is not None:
                # Convert image to k-space
                record["kspace"] = fft2c(to_complex64(record["image"]))
            else:
                raise ValueError("Record must have either kspace or image")
        
        kspace = record["kspace"]
        
        # Determine absolute mean/std based on data amplitude (if relative)
        if self.relative:
            ref_amp = self._compute_reference_amplitude(kspace)
            mean_abs = mean_sampled * ref_amp
            std_abs = std_sampled * ref_amp
            mean_rel = mean_sampled
            std_rel = std_sampled
        else:
            mean_abs = mean_sampled
            std_abs = std_sampled
            # Compute relative values for logging
            ref_amp = self._compute_reference_amplitude(kspace)
            mean_rel = float(mean_abs / (ref_amp + 1e-12))
            std_rel = float(std_abs / (ref_amp + 1e-12))
        
        # Generate complex Gaussian noise: N(mean, std) for real and imaginary parts
        # TorchIO: noise = randn() * std + mean
        noise_real = np.random.randn(*kspace.shape).astype(np.float32) * std_abs + mean_abs
        noise_imag = np.random.randn(*kspace.shape).astype(np.float32) * std_abs + mean_abs
        noise = (noise_real + 1j * noise_imag).astype(np.complex64)
        
        # Add noise to k-space (TorchIO: output = input + noise)
        noisy_kspace = kspace.astype(np.complex64) + noise
        
        # Update record
        record["kspace"] = noisy_kspace
        record["metadata"]["applied"].append({
            "transform": "AddGaussianNoiseKspace",
            "mean_relative": float(mean_rel),
            "mean_absolute": float(mean_abs),
            "std_relative": float(std_rel),
            "std_absolute": float(std_abs),
            "relative": bool(self.relative),
            "reference": str(self.reference)
        })
        
        return record


# Backward compatibility: provide convenience constructors for common patterns
class AddGaussianNoiseKspace_Legacy:
    """Legacy API wrapper for backward compatibility.
    
    Provides the old sigma/sigma_range interface that maps to the new mean/std API.
    For new code, use AddGaussianNoiseKspace directly with mean/std parameters.
    """
    
    @staticmethod
    def from_sigma(
        sigma: Union[float, None] = None,
        sigma_range: Union[Tuple[float, float], None] = None,
        relative: bool = True,
        reference: str = "rms"
    ) -> AddGaussianNoiseKspace:
        """Create noise transform using legacy sigma parameter (maps to std).
        
        Args:
            sigma: Fixed noise std (legacy parameter).
            sigma_range: Range for std (legacy parameter).
            relative: If True, interpret as fraction of k-space amplitude.
            reference: Amplitude reference when relative=True.
            
        Returns:
            AddGaussianNoiseKspace instance with mean=0, std=sigma.
        """
        if sigma is not None:
            std_param = sigma
        elif sigma_range is not None:
            std_param = sigma_range
        else:
            raise ValueError("Either sigma or sigma_range must be provided")
        
        return AddGaussianNoiseKspace(
            mean=0.0,  # Zero-mean noise (most common)
            std=std_param,
            relative=relative,
            reference=reference
        )

