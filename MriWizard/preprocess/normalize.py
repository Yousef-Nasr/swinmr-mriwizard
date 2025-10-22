"""Normalization and standardization transforms for MRI data.

Provides various normalization methods commonly used in medical imaging:
- Z-score normalization (standardization)
- Min-Max normalization (scaling to [0, 1] or custom range)
- Percentile-based normalization
- Robust normalization (IQR-based)

These transforms work on image domain data and follow TorchIO conventions.
"""

import numpy as np
from typing import Union, Tuple, Optional
from MriWizard.core.base import Record
from MriWizard.core.utils import ifft2c, fft2c, to_float32, to_complex64


class Normalize:
    """Base class for normalization transforms.
    
    Provides common functionality for all normalization methods.
    """
    
    def __init__(self, apply_to: str = "image", masking_method: Optional[str] = None):
        """
        Initialize normalization transform.
        
        Args:
            apply_to: Which data to normalize - "image" or "kspace".
                     Default: "image" (recommended for most cases).
            masking_method: Method to compute statistics:
                          None (default): use all voxels
                          "otsu": use Otsu thresholding to exclude background
                          "percentile": exclude bottom 1% (near-zero values)
        """
        self.apply_to = apply_to
        self.masking_method = masking_method
    
    def _get_image(self, record: Record) -> np.ndarray:
        """Get image from record, converting from k-space if needed."""
        if record["image"] is not None:
            return to_float32(record["image"])
        elif record["kspace"] is not None:
            return to_float32(ifft2c(record["kspace"]))
        else:
            raise ValueError("Record must have either kspace or image")
    
    def _compute_mask(self, image: np.ndarray) -> np.ndarray:
        """Compute binary mask for foreground voxels."""
        if self.masking_method is None:
            return np.ones(image.shape, dtype=bool)
        
        elif self.masking_method == "otsu":
            # Simple Otsu-like threshold
            threshold = np.percentile(image, 50)
            return image > threshold
        
        elif self.masking_method == "percentile":
            # Exclude bottom 1% (background)
            threshold = np.percentile(image, 1)
            return image > threshold
        
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement _normalize()")
    
    def __call__(self, record: Record) -> Record:
        """Apply normalization to record."""
        # Get image
        had_kspace_only = record["image"] is None and record["kspace"] is not None
        image = self._get_image(record)
        
        # Apply normalization
        normalized_image = self._normalize(image)
        
        # Update record
        if self.apply_to == "image" or had_kspace_only:
            if had_kspace_only:
                # Convert back to k-space
                record["kspace"] = fft2c(to_complex64(normalized_image))
            else:
                record["image"] = normalized_image
                # Also update k-space if it exists
                if record["kspace"] is not None:
                    record["kspace"] = fft2c(to_complex64(normalized_image))
        else:
            # Normalize k-space directly (less common)
            if record["kspace"] is None:
                record["kspace"] = fft2c(to_complex64(image))
            
            kspace = record["kspace"]
            # Normalize magnitude while preserving phase
            mag = np.abs(kspace)
            phase = np.angle(kspace)
            mag_normalized = self._normalize(mag)
            record["kspace"] = (mag_normalized * np.exp(1j * phase)).astype(np.complex64)
        
        return record


class ZScoreNormalize(Normalize):
    """Z-score normalization (standardization).
    
    Transforms data to have zero mean and unit variance:
        output = (input - mean) / std
    
    This is the most common normalization in deep learning.
    Equivalent to TorchIO's ZNormalization.
    """
    
    def __init__(self, 
                 masking_method: Optional[str] = None,
                 apply_to: str = "image"):
        """
        Initialize Z-score normalization.
        
        Args:
            masking_method: Method to compute statistics (None, "otsu", "percentile").
            apply_to: Which data to normalize ("image" or "kspace").
        
        Example:
            # Basic z-score normalization
            norm = ZScoreNormalize()
            
            # Only compute stats on foreground
            norm = ZScoreNormalize(masking_method="percentile")
        """
        super().__init__(apply_to=apply_to, masking_method=masking_method)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        mask = self._compute_mask(image)
        
        # Compute statistics on masked region
        masked_values = image[mask]
        mean = float(np.mean(masked_values))
        std = float(np.std(masked_values))
        
        # Avoid division by zero
        if std < 1e-10:
            std = 1.0
        
        # Normalize
        normalized = (image - mean) / std
        
        return normalized.astype(np.float32)


class MinMaxNormalize(Normalize):
    """Min-Max normalization (scaling).
    
    Scales data to a specified range (default [0, 1]):
        output = (input - min) / (max - min) * (out_max - out_min) + out_min
    
    Equivalent to TorchIO's RescaleIntensity.
    """
    
    def __init__(self,
                 out_min: float = 0.0,
                 out_max: float = 1.0,
                 percentiles: Optional[Tuple[float, float]] = None,
                 masking_method: Optional[str] = None,
                 apply_to: str = "image"):
        """
        Initialize min-max normalization.
        
        Args:
            out_min: Minimum value of output range. Default: 0.0
            out_max: Maximum value of output range. Default: 1.0
            percentiles: If provided, use percentile values instead of min/max.
                        Example: (1, 99) uses 1st and 99th percentiles.
                        Helps with outliers.
            masking_method: Method to compute statistics.
            apply_to: Which data to normalize ("image" or "kspace").
        
        Example:
            # Scale to [0, 1]
            norm = MinMaxNormalize()
            
            # Scale to [-1, 1]
            norm = MinMaxNormalize(out_min=-1.0, out_max=1.0)
            
            # Use percentiles to handle outliers
            norm = MinMaxNormalize(percentiles=(1, 99))
        """
        super().__init__(apply_to=apply_to, masking_method=masking_method)
        self.out_min = out_min
        self.out_max = out_max
        self.percentiles = percentiles
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply min-max normalization."""
        mask = self._compute_mask(image)
        masked_values = image[mask]
        
        # Compute min/max (or percentiles)
        if self.percentiles is not None:
            in_min = float(np.percentile(masked_values, self.percentiles[0]))
            in_max = float(np.percentile(masked_values, self.percentiles[1]))
        else:
            in_min = float(np.min(masked_values))
            in_max = float(np.max(masked_values))
        
        # Avoid division by zero
        if abs(in_max - in_min) < 1e-10:
            return np.full_like(image, self.out_min, dtype=np.float32)
        
        # Scale to [0, 1]
        normalized = (image - in_min) / (in_max - in_min)
        
        # Scale to [out_min, out_max]
        normalized = normalized * (self.out_max - self.out_min) + self.out_min
        
        # Clip to output range
        normalized = np.clip(normalized, self.out_min, self.out_max)
        
        return normalized.astype(np.float32)


class PercentileNormalize(Normalize):
    """Percentile-based normalization.
    
    Normalizes based on specific percentile values.
    Useful for handling outliers and varying intensity distributions.
    """
    
    def __init__(self,
                 lower_percentile: float = 1.0,
                 upper_percentile: float = 99.0,
                 output_range: Tuple[float, float] = (0.0, 1.0),
                 masking_method: Optional[str] = None,
                 apply_to: str = "image"):
        """
        Initialize percentile normalization.
        
        Args:
            lower_percentile: Lower percentile to use as minimum (0-100).
                            Default: 1.0 (1st percentile).
            upper_percentile: Upper percentile to use as maximum (0-100).
                            Default: 99.0 (99th percentile).
            output_range: Target range for output values.
                         Default: (0.0, 1.0).
            masking_method: Method to compute statistics.
            apply_to: Which data to normalize ("image" or "kspace").
        
        Example:
            # Robust normalization (clips outliers)
            norm = PercentileNormalize(lower_percentile=1, upper_percentile=99)
            
            # More aggressive outlier clipping
            norm = PercentileNormalize(lower_percentile=5, upper_percentile=95)
        """
        super().__init__(apply_to=apply_to, masking_method=masking_method)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.output_range = output_range
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply percentile normalization."""
        mask = self._compute_mask(image)
        masked_values = image[mask]
        
        # Compute percentile values
        p_low = float(np.percentile(masked_values, self.lower_percentile))
        p_high = float(np.percentile(masked_values, self.upper_percentile))
        
        # Avoid division by zero
        if abs(p_high - p_low) < 1e-10:
            return np.full_like(image, self.output_range[0], dtype=np.float32)
        
        # Normalize to [0, 1]
        normalized = (image - p_low) / (p_high - p_low)
        
        # Scale to output range
        out_min, out_max = self.output_range
        normalized = normalized * (out_max - out_min) + out_min
        
        # Clip to output range
        normalized = np.clip(normalized, out_min, out_max)
        
        return normalized.astype(np.float32)


class RobustNormalize(Normalize):
    """Robust normalization using median and IQR.
    
    Uses median and interquartile range (IQR) instead of mean and std.
    More robust to outliers than z-score normalization.
    
    Formula:
        output = (input - median) / IQR
    """
    
    def __init__(self,
                 masking_method: Optional[str] = None,
                 apply_to: str = "image"):
        """
        Initialize robust normalization.
        
        Args:
            masking_method: Method to compute statistics.
            apply_to: Which data to normalize ("image" or "kspace").
        
        Example:
            # Robust standardization
            norm = RobustNormalize()
            
            # With foreground masking
            norm = RobustNormalize(masking_method="percentile")
        """
        super().__init__(apply_to=apply_to, masking_method=masking_method)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply robust normalization."""
        mask = self._compute_mask(image)
        masked_values = image[mask]
        
        # Compute robust statistics
        median = float(np.median(masked_values))
        q25 = float(np.percentile(masked_values, 25))
        q75 = float(np.percentile(masked_values, 75))
        iqr = q75 - q25
        
        # Avoid division by zero
        if iqr < 1e-10:
            iqr = 1.0
        
        # Normalize
        normalized = (image - median) / iqr
        
        return normalized.astype(np.float32)


# Convenience aliases (following sklearn/TorchIO conventions)
StandardScaler = ZScoreNormalize
MinMaxScaler = MinMaxNormalize
RobustScaler = RobustNormalize

