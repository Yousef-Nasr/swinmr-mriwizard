"""Loader for standard image formats (jpg, png, tiff)."""

import numpy as np
from pathlib import Path
from MriWizard.core.base import Record
from MriWizard.core.utils import to_float32, fft2c

class LoadImage:
    """Load standard image files (jpg, png, tiff, etc.)."""
    
    def __init__(self, convert_to_kspace: bool = False, grayscale: bool = True):
        """
        Initialize image loader.
        
        Args:
            convert_to_kspace: If True, convert image to k-space via FFT
            grayscale: If True, convert to grayscale
        """
        self.convert_to_kspace = convert_to_kspace
        self.grayscale = grayscale
    
    def load(self, path: str) -> Record:
        """
        Load image file.
        
        Args:
            path: Path to image file
            
        Returns:
            Record with image (and optionally kspace) and metadata
        """
        import imageio.v3 as iio
        
        image = iio.imread(str(path))
        
        # Convert to grayscale if needed
        if self.grayscale and image.ndim == 3:
            # Simple RGB to grayscale conversion
            if image.shape[-1] == 3:
                image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            elif image.shape[-1] == 4:  # RGBA
                image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
        image = to_float32(image)
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        metadata = {
            "source": str(path),
            "applied": []
        }
        
        # Convert to k-space if requested
        kspace = None
        if self.convert_to_kspace:
            kspace = fft2c(image.astype(np.complex64))
        
        return {
            "kspace": kspace,
            "image": image,
            "mask": None,
            "metadata": metadata
        }
    
    def __call__(self, record: Record) -> Record:
        """Allow use as a transform in pipeline."""
        return record

