"""Gaussian blur via FFT-based convolution."""

import numpy as np
from typing import Union, Tuple
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, ifft2c, to_complex64, to_float32

class RandomGaussianBlurImage:
    """Apply Gaussian blur in image domain via FFT convolution."""
    
    def __init__(self, sigma_px_range: Union[Tuple[float, float], None] = None):
        """
        Initialize Gaussian blur transform.
        
        Args:
            sigma_px_range: Range for Gaussian sigma in pixels (min, max)
        
        Blur is applied in image domain using FFT-based convolution.
        """
        if sigma_px_range is None:
            sigma_px_range = (0.5, 2.0)
        
        self.sigma_px_range = sigma_px_range
    
    def _create_gaussian_kernel(self, shape: Tuple[int, ...], sigma: float) -> np.ndarray:
        """Create Gaussian kernel in frequency domain."""
        # Create frequency grids
        grids = []
        for size in shape:
            freq = np.fft.fftfreq(size).astype(np.float32)
            grids.append(freq)
        
        # Create meshgrid
        if len(shape) == 2:
            ky, kx = np.meshgrid(grids[0], grids[1], indexing='ij')
            k_squared = kx**2 + ky**2
        elif len(shape) == 3:
            kz, ky, kx = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
            k_squared = kx**2 + ky**2 + kz**2
        else:
            k_squared = sum(g**2 for g in np.meshgrid(*grids, indexing='ij'))
        
        # Gaussian in frequency domain
        gaussian_fft = np.exp(-2.0 * (np.pi * sigma)**2 * k_squared).astype(np.float32)
        
        return gaussian_fft
    
    def __call__(self, record: Record) -> Record:
        """
        Apply Gaussian blur to image.
        
        Args:
            record: Input record
            
        Returns:
            Record with blurred image/kspace
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
        
        # Sample sigma
        sigma = np.random.uniform(
            self.sigma_px_range[0],
            self.sigma_px_range[1]
        )
        
        # Get spatial shape (last 2 dimensions)
        spatial_shape = image.shape[-2:]
        
        # Create Gaussian kernel in frequency domain
        gaussian_kernel = self._create_gaussian_kernel(spatial_shape, sigma)
        
        # Process each slice/channel if multi-dimensional
        if image.ndim == 2:
            # Simple 2D image
            image_fft = np.fft.fftn(image)
            blurred_fft = image_fft * gaussian_kernel
            blurred_image = np.real(np.fft.ifftn(blurred_fft)).astype(np.float32)
        else:
            # Multiple slices/channels
            blurred_image = np.zeros_like(image)
            
            # Expand kernel for broadcasting
            kernel_broadcast = gaussian_kernel
            for _ in range(image.ndim - 2):
                kernel_broadcast = np.expand_dims(kernel_broadcast, axis=0)
            kernel_broadcast = np.broadcast_to(kernel_broadcast, image.shape)
            
            # Apply blur via FFT
            image_fft = np.fft.fftn(image, axes=(-2, -1))
            blurred_fft = image_fft * kernel_broadcast
            blurred_image = np.real(np.fft.ifftn(blurred_fft, axes=(-2, -1))).astype(np.float32)
        
        # Update record
        if had_kspace_only:
            # Convert back to k-space
            record["kspace"] = fft2c(to_complex64(blurred_image))
        else:
            record["image"] = blurred_image
            # Also update k-space if it exists
            if record["kspace"] is not None:
                record["kspace"] = fft2c(to_complex64(blurred_image))
        
        record["metadata"]["applied"].append({
            "transform": "RandomGaussianBlurImage",
            "sigma_px": float(sigma)
        })
        
        return record

