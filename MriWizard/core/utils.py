"""Utility functions for FFTs, normalization, and dtype conversions."""

import numpy as np
from typing import Union, Tuple

def fft2c(data: np.ndarray) -> np.ndarray:
    """
    Centered 2D FFT.
    
    Args:
        data: Input array (can be 2D or higher-dimensional)
        
    Returns:
        FFT of data with proper centering
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(data, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

def ifft2c(data: np.ndarray) -> np.ndarray:
    """
    Centered 2D inverse FFT.
    
    Args:
        data: Input k-space array
        
    Returns:
        Image domain array
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(data, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))

def fftnc(data: np.ndarray, axes: Union[Tuple[int, ...], None] = None) -> np.ndarray:
    """
    Centered N-dimensional FFT.
    
    Args:
        data: Input array
        axes: Axes along which to perform FFT (default: all axes)
        
    Returns:
        FFT of data with proper centering
    """
    if axes is None:
        axes = tuple(range(data.ndim))
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data, axes=axes), axes=axes), axes=axes)

def ifftnc(data: np.ndarray, axes: Union[Tuple[int, ...], None] = None) -> np.ndarray:
    """
    Centered N-dimensional inverse FFT.
    
    Args:
        data: Input k-space array
        axes: Axes along which to perform IFFT (default: all axes)
        
    Returns:
        Image domain array
    """
    if axes is None:
        axes = tuple(range(data.ndim))
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(data, axes=axes), axes=axes), axes=axes)

def to_complex64(data: np.ndarray) -> np.ndarray:
    """Convert array to complex64."""
    if np.iscomplexobj(data):
        return data.astype(np.complex64)
    else:
        return data.astype(np.float32).astype(np.complex64)

def to_float32(data: np.ndarray) -> np.ndarray:
    """Convert array to float32, taking magnitude if complex."""
    if np.iscomplexobj(data):
        return np.abs(data).astype(np.float32)
    else:
        return data.astype(np.float32)

def normalize(data: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Normalize array to [0, 1] based on percentile.
    
    Args:
        data: Input array
        percentile: Percentile for normalization (default: 99)
        
    Returns:
        Normalized array
    """
    data = to_float32(data)
    max_val = np.percentile(data, percentile)
    if max_val > 0:
        return np.clip(data / max_val, 0, 1)
    return data

def validate_kspace(kspace: np.ndarray) -> None:
    """
    Validate that k-space array has proper shape and dtype.
    
    Args:
        kspace: k-space array to validate
        
    Raises:
        ValueError: If k-space is invalid
    """
    if not np.iscomplexobj(kspace):
        raise ValueError("k-space must be complex-valued")
    if kspace.ndim < 2:
        raise ValueError(f"k-space must have at least 2 dimensions, got {kspace.ndim}")

def validate_image(image: np.ndarray) -> None:
    """
    Validate that image array has proper shape.
    
    Args:
        image: Image array to validate
        
    Raises:
        ValueError: If image is invalid
    """
    if image.ndim < 2:
        raise ValueError(f"Image must have at least 2 dimensions, got {image.ndim}")

