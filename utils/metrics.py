"""Metrics for image quality assessment: PSNR, SSIM, LPIPS."""

import numpy as np
import torch
from typing import Union, Tuple


def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def calculate_psnr(img1: Union[torch.Tensor, np.ndarray], 
                  img2: Union[torch.Tensor, np.ndarray],
                  data_range: float = 1.0) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    
    Args:
        img1: Predicted/generated image
        img2: Ground truth image
        data_range: Maximum value of data range (typically 1.0 for normalized, 255 for uint8)
        
    Returns:
        PSNR value in dB
    """
    img1 = tensor_to_numpy(img1)
    img2 = tensor_to_numpy(img2)
    
    # Handle batches - compute mean across all samples
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100.0  # Identical images
    
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)


def calculate_ssim(img1: Union[torch.Tensor, np.ndarray],
                  img2: Union[torch.Tensor, np.ndarray],
                  window_size: int = 11,
                  sigma: float = 1.5,
                  data_range: float = 1.0) -> float:
    """
    Calculate SSIM (Structural Similarity Index Measure).
    
    Args:
        img1: Predicted/generated image
        img2: Ground truth image
        window_size: Gaussian window size
        sigma: Gaussian window standard deviation
        data_range: Maximum value of data range
        
    Returns:
        SSIM value (between -1 and 1, typically 0 to 1)
    """
    img1 = tensor_to_numpy(img1)
    img2 = tensor_to_numpy(img2)
    
    # Handle batches
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()
    
    # Ensure float32
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Create Gaussian window
    kernel = _gaussian_kernel(window_size, sigma)
    
    # Constants for SSIM
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    # Compute SSIM
    mu1 = _convolve2d(img1, kernel)
    mu2 = _convolve2d(img2, kernel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = _convolve2d(img1 * img1, kernel) - mu1_sq
    sigma2_sq = _convolve2d(img2 * img2, kernel) - mu2_sq
    sigma12 = _convolve2d(img1 * img2, kernel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(np.mean(ssim_map))


def calculate_lpips(img1: Union[torch.Tensor, np.ndarray],
                   img2: Union[torch.Tensor, np.ndarray],
                   net: str = 'alex') -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).
    
    Requires: pip install lpips
    
    Args:
        img1: Predicted/generated image (tensor or numpy)
        img2: Ground truth image (tensor or numpy)
        net: Backbone network ('alex', 'vgg', 'squeeze')
        
    Returns:
        LPIPS distance
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("LPIPS not available. Install: pip install lpips")
    
    # Convert to tensors if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Ensure right shape: (B, C, H, W)
    if img1.ndim == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
    elif img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    
    if img2.ndim == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img2.ndim == 3:
        img2 = img2.unsqueeze(0)
    
    # For grayscale, expand to 3 channels for LPIPS
    if img1.shape[1] == 1:
        img1 = img1.repeat(1, 3, 1, 1)
    if img2.shape[1] == 1:
        img2 = img2.repeat(1, 3, 1, 1)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = img1.to(device)
    img2 = img2.to(device)
    
    # Compute LPIPS
    loss_fn = lpips.LPIPS(net=net, verbose=False).to(device)
    with torch.no_grad():
        lpips_score = loss_fn(img1, img2)
    
    return float(lpips_score.item())


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create Gaussian kernel."""
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution for SSIM computation."""
    from scipy import signal
    return signal.convolve2d(img, kernel, mode='valid', boundary='symm')


class MetricsComputer:
    """Compute multiple metrics for batch evaluation."""
    
    def __init__(self, metrics: list = None, data_range: float = 1.0):
        """
        Initialize metrics computer.
        
        Args:
            metrics: List of metrics to compute ('psnr', 'ssim', 'lpips')
            data_range: Data range for PSNR/SSIM
        """
        if metrics is None:
            metrics = ['psnr', 'ssim']
        
        self.metrics = metrics
        self.data_range = data_range
        self.results = {m: [] for m in metrics}
    
    def compute(self, pred: Union[torch.Tensor, np.ndarray],
               target: Union[torch.Tensor, np.ndarray]) -> dict:
        """
        Compute all metrics for a sample.
        
        Args:
            pred: Predicted image
            target: Target image
            
        Returns:
            Dictionary of computed metrics
        """
        results = {}
        
        for metric in self.metrics:
            if metric.lower() == 'psnr':
                value = calculate_psnr(pred, target, self.data_range)
                results['psnr'] = value
                self.results['psnr'].append(value)
            
            elif metric.lower() == 'ssim':
                value = calculate_ssim(pred, target, data_range=self.data_range)
                results['ssim'] = value
                self.results['ssim'].append(value)
            
            elif metric.lower() == 'lpips':
                try:
                    value = calculate_lpips(pred, target)
                    results['lpips'] = value
                    self.results['lpips'].append(value)
                except Exception as e:
                    print(f"Warning: LPIPS computation failed: {e}")
        
        return results
    
    def get_average(self) -> dict:
        """Get average of all computed metrics."""
        averages = {}
        for metric, values in self.results.items():
            if len(values) > 0:
                averages[metric] = np.mean(values)
        return averages
    
    def reset(self):
        """Reset results."""
        for metric in self.results:
            self.results[metric] = []
