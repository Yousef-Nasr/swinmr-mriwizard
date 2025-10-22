"""
SSIM Loss Implementation for MRI Reconstruction

Structural Similarity Index (SSIM) loss for image quality assessment.
Particularly useful for MRI reconstruction as it captures perceptual quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIMLoss(nn.Module):
    """
    SSIM Loss Module

    Computes the Structural Similarity Index loss between predicted and target images.
    SSIM measures perceived quality based on luminance, contrast, and structure.

    Args:
        window_size: Size of Gaussian window (default: 11)
        size_average: If True, average loss over batch (default: True)
        channel: Number of input channels (default: 1 for grayscale MRI)
        data_range: Range of input data (default: 1.0 for normalized images)
    """

    def __init__(self, window_size=11, size_average=True, channel=1, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.data_range = data_range
        self.window = self._create_window(window_size, channel)

    def _gaussian(self, window_size, sigma):
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
        Calculate SSIM between two images.

        Args:
            img1: First image (B, C, H, W)
            img2: Second image (B, C, H, W)
            window: Gaussian window
            window_size: Size of window
            channel: Number of channels
            size_average: Average over batch

        Returns:
            SSIM value (higher is better, range [-1, 1])
        """
        # Constants for numerical stability
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        # Ensure window is on the same device as input
        if window.device != img1.device or window.dtype != img1.dtype:
            window = window.to(img1.device).type(img1.dtype)

        # Calculate local means
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate local variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        """
        Forward pass - compute SSIM loss.

        Args:
            img1: Predicted image (B, C, H, W)
            img2: Target image (B, C, H, W)

        Returns:
            SSIM loss (lower is better, range [0, 2])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        # Compute SSIM
        ssim_value = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

        # Return loss (1 - SSIM) so lower is better
        return 1 - ssim_value


class MS_SSIMLoss(nn.Module):
    """
    Multi-Scale SSIM Loss

    Computes SSIM at multiple scales for better perceptual quality assessment.
    Uses a weighted combination of SSIM computed at different resolutions.

    Args:
        window_size: Size of Gaussian window (default: 11)
        size_average: If True, average loss over batch (default: True)
        channel: Number of input channels (default: 1 for grayscale MRI)
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        data_range: Range of input data (default: 1.0)
    """

    def __init__(self, window_size=11, size_average=True, channel=1,
                 weights=None, data_range=1.0):
        super(MS_SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.data_range = data_range

        if weights is None:
            self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        else:
            self.weights = torch.FloatTensor(weights)

        self.ssim_loss = SSIMLoss(window_size, size_average, channel, data_range)

    def forward(self, img1, img2):
        """
        Forward pass - compute MS-SSIM loss.

        Args:
            img1: Predicted image (B, C, H, W)
            img2: Target image (B, C, H, W)

        Returns:
            MS-SSIM loss (lower is better)
        """
        device = img1.device
        weights = self.weights.to(device)

        levels = weights.size(0)
        mssim = []

        for i in range(levels):
            ssim_val = 1 - self.ssim_loss(img1, img2)  # Get SSIM (not loss)
            mssim.append(ssim_val)

            # Downsample for next scale (except last level)
            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

        mssim = torch.stack(mssim)

        # Weighted combination of scales
        ms_ssim_value = torch.prod(mssim ** weights.view(-1, 1))

        # Return loss (1 - MS-SSIM)
        return 1 - ms_ssim_value


class MixedLoss(nn.Module):
    """
    Mixed Loss combining L1/L2 with SSIM

    Combines pixel-wise loss (L1 or L2) with SSIM for better reconstruction quality.
    Common practice in image restoration tasks.

    Args:
        alpha: Weight for pixel loss (default: 0.84)
        beta: Weight for SSIM loss (default: 0.16)
        pixel_loss: 'l1' or 'l2' (default: 'l1')
        window_size: SSIM window size (default: 11)
        channel: Number of channels (default: 1)
        data_range: Data range (default: 1.0)
    """

    def __init__(self, alpha=0.84, beta=0.16, pixel_loss='l1',
                 window_size=11, channel=1, data_range=1.0):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        if pixel_loss == 'l1':
            self.pixel_loss = nn.L1Loss()
        elif pixel_loss == 'l2':
            self.pixel_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown pixel loss: {pixel_loss}")

        self.ssim_loss = SSIMLoss(window_size, True, channel, data_range)

    def forward(self, pred, target):
        """
        Forward pass - compute mixed loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Mixed loss value
        """
        pixel_loss_val = self.pixel_loss(pred, target)
        ssim_loss_val = self.ssim_loss(pred, target)

        return self.alpha * pixel_loss_val + self.beta * ssim_loss_val


def ssim(img1, img2, window_size=11, size_average=True, data_range=1.0):
    """
    Functional interface for SSIM calculation.

    Args:
        img1: First image (B, C, H, W)
        img2: Second image (B, C, H, W)
        window_size: Size of Gaussian window
        size_average: Average over batch
        data_range: Range of data

    Returns:
        SSIM value (higher is better, range [-1, 1])
    """
    (_, channel, _, _) = img1.size()
    ssim_loss = SSIMLoss(window_size, size_average, channel, data_range)
    return 1 - ssim_loss(img1, img2)


def ms_ssim(img1, img2, window_size=11, size_average=True, weights=None, data_range=1.0):
    """
    Functional interface for MS-SSIM calculation.

    Args:
        img1: First image (B, C, H, W)
        img2: Second image (B, C, H, W)
        window_size: Size of Gaussian window
        size_average: Average over batch
        weights: Weights for each scale
        data_range: Range of data

    Returns:
        MS-SSIM value (higher is better)
    """
    (_, channel, _, _) = img1.size()
    ms_ssim_loss = MS_SSIMLoss(window_size, size_average, channel, weights, data_range)
    return 1 - ms_ssim_loss(img1, img2)


# Convenience aliases
SSIM = SSIMLoss
MS_SSIM = MS_SSIMLoss


if __name__ == '__main__':
    """Test SSIM loss implementations."""

    print("="*60)
    print("Testing SSIM Loss Implementation")
    print("="*60)

    # Create test images
    batch_size = 4
    channels = 1
    height, width = 256, 256

    # Perfect reconstruction (loss should be ~0)
    img1 = torch.rand(batch_size, channels, height, width)
    img2 = img1.clone()

    print("\n1. Testing SSIMLoss (identical images):")
    ssim_loss = SSIMLoss(window_size=11, channel=channels)
    loss = ssim_loss(img1, img2)
    print(f"   Loss: {loss.item():.6f} (should be ~0)")

    # Different images
    img3 = torch.rand(batch_size, channels, height, width)
    loss2 = ssim_loss(img1, img3)
    print(f"\n2. Testing SSIMLoss (different images):")
    print(f"   Loss: {loss2.item():.6f} (should be > 0)")

    # MS-SSIM
    print(f"\n3. Testing MS-SSIMLoss:")
    ms_ssim_loss = MS_SSIMLoss(window_size=11, channel=channels)
    loss3 = ms_ssim_loss(img1, img3)
    print(f"   Loss: {loss3.item():.6f}")

    # Mixed Loss
    print(f"\n4. Testing MixedLoss (L1 + SSIM):")
    mixed_loss = MixedLoss(alpha=0.84, beta=0.16, pixel_loss='l1', channel=channels)
    loss4 = mixed_loss(img1, img3)
    print(f"   Loss: {loss4.item():.6f}")

    # Functional interface
    print(f"\n5. Testing functional ssim():")
    ssim_value = ssim(img1, img2)
    print(f"   SSIM: {ssim_value.item():.6f} (should be ~1.0)")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
