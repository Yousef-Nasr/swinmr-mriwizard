import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd
import math


"""
Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2*): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7*): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16*): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""


# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss



def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.
        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


# PyTorch
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


def mask_to_onehot(net_output, gt):
    """
    net_output must be (b, c, x, y(, z)))
    mask with shape (b, 1, x, y(, z)) OR shape (b, x, y(, z)))
    """
    shp_x = net_output.shape
    shp_y = gt.shape
    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
            # print(y_onehot)
    return y_onehot

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        # predict = F.softmax(predict, dim=1)
        # predict = F.sigmoid(predict, dim=1)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class FFTLoss(nn.Module):
    """
    FFT (Frequency Domain) Loss for MRI Reconstruction

    Computes loss in the frequency domain using Fast Fourier Transform.
    Highly beneficial for MRI because:
    - MRI data is acquired in k-space (frequency domain)
    - Preserves frequency structure and fine details
    - Complements spatial domain losses
    - Common in state-of-the-art MRI reconstruction

    Args:
        loss_type: Type of loss ('l1', 'l2', or 'combined')
        reduction: 'mean', 'sum', or 'none'
        alpha: Weight for magnitude loss (default: 1.0)
        beta: Weight for phase loss (default: 0.0, phase usually less important)
    """

    def __init__(self, loss_type='l1', reduction='mean', alpha=1.0, beta=0.0):
        super(FFTLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.alpha = alpha  # Magnitude weight
        self.beta = beta    # Phase weight

        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.loss_type == 'combined':
            self.loss_l1 = nn.L1Loss(reduction=reduction)
            self.loss_l2 = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, pred, target):
        """
        Compute FFT loss between predicted and target images.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            FFT loss value
        """
        # Compute FFT of both images
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        # Compute magnitude loss (most important for MRI)
        if self.alpha > 0:
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)

            if self.loss_type == 'l1':
                mag_loss = self.loss_fn(pred_mag, target_mag)
            elif self.loss_type == 'l2':
                mag_loss = self.loss_fn(pred_mag, target_mag)
            elif self.loss_type == 'combined':
                mag_loss = 0.5 * self.loss_l1(pred_mag, target_mag) + \
                          0.5 * self.loss_l2(pred_mag, target_mag)
        else:
            mag_loss = 0.0

        # Compute phase loss (optional, usually less important)
        if self.beta > 0:
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)

            if self.loss_type == 'l1':
                phase_loss = self.loss_fn(pred_phase, target_phase)
            elif self.loss_type == 'l2':
                phase_loss = self.loss_fn(pred_phase, target_phase)
            elif self.loss_type == 'combined':
                phase_loss = 0.5 * self.loss_l1(pred_phase, target_phase) + \
                            0.5 * self.loss_l2(pred_phase, target_phase)
        else:
            phase_loss = 0.0

        # Combined loss
        total_loss = self.alpha * mag_loss + self.beta * phase_loss

        return total_loss


class KSpaceLoss(nn.Module):
    """
    K-Space Loss for MRI Reconstruction

    Similar to FFT loss but explicitly designed for MRI k-space.
    Optionally applies undersampling mask to focus on acquired k-space data.

    Args:
        loss_type: Type of loss ('l1' or 'l2')
        reduction: 'mean', 'sum', or 'none'
        use_mask: If True, apply undersampling mask (if provided)
        weight_center: Extra weight for k-space center (default: 1.0)
    """

    def __init__(self, loss_type='l1', reduction='mean', use_mask=False, weight_center=1.0):
        super(KSpaceLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        self.use_mask = use_mask
        self.weight_center = weight_center

        if self.loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, pred, target, mask=None):
        """
        Compute k-space loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)
            mask: Optional k-space mask (B, 1, H, W)

        Returns:
            K-space loss value
        """
        # Convert to k-space
        pred_kspace = torch.fft.fft2(pred, norm='ortho')
        target_kspace = torch.fft.fft2(target, norm='ortho')

        # Compute complex-valued loss (magnitude of difference)
        kspace_diff = torch.abs(pred_kspace - target_kspace)
        loss = self.loss_fn(kspace_diff, torch.zeros_like(kspace_diff))

        # Apply mask if provided and requested
        if self.use_mask and mask is not None:
            loss = loss * mask

        # Apply center weighting (k-space center is more important)
        if self.weight_center > 1.0:
            B, C, H, W = loss.shape
            center_h, center_w = H // 2, W // 2
            radius = min(H, W) // 8  # Center region radius

            # Create center weight map
            y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            dist = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2).to(loss.device)
            weight_map = torch.ones_like(dist)
            weight_map[dist <= radius] = self.weight_center
            weight_map = weight_map.unsqueeze(0).unsqueeze(0)

            loss = loss * weight_map

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined Multi-Component Loss for MRI Reconstruction

    Combines multiple loss types with configurable weights:
    - Spatial loss (L1/Charbonnier)
    - FFT/frequency loss
    - Perceptual loss (optional)
    - SSIM loss (optional)

    This is the recommended loss for MRI reconstruction!

    Args:
        spatial_weight: Weight for spatial loss (default: 1.0)
        fft_weight: Weight for FFT loss (default: 0.1)
        perceptual_weight: Weight for perceptual loss (default: 0.0)
        ssim_weight: Weight for SSIM loss (default: 0.0)
        spatial_loss_type: 'l1', 'l2', or 'charbonnier'
        fft_loss_type: 'l1' or 'l2'
    """

    def __init__(self, spatial_weight=1.0, fft_weight=0.1,
                 perceptual_weight=0.0, ssim_weight=0.0,
                 spatial_loss_type='charbonnier', fft_loss_type='l1'):
        super(CombinedLoss, self).__init__()

        self.spatial_weight = spatial_weight
        self.fft_weight = fft_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight

        # Spatial loss
        if spatial_loss_type == 'charbonnier':
            self.spatial_loss = CharbonnierLoss()
        elif spatial_loss_type == 'l1':
            self.spatial_loss = nn.L1Loss()
        elif spatial_loss_type == 'l2':
            self.spatial_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown spatial_loss_type: {spatial_loss_type}")

        # FFT loss
        if fft_weight > 0:
            self.fft_loss = FFTLoss(loss_type=fft_loss_type)

        # Perceptual loss
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()

        # SSIM loss
        if ssim_weight > 0:
            try:
                from .loss_ssim import SSIMLoss
                self.ssim_loss = SSIMLoss()
            except ImportError:
                print("Warning: SSIMLoss not available, skipping SSIM loss")
                self.ssim_weight = 0

    def forward(self, pred, target):
        """
        Compute combined loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Combined loss value
        """
        total_loss = 0.0
        loss_dict = {}

        # Spatial loss
        if self.spatial_weight > 0:
            spatial_loss = self.spatial_loss(pred, target)
            total_loss += self.spatial_weight * spatial_loss
            loss_dict['spatial'] = spatial_loss.item()

        # FFT loss
        if self.fft_weight > 0:
            fft_loss = self.fft_loss(pred, target)
            total_loss += self.fft_weight * fft_loss
            loss_dict['fft'] = fft_loss.item()

        # Perceptual loss
        if self.perceptual_weight > 0:
            # Convert grayscale to 3-channel for VGG
            if pred.shape[1] == 1:
                pred_3ch = pred.repeat(1, 3, 1, 1)
                target_3ch = target.repeat(1, 3, 1, 1)
            else:
                pred_3ch = pred
                target_3ch = target
            perceptual_loss = self.perceptual_loss(pred_3ch, target_3ch)
            total_loss += self.perceptual_weight * perceptual_loss
            loss_dict['perceptual'] = perceptual_loss.item()

        # SSIM loss
        if self.ssim_weight > 0:
            ssim_loss = self.ssim_loss(pred, target)
            total_loss += self.ssim_weight * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()

        return total_loss