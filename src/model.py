import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Small U-Net used by the course project."""
    def __init__(self, base_channels: int = 8):
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4,
                  base_channels * 8, base_channels * 16]
        self.enc = nn.ModuleList(
            [DoubleConv(1, widths[0])] +
            [DoubleConv(a, b) for a, b in zip(widths[:-1], widths[1:])]
        )
        self.dec = nn.ModuleList(
            [DoubleConv(widths[i] + widths[i - 1], widths[i - 1])
             for i in range(4, 0, -1)]
        )
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for level, block in enumerate(self.enc):
            if level:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.out(x)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def paired_augment(images, masks, valid, generator):
    """Apply the same geometry to image, mask, and valid-area map."""
    batch = images.shape[0]
    device = images.device
    angle = (torch.rand(batch, device=device, generator=generator) * 2 - 1) * (10 * np.pi / 180)
    flip = torch.where(torch.rand(batch, device=device, generator=generator) < 0.5, -1.0, 1.0)
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = angle.cos() * flip
    theta[:, 0, 1] = -angle.sin()
    theta[:, 1, 0] = angle.sin() * flip
    theta[:, 1, 1] = angle.cos()
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    masks = F.grid_sample(masks, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    valid = F.grid_sample(valid, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    gain = 0.9 + 0.2 * torch.rand(batch, 1, 1, 1, device=device, generator=generator)
    offset = (torch.rand(batch, 1, 1, 1, device=device, generator=generator) * 2 - 1) * 0.05
    images = (images * gain + offset).clamp(0, 1) * valid
    return images, masks, valid


def segmentation_loss(logits, target, valid, dice_weight=0.0):
    logits = logits.float()
    target = target.float()
    valid = valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum()
    bce = bce / valid.sum().clamp_min(1)
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    soft_dice = (2 * (probability * target * valid).sum(dims) + 1) / (
        (probability * valid).sum(dims) + (target * valid).sum(dims) + 1
    )
    return bce + dice_weight * (1 - soft_dice).mean()


def per_image_metrics(logits, target, valid, threshold=0.5):
    prediction = torch.sigmoid(logits.float()) >= threshold
    truth = target > 0.5
    keep = valid > 0.5
    dims = (1, 2, 3)
    tp = (prediction & truth & keep).sum(dims).float()
    fp = (prediction & ~truth & keep).sum(dims).float()
    fn = (~prediction & truth & keep).sum(dims).float()
    tn = (~prediction & ~truth & keep).sum(dims).float()

    def safe(numerator, denominator):
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1), torch.ones_like(denominator))

    return {
        "dice": safe(2 * tp, 2 * tp + fp + fn),
        "iou": safe(tp, tp + fp + fn),
        "recall": safe(tp, tp + fn),
        "precision": tp / (tp + fp).clamp_min(1),
        "pixel_accuracy": safe(tp + tn, tp + tn + fp + fn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
