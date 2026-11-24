"""模型、数据增强、损失函数和分割评价指标。

本文件是项目中 CNN 的核心实现：使用一个轻量级 U-Net 对甲状腺超声图像
进行像素级结节分割。网络输入为单通道灰度图，输出为每个像素属于结节的
概率对应的 logits。
"""

import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """U-Net 中的基本卷积块。

    每个卷积块包含两次 3×3 卷积。卷积核大小为 3，padding=1，因此空间
    尺寸在卷积块内保持不变；BatchNorm 用于稳定训练，ReLU 提供非线性。
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            # 3×3 卷积是 CNN 提取局部纹理和边缘特征的主要操作。
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """课程项目使用的轻量 U-Net。

    输入尺寸为 [batch, 1, 192, 192]，输出尺寸为 [batch, 1, 192, 192]。
    编码器逐步降低空间分辨率、增加通道数以提取语义特征；解码器逐步恢复
    分辨率，并通过 skip connection 保留结节边界等细节。最后的 1×1 卷积
    将特征图映射为一个前景/背景分割通道。
    """

    def __init__(self, base_channels: int = 8):
        super().__init__()
        # 通道数依次为 8、16、32、64、128。base_channels=8 使模型较轻量，
        # 适合课程实验；增大它可以提升容量，但会增加显存和训练时间。
        widths = [base_channels, base_channels * 2, base_channels * 4,
                  base_channels * 8, base_channels * 16]
        # 编码器：每层先做 DoubleConv，除第一层外再由 forward 中的 2×2
        # 最大池化将高、宽缩小一半。
        self.enc = nn.ModuleList(
            [DoubleConv(1, widths[0])] +
            [DoubleConv(a, b) for a, b in zip(widths[:-1], widths[1:])]
        )
        # 解码器：上采样后与对应编码器特征拼接，所以输入通道数是两路
        # 特征通道数之和，再用 DoubleConv 融合信息。
        self.dec = nn.ModuleList(
            [DoubleConv(widths[i] + widths[i - 1], widths[i - 1])
             for i in range(4, 0, -1)]
        )
        # 1×1 卷积只改变通道数，不改变图像尺寸；输出一个分割 logits 通道。
        self.out = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # 保存编码器各层输出，供解码器进行 skip connection。
        skips = []
        for level, block in enumerate(self.enc):
            if level:
                # 2×2 最大池化，步长默认为 2，使空间尺寸减半。
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        # 从最深层开始逐级恢复分辨率，并拼接同尺度的编码器特征。
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.out(x)


def seed_everything(seed: int = 42):
    """固定随机种子，尽量保证不同运行之间的结果可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def paired_augment(images, masks, valid, generator):
    """对图像、mask 和有效区域同步做数据增强。

    分割任务中图像和 mask 必须进行完全相同的几何变换，否则输入图像和
    标签会错位。这里使用小角度旋转、水平翻转以及亮度/对比度扰动。
    """

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
    """计算带有效区域掩码的 BCE + Dice 组合损失。

    ``logits`` 是模型未经 sigmoid 的输出；BCEWithLogits 内部会完成数值
    更稳定的 sigmoid。补边区域由 valid=0 标记，不参与损失计算。A/B 实验
    的 dice_weight=0，只使用 BCE；C 实验将 dice_weight 设为 1，同时优化
    像素分类和区域重叠。
    """

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
    """将 logits 二值化后，逐张图计算分割指标。

    sigmoid 概率大于等于 0.5 的像素视为结节前景，再根据 TP、FP、FN、TN
    计算 Dice、IoU、Recall、Precision 和 Pixel Accuracy。valid 区域之外
    的补边像素不参与统计。
    """

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
