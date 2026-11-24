"""训练 U-Net，并保存最佳模型、训练曲线和实验配置。

本文件对应课程实验的训练部分：准备批次、执行前向传播和反向传播，使用
优化器更新 CNN 参数，并在每轮结束后用验证集选择 Dice 最好的权重。
"""

from pathlib import Path
import csv
import json
import platform
import sys
import time
import hashlib
import torch
from torch.utils.data import DataLoader
from .data import make_loader
from .model import UNet, seed_everything, paired_augment, segmentation_loss, per_image_metrics

EXPERIMENTS = {
    # A：不做数据增强，只使用 BCE 损失，作为基础对照组。
    "A": {"name": "A_baseline", "augmentation": False, "dice_weight": 0.0},
    # B：在 A 的基础上加入同步几何/亮度增强，观察泛化能力变化。
    "B": {"name": "B_augmentation", "augmentation": True, "dice_weight": 0.0},
    # C：在 B 的基础上加入 Dice 损失，重点改善前景区域的重叠效果。
    "C": {"name": "C_augmentation_dice", "augmentation": True, "dice_weight": 1.0},
}


def _device(requested):
    """根据命令行选项决定使用 CPU 还是 CUDA。"""

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate(model, loader, device, amp, dice_weight):
    """在验证集上前向计算损失和分割指标，不更新模型参数。"""

    model.eval()
    values = {"dice": [], "iou": [], "recall": [], "precision": []}
    total_loss = 0.0
    total_count = 0
    with torch.inference_mode():
        for x, y, valid, _ in loader:
            x = x.to(device).float() / 255
            y = y.to(device).float()
            valid = valid.to(device).float()
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x * 2 - 1)
            total_loss += segmentation_loss(logits, y, valid, dice_weight).item() * len(x)
            total_count += len(x)
            scores = per_image_metrics(logits, y, valid)
            for key in values:
                values[key].extend(scores[key].cpu().tolist())
    return {key: sum(value) / len(value) for key, value in values.items()} | {"loss": total_loss / total_count}


def train(project_root, experiment="C", epochs=30, batch_size=64,
          device="auto", limit=None, image_size=192):
    """训练一个实验配置并保存验证 Dice 最好的模型。

    关键超参数说明：
    - ``epochs``：完整遍历训练集的轮数，默认 30；
    - ``batch_size``：每次参数更新使用的样本数，默认 64；
    - AdamW：负责根据梯度更新 CNN 的卷积核和其他可学习参数；
    - 初始学习率 0.001，使用 CosineAnnealingLR 逐步降低到 0.00001；
    - ``limit``：只取少量样本，用于快速检查代码，不用于正式结果。
    """

    project_root = Path(project_root)
    spec = EXPERIMENTS[experiment.upper()]
    seed_everything(42)
    target_device = _device(device)
    amp = target_device.type == "cuda"
    processed = project_root / "data" / "processed"
    if not (processed / "images.npy").exists():
        raise RuntimeError("Processed data not found. Run: python main.py prepare")
    train_loader = make_loader(project_root, "train", batch_size, shuffle=True, limit=limit)
    val_loader = make_loader(project_root, "val", batch_size, shuffle=False, limit=limit)
    # 建立轻量 U-Net。base_channels=8，第一层有 8 个特征通道，之后逐层
    # 翻倍到 16、32、64、128；卷积核细节见 model.py。
    model = UNet(8).to(target_device, memory_format=torch.channels_last)
    # AdamW 是本实验的优化器，更新模型中所有 requires_grad=True 的参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    # 余弦退火学习率调度器：训练后期使用更小步长，使模型收敛更平稳。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    # CUDA 下启用混合精度以减少显存占用；CPU 下自动关闭。
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    augmentation_generator = torch.Generator(device=target_device).manual_seed(43)
    best_dice = -1.0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum, count = 0.0, 0
        # 一个 epoch 由多个 batch 组成。每个 batch 都执行：取数据 -> 前向
        # 传播 -> 计算损失 -> 反向传播 -> 优化器更新参数。
        for x, y, valid, _ in train_loader:
            x = x.to(target_device).float() / 255
            y = y.to(target_device).float()
            valid = valid.to(target_device).float()
            if spec["augmentation"]:
                # 图像、mask、valid 同步增强，保持监督标签与输入对齐。
                x, y, valid = paired_augment(x, y, valid, augmentation_generator)
            # 清空上一批次累积的梯度。
            optimizer.zero_grad(set_to_none=True)
            # 前向传播：灰度图归一化到 [0,1] 后再变换到 [-1,1]，输入 U-Net。
            with torch.autocast(target_device.type, dtype=torch.float16, enabled=amp):
                logits = model((x * 2 - 1).contiguous(memory_format=torch.channels_last))
            # loss 由 BCE 和可选 Dice 项组成；valid 区域外的补边不计入损失。
            loss = segmentation_loss(logits, y, valid, spec["dice_weight"])
            # 反向传播得到卷积核等参数的梯度。
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 梯度裁剪，避免单个 batch 产生过大的梯度。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # 根据梯度更新参数，并更新混合精度缩放器。
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(x)
            count += len(x)
        validation = _evaluate(model, val_loader, target_device, amp, spec["dice_weight"])
        # 一个 epoch 完成后再调整学习率，并记录验证集结果。
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": loss_sum / count,
               "val_loss": validation["loss"], "val_dice": validation["dice"],
               "val_iou": validation["iou"], "val_recall": validation["recall"],
               "val_precision": validation["precision"]}
        history.append(row)
        print(f"{spec['name']} epoch {epoch + 1:02d}/{epochs} "
              f"train_loss={row['train_loss']:.4f} val_Dice={row['val_dice']:.4f}", flush=True)
        if row["val_dice"] > best_dice:
            # 只保存验证集 Dice 更高时的权重，避免最后一轮反而变差。
            best_dice = row["val_dice"]
            best_state = {"model": model.state_dict(),
                          "config": {"image_size": image_size, "base_channels": 8,
                                     "name": spec["name"], "augmentation": spec["augmentation"],
                                     "dice_weight": spec["dice_weight"]},
                          "epoch": epoch + 1, "best_val_dice": best_dice,
                          "split_sha256": hashlib.sha256((processed / "splits.json").read_bytes()).hexdigest()}
    result_dir = project_root / "results" / spec["name"]
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save(best_state, model_dir / f"{spec['name']}_best.pt")
    if experiment.upper() == "C":
        torch.save(best_state, model_dir / "best_model.pt")
    with (result_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (result_dir / "config.json").write_text(json.dumps(spec | {"epochs": epochs, "batch_size": batch_size}, indent=2), encoding="utf-8")
    (result_dir / "complete.json").write_text(json.dumps({
        "best_epoch": best_state["epoch"], "best_val_dice": best_dice,
        "epochs": epochs, "seconds": time.perf_counter() - started,
        "device": str(target_device), "train_limit": limit,
    }, indent=2), encoding="utf-8")
    return best_state
