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
    "A": {"name": "A_baseline", "augmentation": False, "dice_weight": 0.0},
    "B": {"name": "B_augmentation", "augmentation": True, "dice_weight": 0.0},
    "C": {"name": "C_augmentation_dice", "augmentation": True, "dice_weight": 1.0},
}


def _device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate(model, loader, device, amp, dice_weight):
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
    model = UNet(8).to(target_device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    augmentation_generator = torch.Generator(device=target_device).manual_seed(43)
    best_dice = -1.0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for x, y, valid, _ in train_loader:
            x = x.to(target_device).float() / 255
            y = y.to(target_device).float()
            valid = valid.to(target_device).float()
            if spec["augmentation"]:
                x, y, valid = paired_augment(x, y, valid, augmentation_generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(target_device.type, dtype=torch.float16, enabled=amp):
                logits = model((x * 2 - 1).contiguous(memory_format=torch.channels_last))
            loss = segmentation_loss(logits, y, valid, spec["dice_weight"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(x)
            count += len(x)
        validation = _evaluate(model, val_loader, target_device, amp, spec["dice_weight"])
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": loss_sum / count,
               "val_loss": validation["loss"], "val_dice": validation["dice"],
               "val_iou": validation["iou"], "val_recall": validation["recall"],
               "val_precision": validation["precision"]}
        history.append(row)
        print(f"{spec['name']} epoch {epoch + 1:02d}/{epochs} "
              f"train_loss={row['train_loss']:.4f} val_Dice={row['val_dice']:.4f}", flush=True)
        if row["val_dice"] > best_dice:
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
