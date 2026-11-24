"""数据准备和 PyTorch 数据加载。

负责读取 TN3K 原始图像及 mask，按保持宽高比的方式缩放到统一尺寸，保存
可复用的预处理缓存，并为训练、验证、测试阶段创建 DataLoader。
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_SIZE = 192


def prepare_data(project_root, size=DEFAULT_SIZE, force=False):
    project_root = Path(project_root)
    raw = project_root / "data" / "raw" / "TN3K"
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    required = [processed / name for name in ["images.npy", "masks.npy", "valid.npy", "manifest.json", "splits.json"]]
    if all(p.exists() for p in required) and not force:
        return json.loads((processed / "splits.json").read_text(encoding="utf-8"))

    fold_path = project_root / "data" / "raw" / "tn3k-trainval-fold0.json"
    fold = json.loads(fold_path.read_text(encoding="utf-8"))
    train_ids, val_ids = set(fold["train"]), set(fold["val"])
    jobs = []
    for subset in ["trainval", "test"]:
        image_dir = raw / f"{subset}-image"
        mask_dir = raw / f"{subset}-mask"
        image_paths = sorted(image_dir.glob("*.jpg"))
        for image_path in image_paths:
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            image_id = int(image_path.stem)
            split = "test" if subset == "test" else ("train" if image_id in train_ids else "val")
            jobs.append((image_path, mask_path, subset, split))
    if len(jobs) != 3493:
        raise RuntimeError(f"Expected 3493 image-mask pairs, found {len(jobs)}")

    images = np.zeros((len(jobs), size, size), dtype=np.uint8)
    masks = np.zeros_like(images)
    valid = np.zeros_like(images)
    manifest = []
    splits = {"train": [], "val": [], "test": []}
    for index, (image_path, mask_path, subset, split) in enumerate(jobs):
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {image_path}")
        width, height = image.size
        scale = min(size / width, size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        left, top = (size - new_width) // 2, (size - new_height) // 2
        region = (slice(top, top + new_height), slice(left, left + new_width))
        images[index][region] = np.asarray(image.resize((new_width, new_height), Image.Resampling.BILINEAR))
        masks[index][region] = np.asarray(mask.resize((new_width, new_height), Image.Resampling.NEAREST)) >= 128
        valid[index][region] = 1
        splits[split].append(index)
        manifest.append({"id": f"{subset}/{image_path.stem}", "split": split,
                         "image": str(image_path.relative_to(project_root)),
                         "mask": str(mask_path.relative_to(project_root)),
                         "original_size": [width, height]})

    np.save(processed / "images.npy", images)
    np.save(processed / "masks.npy", masks)
    np.save(processed / "valid.npy", valid)
    (processed / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (processed / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    (processed / "preprocessing.json").write_text(json.dumps({
        "size": size, "mask_threshold": 128,
        "image_resize": "bilinear", "mask_resize": "nearest",
        "padding": "aspect-ratio-preserving; ignored by loss and metrics",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits


class CachedDataset(Dataset):
    def __init__(self, processed_dir, indices):
        processed_dir = Path(processed_dir)
        self.images = np.load(processed_dir / "images.npy", mmap_mode="r")
        self.masks = np.load(processed_dir / "masks.npy", mmap_mode="r")
        self.valid = np.load(processed_dir / "valid.npy", mmap_mode="r")
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return (
            torch.from_numpy(self.images[index].copy()).unsqueeze(0),
            torch.from_numpy(self.masks[index].copy()).unsqueeze(0),
            torch.from_numpy(self.valid[index].copy()).unsqueeze(0),
            index,
        )


def make_loader(project_root, split, batch_size=16, shuffle=False, limit=None):
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    splits = json.loads((processed / "splits.json").read_text(encoding="utf-8"))
    indices = splits[split][:limit] if limit else splits[split]
    return DataLoader(CachedDataset(processed, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())
