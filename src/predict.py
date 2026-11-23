from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter
import torch
from .model import UNet


def predict_file(project_root, image_path, checkpoint=None, output_dir=None, device="auto"):
    project_root = Path(project_root)
    target_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
    checkpoint = Path(checkpoint) if checkpoint else project_root / "models" / "best_model.pt"
    output_dir = Path(output_dir) if output_dir else project_root / "results" / "predictions"
    state = torch.load(checkpoint, map_location=target_device, weights_only=True)
    config = state.get("config", {})
    size = config.get("image_size", 192)
    model = UNet(config.get("base_channels", 8)).to(target_device)
    model.load_state_dict(state["model"])
    model.eval()
    image_path = Path(image_path)
    image = Image.open(image_path).convert("L")
    width, height = image.size
    scale = min(size / width, size / height)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    canvas = np.zeros((size, size), dtype=np.float32)
    canvas[top:top + new_height, left:left + new_width] = np.asarray(
        image.resize((new_width, new_height), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255
    with torch.inference_mode():
        logits = model(torch.from_numpy(canvas)[None, None].to(target_device) * 2 - 1)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    crop = Image.fromarray(probability[top:top + new_height, left:left + new_width], mode="F")
    mask = (np.asarray(crop.resize((width, height), Image.Resampling.BILINEAR)) >= 0.5).astype(np.uint8) * 255
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_image = Image.fromarray(mask)
    mask_image.save(output_dir / f"{stem}_mask.png")
    rgb = np.asarray(image.convert("RGB"))
    overlay = rgb.copy()
    foreground = mask > 0
    overlay[foreground] = (0.65 * overlay[foreground] + 0.35 * np.array([40, 200, 255])).astype(np.uint8)
    edge = np.asarray(mask_image.filter(ImageFilter.MaxFilter(3))) != np.asarray(mask_image.filter(ImageFilter.MinFilter(3)))
    overlay[edge] = [20, 220, 255]
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")
    summary = {"input": str(image_path), "output": str(output_dir),
               "original_size": [width, height], "predicted_area_pixels": int(foreground.sum()),
               "checkpoint_epoch": state.get("epoch"), "threshold": 0.5}
    (output_dir / f"{stem}_prediction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary
