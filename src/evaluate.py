from pathlib import Path
import csv
import json
import math
import torch
from .data import make_loader
from .model import UNet, per_image_metrics
from .train import _device


def evaluate(project_root, experiment="C", device="auto", batch_size=16):
    project_root = Path(project_root)
    target_device = _device(device)
    model_path = project_root / "models" / ("best_model.pt" if experiment.upper() == "C" else f"{experiment}_best.pt")
    state = torch.load(model_path, map_location=target_device, weights_only=True)
    model = UNet(state.get("config", {}).get("base_channels", 8)).to(target_device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = make_loader(project_root, "test", batch_size=batch_size)
    processed = project_root / "data" / "processed"
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    with torch.inference_mode():
        for x, y, valid, indices in loader:
            x = x.to(target_device).float() / 255
            y = y.to(target_device).float()
            valid = valid.to(target_device).float()
            logits = model(x * 2 - 1)
            scores = per_image_metrics(logits, y, valid)
            for position, index in enumerate(indices.tolist()):
                row = {"index": index, "id": manifest[index]["id"]}
                row.update({key: float(value[position]) for key, value in scores.items()})
                rows.append(row)
    metrics = {key: sum(row[key] for row in rows) / len(rows)
               for key in ["dice", "iou", "recall", "precision", "pixel_accuracy"]}
    metrics.update({"test_samples": len(rows), "best_epoch": state.get("epoch"), "threshold": 0.5})
    result_dir = project_root / "results" / f"evaluation_{experiment}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (result_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
