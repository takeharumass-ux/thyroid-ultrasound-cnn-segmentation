"""绘制 A/B/C 实验的训练过程曲线。

分割任务中，训练阶段使用训练集和验证集监控模型，测试集只在训练完成后
进行一次最终评价。因此这里绘制训练/验证 loss，以及验证集 Dice、IoU，
避免把测试集反复用于模型选择。
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt


EXPERIMENT_NAMES = {
    "A": ("A_baseline_history.csv", "A: BCE baseline"),
    "B": ("B_augmentation_history.csv", "B: + augmentation"),
    "C": ("C_augmentation_dice_history.csv", "C: + Dice loss"),
}


def _read_history(path):
    """读取逐轮 CSV，并把需要绘图的字段转换成数值。"""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: [float(row[key]) for row in rows]
        for key in ["epoch", "train_loss", "val_loss", "val_dice", "val_iou"]
    }


def plot_training_curves(project_root):
    """生成一张包含三部分的训练曲线图并返回输出路径。"""

    project_root = Path(project_root)
    result_dir = project_root / "results" / "reference_run"
    history_dir = result_dir / "history"
    figure_dir = result_dir / "figures"
    histories = {
        key: _read_history(history_dir / filename)
        for key, (filename, _) in EXPERIMENT_NAMES.items()
    }

    c_history = histories["C"]
    best_index = max(range(len(c_history["val_dice"])), key=c_history["val_dice"].__getitem__)
    best_epoch = int(c_history["epoch"][best_index])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    figure.suptitle("Training Curves for Thyroid Ultrasound Segmentation", fontsize=15, fontweight="bold")

    # 1. C 组训练集和验证集 loss。
    axes[0].plot(c_history["epoch"], c_history["train_loss"], label="Train loss", linewidth=2.2, color="#2563eb")
    axes[0].plot(c_history["epoch"], c_history["val_loss"], label="Validation loss", linewidth=2.2, color="#f59e0b")
    axes[0].axvline(best_epoch, linestyle="--", linewidth=1.2, color="#64748b", label=f"Best epoch: {best_epoch}")
    axes[0].set_title("Experiment C: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)

    # 2. C 组验证集 Dice 和 IoU，作为分割任务的主要“准确程度”指标。
    axes[1].plot(c_history["epoch"], c_history["val_dice"], label="Validation Dice", linewidth=2.2, color="#16a34a")
    axes[1].plot(c_history["epoch"], c_history["val_iou"], label="Validation IoU", linewidth=2.2, color="#9333ea")
    axes[1].axvline(best_epoch, linestyle="--", linewidth=1.2, color="#64748b", label=f"Best epoch: {best_epoch}")
    axes[1].set_title("Experiment C: validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    # 3. 三组实验的验证 Dice，展示数据增强和 Dice 损失的对照效果。
    colors = {"A": "#64748b", "B": "#0891b2", "C": "#2563eb"}
    for key, (_, label) in EXPERIMENT_NAMES.items():
        history = histories[key]
        axes[2].plot(history["epoch"], history["val_dice"], label=label, linewidth=2.2, color=colors[key])
    axes[2].set_title("Validation Dice comparison")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
        axis.grid(axis="x", visible=False)

    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "training_curves.png"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(f"Saved: {plot_training_curves(root)}")
