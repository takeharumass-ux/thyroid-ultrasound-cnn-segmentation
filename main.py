"""甲状腺超声 CNN 项目的统一命令行入口。

用户只需要运行本文件，并通过第一个位置参数选择功能：
``prepare`` 准备数据，``train`` 训练模型，``evaluate`` 评价模型，
``predict`` 对单张图像进行预测。具体实现分别位于 ``src`` 目录中。
"""

from pathlib import Path
import argparse
from src.data import prepare_data
from src.train import train
from src.evaluate import evaluate
from src.predict import predict_file

PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_processed(force=False):
    """确保训练、评价所需的预处理缓存已经生成。

    如果 ``data/processed/images.npy`` 已存在，就直接复用缓存；否则调用
    ``src.data.prepare_data`` 从原始 TN3K 数据重新生成。这样训练和评价时
    不需要用户手动判断是否已经准备过数据。
    """

    processed = PROJECT_ROOT / "data" / "processed" / "images.npy"
    if force or not processed.exists():
        print("Preparing data...")
        prepare_data(PROJECT_ROOT, force=force)
        print("Data preparation complete.")


def main():
    # 用 argparse 统一管理命令行参数，避免为每个功能单独编写启动脚本。
    parser = argparse.ArgumentParser(description="Thyroid ultrasound U-Net project")
    # 四种模式对应项目的四个主要阶段。
    parser.add_argument("mode", choices=["prepare", "train", "evaluate", "predict"])
    # A/B/C 是三组训练设置；默认使用 C 组（增强 + Dice 损失）。
    parser.add_argument("--experiment", choices=["A", "B", "C"], default="C")
    # 下面三个参数主要用于训练，也可以通过 bat 文件传入。
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Use a small subset for a quick smoke run")
    # --force 用于强制重建预处理缓存；--device 可手动指定 CPU 或 CUDA。
    parser.add_argument("--force", action="store_true", help="Rebuild processed data")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # predict 模式必须额外提供一张输入图像的路径。
    parser.add_argument("--image", type=str, help="Input image for predict mode")
    args = parser.parse_args()

    if args.mode == "prepare":
        # 只执行数据准备，不训练模型。
        prepare_data(PROJECT_ROOT, force=args.force)
        print("Processed data: data/processed")
    elif args.mode == "train":
        # 训练前自动检查并准备缓存数据，然后调用训练模块。
        ensure_processed()
        train(PROJECT_ROOT, args.experiment, args.epochs, args.batch_size, args.device, args.limit)
        print("Model output: models/ and results/")
    elif args.mode == "evaluate":
        # 使用已有权重在测试集上计算 Dice、IoU 等指标。
        ensure_processed()
        evaluate(PROJECT_ROOT, args.experiment, args.device)
        print("Evaluation output: results/")
    elif args.mode == "predict":
        # 单张图像预测不需要遍历整个测试集，但必须提供 --image。
        if not args.image:
            parser.error("predict mode requires --image")
        predict_file(PROJECT_ROOT, args.image, device=args.device)
        print("Prediction output: results/predictions/")


if __name__ == "__main__":
    main()
