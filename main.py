from pathlib import Path
import argparse
from src.data import prepare_data
from src.train import train
from src.evaluate import evaluate
from src.predict import predict_file

PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_processed(force=False):
    processed = PROJECT_ROOT / "data" / "processed" / "images.npy"
    if force or not processed.exists():
        print("Preparing data...")
        prepare_data(PROJECT_ROOT, force=force)
        print("Data preparation complete.")


def main():
    parser = argparse.ArgumentParser(description="Thyroid ultrasound U-Net project")
    parser.add_argument("mode", choices=["prepare", "train", "evaluate", "predict"])
    parser.add_argument("--experiment", choices=["A", "B", "C"], default="C")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="Use a small subset for a quick smoke run")
    parser.add_argument("--force", action="store_true", help="Rebuild processed data")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image", type=str, help="Input image for predict mode")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_data(PROJECT_ROOT, force=args.force)
        print("Processed data: data/processed")
    elif args.mode == "train":
        ensure_processed()
        train(PROJECT_ROOT, args.experiment, args.epochs, args.batch_size, args.device, args.limit)
        print("Model output: models/ and results/")
    elif args.mode == "evaluate":
        ensure_processed()
        evaluate(PROJECT_ROOT, args.experiment, args.device)
        print("Evaluation output: results/")
    elif args.mode == "predict":
        if not args.image:
            parser.error("predict mode requires --image")
        predict_file(PROJECT_ROOT, args.image, device=args.device)
        print("Prediction output: results/predictions/")


if __name__ == "__main__":
    main()
