"""Fine-tune YOLOv8n на нашем объединённом датасете.

Стартует из pretrained yolov8n.pt (модель из COCO).
Дообучает на dataset_merged (phone + book).
Сохраняет результаты в runs/finetune/<name>.
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "dataset_merged" / "data.yaml"
PRETRAINED = ROOT / "models" / "yolov8n.pt"
RUNS_DIR = ROOT / "runs" / "finetune"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pass1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--start-weights", default=str(PRETRAINED),
                   help="Стартовые веса (по умолчанию pretrained yolov8n.pt)")
    args = p.parse_args()

    if not DATA_YAML.exists():
        print(f"Не найден {DATA_YAML}", file=sys.stderr)
        return 1
    start_weights = Path(args.start_weights)
    if not start_weights.exists():
        print(f"Не найден {start_weights}", file=sys.stderr)
        return 1

    print(f"Стартовые веса: {start_weights}")
    model = YOLO(str(start_weights))
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.name,
        patience=args.patience,
        verbose=True,
        exist_ok=False,
    )
    print(f"\nГотово. Результаты: {RUNS_DIR / args.name}")
    print(f"Лучшие веса: {RUNS_DIR / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
