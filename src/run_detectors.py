"""Запуск всех детекторов Дня 2 на тестовом ролике."""
import sys
from pathlib import Path

from detectors import process_video, print_summary

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = ROOT / "data" / "raw" / "smoke.mp4"
    output_path = ROOT / "data" / "processed" / (input_path.stem + "_detectors.mp4")

    results = process_video(input_path, output_path)
    if not results:
        return 1
    print_summary(results)
    print(f"\nАннотированное видео: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
