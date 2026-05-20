"""Скачивает Roboflow датасеты для fine-tune YOLO.

Берёт два датасета:
1. book-34iuu -- Book + Mobile phone, ~159 изображений.
2. mobile-phone-dataset-bb7zb -- 3000+ изображений, только телефоны.

Каждый качается в свою папку под data/datasets/, в формате yolov8.
"""
import os
import sys
from pathlib import Path

from roboflow import Roboflow

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "data" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def load_env() -> str:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "ROBOFLOW_API_KEY":
                    return v.strip()
    return os.environ.get("ROBOFLOW_API_KEY", "")


def main() -> int:
    api_key = load_env()
    if not api_key:
        print("Не найден ROBOFLOW_API_KEY в .env или окружении.", file=sys.stderr)
        return 1

    rf = Roboflow(api_key=api_key)

    targets = [
        # Кандидаты на phone-датасет, пробуем по очереди.
        ("kylewd", "mobile-phones-wjvco", "mobile_phones"),
        ("project-1sru8", "object-detection-cell-phone", "object_detection_cell_phone"),
        ("workspace-f5gtr", "cellphone-bz5u9", "cellphone"),
        # Кандидаты на книги.
        ("priyanka-g", "book-bxoc7", "book_priyanka"),
        ("yolo-ea4dn", "book-detection-e1luo-m59fz", "book_detection"),
    ]

    successes = []
    for workspace, project, local_name in targets:
        out_dir = DATASETS_DIR / local_name
        if out_dir.exists() and (out_dir / "data.yaml").exists():
            print(f"Пропуск (уже есть): {out_dir}")
            successes.append(local_name)
            continue

        print(f"\nКачаю {workspace}/{project} -> {out_dir}")
        try:
            proj = rf.workspace(workspace).project(project)
            versions = proj.versions()
            if not versions:
                print(f"  Нет версий у {project}", file=sys.stderr)
                continue
            version = versions[-1]
            print(f"  Использую версию: {version.version}")
            dataset = version.download("yolov8", location=str(out_dir))
            print(f"  Скачано в: {dataset.location}")
            successes.append(local_name)
        except Exception as e:
            print(f"  Ошибка: {e}", file=sys.stderr)

    print(f"\nГотово. Успешных загрузок: {len(successes)} -- {successes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
