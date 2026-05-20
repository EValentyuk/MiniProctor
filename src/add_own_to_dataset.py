"""Добавляет авторазмеченные кадры own_labeled в dataset_merged.

Делит примерно 80/20 на train/valid (test не трогаем -- он Roboflow-only).
Префиксует имена 'own_' чтобы отличать от Roboflow-кадров.

Идемпотентно: удаляет старые own_-файлы из train/valid перед добавлением.
"""
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWN = ROOT / "data" / "own_labeled"
MERGED = ROOT / "data" / "dataset_merged"

VAL_RATIO = 0.2
SEED = 42


def main() -> int:
    if not (OWN / "images").exists():
        print(f"Не найден {OWN}", file=sys.stderr)
        return 1

    img_files = sorted((OWN / "images").glob("*.jpg"))
    if not img_files:
        print("Нет картинок в own_labeled.", file=sys.stderr)
        return 1
    print(f"Найдено own-кадров: {len(img_files)}")

    # Чистим старые own_-файлы.
    removed = 0
    for split in ("train", "valid"):
        for sub in ("images", "labels"):
            d = MERGED / split / sub
            for f in d.glob("own_*"):
                f.unlink()
                removed += 1
    print(f"Удалено старых own_-файлов: {removed}")

    # Раскидываем.
    rng = random.Random(SEED)
    rng.shuffle(img_files)
    n_val = max(1, int(len(img_files) * VAL_RATIO))
    val_set = set(f.name for f in img_files[:n_val])

    n_train = n_val_added = 0
    for img in img_files:
        lbl = OWN / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        split = "valid" if img.name in val_set else "train"
        out_img = MERGED / split / "images" / f"own_{img.name}"
        out_lbl = MERGED / split / "labels" / f"own_{img.stem}.txt"
        shutil.copy2(img, out_img)
        shutil.copy2(lbl, out_lbl)
        if split == "train":
            n_train += 1
        else:
            n_val_added += 1

    print(f"Train: +{n_train}, Valid: +{n_val_added}")

    # Подсчёт итогов.
    train_total = len(list((MERGED / "train" / "images").glob("*")))
    valid_total = len(list((MERGED / "valid" / "images").glob("*")))
    print(f"\nИтого в dataset_merged:")
    print(f"  train: {train_total}")
    print(f"  valid: {valid_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
