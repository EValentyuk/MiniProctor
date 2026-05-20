"""Объединяет cellphone + book_detection в единый датасет MiniProctor.

Целевая схема классов:
  0 -- phone
  1 -- book

Из cellphone (1 класс "cell phone") берём все изображения, ремап 0 -> 0.
Из book_detection (4 класса 0/Author/Book/Title) берём только изображения,
где есть bbox класса Book (индекс 2), их лейблы фильтруем -- оставляем
только Book-боксы, ремап 2 -> 1.

Результат:
  data/dataset_merged/
    train/images/, train/labels/
    valid/images/, valid/labels/
    test/images/,  test/labels/
    data.yaml
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "data" / "datasets"
PHONE_SRC = DATASETS / "cellphone"
BOOK_SRC = DATASETS / "book_detection"
OUT = ROOT / "data" / "dataset_merged"

SPLITS = ["train", "valid", "test"]

# Ремап классов из исходников в общую схему.
# cellphone: единственный класс "cell phone" -> 0 (phone).
# book_detection: классы 0=0, 1=Author, 2=Book, 3=Title -> только 2 -> 1 (book).
PHONE_REMAP = {0: 0}
BOOK_REMAP = {2: 1}


def remap_labels(src_label: Path, dst_label: Path, remap: dict[int, int]) -> int:
    """Копирует label-файл, оставляя только классы из remap и применяя его.

    Возвращает количество строк в выходном файле.
    """
    lines_out = []
    for line in src_label.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls not in remap:
            continue
        new_cls = remap[cls]
        lines_out.append(" ".join([str(new_cls)] + parts[1:]))
    if not lines_out:
        return 0
    dst_label.write_text("\n".join(lines_out) + "\n")
    return len(lines_out)


def copy_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    remap: dict[int, int],
    prefix: str,
) -> tuple[int, int]:
    """Копирует один сплит. Возвращает (n_images, n_bboxes)."""
    src_img = src_root / split / "images"
    src_lbl = src_root / split / "labels"
    dst_img = dst_root / split / "images"
    dst_lbl = dst_root / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    if not src_img.exists():
        return 0, 0

    n_img = 0
    n_bboxes = 0
    for label_file in src_lbl.glob("*.txt"):
        stem = label_file.stem
        # ищем картинку с любым расширением
        img_candidates = list(src_img.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        img_file = img_candidates[0]

        dst_lbl_path = dst_lbl / f"{prefix}_{stem}.txt"
        n = remap_labels(label_file, dst_lbl_path, remap)
        if n == 0:
            # пропускаем картинки без нужных классов
            continue

        dst_img_path = dst_img / f"{prefix}_{stem}{img_file.suffix}"
        shutil.copy2(img_file, dst_img_path)
        n_img += 1
        n_bboxes += n
    return n_img, n_bboxes


def main() -> int:
    if not PHONE_SRC.exists() or not BOOK_SRC.exists():
        print(f"Не найдены исходники: {PHONE_SRC}, {BOOK_SRC}", file=sys.stderr)
        return 1

    print(f"Сборка датасета в {OUT}")
    if OUT.exists():
        shutil.rmtree(OUT)

    totals = {}
    for split in SPLITS:
        ph_i, ph_b = copy_split(PHONE_SRC, OUT, split, PHONE_REMAP, "phone")
        bk_i, bk_b = copy_split(BOOK_SRC, OUT, split, BOOK_REMAP, "book")
        totals[split] = {
            "phone_imgs": ph_i, "phone_bboxes": ph_b,
            "book_imgs": bk_i, "book_bboxes": bk_b,
        }
        print(f"  {split}: phone {ph_i} img/{ph_b} bbox, "
              f"book {bk_i} img/{bk_b} bbox")

    # data.yaml. Пути -- абсолютные, чтобы Ultralytics не путался.
    data_yaml = OUT / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: phone\n"
        "  1: book\n",
        encoding="utf-8",
    )
    print(f"\nЗаписан {data_yaml}")
    print("Готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
