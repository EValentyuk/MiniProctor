"""Композирует синтетический multi-face клип.

Берёт чистый ролик (clean_baseline.mp4) и накладывает второе лицо
из stock-картинки на часть кадров. Результат -- ролик, на котором
multi-face детектор должен сработать в заданном временном окне.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT_VIDEO = ROOT / "data" / "raw" / "clean_baseline.mp4"
INPUT_FACE = ROOT / "data" / "raw" / "synthetic_face.jpg"
OUTPUT = ROOT / "data" / "raw" / "multi_face.mp4"

# Когда вставлять второе лицо (секунды).
INSERT_START_SEC = 8.0
INSERT_END_SEC = 18.0

# Размер вставки и положение в кадре.
INSERT_WIDTH = 180  # px
POSITION = "top_right"  # top_left | top_right | bottom_left | bottom_right
PADDING = 20


def main() -> int:
    if not INPUT_VIDEO.exists():
        print(f"Не найден: {INPUT_VIDEO}", file=sys.stderr)
        return 1
    if not INPUT_FACE.exists():
        print(f"Не найден: {INPUT_FACE}", file=sys.stderr)
        return 1

    face = cv2.imread(str(INPUT_FACE))
    if face is None:
        print(f"Не удалось прочитать {INPUT_FACE}", file=sys.stderr)
        return 1

    # Кадрируем лицо до квадрата по короткой стороне.
    fh, fw = face.shape[:2]
    side = min(fh, fw)
    top = (fh - side) // 2
    left = (fw - side) // 2
    face = face[top:top + side, left:left + side]
    face = cv2.resize(face, (INSERT_WIDTH, INSERT_WIDTH))

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fh = face.shape[0]
    fw = face.shape[1]
    if POSITION == "top_left":
        x, y = PADDING, PADDING
    elif POSITION == "top_right":
        x, y = width - fw - PADDING, PADDING
    elif POSITION == "bottom_left":
        x, y = PADDING, height - fh - PADDING
    else:
        x, y = width - fw - PADDING, height - fh - PADDING

    insert_start_frame = int(INSERT_START_SEC * fps)
    insert_end_frame = int(INSERT_END_SEC * fps)
    print(f"Вход: {INPUT_VIDEO} ({width}x{height}, {fps:.1f} fps, {total} кадров)")
    print(f"Вставка лица в кадры [{insert_start_frame}, {insert_end_frame}] "
          f"({INSERT_START_SEC}–{INSERT_END_SEC} с)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if insert_start_frame <= frame_idx <= insert_end_frame:
            frame[y:y + fh, x:x + fw] = face

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Готово: {OUTPUT} ({frame_idx} кадров)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
