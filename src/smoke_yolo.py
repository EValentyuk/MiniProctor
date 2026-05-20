"""Smoke-тест YOLOv8n на записанном ролике.

Прогоняет видео через pretrained YOLOv8n, ищет классы интереса
(cell phone, book, laptop), рисует боксы и сохраняет аннотированный файл.
"""
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "smoke.mp4"
OUTPUT = ROOT / "data" / "processed" / "smoke_yolo.mp4"
MODEL_PATH = ROOT / "models" / "yolov8n.pt"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

INTEREST = {"cell phone", "book", "laptop"}


def main() -> int:
    if not INPUT.exists():
        print(f"Не найден входной файл: {INPUT}", file=sys.stderr)
        return 1

    model = YOLO(str(MODEL_PATH) if MODEL_PATH.exists() else "yolov8n.pt")
    if not MODEL_PATH.exists():
        downloaded = Path("yolov8n.pt")
        if downloaded.exists():
            downloaded.replace(MODEL_PATH)
            model = YOLO(str(MODEL_PATH))

    names = model.names
    interest_ids = {i for i, n in names.items() if n in INTEREST}
    print(f"Классы интереса: {sorted(INTEREST)} -> ids {sorted(interest_ids)}")

    cap = cv2.VideoCapture(str(INPUT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Вход: {INPUT} ({width}x{height}, {fps:.1f} fps, {total} кадров)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps, (width, height))

    frames = 0
    detections = Counter()
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        results = model.predict(frame, verbose=False, conf=0.25)
        r = results[0]
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                if cls_id not in interest_ids:
                    continue
                detections[cls_name] += 1
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 0),
                    2,
                )

        writer.write(frame)

    cap.release()
    writer.release()

    elapsed = time.time() - start
    proc_fps = frames / max(elapsed, 1e-6)
    print(f"Обработано кадров: {frames}")
    print(f"Время: {elapsed:.2f} с, FPS обработки: {proc_fps:.1f}")
    print(f"Детекции по классам интереса: {dict(detections) or 'нет'}")
    print(f"Результат: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
