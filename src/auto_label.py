"""Авторазметка кадров с YOLO-World (open-vocabulary детектор).

Принимает клип, текстовые промпты («cell phone», «book») и список ground-truth
интервалов из ground_truth.csv. Для каждого кадра внутри GT-интервала запрашивает
YOLO-World, сохраняет картинку и YOLO-разметку.

Внешний цикл -- по клипам phone_in_hand.mp4 и book_in_view.mp4.
Класс в выходной разметке: 0=phone, 1=book (общая схема dataset_merged).

Результат:
  data/own_labeled/images/<clip>_<frame>.jpg
  data/own_labeled/labels/<clip>_<frame>.txt
  data/own_labeled/preview/<clip>_<frame>.jpg  (с нарисованными bbox для спот-чека)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
GT_CSV = ROOT / "data" / "labels" / "ground_truth.csv"
OUT_DIR = ROOT / "data" / "own_labeled"
PREVIEW_DIR = OUT_DIR / "preview"
IMG_DIR = OUT_DIR / "images"
LBL_DIR = OUT_DIR / "labels"

EVENT_TO_CLASS_ID = {"phone": 0, "book": 1}
EVENT_TO_PROMPT = {"phone": "cell phone", "book": "book"}

# Клипы и какой класс из них извлекаем.
CLIPS = [
    ("phone_in_hand.mp4", "phone"),
    ("book_in_view.mp4", "book"),
]

CONF_THRESHOLD = 0.05  # низкий порог -- лучше иметь шум для отбраковки


def load_gt_intervals(csv_path: Path) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Возвращает {(clip, event_type): [(start, end), ...]}."""
    gt: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt[(row["clip"], row["event_type"])].append(
                (float(row["start_sec"]), float(row["end_sec"]))
            )
    return gt


def is_in_intervals(t: float, intervals: list[tuple[float, float]]) -> bool:
    return any(s <= t <= e for s, e in intervals)


def main() -> int:
    if not GT_CSV.exists():
        print(f"Не найден {GT_CSV}", file=sys.stderr)
        return 1

    for d in (IMG_DIR, LBL_DIR, PREVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)

    gt = load_gt_intervals(GT_CSV)

    print("Загружаю YOLO-World...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    # Задаём набор классов в порядке: phone=0, book=1.
    prompts = [EVENT_TO_PROMPT["phone"], EVENT_TO_PROMPT["book"]]
    model.set_classes(prompts)
    print(f"Промпты: {prompts}")

    totals = {"phone": 0, "book": 0}

    for clip_name, event_type in CLIPS:
        clip_path = RAW_DIR / clip_name
        if not clip_path.exists():
            print(f"Пропуск (нет файла): {clip_path}", file=sys.stderr)
            continue
        intervals = gt.get((clip_name, event_type), [])
        if not intervals:
            print(f"Пропуск (нет интервалов): {clip_name}/{event_type}")
            continue

        target_cls_id = EVENT_TO_CLASS_ID[event_type]
        cls_id_in_model = prompts.index(EVENT_TO_PROMPT[event_type])

        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h_total = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_total = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"\n=== {clip_name} ({event_type}) ===")
        print(f"  {w_total}x{h_total}, {fps:.1f} fps, интервалы: {intervals}")

        frame_idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            frame_idx += 1
            if not is_in_intervals(t, intervals):
                continue

            results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            # Фильтр: только нужный класс.
            lines = []
            preview = frame.copy()
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != cls_id_in_model:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Конвертация в YOLO format (xc, yc, w, h normalized).
                xc = ((x1 + x2) / 2) / w_total
                yc = ((y1 + y2) / 2) / h_total
                w = (x2 - x1) / w_total
                h = (y2 - y1) / h_total
                lines.append(f"{target_cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                # Preview.
                cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 200, 0), 2)
                cv2.putText(preview, f"{event_type} {conf:.2f}",
                            (int(x1), max(20, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            if not lines:
                continue

            stem = f"{clip_name.replace('.mp4', '')}_{frame_idx:04d}"
            img_out = IMG_DIR / f"{stem}.jpg"
            lbl_out = LBL_DIR / f"{stem}.txt"
            prv_out = PREVIEW_DIR / f"{stem}.jpg"
            cv2.imwrite(str(img_out), frame)
            lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            cv2.imwrite(str(prv_out), preview)
            kept += 1
            totals[event_type] += 1

        cap.release()
        print(f"  Кадров сохранено: {kept}")

    print(f"\nВсего: phone={totals['phone']}, book={totals['book']}")
    print(f"Картинки: {IMG_DIR}")
    print(f"Лейблы: {LBL_DIR}")
    print(f"Превью для проверки: {PREVIEW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
