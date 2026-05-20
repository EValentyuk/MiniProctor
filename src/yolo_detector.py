"""Покадровый прогон YOLOv8 на ролике.

Возвращает для каждого кадра, был ли обнаружен телефон/книга/ноутбук
с уверенностью выше порога.
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "yolov8n.pt"

# Каждое событие может матчить несколько имён классов -- pretrained COCO
# использует "cell phone", наш fine-tuned -- "phone".
EVENT_TO_CLASS_SYNONYMS = {
    "phone": ["phone", "cell phone", "mobile phone", "cellphone"],
    "book": ["book", "books"],
    "laptop": ["laptop"],
}


def run_yolo_per_frame(
    clip_path: str | Path,
    weights: str | Path = DEFAULT_WEIGHTS,
    conf_threshold: float = 0.25,
) -> dict[str, list[bool]]:
    """Прогоняет видео через YOLO. Возвращает покадровые булевы маски по классам.

    Ключи -- логические имена событий: phone, book, laptop.
    """
    model = YOLO(str(weights))
    names = model.names  # {id: name}
    cls_id_by_event: dict[str, int | None] = {}
    for event, synonyms in EVENT_TO_CLASS_SYNONYMS.items():
        match_id = next(
            (i for i, n in names.items() if n.lower() in [s.lower() for s in synonyms]),
            None,
        )
        cls_id_by_event[event] = match_id

    cap = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    per_frame: dict[str, list[bool]] = {
        event: [False] * total for event in EVENT_TO_CLASS_SYNONYMS
    }

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, verbose=False, conf=conf_threshold)
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                for event, eid in cls_id_by_event.items():
                    if eid is not None and cls_id == eid:
                        per_frame[event][frame_idx] = True
                        break
        frame_idx += 1

    cap.release()
    return per_frame
