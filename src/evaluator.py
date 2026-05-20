"""Оценщик метрик precision/recall/F1 по покадровым предсказаниям.

Прогоняет на каждом клипе face-детекторы (gaze_away, no_face, multi_face)
и YOLO (phone, book), сравнивает с ground truth из CSV, считает метрики
по каждому типу события.

Результат -- таблица в консоли и CSV в data/metrics/baseline_metrics.csv.
"""
import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from detectors import process_video
from yolo_detector import DEFAULT_WEIGHTS, run_yolo_per_frame

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
GT_CSV = ROOT / "data" / "labels" / "ground_truth.csv"
DEFAULT_OUT_CSV = ROOT / "data" / "metrics" / "baseline_metrics.csv"

# Какие типы событий какому детектору соответствуют.
FACE_EVENTS = {"gaze_away", "no_face", "multi_face"}
YOLO_EVENTS = {"phone", "book"}
ALL_EVENTS = FACE_EVENTS | YOLO_EVENTS

# Какие клипы участвуют в оценке. clean_baseline -- негативный контроль
# (только FP, без TP/FN).
CLIPS = [
    "clean_baseline.mp4",
    "phone_in_hand.mp4",
    "book_in_view.mp4",
    "gaze_side.mp4",
    "gaze_down.mp4",
    "absent.mp4",
    "multi_face.mp4",
]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    def f1(self) -> float | None:
        p = self.precision()
        r = self.recall()
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)


def load_ground_truth(csv_path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Парсит CSV в {clip: {event: [(start, end), ...]}}."""
    gt: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[row["clip"]][row["event_type"]].append(
                (float(row["start_sec"]), float(row["end_sec"]))
            )
    return gt


def make_gt_mask(
    n_frames: int, fps: float, intervals: list[tuple[float, float]]
) -> list[bool]:
    """Преобразует интервалы (секунды) в покадровую булеву маску."""
    mask = [False] * n_frames
    for start_sec, end_sec in intervals:
        start_f = max(0, round(start_sec * fps))
        end_f = min(n_frames - 1, round(end_sec * fps))
        for i in range(start_f, end_f + 1):
            mask[i] = True
    return mask


def frame_counts(gt_mask: list[bool], pred_mask: list[bool]) -> Counts:
    """Покадровый подсчёт TP/FP/FN."""
    n = min(len(gt_mask), len(pred_mask))
    tp = sum(1 for i in range(n) if gt_mask[i] and pred_mask[i])
    fp = sum(1 for i in range(n) if not gt_mask[i] and pred_mask[i])
    fn = sum(1 for i in range(n) if gt_mask[i] and not pred_mask[i])
    return Counts(tp=tp, fp=fp, fn=fn)


def evaluate_clip(
    clip_path: Path,
    gt_for_clip: dict[str, list[tuple[float, float]]],
    yolo_weights: Path = DEFAULT_WEIGHTS,
) -> dict[str, Counts]:
    """Возвращает counts по каждому типу события для одного клипа."""
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # Face-детекторы (один проход по видео).
    face_results = process_video(clip_path)
    n_frames = len(face_results)

    pred_masks: dict[str, list[bool]] = {
        "gaze_away": [r.gaze_flag for r in face_results],
        "no_face": [r.no_face for r in face_results],
        "multi_face": [r.multi_face for r in face_results],
    }

    # YOLO (отдельный проход).
    yolo_per_frame = run_yolo_per_frame(clip_path, weights=yolo_weights)
    pred_masks["phone"] = yolo_per_frame["phone"]
    pred_masks["book"] = yolo_per_frame["book"]

    counts_by_event: dict[str, Counts] = {}
    for event in ALL_EVENTS:
        gt_intervals = gt_for_clip.get(event, [])
        gt_mask = make_gt_mask(n_frames, fps, gt_intervals)
        counts_by_event[event] = frame_counts(gt_mask, pred_masks[event])

    return counts_by_event


def fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "  -  "


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--yolo-weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--out", default=str(DEFAULT_OUT_CSV))
    args = p.parse_args()
    yolo_weights = Path(args.yolo_weights)
    out_csv = Path(args.out)

    if not GT_CSV.exists():
        print(f"Не найден ground truth: {GT_CSV}", file=sys.stderr)
        return 1

    print(f"YOLO weights: {yolo_weights}")
    gt_by_clip = load_ground_truth(GT_CSV)

    # per-clip результаты: {event: {clip: Counts}}.
    per_clip: dict[str, dict[str, Counts]] = {e: {} for e in ALL_EVENTS}

    for clip_name in CLIPS:
        clip_path = RAW_DIR / clip_name
        if not clip_path.exists():
            print(f"Пропуск (не найден): {clip_path}", file=sys.stderr)
            continue
        print(f"\n=== {clip_name} ===")
        gt_for_clip = gt_by_clip.get(clip_name, {})
        counts_by_event = evaluate_clip(clip_path, gt_for_clip, yolo_weights=yolo_weights)
        for event, c in counts_by_event.items():
            per_clip[event][clip_name] = c
            print(f"  {event:<11} TP={c.tp:4d} FP={c.fp:4d} FN={c.fn:4d}  "
                  f"P={fmt(c.precision())} R={fmt(c.recall())} F1={fmt(c.f1())}")

    # Агрегированные метрики по каждому событию.
    print("\n\n=== Сводные метрики (все клипы) ===")
    print(f"{'event':<12}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}")
    aggregate: dict[str, Counts] = {}
    for event in sorted(ALL_EVENTS):
        agg = Counts()
        for clip_name, c in per_clip[event].items():
            agg.add(c)
        aggregate[event] = agg
        print(f"{event:<12}{agg.tp:>6}{agg.fp:>6}{agg.fn:>6}"
              f"{fmt(agg.precision()):>8}{fmt(agg.recall()):>8}{fmt(agg.f1()):>8}")

    # Сохранение CSV.
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "clip", "tp", "fp", "fn", "precision", "recall", "f1"])
        for event in sorted(ALL_EVENTS):
            for clip_name, c in per_clip[event].items():
                w.writerow([event, clip_name, c.tp, c.fp, c.fn,
                            fmt(c.precision()), fmt(c.recall()), fmt(c.f1())])
            agg = aggregate[event]
            w.writerow([event, "ALL", agg.tp, agg.fp, agg.fn,
                        fmt(agg.precision()), fmt(agg.recall()), fmt(agg.f1())])
    print(f"\nМетрики сохранены: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
