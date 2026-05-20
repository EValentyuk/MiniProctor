"""Sweep по pitch/yaw threshold для gaze_away.

Для каждой кандидатной пары порогов применяет тот же сглаживатель
(GAZE_MIN_FRAMES последовательных кадров за порогом), считает
precision/recall/F1 по покадровому сравнению с ground truth.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from detectors import GAZE_MIN_FRAMES, process_video
from evaluator import (
    Counts,
    frame_counts,
    load_ground_truth,
    make_gt_mask,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
GT_CSV = ROOT / "data" / "labels" / "ground_truth.csv"

GAZE_CLIPS = ["gaze_side.mp4", "gaze_down.mp4", "clean_baseline.mp4"]

# Кандидаты на пороги (после исправления конвенции углов).
YAW_CANDIDATES = [10.0, 15.0, 20.0, 25.0, 30.0]
PITCH_CANDIDATES = [10.0, 15.0, 18.0, 20.0, 25.0, 30.0]


def apply_threshold_with_smoothing(
    yaws: list[float],
    pitches: list[float],
    yaw_t: float,
    pitch_t: float,
    min_frames: int,
) -> list[bool]:
    """Применяет порог и сглаживание -- возвращает gaze_flag по кадрам."""
    flags = [False] * len(yaws)
    counter = 0
    for i, (y, p) in enumerate(zip(yaws, pitches)):
        away = abs(y) > yaw_t or abs(p) > pitch_t
        counter = counter + 1 if away else 0
        flags[i] = counter >= min_frames
    return flags


def main() -> int:
    gt_by_clip = load_ground_truth(GT_CSV)

    # Собираем raw углы по каждому клипу одним прогоном детекторов.
    clip_data: dict[str, dict] = {}
    for clip_name in GAZE_CLIPS:
        clip_path = RAW_DIR / clip_name
        if not clip_path.exists():
            print(f"Пропуск: {clip_path}", file=sys.stderr)
            continue
        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        print(f"Обрабатываю {clip_name}...")
        results = process_video(clip_path)
        yaws = [r.yaw for r in results]
        pitches = [r.pitch for r in results]
        gt_intervals = gt_by_clip.get(clip_name, {}).get("gaze_away", [])
        gt_mask = make_gt_mask(len(results), fps, gt_intervals)
        clip_data[clip_name] = {
            "yaws": yaws,
            "pitches": pitches,
            "gt": gt_mask,
            "fps": fps,
        }
        print(f"  pitch range: [{min(pitches):+.1f}, {max(pitches):+.1f}], "
              f"yaw range: [{min(yaws):+.1f}, {max(yaws):+.1f}]")

    # Sweep по двум порогам.
    print(f"\n{'yaw_t':>7}{'pitch_t':>9}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    best = (None, None, -1.0)
    for yaw_t in YAW_CANDIDATES:
        for pitch_t in PITCH_CANDIDATES:
            agg = Counts()
            for clip_name, data in clip_data.items():
                pred = apply_threshold_with_smoothing(
                    data["yaws"], data["pitches"],
                    yaw_t, pitch_t, GAZE_MIN_FRAMES,
                )
                agg.add(frame_counts(data["gt"], pred))
            p = agg.precision()
            r = agg.recall()
            f = agg.f1()
            f_str = f"{f:.3f}" if f is not None else "  -  "
            p_str = f"{p:.3f}" if p is not None else "  -  "
            r_str = f"{r:.3f}" if r is not None else "  -  "
            print(f"{yaw_t:>7.1f}{pitch_t:>9.1f}{agg.tp:>6}{agg.fp:>6}{agg.fn:>6}"
                  f"{p_str:>8}{r_str:>8}{f_str:>8}")
            if f is not None and f > best[2]:
                best = (yaw_t, pitch_t, f)

    if best[0] is not None:
        print(f"\nЛучшая пара (yaw, pitch): ({best[0]}, {best[1]}), F1={best[2]:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
