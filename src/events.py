"""Конвертация покадровых масок в интервалы событий и сводные метрики."""
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

from detectors import process_video
from yolo_detector import run_yolo_per_frame


@dataclass
class Event:
    event_type: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def mask_to_intervals(mask: list[bool], fps: float) -> list[tuple[float, float]]:
    """Группирует подряд идущие True-кадры в интервалы (секунды)."""
    intervals: list[tuple[float, float]] = []
    in_run = False
    start_idx = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start_idx = i
        elif not v and in_run:
            in_run = False
            intervals.append((start_idx / fps, i / fps))
    if in_run:
        intervals.append((start_idx / fps, len(mask) / fps))
    return intervals


def merge_close_intervals(
    intervals: list[tuple[float, float]], max_gap_sec: float
) -> list[tuple[float, float]]:
    """Сливает соседние интервалы, если разрыв между ними <= max_gap_sec."""
    if not intervals:
        return []
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s - cur_e <= max_gap_sec:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def filter_short_intervals(
    intervals: list[tuple[float, float]], min_duration_sec: float
) -> list[tuple[float, float]]:
    """Отбрасывает интервалы длительностью меньше порога."""
    return [(s, e) for s, e in intervals if e - s >= min_duration_sec]


def run_full_analysis(
    clip_path: Path,
    yolo_weights: Path,
    yaw_threshold: float = 10.0,
    pitch_threshold: float = 15.0,
    gaze_min_frames: int = 10,
    merge_gap_sec: float = 0.3,
    min_duration_sec: float = 0.2,
) -> dict:
    """Прогоняет все детекторы на клипе. Возвращает словарь с результатами.

    Ключи:
        fps, duration_sec, frames_total -- метаданные клипа;
        masks -- {event: list[bool]} покадровые предсказания;
        events -- список Event со start_sec/end_sec;
        totals -- {event: {n_events, duration_sec, share_pct}}.
    """
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_sec = frames_total / fps if fps > 0 else 0.0

    face_results = process_video(
        clip_path,
        yaw_threshold=yaw_threshold,
        pitch_threshold=pitch_threshold,
        gaze_min_frames=gaze_min_frames,
    )
    masks = {
        "gaze_away": [r.gaze_flag for r in face_results],
        "no_face": [r.no_face for r in face_results],
        "multi_face": [r.multi_face for r in face_results],
    }
    yolo_masks = run_yolo_per_frame(clip_path, weights=yolo_weights)
    masks["phone"] = yolo_masks["phone"]
    masks["book"] = yolo_masks["book"]

    events: list[Event] = []
    totals: dict[str, dict] = {}
    for event_type, mask in masks.items():
        intervals = mask_to_intervals(mask, fps)
        intervals = merge_close_intervals(intervals, merge_gap_sec)
        intervals = filter_short_intervals(intervals, min_duration_sec)
        n = len(intervals)
        dur = sum(e - s for s, e in intervals)
        totals[event_type] = {
            "n_events": n,
            "duration_sec": dur,
            "share_pct": (dur / duration_sec * 100) if duration_sec else 0.0,
        }
        for s, e in intervals:
            events.append(Event(event_type=event_type, start_sec=s, end_sec=e))
    events.sort(key=lambda ev: (ev.start_sec, ev.event_type))

    return {
        "fps": fps,
        "duration_sec": duration_sec,
        "frames_total": frames_total,
        "masks": masks,
        "events": events,
        "totals": totals,
    }
