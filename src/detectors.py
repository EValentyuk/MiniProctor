"""Детекторы нарушений на базе MediaPipe Face Landmarker.

Три детектора:
1. Head pose / gaze -- yaw/pitch/roll из landmarks, флаг «отвёл взгляд».
2. Multi-face -- больше одного лица в кадре.
3. No face -- лицо отсутствует.
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_landmarker.task"

# --- Конфигурация порогов ---

# Пороги углов (градусы). Подобраны sweep'ом по F1 на gaze_side/gaze_down/clean_baseline.
YAW_THRESHOLD = 10.0    # отвод взгляда вбок
PITCH_THRESHOLD = 15.0  # отвод взгляда вверх/вниз

# Сглаживание: сколько кадров подряд угол за порогом, чтобы зафиксировать нарушение.
GAZE_MIN_FRAMES = 10  # при ~15 FPS реальной вебки это ~0.7 секунды


@dataclass
class FrameResult:
    """Результат обработки одного кадра."""
    frame_idx: int
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    faces_count: int = 0
    gaze_away: bool = False   # взгляд за порогом в этом кадре
    gaze_flag: bool = False   # сработал флаг (дольше N кадров)
    multi_face: bool = False
    no_face: bool = False


@dataclass
class GazeTracker:
    """Счётчик последовательных кадров с отводом взгляда."""
    counter: int = 0
    min_frames: int = GAZE_MIN_FRAMES

    def update(self, gaze_away: bool) -> bool:
        if gaze_away:
            self.counter += 1
        else:
            self.counter = 0
        return self.counter >= self.min_frames


def rotation_matrix_to_euler(rmat: np.ndarray) -> tuple[float, float, float]:
    """Извлекает yaw, pitch, roll (в градусах) из матрицы поворота 3x3.

    Конвенция xyz по scipy. На координатах MediaPipe:
    - pitch (a[0]) -- наклон головы вверх/вниз;
    - yaw (a[1]) -- поворот головы влево/вправо;
    - roll (a[2]) -- наклон головы вбок.
    """
    pitch, yaw, roll = R.from_matrix(rmat).as_euler("xyz", degrees=True)
    return float(yaw), float(pitch), float(roll)


def process_video(
    input_path: str | Path,
    output_path: str | Path | None = None,
    yaw_threshold: float = YAW_THRESHOLD,
    pitch_threshold: float = PITCH_THRESHOLD,
    gaze_min_frames: int = GAZE_MIN_FRAMES,
    max_faces: int = 2,
) -> list[FrameResult]:
    """Прогоняет видео через все детекторы, возвращает покадровые результаты.

    Если output_path задан, сохраняет аннотированное видео.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Не найден входной файл: {input_path}", file=sys.stderr)
        return []

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Вход: {input_path} ({width}x{height}, {fps:.1f} fps, {total} кадров)")

    writer = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=max_faces,
        output_facial_transformation_matrixes=True,
    )

    results: list[FrameResult] = []
    tracker = GazeTracker(min_frames=gaze_min_frames)
    frame_idx = 0
    start = time.time()

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            detection = landmarker.detect_for_video(mp_image, ts_ms)

            fr = FrameResult(frame_idx=frame_idx)
            faces = detection.face_landmarks
            fr.faces_count = len(faces) if faces else 0

            # --- No face ---
            if fr.faces_count == 0:
                fr.no_face = True
                tracker.update(False)

            # --- Multi-face ---
            elif fr.faces_count > 1:
                fr.multi_face = True
                tracker.update(False)

            # --- Head pose (single face) ---
            if fr.faces_count >= 1 and detection.facial_transformation_matrixes:
                tmat = detection.facial_transformation_matrixes[0]
                rmat = np.array(tmat)[:3, :3]
                yaw, pitch, roll = rotation_matrix_to_euler(rmat)
                fr.yaw = yaw
                fr.pitch = pitch
                fr.roll = roll
                fr.gaze_away = abs(yaw) > yaw_threshold or abs(pitch) > pitch_threshold
                fr.gaze_flag = tracker.update(fr.gaze_away)

            results.append(fr)

            # --- Аннотация кадра ---
            if writer:
                draw_overlay(frame, fr)
                writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    elapsed = time.time() - start
    proc_fps = frame_idx / max(elapsed, 1e-6)
    print(f"Обработано: {frame_idx} кадров за {elapsed:.2f} с ({proc_fps:.1f} FPS)")
    return results


def draw_overlay(frame: np.ndarray, fr: FrameResult) -> None:
    """Рисует HUD поверх кадра."""
    h, w = frame.shape[:2]
    color_ok = (0, 200, 0)
    color_warn = (0, 0, 255)

    # Углы головы.
    yaw_color = color_warn if abs(fr.yaw) > YAW_THRESHOLD else color_ok
    pitch_color = color_warn if abs(fr.pitch) > PITCH_THRESHOLD else color_ok
    cv2.putText(frame, f"Yaw: {fr.yaw:+.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 2)
    cv2.putText(frame, f"Pitch: {fr.pitch:+.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, pitch_color, 2)
    cv2.putText(frame, f"Roll: {fr.roll:+.1f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ok, 2)

    # Флаги.
    y = 110
    if fr.gaze_flag:
        cv2.putText(frame, "!! GAZE AWAY !!", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_warn, 2)
        y += 30
    if fr.multi_face:
        cv2.putText(frame, "!! MULTI FACE !!", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_warn, 2)
        y += 30
    if fr.no_face:
        cv2.putText(frame, "!! NO FACE !!", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_warn, 2)

    # Количество лиц.
    faces_color = color_warn if fr.faces_count != 1 else color_ok
    cv2.putText(frame, f"Faces: {fr.faces_count}", (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, faces_color, 2)


def print_summary(results: list[FrameResult]) -> None:
    """Печатает сводку по прогону."""
    total = len(results)
    if total == 0:
        print("Нет данных.")
        return

    no_face = sum(1 for r in results if r.no_face)
    multi = sum(1 for r in results if r.multi_face)
    gaze_away = sum(1 for r in results if r.gaze_away)
    gaze_flag = sum(1 for r in results if r.gaze_flag)

    yaws = [r.yaw for r in results if r.faces_count >= 1]
    pitches = [r.pitch for r in results if r.faces_count >= 1]

    print(f"\n--- Сводка ({total} кадров) ---")
    print(f"Нет лица:           {no_face} ({no_face / total:.1%})")
    print(f"Несколько лиц:      {multi} ({multi / total:.1%})")
    print(f"Взгляд за порогом:  {gaze_away} ({gaze_away / total:.1%})")
    print(f"Флаг GAZE AWAY:     {gaze_flag} ({gaze_flag / total:.1%})")
    if yaws:
        print(f"Yaw  диапазон:      [{min(yaws):+.1f}, {max(yaws):+.1f}], "
              f"mean {np.mean(yaws):+.1f}, std {np.std(yaws):.1f}")
        print(f"Pitch диапазон:     [{min(pitches):+.1f}, {max(pitches):+.1f}], "
              f"mean {np.mean(pitches):+.1f}, std {np.std(pitches):.1f}")
