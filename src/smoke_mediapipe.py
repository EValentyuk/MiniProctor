"""Smoke-тест MediaPipe Face Landmarker (Tasks API) на записанном ролике.

Прогоняет видео через FaceLandmarker, рисует landmarks и сохраняет аннотированный файл.
Печатает FPS обработки и долю кадров с найденным лицом.
"""
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw" / "smoke.mp4"
OUTPUT = ROOT / "data" / "processed" / "smoke_facemesh.mp4"
MODEL = ROOT / "models" / "face_landmarker.task"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

mp_drawing = mp_vision.drawing_utils
mp_styles = mp_vision.drawing_styles
FACE_CONN = mp_vision.FaceLandmarksConnections


def draw_landmarks(image: np.ndarray, face_landmarks_list) -> None:
    for face_landmarks in face_landmarks_list:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=FACE_CONN.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=FACE_CONN.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
        )


def main() -> int:
    if not INPUT.exists():
        print(f"Не найден входной файл: {INPUT}", file=sys.stderr)
        return 1
    if not MODEL.exists():
        print(f"Не найдена модель: {MODEL}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(INPUT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps, (width, height))

    print(f"Вход: {INPUT} ({width}x{height}, {fps:.1f} fps, {total} кадров)")

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=2,
    )

    frames = 0
    frames_with_face = 0
    start = time.time()
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frames / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.face_landmarks:
                frames_with_face += 1
                draw_landmarks(frame, result.face_landmarks)

            writer.write(frame)

    cap.release()
    writer.release()

    elapsed = time.time() - start
    proc_fps = frames / max(elapsed, 1e-6)
    face_share = frames_with_face / max(frames, 1)
    print(f"Обработано кадров: {frames}")
    print(f"С лицом: {frames_with_face} ({face_share:.1%})")
    print(f"Время: {elapsed:.2f} с, FPS обработки: {proc_fps:.1f}")
    print(f"Результат: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
