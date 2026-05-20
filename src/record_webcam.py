"""Запись короткого ролика с вебки для smoke-тестов."""
import sys
import time
from pathlib import Path

import cv2

DEFAULT_DURATION_SEC = 15
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "smoke.mp4"


def parse_args() -> tuple[Path, int]:
    """Принимает: [output_path] [duration_sec]."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DURATION_SEC
    return output, duration


def main() -> int:
    OUTPUT, DURATION_SEC = parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Не удалось открыть вебку (индекс 0).", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps, (width, height))

    print(f"Запись {DURATION_SEC} с в {OUTPUT}")
    print(f"Разрешение: {width}x{height}, FPS источника: {fps:.1f}")
    print("Нажми Q в окне предпросмотра, чтобы остановить раньше.")

    start = time.time()
    frames = 0
    while time.time() - start < DURATION_SEC:
        ok, frame = cap.read()
        if not ok:
            print("Кадр не получен, прерывание.", file=sys.stderr)
            break
        writer.write(frame)
        frames += 1

        elapsed = time.time() - start
        remaining = max(0, DURATION_SEC - elapsed)
        cv2.putText(
            frame,
            f"REC {remaining:4.1f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    real_fps = frames / max(time.time() - start, 1e-6)
    print(f"Готово. Кадров: {frames}, фактический FPS записи: {real_fps:.1f}")
    print(f"Файл: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
