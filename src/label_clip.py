"""Интерактивный разметчик ground truth.

Запуск: python label_clip.py <clip_path> <event_type>

Управление:
- SPACE  -- пауза/воспроизведение
- S      -- начало сегмента в текущий момент
- E      -- конец сегмента в текущий момент
- U      -- отменить последний сегмент или незакрытый старт
- ,/.    -- перемотка -1 / +1 секунда
- стрелки влево/вправо -- то же
- q      -- сохранить и выйти

Результат добавляется в data/labels/ground_truth.csv с колонками:
clip, start_sec, end_sec, event_type.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "labels" / "ground_truth.csv"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("clip")
    p.add_argument("event")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    args = p.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"Не найден: {clip_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps
    print(f"Клип: {clip_path.name} ({duration:.2f} с, {total} кадров, {fps:.1f} fps)")
    print(f"Событие: {args.event}")
    print("S/E -- старт/конец, U -- отмена, ,/. или стрелки -- -/+1 с, SPACE -- пауза, q -- выйти")

    segments: list[tuple[float, float]] = []
    pending_start: float | None = None
    paused = True  # стартуем на паузе, чтобы было время сориентироваться
    frame_idx = 0
    speed = 0.5  # коэффициент воспроизведения (0.25..2.0)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            frame_idx = max(0, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            paused = True
            if not ok:
                break

        t = frame_idx / fps

        # HUD
        state = "PAUSED" if paused else f"PLAY x{speed:.2f}"
        lines = [
            f"{args.event}  {t:5.2f}/{duration:.2f} s  frame {frame_idx}/{total - 1}  [{state}]",
            "S start | E end | U undo | ,/. arrows -/+1s | SPACE pause | -/+ speed | q save",
        ]
        if pending_start is not None:
            lines.append(f"PENDING start = {pending_start:.2f} s")
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Прогресс-бар
        bar_y = height - 40
        cv2.rectangle(frame, (10, bar_y), (width - 10, bar_y + 8), (60, 60, 60), -1)
        progress_x = int(10 + (width - 20) * (frame_idx / max(total - 1, 1)))
        cv2.rectangle(frame, (10, bar_y), (progress_x, bar_y + 8), (0, 200, 0), -1)
        for s, e in segments:
            x1 = int(10 + (width - 20) * (s * fps / max(total - 1, 1)))
            x2 = int(10 + (width - 20) * (e * fps / max(total - 1, 1)))
            cv2.rectangle(frame, (x1, bar_y - 4), (x2, bar_y + 12), (0, 0, 255), 2)

        if segments:
            seg_text = "  ".join(f"[{s:.1f}-{e:.1f}]" for s, e in segments[-4:])
            cv2.putText(frame, f"segments: {seg_text}", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.imshow("Label clip", frame)
        wait = max(1, int(1000 / (fps * speed))) if not paused else 30
        key = cv2.waitKeyEx(wait)

        if key == -1:
            if not paused and frame_idx < total - 1:
                frame_idx += 1
            elif not paused:
                paused = True
            continue

        k = key & 0xFF
        if k == ord('q'):
            break
        elif k == ord(' '):
            paused = not paused
        elif k == ord('s'):
            pending_start = t
            print(f"  start @ {t:.2f}")
        elif k == ord('e'):
            if pending_start is not None:
                segments.append((pending_start, t))
                print(f"  end   @ {t:.2f}  -> [{pending_start:.2f}, {t:.2f}]")
                pending_start = None
            else:
                print("  (нет открытого сегмента)")
        elif k == ord('u'):
            if pending_start is not None:
                print(f"  отмена pending start {pending_start:.2f}")
                pending_start = None
            elif segments:
                last = segments.pop()
                print(f"  отмена сегмента [{last[0]:.2f}, {last[1]:.2f}]")
        elif k == ord(',') or key == 2424832:
            frame_idx = max(0, frame_idx - int(fps))
            paused = True
        elif k == ord('.') or key == 2555904:
            frame_idx = min(total - 1, frame_idx + int(fps))
            paused = True
        elif k == ord('-') or k == ord('_'):
            speed = max(0.1, speed - 0.25)
            print(f"  speed = {speed:.2f}")
        elif k == ord('+') or k == ord('='):
            speed = min(4.0, speed + 0.25)
            print(f"  speed = {speed:.2f}")
        else:
            if not paused and frame_idx < total - 1:
                frame_idx += 1
            elif not paused:
                paused = True

    cap.release()
    cv2.destroyAllWindows()

    if pending_start is not None:
        print(f"  внимание: незакрытый pending start {pending_start:.2f} -- не сохранён")

    if not segments:
        print("Сегментов не размечено, CSV не изменён.")
        return 0

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["clip", "start_sec", "end_sec", "event_type"])
        for s, e in segments:
            w.writerow([clip_path.name, f"{s:.2f}", f"{e:.2f}", args.event])
    print(f"Сохранено {len(segments)} сегментов в {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
