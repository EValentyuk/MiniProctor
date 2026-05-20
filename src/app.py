"""Streamlit-приложение MiniProctor: видео + детекторы + таймлайн событий.

Запуск:
    streamlit run src/app.py
"""
import csv
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from events import run_full_analysis  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
TMP_DIR = ROOT / "data" / "uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VARIANTS = {
    "Pretrained (COCO)": MODELS_DIR / "yolov8n.pt",
    "Pass 1 (Roboflow only)": MODELS_DIR / "yolov8n_finetuned.pt",
    "Pass 2 (Roboflow + own)": MODELS_DIR / "yolov8n_pass2.pt",
}

# Цвета для таймлайна -- единая палитра.
EVENT_COLORS = {
    "phone": "#e53935",       # красный
    "book": "#ef6c00",        # оранжевый
    "gaze_away": "#fbc02d",   # жёлтый
    "no_face": "#757575",     # серый
    "multi_face": "#8e24aa",  # фиолетовый
}

EVENT_LABELS = {
    "phone": "Телефон",
    "book": "Книга/шпаргалка",
    "gaze_away": "Отвод взгляда",
    "no_face": "Нет в кадре",
    "multi_face": "Второе лицо",
}


def list_local_clips() -> list[Path]:
    if not RAW_DIR.exists():
        return []
    return sorted(p for p in RAW_DIR.glob("*.mp4"))


def save_uploaded(uploaded) -> Path:
    out = TMP_DIR / uploaded.name
    out.write_bytes(uploaded.getbuffer())
    return out


def build_timeline_fig(events, duration_sec: float) -> go.Figure:
    """Горизонтальная диаграмма Ганта: ряд на тип события."""
    fig = go.Figure()
    row_order = ["phone", "book", "gaze_away", "no_face", "multi_face"]
    for ev_type in row_order:
        rows = [e for e in events if e.event_type == ev_type]
        if not rows:
            fig.add_trace(go.Bar(
                x=[0], y=[EVENT_LABELS[ev_type]],
                base=[0], orientation="h",
                marker=dict(color=EVENT_COLORS[ev_type], opacity=0.0),
                showlegend=False, hoverinfo="skip",
            ))
            continue
        for e in rows:
            fig.add_trace(go.Bar(
                x=[e.duration_sec],
                y=[EVENT_LABELS[ev_type]],
                base=[e.start_sec],
                orientation="h",
                marker=dict(color=EVENT_COLORS[ev_type]),
                showlegend=False,
                hovertemplate=(
                    f"<b>{EVENT_LABELS[ev_type]}</b><br>"
                    f"{e.start_sec:.2f}–{e.end_sec:.2f} с<br>"
                    f"длительность {e.duration_sec:.2f} с<extra></extra>"
                ),
            ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Время, с", range=[0, duration_sec], showgrid=True),
        yaxis=dict(title="", categoryorder="array",
                   categoryarray=[EVENT_LABELS[k] for k in row_order]),
        height=320,
        margin=dict(l=10, r=10, t=20, b=40),
        plot_bgcolor="#fafafa",
    )
    return fig


def events_to_dataframe(events) -> pd.DataFrame:
    rows = [
        {
            "Тип события": EVENT_LABELS[e.event_type],
            "Начало, с": round(e.start_sec, 2),
            "Конец, с": round(e.end_sec, 2),
            "Длительность, с": round(e.duration_sec, 2),
        }
        for e in events
    ]
    return pd.DataFrame(rows)


def events_to_csv_bytes(events) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["event_type", "start_sec", "end_sec", "duration_sec"])
    for e in events:
        w.writerow([
            e.event_type,
            f"{e.start_sec:.2f}",
            f"{e.end_sec:.2f}",
            f"{e.duration_sec:.2f}",
        ])
    return buf.getvalue().encode("utf-8")


@st.cache_data(show_spinner=False)
def cached_analysis(
    clip_path_str: str,
    weights_path_str: str,
    yaw_t: float,
    pitch_t: float,
    gaze_min: int,
    merge_gap: float,
    min_dur: float,
):
    # Кешируем по сигнатуре аргументов.
    return run_full_analysis(
        Path(clip_path_str),
        yolo_weights=Path(weights_path_str),
        yaw_threshold=yaw_t,
        pitch_threshold=pitch_t,
        gaze_min_frames=gaze_min,
        merge_gap_sec=merge_gap,
        min_duration_sec=min_dur,
    )


def main() -> None:
    st.set_page_config(page_title="MiniProctor", layout="wide")
    st.title("MiniProctor -- прокторинг через CV-детекторы")
    st.caption(
        "Загрузите видео или выберите один из готовых клипов. "
        "Приложение прогонит детекторы и покажет таймлайн нарушений."
    )

    with st.sidebar:
        st.header("Параметры анализа")

        source = st.radio(
            "Источник видео",
            ["Готовый клип", "Загрузить своё"],
            horizontal=True,
        )

        clip_path: Path | None = None
        if source == "Готовый клип":
            clips = list_local_clips()
            if not clips:
                st.warning("В data/raw/ нет mp4-файлов.")
            else:
                names = [c.name for c in clips]
                selected = st.selectbox("Клип", names, index=0)
                clip_path = next(c for c in clips if c.name == selected)
        else:
            uploaded = st.file_uploader(
                "Видео (mp4)", type=["mp4", "mov", "avi"]
            )
            if uploaded:
                clip_path = save_uploaded(uploaded)
                st.success(f"Загружено: {clip_path.name}")

        st.divider()
        model_label = st.radio(
            "Модель для phone/book",
            list(MODEL_VARIANTS.keys()),
            index=2,
            help="Pass 2 -- finetuned на Roboflow + наших кадрах, рекомендуемая.",
        )
        weights_path = MODEL_VARIANTS[model_label]

        st.divider()
        st.subheader("Пороги head pose")
        yaw_t = st.slider("Yaw threshold, °", 5.0, 45.0, 10.0, 1.0)
        pitch_t = st.slider("Pitch threshold, °", 5.0, 45.0, 15.0, 1.0)
        gaze_min = st.slider(
            "Кадров подряд для флага", 1, 30, 10, 1,
            help="Сколько кадров с отводом взгляда должно идти подряд, "
                 "чтобы сработал флаг.",
        )

        st.divider()
        st.subheader("Сглаживание таймлайна")
        merge_gap = st.slider(
            "Слить разрывы до, с", 0.0, 1.0, 0.3, 0.05,
            help="Соседние эпизоды одного типа сливаются в один, если разрыв "
                 "между ними не больше этого значения.",
        )
        min_dur = st.slider(
            "Мин. длительность эпизода, с", 0.0, 1.0, 0.2, 0.05,
            help="Эпизоды короче порога отбрасываются как шум детектора.",
        )

    if not clip_path or not clip_path.exists():
        st.info("Выберите клип в левой панели или загрузите файл.")
        return
    if not weights_path.exists():
        st.error(f"Не найден файл весов: {weights_path}")
        return

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("Видео")
        st.video(str(clip_path))

    with col_right:
        st.subheader("Метаданные")
        st.write(f"**Файл:** `{clip_path.name}`")
        st.write(f"**Модель phone/book:** {model_label}")
        st.write(f"**Пороги:** yaw>{yaw_t}°, pitch>{pitch_t}°, "
                 f"мин. кадров {gaze_min}")
        run_btn = st.button("Запустить анализ", type="primary")

    if not run_btn:
        st.info(
            "Нажмите «Запустить анализ» для прогона детекторов. "
            "На 15-секундном клипе ~10–20 секунд."
        )
        return

    with st.spinner("Прогоняю детекторы..."):
        result = cached_analysis(
            str(clip_path), str(weights_path),
            yaw_t, pitch_t, gaze_min,
            merge_gap, min_dur,
        )

    st.divider()
    st.subheader("Таймлайн нарушений")
    fig = build_timeline_fig(result["events"], result["duration_sec"])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Сводка")
    totals = result["totals"]
    cols = st.columns(len(EVENT_LABELS))
    for col, (ev_type, label) in zip(cols, EVENT_LABELS.items()):
        t = totals.get(ev_type, {"n_events": 0, "duration_sec": 0, "share_pct": 0})
        col.metric(
            label,
            f"{t['n_events']} эпизод(ов)",
            delta=f"{t['duration_sec']:.1f} с ({t['share_pct']:.1f}%)",
            delta_color="off",
        )

    st.subheader("Инциденты")
    df = events_to_dataframe(result["events"])
    if df.empty:
        st.success("Нарушений не зафиксировано. Чистый клип.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            "Скачать CSV",
            data=events_to_csv_bytes(result["events"]),
            file_name=f"{clip_path.stem}_events.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
