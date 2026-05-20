# MiniProctor

[English version](README.en.md)

MVP-система прокторинга на компьютерном зрении. Принимает видео с вебки, прогоняет 5 детекторов нарушений, выдаёт таймлайн событий и CSV-отчёт. Streamlit-приложение для интерактивной демонстрации.

Pet-проект для портфолио. Цель -- показать полный цикл CV/ML: сбор данных → разметка → обучение → метрики → UI → выводы.

## Что внутри

5 детекторов нарушений:

1. **Head pose / gaze** -- углы поворота головы из MediaPipe Face Landmarker, флаг «отвёл взгляд» при выходе за пороги.
2. **Multi-face** -- несколько лиц в кадре, признак подсказчика.
3. **No face** -- отсутствие лица в кадре.
4. **Phone** -- телефон в кадре, fine-tuned YOLOv8n.
5. **Book** -- книга или лист как «шпаргалка» в кадре, fine-tuned YOLOv8n.

## Метрики

Покадровые precision / recall / F1 на 7 размеченных тестовых клипах. Сравнение трёх версий YOLO (pretrained / fine-tune на Roboflow / fine-tune на Roboflow + 89 собственных кадров).

| **Событие** | **F1 Baseline** | **F1 Pass 2 (финал)** | **Δ** |
|:---|:---|:---|:---|
| phone | 0.588 | **0.728** | +0.140 |
| book | 0.072 | **0.368** | +0.296 |
| gaze_away | -- | 0.743 | -- |
| multi_face | -- | 0.997 | -- |
| no_face | -- | 0.602 | -- |

Подробные таблицы и сырые данные -- в [docs/metrics.md](docs/metrics.md).

## Стек

- Python 3.11+, PyTorch с CUDA;
- OpenCV, MediaPipe Tasks API;
- Ultralytics YOLOv8 (детекция объектов), YOLO-World (авторазметка);
- Streamlit, Plotly (UI);
- Roboflow Universe (источник публичных датасетов).

## Быстрый запуск

```bash
git clone https://github.com/EValentyuk/MiniProctor.git
cd MiniProctor

python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Скачать модели MediaPipe (один раз):
.venv/Scripts/python.exe -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'models/face_landmarker.task')"

# Pretrained YOLO скачается автоматически при первом запуске. Fine-tuned
# веса в репозиторий не коммитим -- их надо либо обучить локально, либо
# скачать из GitHub Releases (см. docs/github-upload.md).

# Запуск Streamlit-приложения:
.venv/Scripts/python.exe -m streamlit run src/app.py
```

UI откроется на `http://localhost:8501`. В сайдбаре можно выбрать готовый клип, переключить модель (Pretrained / Pass 1 / Pass 2), покрутить пороги.

## Командные скрипты

```bash
# Запись тестового клипа с вебки на 15 секунд:
python src/record_webcam.py data/raw/my_clip.mp4 15

# Прогон всех детекторов на одном клипе с сохранением аннотированного видео:
python src/run_detectors.py data/raw/my_clip.mp4

# Покадровый оценщик precision/recall/F1:
python src/evaluator.py --yolo-weights models/yolov8n_pass2.pt --out data/metrics/my_eval.csv

# Fine-tune YOLO:
python src/train_yolo.py --name my_run --epochs 20 --batch 8 \
    --start-weights models/yolov8n.pt
```

## Структура проекта

```
data/
  raw/            # исходные клипы (не в git)
  labels/         # ground_truth.csv
  datasets/       # скачанные Roboflow-датасеты
  dataset_merged/ # объединённый датасет для fine-tune
  own_labeled/    # авторазмеченные YOLO-World кадры
  processed/      # аннотированные видео
  metrics/        # CSV с метриками
src/              # код детекторов, оценщика, UI и тренировки
models/           # веса (face_landmarker.task, yolov8n*.pt) -- не в git
runs/             # логи тренировок YOLO -- не в git
docs/             # документация
notebooks/        # Jupyter-эксперименты
```

## Документация

- **[docs/brief.md](docs/brief.md)** -- полная постановка задачи, план на неделю, риски.
- **[docs/portfolio-report.md](docs/portfolio-report.md)** -- отчёт для работодателя: что я делал, зачем каждый шаг и как это связано с системным анализом.
- **[docs/architecture.md](docs/architecture.md)** -- C4-диаграммы (System Context, Containers), карта модулей, поток данных.
- **[docs/metrics.md](docs/metrics.md)** -- сводная таблица метрик по всем версиям моделей и клипам.
- **[docs/github-upload.md](docs/github-upload.md)** -- инструкция по заливке проекта в GitHub.
- **[MiniProctor-results.md](MiniProctor-results.md)** -- журнал сессий: что сделано в какой день, какие метрики получены.

## Ограничения

- **Batch-режим, не real-time.** Анализ обрабатывает файл целиком, не подходит для онлайн-прокторинга.
- **Малая тестовая выборка.** 7 клипов, в основном один экзаменуемый. Метрики не репрезентативны для популяции студентов.
- **Multi-face -- синтетика.** Один тест-клип создан вставкой AI-сгенерированного лица, не реальной съёмкой с двумя людьми.
- **Нет аутентификации, шифрования записей, audit log.** Out of scope для пет-проекта.
- **Book FP-хвост.** На пустых кадрах модель Pass 2 ловит ложные книги. Решается повышением conf threshold или негативными примерами в обучении.

Подробности в [docs/architecture.md](docs/architecture.md), раздел «Что вынесено за скобки».

## Что дальше

- Запись и разметка большего и разнообразного датасета.
- Тонкая калибровка conf thresholds для каждого класса.
- Negative mining: добавить пустые кадры в обучение без меток для подавления FP.
- Real-time pipeline на сокращённой версии моделей.
- Сравнение с альтернативными pretrained-моделями (YOLOv11, RT-DETR).

## Лицензия

Код проекта -- MIT. Веса YOLO -- AGPL по политике Ultralytics, веса MediaPipe -- Apache 2.0. При использовании в коммерческих продуктах с YOLO учитывайте AGPL.
