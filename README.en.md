# MiniProctor

[Russian version](README.md)

A CV-based proctoring MVP. Takes webcam video, runs 5 violation detectors, returns an event timeline and CSV report. Streamlit web app for interactive demo.

Pet project for a portfolio. Goal -- to demonstrate the full CV/ML cycle: data collection, labeling, training, metrics, UI, conclusions.

## What's inside

Five violation detectors:

1. **Head pose / gaze** -- head rotation angles from MediaPipe Face Landmarker, "looked away" flag when thresholds are exceeded.
2. **Multi-face** -- more than one face in frame, a possible helper.
3. **No face** -- the examinee is not in frame.
4. **Phone** -- a phone in the frame, fine-tuned YOLOv8n.
5. **Book** -- a book or sheet of paper as a "cheat sheet" in the frame, fine-tuned YOLOv8n.

## Metrics

Per-frame precision / recall / F1 on 7 labeled test clips. Comparison of three YOLO versions (pretrained / Roboflow fine-tune / Roboflow + 89 own labeled frames).

| **Event** | **F1 Baseline** | **F1 Pass 2 (final)** | **Δ** |
|:---|:---|:---|:---|
| phone | 0.588 | **0.728** | +0.140 |
| book | 0.072 | **0.368** | +0.296 |
| gaze_away | -- | 0.743 | -- |
| multi_face | -- | 0.997 | -- |
| no_face | -- | 0.602 | -- |

Detailed tables and raw data are in [docs/metrics.md](docs/metrics.md).

## Stack

- Python 3.11+, PyTorch with CUDA;
- OpenCV, MediaPipe Tasks API;
- Ultralytics YOLOv8 (object detection), YOLO-World (auto-labeling);
- Streamlit, Plotly (UI);
- Roboflow Universe (source of public datasets).

## Quick start

```bash
git clone https://github.com/EValentyuk/MiniProctor.git
cd MiniProctor

python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt

# Download the MediaPipe model (one-time step):
.venv/Scripts/python.exe -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'models/face_landmarker.task')"

# Pretrained YOLO is auto-downloaded on first run. Fine-tuned weights are
# not committed to the repo -- either train locally or grab from GitHub
# Releases (see docs/github-upload.md).

# Run the Streamlit app:
.venv/Scripts/python.exe -m streamlit run src/app.py
```

The UI opens at `http://localhost:8501`. The sidebar lets you pick a ready-made clip, switch between models (Pretrained / Pass 1 / Pass 2), and tune thresholds.

## Command-line scripts

```bash
# Record a 15-second webcam clip:
python src/record_webcam.py data/raw/my_clip.mp4 15

# Run all detectors on one clip and save the annotated video:
python src/run_detectors.py data/raw/my_clip.mp4

# Per-frame precision/recall/F1 evaluator:
python src/evaluator.py --yolo-weights models/yolov8n_pass2.pt --out data/metrics/my_eval.csv

# Fine-tune YOLO:
python src/train_yolo.py --name my_run --epochs 20 --batch 8 \
    --start-weights models/yolov8n.pt
```

## Project structure

```
data/
  raw/            # source clips (gitignored)
  labels/         # ground_truth.csv
  datasets/       # downloaded Roboflow datasets
  dataset_merged/ # merged dataset for fine-tune
  own_labeled/    # YOLO-World auto-labeled frames
  processed/      # annotated videos
  metrics/        # metrics CSVs
src/              # detectors, evaluator, UI, training code
models/           # weights (face_landmarker.task, yolov8n*.pt) -- gitignored
runs/             # YOLO training logs -- gitignored
docs/             # documentation
notebooks/        # Jupyter experiments
```

## Documentation

- **[docs/brief.md](docs/brief.md)** -- full problem statement, week plan, risks (Russian).
- **[docs/portfolio-report.md](docs/portfolio-report.md)** -- employer-facing report: what was done, why each step, how it connects to systems analysis (Russian).
- **[docs/architecture.md](docs/architecture.md)** -- C4 diagrams (System Context, Containers), module map, data flow.
- **[docs/metrics.md](docs/metrics.md)** -- master metrics table across all model versions and clips.
- **[docs/github-upload.md](docs/github-upload.md)** -- GitHub upload instructions (Russian).
- **[MiniProctor-results.md](MiniProctor-results.md)** -- session journal: what was done each day, what metrics were obtained (Russian).

## Limitations

- **Batch mode, not real-time.** The pipeline processes a whole file, not suited for online proctoring.
- **Small test sample.** Seven clips, mostly one subject. Metrics are not representative of a real student population.
- **Multi-face is synthetic.** One test clip was made by overlaying an AI-generated face, not by actual two-person recording.
- **No authentication, encryption of recordings, or audit log.** Out of scope for a pet project.
- **Book FP tail.** On empty frames the Pass 2 model occasionally reports false books. Mitigated by raising the confidence threshold or adding negative samples to training.

Details in [docs/architecture.md](docs/architecture.md), section "Out of scope".

## What's next

- Record and label a larger, more diverse dataset.
- Fine calibration of confidence thresholds per class.
- Negative mining: add empty frames to training without labels to suppress FPs.
- Real-time pipeline on smaller model variants.
- Comparison with alternative pretrained models (YOLOv11, RT-DETR).

## License

Project code -- MIT. YOLO weights -- AGPL under Ultralytics policy, MediaPipe weights -- Apache 2.0. Keep AGPL in mind for commercial use with YOLO weights.
