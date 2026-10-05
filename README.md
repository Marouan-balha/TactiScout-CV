# TactiScout-CV

Offline computer-vision worker for the TactiLens evidence pipeline.

The worker currently produces observations only: player tracks, team labels, ball detections, an annotated inspection video, and immutable run metadata. It does not produce calibrated pitch coordinates, tactical events, or verified evidence.

## Setup

Use a clean Python 3.11 environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Keep footage and model weights outside Git. The local worker expects them at `data\` and `models\` unless explicit paths are supplied.

## Run

```powershell
.\.venv\Scripts\python run_pipeline.py `
  --video data\match_5min.mp4 `
  --model models\best.pt `
  --output-dir output\tl06-run-001 `
  --device cpu
```

Use `--device cuda:0` on a configured NVIDIA GPU instance. Results are written to the output directory and include `run.json`, `tracks.json`, `tracks_with_teams.json`, `ball_observations.json`, and `tracked.mp4`.

## Scope

The next required stages are pitch homography, held-out annotations, detection/tracking metrics, tactical event extraction, and conversion to validated TactiLens candidate evidence. Raw model output must not be marked as verified.