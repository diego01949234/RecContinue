# models/

Local model weights are not committed to this repository (see `.gitignore`).

## MediaPipe Pose Landmarker (needed for Phase 3 — Simple Movement Analysis)

1. Download the pose landmarker task file from Google's MediaPipe model
   index (`pose_landmarker_lite.task` is sufficient for this prototype's
   upper-body-only landmark subset).
2. Place the downloaded file at `models/pose_landmarker_lite.task`.
3. `movement_analysis.py` (Phase 3) will load the model from that path at
   runtime. Nothing in this project downloads the model automatically.

## Gemma 4 (via Ollama)

Gemma itself is not stored in this repository. It is pulled and served
locally by Ollama:

```bash
ollama pull gemma4:e2b
ollama serve
```
