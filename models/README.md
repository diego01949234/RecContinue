# models/

Local model weights are not committed to this repository (see `.gitignore`).

## MediaPipe Pose Landmarker (needed for Phase 3 — Simple Movement Analysis)

1. Download the pose landmarker task file from Google's MediaPipe model
   index (`pose_landmarker_lite.task` is sufficient for this prototype's
   upper-body-only landmark subset).
2. Place the downloaded file at `models/pose_landmarker_lite.task`.
3. `movement_analysis.py` (Phase 3) will load the model from that path at
runtime. Nothing in this project downloads the model automatically.

## MediaPipe Hand Landmarker (needed for palm opening/closing)

1. Download `hand_landmarker.task` from the official MediaPipe model bundle:
   `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
2. Place it at `models/hand_landmarker.task`.
3. RecContinue uses it only for the **手掌** test. The **頭部** and **手臂**
   tests continue to use the existing Pose Landmarker model.

## Gemma 4 (via Ollama)

Gemma itself is not stored in this repository. It is pulled and served
locally by Ollama:

```bash
ollama pull gemma4:e2b
ollama serve
```
