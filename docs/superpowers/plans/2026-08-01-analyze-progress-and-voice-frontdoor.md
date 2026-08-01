# Analyze Progress Display + Voice Front-Door Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two blank-wait moments in RecContinue's patient flow — Tab 2's "Analyze" and the perceived lack of orientation on Tab 1 — with a live progress readout during Analyze, and an optional local-only voice front-door on Tab 1 that reconfirms the day's one assigned task and pre-fills Tab 3.

**Architecture:** Two independent, additive units. (A) `movement_analysis.analyze_video()` becomes a generator that yields throttled progress dicts before its existing final `(path, metrics)` tuple; `app.py`'s handler and a new Tab 2 Markdown component surface those yields. (B) A new `stt_client.py` (local `faster-whisper`) plus a new lightweight, non-JSON Gemma prompt path in `prompts.py`/`gemma_client.py` power a Tab 1 voice recorder whose transcript pre-fills Tab 3's existing patient-statement field; a small `safety.py` refactor lets the same flagged-phrase scanner used on the full PEO report also run over this short acknowledgment text.

**Tech Stack:** Python 3.12, Gradio 6.22 (generator-valued event handlers), OpenCV + MediaPipe (existing), Ollama/Gemma via `requests` (existing), `faster-whisper` (new, local/on-device CPU inference), pytest with `unittest.mock`.

## Global Constraints

- Everything stays local/on-device — no new network calls at inference time (only faster-whisper's one-time model download, mirroring the existing MediaPipe Pose Landmarker model download).
- No diagnosis, no treatment/exercise recommendation, no safety/normalcy judgment — enforced the same way as the existing PEO report path (a constrained system prompt + a mandatory `safety.py` scan before display).
- The synthetic demo keeps its single hardcoded patient/task (`sample_data/synthetic_patient.json`) — no new task data model.
- Every new interactive element (voice recorder, live progress) must be skippable/optional — it must never block the existing golden path (record → analyze → context → generate → export → review).
- Match existing code style: module-level dataclasses/exceptions for graceful degradation (see `MovementAnalysisUnavailableError`, `OllamaUnavailableError`), `unittest.mock`-based tests with `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` at the top of each test file, no live Ollama/mic/camera required to run the suite.
- Spec reference: `docs/superpowers/specs/2026-08-01-analyze-progress-and-voice-frontdoor-design.md`.

---

### Task 1: `frame_elbow_angle()` pure helper

**Files:**
- Modify: `movement_analysis.py`
- Test: `tests/test_analyze_progress.py` (new)

**Interfaces:**
- Produces: `frame_elbow_angle(frame_data: Optional[dict], selected_arm: Arm) -> Optional[float]` — used by Task 2's generator loop to compute the live angle shown in progress updates. `frame_data` has the same shape as one entry of `analyze_video`'s existing `frames_data` list (keys: `nose`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `elbow`, `wrist`, `index`, `pinky`, each `{"x", "y", "visibility"}`), or `None` if no pose was detected in that frame.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_progress.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movement_analysis import frame_elbow_angle  # noqa: E402

RIGHT_ARM_FRAME = {
    "left_shoulder": {"x": 0.4, "y": 0.3, "visibility": 1.0},
    "right_shoulder": {"x": 0.6, "y": 0.3, "visibility": 1.0},
    "elbow": {"x": 0.6, "y": 0.5, "visibility": 1.0},
    "wrist": {"x": 0.6, "y": 0.7, "visibility": 1.0},
}


def test_frame_elbow_angle_computes_for_right_arm():
    angle = frame_elbow_angle(RIGHT_ARM_FRAME, "right")
    assert angle == 180.0


def test_frame_elbow_angle_returns_none_when_frame_is_none():
    assert frame_elbow_angle(None, "right") is None


def test_frame_elbow_angle_returns_none_when_elbow_missing():
    frame = dict(RIGHT_ARM_FRAME)
    frame["elbow"] = None
    assert frame_elbow_angle(frame, "right") is None


def test_frame_elbow_angle_returns_none_when_wrist_missing():
    frame = dict(RIGHT_ARM_FRAME)
    frame["wrist"] = None
    assert frame_elbow_angle(frame, "right") is None


def test_frame_elbow_angle_uses_left_shoulder_for_left_arm():
    frame = {
        "left_shoulder": {"x": 0.4, "y": 0.3, "visibility": 1.0},
        "right_shoulder": {"x": 0.6, "y": 0.3, "visibility": 1.0},
        "elbow": {"x": 0.4, "y": 0.5, "visibility": 1.0},
        "wrist": {"x": 0.4, "y": 0.7, "visibility": 1.0},
    }
    angle = frame_elbow_angle(frame, "left")
    assert angle == 180.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_analyze_progress.py -v`
Expected: FAIL with `ImportError: cannot import name 'frame_elbow_angle'`

- [ ] **Step 3: Write minimal implementation**

In `movement_analysis.py`, add this function immediately after `compute_session_metrics` (after line 255, before `def _require_cv`):

```python
def frame_elbow_angle(frame_data: Optional[dict], selected_arm: Arm) -> Optional[float]:
    """Compute the 2D elbow angle for a single frame's landmark dict.

    Returns None if no pose was detected this frame (frame_data is None)
    or the elbow/wrist points are missing. Used for the live progress
    readout during analyze_video(); the authoritative peak-reach angle
    still comes from compute_session_metrics() over the full session.
    """
    if frame_data is None:
        return None
    shoulder_key = "left_shoulder" if selected_arm == "left" else "right_shoulder"
    shoulder_pt = frame_data[shoulder_key]
    elbow, wrist = frame_data.get("elbow"), frame_data.get("wrist")
    if elbow is None or wrist is None:
        return None
    return round(
        angle_2d((shoulder_pt["x"], shoulder_pt["y"]), (elbow["x"], elbow["y"]), (wrist["x"], wrist["y"])),
        1,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_analyze_progress.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add movement_analysis.py tests/test_analyze_progress.py
git commit -m "feat: add frame_elbow_angle for per-frame live angle readout"
```

---

### Task 2: `analyze_video()` becomes a progress-yielding generator

**Files:**
- Modify: `movement_analysis.py:295-365` (the `analyze_video` function)
- Test: `tests/test_analyze_progress.py`

**Interfaces:**
- Consumes: `frame_elbow_angle()` from Task 1, existing `RepetitionCounter`, `compute_session_metrics()`.
- Produces: `analyze_video(video_path, selected_arm, output_path=None)` is now a **generator**. It yields zero or more progress dicts shaped `{"frame_index": int, "total_frames": Optional[int], "current_angle": Optional[float], "reps_so_far": int}`, then yields exactly one final `(annotated_video_path: str, metrics: dict)` tuple as its last item — the same shape `analyze_video` returned before this change. Task 3's `app.py` handler consumes this generator and discriminates progress vs. final by `isinstance(item, dict)`. Same exceptions as before (`MovementAnalysisUnavailableError`, `NoPoseDetectedError`, `ValueError`) — still raised, not yielded.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analyze_progress.py`:

```python
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import movement_analysis as ma


def test_analyze_video_is_a_generator_function():
    assert inspect.isgeneratorfunction(ma.analyze_video)


class _FakeLandmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def _fake_pose_landmarks():
    fixed = {
        ma.NOSE: (0.5, 0.2), ma.LEFT_EAR: (0.45, 0.2), ma.RIGHT_EAR: (0.55, 0.2),
        ma.LEFT_SHOULDER: (0.4, 0.3), ma.RIGHT_SHOULDER: (0.6, 0.3),
        ma.RIGHT_ELBOW: (0.6, 0.5), ma.RIGHT_WRIST: (0.6, 0.7),
        ma.RIGHT_INDEX: (0.61, 0.71), ma.RIGHT_PINKY: (0.62, 0.72), ma.RIGHT_THUMB: (0.59, 0.71),
        ma.LEFT_ELBOW: (0.4, 0.5), ma.LEFT_WRIST: (0.4, 0.7),
        ma.LEFT_INDEX: (0.39, 0.71), ma.LEFT_PINKY: (0.38, 0.72), ma.LEFT_THUMB: (0.41, 0.71),
    }
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(23)]
    for idx, (x, y) in fixed.items():
        landmarks[idx] = _FakeLandmark(x, y)
    return [landmarks]


class _FakeCapture:
    def __init__(self, total_frames=12):
        self._remaining = total_frames
        self._total = total_frames

    def isOpened(self):
        return True

    def get(self, prop_id):
        return {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 64,
            cv2.CAP_PROP_FRAME_HEIGHT: 48,
            cv2.CAP_PROP_FRAME_COUNT: self._total,
        }.get(prop_id, 0)

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        pass


class _FakeLandmarker:
    def detect_for_video(self, mp_image, timestamp_ms):
        return SimpleNamespace(pose_landmarks=_fake_pose_landmarks())

    def close(self):
        pass


def test_analyze_video_yields_progress_before_final_result(tmp_path):
    output_path = str(tmp_path / "out.mp4")
    with patch.object(ma, "_require_cv", lambda: None), \
         patch.object(ma.cv2, "VideoCapture", lambda path: _FakeCapture(total_frames=12)), \
         patch.object(ma.cv2, "VideoWriter", lambda *a, **kw: MagicMock()), \
         patch.object(ma.mp_vision.PoseLandmarker, "create_from_options", lambda options: _FakeLandmarker()):
        items = list(ma.analyze_video("fake.mp4", "right", output_path=output_path))

    progress_items = [item for item in items if isinstance(item, dict)]
    final_items = [item for item in items if not isinstance(item, dict)]

    assert len(progress_items) >= 1
    assert progress_items[0]["frame_index"] == 0
    assert progress_items[0]["total_frames"] == 12
    assert progress_items[0]["current_angle"] == 180.0

    assert len(final_items) == 1
    annotated_path, metrics = final_items[0]
    assert annotated_path == output_path
    assert metrics["repetition_count"] == 0
    assert metrics["selected_arm"] == "right"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_analyze_progress.py -v`
Expected: FAIL — `test_analyze_video_is_a_generator_function` fails because `analyze_video` has no `yield` yet (returns a plain tuple, not a generator function).

- [ ] **Step 3: Write minimal implementation**

In `movement_analysis.py`, replace the whole body of `analyze_video` (lines 295-365) with:

```python
PROGRESS_YIELD_EVERY_N_FRAMES = 5


def analyze_video(video_path: str, selected_arm: Arm, output_path: Optional[str] = None):
    """Run local MediaPipe/OpenCV analysis on a video.

    Yields a progress dict every PROGRESS_YIELD_EVERY_N_FRAMES frames:
    {"frame_index", "total_frames" (None if unknown), "current_angle"
    (None if no pose detected that frame), "reps_so_far"}. The last item
    yielded is always the final (annotated_video_path, metrics_dict)
    tuple — the same shape this function returned before it became a
    generator.

    Raises NoPoseDetectedError if no pose was found in any frame, and
    MovementAnalysisUnavailableError if the required dependencies/model
    are not installed. Never fabricates a result for a video it could not
    analyze.
    """
    _require_cv()
    indices = arm_landmark_indices(selected_arm)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    if output_path is None:
        output_path = str(pathlib.Path(video_path).with_name(pathlib.Path(video_path).stem + "_annotated.mp4"))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    base_options = BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO)
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    frames_data: list[dict] = []
    live_rep_counter = RepetitionCounter()
    frame_index = 0
    try:
        while True:
            ok, bgr_frame = cap.read()
            if not ok:
                break
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * (1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_data = None
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                frame_data = {
                    "_arm": selected_arm,
                    "nose": _landmark_point(landmarks, NOSE),
                    "left_ear": _landmark_point(landmarks, LEFT_EAR),
                    "right_ear": _landmark_point(landmarks, RIGHT_EAR),
                    "left_shoulder": _landmark_point(landmarks, LEFT_SHOULDER),
                    "right_shoulder": _landmark_point(landmarks, RIGHT_SHOULDER),
                    "elbow": _landmark_point(landmarks, indices["elbow"]),
                    "wrist": _landmark_point(landmarks, indices["wrist"]),
                    "index": _landmark_point(landmarks, indices["index"]),
                    "pinky": _landmark_point(landmarks, indices["pinky"]),
                }
                palm_point, _ = approximate_palm_center(frame_data["wrist"], frame_data["index"], frame_data["pinky"])
                _draw_overlay(bgr_frame, frame_data, palm_point, width, height)
                frames_data.append(frame_data)

            writer.write(bgr_frame)

            current_angle = frame_elbow_angle(frame_data, selected_arm)
            live_rep_counter.update(current_angle)
            if frame_index % PROGRESS_YIELD_EVERY_N_FRAMES == 0:
                yield {
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "current_angle": current_angle,
                    "reps_so_far": live_rep_counter.repetition_count,
                }

            frame_index += 1
    finally:
        cap.release()
        writer.release()
        landmarker.close()

    if not frames_data:
        raise NoPoseDetectedError("No pose was detected in this video.")

    metrics = compute_session_metrics(frames_data, fps, selected_arm)
    yield output_path, metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_analyze_progress.py -v`
Expected: PASS (7 tests)

Then run the full suite to confirm nothing else broke:

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest -v`
Expected: PASS (all tests — `analyze_video`'s only other caller is `app.py`, updated in Task 3)

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add movement_analysis.py tests/test_analyze_progress.py
git commit -m "feat: analyze_video yields live progress before final result"
```

---

### Task 3: Tab 2 progress display in `app.py`

**Files:**
- Modify: `app.py:301-323` (`use_synthetic_metrics_fallback`, `analyze_video_handler`)
- Modify: `app.py:591-635` (Tab 2 layout + event wiring)
- Test: `tests/test_app_handlers.py` (new)

**Interfaces:**
- Consumes: `analyze_video()` from Task 2 (generator yielding progress dicts then a final tuple).
- Produces: `analyze_video_handler(video_path, selected_arm)` is now a generator yielding 4-tuples `(annotated_output_value, metrics_display_value, metrics_state_value, analysis_progress_md_value)` — consumed directly by Gradio's `.click()`/`.stop_recording()` machinery, and by Task 3's own test via `list(...)`. `use_synthetic_metrics_fallback()` now returns a 3-tuple (adds an empty progress string). `_progress_markdown(progress: dict) -> str` is a new pure helper.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_handlers.py`:

```python
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402


def _fake_analyze_video(video_path, selected_arm):
    yield {"frame_index": 5, "total_frames": 12, "current_angle": 96.0, "reps_so_far": 1}
    yield "annotated.mp4", {"repetition_count": 1, "selected_arm": selected_arm}


def test_analyze_video_handler_yields_progress_then_result():
    with patch("app.analyze_video", side_effect=_fake_analyze_video):
        updates = list(app.analyze_video_handler("session.mp4", "right"))

    assert len(updates) == 2

    progress_update = updates[0]
    assert "frame 5/12" in progress_update[3]
    assert "96.0" in progress_update[3]
    assert "reps so far: 1" in progress_update[3]

    final_update = updates[1]
    assert final_update[0] == "annotated.mp4"
    assert final_update[2] == {"repetition_count": 1, "selected_arm": "right"}
    assert final_update[3] == ""


def test_analyze_video_handler_no_video_selected():
    updates = list(app.analyze_video_handler(None, "right"))
    assert len(updates) == 1
    assert updates[0][0] is None
    assert "No video selected" in updates[0][1]


def test_analyze_video_handler_reports_unavailable_error():
    def _raises(video_path, selected_arm):
        raise app.MovementAnalysisUnavailableError("mediapipe not installed")
        yield  # pragma: no cover - makes this a generator function

    with patch("app.analyze_video", side_effect=_raises):
        updates = list(app.analyze_video_handler("session.mp4", "right"))

    assert len(updates) == 1
    assert updates[0][0] is None
    assert "mediapipe not installed" in updates[0][1]


def test_use_synthetic_metrics_fallback_clears_progress_text():
    metrics, markdown, progress = app.use_synthetic_metrics_fallback()
    assert metrics == app.SYNTHETIC_METRICS
    assert progress == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_app_handlers.py -v`
Expected: FAIL — `test_use_synthetic_metrics_fallback_clears_progress_text` fails with a 2-tuple/3-tuple unpack error, and the others fail because `app.analyze_video_handler` isn't a generator yet (patching `side_effect` with a generator function on a non-generator handler won't produce the expected multi-yield sequence).

- [ ] **Step 3: Write minimal implementation**

In `app.py`, replace `use_synthetic_metrics_fallback` (lines 301-302):

```python
def use_synthetic_metrics_fallback():
    return SYNTHETIC_METRICS, _metrics_markdown(SYNTHETIC_METRICS, is_synthetic=True), ""
```

Replace `analyze_video_handler` (lines 305-323) with:

```python
def _progress_markdown(progress: dict) -> str:
    total = progress["total_frames"]
    total_str = f"/{total}" if total else ""
    angle = f"{progress['current_angle']}°" if progress["current_angle"] is not None else "—"
    return (
        f"Processing frame {progress['frame_index']}{total_str} · "
        f"elbow angle ≈ {angle} · reps so far: {progress['reps_so_far']}"
    )


def analyze_video_handler(video_path, selected_arm):
    if video_path is None:
        yield None, "No video selected. Choose a file above, or use the synthetic metrics fallback.", None, ""
        return

    try:
        for item in analyze_video(video_path, selected_arm):
            if isinstance(item, dict):
                yield gr.update(), gr.update(), gr.update(), _progress_markdown(item)
            else:
                annotated_path, metrics = item
                yield annotated_path, _metrics_markdown(metrics, is_synthetic=False), metrics, ""
    except MovementAnalysisUnavailableError as exc:
        yield None, f"⚠️ {exc}", None, ""
    except NoPoseDetectedError as exc:
        yield (
            None,
            f"⚠️ {exc} Try a video with better lighting and the {selected_arm} arm fully in "
            "frame, or use the synthetic metrics fallback below.",
            None,
            "",
        )
    except ValueError as exc:
        yield None, f"⚠️ {exc} Check that the file is a supported video format (e.g. MP4).", None, ""
```

In `app.py`'s Tab 2 block, add `analysis_progress_md` right after the button row (after line 605, before the `annotated_output` group closes/starts at line 606):

```python
                    video_input = gr.Video(sources=["webcam", "upload"], label="Session camera")
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze video", variant="primary")
                        synthetic_fallback_btn = gr.Button("Use synthetic metrics fallback instead", variant="secondary")
                    analysis_progress_md = gr.Markdown("")
                with gr.Group(elem_classes=["kc-card"]):
                    annotated_output = gr.Video(label="Annotated output", interactive=False)
```

Update the three event wirings right below it (previously lines 618-630):

```python
                analyze_btn.click(
                    fn=analyze_video_handler,
                    inputs=[video_input, arm_radio],
                    outputs=[annotated_output, metrics_display, metrics_state, analysis_progress_md],
                )
                video_input.stop_recording(
                    fn=analyze_video_handler,
                    inputs=[video_input, arm_radio],
                    outputs=[annotated_output, metrics_display, metrics_state, analysis_progress_md],
                )
                synthetic_fallback_btn.click(
                    fn=use_synthetic_metrics_fallback,
                    outputs=[metrics_state, metrics_display, analysis_progress_md],
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_app_handlers.py -v`
Expected: PASS (4 tests)

Then confirm the app still boots (imports cleanly and builds its component tree):

Run: `cd reccontinue && source .venv/bin/activate && python -c "import app; app.build_app()"`
Expected: no exception; prints nothing (or Gradio's usual startup notices), exits 0.

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add app.py tests/test_app_handlers.py
git commit -m "feat: show live analyze progress on Tab 2 instead of a blank wait"
```

---

### Task 4: `stt_client.py` — local Whisper transcription

**Files:**
- Create: `stt_client.py`
- Test: `tests/test_stt_client.py` (new)

**Interfaces:**
- Produces: `transcribe_audio(audio_path: str) -> str` (raises `WhisperUnavailableError` if `faster-whisper` or its model can't be loaded); `WhisperUnavailableError` exception class. Used by Task 8's `app.py` Tab 1 handler.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stt_client.py`:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stt_client  # noqa: E402


class _FakeSegment:
    def __init__(self, text):
        self.text = text


def test_transcribe_audio_joins_and_strips_segments():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([_FakeSegment(" Hello there. "), _FakeSegment(" How are you? ")], None)
    with patch("stt_client._get_model", return_value=fake_model):
        text = stt_client.transcribe_audio("dummy.wav")
    assert text == "Hello there. How are you?"


def test_transcribe_audio_raises_when_dependency_missing():
    with patch("stt_client._FASTER_WHISPER_AVAILABLE", False):
        with pytest.raises(stt_client.WhisperUnavailableError):
            stt_client.transcribe_audio("dummy.wav")


def test_get_model_raises_when_model_load_fails():
    with patch("stt_client._FASTER_WHISPER_AVAILABLE", True), \
         patch("stt_client._model", None), \
         patch("stt_client.WhisperModel", side_effect=RuntimeError("no weights")):
        with pytest.raises(stt_client.WhisperUnavailableError):
            stt_client._get_model()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_stt_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stt_client'`

- [ ] **Step 3: Write minimal implementation**

First, install the new dependency and record the resolved version:

```bash
cd reccontinue && source .venv/bin/activate && pip install faster-whisper
pip freeze | grep -i faster-whisper
```

Append the exact resolved line (e.g. `faster-whisper==1.1.1`) to `requirements.txt`.

Create `stt_client.py`:

```python
"""Local speech-to-text for RecContinue's Tab 1 voice front-door.

Wraps faster-whisper, loaded lazily and run entirely on-device (CPU).
The model weights are downloaded once by faster-whisper the first time
transcribe_audio() runs, then cached locally — same posture as the
gitignored MediaPipe Pose Landmarker model (see models/README.md).
Nothing here makes a network call during actual transcription.
"""
from __future__ import annotations

try:
    from faster_whisper import WhisperModel

    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False

WHISPER_MODEL_SIZE = "base"

_model = None


class WhisperUnavailableError(Exception):
    """Raised when faster-whisper or its model can't be loaded locally."""


def _get_model():
    global _model
    if not _FASTER_WHISPER_AVAILABLE:
        raise WhisperUnavailableError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt`."
        )
    if _model is None:
        try:
            _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        except Exception as exc:
            raise WhisperUnavailableError(
                f"Could not load the local Whisper model ({WHISPER_MODEL_SIZE}): {exc}"
            ) from exc
    return _model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe a local audio file entirely on-device and return the text.

    Raises WhisperUnavailableError if faster-whisper or its model can't be
    loaded. Never makes a network call at transcription time.
    """
    model = _get_model()
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments).strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_stt_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add stt_client.py tests/test_stt_client.py requirements.txt
git commit -m "feat: add local faster-whisper transcription client"
```

---

### Task 5: Acknowledgment prompt in `prompts.py`

**Files:**
- Modify: `prompts.py`
- Test: `tests/test_prompts.py` (new)

**Interfaces:**
- Produces: `ACKNOWLEDGMENT_SYSTEM_PROMPT: str`, `build_acknowledgment_prompt(transcript: str, assigned_task: str) -> str`. Used by Task 6's `gemma_client.generate_acknowledgment`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompts.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import ACKNOWLEDGMENT_SYSTEM_PROMPT, build_acknowledgment_prompt  # noqa: E402


def test_build_acknowledgment_prompt_includes_transcript_and_task():
    prompt = build_acknowledgment_prompt("My arm feels a little stiff today.", "Reach-and-place cup task")
    assert "My arm feels a little stiff today." in prompt
    assert "Reach-and-place cup task" in prompt


def test_acknowledgment_system_prompt_forbids_recommendations():
    lowered = ACKNOWLEDGMENT_SYSTEM_PROMPT.lower()
    assert "must not" in lowered
    assert "recommend" in lowered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_prompts.py -v`
Expected: FAIL with `ImportError: cannot import name 'ACKNOWLEDGMENT_SYSTEM_PROMPT'`

- [ ] **Step 3: Write minimal implementation**

Append to `prompts.py` (after `build_repair_prompt`):

```python
ACKNOWLEDGMENT_SYSTEM_PROMPT = """You are RecContinue, an offline rehabilitation companion.

A patient is about to start a rehabilitation activity that their therapist
already assigned. They just described, in their own words, how they are
feeling about today's session.

Reply with one short, warm, plain-language paragraph (2-3 sentences) that:

1. Briefly acknowledges what they said, in neutral, non-clinical language.
2. Reconfirms the name of the one activity their therapist already
   assigned for today. Do not propose, suggest, or imply any other
   activity or exercise.
3. Makes clear the therapist decides what to do, not you.

You must not:

1. Diagnose, judge, or classify how the patient is doing.
2. Recommend a different exercise, treatment, or amount of activity.
3. Say whether continuing is safe or unsafe, or whether their movement
   is normal or abnormal.
4. Invent facts the patient did not say.

Reply with plain text only - no JSON, no markdown, no lists."""


def build_acknowledgment_prompt(transcript: str, assigned_task: str) -> str:
    """Build the Gemma user-turn prompt for the Tab 1 voice front-door reply.

    `transcript` is the patient's locally transcribed voice note;
    `assigned_task` is the one therapist-assigned activity name for the
    synthetic patient. Only locally held information is included here.
    """
    return (
        f'The patient\'s assigned activity today is: "{assigned_task}".\n\n'
        f'The patient said: "{transcript}"\n\n'
        "Reply following the rules in your system prompt."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_prompts.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add prompts.py tests/test_prompts.py
git commit -m "feat: add acknowledgment prompt for Tab 1 voice front-door"
```

---

### Task 6: `generate_acknowledgment()` in `gemma_client.py`

**Files:**
- Modify: `gemma_client.py:17` (import line), `gemma_client.py:116-149` (`_call_ollama_generate`), add new function after `generate_report`
- Test: `tests/test_gemma_client.py` (append)

**Interfaces:**
- Consumes: `ACKNOWLEDGMENT_SYSTEM_PROMPT`, `build_acknowledgment_prompt()` from Task 5.
- Produces: `generate_acknowledgment(transcript: str, assigned_task: str, host=DEFAULT_OLLAMA_HOST, model=DEFAULT_MODEL, timeout=GENERATE_TIMEOUT_SECONDS) -> str`. Raises the same `OllamaUnavailableError` / `GemmaTimeoutError` / `GemmaModelNotInstalledError` as `generate_report` (reuses `_call_ollama_generate`). Used by Task 8's `app.py` Tab 1 handler.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gemma_client.py`:

```python
def test_generate_acknowledgment_returns_plain_text_and_uses_ack_system_prompt():
    fake_response = _fake_generate_response(
        "Thanks for sharing that. Let's continue with the reach-and-place cup task your therapist assigned."
    )
    with patch("gemma_client.requests.post", return_value=fake_response) as mock_post:
        text = gemma_client.generate_acknowledgment("My arm feels stiff.", "Reach-and-place cup task")

    assert "reach-and-place cup task" in text.lower()
    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["system"] == gemma_client.ACKNOWLEDGMENT_SYSTEM_PROMPT
    assert "format" not in sent_json


def test_generate_report_still_requests_json_format():
    fake_response = _fake_generate_response(json.dumps(VALID_REPORT_JSON))
    with patch("gemma_client.requests.post", return_value=fake_response) as mock_post:
        gemma_client.generate_report(
            "SYN-001-session", {"patient_id": "SYN-001"}, {"repetition_count": 3}, {"patient_statement": "hi"}
        )
    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["format"] == "json"
    assert sent_json["system"] == gemma_client.SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_gemma_client.py -v`
Expected: FAIL with `AttributeError: module 'gemma_client' has no attribute 'generate_acknowledgment'`

- [ ] **Step 3: Write minimal implementation**

In `gemma_client.py`, change the import line (line 17):

```python
from prompts import ACKNOWLEDGMENT_SYSTEM_PROMPT, SYSTEM_PROMPT, build_acknowledgment_prompt, build_repair_prompt, build_user_prompt
```

Replace `_call_ollama_generate` (lines 116-149) with:

```python
def _call_ollama_generate(
    prompt: str,
    host: str,
    model: str,
    timeout: int,
    system: str = SYSTEM_PROMPT,
    json_format: bool = True,
) -> str:
    request_body = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
    }
    if json_format:
        request_body["format"] = "json"

    try:
        response = requests.post(
            f"{host}/api/generate",
            json=request_body,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise GemmaTimeoutError(f"Gemma did not respond within {timeout}s.") from exc
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {host}. Is `ollama serve` running? ({exc.__class__.__name__})"
        ) from exc

    if response.status_code == 404:
        raise GemmaModelNotInstalledError(
            f"Model `{model}` is not installed. Run `ollama pull {model}`."
        )
    response.raise_for_status()

    body = response.json()
    return body.get("response", "")
```

Add this function after `generate_report` (after line 188, before the `if __name__ == "__main__":` block):

```python
def generate_acknowledgment(
    transcript: str,
    assigned_task: str,
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
) -> str:
    """Generate a short, plain-text Tab 1 voice front-door acknowledgment.

    Unlike generate_report, this is plain text, not JSON, and is not
    validated against RecContinueReport. Callers must still run the
    result through safety.validate_text_safety before display — this
    function only calls Gemma, it does not scan the reply.
    """
    prompt = build_acknowledgment_prompt(transcript, assigned_task)
    raw_text = _call_ollama_generate(
        prompt, host, model, timeout, system=ACKNOWLEDGMENT_SYSTEM_PROMPT, json_format=False
    )
    return raw_text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_gemma_client.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add gemma_client.py tests/test_gemma_client.py
git commit -m "feat: add generate_acknowledgment for Tab 1 voice front-door"
```

---

### Task 7: `validate_text_safety()` in `safety.py`

**Files:**
- Modify: `safety.py`
- Test: `tests/test_safety.py` (append)

**Interfaces:**
- Produces: `validate_text_safety(text: str) -> SafetyValidationResult` — same `SafetyValidationResult`/`FlaggedPhrase` types `validate_report_safety` already returns. Used by Task 8's `app.py` Tab 1 handler to gate the Gemma acknowledgment reply before display.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_safety.py`:

```python
from safety import validate_text_safety  # noqa: E402


def test_validate_text_safety_passes_clean_text():
    result = validate_text_safety("Thanks for sharing that. Let's continue with today's reach-and-place task.")
    assert result.passed is True
    assert result.flagged == []


def test_validate_text_safety_flags_recommended_exercise():
    result = validate_text_safety("You should perform additional overhead reaches to help with that.")
    assert result.passed is False
    assert any(f.phrase == "should perform" for f in result.flagged)


def test_validate_text_safety_uses_same_scan_as_report_safety():
    # Same underlying scanner as validate_report_safety - not a separate phrase list.
    result = validate_text_safety("This is unsafe to continue without supervision.")
    assert result.passed is False
    assert any(f.phrase == "unsafe" for f in result.flagged)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_safety.py -v`
Expected: FAIL with `ImportError: cannot import name 'validate_text_safety'`

- [ ] **Step 3: Write minimal implementation**

In `safety.py`, replace `validate_report_safety` (lines 60-74) with:

```python
def _scan_sections(sections: list[str]) -> SafetyValidationResult:
    flagged: list[FlaggedPhrase] = []
    for text in sections:
        for sentence in _split_sentences(text):
            lowered = sentence.lower()
            for phrase in FLAGGED_PHRASES:
                if phrase in lowered:
                    flagged.append(FlaggedPhrase(phrase=phrase, sentence=sentence))
    return SafetyValidationResult(passed=not flagged, flagged=flagged)


def validate_report_safety(report: RecContinueReport) -> SafetyValidationResult:
    """Scan report.clinical_text_sections() for FLAGGED_PHRASES.

    Deliberately excludes report_status and safety_notice, which are
    disclaimer/label fields rather than AI-generated clinical content
    (see RecContinueReport.clinical_text_sections()).
    """
    return _scan_sections(report.clinical_text_sections())


def validate_text_safety(text: str) -> SafetyValidationResult:
    """Scan a single free-text string (e.g. the Tab 1 voice-front-door
    Gemma acknowledgment) using the same FLAGGED_PHRASES rules as
    validate_report_safety.
    """
    return _scan_sections([text])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_safety.py -v`
Expected: PASS (all tests, including the three new ones)

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add safety.py tests/test_safety.py
git commit -m "feat: add validate_text_safety, reusing the report safety scanner"
```

---

### Task 8: Tab 1 voice front-door in `app.py`

**Files:**
- Modify: `app.py:21-43` (imports), Tab 1 layout (`app.py:572-589`), Tab 3 layout end (`app.py:663-670`)
- Test: `tests/test_app_handlers.py` (append)

**Interfaces:**
- Consumes: `transcribe_audio`, `WhisperUnavailableError` (Task 4); `generate_acknowledgment` (Task 6); `validate_text_safety` (Task 7); existing `OllamaUnavailableError`, `GemmaModelNotInstalledError`, `GemmaTimeoutError` (already imported in `app.py`); `SYNTHETIC_PATIENT["task"]`.
- Produces: `voice_frontdoor_handler(audio_path) -> (transcript_markdown: str, acknowledgment_markdown: str, patient_statement_update: dict)`. The third value is a `gr.update(value=...)` (or no-op `gr.update()`) targeting Tab 3's existing `patient_statement_tb`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app_handlers.py`:

```python
def test_voice_frontdoor_handler_no_audio_is_a_noop():
    transcript_md, ack_md, statement_update = app.voice_frontdoor_handler(None)
    assert transcript_md == ""
    assert ack_md == ""
    assert "value" not in statement_update


def test_voice_frontdoor_handler_transcribes_and_acknowledges():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), \
         patch(
             "app.generate_acknowledgment",
             return_value="Thanks for sharing. Let's continue with the reach-and-place cup task.",
         ):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "My arm feels stiff today." in transcript_md
    assert "reach-and-place cup task" in ack_md.lower()
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_suppresses_flagged_acknowledgment():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), \
         patch("app.generate_acknowledgment", return_value="You have a diagnosed rotator cuff injury."):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert ack_md == ""
    assert "My arm feels stiff today." in transcript_md
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_whisper_unavailable_does_not_crash():
    with patch("app.transcribe_audio", side_effect=app.WhisperUnavailableError("faster-whisper not installed")):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "faster-whisper not installed" in transcript_md
    assert ack_md == ""
    assert "value" not in statement_update


def test_voice_frontdoor_handler_gemma_unavailable_still_prefills_statement():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), \
         patch("app.generate_acknowledgment", side_effect=app.OllamaUnavailableError("no ollama")):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert ack_md == ""
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_no_speech_detected():
    with patch("app.transcribe_audio", return_value="   "):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "No speech detected" in transcript_md
    assert ack_md == ""
    assert "value" not in statement_update
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_app_handlers.py -v`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'voice_frontdoor_handler'`

- [ ] **Step 3: Write minimal implementation**

In `app.py`, update the import block (lines 21-42) to add the new imports:

```python
from gemma_client import (
    GemmaInvalidResponseError,
    GemmaModelNotInstalledError,
    GemmaTimeoutError,
    OllamaUnavailableError,
    check_ollama_connection,
    generate_acknowledgment,
    generate_report,
)
from movement_analysis import (
    MovementAnalysisUnavailableError,
    NoPoseDetectedError,
    analyze_video,
)
from packet_export import (
    PacketImportError,
    SafetyGateError,
    build_clinician_packet,
    export_packet,
    export_reviewed_report,
    import_packet,
)
from safety import validate_report_safety, validate_text_safety
from stt_client import WhisperUnavailableError, transcribe_audio
import storage
```

Add this handler in `app.py` near `analyze_video_handler` (module level, before `def build_app():`):

```python
def voice_frontdoor_handler(audio_path):
    if audio_path is None:
        return "", "", gr.update()

    try:
        transcript = transcribe_audio(audio_path)
    except WhisperUnavailableError as exc:
        return f"⚠️ {exc}", "", gr.update()

    if not transcript.strip():
        return "_No speech detected — you can skip this and continue to Step 2._", "", gr.update()

    try:
        acknowledgment = generate_acknowledgment(transcript, SYNTHETIC_PATIENT["task"])
    except (OllamaUnavailableError, GemmaModelNotInstalledError, GemmaTimeoutError):
        acknowledgment = ""

    if acknowledgment and not validate_text_safety(acknowledgment).passed:
        acknowledgment = ""

    return f"**You said:** {transcript}", acknowledgment, gr.update(value=transcript)
```

In Tab 1's layout (right after the `TASK_INSTRUCTIONS_MD` group, currently `app.py:584-585`, before the `PRIVACY_EXPLANATION_MD` group), add:

```python
                with gr.Group(elem_classes=["kc-card"]):
                    gr.Markdown(TASK_INSTRUCTIONS_MD)
                with gr.Group(elem_classes=["kc-card"]):
                    gr.Markdown(
                        '<p class="kc-card-title">How are you feeling about today’s session? (optional)</p>'
                        "Record a short voice note if you like — it's transcribed on this "
                        "device only, never sent anywhere, and Gemma will just confirm "
                        "today's assigned activity back to you. You can skip this entirely "
                        "and continue straight to Step 2."
                    )
                    voice_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice note")
                    voice_transcript_md = gr.Markdown("")
                    voice_ack_md = gr.Markdown("")
                with gr.Group(elem_classes=["kc-card"]):
                    gr.Markdown(PRIVACY_EXPLANATION_MD)
```

`patient_statement_tb` doesn't exist until Tab 3 is built later in the same function, so `voice_input`'s event wiring can't be attached inside Tab 1's block. Add it right after Tab 3's block instead — immediately after the `continue3_btn = gr.Button(...)` line (`app.py:669`) and before `with gr.Tab("4 · Private Session Record", id=3):` starts:

```python
                with gr.Row(elem_classes=["kc-continue"]):
                    continue3_btn = gr.Button("Continue to Step 4: Generate Record →", variant="primary")

                voice_input.stop_recording(
                    fn=voice_frontdoor_handler,
                    inputs=[voice_input],
                    outputs=[voice_transcript_md, voice_ack_md, patient_statement_tb],
                )

            with gr.Tab("4 · Private Session Record", id=3):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest tests/test_app_handlers.py -v`
Expected: PASS (all tests, including the six new ones)

Then confirm the app still builds (this exercises the cross-tab `patient_statement_tb` reference at Blocks-construction time, which unit tests alone don't cover):

Run: `cd reccontinue && source .venv/bin/activate && python -c "import app; app.build_app()"`
Expected: no exception, no `NameError`/`UnboundLocalError` for `patient_statement_tb` or `voice_input`.

- [ ] **Step 5: Commit**

```bash
cd reccontinue
git add app.py tests/test_app_handlers.py
git commit -m "feat: add Tab 1 voice front-door that pre-fills Tab 3's patient statement"
```

---

### Task 9: Full regression pass + docs

**Files:**
- Modify: `README.md` (Implementation status, Setup, Known limitations sections)

**Interfaces:** None — this task only runs verification and updates documentation to match the shipped behavior; no new code.

- [ ] **Step 1: Run the complete test suite**

Run: `cd reccontinue && source .venv/bin/activate && python -m pytest -v`
Expected: PASS — all tests from Tasks 1-8 plus the pre-existing 48, with no regressions.

- [ ] **Step 2: Update `README.md`**

In the "Implementation status" section, add a short paragraph after the Phase 5 bullet (before the Phase 6 bullet) describing what was added:

```markdown
- ✅ **Post-P0 addition — Analyze progress + Tab 1 voice front-door**:
  `movement_analysis.analyze_video()` now yields live progress (frame
  count, current elbow angle, running rep count) while it runs, shown on
  Tab 2 instead of a blank wait. Tab 1 adds an optional voice note,
  transcribed locally with `faster-whisper` (`stt_client.py`), that gets
  a short Gemma acknowledgment reconfirming the one therapist-assigned
  task (never a new one) and pre-fills Tab 3's patient statement. The
  acknowledgment is scanned by the same `safety.py` flagged-phrase
  checker used on the full PEO report before it's ever shown. Both
  features are additive and skippable — see
  `docs/superpowers/specs/2026-08-01-analyze-progress-and-voice-frontdoor-design.md`.
```

In the "Setup" section, after the MediaPipe Pose Landmarker model paragraph, add:

```markdown
To use Tab 1's optional voice front-door, no extra setup is needed beyond
`pip install -r requirements.txt` — `faster-whisper` downloads its small
model automatically the first time you record a voice note, then caches
it locally. Without a microphone or without `faster-whisper` installed,
Tab 1 shows a clear message and the rest of the flow is unaffected.
```

In "Known limitations", add:

```markdown
- The Tab 1 voice front-door (transcription → Gemma acknowledgment →
  Tab 3 pre-fill) was verified with `transcribe_audio`,
  `generate_acknowledgment`, and `validate_text_safety` mocked in
  `tests/test_app_handlers.py`; it has not been exercised with a real
  microphone or a live Ollama instance in this environment. Before the
  live demo, do one manual pass with a real mic and `ollama serve`
  running to confirm end-to-end latency and audio quality.
- The Tab 2 live progress display was verified against a mocked
  MediaPipe/OpenCV pipeline in `tests/test_analyze_progress.py`; do one
  manual pass with a real recorded/uploaded video before the live demo
  to confirm the progress text updates smoothly rather than flooding
  the UI (adjust `PROGRESS_YIELD_EVERY_N_FRAMES` in
  `movement_analysis.py` if it does).
```

- [ ] **Step 3: Commit**

```bash
cd reccontinue
git add README.md
git commit -m "docs: document analyze progress + voice front-door additions"
```
