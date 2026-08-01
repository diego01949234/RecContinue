# Live MediaPipe overlay + in-session time-track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RecContinue's record-then-analyze webcam flow with a live-streaming flow that overlays MediaPipe landmarks on the camera feed in real time and, once finished, shows a time-track line chart of the session's primary metric alongside the existing metrics summary.

**Architecture:** `gr.Image(sources=["webcam"], streaming=True)` streams frames to a `live_frame_handler` that runs the module-appropriate MediaPipe landmarker per frame (via new `create_live_session`/`process_live_frame` functions in `movement_analysis.py`), draws the same restricted overlay the recorded-clip pipeline already draws, and returns the annotated frame for immediate display. A `finish_btn` closes the session via `finalize_live_session`, which reuses the existing `compute_session_metrics`/`compute_palm_closure_metrics`/`compute_head_turn_metrics` unchanged.

**Tech Stack:** Python, Gradio 6.22 (`gr.Image` streaming, `gr.LinePlot`), OpenCV, MediaPipe Tasks (PoseLandmarker/HandLandmarker), pandas (already a transitive Gradio dependency; pin it directly since Task 3 imports it).

## Global Constraints

- Only the landmark subset SPEC.md section 6 already restricts per module (head: nose/ears; arm: shoulder/elbow/wrist + approximate palm; palm: wrist + 5 fingertips) may be read or drawn — enforced by reusing the existing `_draw_overlay`/`_draw_head_overlay`/`_draw_hand_overlay` functions unmodified.
- `observation_module` values are the English strings already canonical in the current codebase: `"head"`, `"palm"`, `"arm"` (see `ACTIVITY_MODULES` keys in `app.py` and `compute_*_metrics`'s `"observation_module"` fields in `movement_analysis.py`). Do not introduce Chinese module-name strings — `tests/test_app_handlers.py` and a few other test files still contain pre-existing Chinese-string assertions (e.g. `"手臂"`, `"手掌"`) left over from an in-progress, unrelated localization pass; 5 tests already fail on `main` for this reason before this plan starts (confirmed via `pytest -q`). Do not fix those — out of scope. Only touch tests this plan's tasks explicitly name.
- Keep `analyze_video` and its three per-module helpers (`_analyze_head_video`, `_analyze_palm_video`, `_analyze_arm_video`) in `movement_analysis.py` untouched — they stay tested by `tests/test_analyze_progress.py` and `tests/test_repetition_counter.py` even though `app.py` stops calling them after Task 3.

---

### Task 1: Per-frame head-turn-proxy helper

**Files:**
- Modify: `movement_analysis.py` (insert after `frame_elbow_angle`, currently ending at line 456)
- Test: `tests/test_angles.py`

**Interfaces:**
- Produces: `frame_head_turn_proxy(frame_data: Optional[dict]) -> Optional[float]` — given one frame's `nose`/`left_ear`/`right_ear` landmark dicts (each `{"x", "y", "visibility"}`), returns `(nose.x - ear_mid_x) / ear_width` rounded to 2 decimals, or `None` if `frame_data` is `None` or an ear/nose landmark is missing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_angles.py`:

```python
from movement_analysis import frame_head_turn_proxy  # noqa: E402

HEAD_FRAME = {
    "nose": {"x": 0.55, "y": 0.2, "visibility": 1.0},
    "left_ear": {"x": 0.4, "y": 0.2, "visibility": 1.0},
    "right_ear": {"x": 0.6, "y": 0.2, "visibility": 1.0},
}


def test_frame_head_turn_proxy_computes_offset():
    assert frame_head_turn_proxy(HEAD_FRAME) == 0.5


def test_frame_head_turn_proxy_returns_none_when_frame_is_none():
    assert frame_head_turn_proxy(None) is None


def test_frame_head_turn_proxy_returns_none_when_ear_missing():
    frame = dict(HEAD_FRAME)
    frame["left_ear"] = None
    assert frame_head_turn_proxy(frame) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.runtime-venv/bin/python -m pytest tests/test_angles.py -k head_turn_proxy -v`
Expected: FAIL with `ImportError: cannot import name 'frame_head_turn_proxy'`

- [ ] **Step 3: Implement**

Insert into `movement_analysis.py` immediately after `frame_elbow_angle` (after its closing line, currently line 456, before `def _require_cv`):

```python
def frame_head_turn_proxy(frame_data: Optional[dict]) -> Optional[float]:
    """Compute the head-turn proxy for one frame's nose/ear landmarks.

    Returns ``None`` when the frame has no detected pose or an ear/nose
    landmark is unavailable. Mirrors `frame_elbow_angle`'s per-frame shape
    so a live session can report a value every frame, not only in
    aggregate the way `compute_head_turn_metrics` does.
    """
    if frame_data is None:
        return None
    nose = frame_data.get("nose")
    left_ear, right_ear = frame_data.get("left_ear"), frame_data.get("right_ear")
    if nose is None or left_ear is None or right_ear is None:
        return None
    ear_mid_x = (left_ear["x"] + right_ear["x"]) / 2
    ear_width = abs(right_ear["x"] - left_ear["x"])
    if ear_width <= 1e-6:
        return None
    return round((nose["x"] - ear_mid_x) / ear_width, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.runtime-venv/bin/python -m pytest tests/test_angles.py -v`
Expected: all PASS (existing `frame_elbow_angle` tests plus the 3 new ones)

- [ ] **Step 5: Commit**

```bash
git add movement_analysis.py tests/test_angles.py
git commit -m "feat: add per-frame head-turn-proxy helper for live sessions"
```

---

### Task 2: Live session — create/process/finalize

**Files:**
- Modify: `movement_analysis.py` (add imports; append new code at end of file, after `analyze_video`, currently ending at line 767)
- Test: `tests/test_live_session.py` (new file)

**Interfaces:**
- Consumes: `frame_head_turn_proxy` (Task 1), and the existing `_draw_overlay`, `_draw_head_overlay`, `_draw_hand_overlay`, `_landmark_point`, `arm_landmark_indices`, `approximate_palm_center`, `frame_elbow_angle`, `palm_closure_ratio`, `compute_session_metrics`, `compute_palm_closure_metrics`, `compute_head_turn_metrics`, `_require_cv`, `_require_hand_runtime`, `MODEL_PATH`, `HAND_MODEL_PATH`, `BaseOptions`, `mp_vision`, `mp`, `cv2`, `NoPoseDetectedError`, `MovementAnalysisUnavailableError`.
- Produces:
  - `LiveSession` dataclass with fields `observation_module: str`, `selected_arm: Arm`, `landmarker: Any = None`, `writer: Any = None`, `output_path: str = ""`, `frames_data: list[dict]`, `hand_frames: list[list[dict]]`, `timeseries: list[tuple[float, float]]`, `start_time: Optional[float] = None`.
  - `create_live_session(observation_module: str, selected_arm: Arm) -> LiveSession`
  - `process_live_frame(session: LiveSession, rgb_frame: "np.ndarray", frame_timestamp_ms: int) -> tuple["np.ndarray", Optional[float]]` — returns `(annotated_rgb_frame, current_metric_value)`
  - `finalize_live_session(session: LiveSession) -> tuple[str, dict, list[tuple[float, float]]]` — returns `(output_path, metrics, timeseries)`
  - `LIVE_STREAM_FPS: float = 4.0` module constant

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_session.py`:

```python
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import movement_analysis as ma  # noqa: E402


class _FakeLandmark:
    def __init__(self, x, y, visibility=1.0):
        self.x, self.y, self.visibility = x, y, visibility


def _fake_pose_landmarks():
    fixed = {
        ma.NOSE: (0.5, 0.2), ma.LEFT_EAR: (0.45, 0.2), ma.RIGHT_EAR: (0.55, 0.2),
        ma.LEFT_SHOULDER: (0.4, 0.3), ma.RIGHT_SHOULDER: (0.6, 0.3),
        ma.RIGHT_ELBOW: (0.6, 0.5), ma.RIGHT_WRIST: (0.6, 0.7),
        ma.RIGHT_INDEX: (0.61, 0.71), ma.RIGHT_PINKY: (0.62, 0.72), ma.RIGHT_THUMB: (0.59, 0.71),
    }
    landmarks = [_FakeLandmark(0.5, 0.5) for _ in range(23)]
    for idx, (x, y) in fixed.items():
        landmarks[idx] = _FakeLandmark(x, y)
    return [landmarks]


def _hand_frame(tip_y: float):
    points = [_FakeLandmark(0.5, 0.5) for _ in range(21)]
    points[0] = _FakeLandmark(0.5, 0.5)
    points[9] = _FakeLandmark(0.5, 0.6)
    for index in (4, 8, 12, 16, 20):
        points[index] = _FakeLandmark(0.5, tip_y)
    return points


class _FakePoseLandmarker:
    def detect_for_video(self, mp_image, timestamp_ms):
        return SimpleNamespace(pose_landmarks=_fake_pose_landmarks())

    def close(self):
        pass


class _FakeHandLandmarker:
    def __init__(self, tip_y=0.2):
        self.tip_y = tip_y

    def detect_for_video(self, mp_image, timestamp_ms):
        return SimpleNamespace(hand_landmarks=[_hand_frame(self.tip_y)])

    def close(self):
        pass


def _fake_cv2():
    return SimpleNamespace(
        COLOR_RGB2BGR="rgb_to_bgr",
        COLOR_BGR2RGB="bgr_to_rgb",
        cvtColor=lambda frame, conversion: frame,
        VideoWriter=lambda *a, **kw: MagicMock(),
        VideoWriter_fourcc=lambda *args: 0,
        circle=lambda *args, **kwargs: None,
        line=lambda *args, **kwargs: None,
    )


def _fake_mp():
    return SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB="srgb"))


def test_process_live_frame_arm_accumulates_frame_and_metric():
    session = ma.LiveSession(observation_module="arm", selected_arm="right", landmarker=_FakePoseLandmarker())
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    with patch.multiple(ma, cv2=_fake_cv2(), mp=_fake_mp(), create=True):
        annotated, metric_value = ma.process_live_frame(session, frame, 1000)

    assert annotated.shape == frame.shape
    assert metric_value == 180.0
    assert len(session.frames_data) == 1
    assert session.timeseries == [(0.0, 180.0)]
    session.writer.write.assert_called_once()


def test_process_live_frame_palm_accumulates_hand_frame_and_metric():
    session = ma.LiveSession(observation_module="palm", selected_arm="right", landmarker=_FakeHandLandmarker(tip_y=0.2))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    with patch.multiple(ma, cv2=_fake_cv2(), mp=_fake_mp(), create=True):
        annotated, metric_value = ma.process_live_frame(session, frame, 1000)

    assert len(session.hand_frames) == 1
    assert metric_value is not None
    assert session.timeseries[0][1] == metric_value


def test_process_live_frame_tracks_elapsed_seconds_from_first_frame():
    session = ma.LiveSession(observation_module="arm", selected_arm="right", landmarker=_FakePoseLandmarker())
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    with patch.multiple(ma, cv2=_fake_cv2(), mp=_fake_mp(), create=True):
        ma.process_live_frame(session, frame, 1000)
        ma.process_live_frame(session, frame, 1500)

    assert [t for t, _ in session.timeseries] == [0.0, 0.5]


def test_finalize_live_session_computes_arm_metrics_and_closes_resources():
    session = ma.LiveSession(observation_module="arm", selected_arm="right", landmarker=MagicMock())
    session.writer = MagicMock()
    session.output_path = "/tmp/fake.mp4"
    session.frames_data = [
        {
            "_arm": "right",
            "nose": {"x": 0.5, "y": 0.2, "visibility": 1.0},
            "left_ear": {"x": 0.45, "y": 0.2, "visibility": 1.0},
            "right_ear": {"x": 0.55, "y": 0.2, "visibility": 1.0},
            "left_shoulder": {"x": 0.4, "y": 0.3, "visibility": 1.0},
            "right_shoulder": {"x": 0.6, "y": 0.3, "visibility": 1.0},
            "elbow": {"x": 0.6, "y": 0.5, "visibility": 1.0},
            "wrist": {"x": 0.6, "y": 0.7, "visibility": 1.0},
            "index": {"x": 0.61, "y": 0.71, "visibility": 1.0},
            "pinky": {"x": 0.62, "y": 0.72, "visibility": 1.0},
        }
    ]
    session.timeseries = [(0.0, 180.0)]

    path, metrics, timeseries = ma.finalize_live_session(session)

    assert path == "/tmp/fake.mp4"
    assert metrics["observation_module"] == "arm"
    assert timeseries == [(0.0, 180.0)]
    session.writer.release.assert_called_once()
    session.landmarker.close.assert_called_once()


def test_finalize_live_session_raises_when_nothing_detected():
    session = ma.LiveSession(observation_module="head", selected_arm="right", landmarker=MagicMock())
    with pytest.raises(ma.NoPoseDetectedError):
        ma.finalize_live_session(session)


def test_create_live_session_raises_when_runtime_unavailable():
    with patch.object(ma, "_load_analysis_runtime", return_value=False):
        with pytest.raises(ma.MovementAnalysisUnavailableError):
            ma.create_live_session("arm", "right")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.runtime-venv/bin/python -m pytest tests/test_live_session.py -v`
Expected: FAIL with `AttributeError: module 'movement_analysis' has no attribute 'LiveSession'`

- [ ] **Step 3: Implement**

Add to the imports at the top of `movement_analysis.py` (replace the current `import math` / `import pathlib` / dataclasses / typing block, lines 13-16):

```python
import math
import pathlib
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
```

Append at the end of `movement_analysis.py` (after `analyze_video`, currently ending at line 767):

```python
LIVE_STREAM_FPS = 4.0  # nominal fps for the annotated output video; matches the
                        # ~0.25-0.3s frame interval the live camera streams at


@dataclass
class LiveSession:
    """Accumulator for one live webcam session: the open MediaPipe landmarker,
    the annotated-output video writer (opened lazily on the first frame, once
    its dimensions are known), and everything needed to compute the same
    metrics `analyze_video` produces for a recorded clip."""

    observation_module: str
    selected_arm: Arm
    landmarker: Any = None
    writer: Any = None
    output_path: str = ""
    frames_data: list[dict] = field(default_factory=list)
    hand_frames: list[list[dict]] = field(default_factory=list)
    timeseries: list[tuple[float, float]] = field(default_factory=list)
    start_time: Optional[float] = None


def create_live_session(observation_module: str, selected_arm: Arm) -> LiveSession:
    """Open the module-appropriate MediaPipe landmarker for a live camera
    session. Raises MovementAnalysisUnavailableError under the same
    conditions `_require_cv`/`_require_hand_runtime` already do."""
    if observation_module == "palm":
        _require_hand_runtime()
        base_options = BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
    else:
        _require_cv()
        base_options = BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp_vision.PoseLandmarkerOptions(base_options=base_options, running_mode=mp_vision.RunningMode.VIDEO)
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    output_path = str(
        pathlib.Path(tempfile.gettempdir()) / f"reccontinue_live_{observation_module}_{uuid.uuid4().hex}.mp4"
    )
    return LiveSession(
        observation_module=observation_module, selected_arm=selected_arm,
        landmarker=landmarker, output_path=output_path,
    )


def _process_arm_or_head_frame(session: LiveSession, bgr_frame, mp_image, frame_timestamp_ms: int,
                                width: int, height: int) -> Optional[float]:
    result = session.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]

    if session.observation_module == "head":
        frame_data = {
            "nose": _landmark_point(landmarks, NOSE),
            "left_ear": _landmark_point(landmarks, LEFT_EAR),
            "right_ear": _landmark_point(landmarks, RIGHT_EAR),
            "left_shoulder": _landmark_point(landmarks, LEFT_SHOULDER),
            "right_shoulder": _landmark_point(landmarks, RIGHT_SHOULDER),
        }
        _draw_head_overlay(bgr_frame, frame_data, width, height)
        session.frames_data.append(frame_data)
        return frame_head_turn_proxy(frame_data)

    indices = arm_landmark_indices(session.selected_arm)
    frame_data = {
        "_arm": session.selected_arm,
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
    session.frames_data.append(frame_data)
    return frame_elbow_angle(frame_data, session.selected_arm)


def _process_palm_frame(session: LiveSession, bgr_frame, mp_image, frame_timestamp_ms: int,
                         width: int, height: int) -> Optional[float]:
    result = session.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    if not result.hand_landmarks:
        return None
    hand_landmarks = [{"x": lm.x, "y": lm.y, "visibility": 1.0} for lm in result.hand_landmarks[0]]
    if len(hand_landmarks) < 21:
        return None
    _draw_hand_overlay(bgr_frame, hand_landmarks, width, height)
    session.hand_frames.append(hand_landmarks)
    return palm_closure_ratio(hand_landmarks)


def process_live_frame(session: LiveSession, rgb_frame, frame_timestamp_ms: int):
    """Detect landmarks in one live-streamed frame, draw the module's
    restricted overlay, buffer the frame/landmarks/metric value on the
    session, and return (annotated_rgb_frame, current_metric_value).
    `current_metric_value` is None when nothing was detected this frame.
    """
    height, width = rgb_frame.shape[0], rgb_frame.shape[1]
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    if session.writer is None:
        session.writer = cv2.VideoWriter(
            session.output_path, cv2.VideoWriter_fourcc(*"mp4v"), LIVE_STREAM_FPS, (width, height)
        )
        session.start_time = frame_timestamp_ms / 1000.0

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    if session.observation_module == "palm":
        metric_value = _process_palm_frame(session, bgr_frame, mp_image, frame_timestamp_ms, width, height)
    else:
        metric_value = _process_arm_or_head_frame(session, bgr_frame, mp_image, frame_timestamp_ms, width, height)

    session.writer.write(bgr_frame)
    if metric_value is not None:
        elapsed_seconds = round(frame_timestamp_ms / 1000.0 - (session.start_time or 0.0), 2)
        session.timeseries.append((elapsed_seconds, metric_value))

    return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), metric_value


def finalize_live_session(session: LiveSession):
    """Close the session's writer/landmarker and compute the final metrics
    from its accumulated frames, via the same compute_*_metrics functions
    the recorded-clip pipeline uses. Returns (output_path, metrics, timeseries)."""
    if session.writer is not None:
        session.writer.release()
    session.landmarker.close()

    if session.observation_module == "palm":
        metrics = compute_palm_closure_metrics(session.hand_frames, LIVE_STREAM_FPS, session.selected_arm)
    elif session.observation_module == "head":
        metrics = compute_head_turn_metrics(session.frames_data, LIVE_STREAM_FPS)
    else:
        metrics = compute_session_metrics(session.frames_data, LIVE_STREAM_FPS, session.selected_arm)

    return session.output_path, metrics, session.timeseries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.runtime-venv/bin/python -m pytest tests/test_live_session.py tests/test_angles.py tests/test_repetition_counter.py tests/test_analyze_progress.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add movement_analysis.py tests/test_live_session.py
git commit -m "feat: add live-session MediaPipe frame processing"
```

---

### Task 3: Wire the live camera + time-track into app.py

**Files:**
- Modify: `app.py`
  - imports (lines 30-34)
  - remove `_progress_markdown` (lines 562-578) and `analyze_video_handler` (lines 581-604); add `_live_status_markdown`, `live_frame_handler`, `finish_session_handler` in their place
  - Tab "2 · MediaPipe analysis" body and its event bindings (lines 941-964)
- Modify: `tests/test_app_handlers.py` — remove the 3 tests that call the removed `analyze_video_handler` (`test_analyze_video_handler_yields_progress_then_result`, `test_analyze_video_handler_no_video_selected`, `test_analyze_video_handler_reports_unavailable_error`) and the now-unused `_fake_analyze_video` helper; add tests for `live_frame_handler`/`finish_session_handler`.
- Modify: `requirements.txt` — pin `pandas` explicitly since `app.py` now imports it directly (it was previously only a transitive Gradio dependency).

**Interfaces:**
- Consumes: `create_live_session`, `process_live_frame`, `finalize_live_session`, `LiveSession`, `MovementAnalysisUnavailableError`, `NoPoseDetectedError` from `movement_analysis.py` (Task 2).
- Produces: `live_frame_handler(rgb_frame, selected_arm, observation_module, session)` and `finish_session_handler(session)`, wired as Gradio callbacks.

- [ ] **Step 1: Write the failing tests**

Replace, in `tests/test_app_handlers.py`, the `_fake_analyze_video` helper and the three `test_analyze_video_handler_*` tests with:

```python
def test_live_frame_handler_creates_session_on_first_frame_and_returns_annotated_frame():
    frame = object()
    annotated = object()
    fake_session = object()

    with patch("app.create_live_session", return_value=fake_session) as create_mock, \
         patch("app.process_live_frame", return_value=(annotated, 96.0)) as process_mock:
        result_frame, result_session, status = app.live_frame_handler(frame, "right", "arm", None)

    create_mock.assert_called_once_with("arm", "right")
    process_mock.assert_called_once()
    assert result_frame is annotated
    assert result_session is fake_session
    assert "96.0" in status


def test_live_frame_handler_reuses_existing_session():
    frame = object()
    existing_session = object()
    annotated = object()

    with patch("app.create_live_session") as create_mock, \
         patch("app.process_live_frame", return_value=(annotated, None)) as process_mock:
        result_frame, result_session, status = app.live_frame_handler(frame, "right", "arm", existing_session)

    create_mock.assert_not_called()
    process_mock.assert_called_once()
    assert result_session is existing_session
    assert result_frame is annotated


def test_live_frame_handler_reports_unavailable_error_on_first_frame():
    with patch("app.create_live_session", side_effect=app.MovementAnalysisUnavailableError("mediapipe not installed")):
        result_frame, result_session, status = app.live_frame_handler(object(), "right", "arm", None)

    assert result_session is None
    assert "mediapipe not installed" in status


def test_live_frame_handler_no_frame_is_a_noop():
    result_frame, result_session, status = app.live_frame_handler(None, "right", "arm", "existing")
    assert result_session == "existing"


def test_finish_session_handler_no_session_warns():
    result = app.finish_session_handler(None)
    assert result[0] is None
    assert "before finishing" in result[4]


def test_finish_session_handler_returns_metrics_and_timeline():
    fake_session = object()
    metrics = {"observation_module": "arm", "repetition_count": 2}
    timeseries = [(0.0, 120.0), (0.5, 150.0)]

    with patch("app.finalize_live_session", return_value=("annotated.mp4", metrics, timeseries)):
        annotated_path, metrics_md, metrics_out, timeline_df, warning = app.finish_session_handler(fake_session)

    assert annotated_path == "annotated.mp4"
    assert metrics_out == metrics
    assert list(timeline_df["elapsed_seconds"]) == [0.0, 0.5]
    assert list(timeline_df["metric_value"]) == [120.0, 150.0]
    assert warning == ""


def test_finish_session_handler_reports_no_pose_detected():
    with patch("app.finalize_live_session", side_effect=app.NoPoseDetectedError("No pose was detected.")):
        result = app.finish_session_handler(object())

    assert result[0] is None
    assert "No pose was detected" in result[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.runtime-venv/bin/python -m pytest tests/test_app_handlers.py -v`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'live_frame_handler'` (and similar for `finish_session_handler`)

- [ ] **Step 3: Implement**

In `app.py`, replace the `movement_analysis` import block (lines 30-34):

```python
from movement_analysis import (
    MovementAnalysisUnavailableError,
    NoPoseDetectedError,
    create_live_session,
    finalize_live_session,
    process_live_frame,
)
```

Add near the top imports (after `import pathlib`, line 14):

```python
import time

import pandas as pd
```

Replace `_progress_markdown` (lines 562-578) and `analyze_video_handler` (lines 581-604) with:

```python
_LIVE_METRIC_LABELS = {"head": "head turn proxy", "palm": "palm opening ratio", "arm": "elbow angle"}


def _live_status_markdown(observation_module: str, metric_value) -> str:
    label = _LIVE_METRIC_LABELS.get(observation_module, _LIVE_METRIC_LABELS["arm"])
    value = f"{metric_value:.2f}" if metric_value is not None else "—"
    return f"🔴 Live observation running · {label} ≈ {value}"


def live_frame_handler(rgb_frame, selected_arm, observation_module, session):
    """Gradio `.stream()` callback: process one live webcam frame.

    Creates the module's `LiveSession` on the first frame (session is None),
    then runs MediaPipe on every subsequent frame and returns the annotated
    frame for immediate display alongside the updated session state.
    """
    if rgb_frame is None:
        return gr.update(), session, gr.update()

    if session is None:
        try:
            session = create_live_session(observation_module or "arm", selected_arm)
        except MovementAnalysisUnavailableError as exc:
            return rgb_frame, None, f"⚠️ {exc}"

    annotated_frame, metric_value = process_live_frame(session, rgb_frame, int(time.time() * 1000))
    return annotated_frame, session, _live_status_markdown(session.observation_module, metric_value)


def finish_session_handler(session):
    """Close out a live session and compute the same metrics shape the old
    recorded-clip pipeline produced, plus a time-track line chart."""
    if session is None:
        return (
            None, _metrics_markdown({}, is_synthetic=False), None, gr.update(),
            "⚠️ Start the live camera before finishing the observation.",
        )

    try:
        annotated_path, metrics, timeseries = finalize_live_session(session)
    except MovementAnalysisUnavailableError as exc:
        return None, f"⚠️ {exc}", None, gr.update(), ""
    except NoPoseDetectedError as exc:
        return (
            None,
            f"⚠️ {exc} Try again with better lighting and the tracked area fully in frame.",
            None,
            gr.update(),
            "",
        )

    timeline_df = pd.DataFrame(timeseries, columns=["elapsed_seconds", "metric_value"])
    return annotated_path, _metrics_markdown(metrics, is_synthetic=False), metrics, timeline_df, ""
```

In `build_app()`, replace the camera card body and its bindings (lines 941-964):

```python
            with gr.Tab("2 · MediaPipe analysis", id=1):
                with gr.Group(elem_classes=["kc-card", "kc-camera-card"]):
                    selected_module_md = gr.Markdown("_Choose an observation first._")
                    gr.Markdown(
                        '<p class="kc-card-title">Live camera observation</p>'
                        "Landmarks are tracked live on your camera feed; keep the selected area in "
                        "frame, then press **Finish observation** when done."
                    )
                    arm_radio = gr.Radio(["left", "right"], value=SYNTHETIC_PATIENT["selected_arm"], label="Arm to analyze")
                    live_camera = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Computer camera")
                    live_session_state = gr.State(value=None)
                    with gr.Row():
                        finish_btn = gr.Button("Finish observation", variant="primary")
                    analysis_progress_md = gr.Markdown("", elem_classes=["kc-analysis-progress"])
                with gr.Group(elem_classes=["kc-card"]):
                    annotated_output = gr.Video(label="MediaPipe annotated result", interactive=False)
                    metrics_display = gr.Markdown(_metrics_markdown({}, is_synthetic=False))
                    timeline_plot = gr.LinePlot(
                        x="elapsed_seconds", y="metric_value", label="Metric over time",
                        x_title="Elapsed seconds", y_title="Metric value",
                    )
                    gr.Markdown('<p class="kc-helper-text">2D camera observation only — not for diagnosis; requires clinician interpretation.</p>')
                with gr.Row(elem_classes=["kc-continue"]):
                    continue2_btn = gr.Button("Generate report →", variant="primary")
                continue2_warning = gr.Markdown("", elem_classes=["kc-hint-warning"])

                live_camera.stream(
                    fn=live_frame_handler,
                    inputs=[live_camera, arm_radio, selected_module_state, live_session_state],
                    outputs=[live_camera, live_session_state, analysis_progress_md],
                    stream_every=0.3,
                )
                finish_btn.click(
                    fn=finish_session_handler,
                    inputs=[live_session_state],
                    outputs=[annotated_output, metrics_display, metrics_state, timeline_plot, analysis_progress_md],
                )
                continue2_btn.click(fn=continue_from_record, inputs=metrics_state, outputs=[tabs, stepper_html, continue2_warning])
```

In `requirements.txt`, add after `numpy==1.26.4`:

```
pandas==2.2.3
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.runtime-venv/bin/python -m pytest tests/test_app_handlers.py -v`
Expected: the 6 new tests PASS. (The 2 pre-existing failures in this file that are unrelated Chinese-string localization issues — `test_recommend_module_from_words_prefills_only_an_existing_observation_view` and `test_empty_metrics_explains_the_next_step_without_an_error` — are untouched by this task and remain failing exactly as before, per Global Constraints.)

Then run the full suite to confirm nothing else broke:
Run: `.runtime-venv/bin/python -m pytest -q`
Expected: same 5 pre-existing failures as the plan's baseline (now `test_analyze_video_handler_no_video_selected` is gone, since that test was deleted in this task — 4 pre-existing failures remain: `test_recommend_module_from_words_prefills_only_an_existing_observation_view`, `test_empty_metrics_explains_the_next_step_without_an_error`, `test_recommend_observation_module_is_limited_to_the_three_existing_views`, `test_palm_closure_metrics_measure_fingertip_distance_change`), plus all new/kept tests passing.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_handlers.py requirements.txt
git commit -m "feat: replace record-then-analyze camera flow with live MediaPipe overlay + time-track"
```

---

### Task 4: Manual verification pass

**Files:** none (verification only)

- [ ] **Step 1: Launch the app locally**

Run: `.runtime-venv/bin/python app.py`

- [ ] **Step 2: Manually verify in a browser with a real webcam**

Not exercisable in a sandboxed/headless environment (same limitation README.md already documents for the pre-existing recorded-clip flow) — requires a human with a webcam:
1. Choose each of Head / Palm / Arm in Step 1.
2. In Step 2, confirm the live camera view shows only that module's restricted overlay points, moving with you in near-real-time.
3. Click **Finish observation**; confirm `annotated_output` plays a video, `metrics_display` shows the same metric shape the old flow showed, and `timeline_plot` shows a line chart with more than one point.
4. Continue to Step 3 (Generate report) and Step 4 (Export) and confirm both still work unchanged.

- [ ] **Step 3: Update docs**

Update the "Record with your computer camera" language in `SPEC.md`'s analysis-flow section and `README.md`'s camera-step description to describe the live-streaming flow instead of record-then-analyze. (No specific line numbers pinned here — read the current section headings for "MediaPipe" / "camera" in each file and edit the paragraph describing the record→analyze step.)
