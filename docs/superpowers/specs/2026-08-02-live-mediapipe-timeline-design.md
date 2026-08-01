# Live MediaPipe overlay + in-session time-track

Status: approved for planning
Date: 2026-08-02

## 1. Problem

Today (Step 2 · MediaPipe 分析 in `app.py`) a patient records a full webcam
clip with `gr.Video(sources=["webcam"])`, and only after recording stops does
`analyze_video()` run MediaPipe over the saved file, producing an annotated
video and a metrics dict once analysis finishes. The patient gets no
feedback about landmark tracking while they are actually moving, and there is
no view of how a metric (elbow angle, palm opening ratio, head turn proxy)
changed moment-to-moment across the session — only session-level min/max/
change numbers.

## 2. Goals

- Show MediaPipe landmark points overlaid on the camera feed live, while the
  patient is moving, not only after the fact.
- Show a time-track (line chart) of the session's primary metric across the
  recording, once the session finishes.
- Preserve everything downstream of "we have a metrics dict + an annotated
  video path" untouched: Gemma report generation, the safety check, session
  storage, and clinician packet export.

## 3. Non-goals

- No client-side (in-browser/WASM) MediaPipe. Server-side processing only,
  to reuse the existing Python landmark/metric code without a parallel JS
  implementation.
- No true zero-latency AR — a small (network + inference) round-trip delay
  per frame is acceptable.
- No change to which landmarks are read or drawn per observation module
  (head/palm/arm) — SPEC.md section 6's restricted point sets stay as-is.
- No persistence of multi-session trend data (that was explicitly descoped
  in the clarifying questions — this spec is single-session only).

## 4. Architecture

Replace the "record clip, then analyze" flow with a live-streaming flow:

- `gr.Video(sources=["webcam"])` in `app.py` is replaced with
  `gr.Image(sources=["webcam"], streaming=True)`, driven by Gradio's
  `.stream()` event (fires roughly every `stream_every` seconds — target
  0.25–0.3s).
- A `gr.State` holds one `LiveSession` object per browser session:
  the open MediaPipe landmarker (Pose or Hands, chosen from
  `selected_module_state`), an open `cv2.VideoWriter` writing the annotated
  frames out as they're produced, the growing `frames_data` list (same shape
  `compute_session_metrics` already consumes), and a
  `(elapsed_seconds, metric_value)` time series list.
- Each streamed frame: run the module-appropriate landmarker on it, draw the
  same restricted overlay as today, write the frame to the `VideoWriter`,
  append to `frames_data` and to the time series, and return the annotated
  frame back to the same `gr.Image` component so the patient sees it
  immediately.
- A new "完成觀察，產生結果" button ends the session: it closes the writer
  and landmarker, then calls the **existing, unmodified**
  `compute_session_metrics` / `compute_palm_closure_metrics` /
  `compute_head_turn_metrics` on the accumulated `frames_data` — so the
  metrics dict shape is unchanged and every downstream consumer (report
  generation, safety check, packet export) needs no changes. The time series
  is rendered as a `gr.LinePlot`.

## 5. Components

### `movement_analysis.py` (new functions, alongside the existing ones)

```
@dataclass
class LiveSession:
    observation_module: str        # "頭部" | "手掌" | "手臂"
    selected_arm: Arm
    landmarker: Any                 # PoseLandmarker or HandLandmarker
    writer: Any                     # cv2.VideoWriter, opened on first frame
    output_path: str
    frames_data: list[dict] = field(default_factory=list)
    hand_frames: list[list[dict]] = field(default_factory=list)  # 手掌 only
    timeseries: list[tuple[float, float]] = field(default_factory=list)
    start_time: float | None = None  # wall-clock seconds, set on first frame
    width: int | None = None
    height: int | None = None

def create_live_session(observation_module: str, selected_arm: Arm) -> LiveSession: ...

def process_live_frame(
    session: LiveSession, rgb_frame: np.ndarray, elapsed_ms: int
) -> tuple[np.ndarray, Optional[float]]:
    """Detect landmarks, draw the module's overlay, buffer the frame and its
    metric value, and return (annotated_rgb_frame, current_metric_value)."""

def finalize_live_session(session: LiveSession) -> tuple[str, dict, list[tuple[float, float]]]:
    """Close the writer/landmarker and compute final metrics from
    session.frames_data via the existing compute_*_metrics functions.
    Returns (annotated_video_path, metrics, timeseries)."""
```

`process_live_frame` dispatches on `session.observation_module` the same way
`analyze_video` does today: 手掌 → `HandLandmarker` + `_draw_hand_overlay` +
the existing per-frame `palm_closure_ratio` for the live metric value;
手臂 → `PoseLandmarker` + `_draw_overlay` + the existing per-frame
`frame_elbow_angle` for the live metric value.

頭部 has no existing per-frame helper — `compute_head_turn_metrics` computes
the head-turn proxy only inline, across the whole session. A new
`frame_head_turn_proxy(frame_data) -> Optional[float]` needs to be added
(mirrors `frame_elbow_angle`'s shape: given one frame's `nose`/`left_ear`/
`right_ear` landmarks, return `(nose.x - ear_mid_x) / ear_width` or `None`
if an ear is missing), used both for the live per-frame value and reusable
by `compute_head_turn_metrics` if useful, though that function stays
unmodified per section 4.

The three `_draw_*_overlay` functions are reused as-is. They currently
assume a BGR frame (drawn with `cv2`, `writer.write()` expects BGR); Gradio
hands us RGB and expects RGB back. `process_live_frame` does the BGR↔RGB
conversion at its boundary (convert once in, draw, convert once out) so the
overlay functions themselves don't change.

`VideoWriter` needs a fixed fps up front, but streaming doesn't have one.
Open it lazily on the first frame using a nominal fps derived from
`1 / stream_every` (i.e. the configured Gradio stream interval), since that
is the actual frame arrival rate we control.

### `app.py`

- Replace `video_input = gr.Video(...)` with:
  `live_camera = gr.Image(sources=["webcam"], streaming=True, label="Computer camera")`
- Add `live_session_state = gr.State(value=None)`
- Add `finish_btn = gr.Button("完成觀察，產生結果", variant="primary")`
- Add `timeline_plot = gr.LinePlot(x="elapsed_seconds", y="metric_value", label="指標隨時間變化")` under `metrics_display`
- Remove the `analyze_btn` / `video_input.stop_recording` bindings that
  called `analyze_video_handler`; replace with:
  - `live_camera.stream(fn=live_frame_handler, inputs=[live_camera, arm_radio, selected_module_state, live_session_state], outputs=[live_camera, live_session_state], stream_every=0.3)`
  - `finish_btn.click(fn=finish_session_handler, inputs=[live_session_state, arm_radio, selected_module_state], outputs=[annotated_output, metrics_display, metrics_state, timeline_plot, analysis_progress_md])`
- `live_frame_handler` creates a `LiveSession` on first call (when state is
  `None`) via `create_live_session`, otherwise calls `process_live_frame`;
  returns the annotated frame and the updated state.
- `finish_session_handler` calls `finalize_live_session`, builds the
  `metrics_markdown` (existing `_metrics_markdown`, unchanged), and builds a
  pandas DataFrame from the timeseries for `gr.LinePlot`.

## 6. Data flow

```
browser webcam
   -> Gradio streams an RGB frame every ~0.3s
   -> live_frame_handler(frame, arm, module, state)
        -> state is None? create_live_session(module, arm)
        -> process_live_frame(state, frame, elapsed_ms)
             -> run module's MediaPipe landmarker
             -> draw restricted overlay for module
             -> append frame's landmarks to state.frames_data
             -> append (elapsed_s, metric_value) to state.timeseries
             -> write annotated frame to state.writer
        -> return annotated frame (shown immediately in gr.Image)
   ... repeats until patient clicks 完成觀察 ...
finish_btn.click
   -> finish_session_handler(state, arm, module)
        -> finalize_live_session(state)
             -> close writer, close landmarker
             -> compute_session_metrics / compute_palm_closure_metrics /
                compute_head_turn_metrics(state.frames_data, ...)   [unchanged]
             -> return (video_path, metrics, timeseries)
        -> render annotated_output, metrics_display, metrics_state, timeline_plot
```

Everything after `finish_session_handler` returns — Step 3 (Gemma report)
and Step 4 (export) — is unaffected: they already only consume
`metrics_state` and `annotated_output`'s path.

## 7. Error handling

- **No opencv/mediapipe available**: `create_live_session` raises the
  existing `MovementAnalysisUnavailableError`; `live_frame_handler` catches
  it on the first frame and surfaces the existing error message instead of
  starting the stream.
- **No landmarks detected in a frame**: same as today — that frame is
  skipped for overlay/data purposes (no point drawn, nothing appended to
  `frames_data`/`timeseries`), but the raw frame is still written to the
  output video and streaming continues. Mirrors current per-frame behavior
  in `analyze_video`.
- **No landmarks detected in the entire session**: `finalize_live_session`
  surfaces the existing `NoPoseDetectedError` from `compute_session_metrics`
  et al.; `finish_session_handler` catches it and shows the same "try better
  lighting / get the arm in frame" message `analyze_video_handler` shows
  today.
- **Patient closes the tab / navigates away mid-session without clicking
  完成觀察**: the `LiveSession`'s writer/landmarker are simply never closed
  or finalized for that browser session. This matches the current
  no-persistence-until-explicit-action model (nothing was being saved
  mid-recording before either) — no new data-loss risk is introduced. Not
  handling orphaned-resource cleanup beyond this is an accepted scope
  limitation for the hackathon timeline.
- **Repetition counting reliability**: flagged as a known trade-off (section
  8), not treated as an error — the count is still shown, just with a wider
  margin of error than the old 30fps analysis.

## 8. Trade-offs (carried over from the approved design discussion)

1. **Lower live sampling rate** (~3–4 samples/sec vs. the recorded clip's
   ~30fps previously). `RepetitionCounter`'s `smoothing_window=5` /
   `confirm_frames=3` defaults were tuned for 30fps data; at 3–4fps that's
   1.5–2.5 seconds of latency before a rep confirms, which may undercount
   fast repetitions. The plan should include re-tuning or documenting this
   explicitly rather than silently shipping less accurate rep counts.
2. **Per-frame round-trip latency**: what the patient sees lags their actual
   movement by roughly one `stream_every` interval plus inference/network
   time. Acceptable for slow rehab movements; should be mentioned in the
   in-app helper text if it becomes noticeable.

## 9. Testing

- **Unit tests** (extend `tests/test_analyze_progress.py` /
  `tests/test_angles.py` style): feed `process_live_frame` synthetic
  landmark-bearing frames (same fixture approach the existing
  `compute_session_metrics`/`compute_palm_closure_metrics` tests likely
  already use) and assert `frames_data`/`timeseries` accumulate correctly,
  and that `finalize_live_session` produces a metrics dict with the same
  keys `compute_session_metrics` produces today.
- **Handler tests** (extend `tests/test_app_handlers.py`'s existing
  patch-and-yield pattern): patch `movement_analysis.create_live_session` /
  `process_live_frame` / `finalize_live_session` the same way
  `_fake_analyze_video` patches `analyze_video` today, and assert
  `live_frame_handler`/`finish_session_handler` wire outputs to the right
  Gradio components.
- **Manual verification**: requires a real webcam, so — same limitation
  README.md already notes for the existing analysis flow — this can't be
  exercised in a sandboxed/headless environment. Manual test plan: run the
  app locally, pick each of 頭部/手掌/手臂, confirm the live overlay tracks
  the right points only, click 完成觀察, confirm the annotated video, metrics
  markdown, and time-track line chart all populate, then confirm Step 3
  (Gemma report) and Step 4 (export) still work unchanged.

## 10. Docs to update alongside implementation

- `SPEC.md` section on the analysis flow (currently describes record-then-
  analyze) and `README.md`'s description of the camera step.
- `models/README.md` if the live session's lazy `VideoWriter` fps choice is
  worth documenting for future contributors.
