# Design: Analyze Progress Display + Voice Front-Door

Date: 2026-08-01
Status: Approved for implementation (pre-hackathon-submission scope)

## Problem

Detection feels slow because both blocking steps in the current flow give
zero feedback while they run:

- Tab 2 **Analyze**: `movement_analysis.analyze_video()` reads the whole
  uploaded/recorded clip frame-by-frame with MediaPipe, writes an
  annotated overlay video, and only returns after every frame is
  processed. The UI shows nothing until that full pass completes.
- Tab 4 **Generate with Gemma**: a single blocking Ollama call with no
  streaming.

Original ask was three ideas: (1) a live ROM gauge, (2) opening the flow
by asking "what are we rehabbing today," (3) a voice front-door that
routes to a Gemma-recommended detection/exercise. Scoping discussion
(see conversation) resolved this to two buildable units, both scoped to
land before hackathon submission:

- The synthetic demo has exactly **one** patient and **one** assigned
  task (`sample_data/synthetic_patient.json`). Adding real multi-task
  routing was explicitly rejected as out of scope for this pass — the
  task stays hardcoded.
- A live-during-recording gauge (true webcam frame streaming) was
  rejected in favor of a lower-risk progress display during the existing
  post-recording Analyze step. Same recording UX, just no more blank
  wait.
- Ideas (2) and (3) collapse into one interaction once there's only one
  task: a Tab 1 voice front-door that asks how the patient is feeling,
  transcribes locally, has Gemma acknowledge and reconfirm the one
  assigned task (never inventing a new one), and carries the transcript
  into Tab 3 so the patient isn't re-typing what they just said.
- Voice is input-only. Gemma's reply is text, not synthesized speech —
  avoids a second live-demo-fragile audio pipeline.
- Speech-to-text must be local (`faster-whisper`), not the browser's
  built-in `SpeechRecognition`, since that ships audio to a cloud
  service and would directly contradict this product's Track 3
  on-device/privacy positioning.

## Feature A — Analyze progress display

**Goal:** replace the blank wait during Tab 2's Analyze step with a
live-updating readout, without changing the record/upload UX or the
final output contract.

**Changes:**

- `movement_analysis.analyze_video()` becomes a generator. While
  looping over frames, it yields a progress dict every ~5 frames (or
  ~200ms, whichever is coarser, to avoid flooding the UI):
  ```python
  {"frame_index": i, "total_frames": total, "current_angle": float | None, "reps_so_far": int}
  ```
  `current_angle` is `None` on frames where the pose/arm landmarks
  weren't confidently detected. After the loop, it performs the same
  final `compute_session_metrics` call as today and yields the existing
  `(annotated_path, metrics)` tuple as its **last** yield. Nothing about
  the returned final shape changes — only intermediate yields are new.
- `app.py`'s `analyze_video_handler` becomes a generator that consumes
  `analyze_video()`'s yields. For every intermediate dict, it yields an
  update to a new `analysis_progress_md` Markdown component (e.g.
  "Processing frame 42/120 · elbow angle ≈ 96° · reps so far: 1") and
  leaves `annotated_output`/`metrics_display` untouched. On the final
  yield (the existing tuple), it yields the same three outputs it
  returns today, plus clearing `analysis_progress_md`. This relies on
  Gradio's existing support for generator-valued event handlers
  (confirmed on the installed `gradio==6.22.0`) — no new component types,
  no change to `sources=["webcam","upload"]` or the `stop_recording`
  wiring.
- `analysis_progress_md` is added to Tab 2 directly under the
  Analyze/fallback button row, empty by default.
- Existing exceptions (`MovementAnalysisUnavailableError`,
  `NoPoseDetectedError`, `ValueError`) are caught exactly as today; the
  handler yields the same error-state tuple it currently returns,
  clearing the progress line.

**Testing:**

- `analyze_video()` on a short synthetic test clip: assert at least one
  intermediate yield occurs before the final tuple, and that the final
  tuple's shape/values match what the current non-generator version
  produces on the same input (regression check).
- No changes to `RepetitionCounter` or `compute_session_metrics` tests —
  the underlying per-frame math is untouched, only when it's surfaced.

## Feature B — Tab 1 voice front-door

**Goal:** let the patient answer "how are you feeling about today's
session" by voice, get a same-task acknowledgment from Gemma, and have
that carry into Tab 3 without retyping — all optional and gracefully
degrading, matching this project's existing pattern for
MediaPipe/Ollama unavailability.

**New module `stt_client.py`** (parallel structure to `gemma_client.py`):

- Wraps `faster-whisper`, lazily loading a small CPU/int8 model (`base`
  or `tiny` — final choice made during implementation based on
  transcription latency on the dev machine) on first use.
- `transcribe_audio(path: str) -> str`, raising a dedicated
  `WhisperUnavailableError` (mirroring
  `movement_analysis.MovementAnalysisUnavailableError`) when the
  dependency or model isn't available, so the caller can show a setup
  message instead of crashing.
- No network calls — local inference only, consistent with the rest of
  the product's on-device claims.

**`prompts.py` addition:**

- `ACKNOWLEDGMENT_SYSTEM_PROMPT`: short variant of the existing system
  prompt's constraints (no diagnosis, no treatment/exercise
  recommendation, no severity judgment) scoped to a one-paragraph
  spoken-style reply.
- `build_acknowledgment_prompt(transcript: str, assigned_task: str) ->
  str`: asks Gemma to briefly acknowledge what the patient said and
  reconfirm the one already-assigned task by name — explicitly
  forbidden from proposing a different exercise or judging whether the
  patient should do it.

**`gemma_client.py` addition:**

- `generate_acknowledgment(transcript, assigned_task, host=..., model=...) -> str`:
  calls Ollama with `"format"` omitted (plain text, not the structured
  JSON report path) using `ACKNOWLEDGMENT_SYSTEM_PROMPT`. Reuses the
  existing `_call_ollama_generate` helper and the same
  `OllamaUnavailableError`/`GemmaTimeoutError`/`GemmaModelNotInstalledError`
  exceptions callers already handle elsewhere.

**Safety:**

- Before display, the acknowledgment text is run through the existing
  `safety.py` scanner (the same flagged-phrase check already used on
  the PEO report). If flagged, the acknowledgment is discarded and only
  the transcript + the static task instructions are shown — never
  surface a flagged sentence in the live demo.

**`app.py` / Tab 1 wiring:**

- New `gr.Audio(sources=["microphone"], label="How are you feeling about today's session? (optional)")`.
- On stop-recording, an auto-triggered handler: transcribe → build +
  call acknowledgment → safety-scan → render transcript and (if not
  flagged) Gemma's acknowledgment text.
- The transcript is also written into the same state/value that backs
  Tab 3's `patient_statement_tb`, so when the patient reaches Tab 3 it's
  pre-filled and still editable (one-way pre-fill, not a live sync —
  editing it on Tab 3 doesn't write back to Tab 1).
- This entire interaction is optional: the "Continue to Step 2: Record →"
  button is not gated on it. If `faster-whisper`/the model isn't
  installed, or the patient records nothing, Tab 1 shows a clear setup
  message or simply stays empty, and the rest of the flow is unaffected.

**Testing (all mocked, no live mic/Ollama needed, matching existing
test philosophy):**

- `stt_client.transcribe_audio()`: returns expected text with a mocked
  `faster-whisper` model; raises `WhisperUnavailableError` when the
  dependency/model is missing.
- `prompts.build_acknowledgment_prompt()`: produces the expected
  prompt shape/content for a given transcript + task.
- A deliberately flagged acknowledgment string is caught by the
  existing `safety.py` scanner and suppressed (reusing existing safety
  test fixtures/approach).

## New dependency

- `faster-whisper`, added to `requirements.txt`. Model file downloaded
  once and cached locally (same posture as the gitignored MediaPipe
  Pose Landmarker model — see `models/README.md`).

## Out of scope for this pass

- Real multi-task assignment/routing (still one hardcoded task).
- Spoken (TTS) Gemma replies.
- True live webcam-streaming ROM display during recording itself.
- Any change to Tab 4's Gemma report generation call or its own
  (separate, unchanged) blocking-wait UX.
