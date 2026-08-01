# RecContinue

Every movement has context. Every record should stay private.

Rehabilitation shouldn't stop when a patient leaves the clinic. RecContinue
is an on-device, privacy-first companion that helps a patient continue an
activity their therapist already assigned, records what happened locally,
and brings that record back for the therapist to review — for the
"Build with Gemma NYC: On-Device AI for Healthcare" hackathon (Track 3 —
On-Device Private Health). The full product specification lives in
[`SPEC.md`](./SPEC.md) (see section 27 for the positioning refinement this
README reflects); this README covers what is currently built and how to
run it.

**RecContinue is not a diagnostic system, a treatment recommendation engine,
or a medical device.** It does not decide what a patient should do, and it
does not judge whether rehabilitation succeeded — the therapist assigns
the activity and reviews the outcome; Gemma 4 only helps organize what
happened in between, locally. It is a decision-support documentation tool
that requires clinician review.

## Implementation status

This build implements **Phases 1–5** of the six-phase plan in SPEC.md
section 24 — Gemma Spike, App Shell, Simple Movement Analysis,
End-to-End Integration, and Safety and Testing. Only **Phase 6
(Submission Support)** remains, and per an explicit decision in this
project, the Kaggle Writeup itself is drafted outside of Claude Code:

- ✅ **Phase 1 — Gemma Spike**: `gemma_client.py` talks to a local Ollama
  daemon, sends synthetic patient/context/metrics data, parses and
  validates the JSON response against the `schemas.RecContinueReport`
  Pydantic model, and attempts one JSON-repair prompt if the first
  response doesn't validate. Runnable standalone: `python gemma_client.py`.
- ✅ **Phase 2 — App Shell**: `app.py` is a branded Gradio Blocks app with a
  focused patient flow: choose one of four synthetic clinician-provided
  modules, record with the computer camera, generate a local Gemma report,
  then export a minimal packet. Clinician review is a separate tab.
- ✅ **Phase 3 — Simple Movement Analysis**: `movement_analysis.py` runs
  MediaPipe Pose Landmarker + OpenCV locally against an uploaded video,
  reading only the upper-body landmark subset in SPEC.md section 6 (head,
  shoulders, selected-arm elbow/wrist, approximate palm), draws the
  matching overlay, and computes all five metrics from section 7
  (repetition count via a smoothed elbow-angle state machine — flexed
  below 115°, extended above 145°, 5-frame moving average, 3-frame
  confirm — session duration, estimated 2D elbow angle at peak reach, max
  hand height relative to shoulder, head-position variation, and
  landmark confidence). Wired into
  Tab 2's **Analyze** button. Verified end-to-end against a real photo of
  a person (see Known limitations — no real *video* of arm movement was
  available in this environment, so repetition counting itself is only
  unit-tested, not verified against real footage).
- ✅ **Phase 4 — End-to-End Integration**: `storage.py` saves each
  successfully generated report to a local JSON vault (`vault/`,
  gitignored) — patient, metrics, patient context, Gemma report, and
  review status. `packet_export.py` builds a patient-controlled,
  minimized clinician packet (`build_clinician_packet`) honoring the
  Tab 5 inclusion checkboxes (raw video and representative frame default
  **off**), exports it as `<patient_id>-session.reccontinue.json`, and
  Tab 6 imports it, displays metrics/report/patient-context, and exports
  a reviewed copy with the clinician's note and reviewed flag. All six
  tabs are now fully wired to real local logic — see the file-by-file
  breakdown above.
- ✅ **Phase 5 — Safety and Testing**: `safety.py` deterministically scans
  every Gemma report's clinical-content sections (never the safety
  disclaimer itself) for the unsupported phrases in SPEC.md section 18
  ("the patient has", "diagnosed with", "Brunnstrom stage", "recommended
  exercise/treatment", "unsafe", "normal/abnormal movement", etc.),
  case-insensitively, and reports the *exact offending sentence* without
  rewriting or deleting the original output. Tab 4 shows the result right
  after generation. If flagged, the session's review status becomes
  "Requires manual editing," that flag travels into the exported
  clinician packet, and `packet_export.export_reviewed_report()` raises
  `SafetyGateError` — surfaced as a clear in-app message — if a clinician
  tries to mark a still-flagged report reviewed (a clinician note can
  still be saved without marking it reviewed). This closes out P0 per
  SPEC.md's acceptance criteria (section 21).
- ✅ **Post-P0 addition — Analyze progress + local voice capability**:
  `movement_analysis.analyze_video()` now yields live progress (frame
  count, current elbow angle, running rep count) while it runs, shown on
  Tab 2 instead of a blank wait. A local, optional voice capability remains
  available in the codebase (`stt_client.py`), but the patient UI now puts
  its focus on module selection, camera tracking, and the report. The
  acknowledgment path is scanned by the same `safety.py` flagged-phrase
  checker used on the full PEO report — see
  `docs/superpowers/specs/2026-08-01-analyze-progress-and-voice-frontdoor-design.md`.
- ⬜ **Phase 6 — Submission Support**: not started (see note above).

## Why this matters

In Taiwan alone, the population needing rehabilitation services exceeds
750,000, while physical/occupational therapists can meet only 40–50% of
that market demand — a gap driven by therapist shortages, long travel
times, and the cost of frequent in-person follow-up, especially in
underserved areas (source: companion vision deck, "AI職能治療智慧輔助系統
/ AI-Driven Remote Rehabilitation Ecosystem," citing Taiwan long-term-care
and physical-therapy staffing news coverage). The time between clinic
visits becomes a documentation blind spot precisely for the patients who
can least afford it. RecContinue targets that gap: not by replacing the
therapist, but by giving patients a private way to keep a therapist's
assigned activity documented between visits.

## Why on-device, why Gemma

Rehabilitation sessions involve video of a patient moving inside their
home, body-movement measurements, and descriptions of daily-life
difficulty — all sensitive. RecContinue keeps capture, analysis, and report
generation entirely local: no cloud AI, no analytics, no automatic
transmission. Gemma 4, run locally through Ollama, is the component that
turns local camera-based measurements and patient-reported context into
organized Person–Environment–Occupation documentation — distinguishing
objective measurement from patient report, flagging missing information,
and drafting clinician follow-up questions, without diagnosing or
recommending treatment. See SPEC.md sections 4, 11, and 13 for the full
rationale and the exact system prompt.

Gemma also powers the optional text onboarding field on Tab 1. A patient can
describe what they want to do and what is difficult in their own words; Gemma
turns that into editable Person-Environment-Occupation context and identifies
missing details. It never changes the activity already assigned by the
therapist, and the generated context is visible for patient editing before it
can appear in a clinician packet.

## Project structure

```text
reccontinue/
├── app.py                 # Gradio Blocks app shell (Phase 2)
├── movement_analysis.py    # MediaPipe/OpenCV movement analysis (Phase 3)
├── gemma_client.py         # Local Ollama/Gemma 4 client (Phase 1)
├── prompts.py               # System prompt + prompt builders (Phase 1)
├── schemas.py               # Pydantic report schema (Phase 1)
├── storage.py                # Local JSON session vault (Phase 4)
├── packet_export.py           # Clinician packet build/export/import (Phase 4)
├── safety.py                   # Deterministic post-generation safety check (Phase 5)
├── sample_data/                 # Synthetic patient + synthetic metrics fixtures
├── assets/logo.svg                # Brand logo
├── models/                         # Pose Landmarker model (gitignored; see models/README.md)
├── vault/, exports/                 # Created at runtime, gitignored — local session data
├── tests/                             # pytest suite
├── requirements.txt
└── SPEC.md                             # Full product specification
```

## Setup

Python 3.11 is specified in SPEC.md; it was not available in this dev
environment, so this build was created and tested on Python 3.12 instead
(noted under Known limitations below).

```bash
cd reccontinue
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use live Gemma generation, install and start Ollama separately:

```bash
ollama pull gemma4:e2b
ollama serve
```

To use real camera-based movement analysis (Tab 2's **Analyze** button),
download the MediaPipe Pose Landmarker model — see
[`models/README.md`](./models/README.md) for the exact URL and where to
place it. Without the model file, Tab 2 shows a clear setup message
instead of crashing, and the synthetic metrics fallback still works.

Then run the app:

```bash
python app.py
```

It binds only to `127.0.0.1:7860` (`share=False`), with
`GRADIO_ANALYTICS_ENABLED=False` set before Gradio is imported.

**The app starts and is fully usable even if Ollama is not running** — Tab
1 and Tab 4 show an exact remediation message (`ollama serve`, then
`ollama pull gemma4:e2b`) instead of crashing.

## Demo workflow (current build)

1. Open the app and choose exactly one of four synthetic, clinician-provided
   rehabilitation modules. Click **Apply selected module**.
2. On **Track movement**, choose the arm and record with the computer
   camera. Stopping the recording starts local MediaPipe analysis. A small
   synthetic-measurements shortcut is available for judging without a camera.
3. On **Gemma report**, generate a validated PEO documentation draft from
   the selected module and the local measurements. The report includes
   objective observations, missing information, and neutral questions for
   clinician discussion; it does not give treatment advice.
4. On **Export**, the default packet contains the measurement summary and
   report, while raw video stays excluded. The separate **Clinician review**
   tab imports an exported packet for annotation and review.

## Tests

```bash
source .venv/bin/activate
python -m pytest -v
```

77 tests, all passing, covering:

- `tests/test_report_schema.py` — Pydantic schema validation (valid
  report, missing required field, wrong field type, default
  `report_status`, and that `clinical_text_sections()` excludes the
  disclaimer/status fields).
- `tests/test_gemma_client.py` — JSON extraction from fenced/bare model
  output; Ollama connection-status detection (unreachable, model missing,
  ok); successful report generation; one-shot JSON-repair recovery;
  failure after repair also fails; timeout handling; connection-failure
  handling. All Ollama calls are mocked — no live Ollama instance is
  required to run the suite.
- `tests/test_angles.py` — `angle_2d()` against known geometric cases
  (180°, 90°, 60°, 0°, a degenerate zero-length ray, and a plausible
  elbow-bend example).
- `tests/test_repetition_counter.py` — `RepetitionCounter`'s smoothed
  elbow-angle state machine (no movement, one clean rep, three reps, a
  dead-zone reading between the flexed/extended thresholds, a brief
  noise spike that must *not* count under the default 5-frame/3-confirm
  smoothing, and a rise that never returns to flexed so must not count
  yet); plus `compute_session_metrics()` on
  synthetic per-frame landmark data (basic metric shape, raises on no
  frames, falls back to wrist-based measurement when hand points are
  low-confidence, flags low overall landmark confidence).
- `tests/test_packet_export.py` — clinician packet building (default
  excludes raw video/frame; only includes raw video when explicitly opted
  in), export/import round trip, corrupt/missing/malformed packet
  handling, and reviewed-report export (with and without the reviewed
  flag set).
- `tests/test_safety.py` — a clean report passes; each representative
  flagged phrase (diagnostic claims, Brunnstrom stage, recommended
  exercise, "unsafe"/"normal movement", case-insensitivity) is caught;
  the flagged sentence is isolated from its surrounding paragraph;
  `report_status` and `safety_notice` are never scanned; the validator
  never mutates the report it's checking.

None of the above require cv2/mediapipe/Ollama to be running — they test
the pure logic with mocked or synthetic inputs. `movement_analysis.py`'s
video pipeline and the full save → export → import → review flow were
additionally exercised manually end-to-end (see Known limitations).

## Known limitations

- Built and tested on Python 3.12, not the Python 3.11 specified in
  SPEC.md (3.11 was not available on this machine). No 3.12-specific
  behavior was relied upon.
- `requirements.txt` pins the exact versions installed and tested here
  (`gradio`, `pydantic`, `requests`, `pytest`, `opencv-python`,
  `mediapipe`, `numpy`), which are newer than the versions originally
  listed in SPEC.md — those older pins were not resolvable for Python
  3.12 in this environment.
- No live end-to-end Gemma generation has been run in this environment
  (Ollama is not installed here); the Ollama round trip is covered by
  mocked unit tests plus a verified graceful-failure path in the UI and
  in `gemma_client.py`'s own `__main__` spike.
- `movement_analysis.analyze_video()` was verified end-to-end (real
  MediaPipe detection, correct overlay, correct metric computation) using
  a short video built from a real public-domain photo of a person, since
  no webcam or real recorded rehab video was available in this sandboxed
  environment. That validates the detection/overlay/metrics pipeline, but
  not repetition counting against genuine repeated elbow flexion/extension
  — that logic is instead covered by `RepetitionCounter` unit tests with
  synthetic elbow-angle sequences.
- The save → export → import → review flow (Phase 4) was verified by
  calling `app.py`'s handler functions directly with a fabricated report
  (no live Ollama available here), plus a full pass through the actual
  Tab 5 UI in-browser. The one path not exercised end-to-end through the
  browser is Tab 6's file-picker import, since this sandboxed browser
  automation can't drive a native OS file dialog; `import_packet()`
  itself is covered by both direct calls and unit tests.
- The safety gate (Phase 5) was verified with a deliberately flagged
  report: it correctly blocked `export_reviewed_report()` from marking
  the session reviewed, and correctly allowed a draft clinician note to
  be saved without the reviewed flag. Not verified: what a *real* Gemma
  model actually tends to write (the flagged-phrase list is exercised
  against hand-written test fixtures, not live model output, since no
  Ollama instance was available here).
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
- Phase 6 (Submission Support) has not been started — per an explicit
  project decision, the Kaggle Writeup and other presentation/pitch
  materials are handled outside of Claude Code, not by this build.
