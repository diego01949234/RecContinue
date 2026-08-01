# Multi-Exercise Movement Profiles — Design

Date: 2026-08-01
Status: Approved by product owner, ready for implementation planning.

## Background

`movement_analysis.py` currently supports exactly one rehabilitation task
(reach-and-place cup, elbow flexion/extension) with the counting logic,
landmark set, and metric labels hardcoded around a single shoulder-elbow-wrist
angle. This came out of `SPEC.md` section 26 ("Do not add: ... Multiple
rehabilitation tasks") and section 27 ("A hand-tracking pivot ... was
considered and explicitly rejected"), both written to keep the one-day
hackathon build's live demo reliable.

This design reverses both of those restrictions, based on an explicit product
decision made in conversation on 2026-08-01:

- The hackathon has not yet been submitted — this is the version being
  submitted, so the "multiple tasks" restriction is being lifted deliberately,
  not accidentally reopened.
- The privacy positioning ("local inference, local storage, no cloud") is
  unaffected by tracking a few more landmarks locally — privacy was never
  about landmark count, it was about data never leaving the device.
- The hand-tracking rejection was really two decisions bundled together: (a)
  no full finger-joint/OT-goniometry-level hand skeleton, and (b) no multiple
  tasks. This design keeps (a) almost entirely intact — it adds exactly one
  new landmark (thumb tip) and one new derived metric (thumb-to-index
  distance), not a finger skeleton — while deliberately reversing (b).
- Exercise-type selection stays a patient/therapist choice ("today's assigned
  activity is X"), consistent with section 27's existing reframe that the
  therapist assigns the activity and the app only records what happened. The
  app does not infer, diagnose, or recommend which exercise a patient needs.

## Goals

- Support 5 exercise types, each with its own landmark-derived rep-counting
  metric: elbow flexion (existing), shoulder raise, neck rotation, wrist
  flexion, grip/pinch.
- Let the patient select the exercise type on Tab 1, alongside the existing
  arm selector.
- Keep each exercise type's logic independently testable, so a bug in one
  profile cannot break the others before the live demo.
- Preserve every existing safety/wording constraint from SPEC.md sections 7,
  8, and 18 (no diagnostic language, camera-based/estimated framing, etc.) —
  this design only changes *which* landmarks/metric drive rep counting, not
  the report-generation or safety-validation layers.

## Non-goals

- No full hand skeleton / per-finger joint angles (that stays rejected).
- No lower-body landmarks (hips/knees/ankles/feet) — still out of scope.
- No automatic inference of which exercise a patient should do — the patient
  or therapist picks the type explicitly.
- No change to the Gemma prompt's clinical-content rules, the safety
  validator's flagged-phrase list, or the packet export/import format beyond
  what's needed to carry the exercise-type label through.

## Architecture

### 1. Generalize `RepetitionCounter`

Currently (`movement_analysis.py:106-164`) hardcoded to
`elbow_angle_degrees`, `flexed_below_degrees`, `extended_above_degrees`,
always starting in state `"flexed"`.

Change to:

```python
@dataclass
class RepetitionCounter:
    low_threshold: float
    high_threshold: float
    smoothing_window: int = 5
    confirm_frames: int = 3
    initial_state: Literal["low", "high"] = "low"

    def update(self, value: Optional[float]) -> None: ...
```

Internal state machine logic (smoothing window, dead zone, confirm-frame
debounce, rep incremented on transition back to the state opposite the one
that completes a cycle) is unchanged — only names and the configurable
starting state change. A rep is counted on `high -> low` if `initial_state ==
"low"`, or `low -> high` if `initial_state == "high"` (grip/pinch starts
"high" = hand open).

### 2. `ExerciseProfile`

New dataclass (in `movement_analysis.py`, or a new `exercise_profiles.py` if
the file grows too large — implementation planning to decide):

```python
@dataclass
class ExerciseProfile:
    key: str                    # "elbow_flexion", "shoulder_raise", ...
    label: str                  # patient-facing name
    instructions_md: str        # Tab 1 task instructions
    metric_label: str           # e.g. "Estimated wrist extension angle at peak"
    low_threshold: float
    high_threshold: float
    initial_state: Literal["low", "high"]
    frame_metric: Callable[[dict, Arm], Optional[float]]  # per-frame scalar
    required_landmark_keys: tuple[str, ...]  # for confidence + fallback checks
```

`frame_metric` receives one frame's landmark dict (same shape already built
in `analyze_video`) and the selected arm, and returns the scalar value fed to
`RepetitionCounter` and tracked for the peak-value metric. Which extremum
counts as the "peak" varies by profile (e.g. elbow/shoulder/wrist care about
the maximum angle/height reached; grip_pinch cares about the minimum
distance, i.e. the most-closed moment) — see the `peak_direction` field in
section 5 below.

### 3. The 5 profiles

| key | frame_metric | low_threshold | high_threshold | initial_state | new landmark? |
|---|---|---|---|---|---|
| `elbow_flexion` (existing) | angle(shoulder, elbow, wrist) | 115° | 145° | low | no |
| `shoulder_raise` | (shoulder_y - palm_y) / shoulder_width | 0.0 | 0.3 | low | no |
| `neck_rotation` | abs((nose_x - shoulder_mid_x) / shoulder_width) | 0.05 | 0.15 | low | no |
| `wrist_flexion` | angle(elbow, wrist, palm) | 150° | 170° | low | no |
| `grip_pinch` | dist(thumb, index) / shoulder_width | 0.05 | 0.15 | **high** | **yes — thumb** |

All thresholds are estimates in the same spirit as the existing elbow
115°/145° pair — not clinically validated, to be sanity-checked against
synthetic test fixtures the same way the elbow thresholds were (no real
rehab video is available in this environment; see README's existing "Known
limitations").

`shoulder_raise`, `neck_rotation` reuse metrics `compute_session_metrics`
already computes today (`palm_height`, `head_offset`) — the generalization
is wiring them into `RepetitionCounter`, not inventing new geometry.

`wrist_flexion` reuses `angle_2d` with different vertex/rays — no new helper
needed.

`grip_pinch` needs a new `distance_2d(a, b)` helper and is the only profile
needing a new landmark read (`thumb`, already defined as `LEFT_THUMB`/
`RIGHT_THUMB` indices in the file but never read into `frame_data` today).

### 4. `analyze_video()` changes

- Add `thumb` to the per-frame landmark dict alongside existing wrist/index/
  pinky reads (cheap — index is already computed, just add one more
  `_landmark_point` call).
- Accept a `profile: ExerciseProfile` argument (replacing the implicit
  elbow-only path).
- Overlay drawing (`_draw_overlay`) stays as today for shoulder/elbow/wrist/
  head — no new overlay lines added for grip_pinch's thumb-index distance
  in this pass (keeps the overlay change surface small); the thumb point
  itself can be drawn as a small dot as a very small addition if trivial,
  otherwise deferred.

### 5. `compute_session_metrics()` changes

Generalized to take a `profile` argument:

- Uses `profile.frame_metric` per frame instead of the hardcoded elbow-angle
  computation.
- Feeds that value into a `RepetitionCounter(profile.low_threshold,
  profile.high_threshold, initial_state=profile.initial_state)`.
- Tracks peak value the same way as today (frame with max metric value,
  except grip_pinch where the notable peak is the *minimum* distance —
  needs a `peak_direction: "max" | "min"` field on `ExerciseProfile` to
  decide which extremum frame to record).
- `head_offset`/landmark confidence tracking stays computed for every
  profile as today (general compensation/quality signal), *except* for
  `neck_rotation`, where the rep-driving metric literally is the head
  offset — computed once, reused for both purposes, not duplicated.
- Output dict's metric key/label switches to `profile.metric_label` instead
  of the hardcoded "Estimated 2D elbow angle at peak reach" string.

## UI changes (`app.py`)

- Tab 1: new `gr.Radio`/`gr.Dropdown` "Exercise type" control next to the
  existing arm selector, options = the 5 profiles' `label`s.
- Tab 1: task instructions `Markdown` becomes dynamic — swap
  `TASK_INSTRUCTIONS_MD` for a function that renders the selected profile's
  `instructions_md`.
- Tab 2: `analyze_video_handler` and the webcam `stop_recording` handler gain
  the exercise-type input and pass it through to `analyze_video()`.
- Tab 2: `_metrics_markdown()` reads `profile.metric_label` instead of the
  hardcoded elbow-angle line.
- Tab 4: the session's "task" description sent to Gemma
  (`generate_with_gemma` → `generate_report`) uses the selected profile's
  label/instructions instead of the fixed `SYNTHETIC_PATIENT["task"]` cup
  task text, so the generated PEO report accurately describes what was
  actually recorded.
- `sample_data/synthetic_metrics.json`: add one synthetic-metrics fixture per
  profile (5 total) so "Use synthetic metrics fallback" / "Use Synthetic Demo
  Session" show data consistent with whichever exercise type is selected,
  instead of always showing elbow-flavored numbers.

## Error handling

- `grip_pinch` needs a low-confidence/unavailable fallback for the thumb
  landmark, mirroring the existing wrist-based palm fallback
  (`approximate_palm_center`'s `is_wrist_based` pattern): if thumb visibility
  is below `CONFIDENCE_THRESHOLD`, report "Grip measurement unavailable" for
  that frame rather than a garbage distance.
- All other profiles reuse landmarks already read for every session (head,
  shoulders, elbow, wrist, palm approximation) — no new failure modes beyond
  what `NoPoseDetectedError`/low-confidence handling already covers.

## Testing plan

- `tests/test_repetition_counter.py`: rename fixture/parameter usage to the
  generalized `value`/`low_threshold`/`high_threshold`/`initial_state` API;
  existing 6 elbow-shaped cases (no movement, one clean rep, three reps,
  dead-zone reading, noise-spike suppression, rise-without-return) continue
  to pass unchanged in meaning.
- Add one parametrized test group per new profile (shoulder_raise,
  neck_rotation, wrist_flexion, grip_pinch) covering: one clean rep, no
  movement, and jitter suppression — reusing the same synthetic-sequence
  pattern as the elbow tests.
- `tests/test_angles.py`: `wrist_flexion`'s angle computation is already
  covered by `angle_2d` tests (different vertex, same function) — no new
  geometry tests needed there.
- New test for `distance_2d()` (grip_pinch): known-distance cases plus a
  degenerate zero-distance case.
- `tests/test_repetition_counter.py` or a new
  `tests/test_exercise_profiles.py`: `compute_session_metrics()` per profile
  against synthetic per-frame landmark data, including the grip_pinch
  low-thumb-confidence fallback path.

## SPEC.md amendment

Add SPEC.md section 28 documenting:

1. The explicit reversal of section 26's "no multiple rehabilitation tasks"
   restriction and section 27's hand-tracking rejection, and why (see
   Background above).
2. The 5 exercise profiles: key, label, instructions text, metric label,
   thresholds.
3. That exercise-type selection is a patient/therapist choice, not an
   AI inference — preserving the "therapist assigns, AI only records"
   framing from section 27.
4. That `grip_pinch` adds exactly one landmark (thumb tip) and one derived
   distance metric — explicitly not a full hand skeleton or per-finger
   goniometry — to make clear how much of section 27's original concern
   still holds.

README's "Implementation status" section gets a short update noting the
5-profile system once built.

## Open items for implementation planning

- Exact instruction text (patient-facing steps) for the 4 new exercise
  types — draft during implementation, following the existing cup-task
  instructions' tone and length.
- Whether `ExerciseProfile` definitions live in `movement_analysis.py` or a
  new `exercise_profiles.py` — implementation planning's call based on how
  large the file gets.
- Whether to add a thumb-point overlay dot for `grip_pinch` in this pass or
  defer it (design defaults to deferring — not required for the metric to
  work).
