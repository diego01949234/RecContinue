# Build RecContinue — Complete Claude Code Specification

You are a senior product engineer, Python developer, computer-vision engineer, and local-LLM engineer. Build a complete, runnable one-day healthcare hackathon prototype called RecContinue.

Do not create only mockups, placeholder functions, or TODO files. Build a reliable end-to-end vertical workflow, test it, and document how to run it.

## 1. Hackathon Context

This project is for:

Build with Gemma NYC: On-Device AI for Healthcare

The event is a one-day healthcare AI hackathon using Google DeepMind's open Gemma 4 models.

Selected track:

Track 3 — On-Device Private Health

Official constraints:

- Gemma 4 must be core to the product.
- The prototype must work during a live demo.
- The product must be decision-support only.
- No diagnosis.
- No treatment recommendation.
- Use only synthetic or public data.
- Never use real patient data.
- Patient data should remain on-device.
- The final submission requires:
  - A Kaggle Writeup under 1,500 words
  - A public code repository
  - A working demo, recording, or clonable notebook
  - A clearly selected track
- Public code must clearly show how Gemma 4 is used.
- Draft or unsubmitted Kaggle Writeups do not qualify.

Official evaluation criteria:

- Healthcare Impact — 30%
- Gemma Integration — 25%
- Functionality — 20%
- Presentation and Writeup — 15%
- Privacy and Safety — 10%

Design and engineering decisions should prioritize these scoring criteria.

## 2. Product Name and Brand

Product name:

RecContinue

Meaning:

- "Rec" represents the recorded session — the concrete record of movement and context a patient leaves behind after a single home rehab session.
- "Continue" represents the product's core goal: home-based rehabilitation only works if it keeps happening. Recovery stalls quietly, one skipped or shortened session at a time, long before a clinician notices.

Primary tagline:

Every movement has context.

Extended tagline:

Every movement has context. Every record should stay private.

One-sentence product definition:

RecContinue is an on-device rehabilitation documentation platform that combines simple upper-body movement measurements with patient-reported daily-life context to generate a clinician-ready PEO report using Gemma 4.

The product is not:

- A diagnostic system
- A replacement for a therapist
- A treatment recommendation engine
- An automatic clinical staging system
- A medical device
- A general healthcare chatbot
- A cloud patient-monitoring platform

The product is:

- A local movement-observation tool
- A rehabilitation documentation assistant
- A patient-context collection tool
- A privacy-preserving clinical handoff prototype
- A decision-support tool requiring clinician review

See section 27 for the continuity/interruption-risk positioning refinement layered on top of this base definition.

## 3. Brand Design

Use this visual identity:

- Primary navy: `#14253D`
- Therapy teal: `#1FA89A`
- Soft mint: `#EAF7F3`
- Warm coral: `#FF7A68`
- Background: `#F7F9FB`
- Main text: `#1E293B`
- Secondary text: `#64748B`

The interface should feel:

- Calm
- Human
- Clinical
- Reassuring
- Privacy-first
- Modern but not futuristic

Avoid:

- Neon gradients
- Cyberpunk styling
- Excessive animation
- Generic AI robot imagery
- Medical claims
- Red warning-heavy interfaces

Create a simple local SVG logo using selected movement nodes to form an "R" whose leg flows into an infinity loop, inside a rounded frame — the letter for "Rec," the loop for "Continue."

Persistent application status badge:

Gemma running locally · No cloud processing · Synthetic data only

Persistent safety badge:

Documentation support only · Clinician review required

## 4. Primary Problem

Rehabilitation sessions generate highly sensitive information:

- Videos of a patient moving inside their home
- Body movement measurements
- Patient voice or written descriptions
- Functional difficulties
- Home-environment information
- Clinical observations

Traditional remote rehabilitation platforms may upload raw video and health information to cloud systems.

RecContinue follows a privacy-first alternative:

1. Capture movement locally.
2. Analyze selected movement points locally.
3. Generate the PEO report locally using Gemma 4.
4. Store everything locally first.
5. Keep raw video on the patient's device by default.
6. Let the patient select what information is included in a clinician packet.
7. Export only the minimum information required for clinical review.

The hackathon prototype must not automatically upload or transmit patient information.

## 5. Core User Story

Use one synthetic patient and one rehabilitation task.

Synthetic patient:

```json
{
  "patient_id": "SYN-001",
  "name": "Ms. Lin",
  "age": 52,
  "condition_source": "Clinician-provided",
  "clinician_provided_condition": "Post-stroke rehabilitation",
  "task": "Reach-and-place cup task",
  "selected_arm": "right",
  "patient_goal": "I want to put cups back on the kitchen shelf without asking for help.",
  "home_context": "The frequently used kitchen shelf is above shoulder height.",
  "data_status": "Entirely synthetic"
}
```

The AI must not infer or diagnose the condition.

Primary task:

Reach-and-Place Cup Task

Instructions:

1. Sit or stand in view of the camera.
2. Place a lightweight cup on a table.
3. Pick it up.
4. Raise it approximately to shoulder height or slightly above.
5. Return it to the table.
6. Repeat three times.

Do not tell the patient whether the movement is medically correct or safe.

## 6. Simplified Movement Scope

MediaPipe may internally detect the full pose, but RecContinue must only display and use a limited set of upper-body landmarks.

Use these landmarks:

Head

- Nose
- Left ear
- Right ear

Upper body

- Left shoulder
- Right shoulder
- Selected-arm elbow
- Selected-arm wrist

Approximate hand/palm

- Selected-arm wrist
- Selected-arm index point
- Selected-arm pinky point
- Selected-arm thumb point

Approximate the palm center using the available wrist, index, and pinky points.

If hand points are unreliable, fall back to the wrist point and clearly label the value as wrist-based.

Do not display or analyze:

- Hips
- Knees
- Ankles
- Feet
- Full facial mesh
- Individual finger joints
- Gait
- Balance
- Lower-body movement

The output overlay should show only:

- Head reference points
- Shoulder line
- Selected upper arm
- Selected forearm
- Approximate palm or wrist point

## 7. Simple Movement Metrics

Do not implement complex rehabilitation biomechanics.

Compute only:

A. Repetition Count

Count a repetition using the estimated 2D elbow angle (selected shoulder →
selected elbow → selected wrist):

1. The session starts in a flexed state (elbow angle below 115 degrees).
2. The angle rises to an extended state (above 145 degrees).
3. The angle returns to a flexed state (below 115 degrees) — this
   completes one repetition.

Smooth the angle with a 5-frame moving average before classifying it, and
require 3 consecutive smoothed frames past a threshold before accepting a
state change, to prevent single-frame jitter from double-counting or
missing a repetition. Readings between the two thresholds are a dead
zone that neither confirms nor resets a pending transition.

B. Session Duration

Measure video duration in seconds.

C. Estimated Elbow Angle at Peak Reach

Use the 2D angle formed by:

- Selected shoulder
- Selected elbow
- Selected wrist

This is only an estimated camera-based observation.

Label it: Estimated 2D elbow angle at peak reach

Never label it clinical range of motion or validated goniometry.

D. Maximum Palm Height Relative to Shoulder

Calculate the vertical distance between the approximate palm center and selected shoulder.

Normalize using shoulder width when possible.

Label it: Maximum observed hand height relative to shoulder

E. Head Movement Indicator

Calculate the horizontal movement of the nose relative to the midpoint of the shoulder line.

Normalize using shoulder width.

Label it: Observed head-position variation

Do not describe it as compensation, neurological impairment, abnormal posture, or unsafe movement.

F. Landmark Confidence

Show average pose confidence.

If confidence is below a reasonable threshold, display: Low landmark confidence. This session requires manual review.

## 8. Required Wording for Movement Results

Allowed wording:

- Observed movement
- Estimated 2D angle
- Camera-based measurement
- Head-position variation
- Movement variation
- Additional clinician review may be useful
- Measurement unavailable
- Low landmark confidence

Prohibited wording:

- Correct movement
- Incorrect movement
- Normal
- Abnormal
- Safe
- Unsafe
- Compensatory movement
- Neurological deficit
- Brunnstrom stage
- Patient improved
- Patient deteriorated
- Recommended exercise
- Recommended treatment

## 9. Patient Context Collection

Required P0 input:

Use editable text fields for:

- What daily activity is the patient trying to regain?
- Where does the difficulty happen?
- Was assistance needed?
- Was discomfort reported?
- What does the patient want to be able to do independently?
- Optional patient statement

Prefill these fields with synthetic content.

Voice input is a P1 stretch feature only.

If voice is implemented:

- Record a maximum of 30 seconds.
- Process transcription locally.
- Never call browser cloud speech-recognition APIs.
- Never call Google Speech-to-Text, OpenAI Whisper API, or another cloud service.
- Prefer Gemma 4 native audio only if local runtime support is verified.
- Otherwise use an optional local transcription adapter.
- Always allow the patient to view and edit the transcript.
- The application must remain fully functional with text input alone.

Do not delay the P0 workflow to implement voice.

## 10. Correct Product Workflow

Step 1 — Patient and Task

- Load synthetic patient.
- Select left or right arm.
- Display task instructions.
- Clearly display "Synthetic patient data."

Step 2 — Record or Upload

- Record a short webcam video or upload an MP4.
- Allow a clearly labeled synthetic sample video or synthetic metrics fallback.
- Save video locally only.

Step 3 — Analyze Locally

- Run MediaPipe and OpenCV locally.
- Draw only selected landmarks.
- Produce an annotated local video or representative annotated frames.
- Calculate the five simple metrics.
- Show confidence and limitations.

Step 4 — Add Patient Context

- Collect the PEO-related text fields.
- Keep objective measurements separate from patient-reported information.

Step 5 — Generate with Gemma 4

- Send only locally held synthetic information to Gemma 4 through Ollama.
- Generate a structured PEO documentation draft.
- Never send information to an external API.

Step 6 — Save to Local Vault

Save locally:

- Session metadata
- Movement metrics
- Patient context
- Gemma report
- Local video path
- Timestamp
- Review status

Do not save real patient data.

Step 7 — Privacy Preview

Let the user decide what may be included in the clinician packet.

Default selections:

- Include movement metrics: Yes
- Include PEO report: Yes
- Include patient statement: Yes
- Include representative frame: No
- Include raw video: No

Show an explicit preview before export.

Step 8 — Export Clinician Packet

Export a local JSON-based file: `SYN-001-session.reccontinue.json`

The packet should contain only patient-approved fields.

Do not automatically email, upload, message, or transmit the packet.

Step 9 — Clinician Import and Review

Provide a separate Clinician Mode tab.

The clinician can:

- Import the `.reccontinue.json` packet
- View movement metrics
- View PEO report
- View patient-reported context
- View included representative frame if authorized
- Edit the documentation
- Mark the report as reviewed
- Add a clinician note
- Export a reviewed local report

The clinician, not Gemma, decides treatment.

## 11. Why Gemma 4 Is Core

Gemma 4 must not be used only to rewrite text.

Gemma 4 is responsible for:

Context Fusion

Combine:

- Objective movement measurements
- Patient-reported information
- Daily-life goal
- Home environment
- Rehabilitation task

PEO Organization

Organize information into:

- Person
- Environment
- Occupation

Missing-Information Detection

Identify missing information without inventing it.

Examples:

- Assistance level not documented
- Discomfort not documented
- Camera confidence too low
- Task completion conditions unclear

Clinician Question Generation

Generate neutral questions a clinician may consider asking.

Do not generate treatment recommendations.

Patient-Friendly Recap

Translate the technical session record into plain language without making clinical conclusions.

Without Gemma, the system only produces numbers. Gemma turns local measurements and patient context into structured, usable documentation.

## 12. Gemma Report Schema

Use Pydantic to validate this schema:

```json
{
  "session_id": "string",
  "report_status": "AI-generated draft requiring clinician review",
  "session_summary": "string",
  "objective_observations": ["string"],
  "patient_reported_information": ["string"],
  "person_factors": ["string"],
  "environment_factors": ["string"],
  "occupation_factors": ["string"],
  "missing_information": ["string"],
  "clinician_follow_up_questions": ["string"],
  "patient_friendly_recap": "string",
  "limitations": ["string"],
  "safety_notice": "string"
}
```

The report must clearly separate:

- Objective camera measurements
- Patient-reported information
- AI-generated organization
- Clinician-authored notes

## 13. Gemma System Prompt

Use this as the primary Gemma system instruction:

```text
You are RecContinue, an offline rehabilitation documentation assistant.

Your role is to organize synthetic patient-reported information and locally measured camera-based movement observations into a structured documentation draft using the Person–Environment–Occupation framework.

You are not a clinician and you do not provide medical advice.

You must:

1. Clearly distinguish objective measurements from patient-reported information.
2. Use neutral, observational, uncertainty-aware language.
3. Organize information into Person, Environment, and Occupation factors.
4. Identify information that is missing or has low confidence.
5. Generate neutral follow-up questions for a licensed rehabilitation professional.
6. Generate a plain-language patient recap.
7. Label the entire report as an AI-generated draft requiring clinician review.
8. Include the limitations of camera-based 2D movement measurement.
9. Output valid JSON matching the requested schema.

You must not:

1. Diagnose a disease or condition.
2. Assign a Brunnstrom stage or any clinical severity classification.
3. Recommend exercises, treatment, medication, or care plans.
4. determine whether a movement is medically correct, normal, abnormal, safe, or unsafe.
5. Interpret head movement as compensation or neurological impairment.
6. Claim clinical validation or diagnostic accuracy.
7. Invent measurements, symptoms, or patient history.
8. Merge patient-reported information with objective measurement.
9. claim that the patient improved or deteriorated.
10. expose internal reasoning.

If information is missing, explicitly state that it is missing.

If landmark confidence is low, state that the camera-based observation requires manual review.

Return JSON only.
```

## 14. Privacy Model

RecContinue follows these principles:

- Local inference
- Local storage
- Synthetic data only
- No cloud AI
- No analytics
- No telemetry
- No automatic data transmission
- Raw video excluded from export by default
- Patient-controlled selective disclosure
- Data minimization
- Clinician review
- Deletion controls

Important wording:

Do not claim that remote clinician transfer can happen without any network.

Instead state:

RecContinue performs capture, analysis, and report generation locally. It creates a patient-controlled, minimized clinical packet that can be transferred through an institution-approved secure channel. Raw video remains local by default.

The hackathon prototype handles export and import only. Network transfer is outside scope.

## 15. Technical Stack

Use:

- Python 3.11
- Gradio Blocks
- OpenCV
- MediaPipe Pose Landmarker
- NumPy
- Pydantic
- Requests
- Ollama local HTTP API
- Local JSON storage
- Pytest

Default Gemma model: `gemma4:e2b`

Ollama endpoint: `http://localhost:11434/api/generate`

Application requirements:

- Bind only to `127.0.0.1`
- Use `share=False`
- Set `GRADIO_ANALYTICS_ENABLED=False`
- No CDN dependencies at runtime
- No Google Fonts loaded from the web
- No cloud database
- No account system
- No authentication
- No API keys
- No hosted analytics
- No external healthcare services
- No automatic uploads

The application must start even if Ollama is unavailable and show exact setup instructions instead of crashing.

## 16. Required Project Structure

```text
reccontinue/
├── app.py
├── movement_analysis.py
├── gemma_client.py
├── prompts.py
├── schemas.py
├── safety.py
├── storage.py
├── packet_export.py
├── models/
│   └── README.md
├── sample_data/
│   ├── synthetic_patient.json
│   ├── synthetic_metrics.json
│   └── README.md
├── assets/
│   └── logo.svg
├── tests/
│   ├── test_angles.py
│   ├── test_repetition_counter.py
│   ├── test_report_schema.py
│   ├── test_safety.py
│   └── test_packet_export.py
├── requirements.txt
├── README.md
└── .gitignore
```

Do not commit:

- Ollama model weights
- MediaPipe model weights if too large
- Generated patient sessions
- Recorded videos
- Audio
- Temporary files
- Any private information

## 17. Required UI

Create these Gradio tabs:

Tab 1 — Patient & Task

- RecContinue header
- Local status indicators
- Synthetic patient card
- Left/right arm selector
- Reach-and-place instructions
- Privacy explanation

Tab 2 — Record & Analyze

- Webcam/video input
- Upload input
- Analyze button
- Annotated video or frame output
- Metrics cards
- Confidence indicator
- Limitations notice
- Synthetic metrics fallback

Tab 3 — Patient Context

- PEO-related questions
- Patient statement
- Optional voice capture only if P1 is implemented
- Clear patient-reported label

Tab 4 — Generate Report

- Ollama connection status
- Generate with Gemma button
- Structured report sections
- Editable report preview
- Safety warnings
- Report-status label

Tab 5 — Privacy & Export

- Local-vault summary
- Data inclusion checkboxes
- Raw-video checkbox default off
- Representative-frame checkbox default off
- Consent confirmation
- Clinician-packet preview
- Export button
- Delete-session button

Tab 6 — Clinician Mode

- Import clinician packet
- Display metrics
- Display PEO report
- Display patient-reported information
- Editable clinician note
- "Reviewed by clinician" checkbox
- Export reviewed report

## 18. Safety Validation

Implement deterministic post-generation validation.

Check only the clinical-content sections, not the safety disclaimer itself.

Flag unsupported phrases such as:

- "you have"
- "the patient has"
- "diagnosed with"
- "Brunnstrom stage"
- "should perform"
- "recommended exercise"
- "recommended treatment"
- "safe to continue"
- "unsafe"
- "normal movement"
- "abnormal movement"
- "neurological deficit"
- "compensation pattern"

If detected:

- Show the exact flagged sentence.
- Mark the report "Requires manual editing."
- Prevent final clinician approval until the text is edited.
- Do not silently rewrite it.
- Do not delete the original output.

## 19. Error Handling

The application must handle:

- No video selected
- Unsupported video format
- Camera permission unavailable
- No pose detected
- Low-confidence landmarks
- Selected arm outside the frame
- Hand points unavailable
- Corrupted clinician packet
- Ollama not running
- Gemma model not installed
- Gemma timeout
- Invalid JSON response
- Local save failure

For invalid Gemma JSON:

1. Preserve the original output.
2. Attempt one JSON-repair prompt.
3. Validate again with Pydantic.
4. If still invalid, show a useful manual-review error.

Never fabricate a successful report.

## 20. Synthetic Demo Mode

Provide a clearly labeled button: Use Synthetic Demo Session

This mode may load:

- Synthetic movement metrics
- Synthetic patient context
- Synthetic report input

It must still call Gemma 4 to generate the report.

Do not present synthetic metrics as camera-derived measurements.

Display: Synthetic demonstration data — not generated from a real patient.

## 21. Acceptance Criteria

The P0 prototype is complete only when all of the following work:

1. The application starts locally.
2. The synthetic patient loads.
3. User can select left or right arm.
4. User can upload or record a short video.
5. Selected upper-body landmarks are detected.
6. An annotated video or representative frame is displayed.
7. Repetitions are counted.
8. Session duration is shown.
9. Estimated 2D elbow angle at peak reach is shown.
10. Maximum hand height relative to shoulder is shown.
11. Head-position variation is shown.
12. Landmark confidence is shown.
13. Patient context can be entered.
14. Gemma 4 generates a valid PEO JSON report locally.
15. The report separates measurements from patient statements.
16. The report contains no diagnosis or treatment recommendation.
17. The session can be saved locally.
18. Patient can preview data before export.
19. Raw video is excluded by default.
20. A clinician packet can be exported.
21. Clinician Mode can import the packet.
22. Clinician can add a note and mark the report reviewed.
23. The session can be deleted.
24. The core workflow works without internet after dependencies and models are installed.
25. Tests pass.
26. README contains complete setup and demo instructions.

## 22. Required Setup Flow

README must include:

```bash
ollama pull gemma4:e2b
ollama serve
```

Then:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Also explain:

- How to obtain the MediaPipe Pose Landmarker model
- Where to place the local model file
- How to verify Ollama
- How to use synthetic demo mode
- How to record the live demo
- How to test offline mode
- Known limitations

## 23. README Content

README must explain:

1. The rehabilitation documentation problem
2. Why rehabilitation video is privacy-sensitive
3. Why on-device inference matters
4. Why Gemma 4 is core
5. How the PEO framework is used
6. Simplified movement-analysis approach
7. Privacy architecture
8. Patient-controlled clinical packet
9. Synthetic-data policy
10. Decision-support limitations
11. Project architecture
12. Setup
13. Demo workflow
14. Tests
15. Hackathon track
16. Future work

Do not include unsupported claims about:

- Diagnostic accuracy
- Clinical validation
- Treatment effectiveness
- Recovery improvement
- Regulatory approval
- Real patient usage

## 24. Implementation Order

Implement in this order:

Phase 1 — Gemma Spike

- Test Ollama connection.
- Send synthetic metrics and context.
- Receive valid JSON.
- Validate with Pydantic.

Do not build the entire UI before testing Gemma.

Phase 2 — App Shell

- Build Gradio tabs.
- Add brand styling.
- Load synthetic patient.
- Add status indicators.

Phase 3 — Simple Movement Analysis

- Video upload
- Selected landmarks
- Repetition count
- Simple metrics
- Annotated output

Phase 4 — End-to-End Integration

- Patient context
- Gemma report
- Local save
- Privacy preview
- Packet export
- Clinician import

Phase 5 — Safety and Testing

- Safety validator
- Error states
- Offline test
- Synthetic demo
- Automated tests

Phase 6 — Submission Support

- Complete README.
- Include architecture diagram in Mermaid.
- Include screenshots only after the working app exists.
- Draft a Kaggle Writeup outline under 1,500 words.

Voice input is implemented only after all P0 acceptance criteria pass.

## 25. Demo Story

The demo should tell this story:

A synthetic patient, Ms. Lin, wants to place cups back on a high kitchen shelf without asking for help.

RecContinue records a short reach-and-place session and analyzes only selected upper-body points locally: head, shoulders, elbow, wrist, and approximate palm.

The system produces simple camera-based observations, not clinical judgments.

Ms. Lin adds context about her kitchen and independence goal.

Gemma 4 runs locally and turns the measurements and patient statement into a structured PEO report.

Before anything leaves the device, Ms. Lin reviews exactly what will be included.

Raw video remains local by default.

A minimized clinician packet is exported and imported into Clinician Mode.

The clinician reviews the information and makes all treatment decisions.

Closing line: Every movement has context. Every record should stay private.

## 26. Final Engineering Instructions

Before declaring completion:

- Run the application.
- Run all tests.
- Test synthetic mode.
- Test one real sample video.
- Test low-confidence handling.
- Test Ollama unavailable state.
- Test invalid Gemma JSON.
- Test export and import.
- Verify raw video is excluded by default.
- Verify no cloud API is called.
- Verify app binds only to localhost.
- Verify generated clinical content contains no diagnosis or treatment advice.
- Document any remaining limitations honestly.

Do not expand scope beyond this specification.

Do not add:

- Lower-body analysis
- Full facial analysis
- Real patient data
- Cloud transmission
- User accounts
- Hospital integration
- Billing
- Scheduling
- Chatbots
- Treatment recommendations
- Clinical staging
- Complex rehabilitation biomechanics
- Multiple rehabilitation tasks

Build the reliable P0 vertical workflow first.

At the end, report:

1. Files created
2. Commands to run
3. Tests executed
4. What works
5. Remaining limitations
6. Exact live-demo steps

## 27. Addendum — Positioning Refinement (decided after initial P0 build)

After Phases 1–5 were built and tested against the original spec above,
the product owner refined RecContinue's positioning while reviewing a
companion vision deck ("Novatera" — a broader AI occupational-therapy
assistance concept). This addendum documents that refinement; it amends
the framing and UI copy, not the underlying architecture, privacy model,
or safety constraints defined in sections 1–26 above, all of which
remain in force.

### The core reframe

The product's center of gravity is **not** "AI measures your rehabilitation
movement angles." It is:

> Rehabilitation shouldn't stop just because a patient leaves the clinic.

RecContinue helps a patient continue an activity their therapist already
assigned, records what happened locally, and brings that record back for
the therapist to review. The therapist sets the activity and reviews the
outcome; Gemma 4 only helps organize what happened in between. This must
never be described as the app deciding what a patient should do, or the
app judging whether rehabilitation succeeded — both remain explicitly
out of scope, consistent with section 26's existing prohibitions.

Not this:
```text
AI decides what the patient should do → AI judges whether rehab succeeded
```
But this:
```text
Therapist assigns an activity → patient does it at home → AI helps
record what happened → therapist reviews it afterward
```

### Confirmed: no finger-level hand tracking

A hand-tracking pivot (MediaPipe Hands, per-finger joint angles, OT-style
goniometry) was considered and explicitly **rejected** in favor of this
reframe. Section 6's simplified upper-body landmark subset (head,
shoulders, selected-arm elbow/wrist, approximate palm from
wrist/index/pinky) stays as the measurement approach. The value
proposition is continuity of care and privacy, not measurement precision.

### Updated taglines

Primary (unchanged, still accurate and tied to the existing logo/brand):
"Every movement has context. Every record should stay private."

Positioning-deck alternatives, usable in pitch/writeup contexts:
- "Private rehabilitation continuity, wherever recovery happens."
- "RecContinue keeps rehabilitation connected without keeping patients watched."
- "Recovery continues at home. Privacy stays there too."

### Privacy Preview checklist (implemented in Tab 5)

```text
Processed locally, never leaves this device:
✓ Raw movement video
✓ Body landmarks
✓ Motion measurements
✓ Gemma 4 report generation

Included in your exported record (if you choose to export):
✓ Session date
✓ Patient-selected context
✓ Summary metrics
✓ PEO report

Not included by default:
✗ Raw video
✗ Facial image
✗ Home background
✗ Real patient identity
```

### Illustrative pitch narrative (writeup/demo use only — not the app's synthetic test data)

For the Kaggle Writeup and pitch narrative, a patient named "Maria" (who
lives far from her clinic) illustrates the continuity-of-care story. This
is separate from the app's actual synthetic demo patient, Ms. Lin
(`SYN-001`), which remains the technical fixture used throughout
`sample_data/` and the tests. When telling Maria's story, do not say she
"doesn't need to go to the hospital" — say: "RecContinue supports
continuity between clinician visits; it does not replace clinical care."

### 30-second pitch (reference text)

> Rehabilitation should not stop when a patient leaves the clinic. For
> people living far from care—or facing mobility and transportation
> barriers—the time between appointments can become a documentation
> blind spot. RecContinue is a privacy-first mobile rehabilitation
> companion that helps patients continue clinician-assigned activities at
> home. Basic movement observations are extracted locally, and Gemma 4
> converts those measurements and the patient's lived context into a
> structured PEO session record. Raw video stays on the device, and the
> patient decides what to share with their therapist. RecContinue supports
> continuity between visits—it does not diagnose, prescribe or replace
> professional care.

### Intended deployment note

The long-term target is a mobile app. This Gradio prototype demonstrates
the same end-to-end local pipeline in a mobile-oriented web layout; when
presenting, say so plainly: "Today's prototype demonstrates the
end-to-end local pipeline in a mobile-oriented web interface. The
intended deployment is an on-device mobile application."
