"""RecContinue Gradio app shell.

All six tabs are wired to real local logic: Tab 1 (Patient & Task), Tab 2
(Record & Analyze, via movement_analysis.py), Tab 3 (Patient Context),
Tab 4 (Generate Report, via gemma_client.py, auto-saving to the local
vault via storage.py on success), Tab 5 (Privacy & Export, via
packet_export.py), and Tab 6 (Clinician Mode, packet import/review/export
via packet_export.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import json
import pathlib

import gradio as gr

from gemma_client import (
    GemmaInvalidResponseError,
    GemmaModelNotInstalledError,
    GemmaTimeoutError,
    OllamaUnavailableError,
    check_ollama_connection,
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
from safety import validate_report_safety
import storage

BASE_DIR = pathlib.Path(__file__).parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
ASSETS_DIR = BASE_DIR / "assets"
EXPORTS_DIR = BASE_DIR / "exports"

SYNTHETIC_PATIENT = json.loads((SAMPLE_DATA_DIR / "synthetic_patient.json").read_text())
SYNTHETIC_METRICS = json.loads((SAMPLE_DATA_DIR / "synthetic_metrics.json").read_text())
LOGO_SVG = (ASSETS_DIR / "logo.svg").read_text()
LOGO_DATA_URI = "data:image/svg+xml;base64," + __import__("base64").b64encode(
    LOGO_SVG.encode("utf-8")
).decode("ascii")

STATUS_BADGE_TEXT = "Gemma running locally · No cloud processing · Synthetic data only"
SAFETY_BADGE_TEXT = "Documentation support only · Clinician review required"

BRAND_CSS = """
:root {
    --kc-navy: #14253D;
    --kc-teal: #1FA89A;
    --kc-mint: #EAF7F3;
    --kc-coral: #FF7A68;
    --kc-bg: #F7F9FB;
    --kc-text: #1E293B;
    --kc-text-secondary: #64748B;
}
.gradio-container { background: var(--kc-bg) !important; color: var(--kc-text) !important; }
#kc-header { display: flex; align-items: center; gap: 16px; padding: 12px 4px; }
#kc-header img { width: 52px; height: 52px; }
#kc-title-block h1 { font-size: 1.5rem; font-weight: 700; color: var(--kc-navy); margin: 0; }
#kc-title-block p { color: var(--kc-text-secondary); margin: 2px 0 0 0; }
.kc-badge-row { margin: 4px 0 12px 0; }
.kc-badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; margin-right: 8px; margin-bottom: 6px;
}
.kc-badge-status { background: var(--kc-mint); color: var(--kc-navy); border: 1px solid var(--kc-teal); }
.kc-badge-safety { background: #FFF1EE; color: var(--kc-navy); border: 1px solid var(--kc-coral); }
"""

POSITIONING_MD = """
Rehabilitation shouldn't stop when a patient leaves the clinic. RecContinue
helps patients continue an activity their therapist already assigned,
records what happened, and brings that record back for the therapist to
review — the therapist sets the activity and reviews the outcome; Gemma 4
only helps organize what happened in between, locally.
"""

TASK_INSTRUCTIONS_MD = """
### Today's Activity — Reach-and-Place Cup Task

Assigned by your therapist as part of Ms. Lin's post-stroke rehabilitation
plan. Continue it at home, at your own pace:

1. Sit or stand in view of the camera.
2. Place a lightweight cup on a table.
3. Pick it up.
4. Raise it approximately to shoulder height or slightly above.
5. Return it to the table.
6. Repeat three times.

RecContinue records what it observes so your therapist has something
concrete to review at your next visit. It does not tell you whether the
movement is medically correct or safe, and it does not decide what to do
next — that judgment stays with your clinician.
"""

PRIVACY_EXPLANATION_MD = """
### Privacy

- Raw video is only ever processed on this device.
- Gemma 4 only receives simple movement metrics and the context you choose
  to enter — never raw video.
- Raw video is excluded from your exported record by default.
- Your record is saved on this device first; nothing leaves it
  automatically.
- You decide if and when to export a record to bring or send to your
  therapist.
- This demo uses synthetic, staged data only — never a real patient.

RecContinue performs capture, analysis, and report generation locally on
this device. It creates a patient-controlled, minimized clinical packet
that can be transferred through an institution-approved secure channel.
This prototype handles export and import only — network transfer is
outside its scope.
"""

PRIVACY_PREVIEW_MD = """
### Privacy Preview

**Processed locally, never leaves this device:**
✓ Raw movement video
✓ Body landmarks
✓ Motion measurements
✓ Gemma 4 report generation

**Included in your exported record (if you choose to export):**
✓ Session date
✓ Patient-selected context
✓ Summary metrics
✓ PEO report

**Not included by default:**
✗ Raw video
✗ Facial image
✗ Home background
✗ Real patient identity
"""

DEFAULT_CONTEXT = {
    "daily_activity": "Putting cups and dishes away on the kitchen shelf.",
    "difficulty_location": "Reaching the shelf above shoulder height in the kitchen.",
    "assistance_needed": "Sometimes asks a family member to reach the top shelf.",
    "discomfort_reported": "Mild tiredness in the arm after several reaches, no pain reported.",
    "independence_goal": "I want to put cups back on the kitchen shelf without asking for help.",
    "patient_statement": "It's frustrating to need help with something this small.",
}


def _patient_card_markdown() -> str:
    p = SYNTHETIC_PATIENT
    return f"""
**{p['data_status']}**

| | |
|---|---|
| Patient | {p['name']} ({p['patient_id']}) |
| Age | {p['age']} |
| Condition source | {p['condition_source']}: {p['clinician_provided_condition']} |
| Task | {p['task']} |
| Patient goal | "{p['patient_goal']}" |
| Home context | {p['home_context']} |
"""


def _ollama_status_markdown() -> str:
    status = check_ollama_connection()
    icon = "✅" if (status.reachable and status.model_installed) else "⚠️"
    return f"{icon} {status.message}"


def _metrics_markdown(metrics: dict, is_synthetic: bool) -> str:
    if not metrics:
        return "_No metrics yet. Analyze a video or use the synthetic metrics fallback._"
    lines = []
    if is_synthetic:
        lines.append("**Synthetic demonstration data — not generated from a real patient.**")
    conf = metrics.get("landmark_confidence", {})
    lines += [
        f"- Repetition count: **{metrics.get('repetition_count', 'Measurement unavailable')}**",
        f"- Session duration: **{metrics.get('session_duration_seconds', 'Measurement unavailable')} s**",
        f"- Estimated 2D elbow angle at peak reach: **{metrics.get('estimated_2d_elbow_angle_at_peak_reach_degrees', 'Measurement unavailable')}°**",
        f"- Maximum observed hand height relative to shoulder: **{metrics.get('maximum_observed_hand_height_relative_to_shoulder', 'Measurement unavailable')}**",
        f"- Observed head-position variation: **{metrics.get('observed_head_position_variation', 'Measurement unavailable')}**",
        f"- Landmark confidence (average): **{conf.get('average', 'Measurement unavailable')}**",
    ]
    if conf.get("low_confidence"):
        lines.append("\n⚠️ Low landmark confidence. This session requires manual review.")
    lines.append(
        "\n_These are camera-based observations only — not a clinical assessment. "
        "Additional clinician review may be useful._"
    )
    return "\n".join(lines)


def use_synthetic_metrics_fallback():
    return SYNTHETIC_METRICS, _metrics_markdown(SYNTHETIC_METRICS, is_synthetic=True)


def analyze_video_handler(video_path, selected_arm):
    if video_path is None:
        return None, "No video selected. Choose a file above, or use the synthetic metrics fallback.", None

    try:
        annotated_path, metrics = analyze_video(video_path, selected_arm)
    except MovementAnalysisUnavailableError as exc:
        return None, f"⚠️ {exc}", None
    except NoPoseDetectedError as exc:
        return (
            None,
            f"⚠️ {exc} Try a video with better lighting and the {selected_arm} arm fully in "
            "frame, or use the synthetic metrics fallback below.",
            None,
        )
    except ValueError as exc:
        return None, f"⚠️ {exc} Check that the file is a supported video format (e.g. MP4).", None

    return annotated_path, _metrics_markdown(metrics, is_synthetic=False), metrics


def _report_markdown(report_dict: dict) -> str:
    def bullets(items):
        return "\n".join(f"- {item}" for item in items) if items else "_None recorded._"

    return f"""
**Report status:** {report_dict['report_status']}

#### Session summary
{report_dict['session_summary']}

#### Objective observations (camera-based measurements)
{bullets(report_dict['objective_observations'])}

#### Patient-reported information
{bullets(report_dict['patient_reported_information'])}

#### Person factors
{bullets(report_dict['person_factors'])}

#### Environment factors
{bullets(report_dict['environment_factors'])}

#### Occupation factors
{bullets(report_dict['occupation_factors'])}

#### Missing information
{bullets(report_dict['missing_information'])}

#### Clinician follow-up questions
{bullets(report_dict['clinician_follow_up_questions'])}

#### Patient-friendly recap
{report_dict['patient_friendly_recap']}

#### Limitations
{bullets(report_dict['limitations'])}

---
{report_dict['safety_notice']}
"""


def _safety_markdown(safety_result) -> str:
    if safety_result.passed:
        return "✅ Passed automated safety check — no unsupported clinical phrases detected."
    lines = [
        "⚠️ **Requires manual editing.** The following sentences were flagged by the "
        "automated safety check and must be edited before a clinician can mark this "
        "report reviewed. The original output has been preserved, not rewritten:",
    ]
    for item in safety_result.flagged:
        lines.append(f'- _"{item.phrase}"_ in: "{item.sentence}"')
    return "\n".join(lines)


def _vault_summary_markdown(session: dict | None) -> str:
    if not session:
        return (
            "**Local-vault summary:** No session saved locally yet. Generate a report on "
            "the Generate Report tab to save one."
        )
    return (
        f"**Local-vault summary:** Session `{session['session_id']}` for "
        f"`{session['patient']['patient_id']}` saved locally · review status: "
        f"{session['review_status']}."
    )


def use_synthetic_demo_session():
    values = [DEFAULT_CONTEXT[k] for k in [
        "daily_activity", "difficulty_location", "assistance_needed",
        "discomfort_reported", "independence_goal", "patient_statement",
    ]]
    return (
        SYNTHETIC_METRICS,
        _metrics_markdown(SYNTHETIC_METRICS, is_synthetic=True),
        *values,
        "Synthetic demonstration data — not generated from a real patient.",
    )


def generate_with_gemma(
    metrics_state, current_session,
    daily_activity, difficulty_location, assistance_needed,
    discomfort_reported, independence_goal, patient_statement,
):
    if not metrics_state:
        return (
            "No movement metrics available. Analyze a video, use the synthetic metrics "
            "fallback on the Record & Analyze tab, or click “Use Synthetic Demo Session” above.",
            "", "{}", current_session, "",
        )

    patient_context = {
        "daily_activity_goal": daily_activity,
        "difficulty_location": difficulty_location,
        "assistance_needed": assistance_needed,
        "discomfort_reported": discomfort_reported,
        "independence_goal": independence_goal,
        "patient_statement": patient_statement,
    }
    session_id = f"{SYNTHETIC_PATIENT['patient_id']}-session"

    try:
        report = generate_report(session_id, SYNTHETIC_PATIENT, metrics_state, patient_context)
    except OllamaUnavailableError as exc:
        return f"⚠️ {exc}", "", "{}", current_session, ""
    except GemmaModelNotInstalledError as exc:
        return f"⚠️ {exc}", "", "{}", current_session, ""
    except GemmaTimeoutError as exc:
        return f"⚠️ {exc}", "", "{}", current_session, ""
    except GemmaInvalidResponseError as exc:
        return (
            f"⚠️ Gemma's response could not be validated: {exc}\n\n"
            "This session requires manual review. Raw model output has been preserved "
            "in the JSON preview below instead of a fabricated report.",
            "",
            exc.raw_text,
            current_session,
            "",
        )

    safety_result = validate_report_safety(report)
    report_dict = report.model_dump()
    review_status = safety_result.status_label if not safety_result.passed else report_dict["report_status"]
    storage.save_session(
        session_id, SYNTHETIC_PATIENT, metrics_state, patient_context, report_dict,
        video_path=None, review_status=review_status,
    )
    session = {
        "session_id": session_id,
        "patient": SYNTHETIC_PATIENT,
        "movement_metrics": metrics_state,
        "patient_context": patient_context,
        "gemma_report": report_dict,
        "video_path": None,
        "representative_frame_path": None,
        "review_status": review_status,
        "safety_check": {
            "passed": safety_result.passed,
            "flagged_sentences": [
                {"phrase": item.phrase, "sentence": item.sentence} for item in safety_result.flagged
            ],
        },
    }
    return (
        "", _report_markdown(report_dict), json.dumps(report_dict, indent=2), session,
        _safety_markdown(safety_result),
    )


def export_clinician_packet_handler(
    session, include_metrics, include_report, include_statement, include_frame, include_raw_video, consent,
):
    if not session:
        return "No saved session yet. Generate a report on the Generate Report tab first."
    if not consent:
        return "Please confirm the consent checkbox before exporting."
    packet = build_clinician_packet(
        session, include_metrics, include_report, include_statement, include_frame, include_raw_video
    )
    path = export_packet(packet, EXPORTS_DIR)
    return f"Exported clinician packet to `{path}`:\n\n```json\n{json.dumps(packet, indent=2)}\n```"


def delete_session_handler(session):
    if not session:
        return "No saved session to delete.", None, _vault_summary_markdown(None)
    storage.delete_session(session["session_id"])
    return f"Deleted local session `{session['session_id']}`.", None, _vault_summary_markdown(None)


def import_clinician_packet_handler(file_obj):
    if file_obj is None:
        return "No file selected.", None, "_Not imported yet._", "", ""

    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    try:
        data = import_packet(file_path)
    except PacketImportError as exc:
        return f"⚠️ {exc}", None, "_Not imported yet._", "", ""

    metrics = data.get("movement_metrics")
    metrics_md = _metrics_markdown(metrics, is_synthetic=False) if metrics else "_Movement metrics not included in this packet._"

    report = data.get("gemma_report")
    if report:
        report_md = _report_markdown(report)
        safety_check = data.get("safety_check")
        if safety_check and not safety_check.get("passed", True):
            flags = "\n".join(
                f'- _"{f["phrase"]}"_ in: "{f["sentence"]}"' for f in safety_check.get("flagged_sentences", [])
            )
            report_md += (
                "\n\n---\n⚠️ **This report was flagged by the automated safety check and "
                f"cannot be marked reviewed until the flagged text is edited:**\n{flags}"
            )
    else:
        report_md = "_PEO report not included in this packet._"

    context = (data.get("patient_context") or {})
    context_md = (
        "\n".join(f"- **{key}:** {value}" for key, value in context.items())
        if context else "_Patient-reported context not included in this packet._"
    )

    return (
        f"✅ Imported packet for session `{data.get('session_id')}` (patient `{data.get('patient_id')}`).",
        data, metrics_md, report_md, context_md,
    )


def export_reviewed_report_handler(imported_packet, clinician_note, reviewed):
    if not imported_packet:
        return "No imported packet to export. Import a clinician packet above first."
    try:
        path = export_reviewed_report(imported_packet, clinician_note, reviewed, EXPORTS_DIR)
    except SafetyGateError as exc:
        return f"⚠️ {exc}"
    return f"Exported reviewed report to `{path}`."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RecContinue") as demo:
        gr.HTML(
            f"""
            <div id="kc-header">
              <img src="{LOGO_DATA_URI}" alt="RecContinue logo" />
              <div id="kc-title-block">
                <h1>RecContinue</h1>
                <p>Every movement has context. Every record should stay private.</p>
              </div>
            </div>
            <div class="kc-badge-row">
              <span class="kc-badge kc-badge-status">{STATUS_BADGE_TEXT}</span>
              <span class="kc-badge kc-badge-safety">{SAFETY_BADGE_TEXT}</span>
            </div>
            """
        )
        gr.Markdown(POSITIONING_MD)

        metrics_state = gr.State(value=None)
        current_session_state = gr.State(value=None)

        with gr.Tab("1 · Today's Activity"):
            ollama_status_md = gr.Markdown(_ollama_status_markdown())
            refresh_status_btn = gr.Button("Recheck local Gemma status", size="sm")
            gr.Markdown(_patient_card_markdown())
            arm_radio = gr.Radio(
                ["left", "right"],
                value=SYNTHETIC_PATIENT["selected_arm"],
                label="Selected arm for this session",
            )
            gr.Markdown(TASK_INSTRUCTIONS_MD)
            gr.Markdown(PRIVACY_EXPLANATION_MD)
            refresh_status_btn.click(fn=_ollama_status_markdown, outputs=ollama_status_md)

        with gr.Tab("2 · Session & Movement Observations"):
            gr.Markdown(
                "Record a short webcam video or upload an MP4 of the reach-and-place task. "
                "Video stays on this device."
            )
            video_input = gr.Video(sources=["upload", "webcam"], label="Session video")
            analyze_btn = gr.Button("Analyze", variant="primary")
            annotated_output = gr.Video(label="Annotated output", interactive=False)
            metrics_display = gr.Markdown(_metrics_markdown({}, is_synthetic=False))
            synthetic_fallback_btn = gr.Button("Use synthetic metrics fallback")
            gr.Markdown(
                "_Limitations: measurements are 2D, camera-based, and estimated from a "
                "limited upper-body landmark subset. They are not validated clinical "
                "measurements._"
            )
            analyze_btn.click(
                fn=analyze_video_handler,
                inputs=[video_input, arm_radio],
                outputs=[annotated_output, metrics_display, metrics_state],
            )
            synthetic_fallback_btn.click(
                fn=use_synthetic_metrics_fallback, outputs=[metrics_state, metrics_display]
            )

        with gr.Tab("3 · How This Fit Your Day"):
            gr.Markdown(
                "**Patient-reported information** (kept separate from objective measurements). "
                "This is where the activity connects back to real life — not the movement "
                "measurements, but what it meant to try it today."
            )
            daily_activity_tb = gr.Textbox(
                label="What daily activity is the patient trying to regain?",
                value=DEFAULT_CONTEXT["daily_activity"],
            )
            difficulty_location_tb = gr.Textbox(
                label="Where does the difficulty happen?",
                value=DEFAULT_CONTEXT["difficulty_location"],
            )
            assistance_needed_tb = gr.Textbox(
                label="Was assistance needed?", value=DEFAULT_CONTEXT["assistance_needed"]
            )
            discomfort_reported_tb = gr.Textbox(
                label="Was discomfort reported?", value=DEFAULT_CONTEXT["discomfort_reported"]
            )
            independence_goal_tb = gr.Textbox(
                label="What does the patient want to be able to do independently?",
                value=DEFAULT_CONTEXT["independence_goal"],
            )
            patient_statement_tb = gr.Textbox(
                label="Optional patient statement",
                value=DEFAULT_CONTEXT["patient_statement"],
                lines=3,
            )

        with gr.Tab("4 · Private Session Record"):
            gr.Markdown(
                "Your therapist assigned the activity; this step only organizes what "
                "happened locally into a record for them to review — it does not judge "
                "or diagnose anything."
            )
            gen_ollama_status_md = gr.Markdown(_ollama_status_markdown())
            demo_session_btn = gr.Button("Use Synthetic Demo Session")
            demo_session_note = gr.Markdown("")
            generate_btn = gr.Button("Generate with Gemma", variant="primary")
            generate_error_md = gr.Markdown("")
            report_preview_md = gr.Markdown("")
            report_json = gr.Code(label="Editable report preview (JSON)", language="json")
            safety_status_md = gr.Markdown("")

            demo_session_btn.click(
                fn=use_synthetic_demo_session,
                outputs=[
                    metrics_state, metrics_display,
                    daily_activity_tb, difficulty_location_tb, assistance_needed_tb,
                    discomfort_reported_tb, independence_goal_tb, patient_statement_tb,
                    demo_session_note,
                ],
            )
            generate_btn.click(
                fn=generate_with_gemma,
                inputs=[
                    metrics_state, current_session_state,
                    daily_activity_tb, difficulty_location_tb, assistance_needed_tb,
                    discomfort_reported_tb, independence_goal_tb, patient_statement_tb,
                ],
                outputs=[generate_error_md, report_preview_md, report_json, current_session_state, safety_status_md],
            )

        with gr.Tab("5 · Privacy Preview & Export") as tab5:
            gr.Markdown(PRIVACY_PREVIEW_MD)
            vault_summary_md = gr.Markdown(_vault_summary_markdown(None))
            gr.Markdown("Choose what would be included in a clinician packet:")
            include_metrics_cb = gr.Checkbox(value=True, label="Include movement metrics")
            include_report_cb = gr.Checkbox(value=True, label="Include PEO report")
            include_statement_cb = gr.Checkbox(value=True, label="Include patient statement")
            include_frame_cb = gr.Checkbox(value=False, label="Include representative frame")
            include_raw_video_cb = gr.Checkbox(value=False, label="Include raw video")
            consent_cb = gr.Checkbox(value=False, label="I confirm I want to export the selections above")
            export_preview_md = gr.Markdown("")
            with gr.Row():
                export_btn = gr.Button("Export clinician packet")
                delete_btn = gr.Button("Delete session", variant="stop")

            tab5.select(fn=_vault_summary_markdown, inputs=current_session_state, outputs=vault_summary_md)
            export_btn.click(
                fn=export_clinician_packet_handler,
                inputs=[
                    current_session_state, include_metrics_cb, include_report_cb,
                    include_statement_cb, include_frame_cb, include_raw_video_cb, consent_cb,
                ],
                outputs=export_preview_md,
            )
            delete_btn.click(
                fn=delete_session_handler,
                inputs=current_session_state,
                outputs=[export_preview_md, current_session_state, vault_summary_md],
            )

        with gr.Tab("6 · Clinician Mode"):
            clinician_packet_state = gr.State(value=None)
            gr.Markdown("Import a `.reccontinue.json` clinician packet exported from the Privacy & Export tab.")
            packet_file = gr.File(label="Clinician packet (.reccontinue.json)")
            clinician_import_status = gr.Markdown("")
            imported_metrics_md = gr.Markdown("_Not imported yet._")
            imported_report_md = gr.Markdown("")
            imported_context_md = gr.Markdown("")
            clinician_note_tb = gr.Textbox(label="Clinician note", lines=3)
            reviewed_cb = gr.Checkbox(value=False, label="Reviewed by clinician")
            export_reviewed_btn = gr.Button("Export reviewed report")

            packet_file.change(
                fn=import_clinician_packet_handler,
                inputs=packet_file,
                outputs=[
                    clinician_import_status, clinician_packet_state,
                    imported_metrics_md, imported_report_md, imported_context_md,
                ],
            )
            export_reviewed_btn.click(
                fn=export_reviewed_report_handler,
                inputs=[clinician_packet_state, clinician_note_tb, reviewed_cb],
                outputs=clinician_import_status,
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, css=BRAND_CSS)
