"""RecContinue's focused local rehabilitation-record workflow.

The patient-facing flow is intentionally short: choose one clinician-provided
module, record with the webcam, generate a local Gemma documentation draft,
then export a minimal clinician packet.  Clinician review remains separate.
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import json
import pathlib
import time

import gradio as gr

from gemma_client import (
    GemmaInvalidResponseError,
    GemmaModelNotInstalledError,
    GemmaTimeoutError,
    OllamaUnavailableError,
    check_ollama_connection,
    generate_acknowledgment,
    generate_onboarding_intake,
    generate_quick_summary,
    generate_report,
    recommend_observation_module,
)
from movement_analysis import (
    MovementAnalysisUnavailableError,
    NoPoseDetectedError,
    analyze_video,
    create_live_session,
    finalize_live_session,
    process_live_frame,
)
from packet_export import (
    PacketImportError,
    SafetyGateError,
    build_clinician_packet,
    export_html,
    export_packet,
    export_reviewed_report,
    import_packet,
)
from safety import validate_report_safety, validate_text_safety
from stt_client import WhisperUnavailableError, transcribe_audio
import storage
from excel_export import build_session_workbook, build_vault_workbook

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

STATUS_BADGE_TEXT = "Local MediaPipe · fast local report · no cloud processing"
SAFETY_BADGE_TEXT = "Camera observation only · clinician review required"

# Three concise camera-observation views. They control how the local results
# are presented; they never prescribe an exercise or make a clinical decision.
ACTIVITY_MODULES = {
    "head": {
        "label": "Head & Neck",
        "task": "Head and neck turning observation",
        "summary": "Face the camera and slowly turn your head side to side. MediaPipe tracks the nose's 2D position relative to both ears to record the turning range.",
        "context": {
            "daily_activity": "A daily activity involving head and upper-body positioning.",
            "difficulty_location": "Not provided.",
            "assistance_needed": "Not recorded for this session.",
            "discomfort_reported": "Not recorded for this session.",
            "independence_goal": "Not recorded for this session.",
            "patient_statement": "No additional patient statement was recorded for this session.",
        },
    },
    "palm": {
        "label": "Palm",
        "task": "Palm opening and closing observation",
        "summary": "Face your palm toward the camera and slowly open and close your hand. MediaPipe Hands measures the 2D fingertip-to-wrist distance change.",
        "context": {
            "daily_activity": "A daily activity involving reaching or placing an object.",
            "difficulty_location": "Not provided.",
            "assistance_needed": "Not recorded for this session.",
            "discomfort_reported": "Not recorded for this session.",
            "independence_goal": "Not recorded for this session.",
            "patient_statement": "No additional patient statement was recorded for this session.",
        },
    },
    "arm": {
        "label": "Arm & Elbow",
        "task": "Elbow bending observation",
        "summary": "Keep your shoulder, elbow, and wrist on the selected side in frame, and slowly bend then straighten your arm. MediaPipe measures the 2D elbow angle frame by frame.",
        "context": {
            "daily_activity": "A daily activity involving arm movement.",
            "difficulty_location": "Not provided.",
            "assistance_needed": "Not recorded for this session.",
            "discomfort_reported": "Not recorded for this session.",
            "independence_goal": "Not recorded for this session.",
            "patient_statement": "No additional patient statement was recorded for this session.",
        },
    },
}

BRAND_CSS = """
:root {
    --kc-navy: #14253D;
    --kc-teal: #1FA89A;
    --kc-teal-dark: #107567;
    --kc-mint: #EAF7F3;
    --kc-mist: #F4FAF8;
    --kc-coral: #FF7A68;
    --kc-bg: #F6F8F9;
    --kc-text: #1E293B;
    --kc-text-secondary: #64748B;
    --kc-border: #E2E8F0;
    --kc-paper: #FFFFFF;
    --kc-shadow: 0 18px 46px rgba(20, 37, 61, 0.08);
}
.gradio-container {
    max-width: 1240px !important; min-height: 100vh; margin: 0 auto !important;
    padding: 32px clamp(20px, 4vw, 56px) 56px !important;
    color: var(--kc-text) !important; background: var(--kc-bg) !important;
    font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif !important;
    letter-spacing: -0.01em;
}
.gradio-container * { box-sizing: border-box; }
#kc-header {
    position: relative; display: grid; grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center; gap: 16px; padding: 8px 0 24px; margin-bottom: 12px;
    border-bottom: 1px solid rgba(20, 37, 61, 0.12);
}
#kc-header::after { content: ""; position: absolute; left: 0; bottom: -1px; width: 118px; height: 2px; background: var(--kc-teal); }
#kc-header img { width: 48px; height: 48px; filter: drop-shadow(0 8px 12px rgba(20, 37, 61, .12)); }
#kc-title-block .kc-overline { margin: 0 0 3px; color: var(--kc-teal-dark); font-size: .68rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
#kc-title-block h1 { font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; font-size: clamp(1.65rem, 3vw, 2.1rem); font-weight: 700; letter-spacing: -0.055em; color: var(--kc-navy); margin: 0; }
#kc-title-block p { color: var(--kc-text-secondary); font-size: .82rem; margin: 2px 0 0; }
.kc-header-note { display: flex; align-items: center; gap: 7px; color: var(--kc-text-secondary); font-size: .72rem; font-weight: 700; white-space: nowrap; }
.kc-header-note::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--kc-teal); box-shadow: 0 0 0 4px var(--kc-mint); }
.kc-badge-row { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin: 18px 0 22px; }
.kc-badge {
    display: inline-flex; align-items: center; padding: 6px 11px; border-radius: 999px;
    font-size: .7rem; font-weight: 800; line-height: 1.2; letter-spacing: .02em;
}
.kc-badge-status { background: var(--kc-mint); color: var(--kc-teal-dark); border: 1px solid #B5E8DD; }
.kc-badge-safety { background: #FFF6F3; color: #9A4D42; border: 1px solid #FFD4CC; }

/* Shared Gradio surface reset */
.gradio-container .prose { max-width: none; }
.gradio-container .prose p { line-height: 1.65; }
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {
    color: var(--kc-navy); letter-spacing: -0.035em;
}
#kc-tabs { margin-top: 6px; }
#kc-tabs > .tab-nav { gap: 6px; padding: 6px; border: 1px solid var(--kc-border); border-radius: 14px; background: rgba(255,255,255,.76); }
#kc-tabs button { border-radius: 9px !important; color: var(--kc-text-secondary); font-size: .78rem; font-weight: 750; transition: background .18s ease, color .18s ease, box-shadow .18s ease; }
#kc-tabs button.selected { color: var(--kc-navy) !important; background: var(--kc-paper) !important; box-shadow: 0 3px 10px rgba(20,37,61,.08); }
#kc-tabs > .tab-nav button::after { display: none !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
    border-color: #D7E0E7 !important; border-radius: 10px !important; background: #FCFDFD !important;
    box-shadow: none !important;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {
    border-color: var(--kc-teal) !important; box-shadow: 0 0 0 3px rgba(31,168,154,.12) !important;
}
.gradio-container label { color: var(--kc-navy) !important; font-size: .82rem !important; font-weight: 700 !important; }
.gradio-container button { border-radius: 10px !important; font-weight: 750 !important; }

/* Step progress bar (patient flow, Tabs 1-5) */
.kc-stepper {
    display: flex; align-items: flex-start; justify-content: space-between; gap: 0;
    margin: 22px 0 20px; padding: 15px 20px 13px; background: rgba(255,255,255,.76);
    border: 1px solid var(--kc-border); border-radius: 16px;
}
.kc-step { display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 70px; }
.kc-step-dot {
    width: 25px; height: 25px; border-radius: 999px; display: flex; align-items: center;
    justify-content: center; font-size: .7rem; font-weight: 800;
    background: #F2F5F7; color: var(--kc-text-secondary); border: 1px solid var(--kc-border);
}
.kc-step-label { font-size: .67rem; color: var(--kc-text-secondary); text-align: center; line-height: 1.25; max-width: 92px; }
.kc-step-done .kc-step-dot { background: var(--kc-mint); color: var(--kc-teal-dark); border-color: var(--kc-teal); }
.kc-step-active .kc-step-dot { background: var(--kc-teal); color: #fff; border-color: var(--kc-teal); box-shadow: 0 0 0 4px rgba(31,168,154,.13); }
.kc-step-active .kc-step-label { color: var(--kc-navy); font-weight: 700; }
.kc-step-line { flex: 1; height: 1px; background: var(--kc-border); margin-top: 12px; }
.kc-stepper-clinician {
    display: block; padding: 13px 18px; background: var(--kc-navy); color: #fff;
    border-radius: 14px; font-weight: 700; font-size: .86rem;
}

/* Card grouping for visual hierarchy inside tabs */
.kc-card {
    position: relative; overflow: hidden; background: var(--kc-paper) !important;
    border: 1px solid var(--kc-border) !important; border-radius: 16px !important;
    padding: 20px 22px !important; margin-bottom: 14px !important;
    box-shadow: 0 5px 15px rgba(20,37,61,.025);
}
.kc-card .styler, .kc-card > .form,
.kc-landing-card .styler, .kc-landing-card > .form {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
.kc-card-title { font-weight: 800; color: var(--kc-navy); margin: 0 0 7px; font-size: .92rem; letter-spacing: -.02em; }
.kc-intake-card { border-color: #BFE4DA !important; background: linear-gradient(135deg, #FFFFFF 0%, #F1FBF8 100%) !important; }
.kc-intake-card::before { content: "START HERE"; display: block; margin-bottom: 10px; color: var(--kc-teal-dark); font-size: .65rem; font-weight: 800; letter-spacing: .13em; }
.kc-intake-card textarea { background: rgba(255,255,255,.92) !important; }
.kc-module-card { border-color: #BFE4DA !important; background: linear-gradient(140deg, #FFFFFF 0%, #F0FAF7 100%) !important; }
.kc-module-card::before { content: "STEP 01 · CHOOSE ONE"; display: block; margin-bottom: 10px; color: var(--kc-teal-dark); font-size: .65rem; font-weight: 800; letter-spacing: .13em; }
.kc-module-picker fieldset { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px !important; }
.kc-module-picker label { margin: 0 !important; min-height: 74px; padding: 16px !important; border: 1px solid #D7E8E4 !important; border-radius: 12px !important; background: rgba(255,255,255,.88) !important; transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease; }
.kc-module-picker label:hover { border-color: var(--kc-teal) !important; box-shadow: 0 8px 20px rgba(16,117,103,.10); transform: translateY(-1px); }
.kc-module-picker input:checked + span { color: var(--kc-teal-dark) !important; font-weight: 800 !important; }
.kc-module-summary { margin: 14px 0 2px; padding: 13px 15px; border-left: 3px solid var(--kc-teal); border-radius: 0 10px 10px 0; background: #F5FBF9; color: #31534D; }
.kc-camera-card { border-color: #C8DCE7 !important; background: linear-gradient(135deg, #FFFFFF 0%, #F4FAFD 100%) !important; }
.kc-report-lead { font-size: 1rem; color: var(--kc-navy); line-height: 1.65; }
.kc-quiet-note { margin: 12px 0 0; color: var(--kc-text-secondary); font-size: .78rem; }
.kc-patient-recap-card { border-color: #BFE4DA !important; background: linear-gradient(135deg, #FFFFFF 0%, #F1FBF8 100%) !important; }
.kc-patient-recap-lead { font-size: 1.08rem; color: var(--kc-navy); line-height: 1.7; }
.kc-dash-summary { display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 14px; }
.kc-dash-tile { flex: 1; min-width: 140px; background: var(--kc-paper); border: 1px solid var(--kc-border); border-radius: 14px; padding: 14px 16px; }
.kc-dash-tile-num { font-size: 1.7rem; font-weight: 800; color: var(--kc-navy); line-height: 1; }
.kc-dash-tile-label { margin-top: 4px; color: var(--kc-text-secondary); font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
@media (max-width: 640px) { .kc-module-picker fieldset { grid-template-columns: 1fr; } }
.kc-optional-card { border-style: dashed !important; background: #FBFCFD !important; }
.kc-optional-card .prose p { margin-bottom: .3rem; }
.kc-analysis-progress { min-height: 27px; margin-top: 12px; padding: 7px 10px; color: var(--kc-teal-dark); border-radius: 8px; background: var(--kc-mist); font-size: .78rem; font-weight: 700; }
.kc-analysis-progress:empty { display: none; }
.kc-status-card { display: flex; align-items: center; gap: 12px; background: #FCFDFD !important; }

/* Button hierarchy */
button.primary { color: #fff !important; background: var(--kc-teal) !important; border-color: var(--kc-teal) !important; box-shadow: 0 5px 13px rgba(31,168,154,.19); }
button.primary:hover { background: var(--kc-teal-dark) !important; border-color: var(--kc-teal-dark) !important; transform: translateY(-1px); }
button.secondary { color: var(--kc-navy) !important; background: #fff !important; border-color: #D7E0E7 !important; }
.kc-continue { justify-content: flex-end; margin-top: 6px; }
.kc-continue button.primary { min-width: 226px; padding: 10px 16px !important; }
.kc-hint-warning { color: #B75545 !important; font-size: .8rem; margin-top: 5px; }
.kc-helper-text { color: var(--kc-text-secondary); font-size: .8rem; line-height: 1.55; }

/* Landing / welcome screen */
.kc-landing-card {
    position: relative; isolation: isolate; overflow: hidden;
    background: linear-gradient(116deg, #FFFFFF 0%, #FFFFFF 58%, #EEF9F6 100%) !important;
    border: 1px solid #D9E9E5 !important; border-radius: 21px !important;
    padding: clamp(26px, 4vw, 46px) !important; max-width: 920px;
    margin: 6px auto 0 !important; box-shadow: var(--kc-shadow);
}
.kc-landing-card::before { content: ""; position: absolute; z-index: -1; right: -122px; top: -168px; width: 390px; height: 390px; border: 48px solid rgba(31,168,154,.10); border-radius: 50%; }
.kc-landing-card::after { content: ""; position: absolute; z-index: -1; right: 89px; bottom: 27px; width: 9px; height: 9px; border-radius: 50%; background: var(--kc-coral); box-shadow: -34px -17px 0 0 var(--kc-teal), 30px -43px 0 0 rgba(31,168,154,.4), 64px -20px 0 0 rgba(31,168,154,.22); }
.kc-landing-eyebrow { color: var(--kc-teal-dark); font-weight: 800; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; margin: 0; }
.kc-landing-title { max-width: 590px; color: var(--kc-navy); font-family: "Iowan Old Style", Georgia, serif; font-size: clamp(2rem, 4vw, 3.1rem); font-weight: 700; line-height: 1.06; letter-spacing: -.058em; margin: 8px 0 16px; }
.kc-landing-card .prose > p:not(.kc-landing-eyebrow):not(.kc-landing-title) { max-width: 590px; color: #4E6072; font-size: .96rem; }
.kc-track-grid {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 9px;
    max-width: 640px; margin: 24px 0 12px;
}
.kc-track-item {
    background: rgba(255,255,255,.78); border: 1px solid #D6E8E3; border-radius: 12px;
    padding: 12px 7px 11px; text-align: left;
}
.kc-track-icon { font-size: 1.2rem; line-height: 1; }
.kc-track-label { font-weight: 800; color: var(--kc-navy); font-size: .75rem; margin-top: 6px; }
.kc-track-note { color: var(--kc-text-secondary); font-size: .65rem; line-height: 1.25; margin-top: 2px; }
.kc-landing-start { margin-top: 21px; max-width: 270px; }
.kc-landing-start button { font-size: .9rem !important; padding: 11px 15px !important; }

@media (max-width: 760px) {
    .gradio-container { padding: 20px 16px 34px !important; }
    #kc-header { grid-template-columns: auto minmax(0, 1fr); gap: 12px; padding-bottom: 18px; }
    #kc-header img { width: 40px; height: 40px; }
    .kc-header-note { display: none; }
    #kc-title-block h1 { font-size: 1.6rem; }
    #kc-title-block p { font-size: .7rem; line-height: 1.35; }
    .kc-badge-row { margin: 13px 0 18px; }
    .kc-badge { font-size: .62rem; }
    .kc-stepper { padding: 12px 8px; overflow-x: auto; justify-content: flex-start; }
    .kc-step { min-width: 64px; }.kc-step-line { min-width: 10px; }
    #kc-tabs > .tab-nav { overflow-x: auto; justify-content: flex-start; }
    #kc-tabs button { flex: 0 0 auto; font-size: .68rem; }
    .kc-card { padding: 17px !important; border-radius: 14px !important; }
    .kc-track-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); max-width: 360px; }
    .kc-landing-card { padding: 27px 22px !important; border-radius: 17px !important; }
    .kc-landing-title { font-size: 2rem; }.kc-landing-card::before { right: -205px; }
    .kc-continue button.primary { width: 100%; min-width: 0; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition-duration: .01ms !important; } }
"""

TRACKED_PARTS = [
    ("🧠", "Head / Neck", "2D side-to-side turning change"),
    ("✋", "Palm", "2D opening / closing change"),
    ("💪", "Arm / Elbow", "2D bending angle"),
]


def _tracked_parts_html() -> str:
    items = "".join(
        f'<div class="kc-track-item"><div class="kc-track-icon">{icon}</div>'
        f'<div class="kc-track-label">{label}</div>'
        f'<div class="kc-track-note">{note}</div></div>'
        for icon, label, note in TRACKED_PARTS
    )
    return f'<div class="kc-track-grid">{items}</div>'


POSITIONING_MD = """
<p class="kc-report-lead">Choose one view, start the camera, and see the real MediaPipe landmarks while you record.</p>
<p class="kc-quiet-note">This makes a short local observation record; it does not diagnose or prescribe treatment.</p>
"""

DOCTOR_CHECK_MD = """
##### ⚕️ 需要看醫生嗎？你可以這樣想

這裡不會幫你做「該不該看醫生」的判斷 — 這個決定永遠是你自己的。以下提醒或許有幫助：

- 出現**新的**疼痛、明顯加劇的不舒服，或做動作時感到不安全 → 建議儘快聯絡治療師或醫生，不必等到下次回診
- 只是熟悉的痠、累，跟平常差不多 → 通常可以按原本安排繼續，下次回診時再提出也可以
- 拿不準的時候，直接打去問治療師或醫生，永遠是最快、最保險的方法

本頁與報告僅為文件記錄用途，不能取代醫療判斷。
"""

PRIVACY_PREVIEW_MD = """
### Export a minimal clinician packet

Your report and measurement summary remain on this device until you choose to export them. Raw video is not included by default.
"""

DEFAULT_CONTEXT = {
    "daily_activity": "Putting cups and dishes away on the kitchen shelf.",
    "difficulty_location": "Reaching the shelf above shoulder height in the kitchen.",
    "assistance_needed": "Sometimes asks a family member to reach the top shelf.",
    "discomfort_reported": "Mild tiredness in the arm after several reaches, no pain reported.",
    "independence_goal": "I want to put cups back on the kitchen shelf without asking for help.",
    "patient_statement": "It's frustrating to need help with something this small.",
}


PATIENT_STEPS = ["Choose observation", "MediaPipe analysis", "Gemma report", "Export record"]


def _stepper_html(current: int) -> str:
    if current >= len(PATIENT_STEPS):
        return (
            '<div class="kc-stepper kc-stepper-clinician">🩺 Clinician Mode — reviewing a '
            "shared record. This is a separate view from the patient flow above.</div>"
        )
    parts = []
    for i, label in enumerate(PATIENT_STEPS):
        if i < current:
            cls, marker = "kc-step kc-step-done", "✓"
        elif i == current:
            cls, marker = "kc-step kc-step-active", str(i + 1)
        else:
            cls, marker = "kc-step", str(i + 1)
        parts.append(
            f'<div class="{cls}"><span class="kc-step-dot">{marker}</span>'
            f'<span class="kc-step-label">{label}</span></div>'
        )
        if i < len(PATIENT_STEPS) - 1:
            parts.append('<div class="kc-step-line"></div>')
    return f'<div class="kc-stepper">{"".join(parts)}</div>'


def on_tab_select(evt: gr.SelectData):
    return _stepper_html(evt.index)


def continue_to_step(target: int):
    return gr.Tabs(selected=target), _stepper_html(target)


def continue_from_record(metrics_state):
    if not metrics_state:
        return (
            gr.Tabs(), _stepper_html(1),
            "⚠️ Please record a video and complete MediaPipe analysis before generating a report.",
        )
    return gr.Tabs(selected=2), _stepper_html(2), ""


def continue_from_generate(session):
    if not session:
        return (
            gr.Tabs(), _stepper_html(3),
            "⚠️ Generate a session record above before continuing to export.",
        )
    return gr.Tabs(selected=4), _stepper_html(4), ""


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


def _module_markdown(module_name: str | None) -> str:
    if not module_name or module_name not in ACTIVITY_MODULES:
        return "_Choose one module to continue._"
    module = ACTIVITY_MODULES[module_name]
    return (
        f"<div class=\"kc-module-summary\"><strong>Selected · {module['label']}</strong><br>"
        f"{module['summary']}</div>"
    )


def apply_module(module_name: str | None):
    """Require a module choice before camera capture begins."""
    if not module_name or module_name not in ACTIVITY_MODULES:
        return None, "⚠️ Please choose Head, Palm, or Arm first.", gr.Tabs(), _stepper_html(0)
    return module_name, _module_markdown(module_name), gr.Tabs(selected=1), _stepper_html(1)


def quick_summary_handler(age, module_name: str | None):
    """Generate the Tab 1 quick-summary card: a fixed doctor self-check plus a
    short, fast, age-personalized welcome. Never touches movement data or the
    session record — purely a courtesy panel, not part of the clinical report.
    """
    if age is None:
        return "⚠️ 請先輸入你的年齡。"

    assigned_task = ACTIVITY_MODULES.get(module_name or "", {}).get("task")
    try:
        blurb = generate_quick_summary(int(age), assigned_task)
    except (OllamaUnavailableError, GemmaModelNotInstalledError, GemmaTimeoutError):
        blurb = ""

    if blurb and not validate_text_safety(blurb).passed:
        blurb = ""

    return DOCTOR_CHECK_MD + ("\n\n" + blurb if blurb else "")


def _patient_for_module(module_name: str | None) -> dict:
    patient = dict(SYNTHETIC_PATIENT)
    if module_name in ACTIVITY_MODULES:
        module = ACTIVITY_MODULES[module_name]
        patient["task"] = module["task"]
        patient["patient_goal"] = module["context"]["independence_goal"]
        patient["home_context"] = module["context"]["difficulty_location"]
    return patient


def _context_for_module(module_name: str | None, completion_barrier: str = "") -> dict:
    module = ACTIVITY_MODULES.get(module_name or "", ACTIVITY_MODULES["arm"])
    context = dict(module["context"])
    context["activity_completion"] = (
        f"Patient reported a barrier during the selected module: {completion_barrier}."
        if completion_barrier.strip()
        else "Completion status was not recorded separately; review the local camera observations."
    )
    return context


def _ollama_status_markdown() -> str:
    status = check_ollama_connection()
    icon = "✅" if (status.reachable and status.model_installed) else "⚠️"
    return f"{icon} {status.message}"


def organize_onboarding(patient_words: str):
    """Use local Gemma to prefill the editable PEO context form from free text."""
    if not patient_words or not patient_words.strip():
        return (
            "Please describe what you want to do, what feels difficult, and any help you need.",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )

    try:
        intake = generate_onboarding_intake(patient_words.strip(), SYNTHETIC_PATIENT["task"])
    except (OllamaUnavailableError, GemmaModelNotInstalledError, GemmaTimeoutError) as exc:
        return f"⚠️ {exc}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    except GemmaInvalidResponseError as exc:
        return (
            f"⚠️ Gemma could not organize this note: {exc}. Your words were not changed.",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )

    missing = "\n".join(f"- {item}" for item in intake.missing_information) or "- None noted"
    summary = (
        f"**Gemma organized your note locally**  \n{intake.acknowledgement}\n\n"
        f"**Connection to today's assigned activity**  \n{intake.task_connection}\n\n"
        f"**You may still want to add**  \n{missing}\n\n"
        "These fields are editable in Step 3 and will be used only as context for the PEO record."
    )
    return (
        summary,
        intake.daily_activity,
        intake.difficulty_location,
        intake.assistance_needed,
        intake.discomfort_reported,
        intake.independence_goal,
        intake.patient_statement,
    )


def recommend_module_from_words(patient_words: str):
    """Ask local Gemma which of the three existing observation views fits best.

    The returned module remains only a pre-selected interface choice. The user
    can always choose another view, and the text explicitly avoids treatment
    advice or diagnostic claims.
    """
    if not patient_words or not patient_words.strip():
        return "Please describe today's situation in your own words first.", gr.update()
    try:
        recommendation = recommend_observation_module(patient_words.strip())
    except (OllamaUnavailableError, GemmaModelNotInstalledError, GemmaTimeoutError) as exc:
        return f"⚠️ {exc}", gr.update()
    except GemmaInvalidResponseError as exc:
        return f"⚠️ Gemma could not organize this description: {exc}", gr.update()

    module_label = ACTIVITY_MODULES.get(recommendation.observation_module, {}).get(
        "label", recommendation.observation_module
    )
    return (
        f"**Gemma suggests starting with: {module_label}**  \n"
        f"{recommendation.explanation}  \n\n"
        "This is only a suggestion for which camera view to start with — not a diagnosis "
        "or a rehabilitation prescription; you can still choose any option.",
        gr.update(value=recommendation.observation_module),
    )


def _metrics_markdown(metrics: dict, is_synthetic: bool) -> str:
    if not metrics:
        return (
            "### No observation results yet\n"
            "Choose **Head, Palm, or Arm**, then record a video with your camera and press "
            "**Analyze recording**."
        )
    lines = []
    if is_synthetic:
        lines.append("**Synthetic demonstration data — not generated from a real patient.**")
    module = metrics.get("observation_module", "arm")
    conf = metrics.get("landmark_confidence", {})
    if module == "palm":
        lines += [
            "#### MediaPipe Hands · 2D hand-aperture observation",
            f"- Fingertip-to-wrist aperture ratio: min **{metrics.get('palm_opening_ratio_min', 'Measurement unavailable')}** · max **{metrics.get('palm_opening_ratio_max', 'Measurement unavailable')}**",
            f"- Observed 2D aperture excursion: **{metrics.get('palm_opening_ratio_change', 'Measurement unavailable')}**",
            f"- Usable landmark samples: **{metrics.get('detected_frame_count', 'Measurement unavailable')}**",
        ]
    elif module == "head":
        lines += [
            "#### MediaPipe Pose · 2D head/neck rotation proxy",
            f"- Nose-to-ear-midpoint rotation proxy range: **{metrics.get('observed_head_turn_proxy_range', 'Measurement unavailable')}**",
            f"- 2D cranial position variation: **{metrics.get('observed_head_position_variation', 'Measurement unavailable')}**",
            f"- Usable pose samples: **{metrics.get('detected_frame_count', 'Measurement unavailable')}**",
        ]
    else:
        lines += [
            "#### MediaPipe Pose · 2D elbow flexion/extension kinematics",
            f"- Elbow joint angle: min **{metrics.get('observed_2d_elbow_angle_min_degrees', 'Measurement unavailable')}°** · max **{metrics.get('observed_2d_elbow_angle_max_degrees', 'Measurement unavailable')}°**",
            f"- Observed angular excursion: **{metrics.get('observed_2d_elbow_angle_change_degrees', 'Measurement unavailable')}°**",
            f"- Usable pose samples: **{metrics.get('detected_frame_count', 'Measurement unavailable')}**",
        ]
    if conf.get("low_confidence"):
        lines.append("\n⚠️ Low landmark confidence. This session requires manual review.")
    lines.append(
        "\n_These are camera-based observations only — not a clinical assessment. "
        "Additional clinician review may be useful._"
    )
    return "\n".join(lines)


def use_synthetic_metrics_fallback():
    return SYNTHETIC_METRICS, _metrics_markdown(SYNTHETIC_METRICS, is_synthetic=True), ""


def _progress_markdown(progress: dict) -> str:
    total = progress["total_frames"]
    total_str = f"/{total}" if total else ""
    header = f"Processing frame {progress['frame_index']}{total_str}"
    module = progress.get("observation_module", "手臂")

    if module in ("head", "頭部"):
        turn = progress.get("current_turn_proxy")
        turn_str = f"{turn}" if turn is not None else "—"
        return f"{header} · head turn proxy ≈ {turn_str}"
    if module in ("palm", "手掌"):
        opening = progress.get("current_opening_ratio")
        opening_str = f"{opening:.2f}" if opening is not None else "—"
        return f"{header} · palm opening ratio ≈ {opening_str}"

    angle = f"{progress['current_angle']}°" if progress.get("current_angle") is not None else "—"
    return f"{header} · elbow angle ≈ {angle} · reps so far: {progress.get('reps_so_far', 0)}"


def analyze_video_handler(video_path, selected_arm, observation_module="arm"):
    # Gradio 6 can send either a filepath string or a FileData-like object,
    # depending on whether the webcam recorder has just converted WebM to MP4.
    # Normalize both shapes before handing the video to OpenCV.
    if not isinstance(video_path, str):
        if isinstance(video_path, dict):
            video_path = video_path.get("path") or video_path.get("name")
        else:
            video_path = getattr(video_path, "path", None) or getattr(video_path, "name", None)
    if video_path is None:
        yield None, "Please record a video with your computer camera, then press \"Analyze recording\".", None, ""
        return

    try:
        for item in analyze_video(video_path, selected_arm, observation_module=observation_module or "arm"):
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
            "frame.",
            None,
            "",
        )
    except ValueError as exc:
        yield None, f"⚠️ {exc} Check that the file is a supported video format (e.g. MP4).", None, ""
    except Exception as exc:
        yield None, f"⚠️ MediaPipe analysis error: {exc}", None, ""


_LIVE_METRIC_LABELS = {"head": "head turn", "palm": "palm opening", "arm": "elbow angle"}


def live_frame_handler(rgb_frame, observation_module, session):
    """Process one live camera frame locally and return the real MediaPipe overlay."""
    if rgb_frame is None:
        return gr.update(), session, gr.update()
    if session is None:
        try:
            # No left/right picker: the first detected pose selects the more
            # visible arm and keeps that MediaPipe side for the whole session.
            session = create_live_session(observation_module or "arm", "auto")
        except MovementAnalysisUnavailableError as exc:
            return rgb_frame, None, f"⚠️ {exc}"
    annotated_frame, value = process_live_frame(session, rgb_frame, int(time.time() * 1000))
    label = _LIVE_METRIC_LABELS.get(session.observation_module, "movement")
    shown_value = f"{value:.2f}" if value is not None else "detecting…"
    return annotated_frame, session, f"🔴 MediaPipe live · {label}: {shown_value}"


def finish_session_handler(session):
    """Finalize the live recording without a second, slow video-analysis pass."""
    if session is None:
        return None, _metrics_markdown({}, False), None, "⚠️ Start the camera first."
    try:
        annotated_path, metrics, _timeseries = finalize_live_session(session)
    except MovementAnalysisUnavailableError as exc:
        return None, f"⚠️ {exc}", None, ""
    except NoPoseDetectedError as exc:
        return None, f"⚠️ {exc} Keep the selected area clearly in frame and try again.", None, ""
    return annotated_path, _metrics_markdown(metrics, False), metrics, ""


def voice_frontdoor_handler(audio_path, module_name: str | None = None):
    """Transcribe an optional voice note and safely pre-fill Tab 3."""
    if audio_path is None:
        return "", "", gr.update()

    try:
        transcript = transcribe_audio(audio_path)
    except WhisperUnavailableError as exc:
        return f"⚠️ {exc}", "", gr.update()

    if not transcript.strip():
        return "_No speech detected — you can skip this and continue to Step 2._", "", gr.update()

    try:
        assigned_task = ACTIVITY_MODULES.get(module_name or "", {}).get("task", SYNTHETIC_PATIENT["task"])
        acknowledgment = generate_acknowledgment(transcript, assigned_task)
    except (OllamaUnavailableError, GemmaModelNotInstalledError, GemmaTimeoutError):
        acknowledgment = ""

    if acknowledgment and not validate_text_safety(acknowledgment).passed:
        acknowledgment = ""

    return f"**You said:** {transcript}", acknowledgment, gr.update(value=transcript)


def _patient_report_markdown(report_dict: dict) -> str:
    """A short, plain-language recap for the patient — no clinical jargon."""
    return f"""
<p class="kc-patient-recap-lead">{report_dict.get('patient_friendly_recap', 'A plain-language recap was not generated for this session.')}</p>

<p class="kc-quiet-note">{report_dict['safety_notice']}</p>
"""


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

#### Discussion ideas for clinician review
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
    """Keep a small judging fallback without adding more patient-flow fields."""
    return (
        SYNTHETIC_METRICS,
        _metrics_markdown(SYNTHETIC_METRICS, is_synthetic=True),
        "Synthetic demonstration measurements loaded. Generate the report when ready.",
    )


def _build_fast_report(session_id: str, metrics: dict, patient_context: dict) -> dict:
    """Make the short clinic-facing record locally; no model wait or repair loop."""
    module = metrics.get("observation_module", "arm")
    if module == "head":
        observation = f"Head turn proxy range: {metrics.get('observed_head_turn_proxy_range', 'not available')}"
    elif module == "palm":
        observation = f"Palm opening change: {metrics.get('palm_opening_ratio_change', 'not available')}"
    else:
        observation = f"Elbow angle change: {metrics.get('observed_2d_elbow_angle_change_degrees', 'not available')}°"
    patient_note = patient_context.get("patient_statement") or "No additional patient note recorded."
    return {
        "session_id": session_id,
        "report_status": "Quick local draft — clinician review required",
        "session_summary": f"{module.title()} camera observation completed.",
        "objective_observations": [observation, f"Detected frames: {metrics.get('detected_frame_count', 'not available')}"],
        "patient_reported_information": [patient_note],
        "person_factors": [],
        "environment_factors": [],
        "occupation_factors": [],
        "missing_information": [],
        "clinician_follow_up_questions": [],
        "patient_friendly_recap": "Your camera observation was saved as a short draft for clinician review.",
        "limitations": ["Camera-based 2D observation only."],
        "safety_notice": "This short local draft is not a diagnosis or treatment recommendation.",
    }


def generate_with_gemma(metrics_state, current_session, module_name, completion_barrier):
    if not metrics_state:
        return (
            "No MediaPipe observation results yet. Please complete a recording analysis first.",
            "", "", "{}", current_session, "",
        )

    if module_name not in ACTIVITY_MODULES:
        return "Choose a rehabilitation module first.", "", "", "{}", current_session, ""

    patient = _patient_for_module(module_name)
    patient_context = _context_for_module(module_name, completion_barrier or "")
    session_id = f"{SYNTHETIC_PATIENT['patient_id']}-session"

    try:
        report = generate_report(session_id, patient, metrics_state, patient_context)
    except (OllamaUnavailableError, GemmaModelNotInstalledError, GemmaTimeoutError) as exc:
        return f"⚠️ {exc}", "", "", "{}", current_session, ""
    except GemmaInvalidResponseError as exc:
        return (
            f"⚠️ Gemma returned an invalid short report: {exc}. Please retry.",
            "", "", exc.raw_text, current_session, "",
        )

    safety_result = validate_report_safety(report)
    report_dict = report.model_dump()
    review_status = safety_result.status_label if not safety_result.passed else report_dict["report_status"]
    storage.save_session(
        session_id, patient, metrics_state, patient_context, report_dict,
        video_path=None, review_status=review_status,
    )
    session = {
        "session_id": session_id,
        "patient": patient,
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
        "", _patient_report_markdown(report_dict), _report_markdown(report_dict),
        json.dumps(report_dict, indent=2), session, _safety_markdown(safety_result),
    )


def export_minimal_packet_handler(session, include_frame, include_raw_video, consent):
    """Export the focused flow's safe defaults without surfacing every field."""
    return export_clinician_packet_handler(
        session,
        include_metrics=True,
        include_report=True,
        include_statement=True,
        include_frame=include_frame,
        include_raw_video=include_raw_video,
        consent=consent,
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


def export_html_handler(session, include_frame, include_raw_video, consent):
    """Export a self-contained HTML clinician summary, viewable on any phone or
    computer without running RecContinue — meant for sending to a clinician
    ahead of a visit. Includes a short continuity table of the patient's prior
    local sessions."""
    if not session:
        return "No saved session yet. Generate a report on the Generate Report tab first.", None
    if not consent:
        return "Please confirm the consent checkbox before exporting.", None
    packet = build_clinician_packet(session, True, True, True, include_frame, include_raw_video)

    patient_id = packet.get("patient_id")
    history = [
        storage.load_session(summary["session_id"])
        for summary in storage.list_sessions()
        if summary.get("patient_id") == patient_id and summary.get("session_id") != session.get("session_id")
    ]
    history.sort(key=lambda s: s.get("saved_at") or "", reverse=True)

    path = export_html(packet, EXPORTS_DIR, history[:5])
    return f"Exported clinician HTML to `{path}`. Open it on any phone or computer browser.", path


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


def export_session_excel_handler(session):
    """Export the current session's metrics + report as a local .xlsx workbook."""
    if not session:
        return "No saved session yet. Generate a report on the Generate Report tab first.", None
    path = build_session_workbook(session, EXPORTS_DIR)
    return f"Exported Excel report to `{path}`.", path


# --- Clinic dashboard: a live view of every locally saved session, for a clinic
# tracking rehabilitation progress between visits. Reads only the local vault
# (storage.py); nothing here is transmitted anywhere.

DASHBOARD_HEADERS = ["Session ID", "Patient ID", "Saved at (UTC)", "Review status"]


def _dashboard_rows() -> list[list]:
    return [
        [s.get("session_id"), s.get("patient_id"), s.get("saved_at"), s.get("review_status")]
        for s in storage.list_sessions()
    ]


def _dashboard_summary_html() -> str:
    sessions = storage.list_sessions()
    total = len(sessions)
    flagged = sum(1 for s in sessions if (s.get("review_status") or "") == "Requires manual editing")
    awaiting = total - flagged

    def tile(num, label):
        return f'<div class="kc-dash-tile"><div class="kc-dash-tile-num">{num}</div><div class="kc-dash-tile-label">{label}</div></div>'

    return (
        '<div class="kc-dash-summary">'
        + tile(total, "Total local sessions")
        + tile(awaiting, "Awaiting clinician review")
        + tile(flagged, "Flagged — needs manual editing")
        + "</div>"
    )


def refresh_dashboard():
    return gr.update(value=_dashboard_rows()), _dashboard_summary_html()


def export_vault_excel_handler():
    sessions = [storage.load_session(s["session_id"]) for s in storage.list_sessions()]
    if not sessions:
        return "No local sessions saved yet.", None
    path = build_vault_workbook(sessions, EXPORTS_DIR)
    return f"Exported clinic tracking sheet to `{path}`.", path


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RecContinue") as demo:
        gr.HTML(
            f"""
            <div id="kc-header">
              <img src="{LOGO_DATA_URI}" alt="RecContinue logo" />
              <div id="kc-title-block">
                <p class="kc-overline">Private rehabilitation record</p>
                <h1>RecContinue</h1>
                <p>Every movement has context. Every record should stay private.</p>
              </div>
              <div class="kc-header-note">On this device</div>
            </div>
            <div class="kc-badge-row">
              <span class="kc-badge kc-badge-status">{STATUS_BADGE_TEXT}</span>
              <span class="kc-badge kc-badge-safety">{SAFETY_BADGE_TEXT}</span>
            </div>
            """
        )
        metrics_state = gr.State(value=None)
        current_session_state = gr.State(value=None)
        selected_module_state = gr.State(value=None)
        gr.Markdown(POSITIONING_MD)
        stepper_html = gr.HTML("", visible=False)

        with gr.Tabs(elem_id="kc-tabs") as tabs:
            with gr.Tab("1 · Start", id=0):
                with gr.Group(elem_classes=["kc-card", "kc-module-card"]):
                    gr.Markdown("<p class=\"kc-card-title\">Choose one area to observe</p>")
                    module_radio = gr.Radio(
                        [(module["label"], key) for key, module in ACTIVITY_MODULES.items()],
                        value=None, label="Observation", elem_classes=["kc-module-picker"],
                    )
                    module_summary_md = gr.Markdown("")
                    gr.Markdown("<p class=\"kc-card-title\">Voice check-in (automatic)</p>"
                                "Record, then stop: local Whisper transcribes automatically and Gemma replies automatically.")
                    voice_input = gr.Audio(sources=["microphone"], type="filepath", label="Record voice for Gemma")
                    voice_transcript_md = gr.Markdown("**Transcript:** waiting for a recording.")
                    voice_acknowledgment_md = gr.Markdown("**Gemma response:** waiting for a recording.")
                    apply_module_btn = gr.Button("Start camera →", variant="primary")

            with gr.Tab("2 · Record & analyze", id=1):
                with gr.Group(elem_classes=["kc-card", "kc-camera-card"]):
                    selected_module_md = gr.Markdown("_Choose an observation first._")
                    gr.Markdown("<p class=\"kc-card-title\">Record with your camera</p>"
                                "Record, press Stop, then press Analyze once. Analysis runs locally and may take a moment.")
                    arm_radio = gr.Radio(["left", "right"], value=SYNTHETIC_PATIENT["selected_arm"], label="Arm to analyze")
                    video_input = gr.Video(sources=["webcam"], label="Camera recording")
                    analyze_btn = gr.Button("Analyze recording", variant="primary")
                    analysis_progress_md = gr.Markdown("", elem_classes=["kc-analysis-progress"])
                with gr.Group(elem_classes=["kc-card"]):
                    annotated_output = gr.Video(label="MediaPipe annotated result", interactive=False)
                    metrics_display = gr.Markdown(_metrics_markdown({}, is_synthetic=False))
                    report_next_btn = gr.Button("Create Gemma report →", variant="primary")

            with gr.Tab("3 · Gemma report", id=2):
                with gr.Group(elem_classes=["kc-card"]):
                    gr.Markdown("<p class=\"kc-card-title\">Custom short report</p>"
                                "Gemma uses your voice/custom note plus the MediaPipe measurements."
                                " Output is capped for a fast result.")
                    clinician_note = gr.Textbox(label="Custom report note", lines=2)
                    generate_btn = gr.Button("Generate with Gemma", variant="primary")
                    generate_error_md = gr.Markdown("")
                    patient_report_md = gr.Markdown("")
                    report_preview_md = gr.Markdown("")
                    report_json = gr.Code(label="Report data", language="json")
                    safety_status_md = gr.Markdown("")

            apply_module_btn.click(
                fn=apply_module, inputs=module_radio,
                outputs=[selected_module_state, module_summary_md, tabs, stepper_html],
            ).then(fn=_module_markdown, inputs=selected_module_state, outputs=selected_module_md)
            analyze_btn.click(
                fn=analyze_video_handler,
                inputs=[video_input, arm_radio, selected_module_state],
                outputs=[annotated_output, metrics_display, metrics_state, analysis_progress_md],
            )
            report_next_btn.click(fn=lambda: gr.Tabs(selected=2), outputs=tabs)
            voice_input.stop_recording(
                fn=voice_frontdoor_handler,
                inputs=[voice_input, selected_module_state],
                outputs=[voice_transcript_md, voice_acknowledgment_md, clinician_note],
            )
            generate_btn.click(
                fn=generate_with_gemma,
                inputs=[metrics_state, current_session_state, selected_module_state, clinician_note],
                outputs=[generate_error_md, patient_report_md, report_preview_md, report_json, current_session_state, safety_status_md],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        css=BRAND_CSS,
    )
