"""Patient-controlled clinician packet export/import for RecContinue (Phase 4).

Exports a minimized, patient-approved subset of a saved session to a local
`<patient_id>-session.reccontinue.json` file. Export and import both
operate on local files only — nothing here transmits data anywhere
(SPEC.md section 14: "The hackathon prototype handles export and import
only. Network transfer is outside scope.").
"""
from __future__ import annotations

import html
import json
import pathlib
from datetime import datetime, timezone
from typing import Any

PACKET_SUFFIX = ".reccontinue.json"


class PacketImportError(Exception):
    """Raised when a clinician packet file is missing, corrupt, or malformed."""


class SafetyGateError(Exception):
    """Raised when a report that failed the automated safety check is marked
    reviewed without first being edited (SPEC.md section 18: "Prevent final
    clinician approval until the text is edited")."""


def build_clinician_packet(
    session: dict[str, Any],
    include_metrics: bool,
    include_report: bool,
    include_patient_statement: bool,
    include_representative_frame: bool,
    include_raw_video: bool,
) -> dict[str, Any]:
    """Build the minimized, patient-approved packet for a saved session.

    Raw video and the representative frame are excluded unless explicitly
    opted in (both default off, per SPEC.md section 10 step 7).
    """
    patient = session.get("patient", {})
    packet: dict[str, Any] = {
        "packet_version": "1.0",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session.get("session_id"),
        "patient_id": patient.get("patient_id"),
        "data_status": patient.get("data_status", "Entirely synthetic"),
        "task": patient.get("task"),
        "selected_arm": (session.get("movement_metrics") or {}).get("selected_arm"),
        "review_status": session.get("review_status"),
    }

    if include_metrics:
        packet["movement_metrics"] = session.get("movement_metrics")
    if include_report:
        packet["gemma_report"] = session.get("gemma_report")
        packet["safety_check"] = session.get("safety_check")
    if include_patient_statement:
        packet["patient_context"] = session.get("patient_context")
    if include_representative_frame:
        packet["representative_frame_path"] = session.get("representative_frame_path")
    if include_raw_video:
        packet["raw_video_path"] = session.get("video_path")

    return packet


def export_packet(packet: dict[str, Any], output_dir: str | pathlib.Path) -> str:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_id = packet.get("patient_id") or "session"
    output_path = output_dir / f"{patient_id}-session{PACKET_SUFFIX}"
    output_path.write_text(json.dumps(packet, indent=2))
    return str(output_path)


def import_packet(path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(path)
    if not path.exists():
        raise PacketImportError(f"Packet file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PacketImportError(f"Packet file is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "session_id" not in data:
        raise PacketImportError("Packet file is missing required fields (session_id).")

    return data


def _history_key_metric(metrics: dict[str, Any] | None) -> str:
    """One representative number per observation module, for the continuity table."""
    if not metrics:
        return "—"
    module = metrics.get("observation_module", "arm")
    if module == "palm":
        value = metrics.get("palm_opening_ratio_change")
    elif module == "head":
        value = metrics.get("observed_head_turn_proxy_range")
    else:
        value = metrics.get("observed_2d_elbow_angle_change_degrees")
    return str(value) if value is not None else "—"


def render_clinician_html(packet: dict[str, Any], history: list[dict[str, Any]] | None = None) -> str:
    """Render a self-contained, mobile-readable HTML page for a clinician packet.

    No external assets or network calls — the file opens directly in any
    phone or desktop browser, independent of the RecContinue app itself, so
    a clinician without a computer in the room can still read the report.
    `history` is prior saved sessions for the same patient (most recent
    first), used only to show a short continuity table across visits.
    """

    def esc(value: Any) -> str:
        return html.escape(str(value)) if value is not None else ""

    def bullets(items: list[str] | None) -> str:
        if not items:
            return "<p class='muted'>None recorded.</p>"
        return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"

    patient_id = esc(packet.get("patient_id"))
    task = esc(packet.get("task"))
    review_status = esc(packet.get("review_status"))

    safety_check = packet.get("safety_check")
    safety_html = ""
    if safety_check is not None:
        if safety_check.get("passed", True):
            safety_html = "<div class='banner ok'>✅ Passed automated safety check.</div>"
        else:
            flags = "".join(
                f"<li><em>\"{esc(f.get('phrase'))}\"</em> in: \"{esc(f.get('sentence'))}\"</li>"
                for f in safety_check.get("flagged_sentences", [])
            )
            safety_html = (
                "<div class='banner warn'>⚠️ Flagged by the automated safety check — "
                f"cannot be marked reviewed until edited.<ul>{flags}</ul></div>"
            )

    report = packet.get("gemma_report")
    report_html = "<p class='muted'>PEO report not included in this packet.</p>"
    if report:
        report_html = f"""
        <p><strong>Report status:</strong> {esc(report.get('report_status'))}</p>
        <h3>Session summary</h3><p>{esc(report.get('session_summary'))}</p>
        <h3>Objective observations</h3>{bullets(report.get('objective_observations'))}
        <h3>Patient-reported information</h3>{bullets(report.get('patient_reported_information'))}
        <h3>Person factors</h3>{bullets(report.get('person_factors'))}
        <h3>Environment factors</h3>{bullets(report.get('environment_factors'))}
        <h3>Occupation factors</h3>{bullets(report.get('occupation_factors'))}
        <h3>Missing information</h3>{bullets(report.get('missing_information'))}
        <h3>Discussion ideas for clinician review</h3>{bullets(report.get('clinician_follow_up_questions'))}
        <h3>Patient-friendly recap</h3><p>{esc(report.get('patient_friendly_recap'))}</p>
        <h3>Limitations</h3>{bullets(report.get('limitations'))}
        <p class='muted'>{esc(report.get('safety_notice'))}</p>
        """

    history_html = ""
    if history:
        rows = "".join(
            "<tr>"
            f"<td>{esc((h.get('saved_at') or '')[:10])}</td>"
            f"<td>{esc(h.get('review_status'))}</td>"
            f"<td>{esc(_history_key_metric(h.get('movement_metrics')))}</td>"
            "</tr>"
            for h in history
        )
        history_html = f"""
        <h2>Recent sessions</h2>
        <p class='muted'>Prior local sessions for this patient, most recent first.</p>
        <table><thead><tr><th>Date</th><th>Review status</th><th>Key metric</th></tr></thead>
        <tbody>{rows}</tbody></table>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RecContinue — {patient_id}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 640px; margin: 0 auto; padding: 16px; line-height: 1.5; color: #1a1a1a; }}
  h1 {{ font-size: 1.3rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 1.5rem; border-top: 1px solid #ddd; padding-top: 1rem; }}
  h3 {{ font-size: 1rem; margin-bottom: 0.2rem; }}
  .muted {{ color: #666; font-size: 0.9rem; }}
  .banner {{ padding: 10px 14px; border-radius: 8px; margin: 10px 0; }}
  .banner.ok {{ background: #e6f4ea; color: #1e4620; }}
  .banner.warn {{ background: #fdecea; color: #611a15; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; }}
  ul {{ margin: 0.2rem 0; padding-left: 1.2rem; }}
</style>
</head>
<body>
  <h1>RecContinue — Clinician summary</h1>
  <p><strong>Patient:</strong> {patient_id} &nbsp; <strong>Task:</strong> {task}</p>
  <p><strong>Status:</strong> {review_status}</p>
  {safety_html}
  <h2>This session</h2>
  {report_html}
  {history_html}
  <p class="muted">Generated locally by RecContinue. Not a diagnosis or treatment recommendation.</p>
</body>
</html>
"""


def export_html(
    packet: dict[str, Any], output_dir: str | pathlib.Path, history: list[dict[str, Any]] | None = None
) -> str:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_id = packet.get("patient_id") or "session"
    output_path = output_dir / f"{patient_id}-session.html"
    output_path.write_text(render_clinician_html(packet, history))
    return str(output_path)


def export_reviewed_report(
    imported_packet: dict[str, Any], clinician_note: str, reviewed: bool, output_dir: str | pathlib.Path
) -> str:
    safety_check = imported_packet.get("safety_check")
    if reviewed and safety_check is not None and not safety_check.get("passed", True):
        raise SafetyGateError(
            "This report was flagged by the automated safety check and cannot be marked "
            "reviewed until the flagged text is edited. See the flagged sentences below."
        )

    reviewed_report = dict(imported_packet)
    reviewed_report["clinician_note"] = clinician_note
    reviewed_report["reviewed_by_clinician"] = reviewed
    reviewed_report["reviewed_at"] = datetime.now(timezone.utc).isoformat() if reviewed else None

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = imported_packet.get("session_id") or "session"
    output_path = output_dir / f"{session_id}-reviewed{PACKET_SUFFIX}"
    output_path.write_text(json.dumps(reviewed_report, indent=2))
    return str(output_path)
