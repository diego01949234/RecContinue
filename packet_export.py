"""Patient-controlled clinician packet export/import for RecContinue (Phase 4).

Exports a minimized, patient-approved subset of a saved session to a local
`<patient_id>-session.reccontinue.json` file. Export and import both
operate on local files only — nothing here transmits data anywhere
(SPEC.md section 14: "The hackathon prototype handles export and import
only. Network transfer is outside scope.").
"""
from __future__ import annotations

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
