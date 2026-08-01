"""Local JSON session vault for RecContinue (Phase 4).

Sessions are written to vault/<session_id>.json on disk. Nothing here ever
leaves the local filesystem, and only the synthetic session data
RecContinue itself generates is ever stored (SPEC.md section 10 step 6).
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any, Optional

BASE_DIR = pathlib.Path(__file__).parent
VAULT_DIR = BASE_DIR / "vault"


class SessionNotFoundError(Exception):
    """Raised when a session_id has no corresponding saved session."""


def _session_path(session_id: str) -> pathlib.Path:
    safe_id = session_id.replace("/", "_").replace("..", "_")
    return VAULT_DIR / f"{safe_id}.json"


def save_session(
    session_id: str,
    patient: dict[str, Any],
    movement_metrics: Optional[dict[str, Any]],
    patient_context: Optional[dict[str, Any]],
    gemma_report: Optional[dict[str, Any]],
    video_path: Optional[str] = None,
    review_status: str = "AI-generated draft requiring clinician review",
) -> str:
    """Save (or overwrite) a session locally and return the file path."""
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    session = {
        "session_id": session_id,
        "patient_id": patient.get("patient_id"),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "patient": patient,
        "movement_metrics": movement_metrics,
        "patient_context": patient_context,
        "gemma_report": gemma_report,
        "video_path": video_path,
        "review_status": review_status,
    }
    path = _session_path(session_id)
    path.write_text(json.dumps(session, indent=2))
    return str(path)


def load_session(session_id: str) -> dict[str, Any]:
    path = _session_path(session_id)
    if not path.exists():
        raise SessionNotFoundError(f"No saved session found for {session_id}.")
    return json.loads(path.read_text())


def list_sessions() -> list[dict[str, Any]]:
    if not VAULT_DIR.exists():
        return []
    summaries = []
    for path in sorted(VAULT_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        summaries.append({
            "session_id": data.get("session_id"),
            "patient_id": data.get("patient_id"),
            "saved_at": data.get("saved_at"),
            "review_status": data.get("review_status"),
        })
    return summaries


def delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    if not path.exists():
        return False
    path.unlink()
    return True
