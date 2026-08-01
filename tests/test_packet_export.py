import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packet_export import (  # noqa: E402
    PacketImportError,
    build_clinician_packet,
    export_packet,
    export_reviewed_report,
    import_packet,
)

SAMPLE_SESSION = {
    "session_id": "SYN-001-session",
    "patient": {
        "patient_id": "SYN-001",
        "name": "Ms. Lin",
        "data_status": "Entirely synthetic",
        "task": "Reach-and-place cup task",
    },
    "movement_metrics": {"selected_arm": "right", "repetition_count": 3},
    "patient_context": {"patient_statement": "It's frustrating to need help."},
    "gemma_report": {"session_summary": "Summary text."},
    "video_path": "/local/only/session.mp4",
    "representative_frame_path": "/local/only/frame.png",
    "review_status": "AI-generated draft requiring clinician review",
}


def test_default_packet_excludes_raw_video_and_frame():
    packet = build_clinician_packet(
        SAMPLE_SESSION,
        include_metrics=True,
        include_report=True,
        include_patient_statement=True,
        include_representative_frame=False,
        include_raw_video=False,
    )
    assert "raw_video_path" not in packet
    assert "representative_frame_path" not in packet
    assert packet["movement_metrics"] == SAMPLE_SESSION["movement_metrics"]
    assert packet["gemma_report"] == SAMPLE_SESSION["gemma_report"]
    assert packet["patient_id"] == "SYN-001"


def test_packet_only_includes_opted_in_raw_video():
    packet = build_clinician_packet(
        SAMPLE_SESSION,
        include_metrics=False,
        include_report=False,
        include_patient_statement=False,
        include_representative_frame=False,
        include_raw_video=True,
    )
    assert packet["raw_video_path"] == "/local/only/session.mp4"
    assert "movement_metrics" not in packet
    assert "gemma_report" not in packet
    assert "patient_context" not in packet


def test_export_and_import_round_trip(tmp_path):
    packet = build_clinician_packet(
        SAMPLE_SESSION,
        include_metrics=True,
        include_report=True,
        include_patient_statement=True,
        include_representative_frame=False,
        include_raw_video=False,
    )
    output_path = export_packet(packet, tmp_path)
    assert output_path.endswith("SYN-001-session.reccontinue.json")

    imported = import_packet(output_path)
    assert imported["session_id"] == "SYN-001-session"
    assert imported["movement_metrics"]["repetition_count"] == 3


def test_import_packet_missing_file_raises():
    with pytest.raises(PacketImportError):
        import_packet("/nonexistent/path/session.reccontinue.json")


def test_import_packet_corrupt_json_raises(tmp_path):
    bad_file = tmp_path / "bad.reccontinue.json"
    bad_file.write_text("{not valid json")
    with pytest.raises(PacketImportError):
        import_packet(bad_file)


def test_import_packet_missing_session_id_raises(tmp_path):
    bad_file = tmp_path / "bad.reccontinue.json"
    bad_file.write_text(json.dumps({"patient_id": "SYN-001"}))
    with pytest.raises(PacketImportError):
        import_packet(bad_file)


def test_export_reviewed_report_round_trip(tmp_path):
    packet = build_clinician_packet(
        SAMPLE_SESSION, include_metrics=True, include_report=True,
        include_patient_statement=True, include_representative_frame=False, include_raw_video=False,
    )
    reviewed_path = export_reviewed_report(packet, "Discussed with patient.", True, tmp_path)
    reviewed = json.loads(Path(reviewed_path).read_text())
    assert reviewed["clinician_note"] == "Discussed with patient."
    assert reviewed["reviewed_by_clinician"] is True
    assert reviewed["reviewed_at"] is not None


def test_export_reviewed_report_not_reviewed_has_no_timestamp(tmp_path):
    packet = build_clinician_packet(
        SAMPLE_SESSION, include_metrics=True, include_report=True,
        include_patient_statement=True, include_representative_frame=False, include_raw_video=False,
    )
    reviewed_path = export_reviewed_report(packet, "", False, tmp_path)
    reviewed = json.loads(Path(reviewed_path).read_text())
    assert reviewed["reviewed_by_clinician"] is False
    assert reviewed["reviewed_at"] is None
