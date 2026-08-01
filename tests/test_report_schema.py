import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import RecContinueReport  # noqa: E402

VALID_REPORT = {
    "session_id": "SYN-001-session",
    "report_status": "AI-generated draft requiring clinician review",
    "session_summary": "Session summary text.",
    "objective_observations": ["Observed hand height variation of 0.21."],
    "patient_reported_information": ["Patient reported needing help reaching the shelf."],
    "person_factors": ["Right arm selected for the reach-and-place task."],
    "environment_factors": ["Kitchen shelf reported above shoulder height."],
    "occupation_factors": ["Goal: independently put cups away."],
    "missing_information": ["Discomfort level not documented."],
    "clinician_follow_up_questions": ["Has assistance level been documented?"],
    "patient_friendly_recap": "Here is a plain-language summary of your session.",
    "limitations": ["Camera-based 2D measurement only."],
    "safety_notice": "AI-generated draft requiring clinician review.",
}


def test_valid_report_parses():
    report = RecContinueReport.model_validate(VALID_REPORT)
    assert report.session_id == "SYN-001-session"
    assert report.report_status == "AI-generated draft requiring clinician review"


def test_missing_required_field_raises():
    incomplete = dict(VALID_REPORT)
    del incomplete["session_summary"]
    with pytest.raises(ValidationError):
        RecContinueReport.model_validate(incomplete)


def test_wrong_type_for_list_field_raises():
    bad = dict(VALID_REPORT)
    bad["objective_observations"] = "not a list"
    with pytest.raises(ValidationError):
        RecContinueReport.model_validate(bad)


def test_default_report_status_applied_when_omitted():
    without_status = dict(VALID_REPORT)
    del without_status["report_status"]
    report = RecContinueReport.model_validate(without_status)
    assert report.report_status == "AI-generated draft requiring clinician review"


def test_clinical_text_sections_excludes_safety_notice_and_status():
    report = RecContinueReport.model_validate(VALID_REPORT)
    sections = report.clinical_text_sections()
    assert report.safety_notice not in sections
    assert report.report_status not in sections
    assert report.session_summary in sections
    assert report.patient_friendly_recap in sections
