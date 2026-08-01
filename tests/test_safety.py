import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety import STATUS_PASSED, STATUS_REQUIRES_MANUAL_EDITING, validate_report_safety, validate_text_safety  # noqa: E402
from schemas import RecContinueReport  # noqa: E402

CLEAN_REPORT = {
    "session_id": "SYN-001-session",
    "report_status": "AI-generated draft requiring clinician review",
    "session_summary": "The session included a reach-and-place task with the right arm.",
    "objective_observations": ["Observed hand height variation of 0.21 relative to shoulder width."],
    "patient_reported_information": ["Patient reported needing help reaching the shelf."],
    "person_factors": ["Right arm selected for the reach-and-place task."],
    "environment_factors": ["Kitchen shelf reported above shoulder height."],
    "occupation_factors": ["Goal: independently put cups away."],
    "missing_information": ["Discomfort level not documented."],
    "clinician_follow_up_questions": ["Has assistance level been documented?"],
    "patient_friendly_recap": "Here is a plain-language summary of your session.",
    "limitations": ["Camera-based 2D measurement only, not validated clinical measurement."],
    "safety_notice": "AI-generated draft requiring clinician review.",
}


def _report_with(**overrides) -> RecContinueReport:
    data = dict(CLEAN_REPORT)
    data.update(overrides)
    return RecContinueReport.model_validate(data)


def test_clean_report_passes():
    result = validate_report_safety(_report_with())
    assert result.passed is True
    assert result.flagged == []
    assert result.status_label == STATUS_PASSED


def test_flags_diagnostic_claim():
    result = validate_report_safety(
        _report_with(session_summary="The patient has a moderate reaching deficit.")
    )
    assert result.passed is False
    assert result.status_label == STATUS_REQUIRES_MANUAL_EDITING
    assert any(f.phrase == "the patient has" for f in result.flagged)


def test_flags_brunnstrom_stage():
    result = validate_report_safety(
        _report_with(person_factors=["Consistent with Brunnstrom stage 4."])
    )
    assert result.passed is False
    assert any(f.phrase == "brunnstrom stage" for f in result.flagged)


def test_flags_recommended_exercise():
    result = validate_report_safety(
        _report_with(clinician_follow_up_questions=["Recommended exercise: overhead reaches daily."])
    )
    assert result.passed is False
    assert any(f.phrase == "recommended exercise" for f in result.flagged)


def test_flags_unsafe_and_normal_movement():
    result = validate_report_safety(
        _report_with(
            objective_observations=[
                "This was normal movement throughout.",
                "It is unsafe for the patient to continue unsupervised.",
            ]
        )
    )
    phrases = {f.phrase for f in result.flagged}
    assert "normal movement" in phrases
    assert "unsafe" in phrases


def test_case_insensitive_matching():
    result = validate_report_safety(
        _report_with(limitations=["DIAGNOSED WITH a rotator cuff issue previously."])
    )
    assert result.passed is False


def test_exact_sentence_is_isolated_not_whole_paragraph():
    result = validate_report_safety(
        _report_with(
            session_summary=(
                "The session went smoothly overall. The patient has a reaching limitation. "
                "No further issues were observed."
            )
        )
    )
    assert len(result.flagged) == 1
    assert result.flagged[0].sentence == "The patient has a reaching limitation."


def test_safety_notice_field_is_never_scanned():
    # "unsafe" appears only in the disclaimer field, which must be excluded.
    result = validate_report_safety(
        _report_with(safety_notice="It would be unsafe to treat this as a diagnosis; clinician review required.")
    )
    assert result.passed is True


def test_report_status_field_is_never_scanned():
    result = validate_report_safety(
        _report_with(report_status="Recommended treatment pending clinician review")
    )
    assert result.passed is True


def test_does_not_mutate_the_report():
    report = _report_with(session_summary="The patient has a reaching limitation.")
    original = report.model_copy(deep=True)
    validate_report_safety(report)
    assert report == original


def test_validate_text_safety_passes_clean_text():
    result = validate_text_safety("Thanks for sharing that. Today's reach-and-place task remains assigned by your therapist.")
    assert result.passed is True
    assert result.flagged == []


def test_validate_text_safety_flags_recommended_exercise():
    result = validate_text_safety("You should perform additional overhead reaches to help with that.")
    assert result.passed is False
    assert any(f.phrase == "should perform" for f in result.flagged)


def test_validate_text_safety_uses_same_scan_as_report_safety():
    result = validate_text_safety("This is unsafe to continue without supervision.")
    assert result.passed is False
    assert any(f.phrase == "unsafe" for f in result.flagged)
