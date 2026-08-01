import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemma_client  # noqa: E402

VALID_REPORT_JSON = {
    "session_id": "SYN-001-session",
    "report_status": "AI-generated draft requiring clinician review",
    "session_summary": "Session summary text.",
    "objective_observations": ["Observed hand height variation of 0.21."],
    "patient_reported_information": ["Patient reported needing help reaching the shelf."],
    "person_factors": ["Right arm selected."],
    "environment_factors": ["Kitchen shelf above shoulder height."],
    "occupation_factors": ["Goal: independently put cups away."],
    "missing_information": ["Discomfort level not documented."],
    "clinician_follow_up_questions": ["Has assistance level been documented?"],
    "patient_friendly_recap": "Plain-language summary.",
    "limitations": ["Camera-based 2D measurement only."],
    "safety_notice": "AI-generated draft requiring clinician review.",
}


def _fake_generate_response(payload_text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"response": payload_text}
    response.raise_for_status = MagicMock()
    return response


def test_extract_json_text_strips_markdown_fence():
    raw = "```json\n" + json.dumps(VALID_REPORT_JSON) + "\n```"
    extracted = gemma_client._extract_json_text(raw)
    assert json.loads(extracted) == VALID_REPORT_JSON


def test_extract_json_text_handles_bare_json():
    raw = json.dumps(VALID_REPORT_JSON)
    extracted = gemma_client._extract_json_text(raw)
    assert json.loads(extracted) == VALID_REPORT_JSON


def test_check_ollama_connection_unreachable():
    with patch("gemma_client.requests.get", side_effect=requests.ConnectionError("refused")):
        status = gemma_client.check_ollama_connection()
    assert status.reachable is False
    assert status.model_installed is False
    assert "ollama serve" in status.message


def test_check_ollama_connection_model_missing():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"models": [{"name": "llama3:8b"}]}
    with patch("gemma_client.requests.get", return_value=response):
        status = gemma_client.check_ollama_connection(model="gemma4:e2b")
    assert status.reachable is True
    assert status.model_installed is False


def test_check_ollama_connection_ok():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"models": [{"name": "gemma4:e2b"}]}
    with patch("gemma_client.requests.get", return_value=response):
        status = gemma_client.check_ollama_connection(model="gemma4:e2b")
    assert status.reachable is True
    assert status.model_installed is True


def test_generate_report_success_on_first_response():
    fake_response = _fake_generate_response(json.dumps(VALID_REPORT_JSON))
    with patch("gemma_client.requests.post", return_value=fake_response) as mock_post:
        report = gemma_client.generate_report(
            "SYN-001-session", {"patient_id": "SYN-001"}, {"repetition_count": 3}, {"patient_statement": "hi"}
        )
    assert report.session_id == "SYN-001-session"
    assert mock_post.call_count == 1


def test_generate_report_repairs_invalid_json_once():
    bad_response = _fake_generate_response("not valid json at all")
    good_response = _fake_generate_response(json.dumps(VALID_REPORT_JSON))
    with patch("gemma_client.requests.post", side_effect=[bad_response, good_response]) as mock_post:
        report = gemma_client.generate_report(
            "SYN-001-session", {"patient_id": "SYN-001"}, {"repetition_count": 3}, {"patient_statement": "hi"}
        )
    assert report.session_id == "SYN-001-session"
    assert mock_post.call_count == 2


def test_generate_report_raises_after_repair_also_fails():
    bad_response = _fake_generate_response("still not json")
    with patch("gemma_client.requests.post", side_effect=[bad_response, bad_response]) as mock_post:
        with pytest.raises(gemma_client.GemmaInvalidResponseError) as excinfo:
            gemma_client.generate_report(
                "SYN-001-session", {"patient_id": "SYN-001"}, {"repetition_count": 3}, {"patient_statement": "hi"}
            )
    assert mock_post.call_count == 2
    assert excinfo.value.raw_text == "still not json"


def test_generate_report_raises_timeout_error():
    with patch("gemma_client.requests.post", side_effect=requests.Timeout("timed out")):
        with pytest.raises(gemma_client.GemmaTimeoutError):
            gemma_client.generate_report(
                "SYN-001-session", {"patient_id": "SYN-001"}, {"repetition_count": 3}, {"patient_statement": "hi"}
            )


def test_generate_report_raises_unavailable_error():
    with patch("gemma_client.requests.post", side_effect=requests.ConnectionError("refused")):
        with pytest.raises(gemma_client.OllamaUnavailableError):
            gemma_client.generate_report(
                "SYN-001-session", {"patient_id": "SYN-001"}, {"repetition_count": 3}, {"patient_statement": "hi"}
            )
