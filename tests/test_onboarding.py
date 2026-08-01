import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemma_client  # noqa: E402


VALID_INTAKE_JSON = {
    "acknowledgement": "You want to put cups away with less help.",
    "daily_activity": "Putting cups away in the kitchen.",
    "difficulty_location": "Top kitchen shelf.",
    "assistance_needed": "Family member helps with the top shelf.",
    "discomfort_reported": "Arm feels tired after several reaches.",
    "independence_goal": "Put cups away independently.",
    "patient_statement": "I want to reach the shelf myself.",
    "missing_information": ["How often this happens was not provided."],
    "task_connection": "The assigned reach-and-place task can be documented alongside this goal.",
}


def _fake_generate_response(payload_text: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"response": payload_text}
    response.raise_for_status = MagicMock()
    return response


def test_generate_onboarding_intake_returns_validated_local_gemma_output():
    response = _fake_generate_response(json.dumps(VALID_INTAKE_JSON))
    with patch("gemma_client.requests.post", return_value=response) as mock_post:
        intake = gemma_client.generate_onboarding_intake(
            "I want to put cups on the high kitchen shelf but need help after a few reaches.",
            "Reach-and-place cup task",
        )

    assert intake.independence_goal == "Put cups away independently."
    assert intake.missing_information == ["How often this happens was not provided."]
    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["json"]["system"] != gemma_client.SYSTEM_PROMPT
