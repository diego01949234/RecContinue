import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import ACKNOWLEDGMENT_SYSTEM_PROMPT, build_acknowledgment_prompt  # noqa: E402


def test_build_acknowledgment_prompt_includes_transcript_and_task():
    prompt = build_acknowledgment_prompt("My arm feels a little stiff today.", "Reach-and-place cup task")
    assert "My arm feels a little stiff today." in prompt
    assert "Reach-and-place cup task" in prompt


def test_acknowledgment_system_prompt_forbids_recommendations():
    lowered = ACKNOWLEDGMENT_SYSTEM_PROMPT.lower()
    assert "must not" in lowered
    assert "recommend" in lowered
