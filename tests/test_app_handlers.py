import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

gradio_stub = types.ModuleType("gradio")
gradio_stub.SelectData = object
gradio_stub.update = lambda **kwargs: kwargs
sys.modules.setdefault("gradio", gradio_stub)

import app  # noqa: E402


def _fake_analyze_video(video_path, selected_arm):
    yield {"frame_index": 5, "total_frames": 12, "current_angle": 96.0, "reps_so_far": 1}
    yield "annotated.mp4", {"repetition_count": 1, "selected_arm": selected_arm}


def test_analyze_video_handler_yields_progress_then_result():
    with patch("app.analyze_video", side_effect=_fake_analyze_video):
        updates = list(app.analyze_video_handler("session.mp4", "right"))

    assert len(updates) == 2
    progress_update = updates[0]
    assert "frame 5/12" in progress_update[3]
    assert "96.0" in progress_update[3]
    assert "reps so far: 1" in progress_update[3]

    final_update = updates[1]
    assert final_update[0] == "annotated.mp4"
    assert final_update[2] == {"repetition_count": 1, "selected_arm": "right"}
    assert final_update[3] == ""


def test_analyze_video_handler_no_video_selected():
    updates = list(app.analyze_video_handler(None, "right"))
    assert len(updates) == 1
    assert updates[0][0] is None
    assert "No video selected" in updates[0][1]


def test_analyze_video_handler_reports_unavailable_error():
    def _raises(video_path, selected_arm):
        raise app.MovementAnalysisUnavailableError("mediapipe not installed")
        yield  # pragma: no cover

    with patch("app.analyze_video", side_effect=_raises):
        updates = list(app.analyze_video_handler("session.mp4", "right"))

    assert len(updates) == 1
    assert updates[0][0] is None
    assert "mediapipe not installed" in updates[0][1]


def test_use_synthetic_metrics_fallback_clears_progress_text():
    metrics, markdown, progress = app.use_synthetic_metrics_fallback()
    assert metrics == app.SYNTHETIC_METRICS
    assert progress == ""


def test_voice_frontdoor_handler_no_audio_is_a_noop():
    transcript_md, acknowledgment_md, statement_update = app.voice_frontdoor_handler(None)
    assert transcript_md == ""
    assert acknowledgment_md == ""
    assert "value" not in statement_update


def test_voice_frontdoor_handler_transcribes_and_acknowledges():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), patch(
        "app.generate_acknowledgment",
        return_value="Thanks for sharing. The assigned reach-and-place cup task remains unchanged.",
    ):
        transcript_md, acknowledgment_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "My arm feels stiff today." in transcript_md
    assert "reach-and-place cup task" in acknowledgment_md.lower()
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_suppresses_flagged_acknowledgment():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), patch(
        "app.generate_acknowledgment", return_value="You have a diagnosed rotator cuff injury."
    ):
        transcript_md, acknowledgment_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert acknowledgment_md == ""
    assert "My arm feels stiff today." in transcript_md
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_whisper_unavailable_does_not_crash():
    with patch("app.transcribe_audio", side_effect=app.WhisperUnavailableError("faster-whisper not installed")):
        transcript_md, acknowledgment_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "faster-whisper not installed" in transcript_md
    assert acknowledgment_md == ""
    assert "value" not in statement_update


def test_voice_frontdoor_handler_gemma_unavailable_still_prefills_statement():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), patch(
        "app.generate_acknowledgment", side_effect=app.OllamaUnavailableError("no ollama")
    ):
        transcript_md, acknowledgment_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert acknowledgment_md == ""
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_no_speech_detected():
    with patch("app.transcribe_audio", return_value="   "):
        transcript_md, acknowledgment_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "No speech detected" in transcript_md
    assert acknowledgment_md == ""
    assert "value" not in statement_update


def test_voice_frontdoor_handler_no_audio_is_a_noop():
    transcript_md, ack_md, statement_update = app.voice_frontdoor_handler(None)
    assert transcript_md == ""
    assert ack_md == ""
    assert "value" not in statement_update


def test_voice_frontdoor_handler_transcribes_and_acknowledges():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), \
         patch(
             "app.generate_acknowledgment",
             return_value="Thanks for sharing. Today's assigned activity is the reach-and-place cup task.",
         ):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "My arm feels stiff today." in transcript_md
    assert "reach-and-place cup task" in ack_md.lower()
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_suppresses_flagged_acknowledgment():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), \
         patch("app.generate_acknowledgment", return_value="You have a diagnosed rotator cuff injury."):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert ack_md == ""
    assert "My arm feels stiff today." in transcript_md
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_whisper_unavailable_does_not_crash():
    with patch("app.transcribe_audio", side_effect=app.WhisperUnavailableError("faster-whisper not installed")):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "faster-whisper not installed" in transcript_md
    assert ack_md == ""
    assert "value" not in statement_update


def test_voice_frontdoor_handler_gemma_unavailable_still_prefills_statement():
    with patch("app.transcribe_audio", return_value="My arm feels stiff today."), \
         patch("app.generate_acknowledgment", side_effect=app.OllamaUnavailableError("no ollama")):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert ack_md == ""
    assert statement_update["value"] == "My arm feels stiff today."


def test_voice_frontdoor_handler_no_speech_detected():
    with patch("app.transcribe_audio", return_value="   "):
        transcript_md, ack_md, statement_update = app.voice_frontdoor_handler("voice.wav")

    assert "No speech detected" in transcript_md
    assert ack_md == ""
    assert "value" not in statement_update
