import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stt_client  # noqa: E402


class _FakeSegment:
    def __init__(self, text):
        self.text = text


def test_transcribe_audio_joins_and_strips_segments():
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([_FakeSegment(" Hello there. "), _FakeSegment(" How are you? ")], None)
    with patch("stt_client._get_model", return_value=fake_model):
        text = stt_client.transcribe_audio("dummy.wav")
    assert text == "Hello there. How are you?"


def test_transcribe_audio_raises_when_dependency_missing():
    with patch("stt_client._FASTER_WHISPER_AVAILABLE", False):
        with pytest.raises(stt_client.WhisperUnavailableError):
            stt_client.transcribe_audio("dummy.wav")


def test_get_model_raises_when_model_load_fails():
    with patch("stt_client._FASTER_WHISPER_AVAILABLE", True), \
         patch("stt_client._model", None), \
         patch("stt_client.WhisperModel", side_effect=RuntimeError("no weights")):
        with pytest.raises(stt_client.WhisperUnavailableError):
            stt_client._get_model()
