"""Local speech-to-text for RecContinue's optional Tab 1 voice note.

The faster-whisper model is loaded only when a patient records audio and
performs transcription on-device. Its model download occurs once on first
use and is cached locally; this module makes no cloud inference calls.
"""
from __future__ import annotations

try:
    from faster_whisper import WhisperModel

    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    _FASTER_WHISPER_AVAILABLE = False


WHISPER_MODEL_SIZE = "base"

_model = None


class WhisperUnavailableError(Exception):
    """Raised when faster-whisper or its local model cannot be loaded."""


def _get_model():
    """Return the lazily loaded CPU/int8 Whisper model."""
    global _model
    if not _FASTER_WHISPER_AVAILABLE:
        raise WhisperUnavailableError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt`."
        )
    if _model is None:
        try:
            _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        except Exception as exc:
            raise WhisperUnavailableError(
                f"Could not load the local Whisper model ({WHISPER_MODEL_SIZE}): {exc}"
            ) from exc
    return _model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe a local audio file on-device and return its plain text."""
    model = _get_model()
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments).strip()
