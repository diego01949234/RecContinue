"""Deterministic post-generation safety validation for RecContinue (Phase 5).

Scans only the report's clinical-content sections (never the safety
disclaimer itself) for unsupported clinical phrases, per SPEC.md
section 18. Never rewrites or deletes the original model output — it only
flags it and reports the exact offending sentence, so a human can edit it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from schemas import RecContinueReport

FLAGGED_PHRASES = (
    "you have",
    "the patient has",
    "diagnosed with",
    "brunnstrom stage",
    "should perform",
    "recommended exercise",
    "recommended treatment",
    "safe to continue",
    "unsafe",
    "normal movement",
    "abnormal movement",
    "neurological deficit",
    "compensation pattern",
)

STATUS_PASSED = "AI-generated draft requiring clinician review"
STATUS_REQUIRES_MANUAL_EDITING = "Requires manual editing"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


@dataclass
class FlaggedPhrase:
    phrase: str
    sentence: str


@dataclass
class SafetyValidationResult:
    passed: bool
    flagged: list[FlaggedPhrase] = field(default_factory=list)

    @property
    def status_label(self) -> str:
        return STATUS_PASSED if self.passed else STATUS_REQUIRES_MANUAL_EDITING


def _scan_sections(sections: list[str]) -> SafetyValidationResult:
    """Scan text sections using the single shared flagged-phrase list."""
    flagged: list[FlaggedPhrase] = []
    for text in sections:
        for sentence in _split_sentences(text):
            lowered = sentence.lower()
            for phrase in FLAGGED_PHRASES:
                if phrase in lowered:
                    flagged.append(FlaggedPhrase(phrase=phrase, sentence=sentence))
    return SafetyValidationResult(passed=not flagged, flagged=flagged)


def validate_report_safety(report: RecContinueReport) -> SafetyValidationResult:
    """Scan report clinical text while excluding label/disclaimer fields."""
    return _scan_sections(report.clinical_text_sections())


def validate_text_safety(text: str) -> SafetyValidationResult:
    """Scan one free-text response using the report safety rules."""
    return _scan_sections([text])
