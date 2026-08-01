"""Local Ollama/Gemma 4 client for RecContinue.

Everything here talks only to a local Ollama daemon (default
http://localhost:11434). Nothing in this module ever performs a network
call to a non-localhost host.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests
from pydantic import ValidationError

from prompts import (
    ACKNOWLEDGMENT_SYSTEM_PROMPT,
    ONBOARDING_SYSTEM_PROMPT,
    OBSERVATION_RECOMMENDATION_SYSTEM_PROMPT,
    QUICK_SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_acknowledgment_prompt,
    build_onboarding_prompt,
    build_onboarding_repair_prompt,
    build_observation_recommendation_prompt,
    build_quick_summary_prompt,
    build_repair_prompt,
    build_user_prompt,
)
from schemas import ObservationRecommendation, OnboardingIntake, RecContinueReport

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e2b"
CONNECT_TIMEOUT_SECONDS = 3
# gemma4:e2b decodes at ~10 tokens/sec on the hackathon dev machine (M1,
# no GPU headroom beyond that). A full 12-field prose report used to take
# ~80-110s of eval time alone. generate_report now only asks Gemma for 6
# short (2-5 word) fields and caps output at REPORT_NUM_PREDICT tokens,
# which keeps real generations to ~9-10s; 240s remains as a generous
# ceiling for cold-load or repair-prompt edge cases, not the expected case.
GENERATE_TIMEOUT_SECONDS = 120
# Empirically, 6 fields x 2-5 words fits well under 100 tokens; this cap is
# a safety net against runaway generation, not the normal stopping point.
REPORT_NUM_PREDICT = 220

FIXED_SAFETY_NOTICE = (
    "AI-generated draft requiring clinician review. This report does not "
    "diagnose, assess severity, or recommend treatment."
)
FIXED_LIMITATIONS = [
    "Camera-based 2D movement measurement is an estimate, not a validated "
    "clinical measurement.",
    "Accuracy can be affected by lighting, camera angle, and occlusion.",
]
FIXED_FOLLOW_UP_QUESTIONS = [
    "Please review the objective measurements and patient-reported "
    "information above for any follow-up questions.",
]
_MISSING_CONTEXT_LABELS = {
    "assistance_needed": "Assistance needed",
    "discomfort_reported": "Discomfort reported",
    "difficulty_location": "Difficulty location",
    "independence_goal": "Independence goal",
}


class OllamaUnavailableError(Exception):
    """Raised when the local Ollama daemon cannot be reached at all."""


class GemmaModelNotInstalledError(Exception):
    """Raised when Ollama is reachable but the requested model is not pulled."""


class GemmaTimeoutError(Exception):
    """Raised when a generate call exceeds GENERATE_TIMEOUT_SECONDS."""


class GemmaInvalidResponseError(Exception):
    """Raised when Gemma's output could not be turned into a valid report,
    even after one repair attempt. Carries the raw text for manual review.
    """

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class OllamaStatus:
    reachable: bool
    model_installed: bool
    message: str


def check_ollama_connection(
    host: str = DEFAULT_OLLAMA_HOST, model: str = DEFAULT_MODEL
) -> OllamaStatus:
    """Check whether the local Ollama daemon is running and the model is pulled.

    Never raises: any connectivity problem is folded into the returned
    OllamaStatus so the UI can render exact setup instructions instead of
    crashing.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=CONNECT_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        return OllamaStatus(
            reachable=False,
            model_installed=False,
            message=(
                "Cannot reach Ollama at "
                f"{host}. Start it with `ollama serve`, then pull the model "
                f"with `ollama pull {model}`. ({exc.__class__.__name__})"
            ),
        )

    try:
        tags = response.json().get("models", [])
    except (ValueError, AttributeError):
        tags = []

    installed_names = {entry.get("name", "") for entry in tags}
    model_installed = any(
        name == model or name.startswith(f"{model}:") or name.split(":")[0] == model.split(":")[0]
        for name in installed_names
    )

    if not model_installed:
        return OllamaStatus(
            reachable=True,
            model_installed=False,
            message=f"Ollama is running, but `{model}` is not installed. Run `ollama pull {model}`.",
        )

    return OllamaStatus(
        reachable=True,
        model_installed=True,
        message=f"Gemma running locally ({model}) via Ollama at {host}.",
    )


def _extract_json_text(raw_text: str) -> str:
    """Strip markdown code fences if the model wrapped its JSON output in them."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]
    return text


def _call_ollama_generate(
    prompt: str,
    host: str,
    model: str,
    timeout: int,
    system: str = SYSTEM_PROMPT,
    json_format: bool = True,
    num_predict: int | None = None,
) -> str:
    request_body = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        # Gemma 4 supports a separate reasoning channel. RecContinue needs a
        # concise, schema-constrained documentation draft, so suppress it to
        # avoid making a local report request wait behind unnecessary thinking.
        "think": False,
        "keep_alive": "5m",
    }
    if json_format:
        request_body["format"] = "json"
    if num_predict is not None:
        request_body["options"] = {"num_predict": num_predict}
    try:
        response = requests.post(
            f"{host}/api/generate",
            json=request_body,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise GemmaTimeoutError(f"Gemma did not respond within {timeout}s.") from exc
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {host}. Is `ollama serve` running? ({exc.__class__.__name__})"
        ) from exc

    if response.status_code == 404:
        raise GemmaModelNotInstalledError(
            f"Model `{model}` is not installed. Run `ollama pull {model}`."
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        # Ollama's own llama-server occasionally returns 5xx on long
        # generations instead of just being slow (observed in
        # ~/.ollama/logs/server.log). Surface this the same way as an
        # unreachable daemon so the UI shows a clear, retryable message
        # instead of an unhandled exception.
        raise OllamaUnavailableError(
            f"Ollama returned an error (HTTP {response.status_code}) while generating. "
            "This can happen when a generation runs long. Please try again."
        ) from exc

    body = response.json()
    return body.get("response", "")


def _compute_missing_information(patient_context: dict[str, Any]) -> list[str]:
    """Flag context fields the patient left blank, without asking Gemma to notice it.

    This was previously part of what Gemma generated per report; computing it
    directly from the same locally held dict Gemma would have read is both
    faster (no extra tokens) and more reliable (no risk of the model missing
    or inventing a gap).
    """
    missing = []
    for key, label in _MISSING_CONTEXT_LABELS.items():
        value = str(patient_context.get(key, "")).strip()
        if not value or value.lower() == "not documented":
            missing.append(f"{label} not documented.")
    return missing


def _fill_deterministic_fields(data: dict[str, Any], patient_context: dict[str, Any]) -> None:
    """Populate the report fields RecContinue no longer asks Gemma to write.

    These are disclaimer/boilerplate text or directly computable from local
    data (see REPORT_DETERMINISTIC_FIELDS in prompts.py) — spending model
    generation time on them added latency without adding real variation.
    """
    data["safety_notice"] = FIXED_SAFETY_NOTICE
    data["limitations"] = list(FIXED_LIMITATIONS)
    data["clinician_follow_up_questions"] = list(FIXED_FOLLOW_UP_QUESTIONS)
    data["missing_information"] = _compute_missing_information(patient_context)
    data.setdefault("patient_friendly_recap", data.get("session_summary", ""))


def _parse_and_validate(
    raw_text: str, session_id: str, patient_context: dict[str, Any]
) -> RecContinueReport:
    json_text = _extract_json_text(raw_text)
    data: Any = json.loads(json_text)
    data.setdefault("session_id", session_id)
    _fill_deterministic_fields(data, patient_context)
    return RecContinueReport.model_validate(data)


def _parse_and_validate_intake(raw_text: str) -> OnboardingIntake:
    data: Any = json.loads(_extract_json_text(raw_text))
    return OnboardingIntake.model_validate(data)


def _parse_and_validate_observation_recommendation(raw_text: str) -> ObservationRecommendation:
    data: Any = json.loads(_extract_json_text(raw_text))
    return ObservationRecommendation.model_validate(data)


def recommend_observation_module(
    patient_words: str,
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
) -> ObservationRecommendation:
    """Choose one existing camera observation view from a patient's own words.

    The constrained schema prevents the model from creating a new activity or
    prescribing rehabilitation; callers show this only as a selectable UI hint.
    """
    raw_text = _call_ollama_generate(
        build_observation_recommendation_prompt(patient_words),
        host,
        model,
        timeout,
        system=OBSERVATION_RECOMMENDATION_SYSTEM_PROMPT,
    )
    try:
        return _parse_and_validate_observation_recommendation(raw_text)
    except (json.JSONDecodeError, ValidationError, AttributeError) as exc:
        raise GemmaInvalidResponseError(
            f"Gemma observation suggestion did not match the required schema: {exc}",
            raw_text=raw_text,
        ) from exc


def generate_onboarding_intake(
    patient_words: str,
    assigned_task: str,
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
) -> OnboardingIntake:
    """Organize a patient's free-text intake locally without changing their task.

    As with report generation, exactly one repair attempt is allowed when Gemma
    does not return JSON that matches the constrained intake schema.
    """
    prompt = build_onboarding_prompt(patient_words, assigned_task)
    raw_text = _call_ollama_generate(
        prompt, host, model, timeout, system=ONBOARDING_SYSTEM_PROMPT
    )
    try:
        return _parse_and_validate_intake(raw_text)
    except (json.JSONDecodeError, ValidationError, AttributeError) as first_error:
        repair_prompt = build_onboarding_repair_prompt(raw_text, str(first_error))
        try:
            repaired_text = _call_ollama_generate(
                repair_prompt, host, model, timeout, system=ONBOARDING_SYSTEM_PROMPT
            )
            return _parse_and_validate_intake(repaired_text)
        except (json.JSONDecodeError, ValidationError, AttributeError) as second_error:
            raise GemmaInvalidResponseError(
                f"Gemma intake output did not match the required schema after one repair "
                f"attempt: {second_error}",
                raw_text=raw_text,
            ) from second_error


def generate_report(
    session_id: str,
    patient: dict[str, Any],
    movement_metrics: dict[str, Any],
    patient_context: dict[str, Any],
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
) -> RecContinueReport:
    """Generate and validate a RecContinueReport from locally held session data.

    Uses one local repair attempt if Gemma returns invalid JSON.
    """
    prompt = build_user_prompt(patient, movement_metrics, patient_context)
    raw_text = _call_ollama_generate(
        prompt, host, model, timeout, num_predict=REPORT_NUM_PREDICT
    )

    try:
        return _parse_and_validate(raw_text, session_id, patient_context)
    except (json.JSONDecodeError, ValidationError, AttributeError) as first_error:
        repair_prompt = build_repair_prompt(session_id, raw_text, str(first_error))
        try:
            repaired_text = _call_ollama_generate(
                repair_prompt, host, model, timeout, num_predict=REPORT_NUM_PREDICT
            )
            return _parse_and_validate(repaired_text, session_id, patient_context)
        except (json.JSONDecodeError, ValidationError, AttributeError) as second_error:
            raise GemmaInvalidResponseError(
                f"Gemma output did not match the required schema after one repair "
                f"attempt: {second_error}", raw_text=raw_text,
            ) from second_error


def generate_acknowledgment(
    transcript: str,
    assigned_task: str,
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
) -> str:
    """Generate a short, plain-text acknowledgment for the Tab 1 voice note.

    Callers must run the returned text through ``validate_text_safety`` before
    displaying it. This client deliberately only communicates with the local
    Ollama endpoint.
    """
    prompt = build_acknowledgment_prompt(transcript, assigned_task)
    raw_text = _call_ollama_generate(
        prompt,
        host,
        model,
        timeout,
        system=ACKNOWLEDGMENT_SYSTEM_PROMPT,
        json_format=False,
    )
    return raw_text.strip()


def generate_quick_summary(
    age: int,
    assigned_task: str | None,
    host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_MODEL,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
) -> str:
    """Generate a short, plain-text personalized welcome for the Tab 1 age input.

    Callers must run the returned text through ``validate_text_safety`` before
    displaying it, same as ``generate_acknowledgment``. No JSON schema and no
    movement data are involved, so this stays fast enough to run immediately
    after the patient enters their age, before any recording happens.
    """
    prompt = build_quick_summary_prompt(age, assigned_task)
    raw_text = _call_ollama_generate(
        prompt,
        host,
        model,
        timeout,
        system=QUICK_SUMMARY_SYSTEM_PROMPT,
        json_format=False,
    )
    return raw_text.strip()


if __name__ == "__main__":
    # Phase 1 spike: prove the local Gemma round trip works end to end
    # before any UI is built, per SPEC.md Phase 1.
    import pathlib

    base_dir = pathlib.Path(__file__).parent
    patient = json.loads((base_dir / "sample_data" / "synthetic_patient.json").read_text())
    movement_metrics = json.loads((base_dir / "sample_data" / "synthetic_metrics.json").read_text())
    patient_context = {
        "daily_activity_goal": patient.get("patient_goal", ""),
        "difficulty_location": patient.get("home_context", ""),
        "assistance_needed": "Not documented",
        "discomfort_reported": "Not documented",
        "independence_goal": patient.get("patient_goal", ""),
        "patient_statement": "I want to put cups back on the kitchen shelf without asking for help.",
    }

    status = check_ollama_connection()
    print(f"[ollama status] {status.message}")

    if not status.reachable or not status.model_installed:
        print("Skipping live generation call; see message above for setup steps.")
    else:
        try:
            report = generate_report("SYN-001-spike", patient, movement_metrics, patient_context)
            print("[gemma report] valid JSON received and validated:")
            print(report.model_dump_json(indent=2))
        except (GemmaTimeoutError, GemmaInvalidResponseError, OllamaUnavailableError, GemmaModelNotInstalledError) as exc:
            print(f"[gemma error] {exc}")
