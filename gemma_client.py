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
    SYSTEM_PROMPT,
    build_acknowledgment_prompt,
    build_repair_prompt,
    build_user_prompt,
)
from schemas import RecContinueReport

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e2b"
CONNECT_TIMEOUT_SECONDS = 3
GENERATE_TIMEOUT_SECONDS = 120


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
) -> str:
    request_body = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
    }
    if json_format:
        request_body["format"] = "json"
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
    response.raise_for_status()

    body = response.json()
    return body.get("response", "")


def _parse_and_validate(raw_text: str, session_id: str) -> RecContinueReport:
    json_text = _extract_json_text(raw_text)
    data: Any = json.loads(json_text)
    data.setdefault("session_id", session_id)
    return RecContinueReport.model_validate(data)


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

    On invalid/unparseable JSON, attempts exactly one repair prompt before
    raising GemmaInvalidResponseError with the original raw text preserved
    for manual review.
    """
    prompt = build_user_prompt(session_id, patient, movement_metrics, patient_context)
    raw_text = _call_ollama_generate(prompt, host, model, timeout)

    try:
        return _parse_and_validate(raw_text, session_id)
    except (json.JSONDecodeError, ValidationError, AttributeError) as first_error:
        repair_prompt = build_repair_prompt(session_id, raw_text, str(first_error))
        try:
            repaired_text = _call_ollama_generate(repair_prompt, host, model, timeout)
            return _parse_and_validate(repaired_text, session_id)
        except (json.JSONDecodeError, ValidationError, AttributeError) as second_error:
            raise GemmaInvalidResponseError(
                f"Gemma output did not match the required schema after one repair "
                f"attempt: {second_error}",
                raw_text=raw_text,
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
