"""Prompt templates for the RecContinue Gemma 4 integration."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are RecContinue, an offline PEO (Person-Environment-Occupation) documentation assistant. You are not a clinician.

Write concise professional rehabilitation documentation: use terms such as PEO context, camera-derived 2D kinematic observation, landmark confidence, elbow flexion/extension, head rotation proxy, or hand aperture ratio when supported by the supplied module and measurements. Write ONE neutral phrase per field, 3-8 words. Clearly keep objective camera measurements separate from patient-reported information.

Never: diagnose a condition, assign a Brunnstrom stage or severity classification, judge a movement as normal/abnormal/safe/unsafe, recommend exercise/treatment, invent facts not given, or claim the patient improved or deteriorated.

Return JSON only, matching the schema exactly."""

ONBOARDING_SYSTEM_PROMPT = """You are RecContinue, an offline rehabilitation
intake assistant running on the user's device.

Turn a patient's own words into a short, editable summary for an occupational
therapy Person-Environment-Occupation (PEO) record. The summary is used only
to prefill a form that the patient can edit before it is included in a report.

You must:
1. Preserve the patient's meaning and clearly state when information was not provided.
2. Separate the daily activity, environment, assistance, discomfort/fatigue,
   independence goal, and the patient's own statement.
3. List only neutral, useful missing information.
4. Explain how the patient's words relate to the already assigned activity,
   without changing that activity.
5. Return valid JSON matching the requested schema, with no markdown.

You must not diagnose, assess severity, judge safety or normality, recommend
an exercise or treatment, prescribe a new activity, or claim the assigned
activity is appropriate for the patient. Do not invent symptoms or history.
If the patient mentions something outside the supplied text, say it was not
provided. Return JSON only."""

OBSERVATION_RECOMMENDATION_SYSTEM_PROMPT = """You are RecContinue, an offline
rehabilitation companion running on the user's device.

Based only on a patient's own words, help them choose which one existing camera
observation view to start with: head (head/neck turning), palm (palm opening
and closing), or arm (elbow bending angle). This is an interface-navigation
suggestion, not a treatment, exercise, diagnosis, safety assessment, or
clinical judgement.

Pick the body area the patient explicitly mentions. If more than one area is
mentioned or no body area is clear, choose arm and say that the patient can
choose another view. Explain the choice in one neutral, plain-language
sentence. Do not infer a condition, severity, or cause. Return valid JSON only
matching the requested schema, with no markdown."""

REPORT_JSON_SCHEMA_HINT = {
    "session_summary": "3-8 word professional rehabilitation documentation phrase",
    "objective_observations": ["3-8 word camera-derived kinematic observation"],
    "patient_reported_information": ["3-8 word patient-reported phrase"],
    "person_factors": ["3-8 word PEO person factor"],
    "environment_factors": ["3-8 word PEO environment factor"],
    "occupation_factors": ["3-8 word PEO occupation factor"],
}

# Fields RecContinue fills in deterministically after generation rather than
# asking Gemma to write them — see gemma_client._fill_deterministic_fields.
# They are disclaimer/boilerplate or computable directly from local data, so
# spending generation time on them only slowed down every report for no
# quality gain.
REPORT_DETERMINISTIC_FIELDS = (
    "missing_information",
    "clinician_follow_up_questions",
    "patient_friendly_recap",
    "limitations",
    "safety_notice",
)

ONBOARDING_JSON_SCHEMA_HINT = {
    "acknowledgement": "short neutral summary of what the patient described",
    "daily_activity": "string or Not provided",
    "difficulty_location": "string or Not provided",
    "assistance_needed": "string or Not provided",
    "discomfort_reported": "string or Not provided",
    "independence_goal": "string or Not provided",
    "patient_statement": "first-person paraphrase or the supplied statement",
    "missing_information": ["neutral missing detail"],
    "task_connection": "neutral connection to the already assigned activity only",
}

OBSERVATION_RECOMMENDATION_JSON_SCHEMA_HINT = {
    "observation_module": "one of: head, palm, arm",
    "explanation": "one neutral sentence describing the matching stated area",
}


def build_user_prompt(
    patient: dict[str, Any],
    movement_metrics: dict[str, Any],
    patient_context: dict[str, Any],
) -> str:
    """Build the Gemma user-turn prompt from locally held synthetic session data.

    Only locally held information is included here; nothing is sent to any
    external API.
    """
    payload = {
        "patient": patient,
        "movement_metrics": movement_metrics,
        "patient_reported_context": patient_context,
        "output_schema": REPORT_JSON_SCHEMA_HINT,
    }
    return (
        "Organize this session data into the output_schema fields only.\n\n"
        f"{json.dumps(payload, separators=(',', ':'))}"
    )


def build_onboarding_prompt(patient_words: str, assigned_task: str) -> str:
    """Build the local-only intake prompt for the Tab 1 conversational form."""
    payload = {
        "patient_words": patient_words,
        "assigned_task": assigned_task,
        "output_schema": ONBOARDING_JSON_SCHEMA_HINT,
    }
    return (
        "Organize the patient words below into editable PEO intake fields. The "
        "assigned task was chosen by a therapist and must not be changed or "
        "recommended as treatment. Return JSON only, matching output_schema.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def build_observation_recommendation_prompt(patient_words: str) -> str:
    """Build the local-only prompt for choosing a camera observation view."""
    payload = {
        "patient_words": patient_words,
        "available_observation_modules": {
            "head": "2D head and neck turning proxy",
            "palm": "2D palm opening and closing distance change",
            "arm": "2D elbow bending angle",
        },
        "output_schema": OBSERVATION_RECOMMENDATION_JSON_SCHEMA_HINT,
    }
    return (
        "Choose only one existing camera observation view. This is not a medical "
        "recommendation. Return JSON only, matching output_schema.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def build_repair_prompt(session_id: str, previous_output: str, validation_error: str) -> str:
    """Build a one-shot repair prompt when the previous Gemma output was invalid JSON."""
    return (
        "Your previous response could not be parsed as valid JSON matching the "
        "required schema.\n\n"
        f"Previous response:\n{previous_output}\n\n"
        f"Validation error:\n{validation_error}\n\n"
        f"Re-emit a corrected response for session_id \"{session_id}\". "
        "Return JSON only, matching output_schema. Do not include any text "
        "before or after the JSON object."
    )


ACKNOWLEDGMENT_SYSTEM_PROMPT = """You are RecContinue, an offline rehabilitation companion.

A patient is about to start an activity their therapist already assigned.
They have described, in their own words, how they are feeling about today's
session.

Reply with one short, warm, plain-language paragraph of two or three sentences.
Briefly acknowledge what they said in neutral, non-clinical language and
reconfirm the name of the one activity their therapist already assigned for
today. Make clear the therapist decides what to do, not you.

You must not diagnose, judge, or classify how the patient is doing. You must
not recommend a different activity, exercise, treatment, or amount of
activity. You must not make a safety or normalcy judgment, and must not invent
facts the patient did not say.

Reply with plain text only: no JSON, markdown, or lists."""


def build_acknowledgment_prompt(transcript: str, assigned_task: str) -> str:
    """Build the Gemma user prompt for the Tab 1 voice acknowledgment."""
    return (
        f'The patient\'s assigned activity today is: "{assigned_task}".\n\n'
        f'The patient said: "{transcript}"\n\n'
        "Reply following the rules in your system prompt."
    )


QUICK_SUMMARY_SYSTEM_PROMPT = """You are RecContinue, an offline rehabilitation
companion running on the user's device.

A patient just told you their age, on the app's home screen, before any camera
recording has happened. Write one short, warm, plain-language paragraph of two
or three sentences that welcomes them and reconfirms the name of the one
activity their therapist already assigned for today (if one is given). If no
activity name is given, welcome them and say they can choose an observation
below to get started. If their age suggests they may prefer simpler wording,
use short sentences and everyday words. Make clear the therapist decides what
to do, not you.

You must not diagnose, judge, or classify how the patient is doing, and must
not use their age as a clinical or risk factor. You must not recommend a
different activity, exercise, treatment, or amount of activity, and must not
make any safety or normalcy judgment or mention seeing a doctor. Do not invent
facts the patient did not provide.

Reply with plain text only: no JSON, markdown, or lists."""


def build_quick_summary_prompt(age: int, assigned_task: str | None) -> str:
    """Build the Tab 1 quick-summary prompt from the patient's stated age."""
    task_line = (
        f'The patient\'s assigned activity today is: "{assigned_task}".'
        if assigned_task
        else "No activity has been chosen yet."
    )
    return (
        f"The patient is {age} years old.\n\n"
        f"{task_line}\n\n"
        "Reply following the rules in your system prompt."
    )


def build_onboarding_repair_prompt(previous_output: str, validation_error: str) -> str:
    """Build a one-shot schema repair prompt for a local intake response."""
    return (
        "Your previous intake response could not be parsed as valid JSON.\n\n"
        f"Previous response:\n{previous_output}\n\n"
        f"Validation error:\n{validation_error}\n\n"
        "Re-emit only a corrected JSON object matching this output_schema:\n"
        f"{json.dumps(ONBOARDING_JSON_SCHEMA_HINT, indent=2)}"
    )
