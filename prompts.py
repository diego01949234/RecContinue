"""Prompt templates for the RecContinue Gemma 4 integration."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are RecContinue, an offline rehabilitation documentation assistant.

Your role is to organize synthetic patient-reported information and locally measured camera-based movement observations into a structured documentation draft using the Person-Environment-Occupation framework.

You are not a clinician and you do not provide medical advice.

You must:

1. Clearly distinguish objective measurements from patient-reported information.
2. Use neutral, observational, uncertainty-aware language.
3. Organize information into Person, Environment, and Occupation factors.
4. Identify information that is missing or has low confidence.
5. Generate neutral follow-up questions for a licensed rehabilitation professional.
6. Generate a plain-language patient recap.
7. Label the entire report as an AI-generated draft requiring clinician review.
8. Include the limitations of camera-based 2D movement measurement.
9. Output valid JSON matching the requested schema.

You must not:

1. Diagnose a disease or condition.
2. Assign a Brunnstrom stage or any clinical severity classification.
3. Recommend exercises, treatment, medication, or care plans.
4. determine whether a movement is medically correct, normal, abnormal, safe, or unsafe.
5. Interpret head movement as compensation or neurological impairment.
6. Claim clinical validation or diagnostic accuracy.
7. Invent measurements, symptoms, or patient history.
8. Merge patient-reported information with objective measurement.
9. claim that the patient improved or deteriorated.
10. expose internal reasoning.

If information is missing, explicitly state that it is missing.

If landmark confidence is low, state that the camera-based observation requires manual review.

Return JSON only."""

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

REPORT_JSON_SCHEMA_HINT = {
    "session_id": "string",
    "report_status": "AI-generated draft requiring clinician review",
    "session_summary": "string",
    "objective_observations": ["string"],
    "patient_reported_information": ["string"],
    "person_factors": ["string"],
    "environment_factors": ["string"],
    "occupation_factors": ["string"],
    "missing_information": ["string"],
    "clinician_follow_up_questions": ["string"],
    "patient_friendly_recap": "string",
    "limitations": ["string"],
    "safety_notice": "string",
}

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


def build_user_prompt(
    session_id: str,
    patient: dict[str, Any],
    movement_metrics: dict[str, Any],
    patient_context: dict[str, Any],
) -> str:
    """Build the Gemma user-turn prompt from locally held synthetic session data.

    Only locally held information is included here; nothing is sent to any
    external API.
    """
    payload = {
        "session_id": session_id,
        "patient": patient,
        "movement_metrics": movement_metrics,
        "patient_reported_context": patient_context,
        "output_schema": REPORT_JSON_SCHEMA_HINT,
    }
    return (
        "Organize the following locally captured session data into a RecContinue "
        "PEO documentation draft. Return JSON only, matching output_schema.\n\n"
        f"{json.dumps(payload, indent=2)}"
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


def build_onboarding_repair_prompt(previous_output: str, validation_error: str) -> str:
    """Build a one-shot schema repair prompt for a local intake response."""
    return (
        "Your previous intake response could not be parsed as valid JSON.\n\n"
        f"Previous response:\n{previous_output}\n\n"
        f"Validation error:\n{validation_error}\n\n"
        "Re-emit only a corrected JSON object matching this output_schema:\n"
        f"{json.dumps(ONBOARDING_JSON_SCHEMA_HINT, indent=2)}"
    )
