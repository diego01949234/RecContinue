"""Pydantic schemas for RecContinue's Gemma-generated PEO report."""
from __future__ import annotations

from pydantic import BaseModel, Field


class OnboardingIntake(BaseModel):
    """Gemma-organized patient context used to prefill the PEO report form.

    This is an intake summary, not a diagnosis or a replacement for a therapist's
    assigned activity. Keeping it structured means the patient can review and
    edit every field before it becomes part of a session record.
    """

    acknowledgement: str
    daily_activity: str
    difficulty_location: str
    assistance_needed: str
    discomfort_reported: str
    independence_goal: str
    patient_statement: str
    missing_information: list[str] = Field(default_factory=list)
    task_connection: str


class RecContinueReport(BaseModel):
    session_id: str
    report_status: str = "AI-generated draft requiring clinician review"
    session_summary: str
    objective_observations: list[str] = Field(default_factory=list)
    patient_reported_information: list[str] = Field(default_factory=list)
    person_factors: list[str] = Field(default_factory=list)
    environment_factors: list[str] = Field(default_factory=list)
    occupation_factors: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clinician_follow_up_questions: list[str] = Field(default_factory=list)
    patient_friendly_recap: str
    limitations: list[str] = Field(default_factory=list)
    safety_notice: str

    def clinical_text_sections(self) -> list[str]:
        """Return the free-text clinical-content fields that safety validation must scan.

        Excludes report_status and safety_notice, which are disclaimer/label
        fields rather than AI-generated clinical content.
        """
        sections: list[str] = [self.session_summary, self.patient_friendly_recap]
        sections.extend(self.objective_observations)
        sections.extend(self.patient_reported_information)
        sections.extend(self.person_factors)
        sections.extend(self.environment_factors)
        sections.extend(self.occupation_factors)
        sections.extend(self.missing_information)
        sections.extend(self.clinician_follow_up_questions)
        sections.extend(self.limitations)
        return sections
