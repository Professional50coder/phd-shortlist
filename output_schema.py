from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class Tier(str, Enum):
    reach = "reach"
    target = "target"
    safety = "safety"


class EvidenceItem(BaseModel):
    title: str
    year: Optional[int]
    url: Optional[str]
    venue: Optional[str] = None
    abstract: Optional[str] = None
    funder: Optional[str] = None
    activity_code: Optional[str] = None
    is_personal_fellowship: Optional[bool] = None


class LinkedProgram(BaseModel):
    name: str
    url: Optional[str] = None
    eligible: bool = True


class ShortlistEntry(BaseModel):
    supervisor_id: str
    name: str
    institution: Optional[str]
    country: str
    contact_email: Optional[str]
    research_focus: Optional[str]
    evidence: List[EvidenceItem]
    why_match: str
    tier: Tier
    linked_programs: List[LinkedProgram] = Field(default_factory=list)
    area: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    @validator("country")
    def country_must_be_iso(cls, value: str) -> str:
        if not value or len(value) != 2 or not value.isalpha() or not value.isupper():
            raise ValueError("country must be an uppercase ISO 3166-1 alpha-2 code")
        return value


class OutputShortlist(BaseModel):
    student_id: str
    generated_at: datetime
    target_intake: dict
    coverage: dict
    shortlist: List[ShortlistEntry]

    @validator("target_intake")
    def intake_must_have_semester_year(cls, value: dict) -> dict:
        if "semester" not in value or "year" not in value:
            raise ValueError("target_intake must contain semester and year")
        return value


def validate_output(payload: dict) -> OutputShortlist:
    return OutputShortlist.parse_obj(payload)
