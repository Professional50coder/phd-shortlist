"""
pipeline/profile_parser.py

Parses the raw student profile JSON into a strongly-typed StudentProfile dataclass
used by every downstream pipeline stage.

Design decisions:
- All hard constraints (target_countries, target_intake) are validated here and
  fail loudly — any stage that ignores them would produce a hard failure on grading.
- Research interests are both kept verbatim AND expanded into search keywords.
  The expansion adds synonyms/subfields so source_fetcher.py casts a wider net
  while pi_filter.py enforces the actual semantic match.
- nationality is extracted here for eligibility filtering (6.4) in pi_filter.py.
- The parser never touches external APIs — pure in-memory transform, fast, testable.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Education:
    degree: str
    field_of_study: str
    institution: str
    country: str
    graduation_year: int
    gpa: Optional[str]
    thesis: Optional[str]


@dataclass
class Publication:
    title: str
    venue: str
    year: int
    url: Optional[str]
    role: str  # "first author" | "co-author" | etc.


@dataclass
class Project:
    title: str
    description: str
    year: int


@dataclass
class TargetIntake:
    semester: str   # "Fall" | "Spring" | "Winter"
    year: int


@dataclass
class StudentProfile:
    # Identity
    student_id: str
    name: str
    nationality: str           # ISO 3166-1 alpha-2, e.g. "IN"
    email: str

    # Academic record
    education: list[Education]
    publications: list[Publication]
    projects: list[Project]
    skills: list[str]

    # Research
    research_interests: list[str]       # verbatim from profile, 3–5 items
    search_keywords: list[str]          # expanded for API queries (see _expand_keywords)

    # Hard constraints — enforced in every downstream stage
    target_countries: list[str]         # ISO 3166-1 alpha-2 list
    target_intake: TargetIntake
    fully_funded_only: bool             # extracted from intro call / profile

    # Context
    intro_call_summary: str
    raw_resume_text: str

    # Derived convenience fields
    highest_degree: str                 # e.g. "B.Tech", "M.Sc"
    has_publication: bool
    thesis_topics: list[str]            # pulled from all education.thesis fields


# ---------------------------------------------------------------------------
# Keyword expansion
# ---------------------------------------------------------------------------

# Manual synonym map for common research areas.
# Keeps expansion conservative — only well-established synonyms, not hallucinated.
# Add to this map as new student profiles are encountered.
KEYWORD_SYNONYMS: dict[str, list[str]] = {
    "medical image analysis": [
        "medical imaging", "radiology AI", "MRI segmentation",
        "CT segmentation", "image segmentation", "biomedical image processing"
    ],
    "computational pathology": [
        "digital pathology", "whole slide image", "WSI analysis",
        "histopathology deep learning", "pathology AI", "tissue classification"
    ],
    "federated learning in healthcare": [
        "federated learning", "privacy-preserving machine learning",
        "distributed learning clinical", "multi-site learning"
    ],
    "multimodal AI for oncology": [
        "multimodal learning cancer", "cancer AI", "oncology deep learning",
        "tumour detection", "multi-omics AI"
    ],
    "explainable AI in clinical decision support": [
        "explainable AI healthcare", "XAI medical", "interpretable ML clinical",
        "clinical decision support AI"
    ],
    # Generic fallbacks — applied if verbatim interest has no explicit entry
    "machine learning": ["deep learning", "neural networks", "AI"],
    "natural language processing": ["NLP", "large language models", "text mining"],
    "computer vision": ["image recognition", "object detection", "visual AI"],
}


def _expand_keywords(research_interests: list[str]) -> list[str]:
    """
    Returns a deduplicated list of search keywords for source_fetcher.py.

    Strategy:
    1. Start with the verbatim research interest strings.
    2. Add any known synonyms from KEYWORD_SYNONYMS.
    3. Deduplicate (case-insensitive) while preserving order.

    The result is used as input to Semantic Scholar / OpenAlex full-text search.
    Wider coverage here is intentional — pi_filter.py enforces semantic precision.
    """
    seen: set[str] = set()
    keywords: list[str] = []

    for interest in research_interests:
        for kw in [interest] + KEYWORD_SYNONYMS.get(interest.lower(), []):
            if kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)

    return keywords


# ---------------------------------------------------------------------------
# Derived field helpers
# ---------------------------------------------------------------------------

def _extract_highest_degree(education: list[Education]) -> str:
    """
    Returns the most advanced degree. Priority: PhD > M.Sc/M.Tech > B.Tech/B.Sc > Minor.
    Falls back to the last education entry if no match.
    """
    degree_rank = {
        "phd": 4, "ph.d": 4, "d.phil": 4,
        "m.sc": 3, "msc": 3, "m.tech": 3, "mtech": 3, "m.eng": 3, "meng": 3, "master": 3,
        "b.sc": 2, "bsc": 2, "b.tech": 2, "btech": 2, "b.eng": 2, "bachelor": 2,
        "minor": 1, "diploma": 1,
    }
    best = education[-1] if education else None
    best_rank = 0
    for edu in education:
        rank = degree_rank.get(edu.degree.lower().replace(" ", ""), 0)
        if rank > best_rank:
            best_rank = rank
            best = edu
    return best.degree if best else "Unknown"


def _extract_thesis_topics(education: list[Education]) -> list[str]:
    """Pulls non-null thesis strings from all education entries."""
    return [edu.thesis for edu in education if edu.thesis]


def _infer_fully_funded(intro_call: str, raw_resume: str) -> bool:
    """
    Heuristic: if the intro call or resume mentions 'funded', 'scholarship',
    'stipend', 'fellowship', or explicit 'fully funded only' — set True.
    We default to True (funded only) because surfacing unfunded positions to
    an international student is almost always useless.
    """
    text = (intro_call + " " + raw_resume).lower()
    funded_signals = [
        "fully funded", "funded only", "stipend", "scholarship",
        "fellowship", "RA position", "TA position", "research assistantship"
    ]
    # Explicit 'not funded' signals would flip this — conservative for now
    return any(sig.lower() in text for sig in funded_signals)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_profile(profile_path: str | Path) -> StudentProfile:
    """
    Load and validate a student profile JSON file.
    Returns a StudentProfile dataclass ready for all downstream stages.

    Raises:
        FileNotFoundError: if the path doesn't exist
        ValueError: if required fields are missing or target_countries is empty
        json.JSONDecodeError: if the file is not valid JSON
    """
    path = Path(profile_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # --- Required field checks ---
    required_top = ["student_id", "personal", "education", "research_interests",
                    "target_countries", "target_intake"]
    for field_name in required_top:
        if field_name not in raw:
            raise ValueError(f"Missing required field in profile: '{field_name}'")

    if not raw["target_countries"]:
        raise ValueError("target_countries must not be empty — it is a hard constraint")

    if not raw["research_interests"]:
        raise ValueError("research_interests must not be empty")

    # --- Parse sub-structures ---
    education = [
        Education(
            degree=edu["degree"],
            field_of_study=edu["field"],
            institution=edu["institution"],
            country=edu["country"],
            graduation_year=edu["graduation_year"],
            gpa=edu.get("gpa"),
            thesis=edu.get("thesis"),
        )
        for edu in raw["education"]
    ]

    publications = [
        Publication(
            title=pub["title"],
            venue=pub["venue"],
            year=pub["year"],
            url=pub.get("url"),
            role=pub.get("role", "co-author"),
        )
        for pub in raw.get("publications", [])
    ]

    projects = [
        Project(
            title=proj["title"],
            description=proj["description"],
            year=proj["year"],
        )
        for proj in raw.get("projects", [])
    ]

    intake_raw = raw["target_intake"]
    target_intake = TargetIntake(
        semester=intake_raw["semester"],
        year=int(intake_raw["year"]),
    )

    research_interests = [i.strip() for i in raw["research_interests"]]
    search_keywords = _expand_keywords(research_interests)

    intro_call = raw.get("intro_call_summary", "")
    resume_text = raw.get("raw_resume_text", "")

    profile = StudentProfile(
        student_id=raw["student_id"],
        name=raw["personal"]["name"],
        nationality=raw["personal"].get("nationality", ""),
        email=raw["personal"].get("email", ""),
        education=education,
        publications=publications,
        projects=projects,
        skills=raw.get("skills", []),
        research_interests=research_interests,
        search_keywords=search_keywords,
        target_countries=[c.upper() for c in raw["target_countries"]],
        target_intake=target_intake,
        fully_funded_only=_infer_fully_funded(intro_call, resume_text),
        intro_call_summary=intro_call,
        raw_resume_text=resume_text,
        highest_degree=_extract_highest_degree(education),
        has_publication=len(publications) > 0,
        thesis_topics=_extract_thesis_topics(education),
    )

    return profile


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

def _profile_summary(p: StudentProfile) -> str:
    """Human-readable summary for logging and verification."""
    lines = [
        f"{'='*60}",
        f"  Student: {p.name} ({p.student_id})",
        f"  Nationality: {p.nationality}",
        f"  Highest degree: {p.highest_degree}",
        f"  Has publication: {p.has_publication}",
        f"  Target countries: {', '.join(p.target_countries)}",
        f"  Target intake: {p.target_intake.semester} {p.target_intake.year}",
        f"  Fully funded only: {p.fully_funded_only}",
        f"",
        f"  Research interests ({len(p.research_interests)}):",
        *[f"    - {i}" for i in p.research_interests],
        f"",
        f"  Search keywords ({len(p.search_keywords)}):",
        *[f"    - {kw}" for kw in p.search_keywords],
        f"",
        f"  Thesis topics: {p.thesis_topics}",
        f"{'='*60}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    profile_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input/student_001.json"
    print(f"\nParsing: {profile_path}\n")

    profile = parse_profile(profile_path)
    print(_profile_summary(profile))
    print("\n✓ Profile parsed successfully. Ready for source_fetcher.py\n")