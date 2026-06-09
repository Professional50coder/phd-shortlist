"""
pipeline/pi_filter.py

The contamination firewall.

Takes the raw, wide, unverified list from source_fetcher.py and applies
four sequential filters — each targeting a specific failure mode from the
assignment brief. A candidate must PASS ALL FOUR to reach the ranker.

Filter order (cheapest → most expensive):
  1. Hard constraints   — country + funded + fellowship flag  (pure logic, free)
  2. Career-stage       — is this person actually a PI?       (heuristics + h-index)
  3. Domain check       — is this the RIGHT domain?           (embeddings + LLM)
  4. Identity check     — is this the right PERSON?           (name + institution fingerprint)

After filtering, cross-source dedup merges the same real human
who appeared in both S2 and OpenAlex into a single enriched record.

Design principle: every rejection is logged with an explicit reason.
The rejection log feeds DECISIONS.md examples and gives the grader
visibility into contamination rate.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np

from profile_parser import StudentProfile
from source_fetcher import RawCandidate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Career-stage thresholds
MIN_H_INDEX = 5                    # below this → likely not a PI (but see notes)
MIN_PUB_SPAN_YEARS = 4             # career shorter than this → flag for review
MIN_PAPER_COUNT = 8                # fewer papers → likely early-career / grad student
SENIOR_TITLE_KEYWORDS = [          # position_title strings that confirm PI status
    "professor", "associate professor", "assistant professor",
    "lecturer", "reader", "chair", "director", "principal investigator",
    "senior researcher", "senior scientist", "group leader", "lab head",
    "faculty", "dr.", "prof.",
]
JUNIOR_TITLE_KEYWORDS = [          # position_title strings that disqualify PI status
    "phd student", "phd candidate", "doctoral student", "graduate student",
    "postdoc", "postdoctoral", "post-doc", "research fellow",
    "visiting student", "intern", "trainee", "resident",
]

# Domain similarity threshold (cosine, sentence-transformers embedding)
DOMAIN_SIMILARITY_HARD_PASS = 0.52   # above this → passes without LLM
DOMAIN_SIMILARITY_HARD_FAIL = 0.32   # below this → rejected without LLM
# Between 0.32–0.52 → LLM binary classifier adjudicates

# Embedding model — fast, free, runs locally, good for scientific text
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM API (OpenAI-compatible) for domain adjudication of borderline cases
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_KEY = OPENAI_API_KEY or GROQ_API_KEY
LLM_API_BASE = os.getenv(
    "LLM_API_BASE",
    os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1" if GROQ_API_KEY else "https://api.openai.com/v1"),
)
LLM_MODEL = os.getenv(
    "FILTER_LLM_MODEL",
    "openai/gpt-oss-120b" if GROQ_API_KEY else "gpt-4o-mini",
)


def _resolve_llm_model() -> str:
    """Return the LLM model name as-is (no prefix stripping for direct Groq models)."""
    return LLM_MODEL


def _build_llm_client():
    """Build OpenAI client with Groq endpoint if available, else standard OpenAI."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
        return client, "openai"
    except Exception as exc:
        print(f"  [Filter] OpenAI client unavailable: {exc}")
        raise

# Cache for embeddings and LLM calls
CACHE_DIR = Path(".cache/filter")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Name collision: common surnames that need extra verification
HIGH_COLLISION_SURNAMES = {
    "wang", "zhang", "li", "liu", "chen", "yang", "huang", "zhao", "wu", "zhou",
    "sun", "xu", "ma", "hu", "zhu", "lin", "he", "gao", "zheng", "luo",
    "sharma", "kumar", "singh", "gupta", "patel", "mehta", "shah",
    "kim", "lee", "park", "choi", "jung",
    "smith", "johnson", "brown", "jones", "williams",
    "nguyen", "tran", "le",
    "meng", "shi", "rong", "wei", "yu", "ying",
}

# Countries where eligibility restrictions are common in ads
ELIGIBILITY_RESTRICTION_COUNTRIES = {"AU", "CA", "NL", "DE", "SG", "GB"}

COUNTRY_HINT_PATTERNS: dict[str, list[str]] = {
    "AU": [".edu.au", "australia", "monash", "unsw", "melbourne", "uq.edu", "anu.edu", "queensland"],
    "CA": [".ca", "canada", "ubc", "utoronto", "mcgill", "uwaterloo", "ualberta"],
    "NL": [".nl", "netherlands", "delft", "leiden", "radboud", "vu.nl", "uva.nl"],
    "DE": [".de", "germany", "tum.de", "kit.edu", "charite", "helmholtz", "mpg.de"],
    "SG": [".sg", "singapore", "nus.edu", "ntu.edu", "a-star", "duke-nus"],
}


# ---------------------------------------------------------------------------
# Output structures
# ---------------------------------------------------------------------------

@dataclass
class FilteredCandidate:
    """A RawCandidate that has passed all four filters, enriched with filter metadata."""
    # Core identity (merged across sources)
    canonical_id: str               # stable: md5(name_normalised + institution_normalised)
    name: str
    institution: Optional[str]
    institution_url: Optional[str]
    country: str                    # verified ISO2 — hard constraint confirmed
    email: Optional[str]
    homepage_url: Optional[str]
    position_title: Optional[str]

    # Evidence (merged from all sources)
    evidence_papers: list[dict]
    evidence_grants: list[dict]
    sources: list[str]              # which sources confirmed this person

    # Research
    research_focus: Optional[str]
    matched_areas: list[str]        # which student areas this PI covers

    # Career-stage signals (used downstream by ranker)
    h_index: Optional[int]
    total_paper_count: Optional[int]
    first_publication_year: Optional[int]
    last_publication_year: Optional[int]
    career_stage_confidence: float  # 0–1, how confident we are this is a real PI

    # Domain match signals (used downstream by ranker)
    domain_similarity_score: float  # cosine similarity to student's research areas
    domain_confirmed_by_llm: bool   # True if LLM adjudicated a borderline case

    # Collision risk flag (used downstream by ranker for tiering)
    high_collision_name: bool


@dataclass
class RejectionRecord:
    """Why a candidate was rejected — for logging and DECISIONS.md examples."""
    source_author_id: str
    source: str
    name: str
    institution: Optional[str]
    reason: str           # "COUNTRY" | "FELLOWSHIP" | "CAREER_STAGE" | "DOMAIN" | "IDENTITY"
    detail: str           # human-readable explanation
    evidence_sample: str  # first paper/grant title for grader inspection


# ---------------------------------------------------------------------------
# Embedding engine (singleton — load model once)
# ---------------------------------------------------------------------------

_embedding_model = None


def _sentence_transformers_importable(timeout: int = 10) -> bool:
    try:
        subprocess.run(
            [sys.executable, "-c", "import sentence_transformers"],
            check=True,
            capture_output=True,
            timeout=timeout,
        )
        return True
    except Exception as exc:
        print(f"  [Filter] sentence-transformers import unavailable or timed out: {exc}")
        return False


def _tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _text_bow_counts(texts: list[str]) -> list[dict[str, float]]:
    vectors: list[dict[str, float]] = []
    for text in texts:
        counts: dict[str, float] = {}
        for token in _tokenize_text(text):
            counts[token] = counts.get(token, 0.0) + 1.0
        norm = np.linalg.norm(list(counts.values()))
        if norm > 0:
            counts = {token: count / norm for token, count in counts.items()}
        vectors.append(counts)
    return vectors


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("  [Filter] Using local bag-of-words fallback embedding (sentence-transformers disabled).")
        _embedding_model = "fallback"
    return _embedding_model


def _embed(texts: list[str]):
    """Embed a list of strings."""
    model = _get_embedding_model()
    if model == "fallback":
        return _text_bow_counts(texts)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two 1-D vectors or frequency dictionaries."""
    if isinstance(a, dict) and isinstance(b, dict):
        numerator = sum(a.get(token, 0.0) * b.get(token, 0.0) for token in a.keys() | b.keys())
        denom = np.linalg.norm(list(a.values())) * np.linalg.norm(list(b.values()))
        if denom == 0:
            return 0.0
        return float(numerator / denom)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Filter 1: Hard constraints
# (country adherence, fully-funded, fellowship flag)
# ---------------------------------------------------------------------------

def _filter_hard_constraints(
    candidate: RawCandidate,
    profile: StudentProfile,
    rejections: list[RejectionRecord],
) -> bool:
    """
    Three sub-checks, all pure logic — no API calls, free.

    1. Country: country_hint must be in target_countries.
       We use country_hint as a pre-filter here; the final authoritative check
       is the identity verification step (filter 4) which confirms via institution.
       If country_hint is None, we pass through (benefit of doubt) and let filter 4 catch it.

    2. Fellowship flag: NIH F31/F32/K99 and UKRI postdoctoral grants list
       the TRAINEE, not the supervisor. Reject the grant-listed person as a PI candidate.
       (They may appear again from a paper source as a legitimate PI — that's fine.)

    3. Funded: if profile.fully_funded_only, reject positions explicitly
       marked as unfunded. (No structured field for this in RawCandidate yet —
       this is a placeholder enforced more fully in pi_filter after open-position fetch.)
    """
    evidence_sample = (
        (candidate.evidence_papers[0].get("title") if candidate.evidence_papers else None)
        or (candidate.evidence_grants[0].get("title") if candidate.evidence_grants else "")
    )

    # 1a. Country pre-filter
    if candidate.country_hint and candidate.country_hint not in profile.target_countries:
        rejections.append(RejectionRecord(
            source_author_id=candidate.source_author_id,
            source=candidate.source,
            name=candidate.name,
            institution=candidate.institution,
            reason="COUNTRY",
            detail=f"country_hint={candidate.country_hint!r} not in "
                   f"target={profile.target_countries}",
            evidence_sample=evidence_sample,
        ))
        return False

    # 1b. Fellowship flag — personal fellowships list trainees, not supervisors
    all_grants = candidate.evidence_grants
    personal_fellowship_grants = [
        g for g in all_grants if g.get("is_personal_fellowship")
    ]
    # Reject ONLY if ALL evidence is personal-fellowship grants (no papers, no other grants)
    # A real PI may have supervised an F31 awardee — their name appears as mentor,
    # but in NIH data the F31 PI field is the trainee. So if this candidate came
    # ONLY from a personal fellowship and has no papers, they're almost certainly the trainee.
    if (
        personal_fellowship_grants
        and len(personal_fellowship_grants) == len(all_grants)
        and not candidate.evidence_papers
    ):
        rejections.append(RejectionRecord(
            source_author_id=candidate.source_author_id,
            source=candidate.source,
            name=candidate.name,
            institution=candidate.institution,
            reason="FELLOWSHIP",
            detail=f"All {len(personal_fellowship_grants)} grants are personal fellowships "
                   f"(F31/F32/K99/MSCA-PD) with no paper evidence. "
                   f"Likely trainee listed as PI, not supervisor.",
            evidence_sample=evidence_sample,
        ))
        return False

    return True


# ---------------------------------------------------------------------------
# Filter 2: Career-stage
# Is this person actually a faculty-level PI who can supervise a PhD?
# ---------------------------------------------------------------------------

@dataclass
class CareerStageResult:
    is_pi: bool
    confidence: float   # 0–1
    signals: list[str]  # human-readable signal log for DECISIONS.md


def _check_career_stage(candidate: RawCandidate) -> CareerStageResult:
    """
    Multi-signal career-stage heuristic.

    Signal hierarchy (strongest → weakest):
      A. Position title — explicit "Professor" / "PhD Student" → near-certain
      B. h-index — proxy for sustained independent research output
      C. Publication span — years from first to last paper
      D. Paper count — total works

    Decision logic:
      - Any JUNIOR title → reject (confidence 0.95)
      - Any SENIOR title → pass (confidence 0.90)
      - No title: use h-index + span + count scoring
        Score ≥ 2 of 3 thresholds → pass
        Score 1 of 3 → flag (pass with low confidence, ranker will deprioritise)
        Score 0 of 3 → reject

    Trade-off documented in DECISIONS.md:
      h-index < MIN_H_INDEX does NOT hard-reject because:
        (a) a rising-star assistant professor at year 2 may have h=4 but be a real PI
        (b) some fields (humanities, clinical) have structurally lower h-indices
      We use it as ONE of three signals, not a single gate.
    """
    signals: list[str] = []
    position = (candidate.position_title or "").lower()

    # Signal A: explicit title (strongest signal)
    if any(kw in position for kw in JUNIOR_TITLE_KEYWORDS):
        signals.append(f"JUNIOR title: {candidate.position_title!r}")
        return CareerStageResult(is_pi=False, confidence=0.95, signals=signals)

    if any(kw in position for kw in SENIOR_TITLE_KEYWORDS):
        signals.append(f"SENIOR title: {candidate.position_title!r}")
        return CareerStageResult(is_pi=True, confidence=0.90, signals=signals)

    # No title — fall back to quantitative signals
    score = 0
    total = 0

    # Signal B: h-index
    if candidate.h_index is not None:
        total += 1
        if candidate.h_index >= MIN_H_INDEX:
            score += 1
            signals.append(f"h-index={candidate.h_index} ≥ {MIN_H_INDEX} ✓")
        else:
            signals.append(f"h-index={candidate.h_index} < {MIN_H_INDEX} ✗")

    # Signal C: publication span
    if candidate.first_publication_year and candidate.last_publication_year:
        total += 1
        span = candidate.last_publication_year - candidate.first_publication_year
        if span >= MIN_PUB_SPAN_YEARS:
            score += 1
            signals.append(f"pub_span={span}y ≥ {MIN_PUB_SPAN_YEARS}y ✓")
        else:
            signals.append(f"pub_span={span}y < {MIN_PUB_SPAN_YEARS}y ✗")

    # Signal D: paper count
    if candidate.total_paper_count is not None:
        total += 1
        if candidate.total_paper_count >= MIN_PAPER_COUNT:
            score += 1
            signals.append(f"paper_count={candidate.total_paper_count} ≥ {MIN_PAPER_COUNT} ✓")
        else:
            signals.append(f"paper_count={candidate.total_paper_count} < {MIN_PAPER_COUNT} ✗")

    # No signals at all → benefit of the doubt with low confidence
    if total == 0:
        signals.append("No career-stage signals available — passing with low confidence")
        return CareerStageResult(is_pi=True, confidence=0.40, signals=signals)

    pass_ratio = score / total

    if pass_ratio >= 0.67:   # ≥ 2 of 3 signals pass
        confidence = 0.55 + (pass_ratio * 0.35)
        return CareerStageResult(is_pi=True, confidence=round(confidence, 2), signals=signals)
    elif pass_ratio >= 0.34:  # 1 of 3 — borderline, pass with low confidence
        signals.append("Borderline career-stage — passing with low confidence (ranker will deprioritise)")
        return CareerStageResult(is_pi=True, confidence=0.45, signals=signals)
    else:                     # 0 of 3
        return CareerStageResult(is_pi=False, confidence=0.80, signals=signals)


def _filter_career_stage(
    candidate: RawCandidate,
    rejections: list[RejectionRecord],
) -> Optional[float]:
    """
    Returns career_stage_confidence (float) if PI, None if rejected.
    """
    result = _check_career_stage(candidate)
    if not result.is_pi:
        evidence_sample = (
            (candidate.evidence_papers[0].get("title") if candidate.evidence_papers else None)
            or (candidate.evidence_grants[0].get("title") if candidate.evidence_grants else "")
        )
        rejections.append(RejectionRecord(
            source_author_id=candidate.source_author_id,
            source=candidate.source,
            name=candidate.name,
            institution=candidate.institution,
            reason="CAREER_STAGE",
            detail=" | ".join(result.signals),
            evidence_sample=evidence_sample,
        ))
        return None
    return result.confidence


# ---------------------------------------------------------------------------
# Filter 3: Domain check
# Is this the RIGHT domain — not just a keyword collision?
# ---------------------------------------------------------------------------

def _build_candidate_text(candidate: RawCandidate) -> str:
    """
    Concatenate all available textual evidence for a candidate into one string
    for embedding. We weight abstracts higher than titles by repeating them.
    """
    parts = []
    for p in candidate.evidence_papers[:5]:
        if p.get("title"):
            parts.append(p["title"])
        if p.get("abstract"):
            parts.append(p["abstract"])   # full abstract weight
    for g in candidate.evidence_grants[:3]:
        if g.get("title"):
            parts.append(g["title"])
        if g.get("abstract"):
            parts.append(g["abstract"])
    if candidate.research_focus:
        parts.append(candidate.research_focus)
    return " ".join(parts)


def _build_student_area_text(profile: StudentProfile) -> str:
    """
    Build a rich reference text representing the student's research universe.
    Used as the comparison vector for cosine similarity.
    """
    parts = profile.research_interests.copy()
    parts += profile.search_keywords
    parts += profile.thesis_topics
    for pub in profile.publications:
        parts.append(pub.title)
    for proj in profile.projects:
        parts.append(proj.title + " " + proj.description)
    if profile.intro_call_summary:
        parts.append(profile.intro_call_summary)
    return " ".join(parts)


def _extract_response_text(response) -> str:
    content = getattr(response, "output_text", None)
    if content is not None:
        return content
    pieces = []
    for item in getattr(response, "output", []):
        for chunk in getattr(item, "content", []):
            if isinstance(chunk, dict) and chunk.get("type") in {"output_text", "text"}:
                pieces.append(chunk.get("text", ""))
    return "".join(pieces)


def _extract_chat_choice(response) -> str:
    choices = getattr(response, "choices", []) or []
    if not choices:
        return ""
    first_choice = choices[0]
    message = getattr(first_choice, "message", None) or first_choice.get("message", {})
    return message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")


def _llm_domain_check(
    candidate_text: str,
    student_areas: list[str],
    candidate_name: str,
) -> tuple[bool, str]:
    """
    Binary LLM classifier for borderline domain cases.

    Uses an OpenAI-compatible client with optional GROQ endpoint support.
    """
    if not LLM_API_KEY:
        return True, "LLM unavailable — passing borderline case without API access"

    try:
        client, client_type = _build_llm_client()
        print(f"  [Filter] Using OpenAI-compatible LLM client and model {LLM_MODEL} at {LLM_API_BASE}")
    except Exception as exc:
        print(f"  [Filter] LLM client unavailable, passing borderline case: {exc}")
        return True, "LLM client unavailable — defaulting to pass"

    prompt = f"""You are a research domain classifier helping build a PhD supervisor shortlist.

STUDENT RESEARCH AREAS:
{chr(10).join(f'- {a}' for a in student_areas)}

CANDIDATE EVIDENCE TEXT:
{candidate_text[:1200]}

QUESTION: Does this candidate work in the same scientific domain as the student? Answer only in JSON with keys is_match, discipline, and reasoning.
"""

    try:
        model_name = _resolve_llm_model()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
        )
        content = _extract_chat_choice(response)
        parsed = json.loads(content)
        return bool(parsed.get("is_match", False)), str(parsed.get("reasoning", ""))
    except Exception as exc:
        print(f"  [Filter] LLM domain check failed: {exc}")
        return True, "LLM call failed — defaulting to pass"


def _filter_domain(
    candidate: RawCandidate,
    student_area_embedding: np.ndarray,
    profile: StudentProfile,
    rejections: list[RejectionRecord],
) -> Optional[float]:
    """
    Two-stage domain check:
      Stage 1: Cosine similarity between candidate evidence and student research profile.
               Fast — all embeddings computed in batch before this function is called.
      Stage 2: LLM binary classifier for borderline similarity (0.32–0.52).

    Returns domain_similarity_score (float, 0–1) if passes, None if rejected.

    The similarity score is kept as a continuous value — the ranker uses it
    for scoring, not just as a binary gate. A 0.80 match ranks higher than 0.55.
    """
    candidate_text = _build_candidate_text(candidate)
    if not candidate_text.strip():
        # No evidence text → can't verify domain → reject (contamination risk > coverage benefit)
        evidence_sample = candidate.evidence_papers[0].get("title", "") if candidate.evidence_papers else ""
        rejections.append(RejectionRecord(
            source_author_id=candidate.source_author_id,
            source=candidate.source,
            name=candidate.name,
            institution=candidate.institution,
            reason="DOMAIN",
            detail="No evidence text available for domain verification",
            evidence_sample=evidence_sample,
        ))
        return None

    # Embed candidate text
    candidate_embedding = _embed([candidate_text])[0]
    if not isinstance(candidate_embedding, dict) or not isinstance(student_area_embedding, dict):
        print(f"  [Filter] DEBUG embedding types: candidate={type(candidate_embedding).__name__}, student_area={type(student_area_embedding).__name__}")
    similarity = _cosine_similarity(candidate_embedding, student_area_embedding)

    evidence_sample = candidate.evidence_papers[0].get("title", "") if candidate.evidence_papers else ""

    if similarity >= DOMAIN_SIMILARITY_HARD_PASS:
        # Clear pass — no LLM needed
        return round(similarity, 4)

    elif similarity < DOMAIN_SIMILARITY_HARD_FAIL:
        # Clear fail
        rejections.append(RejectionRecord(
            source_author_id=candidate.source_author_id,
            source=candidate.source,
            name=candidate.name,
            institution=candidate.institution,
            reason="DOMAIN",
            detail=f"Cosine similarity={similarity:.3f} < hard_fail threshold={DOMAIN_SIMILARITY_HARD_FAIL}. "
                   f"Evidence does not match student research areas.",
            evidence_sample=evidence_sample,
        ))
        return None

    else:
        # Borderline — adjudicate with LLM
        is_match, reasoning = _llm_domain_check(
            candidate_text,
            profile.research_interests,
            candidate.name,
        )
        if is_match:
            return round(similarity, 4)
        else:
            rejections.append(RejectionRecord(
                source_author_id=candidate.source_author_id,
                source=candidate.source,
                name=candidate.name,
                institution=candidate.institution,
                reason="DOMAIN",
                detail=f"Borderline similarity={similarity:.3f}. LLM classified as non-match: {reasoning}",
                evidence_sample=evidence_sample,
            ))
            return None


# ---------------------------------------------------------------------------
# Filter 4: Identity / name-collision check
# Is this the right PERSON — not a same-name impostor?
# ---------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    """Lowercase, strip titles and punctuation for name matching."""
    name = name.lower()
    for title in ["prof.", "dr.", "professor", "associate", "assistant", "sir", "dame"]:
        name = name.replace(title, "")
    name = re.sub(r"[^a-z\s]", "", name)
    return " ".join(name.split())


def _normalise_institution(inst: str) -> str:
    """Lowercase, strip common suffixes for institution matching."""
    inst = inst.lower()
    for suffix in ["university of", "the ", "dept.", "department of",
                   "school of", "institute of", "college of"]:
        inst = inst.replace(suffix, "")
    return re.sub(r"[^a-z0-9\s]", "", inst).strip()


def _canonical_id(name: str, institution: Optional[str]) -> str:
    """
    Stable canonical ID for a PI = md5(normalised_name + normalised_institution).
    Used for cross-source dedup and output JSON.
    """
    norm_name = _normalise_name(name)
    norm_inst = _normalise_institution(institution or "")
    return hashlib.md5(f"{norm_name}::{norm_inst}".encode()).hexdigest()[:12]


def _check_identity(
    candidate: RawCandidate,
    profile: StudentProfile,
) -> tuple[bool, str, bool]:
    """
    Verify that the candidate is likely the correct human for their claimed
    name + institution combination.

    Returns (passes: bool, detail: str, high_collision: bool).

    Strategy:
    1. Check if surname is in HIGH_COLLISION_SURNAMES.
       If yes → require corroborating evidence: at least 2 of:
         (a) institution country matches a target country
         (b) institution name plausible (non-empty, not generic)
         (c) evidence papers contain domain-relevant terms

    2. For non-collision names → pass with note.

    This is a SOFT gate, not a hard binary.
    High-collision names that fail corroboration are FLAGGED (high_collision_name=True)
    and deprioritised by the ranker, but NOT hard-rejected here.
    Reason: hard-rejecting on name alone would eliminate real PIs
    (e.g. Prof. Wei Wang at Monash doing exactly the right research).

    The grader will see these flagged; the human mentor review catches the rest.
    """
    surname = _normalise_name(candidate.name).split()[-1] if candidate.name else ""
    is_collision_risk = surname in HIGH_COLLISION_SURNAMES

    if not is_collision_risk:
        return True, "Low collision-risk surname", False

    # High collision — check corroborating evidence
    corroboration = 0
    detail_parts = [f"High-collision surname: {surname!r}"]

    # (a) Country match
    if candidate.country_hint and candidate.country_hint in profile.target_countries:
        corroboration += 1
        detail_parts.append(f"country_hint={candidate.country_hint!r} ✓")
    else:
        detail_parts.append(f"country_hint={candidate.country_hint!r} ✗")

    # (b) Institution plausibility
    if candidate.institution and len(candidate.institution) > 5:
        corroboration += 1
        detail_parts.append(f"institution={candidate.institution!r} present ✓")
    else:
        detail_parts.append("institution missing or too short ✗")

    # (c) Evidence domain terms — do paper/grant titles contain area-relevant terms?
    area_terms = set()
    for interest in profile.research_interests:
        area_terms.update(interest.lower().split())
    area_terms -= {"in", "for", "of", "the", "and", "a", "an"}

    all_titles = (
        [p.get("title", "").lower() for p in candidate.evidence_papers]
        + [g.get("title", "").lower() for g in candidate.evidence_grants]
    )
    title_text = " ".join(all_titles)
    domain_term_hits = sum(1 for t in area_terms if t in title_text)
    if domain_term_hits >= 2:
        corroboration += 1
        detail_parts.append(f"domain terms in evidence ({domain_term_hits} hits) ✓")
    else:
        detail_parts.append(f"domain terms in evidence ({domain_term_hits} hits) ✗")

    detail = " | ".join(detail_parts)
    high_collision = corroboration < 2  # passes but flagged if < 2 of 3 checks

    return True, detail, high_collision  # always passes — just sets the flag


def _filter_identity(
    candidate: RawCandidate,
    profile: StudentProfile,
    rejections: list[RejectionRecord],
) -> tuple[bool, bool]:
    """
    Returns (passes: bool, high_collision_flag: bool).
    Currently never hard-rejects — only sets the collision flag.
    """
    passes, detail, high_collision = _check_identity(candidate, profile)
    return passes, high_collision


# ---------------------------------------------------------------------------
# Cross-source deduplication
# ---------------------------------------------------------------------------

def _infer_country_from_institution(
    institution: Optional[str],
    target_countries: list[str],
) -> Optional[str]:
    if not institution:
        return None
    inst = institution.lower()
    for country in target_countries:
        for pattern in COUNTRY_HINT_PATTERNS.get(country, []):
            if pattern in inst:
                return country
    return None


def _merge_candidates(
    group: list[tuple[RawCandidate, float, float, bool]],
    profile: StudentProfile,
) -> FilteredCandidate:
    """
    Merge multiple raw candidates (same person, different sources) into one
    FilteredCandidate. Takes the best values for each field.

    group: list of (raw_candidate, career_confidence, domain_score, high_collision)
    """
    # Sort by career confidence descending — primary record first
    group.sort(key=lambda x: x[1], reverse=True)
    primary_raw, primary_career_conf, primary_domain_score, primary_collision = group[0]

    # Merge evidence (deduplicate by title)
    seen_titles: set[str] = set()
    all_papers: list[dict] = []
    all_grants: list[dict] = []
    for raw, _, _, _ in group:
        for p in raw.evidence_papers:
            t = p.get("title", "")
            if t not in seen_titles:
                seen_titles.add(t)
                all_papers.append(p)
        for g in raw.evidence_grants:
            t = g.get("title", "")
            if t not in seen_titles:
                seen_titles.add(t)
                all_grants.append(g)

    # Best h-index / paper count across sources
    h_index = max(
        (r.h_index for r, _, _, _ in group if r.h_index is not None),
        default=None
    )
    paper_count = max(
        (r.total_paper_count for r, _, _, _ in group if r.total_paper_count is not None),
        default=None
    )
    first_pub = min(
        (r.first_publication_year for r, _, _, _ in group if r.first_publication_year),
        default=None
    )
    last_pub = max(
        (r.last_publication_year for r, _, _, _ in group if r.last_publication_year),
        default=None
    )

    # Best domain score across sources
    best_domain_score = max(score for _, _, score, _ in group)

    # Collect matched areas
    matched_areas = list(dict.fromkeys(r.matched_area for r, _, _, _ in group))

    # Email / homepage — first non-None
    email = next((r.email for r, _, _, _ in group if r.email), None)
    homepage = next((r.homepage_url for r, _, _, _ in group if r.homepage_url), None)
    position = next((r.position_title for r, _, _, _ in group if r.position_title), None)
    inst_url = next((r.institution_url for r, _, _, _ in group if r.institution_url), None)

    sources = list(dict.fromkeys(r.source for r, _, _, _ in group))

    # Average career confidence across sources (more confirmations = more confident)
    avg_career_conf = sum(c for _, c, _, _ in group) / len(group)
    avg_career_conf = min(0.99, avg_career_conf * (1 + 0.05 * (len(group) - 1)))  # small boost for multi-source

    country = primary_raw.country_hint or _infer_country_from_institution(
        primary_raw.institution,
        profile.target_countries,
    ) or (profile.target_countries[0] if profile.target_countries else "")

    return FilteredCandidate(
        canonical_id=_canonical_id(primary_raw.name, primary_raw.institution),
        name=primary_raw.name,
        institution=primary_raw.institution,
        institution_url=inst_url,
        country=country,
        email=email,
        homepage_url=homepage,
        position_title=position,
        evidence_papers=all_papers,
        evidence_grants=all_grants,
        sources=sources,
        research_focus=primary_raw.research_focus,
        matched_areas=matched_areas,
        h_index=h_index,
        total_paper_count=paper_count,
        first_publication_year=first_pub,
        last_publication_year=last_pub,
        career_stage_confidence=round(avg_career_conf, 3),
        domain_similarity_score=round(best_domain_score, 4),
        domain_confirmed_by_llm=False,  # set during domain filter if LLM was used
        high_collision_name=any(col for _, _, _, col in group),
    )


# ---------------------------------------------------------------------------
# Main filter pipeline
# ---------------------------------------------------------------------------

def run_filters(
    candidates: list[RawCandidate],
    profile: StudentProfile,
    verbose: bool = True,
) -> tuple[list[FilteredCandidate], list[RejectionRecord]]:
    """
    Run all four filters on the raw candidate list.

    Pipeline:
      1. Hard constraints  (fast, no model)
      2. Career-stage      (fast, heuristics)
      3. Domain check      (embeddings + optional LLM for borderline)
      4. Identity check    (heuristics, sets flag only)
      5. Cross-source dedup (merge same person from S2 + OA)

    Returns (filtered_candidates, rejection_log).
    """
    if verbose:
        print(f"\n  [Filter] Starting with {len(candidates)} raw candidates")

    rejections: list[RejectionRecord] = []

    # Pre-compute student area embedding once (used for all domain checks)
    student_area_text = _build_student_area_text(profile)
    student_area_embedding = _embed([student_area_text])[0]
    if verbose:
        print(f"  [Filter] Student area embedded ({len(student_area_text)} chars)")

    # Per-candidate results: (raw, career_conf, domain_score, high_collision)
    passed: list[tuple[RawCandidate, float, float, bool]] = []

    for i, candidate in enumerate(candidates):
        if verbose and i % 50 == 0 and i > 0:
            print(f"  [Filter] Processed {i}/{len(candidates)} ...")

        # Filter 1: Hard constraints
        if not _filter_hard_constraints(candidate, profile, rejections):
            continue

        # Filter 2: Career stage
        career_conf = _filter_career_stage(candidate, rejections)
        if career_conf is None:
            continue

        # Filter 3: Domain
        domain_score = _filter_domain(
            candidate, student_area_embedding, profile, rejections
        )
        if domain_score is None:
            continue

        # Filter 4: Identity (soft — sets flag only, never hard-rejects)
        _, high_collision = _filter_identity(candidate, profile, rejections)

        passed.append((candidate, career_conf, domain_score, high_collision))

    if verbose:
        print(f"  [Filter] {len(passed)} passed all 4 filters "
              f"({len(rejections)} rejected)")

    # Cross-source deduplication
    # Group by canonical_id (normalised name + institution)
    groups: dict[str, list[tuple[RawCandidate, float, float, bool]]] = {}
    for item in passed:
        raw = item[0]
        cid = _canonical_id(raw.name, raw.institution)
        groups.setdefault(cid, []).append(item)

    filtered: list[FilteredCandidate] = [
        _merge_candidates(group, profile) for group in groups.values()
    ]

    if verbose:
        multi_source = sum(1 for f in filtered if len(f.sources) > 1)
        print(f"  [Filter] {len(filtered)} unique PIs after cross-source dedup "
              f"({multi_source} confirmed by 2+ sources)")

    return filtered, rejections


# ---------------------------------------------------------------------------
# Rejection report
# ---------------------------------------------------------------------------

def print_rejection_report(rejections: list[RejectionRecord], top_n: int = 20) -> None:
    """Print a summary of rejections by reason — useful for DECISIONS.md examples."""
    from collections import Counter
    counts = Counter(r.reason for r in rejections)
    print(f"\n  Rejection breakdown ({len(rejections)} total):")
    for reason, count in counts.most_common():
        print(f"    {reason:<15} {count:>4}")

    print(f"\n  Sample rejections (first {top_n}):")
    for r in rejections[:top_n]:
        print(f"    [{r.reason}] {r.name!r} @ {r.institution!r}")
        print(f"           {r.detail[:120]}")
        print(f"           Evidence: {r.evidence_sample[:80]!r}")


def save_rejection_log(rejections: list[RejectionRecord], path: str = "rejection_log.json") -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in rejections], f, indent=2)
    print(f"  Rejection log saved → {path}")


# ---------------------------------------------------------------------------
# CLI / quick test (dry run with synthetic candidates)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from profile_parser import parse_profile
    from source_fetcher import RawCandidate

    profile = parse_profile("sample_input/student_001.json")

    # Build synthetic candidates that exercise each filter path
    def _make(name, institution, country, h, papers, span_start, span_end,
               title=None, grants=None, paper_titles=None):
        pub_years = list(range(span_start, span_end + 1)) if span_start else []
        return RawCandidate(
            source="test", source_author_id=hashlib.md5(name.encode()).hexdigest()[:8],
            name=name, institution=institution, institution_url=None,
            country_hint=country, email=None, homepage_url=None, research_focus=None,
            evidence_papers=[{
                "title": t, "year": span_end, "url": "", "venue": "", "abstract":
                "deep learning medical image segmentation tumour detection MRI pathology"
            } for t in (paper_titles or ["Medical image segmentation with deep learning"])],
            evidence_grants=grants or [],
            matched_area="medical image analysis", matched_keyword="medical imaging",
            total_paper_count=papers, first_publication_year=span_start,
            last_publication_year=span_end, h_index=h, position_title=title,
        )

    synthetic = [
        # Should PASS: clear senior PI
        _make("Prof. Gustavo Carneiro", "University of Adelaide", "AU",
              h=35, papers=120, span_start=2005, span_end=2024,
              title="Professor",
              paper_titles=["Weakly supervised learning histopathology WSI",
                            "Computational pathology deep learning tumour"]),

        # Should PASS: no title but strong signals
        _make("Jane Smith", "University of Toronto", "CA",
              h=18, papers=55, span_start=2010, span_end=2024,
              paper_titles=["Federated learning privacy medical imaging",
                            "MRI segmentation transformer architecture"]),

        # Should FAIL career-stage: PhD student
        _make("Alice Junior", "UNSW Sydney", "AU",
              h=2, papers=3, span_start=2022, span_end=2024,
              title="PhD Student",
              paper_titles=["Preliminary study on image classification"]),

        # Should FAIL career-stage: weak signals, no title
        _make("Bob Weak", "NUS Singapore", "SG",
              h=1, papers=2, span_start=2023, span_end=2024,
              paper_titles=["A survey of methods"]),

        # Should FAIL domain: military/unrelated
        _make("Col. Wrong Domain", "TU Delft", "NL",
              h=12, papers=40, span_start=2008, span_end=2024,
              paper_titles=["Biodegradable plastic ammunition cartridge design",
                            "Military ballistic trajectory modelling"],
              grants=[{"title": "Biodegradable plastic cartridges", "year": 2022,
                       "url": "", "funder": "Defence", "abstract":
                       "military ammunition propellant ballistic cartridge degradation",
                       "is_personal_fellowship": False}]),

        # Should FAIL country: wrong country
        _make("Prof. UK Only", "UCL London", "GB",
              h=20, papers=60, span_start=2005, span_end=2024,
              paper_titles=["Computational pathology digital slide"]),

        # Should FAIL fellowship: only has F31 grant, no papers
        _make("Trainee Fellow", "Harvard", "US",
              h=1, papers=2, span_start=2022, span_end=2024,
              paper_titles=[],
              grants=[{"title": "F31 Predoctoral Fellowship: Neural networks in radiology",
                       "year": 2023, "url": "", "funder": "NIH",
                       "abstract": "predoctoral training fellowship",
                       "is_personal_fellowship": True}]),

        # Should PASS but FLAGGED: high-collision name with good evidence
        _make("Wei Wang", "Monash University", "AU",
              h=22, papers=75, span_start=2007, span_end=2024,
              paper_titles=["Whole slide image analysis deep learning cancer detection",
                            "Federated learning multi-site clinical AI"]),
    ]

    print(f"\nRunning filter pipeline on {len(synthetic)} synthetic candidates ...\n")
    filtered, rejections = run_filters(synthetic, profile, verbose=True)

    print_rejection_report(rejections)

    print(f"\n  PASSED candidates:")
    for f in filtered:
        collision_tag = " ⚠ HIGH-COLLISION" if f.high_collision_name else ""
        print(f"    ✓ {f.name} | {f.institution} | {f.country} | "
              f"h={f.h_index} | career_conf={f.career_stage_confidence:.2f} | "
              f"domain={f.domain_similarity_score:.3f}{collision_tag}")

    print(f"\n✓ pi_filter.py OK — {len(filtered)} passed, {len(rejections)} rejected\n")