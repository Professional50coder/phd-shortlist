from __future__ import annotations

import json
import os
from typing import List

from profile_parser import StudentProfile
from ranker import RankedCandidate

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_API_KEY = OPENAI_API_KEY or GROQ_API_KEY
LLM_API_BASE = os.getenv(
    "LLM_API_BASE",
    os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1" if GROQ_API_KEY else "https://api.openai.com/v1"),
)
LLM_MODEL = os.getenv(
    "PERSONALISER_LLM_MODEL",
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
        print(f"  [Personaliser] OpenAI client unavailable: {exc}")
        raise


def _build_evidence_snippet(candidate: RankedCandidate) -> str:
    papers = candidate.candidate.evidence_papers[:2]
    grants = candidate.candidate.evidence_grants[:1]
    snippets = []
    if papers:
        snippets.append(f"papers such as '{papers[0].get('title', '')}'")
    if len(papers) > 1:
        snippets.append(f"and '{papers[1].get('title', '')}'")
    if grants:
        snippets.append(f"funded work like '{grants[0].get('title', '')}'")
    return ", ".join(snippets) if snippets else "their published work"


def _normalize_evidence(item: dict) -> dict:
    normalized = item.copy()
    year = normalized.get("year")
    if isinstance(year, str):
        normalized["year"] = int(year) if year.isdigit() else None
    return normalized


def _fallback_why_match(profile: StudentProfile, candidate: RankedCandidate) -> str:
    interest = profile.research_interests[0] if profile.research_interests else "your research"
    evidence = _build_evidence_snippet(candidate)
    return (
        f"{candidate.candidate.name} is a strong fit for {profile.name} because they work on {candidate.coverage_area or interest}, "
        f"which aligns closely with {interest}. Their recent work includes {evidence}. "
        f"This makes them a strong candidate for a fully funded PhD application focused on applied healthcare AI."
    )


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


def _try_llm_generate(profile: StudentProfile, candidates: List[RankedCandidate]) -> List[str]:
    if not LLM_API_KEY:
        return ["" for _ in candidates]

    try:
        client, client_type = _build_llm_client()
    except Exception:
        return ["" for _ in candidates]

    if GROQ_API_KEY:
        print(f"  [Personaliser] Using {client_type.upper()} endpoint for LLM: {LLM_API_BASE}")
    batch_items = []
    for cand in candidates:
        evidence = _build_evidence_snippet(cand)
        batch_items.append({
            "name": cand.candidate.name,
            "institution": cand.candidate.institution,
            "matched_area": cand.coverage_area,
            "evidence": evidence,
        })

    prompt_lines = [
        "You are writing personalised 2-3 sentence PhD match summaries.",
        "Use the student profile and each candidate's evidence to explain why they are a good fit.",
        "Return valid JSON with one object per candidate:",
        "[ { \"supervisor_name\": ..., \"why_match\": ... }, ... ]",
        "",
        "STUDENT PROFILE:",
        f"Name: {profile.name}",
        f"Research interests: {', '.join(profile.research_interests)}",
        f"Target countries: {', '.join(profile.target_countries)}",
        "",
        "CANDIDATES:",
    ]
    for item in batch_items:
        prompt_lines.append(json.dumps(item, ensure_ascii=False))

    prompt = "\n".join(prompt_lines)
    try:
        client, client_type = _build_llm_client()
        print(f"  [Personaliser] Using OpenAI-compatible LLM client and model {LLM_MODEL} at {LLM_API_BASE}")
    except Exception as exc:
        print(f"  [Personaliser] LLM client unavailable: {exc}")
        return ["" for _ in candidates]

    batch_items = []
    for cand in candidates:
        evidence = _build_evidence_snippet(cand)
        batch_items.append({
            "name": cand.candidate.name,
            "institution": cand.candidate.institution,
            "matched_area": cand.coverage_area,
            "evidence": evidence,
        })

    prompt_lines = [
        "You are writing personalised 2-3 sentence PhD match summaries.",
        "Use the student profile and each candidate's evidence to explain why they are a good fit.",
        "Return valid JSON with one object per candidate:",
        "[ { \"supervisor_name\": ..., \"why_match\": ... }, ... ]",
        "",
        "STUDENT PROFILE:",
        f"Name: {profile.name}",
        f"Research interests: {', '.join(profile.research_interests)}",
        f"Target countries: {', '.join(profile.target_countries)}",
        "",
        "CANDIDATES:",
    ]
    for item in batch_items:
        prompt_lines.append(json.dumps(item, ensure_ascii=False))

    prompt = "\n".join(prompt_lines)
    try:
        model_name = _resolve_llm_model()
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        content = _extract_chat_choice(response)
        parsed = json.loads(content)
        return [item.get("why_match", "") for item in parsed]
    except Exception:
        return ["" for _ in candidates]


def personalise_candidates(
    profile: StudentProfile,
    ranked_candidates: List[RankedCandidate],
) -> List[dict]:
    why_texts = _try_llm_generate(profile, ranked_candidates)
    final = []
    for idx, cand in enumerate(ranked_candidates):
        why_match = why_texts[idx] if why_texts[idx] else _fallback_why_match(profile, cand)
        evidence = [
            _normalize_evidence(item) for item in
            (cand.candidate.evidence_papers + cand.candidate.evidence_grants)
        ]
        final.append({
            "supervisor_id": cand.candidate.canonical_id,
            "name": cand.candidate.name,
            "institution": cand.candidate.institution,
            "country": cand.candidate.country,
            "contact_email": cand.candidate.email,
            "research_focus": cand.candidate.research_focus or cand.coverage_area,
            "evidence": evidence,
            "why_match": why_match,
            "tier": cand.tier,
            "linked_programs": [],
            "area": cand.coverage_area or "Unknown",
            "confidence": round(cand.score, 4),
        })
    return final
