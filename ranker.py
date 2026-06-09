from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from profile_parser import StudentProfile
from pi_filter import FilteredCandidate


@dataclass
class RankedCandidate:
    candidate: FilteredCandidate
    score: float
    tier: str
    coverage_area: Optional[str]


def _recency_score(last_publication_year: Optional[int]) -> float:
    if last_publication_year is None:
        return 0.35
    age = max(0, datetime.now().year - last_publication_year)
    return max(0.15, 1.0 - (age / 12.0))


def _normalize_h_index(h_index: Optional[int]) -> float:
    if h_index is None:
        return 0.25
    return min(1.0, h_index / 25.0)


def _compute_score(candidate: FilteredCandidate) -> float:
    score = 0.0
    score += candidate.domain_similarity_score * 0.45
    score += candidate.career_stage_confidence * 0.25
    score += _normalize_h_index(candidate.h_index) * 0.15
    score += _recency_score(candidate.last_publication_year) * 0.10

    evidence_bonus = min(0.05, (len(candidate.evidence_papers) + len(candidate.evidence_grants)) * 0.005)
    score += evidence_bonus

    if candidate.high_collision_name:
        score -= 0.08

    score = max(0.0, min(1.0, score))
    return round(score, 4)


def _assign_tier(score: float, candidate: FilteredCandidate) -> str:
    if score >= 0.78 and not candidate.high_collision_name and candidate.career_stage_confidence >= 0.55:
        return "target"
    if score >= 0.62 and not candidate.high_collision_name:
        return "reach"
    return "safety"


def _match_area(candidate: FilteredCandidate, profile: StudentProfile) -> Optional[str]:
    normalized_interests = {interest.lower(): interest for interest in profile.research_interests}
    for area in candidate.matched_areas:
        if area and area.lower() in normalized_interests:
            return normalized_interests[area.lower()]
    return candidate.matched_areas[0] if candidate.matched_areas else None


def rank_candidates(
    candidates: List[FilteredCandidate],
    profile: StudentProfile,
    top_n: int = 30,
) -> Tuple[List[RankedCandidate], dict[str, int]]:
    ranked: List[RankedCandidate] = []
    for candidate in candidates:
        score = _compute_score(candidate)
        tier = _assign_tier(score, candidate)
        coverage_area = _match_area(candidate, profile)
        ranked.append(RankedCandidate(
            candidate=candidate,
            score=score,
            tier=tier,
            coverage_area=coverage_area,
        ))

    ranked.sort(key=lambda x: x.score, reverse=True)

    # Ensure area coverage by seeding one candidate per interest when available.
    selected: List[RankedCandidate] = []
    seen_ids = set()
    for interest in profile.research_interests:
        for rc in ranked:
            if rc.candidate.canonical_id in seen_ids:
                continue
            if rc.coverage_area and rc.coverage_area.lower() == interest.lower():
                selected.append(rc)
                seen_ids.add(rc.candidate.canonical_id)
                break

    for rc in ranked:
        if len(selected) >= top_n:
            break
        if rc.candidate.canonical_id not in seen_ids:
            selected.append(rc)
            seen_ids.add(rc.candidate.canonical_id)

    coverage_counts: dict[str, int] = {}
    for rc in selected:
        area = rc.coverage_area or "unknown"
        coverage_counts[area] = coverage_counts.get(area, 0) + 1

    return selected[:top_n], coverage_counts
