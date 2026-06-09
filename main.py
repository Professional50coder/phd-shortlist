from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from output_schema import OutputShortlist, validate_output
from personaliser import personalise_candidates
from profile_parser import parse_profile, StudentProfile
from ranker import rank_candidates
from source_fetcher import fetch_candidates
from pi_filter import run_filters, save_rejection_log


def build_payload(
    profile: StudentProfile,
    personalised: List[dict],
    coverage_stats: dict[str, int],
) -> dict:
    return {
        "student_id": profile.student_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "target_intake": {
            "semester": profile.target_intake.semester,
            "year": profile.target_intake.year,
        },
        "coverage": coverage_stats,
        "shortlist": personalised,
    }


def write_output(payload: dict, output_path: Path) -> None:
    validated = validate_output(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(validated.model_dump_json(by_alias=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PhD supervisor shortlist from a student profile.")
    parser.add_argument("profile", help="Path to the student profile JSON file.")
    parser.add_argument("--top-n", type=int, default=30, help="Maximum number of shortlisted PIs.")
    parser.add_argument("--output", default="shortlist.json", help="Path to the output JSON file.")
    parser.add_argument("--rejection-log", default="rejection_log.json", help="Path to save filter rejection log.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = parse_profile(args.profile)

    print(f"Loading student profile: {args.profile}")
    print(f"Target countries: {profile.target_countries}")
    print(f"Research interests: {profile.research_interests}")

    candidates = fetch_candidates(profile)
    print(f"Fetched {len(candidates)} raw candidates")

    filtered, rejections = run_filters(candidates, profile, verbose=True)
    print(f"Filtered to {len(filtered)} qualified candidates")

    ranked, coverage_stats = rank_candidates(filtered, profile, top_n=args.top_n)
    print(f"Ranked top {len(ranked)} candidates")

    personalised = personalise_candidates(profile, ranked)
    payload = build_payload(profile, personalised, coverage_stats)

    write_output(payload, Path(args.output))
    save_rejection_log(rejections, args.rejection_log)

    print(f"Shortlist written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
