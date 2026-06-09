"""
pipeline/source_fetcher.py

Fetches raw candidate supervisor data from multiple sources concurrently.
Returns a list of RawCandidate objects — unfiltered, unranked.
pi_filter.py does all the quality work downstream.

Sources:
  1. Semantic Scholar (S2)  — papers by keyword, author metadata
  2. OpenAlex               — papers + author institutional affiliation
  3. NIH Reporter           — US grants (used for CA/US-adjacent; skipped for AU/NL/DE/SG)
  4. UKRI Gateway           — UK grants (skipped if UK not in target_countries)

Design decisions:
  - All HTTP calls are async (aiohttp) — all sources fetched concurrently per keyword.
  - Responses are cached to disk (JSON) keyed by (source, query, country_filter).
    Same input → same output on re-run. Required for reproducibility (requirement 6).
  - We fetch WIDE here — pi_filter.py enforces precision.
    Better to over-fetch and filter than under-fetch and miss real PIs.
  - Author deduplication by source-specific ID happens here to avoid sending
    the same person to pi_filter.py 10 times (one per keyword hit).
  - Rate limits: S2 = 100 req/5min unauthenticated, 1 req/sec with key.
    OpenAlex = 100k req/day unauthenticated. We add small delays to be safe.
  - NIH / UKRI are grant sources — authors on these are NOT assumed to be PIs.
    That check lives in pi_filter.py (career-stage filter, challenge 6.2).
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import aiohttp

from profile_parser import StudentProfile

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = Path(".cache/fetch")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optional API key for higher S2 rate limits — set via env var
S2_API_KEY = os.getenv("S2_API_KEY", "")

# How many papers to pull per keyword per source
PAPERS_PER_KEYWORD = 20

# How many authors to extract per paper (lead + corresponding only)
MAX_AUTHORS_PER_PAPER = 3

# Concurrency cap — avoids hammering APIs
MAX_CONCURRENT = 8

# Country code → institution domain hints for pre-filtering
# Used as a soft signal only; pi_filter.py does the real country check
COUNTRY_DOMAIN_HINTS: dict[str, list[str]] = {
    "AU": [".edu.au", "australia", "monash", "unsw", "melbourne", "uq.edu", "anu.edu", "queensland"],
    "CA": [".ca", "canada", "ubc", "utoronto", "mcgill", "uwaterloo", "ualberta"],
    "NL": [".nl", "netherlands", "delft", "leiden", "radboud", "vu.nl", "uva.nl"],
    "DE": [".de", "germany", "tum.de", "kit.edu", "charite", "helmholtz", "mpg.de"],
    "SG": [".sg", "singapore", "nus.edu", "ntu.edu", "a-star", "duke-nus"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawCandidate:
    """
    A single candidate supervisor record as returned from a source.
    Deliberately loose — pi_filter.py will validate and enrich this.
    """
    source: str                         # "semantic_scholar" | "openalex" | "nih" | "ukri"
    source_author_id: str               # Source-specific stable ID (for dedup)
    name: str
    institution: Optional[str]
    institution_url: Optional[str]
    country_hint: Optional[str]         # ISO2 if inferable, else None
    email: Optional[str]
    homepage_url: Optional[str]
    research_focus: Optional[str]       # Free text from profile/affiliation

    # Evidence
    evidence_papers: list[dict]         # [{title, year, url, venue, abstract}]
    evidence_grants: list[dict]         # [{title, year, url, funder, abstract}]

    # Which student search area triggered this candidate
    matched_area: str
    matched_keyword: str

    # Raw fields for pi_filter.py to use in career-stage check
    total_paper_count: Optional[int]
    first_publication_year: Optional[int]
    last_publication_year: Optional[int]
    h_index: Optional[int]
    position_title: Optional[str]       # "Professor", "PhD Student", etc. if available


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(source: str, query: str, countries: list[str]) -> str:
    raw = f"{source}::{query}::{'_'.join(sorted(countries))}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> Optional[list[dict]]:
    p = _cache_path(key)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(key: str, data: list[dict]) -> None:
    with open(_cache_path(key), "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _parse_json_response(response: aiohttp.ClientResponse, source_label: str) -> Optional[dict]:
    try:
        data = await response.json()
    except Exception as e:
        print(f"  [{source_label}] JSON parse failed: {e}")
        return None
    if not isinstance(data, dict):
        print(f"  [{source_label}] Unexpected JSON payload: {type(data).__name__}")
        return None
    return data


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

async def _fetch_s2_papers(
    session: aiohttp.ClientSession,
    keyword: str,
    area: str,
    target_countries: list[str],
) -> list[RawCandidate]:
    """
    Search S2 for papers matching keyword.
    Extract up to MAX_AUTHORS_PER_PAPER authors per paper as candidate supervisors.

    S2 /paper/search returns papers; each paper has authors with S2 author IDs.
    We then hit /author/{id} for profile metadata (h-index, paper count, affiliation).

    Rate limit: add 0.1s delay between author lookups.
    """
    cache_key = _cache_key("s2", keyword, target_countries)
    cached = _load_cache(cache_key)
    if cached is not None:
        return [RawCandidate(**c) for c in cached]

    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    candidates: dict[str, RawCandidate] = {}  # keyed by s2_author_id

    try:
        # Step 1: search papers by keyword
        params = {
            "query": keyword,
            "limit": PAPERS_PER_KEYWORD,
            "fields": "title,year,venue,abstract,authors,externalIds",
        }
        async with session.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 429:
                print(f"  [S2] Rate limited for '{keyword}', retrying after backoff")
                await asyncio.sleep(2)
                async with session.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as retry_resp:
                    if retry_resp.status != 200:
                        print(f"  [S2] Non-200 after retry for '{keyword}': {retry_resp.status}")
                        return []
                    data = await _parse_json_response(retry_resp, "S2")
            elif resp.status != 200:
                print(f"  [S2] Non-200 for '{keyword}': {resp.status}")
                return []
            else:
                data = await _parse_json_response(resp, "S2")
            if data is None:
                return []

        papers = data.get("data", [])
        if not papers:
            return []

        # Step 2: for each paper, fetch top authors
        author_ids_seen: set[str] = set()
        author_paper_map: dict[str, list[dict]] = {}  # author_id → papers they appear in

        for paper in papers:
            paper_evidence = {
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
                "venue": paper.get("venue", ""),
                "abstract": (paper.get("abstract") or "")[:500],  # truncate for storage
            }
            for author in (paper.get("authors") or [])[:MAX_AUTHORS_PER_PAPER]:
                if not isinstance(author, dict):
                    continue
                aid = author.get("authorId")
                if not aid:
                    continue
                if aid in author_ids_seen:
                    author_paper_map[aid].append(paper_evidence)
                    continue
                author_ids_seen.add(aid)
                author_paper_map[aid] = [paper_evidence]

        # Step 3: fetch author profiles (h-index, affiliation, etc.)
        await asyncio.sleep(0.1)  # be polite to S2

        for author_id, papers_list in author_paper_map.items():
            try:
                async with session.get(
                    f"https://api.semanticscholar.org/graph/v1/author/{author_id}",
                    params={
                        "fields": "name,affiliations,homepage,hIndex,paperCount,"
                                  "citationCount,papers.year,externalIds"
                    },
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as aresp:
                    if aresp.status != 200:
                        continue
                    adata = await _parse_json_response(aresp, "S2")
                    if adata is None:
                        continue

                affiliations = adata.get("affiliations") or []
                institution = affiliations[0].get("name") if affiliations else None

                # Soft country hint from institution string
                country_hint = _guess_country(institution or "", target_countries)

                # Publication year range for career-stage check
                pub_years = [
                    p["year"] for p in (adata.get("papers") or [])
                    if p.get("year")
                ]

                candidates[author_id] = RawCandidate(
                    source="semantic_scholar",
                    source_author_id=author_id,
                    name=adata.get("name", ""),
                    institution=institution,
                    institution_url=None,
                    country_hint=country_hint,
                    email=None,  # S2 doesn't provide emails
                    homepage_url=adata.get("homepage"),
                    research_focus=None,
                    evidence_papers=papers_list,
                    evidence_grants=[],
                    matched_area=area,
                    matched_keyword=keyword,
                    total_paper_count=adata.get("paperCount"),
                    first_publication_year=min(pub_years) if pub_years else None,
                    last_publication_year=max(pub_years) if pub_years else None,
                    h_index=adata.get("hIndex"),
                    position_title=None,
                )
                await asyncio.sleep(0.05)  # rate limit: ~20 author lookups/sec

            except Exception as e:
                print(f"  [S2] Author fetch failed for {author_id}: {e}")
                continue

    except Exception as e:
        print(f"  [S2] Search failed for '{keyword}': {e}")
        return []

    result = list(candidates.values())
    _save_cache(cache_key, [asdict(c) for c in result])
    return result


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

async def _fetch_openalex_papers(
    session: aiohttp.ClientSession,
    keyword: str,
    area: str,
    target_countries: list[str],
) -> list[RawCandidate]:
    """
    OpenAlex /works search by keyword, filter by institution country.
    OpenAlex has a country_code field on institutions — we use it as a pre-filter
    to reduce noise (not a hard country filter — that's pi_filter.py's job).

    OpenAlex is generous: 100k requests/day free, no key needed.
    Returns richer institutional data than S2 (ROR IDs, country codes).
    """
    cache_key = _cache_key("openalex", keyword, target_countries)
    cached = _load_cache(cache_key)
    if cached is not None:
        return [RawCandidate(**c) for c in cached]

    candidates: dict[str, RawCandidate] = {}

    try:
        # Build country filter — OpenAlex supports |OR| syntax
        country_filter = "|".join(target_countries)

        params = {
            "search": keyword,
            "filter": f"authorships.institutions.country_code:{country_filter}",
            "per-page": PAPERS_PER_KEYWORD,
            "select": "id,title,publication_year,primary_location,authorships,abstract_inverted_index",
            "mailto": "phd-shortlist@example.com",  # OpenAlex polite pool
        }
        async with session.get(
            "https://api.openalex.org/works",
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                print(f"  [OA] Non-200 for '{keyword}': {resp.status}")
                return []
            data = await _parse_json_response(resp, "OA")
            if data is None:
                return []

        works = data.get("results", [])
        if not isinstance(works, list):
            print(f"  [OA] Unexpected results format for '{keyword}': {type(works).__name__}")
            return []

        for work in works:
            if not isinstance(work, dict):
                continue
            primary_location = work.get("primary_location") or {}
            if not isinstance(primary_location, dict):
                primary_location = {}
            paper_evidence = {
                "title": work.get("title", ""),
                "year": work.get("publication_year"),
                "url": primary_location.get("landing_page_url") or work.get("id", ""),
                "venue": (primary_location.get("source") or {}).get("display_name", ""),
                "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
            }

            for authorship in (work.get("authorships") or [])[:MAX_AUTHORS_PER_PAPER]:
                if not isinstance(authorship, dict):
                    continue
                author = authorship.get("author")
                if not isinstance(author, dict):
                    continue
                author_id_raw = author.get("id")
                oa_id = (author_id_raw or "").replace("https://openalex.org/", "")
                if not oa_id:
                    continue
                if oa_id in candidates:
                    candidates[oa_id].evidence_papers.append(paper_evidence)
                    continue

                # Institution from authorship
                institutions = authorship.get("institutions") or []
                inst = institutions[0] if institutions else {}
                institution_name = inst.get("display_name")
                country_code = inst.get("country_code")  # already ISO2
                ror_url = inst.get("ror")

                # Only include if country matches target (soft pre-filter)
                if country_code and country_code not in target_countries:
                    continue

                candidates[oa_id] = RawCandidate(
                    source="openalex",
                    source_author_id=oa_id,
                    name=author.get("display_name", ""),
                    institution=institution_name,
                    institution_url=ror_url,
                    country_hint=country_code,
                    email=None,
                    homepage_url=None,
                    research_focus=None,
                    evidence_papers=[paper_evidence],
                    evidence_grants=[],
                    matched_area=area,
                    matched_keyword=keyword,
                    total_paper_count=None,
                    first_publication_year=None,
                    last_publication_year=None,
                    h_index=None,
                    position_title=None,
                )

        # Enrich with author-level data (h-index, paper count, position)
        for oa_id in list(candidates.keys()):
            try:
                async with session.get(
                    f"https://api.openalex.org/authors/{oa_id}",
                    params={"mailto": "phd-shortlist@example.com"},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as aresp:
                    if aresp.status != 200:
                        continue
                    ad = await aresp.json()

                candidates[oa_id].h_index = ad.get("summary_stats", {}).get("h_index")
                candidates[oa_id].total_paper_count = ad.get("works_count")

                pub_years = [
                    y for y in [
                        ad.get("counts_by_year", [{}])[0].get("year") if ad.get("counts_by_year") else None,
                    ] if y
                ]
                if pub_years:
                    candidates[oa_id].last_publication_year = max(pub_years)

                await asyncio.sleep(0.05)

            except Exception:
                continue

    except Exception as e:
        print(f"  [OA] Search failed for '{keyword}': {e}")
        return []

    result = list(candidates.values())
    _save_cache(cache_key, [asdict(c) for c in result])
    return result


def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """
    OpenAlex stores abstracts as inverted index {word: [positions]}.
    Reconstruct to plain text for domain-check embedding later.
    """
    if not inverted_index:
        return ""
    position_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    return " ".join(position_word[i] for i in sorted(position_word))[:500]


# ---------------------------------------------------------------------------
# NIH Reporter
# ---------------------------------------------------------------------------

async def _fetch_nih_grants(
    session: aiohttp.ClientSession,
    keyword: str,
    area: str,
    target_countries: list[str],
) -> list[RawCandidate]:
    """
    NIH Reporter API — only relevant if CA is in target_countries.
    (NIH funds US and some Canadian research, but mostly US.)

    CRITICAL design decision (challenge 6.2):
    NIH grants include F31/F32 (pre-doctoral/postdoctoral fellowships) where the
    PI listed is the FELLOW, not the supervisor. We tag these grant types explicitly
    so pi_filter.py can exclude them from PI candidacy.

    We do NOT attempt to identify the actual supervisor from F-series grants —
    that information is not in the grant record and would require hallucination.
    """
    # NIH only useful for US/CA targets
    if not any(c in target_countries for c in ["US", "CA"]):
        return []

    cache_key = _cache_key("nih", keyword, target_countries)
    cached = _load_cache(cache_key)
    if cached is not None:
        return [RawCandidate(**c) for c in cached]

    candidates: dict[str, RawCandidate] = {}

    try:
        payload = {
            "criteria": {
                "advanced_text_search": {
                    "operator": "Advanced",
                    "search_field": "all",
                    "search_text": keyword,
                },
            },
            "include_fields": [
                "ProjectTitle", "AbstractText", "PrincipalInvestigators",
                "Organization", "AwardAmount", "FiscalYear", "ProjectStartDate",
                "ProjectEndDate", "FullStudySection", "ActivityCode",
                "ProjectNumber", "ContactPiName",
            ],
            "offset": 0,
            "limit": 20,
            "sort_field": "project_start_date",
            "sort_order": "desc",
        }

        async with session.post(
            "https://api.reporter.nih.gov/v2/projects/search",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                print(f"  [NIH] Non-200 for '{keyword}': {resp.status}")
                return []
            data = await resp.json()

        for project in data.get("results", []):
            activity_code = project.get("activity_code", "")

            # Flag personal fellowships — DO NOT treat as PI candidacy
            # F31=pre-doc, F32=postdoc, K99=career transition
            # These list the FELLOW/TRAINEE, not the supervising PI
            is_personal_fellowship = activity_code in {
                "F30", "F31", "F32", "F33",   # fellowships
                "K99", "K00",                  # career transition
                "T32", "T34",                  # training grants (institution, not PI)
            }

            pis = project.get("principal_investigators") or []
            for pi in pis[:1]:  # take contact PI only
                name = pi.get("full_name", "").strip()
                if not name:
                    continue

                # Use project number as stable ID (NIH has no author ID system)
                proj_num = project.get("project_number", "")
                candidate_id = f"nih_{hashlib.md5(name.encode()).hexdigest()[:8]}"

                grant_evidence = {
                    "title": project.get("project_title", ""),
                    "year": project.get("fiscal_year"),
                    "url": f"https://reporter.nih.gov/project-details/{proj_num}",
                    "funder": "NIH",
                    "abstract": (project.get("abstract_text") or "")[:500],
                    "activity_code": activity_code,
                    "is_personal_fellowship": is_personal_fellowship,  # KEY FLAG
                }

                org = project.get("organization") or {}
                institution = org.get("org_name")
                country_hint = "US"  # NIH is overwhelmingly US

                if candidate_id not in candidates:
                    candidates[candidate_id] = RawCandidate(
                        source="nih",
                        source_author_id=candidate_id,
                        name=name,
                        institution=institution,
                        institution_url=None,
                        country_hint=country_hint,
                        email=pi.get("email"),
                        homepage_url=None,
                        research_focus=None,
                        evidence_papers=[],
                        evidence_grants=[grant_evidence],
                        matched_area=area,
                        matched_keyword=keyword,
                        total_paper_count=None,
                        first_publication_year=None,
                        last_publication_year=None,
                        h_index=None,
                        position_title=pi.get("title"),
                    )
                else:
                    candidates[candidate_id].evidence_grants.append(grant_evidence)

    except Exception as e:
        print(f"  [NIH] Search failed for '{keyword}': {e}")
        return []

    result = list(candidates.values())
    _save_cache(cache_key, [asdict(c) for c in result])
    return result


# ---------------------------------------------------------------------------
# UKRI Gateway
# ---------------------------------------------------------------------------

async def _fetch_ukri_grants(
    session: aiohttp.ClientSession,
    keyword: str,
    area: str,
    target_countries: list[str],
) -> list[RawCandidate]:
    """
    UKRI Gateway to Research API — only if GB/UK in target_countries.
    Similar fellowship-exclusion logic as NIH.
    UKRI MSCA postdoctoral grants list the postdoc, not the supervisor.
    """
    if "GB" not in target_countries and "UK" not in target_countries:
        return []

    cache_key = _cache_key("ukri", keyword, target_countries)
    cached = _load_cache(cache_key)
    if cached is not None:
        return [RawCandidate(**c) for c in cached]

    candidates: dict[str, RawCandidate] = {}

    try:
        params = {
            "q": keyword,
            "size": 20,
            "page": 1,
        }
        async with session.get(
            "https://gtr.ukri.org/gtr/api/projects",
            params=params,
            headers={"Accept": "application/vnd.rcuk.gtr.json-v7"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                print(f"  [UKRI] Non-200 for '{keyword}': {resp.status}")
                return []
            data = await resp.json()

        for project in (data.get("project") or []):
            # UKRI MSCA postdoctoral = personal fellowship, exclude from PI candidacy
            grant_category = project.get("grantCategory", "")
            is_personal_fellowship = "Postdoctoral" in grant_category or \
                                     "Fellowship" in grant_category or \
                                     "Studentship" in grant_category

            pi_info = (project.get("principalInvestigator") or {})
            name = (
                f"{pi_info.get('firstName', '')} {pi_info.get('surname', '')}".strip()
            )
            if not name:
                continue

            candidate_id = f"ukri_{hashlib.md5(name.encode()).hexdigest()[:8]}"

            grant_evidence = {
                "title": project.get("title", ""),
                "year": (project.get("startDate") or "")[:4] or None,
                "url": f"https://gtr.ukri.org/projects?ref={project.get('id', '')}",
                "funder": project.get("funder", {}).get("name", "UKRI"),
                "abstract": (project.get("abstractText") or "")[:500],
                "is_personal_fellowship": is_personal_fellowship,
            }

            org = pi_info.get("organisation") or {}

            if candidate_id not in candidates:
                candidates[candidate_id] = RawCandidate(
                    source="ukri",
                    source_author_id=candidate_id,
                    name=name,
                    institution=org.get("name"),
                    institution_url=None,
                    country_hint="GB",
                    email=pi_info.get("email"),
                    homepage_url=None,
                    research_focus=None,
                    evidence_papers=[],
                    evidence_grants=[grant_evidence],
                    matched_area=area,
                    matched_keyword=keyword,
                    total_paper_count=None,
                    first_publication_year=None,
                    last_publication_year=None,
                    h_index=None,
                    position_title=None,
                )
            else:
                candidates[candidate_id].evidence_grants.append(grant_evidence)

    except Exception as e:
        print(f"  [UKRI] Search failed for '{keyword}': {e}")
        return []

    result = list(candidates.values())
    _save_cache(cache_key, [asdict(c) for c in result])
    return result


# ---------------------------------------------------------------------------
# Country hint helper
# ---------------------------------------------------------------------------

def _guess_country(institution_str: str, target_countries: list[str]) -> Optional[str]:
    """
    Soft heuristic: check institution string against known domain/name patterns.
    Returns ISO2 country code if confident, else None.
    pi_filter.py does the authoritative country check.
    """
    s = institution_str.lower()
    for country, hints in COUNTRY_DOMAIN_HINTS.items():
        if country in target_countries:
            if any(h in s for h in hints):
                return country
    return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def fetch_all_candidates(
    profile: StudentProfile,
    sources: Optional[list[str]] = None,
) -> list[RawCandidate]:
    """
    Fetch candidates from all sources for all research interests concurrently.

    Strategy:
    - For each (area, keyword) pair, fire requests to S2 + OpenAlex in parallel.
    - NIH / UKRI only if relevant target countries are present.
    - Deduplicate by (source, source_author_id) WITHIN a source.
      Cross-source deduplication by name+institution happens in pi_filter.py
      (needs richer data to do safely).

    Returns a flat list of RawCandidate — all sources, all keywords merged.
    """
    if sources is None:
        sources = ["semantic_scholar", "openalex", "nih", "ukri"]

    if "semantic_scholar" in sources and not S2_API_KEY:
        print("  [WARN] No S2_API_KEY set; skipping Semantic Scholar source to avoid unauthenticated rate limits.")
        sources = [s for s in sources if s != "semantic_scholar"]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _guarded(coro):
        async with semaphore:
            return await coro

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:

        tasks = []

        # Use a subset of keywords to stay within latency budget:
        # - All verbatim research interests (high precision)
        # - First 2 synonyms per interest (broadens coverage without explosion)
        keywords_per_area: dict[str, list[str]] = {}
        for interest in profile.research_interests:
            kws = [interest]
            # find synonyms from the expanded list that aren't the verbatim interest
            synonyms = [kw for kw in profile.search_keywords if kw != interest]
            # take up to 2 synonyms per area
            kws += synonyms[:2]
            keywords_per_area[interest] = kws

        for area, keywords in keywords_per_area.items():
            for keyword in keywords:
                if "semantic_scholar" in sources:
                    tasks.append(_guarded(
                        _fetch_s2_papers(session, keyword, area, profile.target_countries)
                    ))
                if "openalex" in sources:
                    tasks.append(_guarded(
                        _fetch_openalex_papers(session, keyword, area, profile.target_countries)
                    ))
                if "nih" in sources:
                    tasks.append(_guarded(
                        _fetch_nih_grants(session, keyword, area, profile.target_countries)
                    ))
                if "ukri" in sources:
                    tasks.append(_guarded(
                        _fetch_ukri_grants(session, keyword, area, profile.target_countries)
                    ))

        print(f"  Firing {len(tasks)} fetch tasks across "
              f"{len(keywords_per_area)} areas "
              f"({sum(len(v) for v in keywords_per_area.values())} keywords) ...")

        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start

        all_candidates: list[RawCandidate] = []
        errors = 0
        for r in results:
            if isinstance(r, Exception):
                errors += 1
            elif isinstance(r, list):
                all_candidates.extend(r)

        # Dedup within-source by source_author_id
        seen: set[str] = set()
        deduped: list[RawCandidate] = []
        for c in all_candidates:
            key = f"{c.source}::{c.source_author_id}"
            if key not in seen:
                seen.add(key)
                deduped.append(c)
            else:
                # Merge evidence onto existing record
                existing = next(x for x in deduped if f"{x.source}::{x.source_author_id}" == key)
                existing.evidence_papers.extend(c.evidence_papers)
                existing.evidence_grants.extend(c.evidence_grants)

        print(f"  Fetched {len(all_candidates)} raw hits → "
              f"{len(deduped)} unique after within-source dedup "
              f"({errors} errors) in {elapsed:.1f}s")

        return deduped


def fetch_candidates(profile: StudentProfile, sources: Optional[list[str]] = None) -> list[RawCandidate]:
    """Synchronous wrapper for use from main.py or tests."""
    return asyncio.run(fetch_all_candidates(profile, sources))


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    profile_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input/student_001.json"
    print(f"\nLoading profile: {profile_path}")
    from profile_parser import parse_profile
    profile = parse_profile(profile_path)

    print(f"Target countries: {profile.target_countries}")
    print(f"Research areas: {profile.research_interests}")
    print(f"\nFetching candidates ...\n")

    candidates = fetch_candidates(profile, sources=["semantic_scholar", "openalex"])

    print(f"\n{'='*60}")
    print(f"  Total candidates: {len(candidates)}")
    by_area: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for c in candidates:
        by_area[c.matched_area] = by_area.get(c.matched_area, 0) + 1
        by_source[c.source] = by_source.get(c.source, 0) + 1
    print(f"  By source: {by_source}")
    print(f"  By area:")
    for area, count in by_area.items():
        print(f"    {area}: {count}")

    # Show first 3 as samples
    print(f"\n  Sample candidates:")
    for c in candidates[:3]:
        print(f"    - {c.name} | {c.institution} | {c.country_hint} | "
              f"h={c.h_index} | papers={c.total_paper_count} | "
              f"evidence={len(c.evidence_papers)}p+{len(c.evidence_grants)}g")
    print(f"{'='*60}\n")
    print("✓ source_fetcher.py done. Ready for pi_filter.py\n")