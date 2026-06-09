# phd-shortlist

Automated PhD supervisor shortlist builder — given a student profile JSON, produces a ranked list of 50–200 faculty supervisors with verifiable paper/grant evidence and personalised outreach context.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env with your keys

# 3. Run on the sample student profile
python main.py student_001.json --top-n 60 --output sample_output/student_001.json
```

The pipeline completes in under 15 minutes on a single laptop. API responses are cached to `.cache/fetch/` — subsequent runs against the same profile are near-instant.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Recommended | For LLM domain classification and why_match generation (free tier available) |
| `OPENAI_API_KEY` | Alternative | OpenAI instead of Groq for LLM calls |
| `S2_API_KEY` | Optional | Semantic Scholar API key for higher rate limits (free, instant approval) |

If no LLM key is set, the pipeline falls back to template-based `why_match` generation and passes borderline domain cases without LLM adjudication.

## Data Sources

| Source | Auth | Usage |
|---|---|---|
| [Semantic Scholar](https://api.semanticscholar.org) | Optional API key | Paper search + author h-index, affiliation, pub span |
| [OpenAlex](https://openalex.org) | None | Author + institution metadata, paper evidence |
| [NIH Reporter](https://reporter.nih.gov/api) | None | US grant records (used for CA-adjacent institutions) |
| [UKRI Gateway](https://gtr.ukri.org) | None | UK grant records (skipped if UK not in target countries) |

Semantic Scholar and OpenAlex are the primary sources. NIH and UKRI supplement grant evidence. No scraping — all official free APIs.

## Pipeline

```
student.json
    ↓
01 profile_parser.py     Extract hard constraints (countries, intake) + soft signals (areas, keywords)
    ↓
02 source_fetcher.py     Async fetch from S2, OpenAlex, NIH, UKRI — disk-cached per (source, query, countries)
    ↓
03 pi_filter.py          4-stage contamination firewall — country → career-stage → domain → identity
    ↓
04 ranker.py             Score by (domain_sim × 0.45) + (career_conf × 0.25) + (h_index × 0.15) + recency
    ↓
05 personaliser.py       LLM why_match generation (batched 10 per call); template fallback if no API key
    ↓
06 output_schema.py      Pydantic validation → shortlist.json
```

## CLI Options

```
python main.py <profile_json> [--top-n N] [--output PATH] [--rejection-log PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--top-n` | 30 | Maximum shortlist entries |
| `--output` | `shortlist.json` | Output path |
| `--rejection-log` | `rejection_log.json` | Filter rejection audit log |

## Output

See `sample_output/student_001.json` for a full example output and `schema.md` for the documented schema.

## Data Quality Design

Four contamination risks are addressed explicitly — see `DECISIONS.md` for full trade-off reasoning:

1. **Career-stage errors** — PhD students and postdocs appear as first authors. Filtered by: explicit title keywords → h-index → publication span → paper count (multi-signal, not hard threshold).
2. **Wrong-domain leakage** — keyword overlap across disciplines. Filtered by: bag-of-words cosine similarity (hard pass ≥ 0.52, hard fail < 0.32) + LLM binary classifier for borderline cases.
3. **Same-name collisions** — common surnames (Wang, Sharma, Kim). Flagged with `high_collision_name=true` and deprioritised in ranking; not hard-rejected.
4. **Personal fellowship misidentification** — NIH F31/F32, MSCA-PD list the trainee not the supervisor. Rejected when all evidence is personal fellowships with no paper record.

All rejections are logged to `rejection_log.json` with reason and detail for auditability.

## Known Limitations

- **Coverage depends on API availability** — S2 and OpenAlex don't index every institution uniformly. Some strong labs in DE and SG are underrepresented vs AU/CA.
- **Email extraction is conservative** — `contact_email` is null for most entries. The pipeline preserves null rather than guessing, per the trade-off documented in DECISIONS.md.
- **why_match quality** — without an LLM key, fallback templates are generic. With Groq (free), quality improves significantly.
- **sentence-transformers not loaded by default** — the embedding falls back to bag-of-words if the library isn't installed. Install `sentence-transformers` for better domain filtering.
- **Windows encoding** — log output may error on Windows terminals with the `→` character. Redirect to file: `python main.py student_001.json > output.log 2>&1`.

## Repo Structure

```
phd-shortlist/
  main.py                  # Single-command entry point
  profile_parser.py        # Parse student JSON → StudentProfile dataclass
  source_fetcher.py        # Async fetch from S2, OpenAlex, NIH, UKRI
  pi_filter.py             # 4-stage contamination firewall
  ranker.py                # Score + tier assignment
  personaliser.py          # LLM why_match generation
  output_schema.py         # Pydantic schema + validation
  student_001.json         # Sample student profile (input)
  sample_output/
    student_001.json       # Full shortlist output for sample student
  .env.example             # Required environment variables
  requirements.txt
  README.md
  DECISIONS.md             # Trade-off writeup
  schema.md                # Output JSON schema documentation
```
