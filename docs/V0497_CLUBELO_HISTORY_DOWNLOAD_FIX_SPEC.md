# v0.49.7 — ClubElo Team-History Download Fix Spec

Status: Implemented  
Date: 2026-06-25  
Version: 0.49.7

## 1. Context

v0.49.6 introduced external ClubElo enrichment, but the first downloader implementation used one full daily ClubElo snapshot per unique match date.

That was correct in principle but impractical for multi-season Big 5 datasets:

```text
5 seasons × many match dates = hundreds of HTTP requests per league
```

The user observed that Premier League ClubElo download was taking far too long.

## 2. Decision

Change the default ClubElo download strategy from:

```text
one full ClubElo ranking snapshot per match date
```

to:

```text
one full ClubElo history per team
```

ClubElo supports both API patterns:

```text
api.clubelo.com/YYYY-MM-DD   -> one date ranking snapshot
api.clubelo.com/CLUBNAME     -> one club's full history
```

The team-history mode is the new default because it scales with team count rather than match-date count.

## 3. Implemented Behavior

### 3.1 Download

Default command:

```bash
python scripts/download_clubelo.py \
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv \
  --registry data/processed/entities/team_registry.csv \
  --out-dir data/external/clubelo
```

Default mode:

```text
--mode team-history
```

Outputs:

```text
data/external/clubelo/teams/clubelo_team_<alias>.csv
data/external/clubelo/clubelo_download_report.json
```

Legacy daily mode remains available:

```bash
python scripts/download_clubelo.py \
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv \
  --out-dir data/external/clubelo \
  --mode daily-snapshot
```

Use legacy mode only when a full daily ranking table is explicitly needed.

### 3.2 Enrichment

`enrich_matches_with_clubelo.py` now supports:

```text
--source-mode auto | team-history | daily-snapshot
```

Default `auto` prefers cached team histories when available and falls back to legacy daily snapshots otherwise.

For team histories, the join is temporal:

```text
match_date between ClubElo From and To
```

If no exact interval exists, the latest prior rating is used.

## 4. Leakage Policy

ClubElo is attached as an external pre-match strength feature. The enrichment uses only the rating interval available as of the match date.

## 5. Validation

Focused validation:

```bash
python -m compileall -q src scripts/download_clubelo.py scripts/enrich_matches_with_clubelo.py tests/test_v0496_external_data_enrichment.py
python -m pytest tests/test_v0496_external_data_enrichment.py -q
```

Expected result:

```text
all tests pass
```

Internet/API calls are not required for unit tests; they use cached simulated histories.
