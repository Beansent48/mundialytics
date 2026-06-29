# v0.49.8 — xG Provider Fallbacks and Understat Block Handling Spec

Status: Implemented  
Date: 2026-06-25  
Version: 0.49.8

## 1. Context

The v0.49.6 xG research downloader attempted to scrape Understat league pages by extracting `datesData` from inline page JavaScript.

A local Big 5 run returned:

```text
status = blocked
Could not find Understat datesData JSON in page.
output_rows = 0
```

This is a provider/markup problem, not a model problem. Understat direct scraping must be treated as best-effort and optional.

## 2. Decision

Keep xG as a desired Level 3 enrichment, but make the ingestion contract provider-agnostic and robust:

```text
provider/manual xG CSV
→ canonical external xG CSV
→ xG enrichment
→ model-ready snapshots
```

Rules:

- Direct Understat scraping remains optional research mode.
- If direct scraping is blocked, the pipeline must not crash with `FileNotFoundError`.
- A canonical empty `understat_xg_matches.csv` is written so batch jobs can continue and report zero xG coverage.
- Users can import any provider/manual xG CSV into the canonical contract.
- The model must never require xG to run.
- Post-match xG is never a pre-match feature directly; only rolling prior xG may become a model feature.

## 3. Implemented Changes

### 3.1 More robust Understat parser

`src/mundialytics/enrichment/understat.py` now supports several historic inline JSON variants:

```text
datesData = JSON.parse('...')
datesData = JSON.parse("...")
datesData = [...]
```

If none are present, the failure is explicit and points to provider/manual CSV import.

### 3.2 Empty canonical output on blocked scraping

When all requested league-seasons fail, the downloader now writes:

```text
data/external/xg/understat/understat_xg_matches.csv
```

with canonical headers and zero rows.

This avoids downstream `FileNotFoundError`.

### 3.3 Provider/manual CSV import

New script:

```bash
python scripts/import_xg_csv.py   --input data/external/xg/provider_export.csv   --provider provider_name   --out-dir data/external/xg/understat
```

Equivalent mode through the existing downloader:

```bash
python scripts/download_understat_xg.py   --input-csv data/external/xg/provider_export.csv   --provider provider_name   --out-dir data/external/xg/understat
```

The importer accepts common aliases such as:

```text
match_date / date
league / competition
home / home_team
away / away_team
xg_home / home_xg
xg_away / away_xg
```

and writes:

```text
understat_xg_matches.csv
understat_xg_download_report.json
```

### 3.4 Missing-xG enrichment fallback

`enrich_matches_with_xg.py` now supports:

```bash
--allow-missing-xg
```

If the xG file does not exist, it writes an enriched output with:

```text
xg_available = false
coverage_rate = 0
status = warning
```

instead of crashing.

## 4. Operational Guidance

Preferred order:

```text
1. Try direct Understat research download.
2. If blocked, import a provider/manual xG CSV.
3. If no xG source is currently available, continue with ClubElo + Football-Data features and mark xG coverage as 0.
```

Do not block ClubElo, snapshots, or validation because xG is unavailable.

## 5. Validation

Focused validation:

```text
python -m compileall -q src scripts/download_understat_xg.py scripts/import_xg_csv.py scripts/enrich_matches_with_xg.py tests/test_v0498_xg_provider_fallbacks.py
python -m pytest tests/test_v0498_xg_provider_fallbacks.py tests/test_v0496_external_data_enrichment.py tests/test_v0495_hybrid_model_ready_snapshots.py -q
```
