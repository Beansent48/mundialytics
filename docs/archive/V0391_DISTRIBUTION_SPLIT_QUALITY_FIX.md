# v0.39.1 Distribution Split + Target Quality Fix

This patch fixes two issues found in the first v0.39 market distribution lab run:

1. **StatsBomb real-target rows were test-only** because some raw event-derived rows had missing dates in the line-signal file. The analyzer now supports `--split-mode auto|chronological|hash|stratified_hash`; `auto` uses deterministic match-id hash splitting when date coverage is poor and chronological splitting otherwise. Date-less matches no longer all fall into test.

2. **Team-level source quality was too often `unknown_quality`**. The analyzer now infers target quality from `data_quality_flag` and `saves_data_quality_flag` when older signal files did not label it cleanly. Direct boxscore/event/provider targets are upgraded to `real_target` or `match_total`; SOT-minus-goals saves are labelled `derived_target`.

It also improves StatsBomb raw extra-team stats metadata handling by reading the `.metadata.json` sidecars written by the downloader, so dates, competition, season and home/away teams are preserved when rebuilding `statsbomb_raw_extra_match_stats.csv`.

Recommended immediate command on an existing large signal file:

```powershell
python scripts/analyze_market_distribution_lab.py `
  --line-signals outputs/event_line_backtest_current/settled_event_line_signals.csv `
  --out-dir outputs/market_distribution_lab_current_v0391 `
  --line-min-model-prob 0.52 `
  --split-mode auto `
  --min-sample 100
```

For a purely diagnostic non-chronological check across every target-quality slice:

```powershell
python scripts/analyze_market_distribution_lab.py `
  --line-signals outputs/event_line_backtest_current/settled_event_line_signals.csv `
  --out-dir outputs/market_distribution_lab_hash_v0391 `
  --line-min-model-prob 0.52 `
  --split-mode hash `
  --min-sample 100
```

For the cleanest future run, rebuild StatsBomb extra stats and event-line signals after this patch so the metadata is present before the line backtest.
