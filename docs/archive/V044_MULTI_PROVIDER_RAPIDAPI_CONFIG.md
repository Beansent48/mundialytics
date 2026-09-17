# v0.44 — Multi-provider RapidAPI config

This version separates API credentials/config from code updates.

## Providers

- `oddspapi`: odds provider. Used for fixtures, markets, current odds, historical odds, CLV and training odds features.
- `creativesdev_live_football`: football-data provider candidate. Used for fixtures, lineups, events and statistics after endpoint coverage is proven.

## Important design choice

The local config is not shipped as a real-key file inside update ZIPs. Put it at:

```text
config/mundialytics_api_config.local.yaml
```

or point to it with:

```powershell
$env:MUNDIALYTICS_API_CONFIG="C:\Users\Vicente\Desktop\BetBot\mundialytics_betting_engine\config\mundialytics_api_config.local.yaml"
```

## Why Creativesdev endpoint paths are external

The public RapidAPI listing confirms that the API exposes football data such as fixtures, events, line-ups, statistics, standings and odds, but the exact endpoint paths are controlled by RapidAPI Playground and can change. Therefore, the code reads Creativesdev endpoint paths from the local config instead of hard-coding guesses.

## New scripts

```powershell
python scripts/provider_config_check.py --config config/mundialytics_api_config.local.yaml
python scripts/provider_rapidapi_audit.py --config config/mundialytics_api_config.local.yaml
python scripts/creativesdev_probe.py --config config/mundialytics_api_config.local.yaml --endpoint-key fixtures_by_date --vars date=2026-06-23 --dry-run
```

## Training rule

Do not train betting ROI/EV models with post-kickoff or post-match odds. Use pre-kickoff snapshots aligned to the intended betting time, e.g. 60 minutes pre-match.
