# v0.17.1 Player Identity Resolver

This patch fixes a critical operational issue found during safe lineup inference:
well-known current players could fall back to `generic_prior` because the current
lineup used a short public name while historical data used a longer/full display
name.

Example failure mode:

- Current lineup: `Federico Valverde`
- Historical event name: `Federico Santiago Valverde Dipetta`
- Previous behavior: `player_federico_valverde` did not match full-name ID, so
  sample minutes became 0 and generic priors were used.

## New behavior

The model now builds a conservative `PlayerIdentityResolver` from historical
player baselines. It resolves lineup players using:

1. exact `player_id_global`,
2. exact normalized name,
3. unambiguous token-subset full-name match,
4. initial + surname match,
5. unique single-token name match,
6. sequence similarity fallback when strong enough.

The resolver never adds candidates. It only maps players already supplied by
`current_lineups.csv` to historical identities.

## New inference columns

Safe player props now include:

- `resolved_player_id_global`
- `matched_player_name`
- `player_match_method`
- `player_match_confidence`
- `player_match_status`

Warnings now include:

- `unmatched_player_identity_using_prior`
- `ambiguous_player_identity_match`
- `player_identity_resolved_by_<method>`

## Diagnostic script

Use this to inspect whether important players are matched before trusting a
matchday output:

```powershell
python scripts/diagnose_player_identity.py `
  --player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --players "Federico Valverde" "Alvaro Morata" "Lamine Yamal"
```

## Audit rule

For real matchday use, any expected star/current player returning
`player_match_status != matched` or `sample_size = 0` should be treated as a data
quality issue before paper picks are trusted.
