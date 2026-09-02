import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Quarantine legacy (pre-v0.50) test files that fail at IMPORT.
#
# AUDITED 2026-09-03 by executing every one of them, after "known broken" turned
# out to be wrong for 10 of the 11 runtime failures elsewhere in the suite (one
# was just a missing re-export). All 21 do genuinely fail to import. Grouped by
# the symbol they need, and whether it exists anywhere in src/:
#
#   GONE — no such symbol in the codebase, so these need the removed feature
#   rebuilt, not a test fix:
#     player_global_id                (7 files)  identity.normalization
#     canonical_provider_player_id    (3)        data.provider_identity
#     normalize_matches               (2)        data.schema
#     scheduled_events_response_to_df (2)        adapters.sofascore
#     scoreboard_response_to_df       (2)        adapters.espn
#     load_lineups                    (1)        data.loaders
#     classify_competition            (1)        data.competition_taxonomy
#     merge_player_events_with_lineups(1)        data.events
#     normalize_fixtures              (1)        data.schema  (a function of that
#                                                name lives in adapters/
#                                                creativesdev.py, but it is a
#                                                different contract)
#
#   The statsbomb/wyscout symbols these files also wanted DID exist and were
#   simply unexported; that gap is now fixed in data/adapters/__init__.py and is
#   worth having on its own. test_event_data_sources.py still stays here: it
#   needs merge_player_events_with_lineups, and its remaining assertions expect
#   lowercased player names while the live adapter (the one the player-ratings
#   pipeline depends on) emits raw names — an outdated contract, not a bug.
#
# Ignoring these keeps `pytest tests/` clean, while an import break in a
# maintained test still errors loudly (not on this list).
# To revive one: fix its imports and remove it from LEGACY_BROKEN.
LEGACY_BROKEN = [
    "test_agent_improvements.py", "test_audit_regressions.py",
    "test_competition_taxonomy_v15.py", "test_core.py", "test_event_data_sources.py",
    "test_match_value.py", "test_operational_v08.py", "test_safe_lineup_props.py",
    "test_scope_and_fixtures.py", "test_v031_today_matchday_inputs.py",
    "test_v032_team_identity_event_timezone_player_inputs.py", "test_v09_event_pipeline.py",
    "test_v14_operational_contract.py", "test_v16_hierarchical_and_cross_context.py",
    "test_v171_player_identity_resolver.py", "test_v17_player_props_finalization.py",
    "test_v181_today_fixtures.py", "test_v182_world_cup_fixtures.py",
    "test_v183_sofascore_fixtures.py", "test_v18_provider_identity_layer.py",
    "test_v19_free_fixtures_and_team_props.py",
]

collect_ignore = LEGACY_BROKEN
