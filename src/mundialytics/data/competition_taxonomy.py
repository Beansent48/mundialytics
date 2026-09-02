from __future__ import annotations

import pandas as pd

DEFAULT_TEAM_SCOPE = "club"
DEFAULT_COMPETITION = "unknown"


def enrich_competition_metadata(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """Ensure the taxonomy columns (`team_scope`, `competition`) are present.

    `overwrite` exists because eleven call sites across the repo have always
    passed it -- three in data/adapters/api_football.py, eight in
    features/team_match_stats.py, one in scripts/calibrate_player_props.py --
    while this function only ever took `df`. Every one of them raised
    TypeError, so those paths have been dead since the initial commit (the same
    latent-breakage pattern as the `normalize_matches` import documented in
    scripts/build_foundation_big5_historical.py).

    Semantics are deliberately non-destructive, since the richer version the
    callers were written against does not exist to copy:
      overwrite=False (default)  create the columns only when absent -- exactly
                                 the previous behaviour, so nothing changes for
                                 existing callers;
      overwrite=True             additionally fill blanks (NaN/empty) inside
                                 columns that already exist.
    A real value is never replaced under either setting.
    """
    out = df.copy()
    for col, default in (("team_scope", DEFAULT_TEAM_SCOPE),
                         ("competition", DEFAULT_COMPETITION)):
        if col not in out.columns:
            out[col] = default
        elif overwrite:
            blank = out[col].isna() | (out[col].astype(str).str.strip() == "")
            out.loc[blank, col] = default
    return out
