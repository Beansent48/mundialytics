from __future__ import annotations

import pandas as pd

def infer_single_scope(matches: pd.DataFrame) -> str:
    scopes = matches.get("team_scope", pd.Series(["unknown"])).dropna().astype(str).unique().tolist()
    scopes = [s for s in scopes if s and s != "unknown"]
    return scopes[0] if len(scopes) == 1 else ("mixed" if len(scopes) > 1 else "unknown")

def validate_fixture_scope(fixtures: pd.DataFrame, expected_scope: str | None = None) -> dict:
    scope = infer_single_scope(fixtures)
    ok = expected_scope is None or scope in {expected_scope, "unknown"}
    return {"scope": scope, "expected_scope": expected_scope, "ok": ok}
