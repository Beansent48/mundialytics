"""Prices for a team the engine has never seen, so no fixture goes unpredicted.

A club new to the Big Five (Elversberg in 2026/27, one to three a season) has
no rows in the foundation until its first match lands there, a round or two
after it is played. The logger used to skip those fixtures -- every debutant's
opening matchdays were lost to the track record -- and the engine itself is
worse than skipping: an unknown name silently takes team index 0, i.e. another
club's ratings.

The stand-in is the clubs that went down from the same league last season: the
level a promoted side replaces. Each is priced against the real opponent and
the lambdas are averaged, then every market is rebuilt from those lambdas with
the deployed parameters. Validated before use (scripts/experiment_debutant_proxy.py).
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

STAT_KEYS = ("shots", "sot", "corners", "fouls", "yellows")


def relegated_last_season(df: pd.DataFrame, competition: str, season: str,
                          current_teams: set[str]) -> list[str]:
    """Clubs in `competition` the season before `season` that are not in it now.

    `current_teams` comes from the calendar, not the foundation: on matchday one
    the foundation holds no match of the new season yet.
    """
    d = df[(df["competition"] == competition) & (df["season"] < season)]
    if d.empty:
        return []
    prev = d[d["season"] == d["season"].max()]
    return sorted((set(prev["home_team"]) | set(prev["away_team"])) - set(current_teams))


def predict_match_or_proxy(eng, df: pd.DataFrame, home: str, away: str,
                           competition: str, season: str, current_teams: set[str],
                           known: set[str] | None = None):
    """(prediction, stand-ins) -- stand-ins is {} unless a side was unknown.

    The prediction has the attributes the logger and the API read off a
    MatchPrediction.
    """
    known = known if known is not None else set(df["home_team"]) | set(df["away_team"])
    if home in known and away in known:
        return eng.predict_match(home, away, competition=competition, neutral=False), {}

    proxies = [t for t in relegated_last_season(df, competition, season, current_teams)
               if t in known]
    if not proxies:
        raise LookupError(f"no stand-in for {home if home not in known else away}")
    homes = proxies if home not in known else [home]
    aways = proxies if away not in known else [away]
    preds = [eng.predict_match(h, a, competition=competition, neutral=False)
             for h in homes for a in aways if h != a]
    if not preds:
        raise LookupError("no stand-in pairing")

    def mean(attr):
        return sum(float(getattr(p, attr)) for p in preds) / len(preds)

    lh, la = mean("lambda_home"), mean("lambda_away")
    probs, dist = eng.markets_from_lambdas(lh, la)
    ns = SimpleNamespace(
        home_team=home, away_team=away, competition=competition,
        lambda_home=lh, lambda_away=la,
        p_home_win=probs["p_home_win"], p_draw=probs["p_draw"], p_away_win=probs["p_away_win"],
        p_btts=probs["p_btts"], p_over_15=probs["p_over_15"], p_over_25=probs["p_over_25"],
        p_over_35=probs["p_over_35"], p_under_25=probs["p_under_25"],
        top_scorelines=dist.top_scorelines(8), score_matrix=dist.matrix,
        model_source="debutant_proxy",
    )
    for k in STAT_KEYS:
        for side in ("home", "away"):
            setattr(ns, f"expected_{k}_{side}", mean(f"expected_{k}_{side}"))
    unknown = [t for t in (home, away) if t not in known]
    return ns, {t: proxies for t in unknown}
