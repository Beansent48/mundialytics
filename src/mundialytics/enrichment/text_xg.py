"""xG from match commentary text: our own shot model, fed by ESPN.

Every live xG source we used is gone (Understat froze in 05/2026, FBref serves pages
without xG, Sofascore answers 403 since 2026-09-26). ESPN's commentary describes every
attempt in the Opta standard, e.g.

    Attempt saved. Evanilson (Bournemouth) left footed shot from the left side of the six
    yard box is saved in the bottom left corner by Alisson Becker (Liverpool). Assisted by
    Marcus Tavernier with a cross.

Zone, body part, situation and the kind of assist are what a basic xG model uses, so a
logistic regression on those, fitted on shots whose outcome we know, gives a per-shot xG
(penalties at a fixed rate). The count of parsed attempts matches ESPN's own shot totals.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PENALTY_XG = 0.76

_OUTCOME = [
    (re.compile(r"^Goal!"), "goal"),
    (re.compile(r"^Attempt saved\."), "saved"),
    (re.compile(r"^Attempt missed\."), "missed"),
    (re.compile(r"^Attempt blocked\."), "blocked"),
    (re.compile(r"^Penalty saved!"), "saved"),
    (re.compile(r"^Penalty missed!"), "missed"),
]
_WOODWORK = re.compile(r"\((?P<team>[^()]+)\) (?:right footed shot|left footed shot|header|shot).* hits the "
                       r"(?:left |right )?(?:post|bar|crossbar)")
_ZONE = re.compile(r" from (very close range|the (?:centre|left side|right side) of the six yard box"
                   r"|the (?:centre|left side|right side) of the box|outside the box"
                   r"|a difficult angle(?: on the (?:left|right))?|long range(?: on the (?:left|right))?"
                   r"|more than \d+ yards)")
_BODY = re.compile(r"(right footed shot|left footed shot|header|shot)")


def parse_shot(text: str) -> dict | None:
    """One commentary line -> shot dict (team, outcome, zone, body, situation, assist,
    penalty), or None when the line is not an attempt. Own goals are not attempts."""
    if not text:
        return None
    t = text.strip()
    outcome = next((o for rx, o in _OUTCOME if rx.match(t)), None)
    if outcome is None:
        if _WOODWORK.search(t):
            outcome = "woodwork"
        else:
            return None
    body_part = t
    if outcome == "goal":
        m = re.match(r"^Goal! .*?\d+, .*?\d+\. ", t)
        body_part = t[m.end():] if m else t
        if body_part.startswith("Own Goal"):
            return None
    m = re.search(r"\(([^()]+)\)", body_part)
    if not m:
        return None
    team = m.group(1)
    rest = body_part[m.end():]
    penalty = ("Penalty" in t[:16]) or ("converts the penalty" in rest) or ("penalty" in rest[:40])
    z = _ZONE.search(rest)
    zone = z.group(1) if z else ("penalty spot" if penalty else "unknown")
    zone = re.sub(r" on the (left|right)$", "", zone)
    zone = re.sub(r"more than \d+ yards", "more than 35 yards", zone)
    b = _BODY.search(rest)
    body = {"right footed shot": "foot", "left footed shot": "foot", "header": "header"}.get(
        b.group(1) if b else "", "other")
    if "direct free kick" in t:
        situation = "free_kick"
    elif penalty:
        situation = "penalty"
    elif "following a corner" in t:
        situation = "corner"
    elif "set piece situation" in t:
        situation = "set_piece"
    elif "fast break" in t:
        situation = "fast_break"
    else:
        situation = "open_play"
    if "with a cross" in t:
        assist = "cross"
    elif "with a through ball" in t:
        assist = "through_ball"
    elif "with a headed pass" in t:
        assist = "headed_pass"
    elif "Assisted by" in t:
        assist = "pass"
    else:
        assist = "none"
    return {"team": team, "outcome": outcome, "goal": outcome == "goal", "zone": zone, "body": body,
            "situation": situation, "assist": assist, "penalty": bool(penalty)}


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower().replace("&", "and"))


def side_of(team: str, home: str, away: str) -> str:
    """'home' / 'away' for a commentary team name ("Bournemouth" for "AFC Bournemouth",
    "Brighton and Hove Albion" for "Brighton & Hove Albion"), '?' when it is neither."""
    from difflib import SequenceMatcher

    k, h, a = _key(team), _key(home), _key(away)
    if k == h or (k and (k in h or h in k)):
        return "home"
    if k == a or (k and (k in a or a in k)):
        return "away"
    rh, ra = SequenceMatcher(None, k, h).ratio(), SequenceMatcher(None, k, a).ratio()
    if max(rh, ra) >= 0.5 and abs(rh - ra) >= 0.1:
        return "home" if rh > ra else "away"
    return "?"


def shots_from_commentary(lines, home: str | None = None, away: str | None = None) -> pd.DataFrame:
    """All attempts in one match's commentary lines (strings); with the two team names,
    each attempt also gets its side."""
    rows = [s for s in (parse_shot(x) for x in lines) if s is not None]
    out = pd.DataFrame(rows, columns=["team", "outcome", "goal", "zone", "body", "situation", "assist", "penalty"])
    if home is not None and away is not None:
        sides = {t: side_of(t, home, away) for t in out["team"].unique()}
        out["side"] = out["team"].map(sides)
    return out


def _design(shots: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame({
        "zone_body": shots["zone"] + "|" + shots["body"],
        "zone": shots["zone"], "body": shots["body"],
        "situation": shots["situation"], "assist": shots["assist"],
        "header_cross": ((shots["body"] == "header") & (shots["assist"] == "cross")).map(str),
    })
    return pd.get_dummies(x, dtype=float)


class TextXG:
    """Logistic xG on the parsed shot description; penalties at PENALTY_XG."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, shots: pd.DataFrame) -> "TextXG":
        from sklearn.linear_model import LogisticRegression

        s = shots[~shots["penalty"]]
        X = _design(s)
        self.columns_ = list(X.columns)
        self.model_ = LogisticRegression(C=self.C, max_iter=2000).fit(X.values, s["goal"].astype(int).values)
        return self

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        X = _design(shots).reindex(columns=self.columns_, fill_value=0.0)
        p = self.model_.predict_proba(X.values)[:, 1]
        return np.where(shots["penalty"].values, PENALTY_XG, p)


# ── seasons of commentary (data/external/advanced/espn/commentary/<season>.jsonl) ──

def read_commentary(path, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(shots, games) of one season file written by scripts/espn_xg/fetch_commentary.py."""
    import json
    from pathlib import Path

    shots, games = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        g = json.loads(line)
        games.append({"event_id": g["event_id"], "season": season, "date": g["date"],
                      "competition": g["competition"], "home_raw": g["home"], "away_raw": g["away"],
                      "home_goals": g["home_goals"], "away_goals": g["away_goals"]})
        s = shots_from_commentary([c["text"] for c in g["commentary"]], g["home"], g["away"])
        if len(s):
            shots.append(s.assign(event_id=g["event_id"], season=season))
    cols = ["team", "outcome", "goal", "zone", "body", "situation", "assist", "penalty", "side", "event_id", "season"]
    return (pd.concat(shots, ignore_index=True) if shots else pd.DataFrame(columns=cols)), pd.DataFrame(games)


def cross_fit(shots: pd.DataFrame, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold xG: every match is scored by a model that did not see it."""
    ev = shots["event_id"].unique()
    fold = dict(zip(ev, np.random.default_rng(seed).integers(0, k, len(ev))))
    f = shots["event_id"].map(fold).values
    xg = np.zeros(len(shots))
    for i in range(k):
        xg[f == i] = TextXG().fit(shots[f != i]).predict(shots[f == i])
    return xg


OPEN_PLAY = {"open_play", "fast_break"}
SET_PIECE = {"corner", "set_piece", "free_kick"}


def per_match(shots: pd.DataFrame) -> pd.DataFrame:
    """Per match and side: xg, npxg, open-play and set-piece xg, attempts, goals
    (the Understat/Sofascore convention: penalties only in xg)."""
    s = shots[shots["side"].isin(["home", "away"])].assign(
        npxg=lambda x: np.where(x["penalty"], 0.0, x["xg"]),
        xg_op=lambda x: np.where(x["situation"].isin(OPEN_PLAY), x["xg"], 0.0),
        xg_sp=lambda x: np.where(x["situation"].isin(SET_PIECE), x["xg"], 0.0),
        n=1, g=lambda x: x["goal"].astype(int))
    t = s.pivot_table(index="event_id", columns="side", values=["xg", "npxg", "xg_op", "xg_sp", "n", "g"],
                      aggfunc="sum", fill_value=0)
    t.columns = [f"{side}_{v}" for v, side in t.columns]
    for side in ("home", "away"):
        for v in ("xg", "npxg", "xg_op", "xg_sp", "n", "g"):
            if f"{side}_{v}" not in t:
                t[f"{side}_{v}"] = 0
    return t


def join_to_matches(games: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """ESPN event_id -> the row of `matches` (date, home_team, away_team, home_goals,
    away_goals, + whatever else) played within a day with the same score and the most
    similar pair of names. Returns games with the matched columns suffixed _m."""
    from difflib import SequenceMatcher

    def sim(a, b):
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    m = matches.assign(date=pd.to_datetime(matches["date"], errors="coerce"))
    rows = []
    for r in games.assign(date=pd.to_datetime(games["date"])).itertuples(index=False):
        c = m[((m["date"] - r.date).abs() <= pd.Timedelta(days=1))
              & (m["home_goals"] == r.home_goals) & (m["away_goals"] == r.away_goals)]
        if c.empty:
            continue
        best = max((sim(r.home_raw, h) + sim(r.away_raw, a), i) for h, a, i in zip(c["home_team"], c["away_team"], c.index))
        if best[0] >= 0.8:
            rows.append({"event_id": r.event_id, **{f"{k}_m": v for k, v in c.loc[best[1]].items()}})
    return pd.DataFrame(rows)


def level(txg: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """Reference xG over text xG, per competition, on matches both cover (`txg` and `ref`
    indexed by event_id with home_xg/away_xg and competition). A competition the
    reference does not cover takes the overall ratio."""
    j = txg.join(ref[["home_xg", "away_xg"]], rsuffix="_ref").dropna(subset=["home_xg_ref"])
    per = (j.groupby("competition")[["home_xg_ref", "away_xg_ref"]].sum().sum(1)
           / j.groupby("competition")[["home_xg", "away_xg"]].sum().sum(1))
    overall = j[["home_xg_ref", "away_xg_ref"]].values.sum() / j[["home_xg", "away_xg"]].values.sum()
    return per.reindex(sorted(txg["competition"].unique())).fillna(overall)
