"""Cards and real squads, turned into the players the live engine plays with.

The live engine (match_engine.py) needs, for each man on the pitch, what he does
per 90 minutes: goals, assists, bookings, whether he takes the penalties. For a
card that comes from data/processed/squadlab_card_rates.csv, built by
scripts/build_squadlab_card_rates.py from the same sources as the player-props
model. For a club outside the big five (two thirds of a Champions field), no
card exists and only this season's ESPN rows do, so the rate is its record per
appearance shrunk towards its position, as rival_squads.py already reads it.

Nothing here decides how GOOD a side is — Elo does that — only whose name goes
on each event.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from mundialytics.statistical_core.squadlab.match_engine import MatchPlayer

ROOT = Path(__file__).resolve().parents[4]
CARD_RATES = ROOT / "data/processed/squadlab_card_rates.csv"

# Per-90 fallbacks by position, the props model's big-five priors rounded; used
# only when a card has no row in the rates file (an out-of-date build)
FALLBACK = {
    "Goalkeeper": {"g": 0.0, "a": 0.004, "sh": 0.01, "yc": 0.07},
    "Defender": {"g": 0.05, "a": 0.06, "sh": 0.7, "yc": 0.21},
    "Midfielder": {"g": 0.11, "a": 0.12, "sh": 1.4, "yc": 0.22},
    "Forward": {"g": 0.28, "a": 0.14, "sh": 2.3, "yc": 0.19},
}
BENCH_SHAPE = {"Goalkeeper": 1, "Defender": 2, "Midfielder": 2, "Forward": 2}
ELEVEN_SHAPE = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}


@lru_cache(maxsize=2)
def _rates(path: str | None = None) -> dict[str, dict]:
    p = Path(path) if path else CARD_RATES
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {str(r["card_id"]): r for r in df.to_dict("records")}


def from_card(card) -> MatchPlayer:
    r = _rates().get(str(card.card_id))
    fb = FALLBACK.get(card.position, FALLBACK["Midfielder"])

    def v(k):
        x = r.get(k) if r else None
        return float(x) if x is not None and pd.notna(x) else fb.get(k, 0.0)

    return MatchPlayer(key=str(card.card_id), name=str(card.player), position=str(card.position),
                       overall=float(card.overall), g=v("g"), a=v("a"), sh=v("sh"), yc=v("yc"),
                       taker=float(r.get("taker", 0.0) or 0.0) if r else 0.0)


def typical_bench_gap(cards_df: pd.DataFrame) -> dict[str, float]:
    """How much worse a real club's first change is than the man it replaces.

    Per position, the median over clubs of (mean starter overall − mean of the
    best two not in the eleven). The live engine charges a substitution only
    for the part of the drop BEYOND this: real matches, and so the measured
    goal clock, already contain ordinary changes.
    """
    from mundialytics.statistical_core.squadlab.cards import best_eleven

    act = cards_df[cards_df["kind"] == "actual"]
    gaps: dict[str, list[float]] = {}
    for club, grp in act.groupby("club"):
        xi = best_eleven(cards_df, club)
        if len(xi) < 11:
            continue
        ids = {c.card_id for c in xi}
        rest = grp[~grp["card_id"].isin(ids)]
        for pos in ("Goalkeeper", "Defender", "Midfielder", "Forward"):
            st = [c.overall for c in xi if c.position == pos]
            b = rest.loc[rest["position"] == pos, "overall"].nlargest(2 if pos != "Goalkeeper" else 1)
            if st and len(b):
                gaps.setdefault(pos, []).append(float(np.mean(st) - b.mean()))
    return {k: round(float(np.median(v)), 2) for k, v in gaps.items()}


def club_side_from_cards(cards_df: pd.DataFrame, club: str,
                         drafted: set[str]) -> tuple[list[MatchPlayer], list[MatchPlayer]]:
    """A big-five club's best eleven and a seven-man bench, from its actual cards."""
    from mundialytics.statistical_core.squadlab.cards import cards_from_frame
    from mundialytics.identity.current_squads import short_key

    grp = cards_df[(cards_df["kind"] == "actual") & (cards_df["club"] == club)]
    grp = grp[~grp["player"].map(lambda n: short_key(str(n))).isin(drafted)]
    cards = sorted(cards_from_frame(grp), key=lambda c: -c.overall)
    return _split(cards, lambda c: c.position, from_card)


def club_side_from_rows(rows: pd.DataFrame, drafted: set[str]) -> tuple[list[MatchPlayer], list[MatchPlayer]]:
    """A club with no cards: its ESPN squad, eleven by appearances this season."""
    from mundialytics.identity.current_squads import short_key
    from mundialytics.statistical_core.squadlab.rival_squads import (
        FALLBACK_ASSIST_RATE, FALLBACK_GOAL_RATE, SHRINK_APPS,
    )

    d = rows.copy()
    d = d[~d["player"].map(lambda n: short_key(str(n))).isin(drafted)]
    for col in ("apps", "goals", "assists", "yellow_cards", "starts",
                "uefa_apps", "uefa_goals", "uefa_assists"):
        if col not in d.columns:
            d[col] = 0.0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    d["pos"] = d.get("pos_group", pd.Series("Midfielder", index=d.index)).fillna("Midfielder")
    apps = d["apps"] + d["uefa_apps"]
    gp = d["pos"].map(FALLBACK_GOAL_RATE).fillna(0.08)
    ap = d["pos"].map(FALLBACK_ASSIST_RATE).fillna(0.08)
    # per appearance; a starter's appearance is ~80 minutes, near enough per 90
    # for weights that only ever compete inside one side
    d["g"] = (d["goals"] + d["uefa_goals"] + gp * SHRINK_APPS) / (apps + SHRINK_APPS)
    d["a"] = (d["assists"] + d["uefa_assists"] + ap * SHRINK_APPS) / (apps + SHRINK_APPS)
    d["yc"] = (d["yellow_cards"] + 0.2 * SHRINK_APPS) / (d["apps"] + SHRINK_APPS)
    d.loc[d["pos"] == "Goalkeeper", "g"] = 0.0
    d = d.sort_values(["starts", "apps"], ascending=False)

    players = [MatchPlayer(key=f"r:{r.player}", name=str(r.player), position=str(r.pos),
                           overall=70.0, g=float(r.g), a=float(r.a), sh=0.0, yc=float(r.yc))
               for r in d.itertuples()]
    xi, bench = _split(players, lambda p: p.position, lambda p: p)
    # the side's penalty taker: its most prolific outfielder, lacking a record
    if xi:
        best = max((p for p in xi if p.position != "Goalkeeper"), key=lambda p: p.g, default=None)
        if best is not None:
            best.taker = 1.0
    return xi, bench


def _split(items, pos_of, make) -> tuple[list[MatchPlayer], list[MatchPlayer]]:
    """First the eleven by shape, then the bench by shape, then anyone left."""
    xi, bench, used = [], [], set()
    for shape, out in ((ELEVEN_SHAPE, xi), (BENCH_SHAPE, bench)):
        for pos, n in shape.items():
            for it in [x for x in items if pos_of(x) == pos and id(x) not in used][:n]:
                used.add(id(it))
                out.append(make(it))
    # a squad short in one line fills the eleven from whoever is left
    rest = [x for x in items if id(x) not in used and pos_of(x) != "Goalkeeper"]
    while len(xi) < 11 and rest:
        it = rest.pop(0)
        used.add(id(it))
        xi.append(make(it))
    return xi, bench
