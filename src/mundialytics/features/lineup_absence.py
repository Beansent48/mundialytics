"""Which of a team's usual starters are missing today, and whether we could have known.

Regulars = the 11 players with the most starts in the team's last 10 league matches of
the current season, with at least 5 played. The ESPN history the passes read holds the
current season only, and the backtests use the same rule, so the deployed coefficients
match what is served.

Two signals survived the backtest (scripts/lineup_pass/eval_signals.py, weekly-refit
walk-forward with the deployed squad-value shift, LOSO over six seasons):

  * lineup pass, ~1h before kick-off: the market-value share of the regulars who are not
    even in the matchday squad (starters + bench) -> 1X2 RPS -0.00118, 6/6 seasons.
    A regular on the BENCH carries no signal (0/6): that is rotation, and a team rests
    players when it can afford to. The first deployed version counted "not starting",
    which mixes the two and was worth half as much (-0.00052).
  * morning pass, known days ahead: regulars missing from the previous league match's
    squad without serving a ban there, and not banned now -- most likely still injured
    -> -0.00026, 6/6. Bans alone carry nothing (3/6), and neither does "out of the last
    two squads" (4/6).
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

WINDOW = 10
MIN_MATCHES = 5
XI = 11


def regular_xi(history: pd.DataFrame, before) -> list | None:
    """The team's usual starters before `before`, or None when there is too little history.

    `history`: one team's rows for one season with columns date, player, starter (bool).
    Ties in starts go to the player who started more recently.
    """
    h = history[pd.to_datetime(history["date"]) < pd.Timestamp(before)]
    dates = sorted(pd.to_datetime(h["date"]).unique())
    if len(dates) < MIN_MATCHES:
        return None
    recent = h[pd.to_datetime(h["date"]).isin(dates[-WINDOW:])]
    st = recent[recent["starter"].astype(bool)]
    if st.empty:
        return None
    agg = (st.assign(date=pd.to_datetime(st["date"]))
             .groupby("player").agg(starts=("date", "size"), last=("date", "max"))
             .sort_values(["starts", "last"], ascending=False))
    return list(agg.index[:XI])


def absent_share(regulars: list, starters) -> float:
    """Share of `regulars` not in `starters` (0.0 = full-strength XI)."""
    if not regulars:
        return 0.0
    s = set(starters)
    return sum(p not in s for p in regulars) / len(regulars)


def unavailable_share(regulars: list, squad, values: dict | None = None) -> float:
    """Share of `regulars` not in today's matchday `squad` (starters + bench).

    With `values` ({player: EUR}) the share is by market value; a regular without a value
    counts at the median of the ones that have one (all equal when none has).
    """
    if not regulars:
        return 0.0
    s = set(squad)
    known = [values[p] for p in regulars if values and values.get(p)]
    fill = float(pd.Series(known).median()) if known else 1.0
    w = [float(values.get(p) or fill) if values else 1.0 for p in regulars]
    return sum(wi for p, wi in zip(regulars, w) if p not in s) / sum(w)


# ── bans from the card record (league matches only) ────────────────────────────

def _yellow_ban(competition: str, n_yellow: int, match_no: int, recent: int) -> int:
    """Matches banned on receiving league yellow number `n_yellow`."""
    if competition == "Premier League":
        if n_yellow == 5 and match_no <= 19:
            return 1
        if n_yellow == 10 and match_no <= 32:
            return 2
        return 3 if n_yellow == 15 else 0
    if competition == "LaLiga":
        return 1 if n_yellow % 5 == 0 else 0
    if competition == "Bundesliga":
        return 1 if n_yellow in (5, 10, 15) else 0
    if competition == "Serie A":
        return 1 if n_yellow in (5, 10, 14, 17) or n_yellow >= 19 else 0
    if competition == "Ligue 1":
        return 1 if recent >= 3 else 0          # 3 yellows within 10 matches
    return 0


def league_bans(history: pd.DataFrame, competition: str) -> list[set]:
    """Players banned for each of the team's league matches, plus one entry for the NEXT.

    `history`: one team's season with columns date, player, yellow_cards, red_cards.
    A red card bans the next match; the yellow before a second yellow does not count.
    Returns len(match dates) + 1 sets, in date order.
    """
    owed: dict = {}
    yel: dict = {}
    fr: dict = {}
    out = []
    h = history.assign(date=pd.to_datetime(history["date"]))
    for i, (_, day) in enumerate(sorted(h.groupby("date"), key=lambda x: x[0]), start=1):
        today = {p for p, k in owed.items() if k > 0}
        out.append(today)
        for p in today:
            owed[p] -= 1
        for p, y, r in zip(day["player"], day["yellow_cards"].fillna(0), day["red_cards"].fillna(0)):
            if r > 0:
                owed[p] = owed.get(p, 0) + 1
            elif y > 0:
                yel[p] = yel.get(p, 0) + 1
                fr[p] = [m for m in fr.get(p, []) if m > i - 10] + [i]
                n = _yellow_ban(competition, yel[p], i, len(fr[p]))
                if n:
                    owed[p] = owed.get(p, 0) + n
                    fr[p] = []
    out.append({p for p, k in owed.items() if k > 0})
    return out


def still_out(history: pd.DataFrame, before, competition: str) -> tuple[float, list] | None:
    """(share of the regular XI, names) likely still unavailable for the match on `before`.

    Regulars missing from the previous league match's squad who were not serving a ban
    there and are not banned now. `history` must end at the team's previous league match:
    call it only for the team's NEXT league fixture. None without enough history.
    """
    h = history[pd.to_datetime(history["date"]) < pd.Timestamp(before)]
    regs = regular_xi(h, before)
    if regs is None:
        return None
    last = pd.to_datetime(h["date"]).max()
    prev_squad = set(h.loc[pd.to_datetime(h["date"]) == last, "player"])
    bans = league_bans(h, competition)
    out = [p for p in regs if p not in prev_squad and p not in bans[-2] and p not in bans[-1]]
    return len(out) / len(regs), out


# ── ESPN names -> Transfermarkt values ─────────────────────────────────────────

def _norm(s) -> str:
    s = str(s).replace("ß", "ss").replace("ı", "i").replace("ł", "l").replace("ø", "o")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("-", " ").replace("'", "")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", s)).strip()


def match_values(names, squad: pd.DataFrame) -> dict:
    """{name: EUR} for the ESPN `names` found in one club's Transfermarkt `squad`
    (columns player, value_eur). Exact name, same tokens, surname + first initial, a
    mononym, then a close spelling; ambiguous or unmatched names are left out."""
    cand = [(_norm(p), float(v)) for p, v in zip(squad["player"], squad["value_eur"]) if pd.notna(v) and v > 0]
    out = {}
    for name in names:
        k = _norm(name)
        et = k.split()
        if not et:
            continue
        hits = [v for ck, v in cand if ck == k]
        if not hits:
            hits = [v for ck, v in cand if set(ck.split()) == set(et)]
        if not hits and len(et) > 1 and len(et[-1]) >= 3:
            hits = [v for ck, v in cand if ck.split() and et[-1] in ck.split() and ck[0] == et[0][0]]
        if not hits and len(et) == 1:
            hits = [v for ck, v in cand if et[0] in ck.split()]
        if not hits:
            close = [(SequenceMatcher(None, k, ck).ratio(), v) for ck, v in cand]
            close = [c for c in close if c[0] >= 0.85]
            hits = [max(close)[1]] if close else []
        if len(hits) == 1:
            out[name] = hits[0]
    return out
