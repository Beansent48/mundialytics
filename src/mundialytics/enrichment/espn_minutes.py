"""Minutes played, from ESPN's match commentary.

ESPN's player file says who started and who came on, never for how long, so
the player model estimated minutes from each player's usual role (v0.57.0).
The commentary names every change with its minute --

    62'  Substitution, Coventry City. Victor Torp replaces Caleb Yirenkyi.
    66'  Ezri Konsa (Aston Villa) is shown the red card.
    Second yellow card to Wesley Fofana (Chelsea) for a bad foul.

-- so the minutes can be read instead. Understat's convention is kept: a full
match is 90 and the clock runs on: Understat counts first-half stoppage into
the second half (a change at 62' after 45'+3' is about minute 64 -- read off 53k
player-matches of 2025/26, where the plain commentary minute ran 3' short for
every starter taken off). Second-half stoppage is cut at 90, and a player who
came on in it played 1.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

FULL = 90
_SUB = re.compile(r"^Substitution, [^.]+\. (?P<inn>.+?) replaces (?P<out>.+?)\.\s*$")
_RED = re.compile(r"^(?P<who>.+?) \([^()]+\) is shown the red card")
_SECOND = re.compile(r"^Second yellow card to (?P<who>.+?) \([^()]+\)")


def minute(t: str) -> int | None:
    """"62'" -> 62, "45'+3'" -> 45, "" -> None."""
    m = re.match(r"^\s*(\d+)'", str(t or ""))
    return int(m.group(1)) if m else None


def _added(t: str) -> int:
    """"45'+3'" -> 3, "62'" -> 0."""
    m = re.search(r"\+(\d+)'", str(t or ""))
    return int(m.group(1)) if m else 0


def clock(t: str, first_half_added: int) -> int | None:
    """Understat's continuous minute for a commentary time, capped at FULL."""
    base = minute(t)
    if base is None:
        return None
    if base <= 45:
        return min(base + _added(t), FULL)
    # one minute less than the full first-half stoppage: picked on Aug-Dec 2025,
    # held out on Jan-Jun 2026 (MAE 1.59' vs 1.81' with all of it, 9.9' for the
    # role-mean estimate it replaces; 90% within a minute)
    return min(base + max(first_half_added - 1, 0), FULL)


def minutes_from_commentary(commentary: list[dict], starters: list[str],
                            squad: list[str]) -> dict[str, int]:
    """Minutes per squad name (0 for a player who never came on).

    `commentary`: the match's [{"t": "62'", "text": ...}] lines; `starters` and
    `squad` are names as the lineup lists them. Commentary names are matched to
    the squad by name (accents, short forms), so a line naming someone outside
    the squad is ignored rather than guessed."""
    from mundialytics.identity.current_squads import NameIndex

    idx = NameIndex()
    for name in squad:
        idx.add(name, name)
    on = {name: 0 for name in starters}
    off: dict[str, int] = {}
    fh = max((_added(c.get("t")) for c in commentary if minute(c.get("t")) == 45), default=0)
    for c in commentary:
        t, text = clock(c.get("t"), fh), str(c.get("text", "")).strip()
        if t is None:
            continue
        m = _SUB.match(text)
        if m:
            inn, out = idx.lookup(m.group("inn")), idx.lookup(m.group("out"))
            if inn is not None and inn not in on:
                on[inn] = min(t, FULL)
            if out is not None and out in on and out not in off:
                off[out] = min(t, FULL)
            continue
        m = _RED.match(text) or _SECOND.match(text)
        if m:
            who = idx.lookup(m.group("who"))
            if who is not None and who in on and who not in off:
                off[who] = min(t, FULL)
    out = {name: 0 for name in squad}
    for name, start in on.items():
        end = off.get(name, FULL)
        out[name] = max(end - start, 1)
    return out


def season_minutes(path: str | Path, squads: pd.DataFrame) -> pd.DataFrame:
    """event_id, player, minutes for every match of one commentary season file.

    `squads`: the ESPN player rows (event_id, player, starter) of that season;
    events with no commentary are simply absent from the result."""
    p = Path(path)
    if not p.exists() or squads.empty:
        return pd.DataFrame(columns=["event_id", "player", "minutes"])
    sq = squads.assign(event_id=squads["event_id"].astype(str))
    by_event = {e: g for e, g in sq.groupby("event_id")}
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        g = json.loads(line)
        s = by_event.get(str(g["event_id"]))
        if s is None:
            continue
        started = s["starter"].astype(str).str.lower().isin(["true", "1"])
        mins = minutes_from_commentary(g.get("commentary", []), list(s.loc[started, "player"]),
                                       list(s["player"]))
        rows += [{"event_id": str(g["event_id"]), "player": k, "minutes": v} for k, v in mins.items()]
    return pd.DataFrame(rows, columns=["event_id", "player", "minutes"])
