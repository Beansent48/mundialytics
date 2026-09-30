#!/usr/bin/env python3
"""Measure how a football match actually unfolds, for SquadLab's live engine.

SquadLab drew a scoreline and then decorated it: goals at invented minutes, no
substitutions, penalties with no taker, a coin toss for a shoot-out. Every one
of those things is written down, minute by minute, in ESPN's commentary of the
big five (2024/25 onwards, ~3,750 matches). This reads it once and writes the
numbers the simulator plays with, so nothing in the live engine is a guess:

  goal clock        how the scoring rate moves through 90 minutes + stoppage
  stoppage          how many added minutes each half gets
  game state        how the scoring rate moves when a side leads / trails
  red cards         how often, when, and what a man down does to both sides
  penalties         how often per expected goal, how often scored/saved/missed
  own goals         share of goals
  assists           share of goals with a named assister
  yellows           when they come, and the second-yellow hazard
  substitutions     real per-team patterns (how many, at which minutes,
                    whether forced by injury), resampled whole by the engine
  who comes off     P(taken off | position) for starters, from 2026/27 rows

Rates are measured against an EXPECTATION built from each team's own season
scoring at that venue (home/away), spread over the match by the goal clock, so
a factor of 0.7 means "70% of what this team would have scored in those
minutes", not a raw count that mixes strong and weak sides.

Writes data/processed/squadlab_match_dynamics.json.

Run:
    python scripts/measure_squadlab_match_dynamics.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COMM = ROOT / "data/external/advanced/espn/commentary"
OUT = ROOT / "data/processed/squadlab_match_dynamics.json"
PLAYERS = ROOT / "data/external/advanced/espn/espn_player_match_current.csv"

_T = re.compile(r"^\s*(\d+)'(?:\+(\d+)')?")
_GOAL = re.compile(r"^Goal! .*?\d+, .*?\d+\. (?P<who>.+?) \((?P<team>[^()]+)\)")
_OWN = re.compile(r"^Own Goal by (?P<who>.+?), (?P<team>[^.]+?)\.")
_YEL = re.compile(r"^(?P<who>.+?) \((?P<team>[^()]+)\) is shown the yellow card")
_RED = re.compile(r"^(?P<who>.+?) \((?P<team>[^()]+)\) is shown the red card")
_SECOND = re.compile(r"^Second yellow card to (?P<who>.+?) \((?P<team>[^()]+)\)")
_SUB = re.compile(r"^Substitution, (?P<team>.+?)\. (?P<inn>.+?) replaces (?P<out>.+?)(?P<inj> because of an injury)?\.\s*$")
_PEN_CONC = re.compile(r"^Penalty conceded by (?P<who>.+?) \((?P<team>[^()]+)\)")
_PEN_SAVED = re.compile(r"^Penalty saved[!.] .*?\((?P<team>[^()]+)\)")
_PEN_MISSED = re.compile(r"^Penalty missed[!.] .*?\((?P<team>[^()]+)\)")
_ADDED = re.compile(r"announced (\d+) minutes? of added time")


def tmin(t: str):
    """"62'" -> (62, 0); "45'+3'" -> (45, 3); "" -> None."""
    m = _T.match(str(t or ""))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def slot(base: int, add: int) -> str:
    """A minute on the simulator's clock: '1'..'90', or '45+k' / '90+k'."""
    if base >= 90 and add:
        return f"90+{add}"
    if base == 45 and add:
        return f"45+{add}"
    return str(min(base, 90))


def order_key(base: int, add: int) -> float:
    """Chronological order of a displayed minute (first-half stoppage before 46')."""
    if base <= 45:
        return base + add / 100.0
    return 100 + base + add / 100.0


def side_of(team: str, home: str, away: str) -> str | None:
    t = team.strip().lower()
    h, a = home.lower(), away.lower()
    if t == h or t in h or h in t:
        return "h"
    if t == a or t in a or a in t:
        return "a"
    return None


def parse(g: dict) -> dict:
    """One match -> its dated events, in order."""
    home, away = g["home"], g["away"]
    ev, added = [], {1: 0, 2: 0}
    for e in g.get("commentary", []):
        text = str(e.get("text", ""))
        tm = tmin(e.get("t", ""))
        m = _ADDED.search(text)
        if m and tm:
            added[1 if tm[0] <= 45 else 2] = int(m.group(1))
            continue
        if tm is None:
            continue
        base, add = tm
        kind, side, extra = None, None, {}
        if (m := _GOAL.match(text)):
            kind, side = "goal", side_of(m["team"], home, away)
            extra = {"assisted": "Assisted by" in text,
                     "pen": "penalty" in text[:200].lower()}
        elif (m := _OWN.match(text)):
            # the own-goal scorer's team CONCEDES it
            s = side_of(m["team"], home, away)
            kind, side, extra = "goal", {"h": "a", "a": "h"}.get(s), {"own": True}
        elif (m := _SECOND.match(text)):
            kind, side = "red2", side_of(m["team"], home, away)
        elif (m := _RED.match(text)):
            kind, side = "red", side_of(m["team"], home, away)
        elif (m := _YEL.match(text)):
            kind, side = "yellow", side_of(m["team"], home, away)
        elif (m := _SUB.match(text)):
            kind, side, extra = "sub", side_of(m["team"], home, away), {"inj": bool(m["inj"]), "out": m["out"]}
        elif (m := _PEN_CONC.match(text)):
            s = side_of(m["team"], home, away)
            kind, side = "pen_awarded", {"h": "a", "a": "h"}.get(s)
        elif (m := _PEN_SAVED.match(text)):
            kind, side = "pen_saved", side_of(m["team"], home, away)
        elif (m := _PEN_MISSED.match(text)):
            kind, side = "pen_missed", side_of(m["team"], home, away)
        if kind and side:
            ev.append({"k": kind, "s": side, "base": base, "add": add,
                       "o": order_key(base, add), **extra})
    ev.sort(key=lambda x: x["o"])
    return {"id": str(g["event_id"]), "season": None, "home": home, "away": away,
            "hg": g.get("home_goals"), "ag": g.get("away_goals"), "ev": ev, "added": added}


def load() -> list[dict]:
    out = []
    for f in sorted(COMM.glob("*.jsonl")):
        season = f.stem
        for line in f.read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            if not g.get("commentary"):
                continue
            p = parse(g)
            p["season"] = season
            # keep only matches whose commentary goals reproduce the final score
            gh = sum(1 for e in p["ev"] if e["k"] == "goal" and e["s"] == "h")
            ga = sum(1 for e in p["ev"] if e["k"] == "goal" and e["s"] == "a")
            p["ok"] = (gh == p["hg"] and ga == p["ag"])
            out.append(p)
    return out


# ── the goal clock ─────────────────────────────────────────────────────────────
def minute_axis(added1: int, added2: int) -> list[tuple[str, float]]:
    """The simulator's minutes for one match, in order, with a position 0..1."""
    return ([(str(i), i) for i in range(1, 46)] + [(f"45+{k}", 45 + k / 10) for k in range(1, added1 + 1)]
            + [(str(i), i) for i in range(46, 91)] + [(f"90+{k}", 90 + k / 10) for k in range(1, added2 + 1)])


def goal_clock(ms: list[dict]) -> dict:
    """Goals per played minute, per regulation 5-minute bin and per stoppage minute.

    Regulation minutes are played in every match; a stoppage minute k only in
    matches that announced at least k. Dividing by exposure keeps a long
    stoppage from looking like a scoring surge.
    """
    reg = Counter()
    st = {1: Counter(), 2: Counter()}
    exp_st = {1: Counter(), 2: Counter()}
    n = 0
    for m in ms:
        n += 1
        for h in (1, 2):
            for k in range(1, m["added"][h] + 1):
                exp_st[h][k] += 1
        for e in m["ev"]:
            if e["k"] != "goal":
                continue
            if e["add"] and e["base"] in (45, 90):
                h = 1 if e["base"] == 45 else 2
                st[h][min(e["add"], 12)] += 1
            else:
                reg[(min(e["base"], 90) - 1) // 5] += 1
    per_min_reg = {int(b): reg[b] / (5 * n) for b in range(18)}
    per_min_st = {}
    for h in (1, 2):
        tot_g = sum(st[h].values())
        tot_exp = sum(exp_st[h].values())
        per_min_st[h] = tot_g / max(tot_exp, 1)
    mean_rate = sum(per_min_reg.values()) * 5 / 90
    return {
        "per_minute_by_5min_bin": [round(per_min_reg[b] / mean_rate, 4) for b in range(18)],
        "stoppage_first_half": round(per_min_st[1] / mean_rate, 4),
        "stoppage_second_half": round(per_min_st[2] / mean_rate, 4),
        "note": "relative scoring intensity per minute; 1.0 = the match-average minute",
        "n_matches": n,
    }


def stoppage(ms: list[dict]) -> dict:
    out = {}
    for h in (1, 2):
        c = Counter(min(m["added"][h], 15) for m in ms if m["added"][h] > 0)
        tot = sum(c.values())
        out[f"half{h}"] = {str(k): round(v / tot, 4) for k, v in sorted(c.items())}
    return out


def intensity_at(clock: dict, base: int, add: int) -> float:
    if add and base == 45:
        return clock["stoppage_first_half"]
    if add and base >= 90:
        return clock["stoppage_second_half"]
    return clock["per_minute_by_5min_bin"][(min(base, 90) - 1) // 5]


# ── expectation per team-match ─────────────────────────────────────────────────
def team_rates(ms: list[dict]) -> dict:
    """Goals per match by (season, team, venue), each match's own goals left out."""
    tot = defaultdict(lambda: [0.0, 0])
    for m in ms:
        tot[(m["season"], m["home"], "h")][0] += m["hg"]
        tot[(m["season"], m["home"], "h")][1] += 1
        tot[(m["season"], m["away"], "a")][0] += m["ag"]
        tot[(m["season"], m["away"], "a")][1] += 1
    lg = np.mean([m["hg"] + m["ag"] for m in ms]) / 2
    return tot, lg


def expected_rate(tot, lg, m, side, own_goals_this_match) -> float:
    team = m["home"] if side == "h" else m["away"]
    g, n = tot[(m["season"], team, side)]
    g -= own_goals_this_match
    n -= 1
    # shrink toward the league mean with 5 matches of prior
    return (g + 5 * lg) / (n + 5)


def state_effects(ms: list[dict], clock: dict) -> dict:
    """Scoring relative to expectation, by the scoreline the minute is played at,
    and after a red card (for the side down a man and for the side up one).

    Walks each match minute by minute, holding the state that applies at each
    minute; a match is left once a second red card makes it 9v10 / 10v10 etc.
    """
    tot, lg = team_rates(ms)
    obs, expc = Counter(), Counter()
    for m in ms:
        lam = {s: expected_rate(tot, lg, m, s, m["hg"] if s == "h" else m["ag"]) for s in ("h", "a")}
        # per-minute expected goals for this match, scaled by the clock
        axis = minute_axis(m["added"][1], m["added"][2])
        weights = []
        for lab, _ in axis:
            b, a = (int(lab.split("+")[0]), int(lab.split("+")[1])) if "+" in lab else (int(lab), 0)
            weights.append(intensity_at(clock, b, a))
        # the per-match team rate covers the WHOLE match, stoppage included, so
        # it is spread over every minute this match actually played
        wsum = sum(weights) or 1.0
        scale = 90.0 / wsum
        events = m["ev"]
        score = {"h": 0, "a": 0}
        men = {"h": 11, "a": 11}
        ei = 0
        for (lab, pos), w in zip(axis, weights):
            b, a = (int(lab.split("+")[0]), int(lab.split("+")[1])) if "+" in lab else (int(lab), 0)
            o = order_key(b, a)
            # events stamped this minute happen within it: count goals at the
            # state in force when the minute began, then apply the events
            goals_now = {"h": 0, "a": 0}
            j = ei
            while j < len(events) and events[j]["o"] <= o + 1e-9:
                e = events[j]
                if e["k"] == "goal":
                    goals_now[e["s"]] += 1
                j += 1
            if min(men.values()) >= 10 and not (men["h"] == 10 and men["a"] == 10):
                for s in ("h", "a"):
                    other = "a" if s == "h" else "h"
                    gd = score[s] - score[other]
                    st = "level" if gd == 0 else ("lead1" if gd == 1 else "lead2" if gd >= 2
                                                  else "trail1" if gd == -1 else "trail2")
                    mu = lam[s] * w * scale / 90.0
                    if men[s] == 11 and men[other] == 11:
                        key = ("state", st)
                    elif men[s] == 10:
                        key = ("red", "down")
                    else:
                        key = ("red", "up")
                    obs[key] += goals_now[s]
                    expc[key] += mu
            for k in range(ei, j):
                e = events[k]
                if e["k"] == "goal":
                    score[e["s"]] += 1
                elif e["k"] in ("red", "red2"):
                    men[e["s"]] -= 1
            ei = j
    out = {}
    for key in sorted(expc):
        out[f"{key[0]}:{key[1]}"] = {"factor": round(obs[key] / expc[key], 4),
                                     "goals": int(obs[key]), "expected": round(expc[key], 1)}
    return out


def red_cards(ms: list[dict]) -> dict:
    n_tm = 2 * len(ms)
    direct = sum(1 for m in ms for e in m["ev"] if e["k"] == "red")
    second = sum(1 for m in ms for e in m["ev"] if e["k"] == "red2")
    mins = [e["base"] for m in ms for e in m["ev"] if e["k"] == "red"]
    return {
        "direct_per_team_match": round(direct / n_tm, 5),
        "direct_by_15min": np.histogram(mins, bins=[0, 15, 30, 45, 60, 75, 91])[0].tolist(),
        "second_yellow_per_team_match": round(second / n_tm, 5),
    }


def yellows(ms: list[dict]) -> dict:
    """When yellows come, and the second-yellow hazard for a booked player."""
    mins = [min(e["base"], 90) for m in ms for e in m["ev"] if e["k"] == "yellow"]
    hist = np.histogram(mins, bins=list(range(0, 91, 5)) + [91])[0]
    # hazard: second yellows / booked player-minutes remaining
    booked_minutes, seconds = 0.0, 0
    for m in ms:
        for e in m["ev"]:
            if e["k"] == "yellow":
                booked_minutes += max(92 - min(e["base"], 90), 1)
            elif e["k"] == "red2":
                seconds += 1
    return {
        "by_5min_share": [round(x / hist.sum(), 4) for x in hist.tolist()],
        "second_yellow_hazard_per_min": round(seconds / booked_minutes, 6),
    }


def penalties(ms: list[dict]) -> dict:
    awarded = sum(1 for m in ms for e in m["ev"] if e["k"] == "pen_awarded")
    saved = sum(1 for m in ms for e in m["ev"] if e["k"] == "pen_saved")
    missed = sum(1 for m in ms for e in m["ev"] if e["k"] == "pen_missed")
    scored = sum(1 for m in ms for e in m["ev"] if e["k"] == "goal" and e.get("pen"))
    goals = sum(m["hg"] + m["ag"] for m in ms)
    taken = scored + saved + missed
    return {
        "awarded_per_team_match": round(awarded / (2 * len(ms)), 5),
        "taken": taken, "scored": scored, "saved": saved, "missed": missed,
        "conversion": round(scored / max(taken, 1), 4),
        "saved_share_of_failures": round(saved / max(saved + missed, 1), 4),
        "penalty_goals_share_of_goals": round(scored / max(goals, 1), 4),
    }


def goal_details(ms: list[dict]) -> dict:
    goals = [e for m in ms for e in m["ev"] if e["k"] == "goal"]
    own = sum(1 for e in goals if e.get("own"))
    open_play = [e for e in goals if not e.get("own") and not e.get("pen")]
    return {
        "own_goal_share": round(own / len(goals), 4),
        "assisted_share_non_penalty": round(sum(e["assisted"] for e in open_play) / len(open_play), 4),
    }


def substitutions(ms: list[dict]) -> dict:
    """Every team-match's real substitution pattern, to be resampled whole.

    Keyed by the side's state at the hour (leading / level / trailing) because
    who is chasing a game changes WHEN the manager moves; the engine picks a
    pattern from the matching bucket.
    """
    buckets = {"lead": [], "level": [], "trail": []}
    n_inj = n_subs = 0
    for m in ms:
        score = {"h": 0, "a": 0}
        at60 = None
        subs = {"h": [], "a": []}
        for e in m["ev"]:
            if at60 is None and e["o"] > 160:
                at60 = dict(score)
            if e["k"] == "goal":
                score[e["s"]] += 1
            elif e["k"] == "sub":
                lab = slot(e["base"], e["add"])
                subs[e["s"]].append([lab, int(e["inj"])])
                n_subs += 1
                n_inj += int(e["inj"])
        at60 = at60 or score
        for s in ("h", "a"):
            o = "a" if s == "h" else "h"
            gd = at60[s] - at60[o]
            b = "level" if gd == 0 else ("lead" if gd > 0 else "trail")
            # a side with no change at all is almost always a commentary gap
            # (5 subs are allowed; 4.7% "none" is far above the real rate)
            if subs[s]:
                buckets[b].append(subs[s][:5])
    counts = {b: Counter(len(p) for p in v) for b, v in buckets.items()}
    return {
        "patterns": buckets,
        "count_distribution": {b: {str(k): v for k, v in sorted(c.items())} for b, c in counts.items()},
        "injury_share": round(n_inj / max(n_subs, 1), 4),
    }


def own_goal_weights() -> dict:
    """Own goals per started match, by position group (2026/27 player rows).

    Only the RELATIVE sizes are used: who, of the side's players on the pitch,
    puts the ball into his own net once the engine has decided one goes in.
    """
    if not PLAYERS.exists():
        return {}
    p = pd.read_csv(PLAYERS)
    st = p[p["starter"].astype(str).str.lower().isin(["true", "1"])].copy()
    st["g"] = st["position"].map(_GROUP)
    r = st.dropna(subset=["g"]).groupby("g")["own_goals"].agg(["sum", "size"])
    return {g: round(float(row["sum"] / row["size"]), 5) for g, row in r.iterrows()}


def injury_absence() -> dict:
    """How long a player taken off injured stays out, in DAYS (Kaplan-Meier).

    Only 2026/27 can say: its ESPN player file lists every matchday squad,
    bench included, so "absent" means left out of the eighteen. Counted in days
    rather than matches because a Champions side plays every three or four
    days and the game's calendar is the real one. Players still out when the
    data ends are censored, not dropped -- dropping them would keep only the
    short injuries and make every knock look like a day off.
    """
    import unicodedata

    def key(s):
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()

    f = COMM / "2026-2027.jsonl"
    if not (PLAYERS.exists() and f.exists()):
        return {}
    p = pd.read_csv(PLAYERS)
    p["pk"] = p["player"].map(key)
    p["event_id"] = p["event_id"].astype(str)
    p["date"] = pd.to_datetime(p["date"])
    team_dates = p[["team", "event_id", "date"]].drop_duplicates()
    seen = set(zip(p["event_id"], p["pk"]))
    inj = []
    for line in f.read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        for e in g.get("commentary", []):
            m = _SUB.match(str(e.get("text", "")))
            if m and m["inj"]:
                inj.append((str(g["event_id"]), key(m["out"])))
    obs = []                                   # (days, returned?)
    for ev, pk in inj:
        r = p[(p["event_id"] == ev) & (p["pk"] == pk)]
        if r.empty:
            continue
        team, d0 = r["team"].iloc[0], r["date"].iloc[0]
        later = team_dates[(team_dates["team"] == team) & (team_dates["date"] > d0)].sort_values("date")
        back = next((row.date for row in later.itertuples() if (row.event_id, pk) in seen), None)
        if back is not None:
            obs.append(((back - d0).days, True))
        elif len(later):
            obs.append(((later["date"].max() - d0).days, False))
    if not obs:
        return {}
    # Kaplan-Meier on "still out after d days", evaluated at the days a return
    # happened; the engine reads it as a step function
    obs.sort()
    at_risk, surv, curve = len(obs), 1.0, []
    i = 0
    while i < len(obs):
        d = obs[i][0]
        ret = sum(1 for x in obs if x[0] == d and x[1])
        n_d = sum(1 for x in obs if x[0] == d)
        if ret:
            surv *= 1 - ret / at_risk
            curve.append([int(d), round(surv, 4)])
        at_risk -= n_d
        i += n_d
    return {"n_injuries": len(obs), "n_returned": sum(1 for x in obs if x[1]),
            "max_followup_days": int(max(x[0] for x in obs)),
            "still_out_after_days": curve}


def xg_model() -> dict:
    """Non-penalty xG of a side in one match, given its goals and shots.

    npxG ≈ a·(non-penalty goals) + b·(other shots), fitted on our own ESPN
    text-xG (2026/27); a penalty adds its own fixed 0.76.
    """
    x = ROOT / "data/processed/espn_text_team_match_xg.csv"
    if not x.exists():
        return {}
    d = pd.read_csv(x)
    np_ = np.r_[d["home_npxg"], d["away_npxg"]]
    sh = np.r_[d["home_shots"], d["away_shots"]]
    gl = np.r_[d["home_goals"], d["away_goals"]]
    A = np.c_[gl, np.clip(sh - gl, 0, None)]
    coef = np.linalg.lstsq(A, np_, rcond=None)[0]
    resid = np_ - A @ coef
    return {"per_goal": round(float(coef[0]), 4), "per_other_shot": round(float(coef[1]), 4),
            "resid_sd": round(float(resid.std()), 4), "penalty_xg": 0.76, "n": int(len(np_))}


def main() -> None:
    ms = load()
    ok = [m for m in ms if m["ok"]]
    print(f"matches parsed {len(ms)}, score reproduced by the commentary {len(ok)}")
    clock = goal_clock(ok)
    out = {
        "source": "ESPN commentary, big five, " + ", ".join(sorted({m["season"] for m in ok})),
        "n_matches": len(ok),
        "goal_clock": clock,
        "stoppage": stoppage(ok),
        "state_effects": state_effects(ok, clock),
        "red_cards": red_cards(ok),
        "yellows": yellows(ok),
        "penalties": penalties(ok),
        "goals": goal_details(ok),
        "substitutions": substitutions(ok),
        "who_comes_off": off_rates(ok),
        "injury_absence": injury_absence(),
        "own_goal_weights": own_goal_weights(),
        "xg": xg_model(),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    # the live engine's normalising constant depends on the rules just written,
    # so it is found here, once, instead of on the API's first request
    from mundialytics.statistical_core.squadlab import match_engine
    match_engine.load_dynamics.cache_clear()
    out["norm"] = round(match_engine.compute_norm(), 5)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    show = {k: v for k, v in out.items() if k != "substitutions"}
    show["substitutions"] = {k: v for k, v in out["substitutions"].items() if k != "patterns"}
    print(json.dumps(show, indent=1, ensure_ascii=False))


_GROUP = {"G": "Goalkeeper"}
for _c in ("CD", "CD-L", "CD-R", "LB", "RB", "LWB", "RWB", "D", "SW"):
    _GROUP[_c] = "Defender"
for _c in ("CM", "CM-L", "CM-R", "DM", "LM", "RM", "AM", "AM-L", "AM-R", "M"):
    _GROUP[_c] = "Midfielder"
for _c in ("F", "CF", "CF-L", "CF-R", "LF", "RF", "ST"):
    _GROUP[_c] = "Forward"


def off_rates(ms: list[dict]) -> dict:
    """P(taken off) and mean minute, by position group, for 2026/27 starters."""
    if not PLAYERS.exists():
        return {}
    p = pd.read_csv(PLAYERS)
    p["starter"] = p["starter"].astype(str).str.lower().isin(["true", "1"])
    st = p[p["starter"]].copy()
    st["g"] = st["position"].map(_GROUP)
    off = {}
    for m in ms:
        for e in m["ev"]:
            if e["k"] == "sub":
                off[(m["id"], e["out"].strip())] = e["base"]
    st["off"] = [off.get((str(i), str(n).strip())) for i, n in zip(st["event_id"], st["player"])]
    res = {}
    for g_, d in st.dropna(subset=["g"]).groupby("g"):
        taken = d["off"].notna()
        res[g_] = {"p_off": round(float(taken.mean()), 4),
                   "mean_minute": round(float(d.loc[taken, "off"].mean()), 1) if taken.any() else None,
                   "n": int(len(d))}
    return res


if __name__ == "__main__":
    main()
