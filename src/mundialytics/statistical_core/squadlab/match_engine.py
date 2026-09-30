"""SquadLab's live match: ninety-odd minutes played one at a time.

WHAT IT REPLACES. A SquadLab match used to be a Poisson scoreline, decorated
afterwards: goals at random minutes, no substitutions, penalties with no taker,
a coin toss for a shoot-out. The team model's expected goals were right; the
match around them was invented.

WHAT IT DOES. The expected goals still come from the team model (Elo, and the
squad's shape — see champions.py); this module only decides how they unfold,
and every rule it plays by is a number measured on the big five's ESPN
commentary (scripts/measure_squadlab_match_dynamics.py, ~3,100 matches):

  goal clock       the scoring rate through the match, stoppage time included
  game state       a side a goal up scores 7% less, a goal down 2% more
  red cards        a side down a man scores 49% of its rate, the other 148%
  penalties        ~10% of a side's expected goals arrive as an awarded
                   penalty, scored 77% of the time — by the designated taker,
                   at the population rate (taker conversion does not persist
                   across seasons, r = −0.07)
  own goals        2.9% of goals, put in mostly by defenders
  assists          77% of open-play goals carry one
  bookings         the side's expected yellows spread by the booking clock;
                   a booked player's second-yellow hazard per minute
  substitutions    whole real patterns (how many changes, when, whether one was
                   forced by injury), chosen by the score at the hour
  who comes off    by position, from 2026/27 starters (forwards 51%, keepers 1%)

Who gets each event is read off the players' own per-90 rates
(scripts/build_squadlab_card_rates.py): open-play goals by the goal rate,
assists by the creation rate, bookings by the booking rate.

A substitution changes the side's strength when its manager has a real bench
(the drafted squad): the change is measured against the TYPICAL starter-bench
gap of real clubs, because the goal clock was measured on real matches that
already contain real substitutions. A like-for-like change costs nothing
extra; bringing on a bronze for an icon does.

Expected goals are kept: the state and red-card factors move goals around a
match, and a normalising constant (fitted once, by simulation) keeps the
average total on the team model's number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DYNAMICS = ROOT / "data/processed/squadlab_match_dynamics.json"

POSITIONS = ("Goalkeeper", "Defender", "Midfielder", "Forward")
# Elo points per unit of goal-supremacy log-rate: the European calibration's b
ELO_B = 0.7393927801465834
MAX_SUBS = 5


@dataclass
class MatchPlayer:
    """What the engine needs to know about one man."""
    key: str
    name: str
    position: str
    overall: float = 70.0
    g: float = 0.1      # open-play goals per 90 (0.7 npxG + 0.3 npG)
    a: float = 0.1      # assists per 90 (0.7 xA + 0.3 A)
    sh: float = 1.0     # shots per 90
    yc: float = 0.2     # yellow cards per 90
    taker: float = 0.0  # penalty-taker share, 0..1


@dataclass
class Side:
    name: str
    starters: list[MatchPlayer]
    bench: list[MatchPlayer] = field(default_factory=list)
    lam: float = 1.3                 # expected goals for this side over the match
    yellows_lam: float = 2.0
    shots_lam: float = 12.0
    corners_lam: float = 5.0
    # Elo per point of the eleven's mean overall (the squad line's slope); 0 means
    # a substitution never changes this side's strength (the rest of the field,
    # whose strength comes from Elo alone)
    elo_per_ovr: float = 0.0


@dataclass
class MatchEvent:
    minute: str          # '17', '45+2', '90+4', '105'
    order: float
    kind: str            # goal | pen_goal | pen_saved | pen_missed | own_goal |
                         # yellow | second_yellow | red | sub | injury
    side: str            # 'home' | 'away'
    player: str
    assist: str | None = None
    player_in: str | None = None


@dataclass
class PlayerLine:
    key: str
    name: str
    position: str
    minutes: int = 0
    started: bool = True
    goals: int = 0
    pen_goals: int = 0
    pens_missed: int = 0
    own_goals: int = 0
    assists: int = 0
    yellows: int = 0
    red: bool = False
    injured: bool = False
    rating: float = 6.5


@dataclass
class MatchLog:
    home: str
    away: str
    home_goals: int
    away_goals: int
    events: list[MatchEvent]
    lines: dict[str, dict[str, PlayerLine]]      # side -> key -> line
    stats: dict[str, tuple]
    extra_time: bool = False
    shootout: tuple[int, int] | None = None
    shootout_kicks: list[dict] = field(default_factory=list)

    @property
    def winner(self) -> str | None:
        if self.shootout:
            return self.home if self.shootout[0] > self.shootout[1] else self.away
        if self.home_goals == self.away_goals:
            return None
        return self.home if self.home_goals > self.away_goals else self.away


# ── the measured rules ─────────────────────────────────────────────────────────
@dataclass
class Dynamics:
    raw: dict

    @property
    def clock(self) -> list[float]:
        return self.raw["goal_clock"]["per_minute_by_5min_bin"]

    def intensity(self, base: int, add: int) -> float:
        gc = self.raw["goal_clock"]
        if add and base == 45:
            return gc["stoppage_first_half"]
        if add and base >= 90:
            return gc["stoppage_second_half"]
        return self.clock[(min(base, 90) - 1) // 5]

    def draw_stoppage(self, half: int, rng) -> int:
        dist = self.raw["stoppage"][f"half{half}"]
        ks = [int(k) for k in dist]
        ps = np.array([dist[k] for k in dist], dtype=float)
        return int(rng.choice(ks, p=ps / ps.sum()))

    @property
    def mean_weight(self) -> float:
        """Total clock weight of an average match (the rate's denominator)."""
        w = 5.0 * sum(self.clock)
        for h, key in ((1, "stoppage_first_half"), (2, "stoppage_second_half")):
            dist = self.raw["stoppage"][f"half{h}"]
            mean_st = sum(int(k) * v for k, v in dist.items())
            w += mean_st * self.raw["goal_clock"][key]
        return w

    def state_factor(self, gd: int) -> float:
        se = self.raw["state_effects"]
        key = ("state:level" if gd == 0 else "state:lead1" if gd == 1 else "state:lead2" if gd >= 2
               else "state:trail1" if gd == -1 else "state:trail2")
        return float(se[key]["factor"])

    def red_factor(self, men: int, other: int) -> float:
        se = self.raw["state_effects"]
        down, up = float(se["red:down"]["factor"]), float(se["red:up"]["factor"])
        # each man short multiplies once; equal numbers cancel (10v10 plays as 11v11)
        diff = other - men
        return down ** diff if diff > 0 else up ** (-diff) if diff < 0 else 1.0

    @property
    def pen(self) -> dict:
        return self.raw["penalties"]

    @property
    def own_goal_share(self) -> float:
        return float(self.raw["goals"]["own_goal_share"])

    @property
    def assisted_share(self) -> float:
        return float(self.raw["goals"]["assisted_share_non_penalty"])

    def _mean_stoppage(self, half: int) -> float:
        dist = self.raw["stoppage"][f"half{half}"]
        return sum(int(k) * v for k, v in dist.items())

    def yellow_intensity(self, base: int, add: int) -> float:
        """Share of a match's yellows falling in this minute (sums to ~1).

        The measured bins are by the DISPLAYED minute, so 45'+k sits in the
        45-49 bin and 90'+k in the last one; each is spread over the minutes
        it really covers, stoppage included.
        """
        sh = self.raw["yellows"]["by_5min_share"]
        if base >= 90:
            return sh[-1] / (1.0 + self._mean_stoppage(2))
        b = base // 5
        if b == 9:
            return sh[9] / (5.0 + self._mean_stoppage(1))
        return sh[b] / (4.0 if b == 0 else 5.0)

    @property
    def second_yellow_hazard(self) -> float:
        return float(self.raw["yellows"]["second_yellow_hazard_per_min"])

    def direct_red_rate(self, base: int) -> float:
        """Direct reds per minute for one side, following the measured timing."""
        r = self.raw["red_cards"]
        hist = np.array(r["direct_by_15min"], dtype=float)
        share = hist / hist.sum()
        b = min((min(base, 90) - 1) // 15, len(share) - 1)
        return r["direct_per_team_match"] * share[b] / 15.0

    def sub_pattern(self, bucket: str, rng) -> list:
        pats = self.raw["substitutions"]["patterns"][bucket]
        return pats[int(rng.integers(len(pats)))] if pats else []

    def off_weight(self, position: str) -> float:
        w = self.raw.get("who_comes_off", {}).get(position, {})
        return float(w.get("p_off", 0.3))

    def own_goal_weight(self, position: str) -> float:
        return float(self.raw.get("own_goal_weights", {}).get(position, 0.002))

    @property
    def xg(self) -> dict:
        return self.raw["xg"]

    def injury_days(self, rng) -> int:
        """Days out, read off the Kaplan-Meier curve of 2026/27 injuries.

        The curve stops at the data's follow-up (~30 days); a player still out
        there is given that follow-up, a floor rather than a guess at the tail.
        """
        inj = self.raw.get("injury_absence") or {}
        curve = inj.get("still_out_after_days") or []
        if not curve:
            return 0
        u = rng.random()
        for day, surv in curve:
            if u > surv:
                return int(day)
        return int(inj.get("max_followup_days", curve[-1][0]))


@lru_cache(maxsize=1)
def load_dynamics(path: str | None = None) -> Dynamics:
    return Dynamics(json.loads(Path(path or DYNAMICS).read_text(encoding="utf-8")))


# ── the match ──────────────────────────────────────────────────────────────────
def _timeline(dyn: Dynamics, rng) -> list[tuple[str, int, int, float]]:
    """[(label, base, add, order)] for one match's regulation + stoppage."""
    s1, s2 = dyn.draw_stoppage(1, rng), dyn.draw_stoppage(2, rng)
    out = [(str(i), i, 0, float(i)) for i in range(1, 46)]
    out += [(f"45+{k}", 45, k, 45 + k / 100) for k in range(1, s1 + 1)]
    out += [(str(i), i, 0, 100.0 + i) for i in range(46, 91)]
    out += [(f"90+{k}", 90, k, 190 + k / 100) for k in range(1, s2 + 1)]
    return out


class _Team:
    """A side's live state during one match."""

    def __init__(self, side: Side, tag: str, typical_gap: dict[str, float]):
        self.side, self.tag = side, tag
        self.on = list(side.starters)
        self.bench = list(side.bench)
        self.lines = {p.key: PlayerLine(p.key, p.name, p.position) for p in side.starters}
        self.entered = {p.key: None for p in side.starters}      # key -> order entered (None = kick-off)
        self.subs_made = 0
        self.booked: set[str] = set()
        self.men = len(self.on)
        self.goals = 0
        self.strength = 1.0          # multiplicative, from substitutions
        self.gap = typical_gap
        self.pattern: list = []
        self.pens = 0

    # who is on
    def outfield(self) -> list[MatchPlayer]:
        return [p for p in self.on if p.position != "Goalkeeper"]

    def keeper(self) -> MatchPlayer | None:
        return next((p for p in self.on if p.position == "Goalkeeper"), None)

    def pick(self, rng, weights) -> MatchPlayer | None:
        pool = [p for p in self.on]
        w = np.array([max(float(weights(p)), 0.0) for p in pool])
        if not len(pool) or w.sum() <= 0:
            return None
        return pool[int(rng.choice(len(pool), p=w / w.sum()))]

    def taker(self) -> MatchPlayer | None:
        """The designated taker on the pitch: highest taker share, else best finisher."""
        cand = self.outfield() or self.on
        if not cand:
            return None
        best = max(cand, key=lambda p: (p.taker, p.g))
        return best


def play_match(home: Side, away: Side, rng: np.random.Generator, *,
               dyn: Dynamics | None = None, neutral: bool = False,
               extra_time_if_level: bool = False, shootout_if_level: bool = False,
               typical_gap: dict[str, float] | None = None,
               start_score: tuple[int, int] | None = None) -> MatchLog:
    """Play one match minute by minute.

    `lam` on each Side is the expected goals over the match (neutral or not is
    already inside it). `extra_time_if_level` plays thirty more minutes when the
    score — `start_score` + this match, for a second leg — is level after ninety,
    and `shootout_if_level` settles it from the spot if it still is.
    """
    dyn = dyn or load_dynamics()
    gap = typical_gap or {}
    H, A = _Team(home, "home", gap), _Team(away, "away", gap)
    other = {"home": A, "away": H}
    events: list[MatchEvent] = []
    tl = _timeline(dyn, rng)
    W = dyn.mean_weight
    norm = (_NORM_OVERRIDE if _NORM_OVERRIDE is not None
            else float(dyn.raw.get("norm") or _norm_constant()))
    pen_awards_per_goal = dyn.pen["awarded_per_team_match"] / max(1e-9, _league_goals_per_team(dyn))
    pen_conv = float(dyn.pen["conversion"])
    saved_share = float(dyn.pen["saved_share_of_failures"])
    og_share = dyn.own_goal_share
    # the side's goal rate splits into open play, penalties and own goals FOR it
    open_share = max(1.0 - og_share - pen_awards_per_goal * pen_conv, 0.5)

    # every side starts from a real substitution pattern of a level game; at
    # the hour a side ahead or behind switches to one from its own bucket
    for t in (H, A):
        t.pattern = list(dyn.sub_pattern("level", rng))

    # In a second leg a side "leads" on aggregate, not on the night: 1-0 down
    # tonight after winning the first leg 3-0 is still a side protecting a lead.
    base_h = start_score or (0, 0)
    carried = {"home": base_h[0], "away": base_h[1]}

    def gd_of(t: _Team) -> int:
        o = other[t.tag]
        return (t.goals + carried[t.tag]) - (o.goals + carried[o.tag])

    def rate_scale(t: _Team) -> float:
        # a change that strengthens one side raises its rate and lowers the
        # other's by the same Elo step, as the Elo->goals mapping does
        o = other[t.tag]
        return (t.side.lam * norm / W * dyn.state_factor(gd_of(t))
                * dyn.red_factor(t.men, o.men) * t.strength / o.strength)

    def do_sub(t: _Team, order: float, label: str, injury: bool) -> None:
        if t.subs_made >= MAX_SUBS or not t.bench:
            return
        if injury:
            cands = t.on
            if not cands:
                return
            out = cands[int(rng.integers(len(cands)))]
        else:
            starters = [p for p in t.on if t.entered.get(p.key) is None]
            if not starters:
                return
            w = np.array([dyn.off_weight(p.position) for p in starters])
            if w.sum() <= 0:
                return
            out = starters[int(rng.choice(len(starters), p=w / w.sum()))]
        same = [b for b in t.bench if b.position == out.position]
        if out.position == "Goalkeeper" and not same:
            return                                  # nobody to put in goal
        pool = same or [b for b in t.bench if b.position != "Goalkeeper"]
        if not pool:
            return
        inn = max(pool, key=lambda b: b.overall)
        t.bench.remove(inn)
        t.on[t.on.index(out)] = inn
        t.entered[inn.key] = order
        t.lines[inn.key] = PlayerLine(inn.key, inn.name, inn.position, started=False)
        t.subs_made += 1
        if injury:
            t.lines[out.key].injured = True
            events.append(MatchEvent(label, order, "injury", t.tag, out.name))
        events.append(MatchEvent(label, order, "sub", t.tag, out.name, player_in=inn.name))
        _left(t, out, order)
        if t.side.elo_per_ovr:
            delta = (inn.overall - out.overall) + gap.get(out.position, 0.0)
            d_elo = t.side.elo_per_ovr * delta / 11.0
            t.strength *= float(np.exp(ELO_B * d_elo / 400.0))

    def send_off(t: _Team, p: MatchPlayer, order: float, label: str, kind: str) -> None:
        t.lines[p.key].red = True
        events.append(MatchEvent(label, order, kind, t.tag, p.name))
        t.on.remove(p)
        t.men -= 1
        _left(t, p, order)
        # a keeper sent off: an outfield player goes in goal; if a keeper is on
        # the bench and a change is left, the manager sacrifices an outfielder
        if p.position == "Goalkeeper" and t.subs_made < MAX_SUBS:
            gk = next((b for b in t.bench if b.position == "Goalkeeper"), None)
            of = t.outfield()
            if gk and of:
                out = min(of, key=lambda x: x.g)
                t.bench.remove(gk)
                t.on[t.on.index(out)] = gk
                t.entered[gk.key] = order
                t.lines[gk.key] = PlayerLine(gk.key, gk.name, gk.position, started=False)
                t.subs_made += 1
                events.append(MatchEvent(label, order, "sub", t.tag, out.name, player_in=gk.name))
                _left(t, out, order)

    def _left(t: _Team, p: MatchPlayer, order: float) -> None:
        t.lines[p.key].minutes = _minutes_between(t.entered.get(p.key), order)
        t.entered[p.key] = ("gone", t.entered.get(p.key))

    def score_goal(t: _Team, order: float, label: str) -> None:
        o = other[t.tag]
        u = rng.random()
        if u < og_share / (og_share + open_share) and o.on:
            who = o.pick(rng, lambda p: dyn.own_goal_weight(p.position))
            if who is not None:
                o.lines[who.key].own_goals += 1
                events.append(MatchEvent(label, order, "own_goal", t.tag, who.name))
                t.goals += 1
                return
        scorer = t.pick(rng, lambda p: p.g if p.position != "Goalkeeper" else 0.0)
        if scorer is None:
            return
        assister = None
        if rng.random() < dyn.assisted_share and len(t.on) > 1:
            mates = [p for p in t.on if p.key != scorer.key]
            w = np.array([max(p.a, 1e-4) for p in mates])
            assister = mates[int(rng.choice(len(mates), p=w / w.sum()))]
        t.lines[scorer.key].goals += 1
        if assister:
            t.lines[assister.key].assists += 1
        events.append(MatchEvent(label, order, "goal", t.tag, scorer.name,
                                 assist=assister.name if assister else None))
        t.goals += 1

    def penalty(t: _Team, order: float, label: str) -> None:
        taker = t.taker()
        if taker is None:
            return
        t.pens += 1
        if rng.random() < pen_conv:
            t.lines[taker.key].goals += 1
            t.lines[taker.key].pen_goals += 1
            events.append(MatchEvent(label, order, "pen_goal", t.tag, taker.name))
            t.goals += 1
        else:
            t.lines[taker.key].pens_missed += 1
            kind = "pen_saved" if rng.random() < saved_share else "pen_missed"
            gk = other[t.tag].keeper()
            events.append(MatchEvent(label, order, kind, t.tag, taker.name,
                                     assist=gk.name if (gk and kind == "pen_saved") else None))

    def minute(label: str, base: int, add: int, order: float, et: bool = False) -> None:
        # substitutions due this minute (the pattern is re-drawn at the hour by score)
        if not et and label == "60":
            for t in (H, A):
                gd = gd_of(t)
                bucket = "level" if gd == 0 else ("lead" if gd > 0 else "trail")
                if bucket != "level":
                    fresh = dyn.sub_pattern(bucket, rng)
                    t.pattern = [e for e in t.pattern if _order_of(e[0]) < order] + \
                                [e for e in fresh if _order_of(e[0]) >= order]
        for t in (H, A):
            # ESPN stamps half-time changes 45'; they happen at the restart
            for lab, inj in [e for e in t.pattern if _due(e[0]) == label]:
                do_sub(t, order, label, bool(inj))
        w = 1.0 if et else dyn.intensity(base, add)
        for t in (H, A):
            if not t.on:
                continue
            r = rate_scale(t) * w
            n_open = rng.poisson(r * (open_share + og_share))
            for _ in range(int(n_open)):
                score_goal(t, order, label)
            if rng.random() < r * pen_awards_per_goal:
                penalty(t, order, label)
        # bookings
        for t in (H, A):
            yl = t.side.yellows_lam * (1.0 / 3.0 / 30.0 if et else dyn.yellow_intensity(base, add))
            if rng.random() < yl:
                fresh = [p for p in t.on if p.key not in t.booked]
                if fresh:
                    w_ = np.array([max(p.yc, 1e-3) for p in fresh])
                    p = fresh[int(rng.choice(len(fresh), p=w_ / w_.sum()))]
                    t.booked.add(p.key)
                    t.lines[p.key].yellows += 1
                    events.append(MatchEvent(label, order, "yellow", t.tag, p.name))
            for p in [q for q in t.on if q.key in t.booked]:
                if rng.random() < dyn.second_yellow_hazard:
                    t.lines[p.key].yellows += 1
                    send_off(t, p, order, label, "second_yellow")
            if rng.random() < dyn.direct_red_rate(base if not et else 90):
                p = t.pick(rng, lambda q: max(q.yc, 1e-3))
                if p is not None:
                    send_off(t, p, order, label, "red")

    for label, base, add, order in tl:
        minute(label, base, add, order)
    end = tl[-1][3]

    extra = False
    shoot = None
    kicks: list[dict] = []
    agg_level = (H.goals + base_h[0]) == (A.goals + base_h[1])
    if extra_time_if_level and agg_level:
        extra = True
        # extra time: thirty minutes at the match-average intensity
        for i in range(91, 121):
            minute(str(i), i, 0, 200.0 + i, et=True)
        end = 320.0
        agg_level = (H.goals + base_h[0]) == (A.goals + base_h[1])
    if shootout_if_level and agg_level:
        shoot, kicks = _shootout(H, A, rng, pen_conv, saved_share)

    for t in (H, A):
        for p in t.on:
            if not isinstance(t.entered.get(p.key), tuple):
                t.lines[p.key].minutes = _minutes_between(t.entered.get(p.key), end)

    stats = _stats(H, A, dyn, rng)
    events.sort(key=lambda e: e.order)
    return MatchLog(home.name, away.name, H.goals, A.goals, events,
                    {"home": H.lines, "away": A.lines}, stats, extra, shoot, kicks)


def _due(label: str) -> str:
    """The minute a pattern's change is made in: '45' (half-time) -> '46'."""
    return "46" if label == "45" else label


def _order_of(label: str) -> float:
    if "+" in label:
        b, a = label.split("+")
        return (45 + int(a) / 100) if b == "45" else (190 + int(a) / 100)
    m = int(label)
    return float(m) if m <= 45 else 100.0 + m


def _clock(order: float | None) -> float:
    """A position in the match on the playing clock: 45+2 is 45, 90+3 is 90."""
    if order is None:
        return 0.0
    if order < 100:
        return min(order, 45.0)
    if order < 200:
        return min(order - 100.0, 90.0)
    return order - 200.0              # extra time, 91..120


def _minutes_between(start: float | None, end_order: float) -> int:
    """Minutes on the pitch, at least one for anybody who came on."""
    return int(max(round(_clock(end_order) - _clock(start)), 1))


def _shootout(H: _Team, A: _Team, rng, conv: float, saved_share: float) -> tuple[tuple[int, int], list]:
    """Five each, then sudden death, takers in order of who takes them.

    Every kick converts at the measured rate: across seasons a taker's
    conversion does not persist, so skill cannot be read here — the order only
    decides whose name goes on the kick."""
    def order(t: _Team) -> list[MatchPlayer]:
        return sorted(t.on, key=lambda p: (p.position == "Goalkeeper", -p.taker, -p.g))

    oh, oa = order(H), order(A)
    sh = sa = 0
    kicks = []
    k = 0
    while True:
        for t, lst, tag in ((H, oh, "home"), (A, oa, "away")):
            if not lst:
                continue
            p = lst[k % len(lst)]
            scored = rng.random() < conv
            kicks.append({"side": tag, "player": p.name, "scored": bool(scored)})
            if tag == "home":
                sh += int(scored)
            else:
                sa += int(scored)
        k += 1
        if k < 5:
            left = 5 - k
            if sh > sa + left or sa > sh + left:
                break
            continue
        if sh != sa:
            break
        if k > 30:
            break
    return (sh, sa), kicks


def _stats(H: _Team, A: _Team, dyn: Dynamics, rng) -> dict[str, tuple]:
    """Shots, shots on target, corners and xG consistent with what happened.

    Shots are drawn from the side's expected count but never below what the
    match needs (every non-own-goal goal was a shot on target, every saved or
    missed penalty a shot); xG follows our own text-xG fit on goals and shots.
    """
    out: dict[str, list] = {"shots": [], "sot": [], "corners": [], "xg": [], "yellows": []}
    x = dyn.xg
    for t in (H, A):
        goals_shot = sum(l.goals for l in t.lines.values())
        pens = t.pens
        shots = max(int(rng.poisson(t.side.shots_lam)), goals_shot + max(pens - sum(l.pen_goals for l in t.lines.values()), 0))
        sot = max(int(rng.binomial(shots, 0.34)), goals_shot)
        sot = min(sot, shots)
        np_goals = goals_shot - sum(l.pen_goals for l in t.lines.values())
        xg = (x["per_goal"] * np_goals + x["per_other_shot"] * max(shots - np_goals - pens, 0)
              + x["penalty_xg"] * pens + rng.normal(0.0, x["resid_sd"] * 0.5))
        out["shots"].append(shots)
        out["sot"].append(sot)
        out["corners"].append(int(rng.poisson(t.side.corners_lam)))
        out["xg"].append(round(max(float(xg), 0.05), 2))
        out["yellows"].append(sum(l.yellows for l in t.lines.values()))
    return {k: tuple(v) for k, v in out.items()}


def _league_goals_per_team(dyn: Dynamics) -> float:
    """Goals per side per match in the matches the dynamics were measured on."""
    d = dyn.raw
    p = d["penalties"]
    goals = p["scored"] / max(p["penalty_goals_share_of_goals"], 1e-9)
    return float(goals / (2 * d["n_matches"]))


@lru_cache(maxsize=1)
def _norm_constant() -> float:
    """See compute_norm; cached for a dynamics file written without one."""
    return compute_norm()


def compute_norm(dyn: Dynamics | None = None) -> float:
    """Scale that keeps simulated goals on the team model's expectation.

    The game-state and red-card factors are measured relative to each side's
    OWN expectation, but applied inside a match they shift the average a little
    (−1.7% on the measured matches). One simulation pass at a typical pair of
    rates finds the constant; it is not sensitive to the pair.
    """
    dyn = dyn or load_dynamics()
    rng = np.random.default_rng(12345)
    lam = (1.45, 1.20)
    n = 8000

    shape = ["Goalkeeper"] + ["Defender"] * 4 + ["Midfielder"] * 3 + ["Forward"] * 3

    def dummy(name):
        return [MatchPlayer(f"{name}{i}", f"{name}{i}", pos) for i, pos in enumerate(shape)]

    tot = 0.0
    global _NORM_OVERRIDE
    _NORM_OVERRIDE = 1.0
    try:
        for _ in range(n):
            log = play_match(Side("h", dummy("h"), lam=lam[0]), Side("a", dummy("a"), lam=lam[1]),
                             rng, dyn=dyn)
            tot += log.home_goals + log.away_goals
    finally:
        _NORM_OVERRIDE = None
    return float(sum(lam) * n / max(tot, 1.0))


_NORM_OVERRIDE: float | None = None


# ── ratings ────────────────────────────────────────────────────────────────────
# Sofascore-style 0-10 marks. HEURISTIC, as the old ones were (player_rating.py):
# no public ground truth exists to fit a match rating on, so the bonuses are
# the old ones plus the events the live match now has — a missed penalty, an
# own goal, a sending-off, a keeper's penalty save — and everything scales with
# the minutes played, so a late substitute stays near the base mark.
_PEN_MISS = 0.6
_OWN_GOAL = 0.8
_RED = 1.3
_PEN_SAVE = 0.8


def rate_lines(log: MatchLog, rng) -> None:
    from mundialytics.statistical_core.squadlab.player_rating import (
        ASSIST_BONUS, CLEAN_SHEET_BONUS, CONCESSION_PENALTY_CAP, CONCESSION_PENALTY_PER_GOAL,
        RATING_BASE, RATING_MAX, RATING_MIN, YELLOW_CARD_PENALTY, _goal_bonus,
    )
    from mundialytics.statistical_core.player_strength import POSITION_DEFENSE_WEIGHT

    saves = {}
    for e in log.events:
        if e.kind == "pen_saved" and e.assist:
            saves[e.assist] = saves.get(e.assist, 0) + 1
    for side, conceded in (("home", log.away_goals), ("away", log.home_goals)):
        for line in log.lines[side].values():
            share = min(line.minutes, 90) / 90.0
            r = _goal_bonus(line.goals) + ASSIST_BONUS * line.assists
            if conceded == 0 and line.minutes >= 60:
                r += CLEAN_SHEET_BONUS.get(line.position, 0.0)
            r -= min(CONCESSION_PENALTY_PER_GOAL * conceded
                     * POSITION_DEFENSE_WEIGHT.get(line.position, 0.40), CONCESSION_PENALTY_CAP) * share
            r -= YELLOW_CARD_PENALTY * min(line.yellows, 1)
            r -= _PEN_MISS * line.pens_missed + _OWN_GOAL * line.own_goals + _RED * line.red
            if line.position == "Goalkeeper":
                r += _PEN_SAVE * saves.get(line.name, 0)
            r += float(rng.normal(0.0, 0.35)) * max(share, 0.3)
            line.rating = float(np.clip(RATING_BASE + r, RATING_MIN, RATING_MAX))
