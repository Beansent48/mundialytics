"""
SquadLab for the HTTP API.

Two modes reach the web: Draft (your club replaces the league's bottom side and
you sign eleven players, each pick removed from the global pool) and Sandbox
(any eleven, no constraints). Both wire the squad into the same Poisson /
Dixon-Coles machinery the real clubs already use, so a made-up team is priced by
the engine rather than by a separate toy.

The season is played once and returned whole. Streaming it matchday by matchday
would be a nicer animation and a much worse contract: the client wants to move
at its own pace and to jump to the final table, and both are trivial once it
holds every result.
"""
from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SQUAD_TEAM_NAME = "Your XI"

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}

# How many of each card kind the shortlist carries, per position. The catalogue
# holds 138 primes and 62 icons against 2,811 current players, but sorting the
# lot by rating floats every special to the top and a draft becomes "take the
# eleven legends" — the shortlist has to carry the scarcity the pack odds carry
# in the Streamlit draft.
POOL_MIX = {"actual": 34, "prime": 4, "icono": 2}

# The player-profile file and the results file spell competitions differently.
PLAYER_COMP = {
    "LaLiga": "La Liga",
    "Premier League": "Premier League",
    "Serie A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue 1",
}


@lru_cache(maxsize=1)
def strength_model():
    from mundialytics.statistical_core.player_strength import PlayerStrengthModel

    model = PlayerStrengthModel()
    model.fit()
    return model


def _card_entry(c) -> dict:
    """One card in the shape the client draws.

    `player` is the CARD id, not the man's name: a player has up to three cards
    (this season's, a prime season, an icon) and the client has to be able to
    post back which one it drafted.
    """
    return {
        "player": c.card_id,
        "display": c.display,
        "team": c.club if c.kind != "actual" else c.club.title(),
        "position": c.position,
        "role": c.role,
        "kind": c.kind,
        "season": c.season_label or None,
        "overall": round(float(c.overall), 1),
        # A keeper has one rating, shot-stopping, and the client reads it out of
        # the attack slot (see player-card.tsx). The other two axes are not
        # small for him, they do not exist.
        "attack": round(float(c.gk if c.position == "Goalkeeper" else c.attack), 1),
        "defense": None if c.position == "Goalkeeper" else round(float(c.defense), 1),
        "creation": None if c.position == "Goalkeeper" else round(float(c.creation), 1),
        "measured": bool(c.n_def > 0),
        "matches": int(c.matches),
    }


def pool(competition_id: str, limit_per_position: int = 40) -> dict:
    """The cards available to draft, by position, best first.

    Reads the card catalogue rather than the rating model. The model files every
    player under the club he is REMEMBERED for and knows one version of him, so
    the draft used to offer Messi at Barcelona and nothing else; the catalogue is
    built from this week's squads and carries the primes and the icons too. See
    scripts/build_squadlab_cards.py.

    The whole shortlist goes to the client in one response rather than five
    candidates at a time: rerolling a slot then costs nothing, and the draft
    stops feeling like a form.
    """
    from mundialytics.statistical_core.squadlab.cards import cards_from_frame, load_cards

    df = load_cards()
    if df.empty:                      # catalogue not built: fall back to the model
        return _pool_from_profiles(competition_id, limit_per_position)

    comp = PLAYER_COMP.get(competition_id, competition_id)
    # Primes and icons belong to no current league, and excluding them from a
    # league draft would remove the two card types worth chasing.
    keep = (df["kind"] != "actual") | (df["league"].isin({comp, competition_id}))
    by_pos: dict[str, list[dict]] = {p: [] for p in POSITIONS}
    for pos in POSITIONS:
        at_pos = df[keep & (df["position"] == pos)]
        picked: list[dict] = []
        for kind, n in POOL_MIX.items():
            grp = at_pos[at_pos["kind"] == kind]
            if grp.empty:
                continue
            if kind == "actual":
                # the best current players in the league, straightforwardly
                chosen = grp.nlargest(n, "overall")
            else:
                # a stable sample rather than the top n: otherwise every league
                # in every session is offered the same two icons, and the rarest
                # cards in the game turn into fixtures of the furniture
                chosen = grp.nlargest(min(len(grp), n * 6), "overall")
                chosen = _stable_sample(chosen, f"{competition_id}|{pos}|{kind}", n)
            picked += [_card_entry(c) for c in cards_from_frame(chosen)]
        by_pos[pos] = sorted(picked, key=lambda x: -x["overall"])[:limit_per_position]
    return {"slots": SLOTS, "positions": POSITIONS, "players": by_pos}


def _stable_sample(df: pd.DataFrame, seed: str, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    seed_int = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
    return df.sample(n=n, random_state=seed_int)


def _pool_from_profiles(competition_id: str, limit_per_position: int) -> dict:
    """The pre-catalogue pool, kept so a missing CSV degrades instead of 503s."""
    model = strength_model()
    comp = PLAYER_COMP.get(competition_id, competition_id)
    by_pos: dict[str, list[dict]] = {p: [] for p in POSITIONS}
    for name, prof in model.profiles_.items():
        if prof.position not in by_pos or getattr(prof, "matches", 0) < 3:
            continue
        if competition_id and getattr(prof, "competition", None) not in (comp, None):
            continue
        by_pos[prof.position].append({
            "player": name,
            "display": _display(name),
            "team": str(getattr(prof, "team", "") or "").title(),
            "position": prof.position,
            "role": getattr(prof, "role", ""),
            "kind": "actual",
            "season": None,
            "overall": round(float(prof.overall), 1),
            "attack": round(float(prof.gk_strength if prof.position == "Goalkeeper"
                                  else prof.offensive_strength), 1),
            "defense": round(float(prof.defensive_strength), 1),
            "creation": round(float(prof.creation_strength), 1),
            "measured": True,
            "matches": int(getattr(prof, "matches", 0)),
        })
    for p in by_pos:
        by_pos[p] = sorted(by_pos[p], key=lambda x: -x["overall"])[:limit_per_position]
    return {"slots": SLOTS, "positions": POSITIONS, "players": by_pos}


def _display(name: str) -> str:
    """Short name for the UI, clone mark handled first.

    `display_name` shortens to the recognisable part, which for "Lamine Yamal ✦"
    is the last token — the mark itself. Every drafted scorer came back as "✦"
    until the mark was stripped before shortening. Squad membership is carried
    by the `isSquad` flag, so the name does not need to encode it.
    """
    from mundialytics.serving.squad_roster import strip_clone
    from mundialytics.statistical_core.squadlab.cards import split_card_mark

    # A prime and an icon carry their own mark for the same reason a clone does:
    # they are separate identities in the scorer race. Shortening eats it —
    # "Suárez 15/16" came back as "15/16" — so it is put back afterwards.
    base, mark = split_card_mark(strip_clone(name))
    try:
        from mundialytics.identity.display_names import display_name

        return display_name(base) + mark
    except Exception:
        return base + mark


def seeded_sample(candidates: list[dict], seed: str, n: int = 5) -> list[dict]:
    """A stable random five from the shortlist.

    Seeded per slot so Defender #1 and Defender #4 see different faces instead
    of the same top five, and so a reload of the same slot shows the same set.
    """
    if len(candidates) <= n:
        return candidates
    seed_int = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2**32)
    sample = random.Random(seed_int).sample(candidates, n)
    return sorted(sample, key=lambda x: -x["overall"])


def league_opponents(df_clubs: pd.DataFrame, competition_id: str,
                     drop_last: bool) -> tuple[list[str], str | None]:
    """The real clubs the squad plays against, and the one it replaced."""
    season = sorted(df_clubs.loc[df_clubs["competition"] == competition_id,
                                 "season"].unique())[-1]
    rows = df_clubs[(df_clubs["competition"] == competition_id)
                    & (df_clubs["season"] == season)]
    teams = sorted(set(rows["home_team"]) | set(rows["away_team"]))
    if not drop_last:
        return teams, None

    points: dict[str, int] = {t: 0 for t in teams}
    for r in rows.itertuples():
        if pd.isna(r.home_goals) or pd.isna(r.away_goals):
            continue
        h, a = int(r.home_goals), int(r.away_goals)
        points[r.home_team] += 3 if h > a else 1 if h == a else 0
        points[r.away_team] += 3 if a > h else 1 if h == a else 0
    bottom = min(points, key=lambda t: points[t])
    return [t for t in teams if t != bottom], bottom


def play_season(engine, df_clubs: pd.DataFrame, competition_id: str,
                squad_names: list[str], drop_last: bool,
                n_sims: int = 20_000) -> dict:
    """Play the full season once, then price it with a Monte Carlo layer."""
    from mundialytics.statistical_core.squadlab.calendar import (
        generate_double_round_robin,
    )
    from mundialytics.statistical_core.squadlab.lambda_source import (
        RealTeamLambdaSource,
        SeasonLambdaSource,
    )
    from mundialytics.statistical_core.squadlab.season_simulator import (
        SeasonOrchestrator,
    )
    from mundialytics.statistical_core.squadlab.squad_lambda_model import (
        SquadLambdaModel,
    )

    model = strength_model()
    squad = resolve_squad(model, squad_names)
    if len(squad) < 11:
        raise ValueError("The squad needs eleven players the model knows")

    opponents, replaced = league_opponents(df_clubs, competition_id, drop_last)

    real_source = RealTeamLambdaSource(engine)
    bridge = SquadLambdaModel(model)
    lambda_source = SeasonLambdaSource(
        SQUAD_TEAM_NAME, squad, bridge, real_source, engine.ad_model_
    )
    fixtures = generate_double_round_robin([SQUAD_TEAM_NAME] + opponents)

    # Drafted players are clones: the real player keeps playing for his club and
    # both appear separately in the scorer race. Without cloning, one name in two
    # rosters double-counts his goals.
    from mundialytics.serving.squad_roster import clone_for_squad, real_team_rosters

    rosters = {SQUAD_TEAM_NAME: [clone_for_squad(p, SQUAD_TEAM_NAME) for p in squad]}
    rosters.update(real_team_rosters(model, opponents))

    orch = SeasonOrchestrator(lambda_source, fixtures, squad_roster=rosters,
                              competition=competition_id)
    season = orch.play_once(narrative=True)
    mc = orch.run_monte_carlo(n_sims=n_sims)

    return _shape_season(season, mc, replaced)


def resolve_squad(model, names: list[str]):
    """Card ids (or, for an old client, player names) -> playable profiles.

    A profile built from a card carries the MEASURED axes, not the ones printed
    on its face: the scorer weights and the squad-to-lambda bridge were fitted
    against the measured scale, and handing them card-face numbers would flatten
    the gap between a striker and a full-back. See squadlab/cards.py.
    """
    from mundialytics.statistical_core.squadlab.cards import cards_from_frame, load_cards

    df = load_cards()
    by_id = {}
    if not df.empty:
        by_id = {c.card_id: c for c in cards_from_frame(df)}
    out = []
    for n in names:
        card = by_id.get(str(n))
        if card is not None:
            out.append(card.to_profile())
            continue
        prof = model.get(n)             # a name, from a client older than the catalogue
        if prof is not None:
            out.append(prof)
    return out


def _shape_season(season, mc: pd.DataFrame, replaced: str | None) -> dict:
    """The season in the shape the client reveals matchday by matchday."""
    by_md: dict[int, list[dict]] = {}
    for m in season.matches:
        is_squad = SQUAD_TEAM_NAME in (m.home, m.away)
        by_md.setdefault(int(m.matchday), []).append({
            "home": _label(m.home),
            "away": _label(m.away),
            "homeGoals": int(m.home_goals),
            "awayGoals": int(m.away_goals),
            "isSquad": is_squad,
            # The narrative detail only travels for the eleven's own matches:
            # it is the only one the client plays out, and shipping it for all
            # 380 would multiply the response for nothing.
            "events": _timeline(m) if is_squad else None,
            "stats": _match_stats(m) if is_squad else None,
            "ratings": _squad_ratings(m) if is_squad else None,
        })

    table = season.table.copy()
    table["team"] = table["team"].map(_label)
    standings = [
        {
            "rank": i + 1,
            "team": r["team"],
            "played": int(r["played"]),
            "points": int(r["pts"]),
            "goalsFor": int(r["gf"]),
            "goalsAgainst": int(r["ga"]),
            "isSquad": r["team"] == SQUAD_TEAM_NAME,
        }
        for i, r in enumerate(table.to_dict("records"))
    ]

    tallies = season.player_season_tallies
    scorers = []
    if tallies is not None and len(tallies) and "goals" in tallies.columns:
        top = tallies.nlargest(10, "goals")
        scorers = [
            {
                "player": _display(str(r.player)),
                "team": _label(str(getattr(r, "team", "") or "")),
                "goals": int(r.goals),
                "assists": int(getattr(r, "assists", 0) or 0),
                "isSquad": str(getattr(r, "team", "")) == SQUAD_TEAM_NAME,
            }
            for r in top.itertuples()
        ]

    odds = None
    if mc is not None and len(mc):
        row = mc[mc["team"] == SQUAD_TEAM_NAME]
        if len(row):
            r = row.iloc[0]
            # Column names come from SeasonOrchestrator.run_monte_carlo:
            # p_champion / p_top2 / p_top4 / p_relegation / avg_pts / avg_goals.
            odds = {
                key: (None if col not in mc.columns else round(float(r[col]), 4))
                for key, col in [
                    ("title", "p_champion"),
                    ("top2", "p_top2"),
                    ("top4", "p_top4"),
                    ("relegation", "p_relegation"),
                ]
            }
            if "avg_pts" in mc.columns:
                odds["expectedPoints"] = round(float(r["avg_pts"]), 1)
            if "avg_goals" in mc.columns:
                odds["expectedGoals"] = round(float(r["avg_goals"]), 1)

    squad_row = next((s for s in standings if s["isSquad"]), None)
    return {
        "teamName": SQUAD_TEAM_NAME,
        "replaced": _label(replaced) if replaced else None,
        "matchdays": [{"matchday": k, "fixtures": v} for k, v in sorted(by_md.items())],
        "standings": standings,
        "scorers": scorers,
        "odds": odds,
        "finish": squad_row,
    }


def _label(team: str) -> str:
    return team if team == SQUAD_TEAM_NAME else str(team).title()


def _timeline(m) -> list[dict]:
    """Goals and bookings in the order they happened.

    The simulator returns *what* happened, not *when*. Conditional on the score,
    a Poisson process places its events uniformly across the ninety minutes, so
    the clock is drawn here from the same model that produced the score. It is
    done at serving time on a separate seeded stream rather than inside the
    core: drawing there would consume the simulator's RNG and move every result
    in the season.
    """
    rng = random.Random(
        f"{m.matchday}|{m.home}|{m.away}|{m.home_goals}-{m.away_goals}"
    )
    out: list[dict] = []
    for side, goals, cards in (
        ("home", m.home_goal_events, m.home_card_players),
        ("away", m.away_goal_events, m.away_card_players),
    ):
        for entry in goals or []:
            pair = entry if isinstance(entry, (tuple, list)) else (entry, None)
            assist = pair[1] if len(pair) > 1 else None
            out.append({
                "type": "goal",
                "side": side,
                "player": _display(str(pair[0])),
                "assist": _display(str(assist)) if assist else None,
                "minute": rng.randint(1, 90),
            })
        for player in cards or []:
            out.append({
                "type": "card",
                "side": side,
                "player": _display(str(player)),
                "assist": None,
                "minute": rng.randint(1, 90),
            })

    # A handful of clubs have no rated players in the profile file, so the
    # simulator scores their goals without naming a scorer. Unnamed goals still
    # have to appear, or the running score would stop short of the real result
    # and the scoreboard would jump at full time.
    for side, team, goals in (
        ("home", m.home, int(m.home_goals)),
        ("away", m.away, int(m.away_goals)),
    ):
        named = sum(1 for e in out if e["type"] == "goal" and e["side"] == side)
        for _ in range(max(goals - named, 0)):
            out.append({
                "type": "goal",
                "side": side,
                "player": _label(team),
                "assist": None,
                "minute": rng.randint(1, 90),
            })

    out.sort(key=lambda e: e["minute"])
    return out


# Everything the event-lambda model draws for a match, in the order a reader
# scans it: the two figures that decide the game, then the volume behind them.
STAT_KEYS = [
    ("xg", "home_xg", "away_xg"),
    ("shots", "home_shots", "away_shots"),
    ("sot", "home_sot", "away_sot"),
    ("corners", "home_corners", "away_corners"),
    ("fouls", "home_fouls", "away_fouls"),
    ("cards", "home_yellow_cards", "away_yellow_cards"),
]


def _match_stats(m) -> list[dict]:
    return [
        {
            "key": key,
            "home": round(float(getattr(m, h, 0) or 0), 2),
            "away": round(float(getattr(m, a, 0) or 0), 2),
        }
        for key, h, a in STAT_KEYS
    ]


def _squad_ratings(m) -> list[dict]:
    """The eleven's own player ratings for this match, best first."""
    events = m.home_events if m.home == SQUAD_TEAM_NAME else m.away_events
    rows = [
        {
            "player": _display(str(e.player)),
            "rating": round(float(getattr(e, "rating", 0) or 0), 1),
            "goals": int(getattr(e, "goals", 0) or 0),
            "assists": int(getattr(e, "assists", 0) or 0),
            "cards": int(getattr(e, "yellow_cards", 0) or 0),
        }
        for e in (events or {}).values()
    ]
    return sorted(rows, key=lambda r: -r["rating"])
