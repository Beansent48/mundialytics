"""
SquadLab for the HTTP API.

Your drafted eleven ALWAYS plays the real Champions League. Two build methods
reach the web: Draft (sign eleven from packs, each pick removed from the global
pool) and Sandbox (any eleven, no constraints). Either way the squad takes a
RANDOM real club's slot in the actual 36-team draw, so the eight league-phase
opponents and the knockout path change every time you play (see
squadlab/champions.py). The domestic-league season was retired: two thirds of a
Champions field have no row in the club engine's AttackDefenseModel, so the
tournament runs on Elo instead, and the squad reads its Elo off a line fitted
against the real clubs' own best elevens.

The whole run is played once and returned whole. Streaming it round by round
would be a nicer animation and a much worse contract: the client wants to move
at its own pace and to jump to the final, and both are trivial once it holds
every result.
"""
from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from pathlib import Path

import pandas as pd

from mundialytics.statistical_core.squadlab.champions import (
    ROUND_LABELS,
    SQUAD_TEAM_NAME,
)

ROOT = Path(__file__).resolve().parents[1]

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}

MICRO_STATS = ROOT / "data/processed/player_micro_stats.csv"

# The measured sub-stats, per player, that feed the role rating — the depth
# behind a card. Outfield first, keeper last; a card only carries the set for
# its own position. Percentiles within (season, position), 0-100 on the wire.
OUTFIELD_SUBSTATS = [
    "FIN", "BOX", "CONV", "XA", "BIGC", "THRU", "CROSS", "SETP_TAKE", "SETP_FIN",
    "DRIB", "PROG", "LONG", "RETEN", "FOULDRAWN", "DUEL", "AER", "INT", "RECOV",
    "HIGHREG", "GA_ON",
]
GK_SUBSTATS = ["SHOTSTOP", "AERIAL_GK", "SWEEP", "DISTRIB"]

# Human labels for the popup, so a reader sees "Finalización", not "FIN".
SUBSTAT_LABELS = {
    "FIN": "Finalización", "BOX": "Presencia en área", "CONV": "Conversión de tiro",
    "XA": "Asistencia esperada", "BIGC": "Ocasiones claras creadas",
    "THRU": "Pases al hueco", "CROSS": "Centros", "SETP_TAKE": "Balón parado (ejecuta)",
    "SETP_FIN": "Balón parado (remate)", "DRIB": "Regate", "PROG": "Progresión",
    "LONG": "Pase largo / cambio", "RETEN": "Retención de balón",
    "FOULDRAWN": "Faltas recibidas", "DUEL": "Duelos ganados", "AER": "Juego aéreo",
    "INT": "Intercepciones", "RECOV": "Recuperaciones", "HIGHREG": "Presión alta",
    "GA_ON": "Goles evitados", "SHOTSTOP": "Paradas", "AERIAL_GK": "Salidas aéreas",
    "SWEEP": "Portero-líbero", "DISTRIB": "Distribución",
}

# The role formula: how much each sub-stat weighs in a role's rating, /100.
# MIRROR of ROLE_WEIGHTS v3 in scripts/build_role_ratings.py (the source of
# truth that produces the ratings) — kept here only to show the formula in the
# card popup. If the weights change there, change them here.
ROLE_WEIGHTS = {
    "Killer":             {"FIN": 42, "BOX": 22, "CONV": 12, "XA": 8, "SETP_FIN": 6, "AER": 5, "HIGHREG": 5},
    "Target man":         {"FIN": 28, "AER": 18, "BOX": 16, "SETP_FIN": 12, "RETEN": 10, "CONV": 8, "FOULDRAWN": 8},
    "Delantero completo": {"FIN": 30, "XA": 17, "DRIB": 14, "BOX": 13, "CONV": 11, "PROG": 10, "AER": 5},
    "Falso 9":            {"XA": 26, "FIN": 18, "THRU": 16, "DRIB": 12, "FOULDRAWN": 12, "RETEN": 10, "HIGHREG": 6},
    "Extremo":            {"DRIB": 28, "FIN": 30, "CROSS": 12, "PROG": 12, "BOX": 12, "FOULDRAWN": 6},
    "Extremo interior":   {"FIN": 28, "DRIB": 22, "XA": 18, "BOX": 14, "CONV": 10, "HIGHREG": 8},
    "Mediapunta":         {"XA": 26, "BIGC": 16, "THRU": 14, "FIN": 14, "DRIB": 12, "PROG": 10, "SETP_TAKE": 8},
    "Creador":            {"XA": 22, "THRU": 18, "PROG": 18, "RETEN": 14, "DRIB": 10, "LONG": 10, "SETP_TAKE": 8},
    "Box-to-box":         {"PROG": 16, "FIN": 12, "DUEL": 12, "HIGHREG": 12, "XA": 12, "RECOV": 10, "RETEN": 11, "GA_ON": 10, "AER": 5},
    "Pivote organizador": {"RETEN": 20, "PROG": 18, "LONG": 14, "DUEL": 12, "GA_ON": 12, "INT": 10, "RECOV": 8, "XA": 6},
    "Destructor":         {"DUEL": 26, "INT": 18, "RECOV": 16, "HIGHREG": 12, "GA_ON": 12, "AER": 10, "RETEN": 6},
    "Central stopper":    {"DUEL": 22, "AER": 22, "GA_ON": 20, "INT": 14, "RETEN": 8, "PROG": 8, "SETP_FIN": 6},
    "Central de salida":  {"RETEN": 18, "PROG": 16, "GA_ON": 16, "DUEL": 15, "AER": 12, "INT": 9, "LONG": 8, "THRU": 6},
    "Lateral ofensivo":   {"PROG": 16, "XA": 14, "DRIB": 14, "GA_ON": 14, "CROSS": 12, "DUEL": 12, "RETEN": 10, "AER": 8},
    "Lateral defensivo":  {"DUEL": 22, "GA_ON": 18, "INT": 14, "RETEN": 14, "RECOV": 12, "AER": 10, "PROG": 10},
    "Portero":            {"SHOTSTOP": 55, "AERIAL_GK": 18, "DISTRIB": 15, "SWEEP": 12},
}

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


def _fold_name(s: object) -> str:
    """Accent/transliteration-insensitive key, matching the identity layer."""
    import unicodedata

    try:
        from mundialytics.identity.current_squads import _fold

        return _fold(str(s))
    except Exception:
        text = str(s).translate(str.maketrans(
            {"ı": "i", "İ": "i", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l",
             "đ": "d", "Đ": "d", "ð": "d", "Ð": "d", "ß": "ss", "æ": "ae"}))
        return "".join(ch for ch in unicodedata.normalize("NFKD", text)
                       if not unicodedata.combining(ch)).lower().strip()


@lru_cache(maxsize=1)
def _micro_index() -> dict:
    """folded player name -> {SUBSTAT: percentile 0-100}, the measured profile.

    Read straight off player_micro_stats.csv (the depth behind a role rating).
    On a name collision the row with the most minutes wins — the man actually
    playing this season, not a lower-league namesake.
    """
    if not MICRO_STATS.exists():
        return {}
    df = pd.read_csv(MICRO_STATS)
    codes = [c for c in (OUTFIELD_SUBSTATS + GK_SUBSTATS) if c in df.columns]
    if "n90" in df.columns:
        df = df.sort_values("n90", ascending=False)
    out: dict[str, dict[str, int]] = {}
    for r in df.itertuples(index=False):
        key = _fold_name(getattr(r, "player", ""))
        if not key or key in out:
            continue
        prof = {}
        for code in codes:
            v = getattr(r, code, None)
            if v is not None and not pd.isna(v):
                prof[code] = int(round(float(v) * 100))
        if prof:
            out[key] = prof
    return out


def _substats_for(name: object) -> dict | None:
    return _micro_index().get(_fold_name(name))


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
        # The measured sub-stat profile behind the rating (percentiles 0-100),
        # for the card popup. None for players with no advanced data (projected).
        "substats": _substats_for(c.player),
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


def champions_pool(limit_per_position: int = 40) -> dict:
    """The cards available to draft, from ANY league, by position, best first.

    The Champions field is pan-European, so unlike the retired domestic mode the
    pool is not filtered to one competition: the best forwards on offer might be
    from LaLiga, the Bundesliga and the Eredivisie at once. Primes and icons ride
    along at the same scarcity the pack odds carry (POOL_MIX).
    """
    from mundialytics.statistical_core.squadlab.cards import cards_from_frame, load_cards

    df = load_cards()
    if df.empty:                      # catalogue not built: fall back to the model
        return _pool_from_profiles("", limit_per_position)

    by_pos: dict[str, list[dict]] = {p: [] for p in POSITIONS}
    for pos in POSITIONS:
        at_pos = df[df["position"] == pos]
        picked: list[dict] = []
        for kind, n in POOL_MIX.items():
            grp = at_pos[at_pos["kind"] == kind]
            if grp.empty:
                continue
            if kind == "actual":
                chosen = grp.nlargest(n, "overall")
            else:
                # a stable sample rather than the top n, so the two rarest card
                # types are not the same fixtures of the furniture every session
                chosen = grp.nlargest(min(len(grp), n * 6), "overall")
                chosen = _stable_sample(chosen, f"champions|{pos}|{kind}", n)
            picked += [_card_entry(c) for c in cards_from_frame(chosen)]
        by_pos[pos] = sorted(picked, key=lambda x: -x["overall"])[:limit_per_position]
    return {
        "slots": SLOTS,
        "positions": POSITIONS,
        "players": by_pos,
        # metadata for the card popup: the role formulas and readable labels
        "roleWeights": ROLE_WEIGHTS,
        "substatLabels": SUBSTAT_LABELS,
    }


@lru_cache(maxsize=1)
def _champions_resources():
    """The real CL field (elo + drawn fixtures) and the squad-rating -> Elo line.

    Cached: the draw is read off disk (a game wants a stable field), and the
    fitted line does not change between requests. The RANDOM slot the squad
    takes is drawn per playthrough inside ChampionsRun, not here.
    """
    from mundialytics.statistical_core.squadlab.cards import load_cards
    from mundialytics.statistical_core.squadlab.champions import (
        load_field, squad_elo_scale,
    )

    field = load_field()
    elo_df = pd.read_csv(ROOT / "data/processed/clubelo_local.csv").dropna(
        subset=["club", "elo"])
    scale = squad_elo_scale(
        None, load_cards(),
        dict(zip(elo_df["club"].astype(str), elo_df["elo"].astype(float))),
    )
    return field, scale


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


def play_champions(squad_names: list[str], seed: int | None = None) -> dict:
    """Play the whole Champions League once with the chosen eleven.

    The squad takes a RANDOM club's slot in the real draw (a fresh rng per call,
    so the path changes every time). Its Elo is read off the fitted line; the
    league phase and the bracket then run on the European Elo layer.
    """
    import numpy as np

    from mundialytics.statistical_core.squadlab.champions import ChampionsRun

    model = strength_model()
    squad = resolve_squad(model, squad_names)
    if len(squad) < 11:
        raise ValueError("The squad needs eleven players the model knows")

    field, scale = _champions_resources()
    squad_elo = scale.elo_for([getattr(p, "overall", 0.0) for p in squad])

    if seed is None:
        seed = random.randrange(2 ** 32)
    rng = np.random.default_rng(int(seed))
    run = ChampionsRun(SQUAD_TEAM_NAME, squad, squad_elo, field, rng=rng)
    res = run.play()
    return _shape_champions(res, run.replaced, int(seed))


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


def _shape_champions(res, replaced: str, seed: int) -> dict:
    """The Champions run in the shape the client reveals round by round.

    Only the squad's own matches carry the narrative detail (events, stats,
    ratings): they are the only ones the client plays out, and the other 35
    clubs' 144 games resolve as scorelines inside the league-phase table.
    """
    table = res.table
    league_phase = [
        {
            "rank": int(r["pos"]),
            "team": _label(str(r["team"])),
            "played": int(r["played"]),
            "points": int(r["pts"]),
            "goalsFor": int(r["gf"]),
            "goalsAgainst": int(r["ga"]),
            "goalDiff": int(r["gd"]),
            "isSquad": str(r["team"]) == SQUAD_TEAM_NAME,
        }
        for r in table.to_dict("records")
    ]

    own = [m for m in res.matches if SQUAD_TEAM_NAME in (m.home, m.away)]
    matches = [_champ_fixture(m) for m in own]

    scorers = []
    sc = res.scorers
    if sc is not None and len(sc):
        for r in sc.head(12).itertuples():
            scorers.append({
                "player": _display(str(r.jugador)),
                "goals": int(r.goles),
                "assists": int(r.asistencias),
                # only the squad's own goals are attributed to a scorer
                "isSquad": True,
            })

    return {
        "teamName": SQUAD_TEAM_NAME,
        "replaced": _label(replaced) if replaced else None,
        "squadElo": round(float(res.squad_elo), 0),
        "seed": int(seed),
        "champion": _label(res.champion),
        "runnerUp": _label(res.runner_up),
        "stage": res.squad_stage,
        "leaguePhaseRank": int(res.squad_position),
        "leaguePhase": league_phase,
        "matches": matches,
        "bracket": _champ_bracket(res.rounds),
        "scorers": scorers,
    }


def _label(team: str) -> str:
    return team if team == SQUAD_TEAM_NAME else str(team).title()


def _stage_label(stage: str, matchday: int) -> str:
    if stage == "liga":
        return f"Fase liga · J{matchday}"
    if stage == "final":
        return "Final"
    return f"{ROUND_LABELS.get(stage, stage)} · {'Ida' if matchday == 1 else 'Vuelta'}"


def _champ_fixture(m) -> dict:
    """One of the squad's own matches, in the client's fixture shape."""
    return {
        "stage": m.stage,
        "stageLabel": _stage_label(m.stage, m.matchday),
        "home": _label(m.home),
        "away": _label(m.away),
        "homeGoals": int(m.home_goals),
        "awayGoals": int(m.away_goals),
        "isSquad": True,
        "events": _champ_timeline(m),
        "stats": _champ_stats(m),
        "ratings": _champ_ratings(m),
    }


def _champ_timeline(m) -> list[dict]:
    """Goals and bookings in the order they happened.

    The squad's goals are named (scorer, assister); the opponent's are not —
    only the squad's roster is attributed, exactly as the domestic season did.
    Conditional on the score, the minute is drawn on a separate seeded stream so
    it never disturbs the simulator's own RNG.
    """
    rng = random.Random(
        f"{m.stage}|{m.matchday}|{m.home}|{m.away}|{m.home_goals}-{m.away_goals}"
    )
    squad_home = bool(m.squad_is_home)
    squad_side = "home" if squad_home else "away"
    opp_side = "away" if squad_home else "home"
    squad_goals = int(m.home_goals if squad_home else m.away_goals)
    opp_goals = int(m.away_goals if squad_home else m.home_goals)
    opp_team = m.away if squad_home else m.home

    out: list[dict] = []
    for entry in m.goal_events or []:
        pair = entry if isinstance(entry, (tuple, list)) else (entry, None)
        assist = pair[1] if len(pair) > 1 else None
        out.append({
            "type": "goal",
            "side": squad_side,
            "player": _display(str(pair[0])),
            "assist": _display(str(assist)) if assist else None,
            "minute": rng.randint(1, 90),
        })
    # any squad goal the attribution did not name still has to show, or the
    # running score would stop short of the real result
    for _ in range(max(squad_goals - len(m.goal_events or []), 0)):
        out.append({"type": "goal", "side": squad_side,
                    "player": _label(m.home if squad_home else m.away),
                    "assist": None, "minute": rng.randint(1, 90)})
    # the opponent's goals are unnamed: attribution only covers the squad
    for _ in range(opp_goals):
        out.append({"type": "goal", "side": opp_side, "player": _label(opp_team),
                    "assist": None, "minute": rng.randint(1, 90)})
    for player in m.card_players or []:
        out.append({"type": "card", "side": squad_side,
                    "player": _display(str(player)), "assist": None,
                    "minute": rng.randint(1, 90)})
    out.sort(key=lambda e: e["minute"])
    return out


# The event-lambda figures the Champions match draws, in reading order. No
# fouls: the European event calibration does not carry them.
_CHAMP_STAT_KEYS = [
    ("xg", "xg"), ("shots", "shots"), ("sot", "sot"),
    ("corners", "corners"), ("cards", "yellows"),
]


def _champ_stats(m) -> list[dict]:
    out = []
    for key, attr in _CHAMP_STAT_KEYS:
        pair = getattr(m, attr, (0, 0)) or (0, 0)
        out.append({"key": key, "home": round(float(pair[0]), 2),
                    "away": round(float(pair[1]), 2)})
    return out


def _champ_ratings(m) -> list[dict]:
    """The eleven's own player ratings for this match, best first."""
    rows = [
        {
            "player": _display(str(getattr(e, "player", ""))),
            "rating": round(float(getattr(e, "rating", 0) or 0), 1),
            "goals": int(getattr(e, "goals", 0) or 0),
            "assists": int(getattr(e, "assists", 0) or 0),
            "cards": int(getattr(e, "yellow_cards", 0) or 0),
        }
        for e in (m.ratings or {}).values()
    ]
    return sorted(rows, key=lambda r: -r["rating"])


def _champ_bracket(rounds: dict) -> dict:
    """Every knockout tie, by round, with the squad's own ties flagged."""
    out: dict[str, list[dict]] = {}
    for key in ("playoff", "r16", "qf", "sf", "final"):
        out[key] = [
            {
                "round": key,
                "roundLabel": ROUND_LABELS[key],
                "teamA": _label(t["team_a"]),
                "teamB": _label(t["team_b"]),
                "leg1": t.get("leg1", ""),
                "leg2": t.get("leg2", ""),
                "agg": t.get("agg", ""),
                "winner": _label(t["winner"]),
                "note": t.get("note", ""),
                "isSquad": SQUAD_TEAM_NAME in (t["team_a"], t["team_b"]),
            }
            for t in rounds.get(key, [])
        ]
    return out
