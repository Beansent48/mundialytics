"""The SquadLab card catalogue and the Champions draft, pinned where they rot.

A draft game fails silently in ways a model does not. A pool missing goalkeepers
still deals cards; a rarity table that never reaches its rarest tier still runs;
a Champions field one club short still plays a tournament, just not the right
one. Each of those looks like a working game right up to the moment someone
reads the screen carefully — which is exactly what happened to the squads these
cards replaced.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mundialytics.statistical_core.squadlab import cards as C

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_squadlab_cards as C_BUILD  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "data/processed/squadlab_cards.csv"
_CL_CSV = ROOT / "data/external/uefa/raw_champions-league_2026.csv"
_needs_cl = pytest.mark.skipif(
    not _CL_CSV.exists(), reason="Champions League CSV not in repo (gitignored)"
)


def _cards() -> pd.DataFrame:
    df = C.load_cards()
    if df.empty:
        pytest.skip("squadlab_cards.csv not built")
    return df


# ── the catalogue ──────────────────────────────────────────────────────────────
def test_every_club_can_field_an_eleven():
    """The Elo scale is fitted on every club's own best eleven, so a club that
    cannot produce one silently drops out of the calibration."""
    df = _cards()
    short = {club: len(C.best_eleven(df, club))
             for club in df.loc[df["kind"] == "actual", "club"].unique()}
    bad = {k: v for k, v in short.items() if v < 11}
    assert not bad, f"clubes que no llegan a once: {bad}"


def test_the_pool_is_deep_enough_at_every_position():
    """The draw deals five candidates per slot and refuses to repeat a player,
    so a thin position turns into the same faces every draft."""
    df = _cards()
    counts = df[df["kind"] == "actual"]["position"].value_counts()
    for pos, minimum in (("Goalkeeper", 120), ("Defender", 300),
                         ("Midfielder", 300), ("Forward", 250)):
        assert counts.get(pos, 0) >= minimum, f"{pos}: {counts.get(pos, 0)}"


def test_the_three_kinds_exist_and_are_ordered():
    """Rarity has to buy something. A prime that rates below an ordinary card
    is a worse prize than the common it replaced."""
    df = _cards()
    means = df.groupby("kind")["overall"].mean()
    assert set(means.index) == {"actual", "prime", "icono"}
    assert means["actual"] < means["prime"] < means["icono"]


def test_primes_carry_a_year_and_icons_do_not():
    """The year IS the claim a prime makes; an icon's claim is the opposite."""
    df = _cards()
    prime = df[df["kind"] == "prime"]
    icon = df[df["kind"] == "icono"]
    assert (prime["season_label"].astype(str).str.len() > 0).all()
    assert (icon["season_label"].fillna("").astype(str).str.len() == 0).all()


def test_no_icon_is_still_playing():
    """An icon who turns up in a current squad is not an icon, he is a player
    with two cards — and the pool would deal both."""
    from mundialytics.identity.current_squads import NameIndex, load_current_squads

    df = _cards()
    squads = load_current_squads()
    if squads.empty:
        pytest.skip("no current squads")
    idx = NameIndex()
    for t, p in zip(squads["team"], squads["player"]):
        idx.add(p, t)
    active = [(r.player, idx.lookup(r.player))
              for r in df[df["kind"] == "icono"].itertuples()
              if idx.lookup(r.player)]
    assert not active, f"iconos en activo: {active}"


def test_an_axis_is_not_a_restatement_of_the_level():
    """An axis must describe the player, not repeat how good he is.

    It used to be `level + ROLE_OFFSET[axis] + a nudge`, so every axis moved
    with the level: a forward's DEFENCE correlated +0.946 with his own overall,
    which is not a defensive measurement, it is the rating wearing a different
    label. The icons, whose axes are set by hand, sit near +0.19 — and that is
    why their cards read right and these did not. The axes now come from their
    own measurement (see AXIS_SCALE).
    """
    df = _cards()
    a = df[df["kind"] == "actual"]
    fw = a[a["position"] == "Forward"]
    r = float(fw["overall"].corr(fw["defense"]))
    assert abs(r) < 0.45, f"la defensa de un delantero copia su media (r={r:+.3f})"
    mid = a[a["position"] == "Midfielder"]
    assert abs(float(mid["overall"].corr(mid["defense"]))) < 0.45


def test_the_overall_agrees_with_its_own_axes():
    """A card is one object: the role weights the axes and the overall is what
    that weighting comes to. The two used to be computed by independent routes
    and could contradict each other on the same card — Haaland read 92.1 overall
    beside 84.2 in attack."""
    df = _cards()
    a = df[(df["kind"] == "actual") & (df["position"] != "Goalkeeper")]
    got = [(o, C_BUILD.overall_from_axes(str(rl), str(p),
                                         {"attack": at, "defense": de, "creation": cr}))
           for o, rl, p, at, de, cr in zip(a["overall"], a["role"], a["position"],
                                           a["attack"], a["defense"], a["creation"])]
    got = [(o, v) for o, v in got if v is not None]
    assert len(got) > 500
    ovr = pd.Series([o for o, _ in got])
    axw = pd.Series([v for _, v in got])
    assert float(ovr.corr(axw)) > 0.6, "la media no se parece a sus propios ejes"


def test_an_ordinary_card_never_outranks_a_rare_one():
    """Primes and icons are the scarce families and must sit above ACTUAL. With
    the level bump grown, nine actual cards passed the best prime and 62 the
    median icon before the ceiling was brought down to 94."""
    df = _cards()
    top = df.groupby("kind")["overall"].max()
    if not {"actual", "prime", "icono"} <= set(top.index):
        pytest.skip("catalogue missing a card family")
    assert top["actual"] < top["prime"] <= top["icono"], top.to_dict()
    assert top["actual"] <= C_BUILD.LEVEL_MAX


def test_a_striker_does_not_defend_like_a_centre_back():
    """The first pass gave elite forwards 80+ in defence, because the personal
    term was measured against the position median instead of against what the
    rating already implied."""
    df = _cards()
    fw = df[(df["position"] == "Forward") & (df["overall"] >= 88)]
    gap = (fw["overall"] - fw["defense"])
    assert gap.min() >= 15, fw.loc[gap.idxmin(), ["display", "overall", "defense"]].to_dict()
    assert df[df["position"] == "Forward"]["defense"].median() < 55


def test_the_level_is_not_one_number_repeated():
    """Over half of `player_ratings_roles.csv` sits exactly on the bottom anchor
    of its display curve — `np.interp` holds flat below its first point, so
    everyone under it lands on the same value. Nick Pope with 496 matches and
    ter Stegen with 460 both read 66.0. A catalogue that inherits that is
    ranking by nothing, so a floor-pinned rating is treated as missing."""
    df = _cards()
    a = df[df["kind"] == "actual"]
    top = a["overall"].round(1).value_counts()
    share = float(top.iloc[0]) / len(a)
    assert share < 0.05, f"{share:.1%} de las cartas comparten la media {top.index[0]}"


def test_the_level_has_no_ceiling_plateau_either():
    """The attacking term's first run pushed thirteen cards onto exactly 96.0 —
    the same flattening the axes had, one layer up. The level gets the same soft
    ceiling."""
    df = _cards()
    a = df[df["kind"] == "actual"]
    assert int((a["overall"] >= C_BUILD.LEVEL_MAX).sum()) <= 2,         a.loc[a["overall"] >= C_BUILD.LEVEL_MAX, ["display", "overall"]].to_string()
    # relative to the ceiling, not an absolute number: the ceiling is a design
    # dial (96 -> 99 -> 94 in one week) and a hard 90 here just breaks whenever
    # it moves. What matters is that the best cards approach it.
    assert a["overall"].max() >= C_BUILD.LEVEL_MAX - 6, "el techo aprieta demasiado"


def test_a_floor_pinned_rating_is_recognised_as_absent():
    df = _cards()
    assert C_BUILD.is_floor_pinned("Killer", 66.0)
    assert C_BUILD.is_floor_pinned("Central stopper", 64.0)
    assert not C_BUILD.is_floor_pinned("Killer", 66.4)
    assert not C_BUILD.is_floor_pinned("Killer", 64.0)   # not HIS curve's floor
    assert len(df) > 0


def test_no_forward_defends_above_the_cap():
    """A stated rule, not a measurement: nobody who plays up front is printed
    above 65 in defending.

    Without it the axis read a forward's greatness as tackling — it is anchored
    on the rating, so a 97 Messi came out at 74 in defence purely for being a
    97. The cap binds on about a dozen cards and every one of them is an icon
    or a prime.
    """
    df = _cards()
    fw = df[df["position"] == "Forward"]
    over = fw[fw["defense"] > C_BUILD.POSITION_DEF_CAP["Forward"] + 1e-6]
    assert over.empty, over[["display", "kind", "overall", "defense"]].to_string()


def test_the_attacking_axis_is_measured_too():
    """The attacking half of the FBref import, and the half that could actually
    be validated: its weights were fitted on 23/24 features against 24/25 goals
    + assists per 90, out of sample. Like the defensive axis it has to SAY
    something — a measured attack must spread wider than the role baseline."""
    df = _cards()
    if "att_source" not in df.columns:
        pytest.skip("catalogue predates the FBref attacking source")
    a = df[(df["kind"] == "actual") & (df["position"].isin(["Forward", "Midfielder"]))]
    measured = (a["att_source"] == "fbref")
    assert measured.mean() >= 0.45, f"solo {measured.mean():.0%} de atacantes medidos"
    assert a.loc[measured, "attack"].std() > a.loc[~measured, "attack"].std()


def test_a_prime_is_not_scored_on_a_season_it_did_not_play():
    """FBref reaches back to 2023/24 here (and 2025/26 via the scouting merge),
    and a prime card is one named past season. Suárez 15/16 must not be measured
    on Suárez in 2024/25 — that is the retired-goalscorer bug wearing a
    different hat."""
    df = _cards()
    if "att_source" not in df.columns:
        pytest.skip("catalogue predates the FBref attacking source")
    measured_seasons = ["23/24", "24/25", "25/26"]
    pri = df[(df["kind"] == "prime") & (~df["season_label"].isin(measured_seasons))]
    for col in ("att_source", "def_source"):
        bad = pri[pri[col] == "fbref"]
        assert bad.empty, bad[["display", "season_label", col]].to_string()


def test_the_best_defenders_break_ninety():
    """They could not before: two thirds of defenders have no defensive sample
    at all, so the measured axis capped the whole position at 75."""
    df = _cards()
    best = df[df["position"] == "Defender"].nlargest(8, "overall")
    assert best["defense"].max() >= 90, best[["display", "overall", "defense"]].to_string()
    assert (best["defense"] >= best["overall"] - 6).all()


def test_ninety_nine_is_almost_unreachable():
    """A 99 that half the strikers have is not a 99.

    The role offsets alone put an elite Killer at 98 before his own residual is
    added, so without a ceiling the whole top of the game flattened onto one
    number. Two or three cards in the catalogue may touch it; a dozen may not.
    """
    df = _cards()
    peak = df[["attack", "defense", "creation", "gk"]].max(axis=1)
    assert int((peak >= 99).sum()) <= 3, df.loc[peak >= 99, ["display", "overall"]].to_string()
    assert int((peak >= 96).sum()) <= 25
    assert peak.max() >= 93, "nadie llega arriba: el techo aprieta demasiado"


def test_defenders_and_keepers_get_special_cards_too():
    """The per-season ratings have no keeper rows at all and, for defenders,
    only StatsBomb's 2015/16 — which is why the one Van Dijk season on record is
    at Southampton. Those cards are authored; this pins that they exist."""
    df = _cards()
    special = df[df["kind"] != "actual"]
    for pos, minimum in (("Defender", 30), ("Goalkeeper", 8)):
        n = int((special["position"] == pos).sum())
        assert n >= minimum, f"{pos}: solo {n} cartas especiales"
    names = set(special["display"])
    for who in ("van Dijk", "Cannavaro", "Yashin"):
        assert any(who in n for n in names), f"falta {who}"


def test_the_engine_axes_stay_on_the_measured_scale():
    """The face is rebuilt for the reader; the simulator must keep the scale
    its scorer weights and its lambda bridge were fitted on."""
    df = _cards()
    assert df["raw_attack"].median() < 60 < df["attack"].max()
    fw = df[df["position"] == "Forward"]
    assert fw["raw_attack"].median() < fw["attack"].median() - 10


def test_the_defensive_axis_is_mostly_measured_now():
    """StatsBomb measured 17% of defenders defending. FBref's published season
    tables — free, and covering the Championship, Segunda and 2. Bundesliga
    where the promoted clubs' players come from — take it past half."""
    df = _cards()
    if "def_source" not in df.columns:
        pytest.skip("catalogue predates the FBref defensive source")
    d = df[(df["kind"] == "actual") & (df["position"] == "Defender")]
    measured = (d["def_source"] == "fbref")
    assert measured.mean() >= 0.5, f"solo {measured.mean():.0%} de defensas medidos"
    # and it has to SAY something: the measured axis must carry a real spread,
    # not sit on a placeholder. It is percentile-mapped now (a controlled range),
    # so the old 'measured spreads wider than role-implied' no longer holds —
    # the role fallback inherits the wide overall — but a std this size and a
    # clear gap between the best measured defenders and the median both confirm
    # the measurement is doing the ordering.
    md = d.loc[measured, "defense"]
    assert md.std() > 4.0, f"defensa medida casi plana (std {md.std():.1f})"
    assert md.quantile(0.9) - md.median() > 3.0


def test_an_unmeasured_defender_falls_back_to_his_role():
    """No defensive sample must mean "what his role implies", not a silent
    penalty for the neutral 0.50 placeholder sitting in the data."""
    df = _cards()
    # "no measurement" is now def_source, not n_def: StatsBomb having nothing on
    # a player no longer means nobody does — FBref usually does.
    src = df["def_source"] if "def_source" in df.columns else pd.Series("rol", index=df.index)
    unmeasured = df[(df["position"] == "Defender") & (src == "rol")
                    & (df["n_def"] == 0) & (df["kind"] == "actual")]
    if unmeasured.empty:
        pytest.skip("every defender has a defensive sample")
    # the axis must be exactly "rating plus what the role implies" — same offset
    # for everyone in that role, because there is nothing to tell them apart
    gap = unmeasured["defense"] - unmeasured["overall"]
    for role, grp in unmeasured.groupby("role"):
        if len(grp) < 5:
            continue
        spread = float((grp["defense"] - grp["overall"]).std())
        assert spread < 0.2, f"{role}: la defensa varía {spread:.2f} sin medición"
    assert gap.between(-9, 2).all(), unmeasured.loc[gap.abs().idxmax(),
                                                    ["display", "role", "overall", "defense"]].to_dict()


# ── the draw ───────────────────────────────────────────────────────────────────
def test_kind_probabilities_sum_to_one():
    assert abs(sum(C.KIND_P.values()) - 1.0) < 1e-9
    for kind, table in C.TIER_P.items():
        assert abs(sum(table.values()) - 1.0) < 1e-9, kind


def test_a_draft_always_fills_every_slot():
    """A roll landing on an empty bucket must fall back, not return fewer
    candidates: an empty slot reads as a bug, not as bad luck."""
    df = _cards()
    rng = np.random.default_rng(4)
    pool = C.DraftPool(df, rng)
    for pos, n in C.FORMATION_SLOTS["4-3-3"].items():
        for _ in range(n):
            cands = pool.deal(pos, 5)
            assert len(cands) == 5, f"{pos} solo dio {len(cands)} candidatos"
            assert all(c.position == pos for c in cands)
            pool.take(cands[0])


def test_a_player_cannot_be_drafted_twice_in_two_versions():
    """Drafting Messi must remove his icon, his prime and his current card at
    once, or an eleven can field three of him."""
    df = _cards()
    pool = C.DraftPool(df, np.random.default_rng(1))
    icon = df[df["kind"] == "icono"].iloc[0]
    card = C.cards_from_frame(df[df["card_id"] == icon["card_id"]])[0]
    pool.take(card)
    same = df[df["player"] == icon["player"]]
    for other in C.cards_from_frame(same):
        assert pool.is_taken(other)


def test_the_rarest_tier_is_actually_rare_but_reachable():
    """Both halves matter: unreachable is not a chase, common is not a prize."""
    df = _cards()
    rng = np.random.default_rng(7)
    kinds = []
    for _ in range(40):
        pool = C.DraftPool(df, rng)
        for pos, n in C.FORMATION_SLOTS["4-3-3"].items():
            for _ in range(n):
                kinds += [c.kind for c in pool.deal(pos, 5)]
    share = pd.Series(kinds).value_counts(normalize=True)
    assert 0.005 < share.get("icono", 0) < 0.06, share.to_dict()
    assert 0.04 < share.get("prime", 0) < 0.20, share.to_dict()


def test_pack_luck_is_centred_on_one():
    """An average draft has to score ~1.0 or the number says nothing."""
    df = _cards()
    rng = np.random.default_rng(3)
    ratios = []
    for _ in range(30):
        pool = C.DraftPool(df, rng)
        xi = []
        for pos, n in C.FORMATION_SLOTS["4-3-3"].items():
            for _ in range(n):
                c = pool.deal(pos, 5)[0]      # take the first, i.e. no choosing
                pool.take(c)
                xi.append(c)
        ratios.append(C.pack_luck(xi)["ratio"])
    assert 0.7 < float(np.mean(ratios)) < 1.4, float(np.mean(ratios))


# ── formations ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("formation", list(C.FORMATION_SLOTS))
def test_every_formation_is_eleven_players(formation):
    slots = C.FORMATION_SLOTS[formation]
    assert sum(slots.values()) == 11
    assert slots["Goalkeeper"] == 1
    outfield = "-".join(str(slots[p]) for p in ("Defender", "Midfielder", "Forward"))
    assert outfield == formation


def test_the_pitch_has_a_spot_for_every_slot():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    import squadlab_draft as D

    for formation, slots in C.FORMATION_SLOTS.items():
        coords = D.FORMATION_COORDS[formation]
        for pos, n in slots.items():
            assert len(coords[pos]) == n, f"{formation} {pos}"


# ── the Champions field ────────────────────────────────────────────────────────
@_needs_cl
def test_the_champions_field_is_thirty_six_teams_of_eight_games():
    """35 teams is not a Champions League: the league phase and the whole
    bracket are built on the field being exactly 36."""
    from mundialytics.statistical_core.squadlab.champions import load_field

    field = load_field()
    assert len(field["elo"]) == 36, sorted(field["elo"])
    played = pd.concat([field["fixtures"]["home"], field["fixtures"]["away"]]).value_counts()
    assert set(played.unique()) == {8}, played[played != 8].to_dict()


@_needs_cl
def test_the_squad_replaces_the_weakest_side_and_inherits_its_draw():
    from mundialytics.statistical_core.squadlab.champions import (
        ChampionsRun, load_field, weakest_team,
    )

    df = _cards()
    field = load_field()
    weakest = weakest_team(field["elo"])
    xi = [c.to_profile() for c in C.best_eleven(df, "real madrid")]
    run = ChampionsRun("Tu Equipo", xi, 1800.0, field, rng=np.random.default_rng(2))
    assert run.replaced == weakest
    assert weakest not in run.elo and "Tu Equipo" in run.elo
    mine = run.fixtures[(run.fixtures["home"] == "Tu Equipo")
                        | (run.fixtures["away"] == "Tu Equipo")]
    assert len(mine) == 8


@_needs_cl
def test_a_full_run_produces_a_champion_and_your_own_matches():
    from mundialytics.statistical_core.squadlab.champions import ChampionsRun, load_field

    df = _cards()
    field = load_field()
    xi = [c.to_profile() for c in C.best_eleven(df, "bayern munich")]
    res = ChampionsRun("Tu Equipo", xi, 1900.0, field,
                       rng=np.random.default_rng(6)).play()
    assert len(res.table) == 36
    assert (res.table["played"] == 8).all()
    assert res.champion and res.runner_up and res.champion != res.runner_up
    own = [m for m in res.matches
           if m.stage == "liga" and "Tu Equipo" in (m.home, m.away)]
    assert len(own) == 8
    # goals scored by your side must all be attributed to one of your players
    names = {p.player for p in xi}
    for m in own:
        gf = m.home_goals if m.home == "Tu Equipo" else m.away_goals
        assert len(m.goal_events) == gf
        assert all(s in names for s, _ in m.goal_events)


def test_squad_elo_is_monotone_in_squad_quality():
    """A better eleven must never come out weaker — the two-term fit did
    exactly that, and it is the one property a game cannot get wrong."""
    from mundialytics.statistical_core.squadlab.champions import squad_elo_scale

    df = _cards()
    elo = pd.read_csv(ROOT / "data/processed/clubelo_local.csv").dropna(subset=["club", "elo"])
    scale = squad_elo_scale(None, df, dict(zip(elo["club"], elo["elo"])))
    prev = -1e9
    for level in (60, 66, 72, 78, 84, 90):
        e = scale.elo_for([level] * 11)
        assert e >= prev, f"{level} bajó el Elo"
        prev = e
    assert scale.elo_for([92] * 11) > scale.elo_for([75] * 11)
