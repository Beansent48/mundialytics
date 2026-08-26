from __future__ import annotations

"""Historical team catalog for SquadLab's "Champions histórica".

Two sources, one common strength scale:

 1. SEASON TEAMS — (club, season) rosters from player_profiles_by_season.csv.
    ⚠️ Player QUALITY comes from the CAREER profiles, not that season's rows:
    the season file is StatsBomb's Barcelona-centric La Liga dump, where a rival
    appears in only its 1-2 games vs Barça (Real Madrid 2016/17: median 2
    matches). Rating those players on 2 matches shrinks them to mediocrity and
    made La Décima's Madrid look bottom-half. Names match 100% between the two
    files, so the season file is used for WHO was in the squad and the career
    file for HOW GOOD each player is.

 2. CURATED SQUADS — data/curated/legend_squads.csv, an explicit XI per iconic
    side. Needed because the career file gives each player exactly ONE team
    (usually a later one), so famous squads are scattered and unreachable by
    club label: Ajax 2019's XI lives under Barcelona (de Jong), Netherlands
    (de Ligt), Morocco (Ziyech) and Southampton (Tadić). Names resolve
    strictly — every token must match — because a loose surname fallback
    silently mapped "Deco" to Naby Keïta and "Júlio César" to Julio Enciso.

 3. AUTO-LEGEND TEAMS — clubs absent from the season file that still have a
    full XI under one label in the career profiles. Held to a quality floor so
    StatsBomb's Indian-Super-League bulk doesn't pad the catalog.

Every XI goes through SquadLab's OWN player->strength bridge, so all eras land
on ONE comparable scale, then gets an Elo-equivalent for the validated European
tournament simulator.

COVERAGE IMPUTATION (see _impute_axes): team_strength() reads offensive/creation/
defensive_strength, and those three axes are credibility-shrunk to ~50 for any
player StatsBomb barely covers — std 1.23 at 0-5 matches vs 8.02 at 150+. The
`overall` rating is NOT (std 4.29 vs 5.92) because its anchor curves are era-
scoped. Left raw, the catalog ranked Leicester 2016 (a full StatsBomb release)
above Bayern 2013, i.e. it measured data coverage, not quality. So thin-coverage
axes are imputed from `overall` per position instead of falling back to the flat
50. This lives HERE, not in player_strength.py: the deployed rating model is
untouched.

Output: data/processed/historical_teams.csv
"""

import dataclasses
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.player_strength import PlayerStrengthModel  # noqa: E402
from mundialytics.statistical_core.squadlab import calibration_constants as CC  # noqa: E402

OUT = ROOT / "data/processed/historical_teams.csv"
SEASON_FILE = ROOT / "data/processed/player_profiles_by_season.csv"
CURATED_FILE = ROOT / "data/curated/legend_squads.csv"
SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}
# a real historical XI need not fit 4-3-3: the 2004 Porto squad in the data has
# only 3 defenders, and requiring 4 dropped the whole team.
MIN_SLOTS = {"Goalkeeper": 1, "Defender": 3, "Midfielder": 2, "Forward": 1}
MIN_SEASON_PLAYERS = 11
AUTO_LEGEND_MIN_OVERALL = 70.0
ELO_CENTRE, ELO_B = 1780.0, 0.7394
# credibility for the coverage-imputation blend: axis = w·raw + (1-w)·imputed,
# w = matches/(matches+MATCH_CRED). At 2 matches ~88% imputed, at 100 ~87% raw.
MATCH_CRED = 15.0
IMPUTE_FIT_MIN_MATCHES = 50  # only well-covered players train the overall->axis fit
AXES = ("offensive_strength", "creation_strength", "defensive_strength")
# national teams live in the same pool; they are not clubs, keep them out
NATIONS = {
    "brazil", "argentina", "germany", "england", "france", "spain", "italy", "netherlands",
    "portugal", "mexico", "peru", "uruguay", "belgium", "croatia", "colombia", "chile",
    "denmark", "sweden", "switzerland", "poland", "japan", "south korea", "united states",
    "nigeria", "cameroon", "ghana", "senegal", "morocco", "tunisia", "egypt", "australia",
    "costa rica", "ecuador", "paraguay", "venezuela", "bolivia", "honduras", "panama",
    "iran", "saudi arabia", "qatar", "china", "russia", "ukraine", "serbia", "wales",
    "scotland", "republic of ireland", "northern ireland", "austria", "czech republic",
    "greece", "turkey", "romania", "hungary", "norway", "finland", "iceland", "slovakia",
    "slovenia", "bulgaria", "algeria", "ivory coast", "south africa", "jamaica", "canada",
    "new zealand", "north korea", "iraq", "kuwait", "uae", "bosnia and herzegovina",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", s).strip()


def _build_xi(profiles: list) -> list | None:
    """Best XI in a 4-3-3 shape from a pool of career profiles."""
    xi: list = []
    for pos, n in SLOTS.items():
        cands = sorted([p for p in profiles if p.position == pos], key=lambda p: -p.overall)[:n]
        if len(cands) < n:
            return None
        xi += cands
    return xi


def _impute_axes(prof: dict) -> None:
    """Replace coverage-shrunk off/cre/def with an overall-based estimate.

    team_strength() reads these three axes, but StatsBomb barely covers most
    players, so they collapse toward 50 (std 1.2 at 0-5 matches vs 8.0 at 150+)
    while `overall` keeps its spread. Fit overall->axis per position on the
    well-covered players and blend each player's raw axis toward that fit by a
    credibility weight. Mutates profiles in place (this catalog only; the
    deployed rating model is untouched).
    """
    players = list(prof.values())
    fits: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for pos in {p.position for p in players}:
        train = [p for p in players if p.position == pos and p.matches >= IMPUTE_FIT_MIN_MATCHES]
        if len(train) < 30:
            continue
        ov = np.array([p.overall for p in train], dtype=float)
        for ax in AXES:
            y = np.array([getattr(p, ax) for p in train], dtype=float)
            slope, intercept = np.polyfit(ov, y, 1)
            lo, hi = np.percentile(y, 2), np.percentile(y, 98)
            fits[(pos, ax)] = (slope, intercept, lo, hi)

    for p in players:
        w = p.matches / (p.matches + MATCH_CRED)
        for ax in AXES:
            f = fits.get((p.position, ax))
            if f is None:
                continue
            slope, intercept, lo, hi = f
            imputed = float(np.clip(slope * p.overall + intercept, lo, hi))
            setattr(p, ax, round(w * getattr(p, ax) + (1 - w) * imputed, 1))


def _xi_ordered(found: list) -> list | None:
    """Curated XI: honour the listed order (starters first), guarantee one GK.

    Unlike _build_xi_flexible this does NOT reorder by overall — a hand-picked
    side like Ajax 2019 must keep de Ligt (overall 70 in our data) over a
    higher-rated bench striker.
    """
    gk = next((p for p in found if p.position == "Goalkeeper"), None)
    if gk is None:
        return None
    outfield = [p for p in found if p is not gk]
    xi = [gk] + outfield[:10]
    return xi if len(xi) == 11 else None


def _build_xi_flexible(profiles: list) -> list | None:
    """Best XI without imposing 4-3-3: fill minimum quotas, then best available."""
    if len(profiles) < 11:
        return None
    by_pos: dict[str, list] = {}
    for p in profiles:
        by_pos.setdefault(p.position, []).append(p)
    for v in by_pos.values():
        v.sort(key=lambda p: -p.overall)

    xi, used = [], set()
    for pos, n in MIN_SLOTS.items():
        for p in by_pos.get(pos, [])[:n]:
            xi.append(p)
            used.add(p.player)
    if not any(p.position == "Goalkeeper" for p in xi):
        return None
    rest = sorted([p for p in profiles if p.player not in used], key=lambda p: -p.overall)
    xi += rest[: 11 - len(xi)]
    return xi if len(xi) == 11 else None


def _resolve_curated(names: list[str], prof: dict) -> tuple[list, list[str]]:
    """Strict name resolution: every token of the query must appear in the key."""
    by_token: dict[str, set] = {}
    exact: dict[str, str] = {}
    for k in prof:
        n = _norm(k)
        exact.setdefault(n, k)
        for t in n.split():
            by_token.setdefault(t, set()).add(k)

    found, missing = [], []
    for q in names:
        n = _norm(q)
        if n in exact:
            found.append(prof[exact[n]])
            continue
        cands: set | None = None
        for t in n.split():
            hits = by_token.get(t, set())
            cands = hits if cands is None else (cands & hits)
            if not cands:
                break
        if cands:
            found.append(prof[max(cands, key=lambda k: prof[k].overall)])
        else:
            missing.append(q)
    return found, missing


def _bridge(model: PlayerStrengthModel, xi: list) -> tuple[float, float, float, float]:
    st = model.team_strength(xi)
    atk_idx, def_idx = st["attack_index"], st["defense_index"]
    atk = float(np.clip(CC.GOAL_ATTACK_SLOPE * atk_idx + CC.GOAL_ATTACK_INTERCEPT,
                        *CC.ATTACK_PARAM_CLIP))
    dfn = float(np.clip(CC.GOAL_DEFENSE_SLOPE * def_idx + CC.GOAL_DEFENSE_INTERCEPT,
                        *CC.DEFENSE_PARAM_CLIP))
    return atk_idx, def_idx, atk, dfn


def _row(model: PlayerStrengthModel, team: str, ssn: str, label: str,
         kind: str, xi: list, n_pool: int) -> dict:
    atk_idx, def_idx, atk, dfn = _bridge(model, xi)
    return {
        "team": team, "season": ssn, "label": label, "kind": kind,
        "atk_idx": round(atk_idx, 2), "def_idx": round(def_idx, 2),
        "attack_param": round(atk, 4), "defense_param": round(dfn, 4),
        "strength": round((atk + dfn) / 2, 4), "n_pool": n_pool,
        "stars": ", ".join(p.player for p in sorted(xi, key=lambda x: -x.overall)[:3]),
        "avg_overall": round(float(np.mean([p.overall for p in xi])), 1),
        "xi": " | ".join(p.player for p in xi),
    }


def main() -> None:
    model = PlayerStrengthModel().fit()
    prof = model.profiles_
    print(f"career profiles: {len(prof)}")
    _impute_axes(prof)

    rows: list[dict] = []

    # ── 1. season teams (roster by season, quality from career) ───────────────
    season = pd.read_csv(SEASON_FILE)
    for (team, ssn), g in season.groupby(["team", "season"]):
        if len(g) < MIN_SEASON_PLAYERS:
            continue
        pool = [prof[p] for p in g["player"] if p in prof]
        xi = _build_xi(pool)
        if xi is not None:
            rows.append(_row(model, team, ssn, f"{team} {ssn.split('/')[0]}",
                             "season", xi, len(pool)))

    # ── 2. curated iconic squads (explicit XI, strict name resolution) ─────────
    curated_labels: set[str] = set()
    if CURATED_FILE.exists():
        cur = pd.read_csv(CURATED_FILE)
        for label, g in cur.groupby("label"):
            found, missing = _resolve_curated(list(g["player"]), prof)
            xi = _xi_ordered(found)
            status = f"{len(found)}/{len(g)}"
            if xi is None:
                print(f"  SKIP curated {label}: only {status} players resolved")
                continue
            curated_labels.add(_norm(label.rsplit(" ", 1)[0]))
            rows.append(_row(model, label.rsplit(" ", 1)[0], label.rsplit(" ", 1)[-1],
                             label, "curated", xi, len(found)))
            if missing:
                print(f"  curated {label}: {status} resolved, missing {', '.join(missing)}")

    season_clubs = {_norm(t) for t in season["team"].unique()}

    # ── 3. auto-legend teams (career-only clubs with a full XI) ───────────────
    by_club: dict[str, list] = {}
    for p in prof.values():
        by_club.setdefault(str(p.team).lower(), []).append(p)
    for club, pool in by_club.items():
        if _norm(club) in season_clubs or _norm(club) in curated_labels:
            continue
        if club in NATIONS or len(pool) < 11:
            continue
        xi = _build_xi_flexible(pool)
        if xi is None:
            continue
        if float(np.mean([p.overall for p in xi])) < AUTO_LEGEND_MIN_OVERALL:
            continue
        rows.append(_row(model, club, "leyenda", f"{club} (leyenda)",
                         "legend", xi, len(pool)))

    cat = pd.DataFrame(rows)
    cat["elo"] = (ELO_CENTRE + (400.0 / ELO_B) * (cat["strength"] - cat["strength"].mean())).round(0)
    cat = cat.sort_values("strength", ascending=False).reset_index(drop=True)
    cat.to_csv(OUT, index=False)

    counts = cat.kind.value_counts().to_dict()
    print(f"WROTE {OUT}: {len(cat)} teams {counts}, "
          f"{cat.team.nunique()} clubs, elo {cat.elo.min():.0f}-{cat.elo.max():.0f}")


if __name__ == "__main__":
    main()
