#!/usr/bin/env python3
"""Build the CURRENT-SQUAD table from ESPN.

WHY THIS EXISTS. Nothing in the project knew who plays for a club today. Both
player-facing layers inferred the squad from appearance history, and both were
wrong in the same way: the props model read it off the club's last ten Understat
games (Hull's 2026/27 attack came back as Harry Maguire and Oumar Niasse, last
seen there in 2017), SquadLab off career profiles (Barcelona lined up Xavi,
Puyol and Messi). Understat stopped serving its embedded JSON after May 2026, so
the fix could not be "refresh Understat" — the squad has to come from the one
feed that is still current.

WHERE THE TWO HALVES COME FROM. Membership and position come from ESPN's team
roster endpoint, which is the registered squad as of now: a player sold in the
window stops appearing for his old club, and a signing appears before he debuts.
Playing time and season tallies come from the match rosters already downloaded
by fetch_espn_match_events.py — the roster endpoint knows who is at the club,
only the match data knows who actually plays.

The match rosters cannot supply the position on their own: ESPN lists every
substitute as "SUB", so a squad built from them turns two thirds of the league
into defenders.

WHAT IT PRODUCES. One row per (club, player) for the season in progress.
Consumers match it back to their own historical data by name — see
mundialytics.identity.current_squads.

Run with the project venv (fetches ~96 rosters, one request per club):
    .venv/Scripts/python.exe scripts/build_current_squads.py
    .venv/Scripts/python.exe scripts/build_current_squads.py --no-fetch   # cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.providers.espn_fixtures import espn_json  # noqa: E402
from mundialytics.identity.current_squads import (  # noqa: E402
    DEFAULT_CURRENT_SQUADS_PATH, full_key, position_group, short_key,
)

DIR = ROOT / "data/external/advanced/espn"
SRC_MATCH = DIR / "espn_player_match_current.csv"
CACHE_ROSTER = DIR / "espn_team_rosters_current.csv"
ALIASES = ROOT / "data/curated/fixture_team_aliases.csv"
OUT = DEFAULT_CURRENT_SQUADS_PATH
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}"

LEAGUES = {
    "eng.1": "Premier League",
    "esp.1": "LaLiga",
    "ger.1": "Bundesliga",
    "ita.1": "Serie A",
    "fra.1": "Ligue 1",
}

# The 108 clubs drawn into Europe. Two thirds of them play outside the big five
# and had no squad at all, which is why a Champions run could only ever name
# the drafted eleven: the other 35 clubs scored anonymously.
UEFA_LEAGUES = {
    "uefa.champions": "Champions League",
    "uefa.europa": "Europa League",
    "uefa.europa.conf": "Conference League",
}
CACHE_ROSTER_UEFA = DIR / "espn_team_rosters_uefa.csv"
SRC_MATCH_UEFA = DIR / "espn_player_match_uefa.csv"

# What a player did in Europe, kept in its OWN columns. Folding it into `goals`
# or `apps` would quietly redefine those for all 96 clubs that already have
# them -- the scorer race, the props and squad_share all read them as league
# numbers -- so European totals sit beside the domestic ones, never inside.
UEFA_TALLIES = {"uefa_matches": "squad_matches", "uefa_apps": "apps",
                "uefa_goals": "goals", "uefa_assists": "assists"}

NUMERIC = ["goals", "shots", "sot", "assists", "yellow_cards", "red_cards",
           "own_goals", "appearances", "sub_ins"]


def _get(url: str, tries: int = 3) -> dict:
    """Thin alias for the shared caller: the User-Agent rule lives in one place."""
    return espn_json(url, timeout=40, tries=tries)


def _canon():
    from mundialytics.identity.normalization import canonical_team_name
    alias = {}
    if ALIASES.exists():
        a = pd.read_csv(ALIASES)
        alias = dict(zip(a.iloc[:, 0].astype(str), a.iloc[:, 1].astype(str)))

    def canon(name: str) -> str:
        return alias.get(name, canonical_team_name(name))

    return canon


def fetch_rosters(leagues: dict[str, str] | None = None) -> pd.DataFrame:
    """Registered squad + listed position for every club in `leagues`."""
    canon = _canon()
    rows: list[dict] = []
    for code, comp in (leagues or LEAGUES).items():
        try:
            teams = _get(f"{BASE.format(code=code)}/teams")["sports"][0]["leagues"][0]["teams"]
        except Exception as exc:
            print(f"  {comp:15s} FALLO listado de equipos: {type(exc).__name__}")
            continue
        n_ok = 0
        for entry in teams:
            t = entry["team"]
            try:
                r = _get(f"{BASE.format(code=code)}/teams/{t['id']}/roster")
            except Exception:
                continue
            team = canon(r.get("team", {}).get("displayName") or t["displayName"])
            for a in r.get("athletes", []) or []:
                who = a.get("displayName")
                if not who:
                    continue
                rows.append({
                    "competition": comp, "team": team, "player": who,
                    "espn_athlete_id": str(a.get("id") or ""),
                    "position": ((a.get("position") or {}).get("abbreviation") or ""),
                    "jersey": a.get("jersey"),
                    "age": a.get("age"),
                })
            n_ok += 1
            time.sleep(0.2)
        print(f"  {comp:15s} {n_ok}/{len(teams)} plantillas")
    return pd.DataFrame(rows)


def match_tallies(src: Path) -> pd.DataFrame:
    """Minutes played and season tallies, from the stored match rosters."""
    df = pd.read_csv(src, low_memory=False)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)
    df["starter"] = df.get("starter", False).astype(bool)
    df["played"] = (df["appearances"] > 0) | df["starter"] | df.get("subbed_in", False).astype(bool)

    team_matches = df.groupby("team")["event_id"].nunique().rename("team_matches")
    g = df.groupby(["team", "player"], as_index=False).agg(
        season=("season", "last"),
        competition=("competition", "last"),
        squad_matches=("event_id", "nunique"),
        apps=("played", "sum"),
        starts=("starter", "sum"),
        goals=("goals", "sum"),
        shots=("shots", "sum"),
        sot=("sot", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        last_date=("date", "max"),
    )
    return g.merge(team_matches, on="team", how="left")


POS_GROUPS = {"Goalkeeper", "Defender", "Midfielder", "Forward"}


def match_positions(src: Path) -> dict[str, str]:
    """Who actually kept goal, from the MATCH rosters.

    The team-roster endpoint is the registered squad and it is occasionally
    simply wrong about a position: it files Paris FC's Kevin Trapp — a
    goalkeeper who has started in goal — under "D", and the card came out a
    defender rated 87.6. The match rosters say "G" for the matches he played,
    which is not an opinion about his position but a record of where he stood.

    Returns the most common position group per player. The caller arbitrates
    only the KEEPER boundary with it: forward-versus-midfielder disagrees
    between the two feeds for ~100 players and neither is wrong, while a keeper
    filed outfield — or a midfielder filed in goal — is simply wrong, and it is
    the one error that ruins a squad.
    """
    if not src.exists():
        return {}
    df = pd.read_csv(src, usecols=lambda c: c in {"player", "position"}, low_memory=False)
    # A file written by a fetcher that did not record positions (every match
    # from 2026-09-07 to 09-21, and any file rebuilt in that window) has no
    # such column. That means "no evidence", not a crash in the squad build.
    if "position" not in df.columns:
        return {}
    df["pg"] = df["position"].fillna("").map(position_group)
    # "SUB" maps to "Unknown", not to the empty string, and an unused backup
    # keeper is SUB in every match he is named in — left in, "Unknown" wins his
    # vote and 143 real goalkeepers lose the label.
    df = df[df["pg"].isin(POS_GROUPS)]
    if df.empty:
        return {}
    top = df.groupby("player")["pg"].agg(lambda s: s.value_counts().idxmax())
    return {str(k): str(v) for k, v in top.items()}


def build(rosters: pd.DataFrame, tallies: pd.DataFrame,
          played: dict[str, str] | None = None) -> pd.DataFrame:
    """Registered squads, carrying whatever each player has done this season.

    A club whose roster fetch failed falls back to the players seen in its
    matches, so one bad request costs positions for that club, never the club.
    """
    season = str(tallies["season"].dropna().iloc[0]) if len(tallies["season"].dropna()) else ""
    if rosters.empty:
        out = tallies.copy()
        out["position"] = ""
    else:
        covered = set(rosters["team"])
        missing = tallies[~tallies["team"].isin(covered)]
        rcols = ["competition", "team", "player", "position"]
        for c in ("espn_athlete_id", "age"):     # carry ESPN's stable id + age
            if c in rosters.columns:
                rcols.append(c)
        base = rosters[rcols].copy()
        if not missing.empty:
            print(f"  sin plantilla ESPN, se usan los partidos: {', '.join(sorted(set(missing['team'])))}")
            base = pd.concat([base, missing[["competition", "team", "player"]].assign(position="")],
                             ignore_index=True)
        out = base.merge(
            tallies.drop(columns=["competition"]), on=["team", "player"], how="left")

    # a squad member who has not featured yet is still a squad member
    for c in ["squad_matches", "apps", "starts", "goals", "shots", "sot",
              "assists", "yellow_cards", "red_cards"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0)
    out["team_matches"] = out.groupby("team")["team_matches"].transform(
        lambda s: s.fillna(s.max()))
    out["team_matches"] = pd.to_numeric(out["team_matches"], errors="coerce").fillna(0.0)
    out["season"] = out.get("season").fillna(season) if "season" in out.columns else season
    out["last_date"] = out.get("last_date", pd.Series(index=out.index, dtype=object))

    out["squad_share"] = (out["squad_matches"] / out["team_matches"].clip(lower=1)).round(3)
    out["pos_group"] = out["position"].fillna("").map(position_group)
    if played:
        seen = out["player"].map(played).fillna("")
        # only the keeper boundary is arbitrated: forward-versus-midfielder
        # disagrees for ~100 players and neither feed is wrong there
        gk_now = out["pos_group"] == "Goalkeeper"
        gk_played = seen == "Goalkeeper"
        wrong = (seen != "") & (gk_now != gk_played)
        for who, was, now in zip(out.loc[wrong, "player"], out.loc[wrong, "pos_group"],
                                 seen[wrong]):
            print(f"  posicion corregida por los partidos: {who} ({was or 'sin posicion'} -> {now})")
        out.loc[wrong, "pos_group"] = seen[wrong]
    out["full_key"] = out["player"].map(full_key)
    out["short_key"] = out["player"].map(short_key)
    # A stable identity for the player, so a name collision or a club move never
    # confuses him with someone else. ESPN's athlete id is unique on the squad
    # side; birth year (from his age at the fetch) is invariant across clubs and
    # matches the year the stats sources carry — together they let the card
    # builder pick the right one of two same-named players. See NameIndex.
    if "espn_athlete_id" not in out.columns:
        out["espn_athlete_id"] = ""
    out["espn_athlete_id"] = out["espn_athlete_id"].fillna("").astype(str)
    start_year = int(str(season)[:4]) if str(season)[:4].isdigit() else 0
    age = pd.to_numeric(out.get("age"), errors="coerce")
    out["birth_year"] = (start_year - age).round() if start_year else pd.NA

    cols = ["season", "competition", "team", "player", "full_key", "short_key",
            "espn_athlete_id", "birth_year",
            "position", "pos_group", "team_matches", "squad_matches", "squad_share",
            "apps", "starts", "goals", "shots", "sot", "assists",
            "yellow_cards", "red_cards", "last_date"]
    out = out.drop_duplicates(subset=["team", "player"])
    return out[cols].sort_values(["team", "pos_group", "player"]).reset_index(drop=True)


COLUMNS_EURO = ["uefa_competition", "uefa_matches", "uefa_apps", "uefa_goals",
                "uefa_assists"]



def add_european(out: pd.DataFrame, euro_rosters: pd.DataFrame,
                 euro_src: Path) -> pd.DataFrame:
    """Mark who plays in Europe, and what they have done there.

    The domestic half is left exactly as it was. A club in the big five keeps
    its league competition and its league tallies; the European ones only ever
    ADD columns. Clubs that play no big-five football arrive as new rows with
    their UEFA competition and zeroed domestic numbers, which is not a gap but
    the truth: Sabah have not scored in LaLiga.
    """
    for col in UEFA_TALLIES:
        out[col] = 0.0
    out["uefa_competition"] = ""
    if euro_rosters.empty:
        return out

    euro = euro_rosters.drop_duplicates(subset=["team", "player"])
    comp = dict(zip(zip(euro["team"], euro["player"]), euro["competition"]))
    out["uefa_competition"] = [
        comp.get((t, pl), "") for t, pl in zip(out["team"], out["player"])
    ]

    if euro_src.exists():
        tal = match_tallies(euro_src)
        keep = ["team", "player"] + list(UEFA_TALLIES.values())
        tal = tal[keep].rename(columns={v: k for k, v in UEFA_TALLIES.items()})
        out = out.drop(columns=list(UEFA_TALLIES)).merge(
            tal, on=["team", "player"], how="left")
        for col in UEFA_TALLIES:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SRC_MATCH))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-fetch", action="store_true",
                    help="usa el volcado de plantillas ya descargado")
    ap.add_argument("--no-europe", action="store_true",
                    help="solo big-5, sin las plantillas de las competiciones UEFA")
    ap.add_argument("--europe-source", default=str(SRC_MATCH_UEFA))
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(
            f"falta {src}\nEjecuta antes: .venv/Scripts/python.exe scripts/fetch_espn_match_events.py")

    def load_half(leagues, cache: Path, label: str) -> pd.DataFrame:
        if args.no_fetch:
            got = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
            if got.empty:
                print(f"AVISO: sin cache de plantillas ({label})")
            return got
        print(f"ESPN | plantillas actuales ({label})")
        got = fetch_rosters(leagues)
        if not got.empty:
            cache.parent.mkdir(parents=True, exist_ok=True)
            got.to_csv(cache, index=False)
        elif cache.exists():
            print("AVISO: descarga vacia, se reutiliza la cache anterior")
            got = pd.read_csv(cache)
        return got

    rosters = load_half(LEAGUES, CACHE_ROSTER, "big-5")
    euro_rosters = pd.DataFrame()
    if not args.no_europe:
        euro_rosters = load_half(UEFA_LEAGUES, CACHE_ROSTER_UEFA, "UEFA")

    # A club in the big five keeps its league as its competition; only clubs
    # with no domestic coverage are filed under the tournament they were drawn
    # into. Getting this the other way round would take Vinicius out of every
    # path keyed on LaLiga.
    membership = rosters
    if not euro_rosters.empty:
        membership = pd.concat([rosters, euro_rosters], ignore_index=True)
        membership = membership.drop_duplicates(subset=["team", "player"], keep="first")

    out = build(membership, match_tallies(src), match_positions(src))
    if out.empty:
        raise SystemExit("sin filas: ESPN no ha devuelto plantillas")
    out = add_european(out, euro_rosters, Path(args.europe_source))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    unknown = int((out["pos_group"] == "Unknown").sum())
    print(f"\nESCRITO {args.out}")
    print(f"  {len(out):,} jugador-club | {out['team'].nunique()} clubes | "
          f"temporada {out['season'].iloc[0]}")
    if "uefa_competition" in out.columns:
        eu = out[out["uefa_competition"] != ""]
        print(f"  en Europa: {eu['team'].nunique()} clubes, {len(eu):,} jugadores | "
              + " · ".join(f"{k} {v}" for k, v in
                           eu.groupby('uefa_competition')['team'].nunique().items()))
    print("  " + " · ".join(f"{k} {v}" for k, v in out['pos_group'].value_counts().items()))
    if unknown:
        print(f"  posicion desconocida: {unknown} ({unknown / len(out):.1%})")
    thin = out.groupby("team").size()
    thin = thin[thin < 14]
    if len(thin):
        print(f"  clubes con plantilla corta (<14): {', '.join(thin.index)}")


if __name__ == "__main__":
    main()
