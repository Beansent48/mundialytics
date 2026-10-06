"""Do a player's European appearances improve his NEXT LEAGUE match?

THE GAP. `espn_player_match_uefa.csv` is downloaded every day by update_season
step 7d2 and nothing in the player model reads it: only
`espn_player_match_current.csv`, the big-five file, reaches the player state. So
a striker's Champions League hat-trick on Tuesday does not move his Saturday
price, even though PR #19 established that a stale player state is expensive.

WHY IT MIGHT STILL BE WRONG. The team-level version of this idea was measured on
2026-10-05 and came out NEGATIVE (`wf_european_form.py`): European matches push
league matches out of the recency window and replace them with a different
competition against an opponent nobody modelled, and the harm was worst exactly
where the signal should have been strongest. The player model has the same
shape -- a last-15 window over squad rows -- so the same mechanism can bite.
Hence this is measured before anything is wired, not after.

THE TEST. For every league START by a player, predict his shots in it from his
previous starts:

    league    the last N league starts            (production today)
    all       the last N starts in any competition (league + European)

Both shrink to the global mean with K starts, so the arms differ only in which
matches fill the window. Scored on MAE and Poisson deviance against the shots he
actually took, per season.

THE CONFOUND, AND WHAT IS DONE ABOUT IT. League rows come from Understat and
European rows from ESPN, two providers that need not count a shot the same way,
and there is no overlapping match to calibrate on: Understat stopped in May 2026
and ESPN's league file starts in 2026/27. So European shot counts are NORMALISED
-- scaled by one global factor so their mean per start equals the league mean
per start -- which removes any level difference, provider and competition
together, and leaves only the question worth asking: does what this player did
in Europe say anything about this player? `--raw` skips the normalisation, and
the two should agree; if they do not, the result is a level artefact.

Restricting to STARTS is the other control: Understat has real minutes and ESPN
has none, so per-90 rates are not comparable across the two, while a start is a
start.

    python scripts/european_form/wf_player_uefa_rows.py [--window 10] [--raw]
        [--uefa <csv>]   European rows to read instead of the production file,
                         which holds the current season only

RESULT 2026-10-06: NOT DEMONSTRATED. Nothing was wired.

Two European seasons were backfilled from ESPN to run this (2024/25, 2025/26;
2026/27 has one matchday). Deltas against the league-only arm, negative better:

                              dMAE all   ddev all  dMAE half  ddev half
    all league starts          -0.0001    +0.0000    +0.0002    +0.0005
    a European match in window -0.0023    +0.0001    +0.0038    +0.0116
    2+ European in window      -0.0018    +0.0020    +0.0050    +0.0150

The full-weight arm pulls the central estimate slightly closer (MAE) and leaves
the distribution fit flat to worse (deviance), winning 0-1 of 2 testable
seasons. The half-weight arm is clearly worse on both -- which is itself a
reason not to ship: the result is very sensitive to a weight nobody can set from
first principles, and in the opposite direction to the team-level version, where
damping helped.

Two testable seasons cannot meet the usual 5-fold bar in any case. Deciding it
would mean backfilling three more European seasons (~1h of fetching each); on an
effect this small and this inconsistent, that was not judged worth it.

An honest caveat about what the positive MAE sliver even measures: a European
start is more RECENT than the league start it displaces in the window, so part
of any gain is recency rather than European information. Separating them needs a
control arm that keeps the recency and removes the new observation.

AND IT COST SOMETHING. The backfill wrote into the production
espn_player_match_uefa.csv, the 06:30 job rebuilt the squads over it, and
uefa_apps went from 1 to as many as 35 for European players. Files were trimmed
back and the squads rebuilt; build_current_squads.match_tallies now ignores older
seasons so it cannot happen again. Use --uefa to point at a copy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

UNDERSTAT = ROOT / "data/external/advanced/understat/understat_player_match.csv"
TEAM_XG = ROOT / "data/processed/understat_team_match_xg.csv"
UEFA = ROOT / "data/external/advanced/espn/espn_player_match_uefa.csv"
SHRINK_K = 4.0          # starts of prior weight on the global mean
MIN_PRIOR = 3           # a window needs this many starts before it is scored


def _season(d: pd.Series) -> pd.Series:
    """Aug-Jul football season label from a date."""
    y = d.dt.year - (d.dt.month < 8).astype(int)
    return y.astype(str) + "-" + (y + 1).astype(str)


def load_league() -> pd.DataFrame:
    tmx = (pd.read_csv(TEAM_XG)[["provider_match_id", "date"]]
           .rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id"))
    pm = pd.read_csv(UNDERSTAT, low_memory=False).merge(tmx, on="game_id", how="left")
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
    pm = pm.dropna(subset=["date", "player", "shots"])
    # a start, in Understat's shape: position is the role, "Sub" means off the bench
    pm = pm[pm.get("position").astype(str) != "Sub"]
    pm = pm[pd.to_numeric(pm["minutes"], errors="coerce").fillna(0) >= 45]
    return pd.DataFrame({"player": pm["player"].map(canonical_name), "date": pm["date"],
                         "shots": pd.to_numeric(pm["shots"], errors="coerce"),
                         "comp": "league"}).dropna()


def load_uefa(league: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """European starts, with each name resolved onto a league player.

    ESPN and Understat spell players differently -- "Kylian Mbappé" against
    "Kylian Mbappe-Lottin", "Pape Gueye" against "Pape Alassane Gueye" --
    and canonical_name does not reconcile those, so a plain join silently drops
    exactly the players this test is about. NameIndex is the repo's matcher for
    it; ranking by recency so a live player beats a retired namesake.
    """
    from mundialytics.identity.current_squads import NameIndex

    u = pd.read_csv(path or UEFA, low_memory=False)
    u["date"] = pd.to_datetime(u["date"], errors="coerce")
    u = u.dropna(subset=["date", "player", "shots"])
    u = u[u["starter"].astype(str).str.lower().isin({"true", "1", "yes"})]

    # Only big-five clubs. Two thirds of a European field plays outside the five
    # leagues, and a name lookup with no club to confirm it would hand an Ajax
    # or Benfica player's shots to a same-named player the league data knows.
    found = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv",
                        usecols=["home_team", "away_team"], low_memory=False)
    big5 = set(found["home_team"]) | set(found["away_team"])
    from mundialytics.identity.normalization import canonical_team_name
    before_teams = u["team"].nunique()
    u = u[u["team"].astype(str).map(canonical_team_name).isin(big5)]
    print(f"European clubs kept (big-five only): "
          f"{u['team'].nunique()}/{before_teams}")

    idx = NameIndex()
    last = league.groupby("player")["date"].max()
    for name, when in last.items():
        idx.add(name, name, rank=float(pd.Timestamp(when).timestamp()))
    resolved = {n: idx.lookup(n) for n in u["player"].astype(str).unique()}
    hit = sum(v is not None for v in resolved.values())
    print(f"European names resolved onto a league player: {hit}/{len(resolved)} "
          f"({hit/max(len(resolved),1):.0%})")

    u = u.assign(player=u["player"].astype(str).map(resolved))
    return pd.DataFrame({"player": u["player"], "date": u["date"],
                         "shots": pd.to_numeric(u["shots"], errors="coerce"),
                         "comp": "europe"}).dropna()


def run(window: int, raw: bool, uefa_path: Path | None = None) -> None:
    lg = load_league()
    eu = load_uefa(lg, uefa_path)
    print(f"league starts: {len(lg):,} ({lg.date.min():%Y-%m} to {lg.date.max():%Y-%m})")
    print(f"European starts: {len(eu):,} ({eu.date.min():%Y-%m} to {eu.date.max():%Y-%m})")

    if not raw:
        k = lg["shots"].mean() / max(eu["shots"].mean(), 1e-9)
        eu = eu.assign(shots=eu["shots"] * k)
        print(f"European shots normalised by x{k:.3f} "
              f"(league {lg['shots'].mean():.3f} vs European {lg['shots'].mean()/k:.3f} per start)")

    both = pd.concat([lg, eu]).sort_values(["player", "date"]).reset_index(drop=True)
    gmean = float(lg["shots"].mean())

    rows = []
    for player, g in both.groupby("player", sort=False):
        s = g["shots"].to_numpy(float)
        isleague = (g["comp"].to_numpy() == "league")
        dates = g["date"].to_numpy()
        lg_hist: list[float] = []
        all_hist: list[tuple[float, float]] = []      # (shots, weight)
        for i in range(len(g)):
            if isleague[i] and len(lg_hist) >= MIN_PRIOR and len(all_hist) >= MIN_PRIOR:
                a = lg_hist[-window:]
                b = all_hist[-window:]
                row = {"date": dates[i], "player": player, "act": s[i],
                       "league": (sum(a) + SHRINK_K * gmean) / (len(a) + SHRINK_K),
                       "n_eu": sum(1 for v, w in b if w != 1.0)}
                # `all` counts a European start like a league one; `half` counts it
                # at half weight. Both are reported every run: with two arms and
                # one winner the honest reading is "we tested two", not "look at
                # this one".
                for name, euw in (("all", 1.0), ("half", 0.5)):
                    num = sum(v * (1.0 if w == 1.0 else euw) for v, w in b)
                    den = sum(1.0 if w == 1.0 else euw for v, w in b)
                    row[name] = (num + SHRINK_K * gmean) / (den + SHRINK_K)
                rows.append(row)
            all_hist.append((s[i], 1.0 if isleague[i] else 0.0))
            if isleague[i]:
                lg_hist.append(s[i])

    m = pd.DataFrame(rows)
    if m.empty:
        print("no scorable rows")
        return
    m["season"] = _season(pd.to_datetime(m["date"]))
    print(f"\nscorable league starts: {len(m):,}; "
          f"with a European match inside the window: {(m.n_eu > 0).sum():,}")

    def dev(p, a):
        p = np.clip(p, 1e-6, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0) / p), 0.0)
        return float(2 * np.sum(term - (a - p)) / len(a))

    def table(d: pd.DataFrame, label: str) -> None:
        if len(d) < 50:
            print(f"\n{label}: too few rows ({len(d)})")
            return
        arms = ("all", "half")
        print(f"\n{label} (n={len(d):,})   deltas vs the league-only arm, negative = better")
        head = f"  {'season':12s} {'n':>7s} {'MAE lg':>8s} {'dev lg':>8s}"
        for a in arms:
            head += f" {'dMAE ' + a:>11s} {'ddev ' + a:>11s}"
        print(head)
        # A season with no European data at all has identical arms; counting it
        # as a loss (or a win) would be a lie about how often this was tested.
        won = {a: 0 for a in arms}
        live = 0
        for s in sorted(d.season.unique()):
            g = d[d.season == s]
            testable = bool((g["n_eu"] > 0).any())
            live += testable
            ml = np.abs(g["league"] - g.act).mean()
            dl = dev(g["league"].to_numpy(), g.act.to_numpy())
            line = f"  {s:12s} {len(g):7,d} {ml:8.4f} {dl:8.4f}"
            for a in arms:
                ma = np.abs(g[a] - g.act).mean()
                da = dev(g[a].to_numpy(), g.act.to_numpy())
                won[a] += testable and da < dl
                line += f" {ma-ml:+11.4f} {da-dl:+11.4f}"
            print(line + ("" if testable else "   (no European data — arms identical)"))
        ml = np.abs(d["league"] - d.act).mean()
        dl = dev(d["league"].to_numpy(), d.act.to_numpy())
        line = f"  {'POOLED':12s} {len(d):7,d} {ml:8.4f} {dl:8.4f}"
        for a in arms:
            line += (f" {np.abs(d[a]-d.act).mean()-ml:+11.4f}"
                     f" {dev(d[a].to_numpy(), d.act.to_numpy())-dl:+11.4f}")
        print(line)
        print("   deviance won: " + ", ".join(f"{a} {won[a]}/{live}" for a in arms)
              + " testable seasons")

    table(m, "ALL league starts")
    table(m[m.n_eu > 0], "ONLY where a European match is inside the window")
    table(m[m.n_eu >= 2], "ONLY with 2+ European matches inside the window")
    print("\nBar: deviance better in every season on the slice where it applies,"
          " and not worse overall.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--raw", action="store_true", help="skip the level normalisation")
    ap.add_argument("--uefa", default=None,
                    help="European rows to read instead of the production file, which holds "
                         "the current season only (see the 2026-10-06 backfill incident)")
    a = ap.parse_args()
    run(a.window, a.raw, Path(a.uefa) if a.uefa else None)


if __name__ == "__main__":
    main()
