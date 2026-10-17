"""Score one engine change against the deployed baseline, fold by fold.

Both sides are directories of `wf_<season>.csv` dumps from `wf_lambdas.py`, so a
change is judged on the lambdas AS SERVED through the whole deployed chain --
`chain()` here is the same one `score_level.py` verifies against
`deployed_markets_from_lambdas` and against the live log's own probabilities.

    python scripts/goals_level/compare_arms.py outputs/form_age [--base outputs/goals_level]

Matches are joined on (season, date, home, away), so only fixtures both arms
priced are compared and a fold the arm did not run is simply absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/goals_level"))
from score_level import chain  # noqa: E402

KEY = ["season", "date", "home", "away"]


def load(d: Path) -> pd.DataFrame:
    files = sorted(d.glob("wf_*.csv"))
    if not files:
        raise SystemExit(f"no wf_*.csv in {d}")
    out = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.drop_duplicates(KEY)


def metrics(d: pd.DataFrame, lh: str, la: str) -> pd.DataFrame:
    P, o25 = chain(d[lh].values, d[la].values)
    tot = d.hg + d.ag
    o = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    Y = np.eye(3)[o]
    over = (tot > 2.5).astype(float).values
    return pd.DataFrame({
        "rps": ((P.cumsum(1)[:, :2] - Y.cumsum(1)[:, :2]) ** 2).sum(1) / 2,
        "ou": -(over * np.log(np.clip(o25, 1e-9, 1))
                + (1 - over) * np.log(np.clip(1 - o25, 1e-9, 1))),
        "lam_err": (tot - (d[lh] + d[la])).abs().values,
        "bias_num": tot.values, "bias_den": (d[lh] + d[la]).values,
    }, index=d.index)


def main() -> None:
    arm_dir = Path(sys.argv[1])
    base_dir = Path(sys.argv[sys.argv.index("--base") + 1]) if "--base" in sys.argv \
        else ROOT / "outputs/goals_level"
    base, arm = load(ROOT / base_dir if not base_dir.is_absolute() else base_dir), \
        load(ROOT / arm_dir if not arm_dir.is_absolute() else arm_dir)
    m = base.merge(arm[KEY + ["lh", "la"]], on=KEY, suffixes=("_b", "_a"))
    print(f"baseline {len(base)} | arm {len(arm)} | compared {len(m)} matches")
    print(f"folds: {sorted(m.season.unique())}")
    if m.empty:
        raise SystemExit("nothing in common")

    b, a = metrics(m, "lh_b", "la_b"), metrics(m, "lh_a", "la_a")
    d = m.assign(d_rps=a.rps - b.rps, d_ou=a.ou - b.ou,
                 d_err=a.lam_err - b.lam_err,
                 bias_b=b.bias_num / b.bias_den, bias_a=a.bias_num / a.bias_den)
    per = d.groupby("season").agg(
        n=("d_rps", "size"), d_rps=("d_rps", "mean"), d_ou=("d_ou", "mean"),
        d_err=("d_err", "mean"))
    d["goals"] = d.hg + d.ag
    d["lam_b"] = d.lh_b + d.la_b
    d["lam_a"] = d.lh_a + d.la_a
    tot = d.groupby("season")[["goals", "lam_b", "lam_a"]].sum()
    per["bias_base"] = tot.goals / tot.lam_b     # actual goals / lambda served
    per["bias_arm"] = tot.goals / tot.lam_a
    print("\n=== per fold (negative = the arm is better) ===")
    print(per.round(5).to_string())
    hist = per.drop(index=[s for s in per.index if s.startswith("2026")], errors="ignore")
    print(f"\nALL     1X2 RPS {d.d_rps.mean():+.5f}   O/U {d.d_ou.mean():+.5f}")
    if len(hist):
        print(f"HIST    1X2 RPS {hist.d_rps.mean():+.5f} ({int((hist.d_rps < 0).sum())}/{len(hist)} folds)"
              f"   O/U {hist.d_ou.mean():+.5f} ({int((hist.d_ou < 0).sum())}/{len(hist)} folds)")
    for s in [x for x in per.index if x.startswith("2026")]:
        r = per.loc[s]
        print(f"{s}  1X2 RPS {r.d_rps:+.5f}   O/U {r.d_ou:+.5f}   "
              f"bias {r.bias_base:.3f} -> {r.bias_arm:.3f}")
    moved = (d.lh_a != d.lh_b) | (d.la_a != d.la_b)
    print(f"\nmatches whose lambdas moved at all: {moved.sum()} of {len(d)} ({moved.mean():.1%})")
    if moved.any():
        sub = d[moved]
        print(f"  among those: 1X2 RPS {sub.d_rps.mean():+.5f}, O/U {sub.d_ou.mean():+.5f}, "
              f"|goals-lambda| {sub.d_err.mean():+.4f}")


if __name__ == "__main__":
    main()
