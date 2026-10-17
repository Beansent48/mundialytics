"""Score the stale-state arm against the deployed one, per market and per fold.

Both probabilities come from `TeamPropsModel.predict_fixture`, so this compares
what the board would have quoted, not a reimplementation of it.

The headline is the `touched` subset -- matches involving a club without 19
matches inside two years. The untouched rows must be identical and are checked
rather than assumed: a patch that leaked into them would mean the stale set or
the state copy is wrong.

    python scripts/props_stale_form/score.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DUMPS = ROOT / "outputs/props_stale"


def ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    files = sorted(DUMPS.glob("wf_*.csv"))
    if not files:
        raise SystemExit(f"no wf_*.csv in {DUMPS}")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pd.set_option("display.width", 220)
    d["ll_base"] = ll(d.p_base.to_numpy(float), d.over.to_numpy(float))
    d["ll_arm"] = ll(d.p_arm.to_numpy(float), d.over.to_numpy(float))
    d["d"] = d.ll_arm - d.ll_base

    untouched = d[~d.touched]
    leak = float((untouched.p_base - untouched.p_arm).abs().max()) if len(untouched) else 0.0
    print(f"{len(d)} rows, {len(files)} folds | untouched rows: {len(untouched)}, "
          f"max |p_base - p_arm| = {leak:.2e}"
          f"{'  <-- LEAK' if leak > 1e-9 else '  (clean)'}")

    t = d[d.touched]
    print(f"\ntouched rows: {len(t)} ({len(t) / len(d):.1%} of all), "
          f"{t.date.nunique()} match dates")

    print("\n=== touched subset, by market (negative = the arm is better) ===")
    g = t.groupby("mk").apply(lambda x: pd.Series({
        "n": len(x), "ll_base": x.ll_base.mean(), "ll_arm": x.ll_arm.mean(),
        "delta": x.d.mean(), "t": x.d.mean() / (x.d.std(ddof=1) / np.sqrt(len(x)))
        if len(x) > 2 and x.d.std(ddof=1) > 0 else np.nan,
    }), include_groups=False)
    hist = t[~t.season.astype(str).str.startswith("2026")]
    folds = hist.groupby(["mk", "season"]).d.mean().unstack()
    g["folds_better"] = [f"{int((folds.loc[m] < 0).sum())}/{folds.loc[m].notna().sum()}"
                         if m in folds.index else "-" for m in g.index]
    print(g.round(5).to_string())

    print("\n=== per fold, the patched markets only ===")
    pm = t[t.mk.isin(["fouls", "yellows"])]
    print(pm.pivot_table(index="season", columns="mk", values="d",
                         aggfunc="mean").round(5).to_string())

    print("\n=== effect on ALL rows (what the board as a whole would see) ===")
    allg = d.groupby("mk").d.agg(["size", "mean"]).rename(
        columns={"size": "n", "mean": "delta"})
    print(allg.round(6).to_string())
    tot = d.d.mean()
    print(f"\nwhole board: {tot:+.6f} per quoted line "
          f"({len(t)} of {len(d)} rows can move at all)")


if __name__ == "__main__":
    main()
