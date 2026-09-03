from __future__ import annotations

"""Reliability diagram for the deployed walk-forward 1X2 predictions.

Reads the same walk-forward file the Bet365 benchmark uses, so the picture and
the headline RPS describe one set of predictions. Writes a PNG for the README.

    python scripts/plot_calibration.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
OUT = ROOT / "docs/img/calibration_1x2.png"

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"
SERIES = [
    ("Home win", "ph", "#2a78d6", "o", (8, -4)),
    ("Draw", "pd", "#eb6834", "s", (-34, 16)),
    ("Away win", "pa", "#1baf7a", "^", (6, -12)),
]

N_BINS = 10
MIN_BIN = 40  # below this a bin is noise, not calibration


def rps3(y_idx: np.ndarray, p: np.ndarray) -> float:
    """Mean ranked probability score over the ordered outcomes home/draw/away."""
    cum_p = np.cumsum(p, axis=1)[:, :2]
    obs = np.zeros_like(p)
    obs[np.arange(len(p)), y_idx] = 1.0
    cum_o = np.cumsum(obs, axis=1)[:, :2]
    return float(((cum_p - cum_o) ** 2).sum(axis=1).mean() / 2)


def main() -> None:
    df = pd.read_csv(PREDS).dropna(subset=["hg", "ag", "ph", "pd", "pa"])
    diff = df["hg"].to_numpy(int) - df["ag"].to_numpy(int)
    outcome = np.where(diff > 0, 0, np.where(diff == 0, 1, 2))
    probs = df[["ph", "pd", "pa"]].to_numpy(float)

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(7.4, 6.6), height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    fig.patch.set_facecolor(SURFACE)

    for a in (ax, ax_n):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK_MUTED, labelsize=9, length=3, color=GRID)

    ax.plot([0, 1], [0, 1], color="#c9c8c3", lw=1.5, zorder=1,
            label="perfect calibration")

    for label, col, colour, marker, offset in SERIES:
        p = df[col].to_numpy(float)
        hit = (outcome == [c for _, c, *_ in SERIES].index(col)).astype(float)
        idx = np.clip(np.digitize(p, edges) - 1, 0, N_BINS - 1)

        xs, ys, ns = [], [], []
        for b in range(N_BINS):
            m = idx == b
            n = int(m.sum())
            ns.append(n)
            if n >= MIN_BIN:
                xs.append(p[m].mean())
                ys.append(hit[m].mean())

        ax.plot(xs, ys, color=colour, lw=2, marker=marker, ms=8,
                mec=SURFACE, mew=1.5, zorder=3, label=label)
        ax.annotate(
            label, xy=(xs[-1], ys[-1]), xytext=offset, textcoords="offset points",
            color=colour, fontsize=9.5, fontweight="medium", va="center", zorder=4,
        )
        n_arr = np.array(ns, dtype=float)
        shown = n_arr >= 1
        ax_n.plot(centres[shown], n_arr[shown], color=colour, lw=1.5,
                  marker=marker, ms=5, mec=SURFACE, mew=1, zorder=3)

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylabel("Observed frequency", color=INK_MUTED, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9.5,
                    borderaxespad=0.8,
                    labelcolor=INK_MUTED, handlelength=1.6)
    leg.set_zorder(5)

    ax_n.set_yscale("log")
    ax_n.set_ylim(bottom=1)
    ax_n.set_ylabel("Matches", color=INK_MUTED, fontsize=10)
    ax_n.set_xlabel("Predicted probability", color=INK_MUTED, fontsize=10)
    ax_n.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax_n.set_axisbelow(True)

    ax.set_title(
        f"1X2 calibration — {len(df):,} walk-forward predictions, "
        f"Big 5 leagues 2020/21–2025/26\nRPS {rps3(outcome, probs):.4f}   ·   "
        f"bins with fewer than {MIN_BIN} matches omitted",
        color=INK, fontsize=11.5, loc="left", pad=14, linespacing=1.5,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(df):,} predictions)")


if __name__ == "__main__":
    main()
