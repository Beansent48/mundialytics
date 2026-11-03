"""
Score the live 1X2 track record of the current season, by model era.

The log holds what was served before kick-off, with a model fingerprint per row
(`model_fp`). This aggregates it into RPS, log loss and calibration, split by
fingerprint, so the question "is the deployed engine performing like its
backtest?" gets a number instead of an impression.

It reads only. Settlement of European rows reuses the project's own
`_european_results`, so this never becomes a second opinion on what counts as a
result.

Run:  python scripts/tfg/score_live_track_record.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.serving.track_record import (  # noqa: E402
    PRED_LOG,
    attach_results,
)

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"

# Published backtest reference, 10,080 Big 5 matches 2020/21-2025/26.
BACKTEST_ENGINE = 0.1998
BACKTEST_MARKET = 0.1946
BACKTEST_BASE = 0.2308


def rps_1x2(p: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Ranked probability score per match for ordered categories (1, X, 2).

    `p` is (n, 3) in home/draw/away order; `outcome` is (n, 3) one-hot.
    """
    cp = np.cumsum(p, axis=1)[:, :2]
    co = np.cumsum(outcome, axis=1)[:, :2]
    return ((cp - co) ** 2).sum(axis=1) / 2.0


def log_loss_1x2(p: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    picked = (p * outcome).sum(axis=1)
    return -np.log(np.clip(picked, 1e-15, 1.0))


def bootstrap_ci(values: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def load_settled() -> tuple[pd.DataFrame, dict]:
    """The season's 1X2 rows with provenance, joined to real results."""
    log = pd.read_csv(PRED_LOG, low_memory=False)
    season = str(log["season"].dropna().max())

    rows = log[(log["season"] == season) & (log["mercado"] == "1X2")].copy()
    diag = {"season": season, "rows_1x2": int(len(rows))}

    # Rows logged before v0.53.0 carry only the pick, not the trio: they cannot
    # be scored probabilistically and are dropped rather than half-counted.
    usable = rows[rows["model_fp"].notna() & rows["p_home"].notna()].copy()
    diag["without_provenance"] = int(len(rows) - len(usable))
    diag["usable"] = int(len(usable))

    # One shared join: the foundation, then UEFA results, then the fixture
    # calendars, then a two-day window for fixtures that moved. Settlement and
    # `scripts/audit_settlement.py` use this same function.
    found = pd.read_csv(FOUND, low_memory=False)
    m = attach_results(usable, found)
    diag["foundation_last_date"] = str(found["date"].astype(str).str[:10].max())
    unsettled = m[m["home_goals"].isna()]
    diag["unsettled"] = int(len(unsettled))
    diag["unsettled_by_date"] = (unsettled.groupby("fecha").size()
                                 .astype(int).to_dict())

    settled = m.dropna(subset=["home_goals"]).copy()
    diag["settled"] = int(len(settled))

    trio = settled[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    drift = np.abs(trio.sum(axis=1) - 1.0)
    diag["max_trio_drift"] = float(drift.max()) if len(drift) else 0.0

    hg = settled["home_goals"].to_numpy(dtype=float)
    ag = settled["away_goals"].to_numpy(dtype=float)
    outcome = np.zeros_like(trio)
    outcome[hg > ag, 0] = 1.0
    outcome[hg == ag, 1] = 1.0
    outcome[hg < ag, 2] = 1.0

    settled["rps"] = rps_1x2(trio, outcome)
    settled["logloss"] = log_loss_1x2(trio, outcome)
    settled["real"] = np.where(hg > ag, "1", np.where(hg == ag, "X", "2"))
    settled["is_europe"] = settled["model_fp"].astype(str).str.startswith("euro-")
    return settled, diag


def base_rate_reference(settled: pd.DataFrame) -> float:
    """RPS of the observed outcome frequencies on this very sample.

    The backtest's 0.2308 comes from a different sample, so comparing against it
    would be comparing two things at once.
    """
    freq = np.array([(settled["real"] == k).mean() for k in ("1", "X", "2")])
    trio = np.tile(freq, (len(settled), 1))
    outcome = np.zeros_like(trio)
    for i, k in enumerate(("1", "X", "2")):
        outcome[(settled["real"] == k).to_numpy(), i] = 1.0
    return float(rps_1x2(trio, outcome).mean())


def outcome_bias(settled: pd.DataFrame) -> pd.DataFrame:
    """Predicted mass vs what happened, per outcome.

    This is the diagnostic that found the draw deficit: if the engine is short
    on an outcome it shows up here before it shows up in RPS. z treats the
    matches as independent Bernoulli draws with their own probabilities, so
    Var = sum p(1-p).
    """
    recs = []
    for col, key, name in (("p_home", "1", "Local"),
                           ("p_draw", "X", "Empate"),
                           ("p_away", "2", "Visitante")):
        p = settled[col].astype(float).to_numpy()
        y = (settled["real"] == key).to_numpy().astype(float)
        var = float((p * (1 - p)).sum())
        z = (y.sum() - p.sum()) / np.sqrt(var) if var > 0 else float("nan")
        recs.append({"resultado": name, "predicho": p.mean(),
                     "observado": y.mean(), "dif": y.mean() - p.mean(),
                     "z": z})
    return pd.DataFrame(recs)


def calibration(settled: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Predicted vs observed, pooling the three outcomes of every match."""
    recs = []
    for col, key in (("p_home", "1"), ("p_draw", "X"), ("p_away", "2")):
        recs.append(pd.DataFrame({
            "p": settled[col].astype(float).to_numpy(),
            "y": (settled["real"] == key).astype(float).to_numpy(),
        }))
    d = pd.concat(recs, ignore_index=True)
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = d.groupby("bin", observed=True).agg(n=("y", "size"),
                                            predicho=("p", "mean"),
                                            observado=("y", "mean"))
    g["ece_contrib"] = (g["predicho"] - g["observado"]).abs() * g["n"] / len(d)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    settled, diag = load_settled()
    if not len(settled):
        print("Nada liquidado todavia.")
        return 1

    print(f"=== TRACK RECORD EN VIVO · temporada {diag['season']} ===\n")
    print(f"Filas 1X2 registradas          : {diag['rows_1x2']}")
    print(f"  sin trazabilidad (descartadas): {diag['without_provenance']}")
    print(f"  usables                       : {diag['usable']}")
    print(f"  liquidadas                    : {diag['settled']}")
    print(f"  pendientes                    : {diag['unsettled']}")
    print(f"Ultima fecha en la foundation  : {diag['foundation_last_date']}")
    print(f"Desvio maximo del trio de 1.0  : {diag['max_trio_drift']:.2e}\n")

    dom = settled[~settled["is_europe"]]
    eu = settled[settled["is_europe"]]

    for name, d in (("TODO", settled), ("Ligas domesticas", dom), ("Europa", eu)):
        if not len(d):
            continue
        r = d["rps"].to_numpy()
        lo, hi = bootstrap_ci(r)
        print(f"{name:<18} n={len(d):<4} RPS={r.mean():.4f} "
              f"[{lo:.4f}, {hi:.4f}]  log loss={d['logloss'].mean():.4f}")

    print(f"\nTasas base de ESTA muestra     : RPS {base_rate_reference(settled):.4f}")
    print(f"Backtest publicado (10.080)    : motor {BACKTEST_ENGINE:.4f} · "
          f"Bet365 {BACKTEST_MARKET:.4f} · tasas base {BACKTEST_BASE:.4f}")

    print("\n=== SESGO POR RESULTADO ===")
    print(outcome_bias(settled).to_string(index=False,
                                          float_format=lambda v: f"{v:.4f}"))

    print("\n=== POR ERA DE MODELO ===")
    print(f"{'model_fp':<14} {'n':>4}  {'RPS':>7}  {'IC 95%':>18}  {'log loss':>8}  ventana")
    for fp, d in settled.groupby("model_fp"):
        r = d["rps"].to_numpy()
        lo, hi = bootstrap_ci(r)
        print(f"{fp:<14} {len(d):>4}  {r.mean():.4f}  "
              f"[{lo:.4f}, {hi:.4f}]  {d['logloss'].mean():>8.4f}  "
              f"{d['fecha'].min()} -> {d['fecha'].max()}")

    print("\n=== CALIBRACION (los tres resultados agrupados) ===")
    cal = calibration(settled)
    print(cal.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nError esperado de calibracion (ECE): {cal['ece_contrib'].sum():.4f}")

    print("\n=== PENDIENTES POR FECHA DE PARTIDO ===")
    for fecha, n in sorted(diag["unsettled_by_date"].items()):
        flag = "  <- posterior a la foundation" if fecha > diag["foundation_last_date"] else ""
        print(f"  {fecha}  {n}{flag}")

    if args.json:
        payload = {
            "diag": diag,
            "overall": {"n": int(len(settled)),
                        "rps": float(settled["rps"].mean()),
                        "logloss": float(settled["logloss"].mean())},
            "by_era": {fp: {"n": int(len(d)), "rps": float(d["rps"].mean())}
                       for fp, d in settled.groupby("model_fp")},
            "ece": float(cal["ece_contrib"].sum()),
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nEscrito: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
