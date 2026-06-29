from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

BASE = Path("outputs/validation_national_elite_recent")
POISSON_PATH = BASE / "backtest_poisson.csv"
RF_PATH = BASE / "backtest_random_forest_lambda.csv"
OUT_CSV = BASE / "ensemble_blend_results.csv"
OUT_JSON = BASE / "ensemble_blend_best.json"

PROB_COLS = ["p_home_win", "p_draw", "p_away_win"]
LABELS = ["H", "D", "A"]

def onehot(outcomes):
    mapping = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}
    return np.array([mapping[x] for x in outcomes])

def rps(P, O):
    P_cum = np.cumsum(P, axis=1)
    O_cum = np.cumsum(O, axis=1)
    return float(np.mean(np.sum((P_cum - O_cum) ** 2, axis=1) / (P.shape[1] - 1)))

def brier(P, O):
    return float(np.mean(np.sum((P - O) ** 2, axis=1)))

def evaluate(df):
    P = df[PROB_COLS].to_numpy(dtype=float)
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    y = df["actual_outcome"].astype(str).to_numpy()
    O = onehot(y)

    picked_idx = P.argmax(axis=1)
    picked = np.array(LABELS)[picked_idx]
    picked_prob = P.max(axis=1)

    return {
        "n_predictions": int(len(df)),
        "log_loss": float(log_loss(y, P, labels=LABELS)),
        "rps": rps(P, O),
        "brier_multiclass": brier(P, O),
        "accuracy_pick_max": float(np.mean(picked == y)),
        "avg_picked_probability": float(np.mean(picked_prob)),
    }

if not POISSON_PATH.exists():
    raise FileNotFoundError(f"No existe: {POISSON_PATH}")
if not RF_PATH.exists():
    raise FileNotFoundError(f"No existe: {RF_PATH}")

p = pd.read_csv(POISSON_PATH)
r = pd.read_csv(RF_PATH)

# Alinear por match_id si existe. Si no, por orden.
if "match_id" in p.columns and "match_id" in r.columns:
    key_cols = ["match_id"]
else:
    key_cols = None

if key_cols:
    keep_cols = key_cols + ["actual_outcome"] + PROB_COLS
    p2 = p[keep_cols].rename(columns={c: f"{c}_poisson" for c in PROB_COLS})
    r2 = r[key_cols + PROB_COLS].rename(columns={c: f"{c}_rf" for c in PROB_COLS})
    df = p2.merge(r2, on=key_cols, how="inner")
else:
    if len(p) != len(r):
        raise ValueError("Los CSV no tienen la misma longitud y no hay match_id para alinear.")
    df = pd.DataFrame({"actual_outcome": p["actual_outcome"]})
    for c in PROB_COLS:
        df[f"{c}_poisson"] = p[c]
        df[f"{c}_rf"] = r[c]

rows = []

# Incluimos también extremos 1.0 y 0.0 para comparar con modelos puros.
for w in [round(x, 2) for x in np.linspace(0, 1, 21)]:
    # w = peso Poisson; 1-w = peso Random Forest
    blend = df.copy()
    for c in PROB_COLS:
        blend[c] = w * blend[f"{c}_poisson"] + (1 - w) * blend[f"{c}_rf"]

    metrics = evaluate(blend)
    metrics["poisson_weight"] = float(w)
    metrics["rf_weight"] = float(1 - w)
    rows.append(metrics)

results = pd.DataFrame(rows)
results = results.sort_values(["log_loss", "rps", "brier_multiclass"], ascending=True).reset_index(drop=True)
results.to_csv(OUT_CSV, index=False)

best = results.iloc[0].to_dict()
OUT_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")

print("\nTOP 10 BLENDS")
print(results.head(10).to_string(index=False))

print("\nBEST BLEND")
print(json.dumps(best, indent=2))

print(f"\nGuardado CSV: {OUT_CSV}")
print(f"Guardado JSON: {OUT_JSON}")
