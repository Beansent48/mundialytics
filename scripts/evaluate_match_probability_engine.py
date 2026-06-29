#!/usr/bin/env python
"""
v0.51-core: Match Probability Engine – evaluation script (final)
----------------------------------------------------------------
Evalúa sobre la última temporada: LogLoss, Brier, curvas de calibración
para 1X2, y MAE/RMSE para córners, faltas, tarjetas.
"""
import pandas as pd
import numpy as np
from scipy.stats import poisson
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve   # <-- import correcto
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# 1. Configuración
# -------------------------------------------------------------------
MODEL_BUNDLE_PATH = "models/v051_match_engine/model_bundle.pkl"
DATA_PATH = "data/processed/model_ready/foundation_big5_multi_season_advanced/model_ready_match_snapshots.csv"
OUT_DIR = Path("models/v051_match_engine/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not Path(MODEL_BUNDLE_PATH).exists():
    print("Modelo no encontrado. Ejecuta primero train_match_probability_engine.py")
    exit()

# -------------------------------------------------------------------
# 2. Cargar modelo
# -------------------------------------------------------------------
bundle = joblib.load(MODEL_BUNDLE_PATH)

# -------------------------------------------------------------------
# 3. Datos de test (última temporada)
# -------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

last_season = sorted(df["season"].unique())[-1]
test_df = df[df["season"] == last_season].copy()

# Pipeline desde el grupo "goals" (es el mismo para todos)
features = bundle["goals"]["features"]
imputer  = bundle["goals"]["imputer"]
scaler   = bundle["goals"]["scaler"]

X_raw = test_df[features]
X_imp = pd.DataFrame(imputer.transform(X_raw), columns=features, index=X_raw.index)
X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=features, index=X_imp.index)

# -------------------------------------------------------------------
# 4. Evaluación de goles y 1X2
# -------------------------------------------------------------------
goals = bundle["goals"]
pred_h_goals = goals["home"].predict(X_scaled)
pred_a_goals = goals["away"].predict(X_scaled)

true_h_goals = test_df["target_home_goals"].values
true_a_goals = test_df["target_away_goals"].values

valid = ~np.isnan(true_h_goals) & ~np.isnan(true_a_goals)
ph = pred_h_goals[valid]
pa = pred_a_goals[valid]
th = true_h_goals[valid]
ta = true_a_goals[valid]

# Derivar probabilidades 1X2 (Poisson independiente)
max_g = 8
probs_h = np.array([poisson.pmf(np.arange(max_g+1), mu=l) for l in ph])
probs_a = np.array([poisson.pmf(np.arange(max_g+1), mu=l) for l in pa])

prob_H = np.zeros(len(ph))
prob_D = np.zeros(len(ph))
prob_A = np.zeros(len(ph))
for i in range(len(ph)):
    p_h = probs_h[i]
    p_a = probs_a[i]
    for h in range(max_g+1):
        for a in range(max_g+1):
            if h > a:
                prob_H[i] += p_h[h] * p_a[a]
            elif h == a:
                prob_D[i] += p_h[h] * p_a[a]
            else:
                prob_A[i] += p_h[h] * p_a[a]

true_1X2 = np.where(th > ta, 0, np.where(th == ta, 1, 2))

# Métricas
ll = log_loss(true_1X2, np.column_stack([prob_H, prob_D, prob_A]))
brier_h = brier_score_loss((true_1X2 == 0).astype(int), prob_H)
print(f"Goles - LogLoss: {ll:.4f}, Brier (H): {brier_h:.4f}")

# Curva de calibración – victoria local
prob_true, prob_pred = calibration_curve((true_1X2 == 0).astype(int), prob_H, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], '--', color='gray')
plt.xlabel('Probabilidad predicha')
plt.ylabel('Fracción real')
plt.title('Calibración – Victoria Local')
plt.savefig(OUT_DIR / "calib_home_win.png")
plt.close()

# -------------------------------------------------------------------
# 5. Evaluación del resto de objetivos
# -------------------------------------------------------------------
for group_name in ["corners", "fouls", "yellow_cards", "red_cards"]:
    if group_name not in bundle:
        continue
    models = bundle[group_name]
    pred_h = models["home"].predict(X_scaled)
    pred_a = models["away"].predict(X_scaled)
    true_h = test_df[f"target_home_{group_name}"].values
    true_a = test_df[f"target_away_{group_name}"].values
    v = ~np.isnan(true_h) & ~np.isnan(true_a)
    mae_h = np.mean(np.abs(pred_h[v] - true_h[v]))
    rmse_h = np.sqrt(np.mean((pred_h[v] - true_h[v])**2))
    mae_a = np.mean(np.abs(pred_a[v] - true_a[v]))
    rmse_a = np.sqrt(np.mean((pred_a[v] - true_a[v])**2))
    print(f"{group_name}: Home MAE={mae_h:.2f}, RMSE={rmse_h:.2f} | Away MAE={mae_a:.2f}, RMSE={rmse_a:.2f}")

# Guardar métricas
report = {"goals_log_loss": ll, "goals_brier_home": brier_h}
with open(OUT_DIR / "evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Evaluación completada. Resultados en {OUT_DIR}")