#!/usr/bin/env python
"""
v0.51-core: Match Probability Engine – training script (final)
--------------------------------------------------------------
Entrena modelos Poisson (goles, córners, tarjetas) usando
walk‑forward manual por temporada.
Guarda el modelo y las métricas.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import poisson
import joblib
import json
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# 1. Configuración
# -------------------------------------------------------------------
CSV_PATH = "data/processed/model_ready/foundation_big5_multi_season_advanced/model_ready_match_snapshots.csv"
OUT_DIR = Path("models/v051_match_engine")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_GROUPS = {
    "goals":        {"home": "target_home_goals",      "away": "target_away_goals"},
    "corners":      {"home": "target_home_corners",    "away": "target_away_corners"},
    "fouls":        {"home": "target_home_fouls",      "away": "target_away_fouls"},
    "yellow_cards": {"home": "target_home_yellow_cards","away": "target_away_yellow_cards"},
    "red_cards":    {"home": "target_home_red_cards",  "away": "target_away_red_cards"},
}

MIN_COVERAGE = 0.9   # fracción de no‑NaN para conservar una feature en el fold

# -------------------------------------------------------------------
# 2. Cargar datos
# -------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Identificar columnas a excluir como features
exclude_cols = set()
for group in TARGET_GROUPS.values():
    exclude_cols.add(group["home"])
    exclude_cols.add(group["away"])
exclude_cols.update(["match_id", "date", "competition", "season", "stage",
                     "team_scope", "home_team", "away_team", "neutral",
                     "venue_country", "venue_city", "home_team_country",
                     "away_team_country", "home_team_city", "away_team_city"])
# Excluir todas las que empiecen por "target_"
target_cols = [c for c in df.columns if c.startswith("target_")]
exclude_cols.update(target_cols)

# Lista preliminar de features (solo numéricas)
feature_candidates = [c for c in df.columns if c not in exclude_cols]
# Mantener solo columnas numéricas (int o float)
numeric_features = df[feature_candidates].select_dtypes(include=[np.number]).columns.tolist()
print(f"Columnas numéricas candidatas: {len(numeric_features)}")

# -------------------------------------------------------------------
# 3. Walk‑forward manual por temporada
# -------------------------------------------------------------------
seasons = sorted(df["season"].unique())
if len(seasons) < 2:
    raise ValueError("Se necesitan al menos 2 temporadas.")

n_folds = len(seasons) - 1   # p. ej. 5 temporadas → 4 folds
if n_folds == 0:
    n_folds = 1

all_metrics = {group: [] for group in TARGET_GROUPS}
baseline_metrics = {group: [] for group in TARGET_GROUPS}

final_models = {}
final_features = None
final_imputer = None
final_scaler = None

for fold_idx in range(n_folds):
    train_seasons = seasons[:fold_idx+1]
    test_seasons  = [seasons[fold_idx+1]]

    train_mask = df["season"].isin(train_seasons)
    test_mask  = df["season"].isin(test_seasons)

    # Solo las columnas numéricas (ya filtradas)
    X_train_raw = df.loc[train_mask, numeric_features].copy()
    X_test_raw  = df.loc[test_mask, numeric_features].copy()

    # Selección de features por cobertura en el conjunto de entrenamiento
    coverage = X_train_raw.notna().mean()
    selected = coverage[coverage >= MIN_COVERAGE].index.tolist()
    if not selected:
        print(f"Fold {fold_idx}: ninguna feature alcanza cobertura {MIN_COVERAGE}, saltando.")
        continue

    X_train = X_train_raw[selected].copy()
    X_test  = X_test_raw[selected].copy()

    # Imputación con mediana (todas numéricas, no dará error)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train),
                               columns=selected, index=X_train.index)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test),
                               columns=selected, index=X_test.index)

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp),
                                  columns=selected, index=X_train_imp.index)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test_imp),
                                  columns=selected, index=X_test_imp.index)

    # Actualizar pipeline final (último fold)
    final_features = selected
    final_imputer = imputer
    final_scaler = scaler

    # -------------------------------
    # Para cada grupo de targets
    # -------------------------------
    for group_name, targets in TARGET_GROUPS.items():
        home_t = targets["home"]
        away_t = targets["away"]

        y_train_home = df.loc[train_mask, home_t]
        y_train_away = df.loc[train_mask, away_t]
        y_test_home  = df.loc[test_mask, home_t]
        y_test_away  = df.loc[test_mask, away_t]

        # Filtrar NaN en targets
        valid_tr = y_train_home.notna() & y_train_away.notna()
        valid_te = y_test_home.notna() & y_test_away.notna()

        X_tr = X_train_scaled.loc[valid_tr]
        y_tr_h = y_train_home[valid_tr]
        y_tr_a = y_train_away[valid_tr]
        X_te = X_test_scaled.loc[valid_te]
        y_te_h = y_test_home[valid_te]
        y_te_a = y_test_away[valid_te]

        if len(X_tr) < 10 or len(X_te) < 5:
            print(f"Fold {fold_idx}, {group_name}: muestras insuficientes, saltando.")
            continue

        # Modelos Poisson
        model_h = PoissonRegressor(alpha=0.1, max_iter=1000)
        model_a = PoissonRegressor(alpha=0.1, max_iter=1000)
        model_h.fit(X_tr, y_tr_h)
        model_a.fit(X_tr, y_tr_a)

        # Predicción y métricas
        pred_h = model_h.predict(X_te)
        pred_a = model_a.predict(X_te)

        fold_metrics = {
            "home_mae":  np.mean(np.abs(pred_h - y_te_h)),
            "away_mae":  np.mean(np.abs(pred_a - y_te_a)),
            "home_rmse": np.sqrt(np.mean((pred_h - y_te_h)**2)),
            "away_rmse": np.sqrt(np.mean((pred_a - y_te_a)**2)),
        }
        all_metrics[group_name].append(fold_metrics)

        # Baseline (media del train)
        base_h = y_tr_h.mean()
        base_a = y_tr_a.mean()
        baseline_metrics[group_name].append({
            "home_mae": np.mean(np.abs(base_h - y_te_h)),
            "away_mae": np.mean(np.abs(base_a - y_te_a)),
        })

        # Si es el último fold, guardar modelos
        if fold_idx == n_folds - 1:
            final_models[group_name] = {
                "home": model_h,
                "away": model_a,
            }

# -------------------------------------------------------------------
# 4. Añadir metadatos del pipeline a cada grupo y guardar bundle
# -------------------------------------------------------------------
if not final_models:
    print("ERROR: No se generó ningún modelo.")
    exit(1)

for group_name in final_models:
    final_models[group_name]["features"] = final_features
    final_models[group_name]["imputer"] = final_imputer
    final_models[group_name]["scaler"] = final_scaler

bundle_path = OUT_DIR / "model_bundle.pkl"
joblib.dump(final_models, bundle_path)
print(f"Modelo guardado en {bundle_path}")

# -------------------------------------------------------------------
# 5. Informe resumido de métricas
# -------------------------------------------------------------------
summary = {}
for group in all_metrics:
    if not all_metrics[group]:
        continue
    avg_h_mae = np.mean([m["home_mae"] for m in all_metrics[group]])
    avg_a_mae = np.mean([m["away_mae"] for m in all_metrics[group]])
    avg_h_rmse = np.mean([m["home_rmse"] for m in all_metrics[group]])
    avg_a_rmse = np.mean([m["away_rmse"] for m in all_metrics[group]])
    base_h_mae = np.mean([m["home_mae"] for m in baseline_metrics[group]])
    base_a_mae = np.mean([m["away_mae"] for m in baseline_metrics[group]])
    summary[group] = {
        "avg_home_mae": avg_h_mae,
        "avg_away_mae": avg_a_mae,
        "avg_home_rmse": avg_h_rmse,
        "avg_away_rmse": avg_a_rmse,
        "baseline_home_mae": base_h_mae,
        "baseline_away_mae": base_a_mae,
    }

with open(OUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Entrenamiento finalizado.")