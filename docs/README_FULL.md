# Mundialytics Betting Engine

Motor Python para predicción de fútbol, **simulación estadística**, player props y análisis de valor en paper mode.

Aunque se llama Mundialytics, **no funciona solo para el Mundial**. El mismo engine sirve para:

- selecciones (`team_scope = national`),
- clubes (`team_scope = club`).

La regla de diseño es estricta: **no mezclar clubes y selecciones en el mismo modelo**. Si entrenas con clubes, solo predices fixtures de clubes. Si entrenas con selecciones, solo predices fixtures de selecciones.



## v0.49.9 — Free xG via StatsBomb Open Data

Understat direct scraping can be blocked. For a free and more stable source, use the local StatsBomb Open Data checkout:

```powershell
python scripts/import_statsbomb_open_xg.py `
  --data-dir data/raw/statsbomb/open-data/data `
  --out-dir data/external/xg/statsbomb
```

This writes:

```text
data/external/xg/statsbomb/statsbomb_xg_matches.csv
data/external/xg/statsbomb/statsbomb_xg_shots.csv
data/external/xg/statsbomb/statsbomb_xg_import_report.json
```

Then enrich a league dataset:

```powershell
python scripts/enrich_matches_with_xg.py `
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv `
  --xg data/external/xg/statsbomb/statsbomb_xg_matches.csv `
  --registry data/processed/entities/team_registry.csv `
  --provider statsbomb_open_data `
  --provider-alias-column statsbomb_name `
  --out-dir data/processed/enriched/epl_clubelo_statsbomb_xg `
  --dataset-name epl_clubelo_statsbomb_xg `
  --allow-missing-xg
```

StatsBomb Open Data is official free event data, but its coverage is partial. Missing xG must remain explicit through `xg_available = false`.


## v0.49.8 — xG fallback when Understat direct scrape is blocked

Understat direct scraping is optional research mode. If the downloader reports:

```text
Could not find Understat datesData JSON in page
```

the pipeline should not stop. v0.49.8 writes an empty canonical xG file and supports provider/manual CSV import.

Try direct mode:

```powershell
python scripts/download_understat_xg.py `
  --league-season EPL:2021 EPL:2022 EPL:2023 EPL:2024 EPL:2025 `
  --out-dir data/external/xg/understat
```

If blocked, import a provider/manual xG CSV:

```powershell
python scripts/import_xg_csv.py `
  --input data/external/xg/provider_export.csv `
  --provider provider_csv `
  --out-dir data/external/xg/understat
```

Then enrich as usual:

```powershell
python scripts/enrich_matches_with_xg.py `
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv `
  --xg data/external/xg/understat/understat_xg_matches.csv `
  --registry data/processed/entities/team_registry.csv `
  --provider provider_csv `
  --out-dir data/processed/enriched/epl_clubelo_xg `
  --dataset-name epl_clubelo_xg
```

For batch jobs where xG is unavailable but the rest should continue:

```powershell
python scripts/enrich_matches_with_xg.py `
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv `
  --xg data/external/xg/understat/understat_xg_matches.csv `
  --out-dir data/processed/enriched/epl_clubelo_xg `
  --allow-missing-xg
```


## v0.49.6 — Obtener y enriquecer datos externos

La fase v0.49.6 añade scripts para convertir el contrato híbrido en datos enriquecidos reales:

```text
team_registry.csv
ClubElo cached daily snapshots
canonical_matches_with_clubelo.csv
optional Understat xG research CSVs
canonical_matches_with_xg.csv
enriched model_ready_match_snapshots.csv
```

### 1) Crear registry Big 5

```powershell
python scripts/build_team_registry.py `
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv `
            data/processed/foundation_laliga_multi_season/canonical_matches.csv `
            data/processed/foundation_seriea_multi_season/canonical_matches.csv `
            data/processed/foundation_bundesliga_multi_season/canonical_matches.csv `
            data/processed/foundation_ligue1_multi_season/canonical_matches.csv `
  --out-dir data/processed/entities `
  --dataset-name big5_team_registry
```

Revisar manualmente:

```text
data/processed/entities/team_registry.csv
```

especialmente filas con `alias_status = generated_review_needed`.

### 2) Descargar ClubElo

```powershell
python scripts/download_clubelo.py `
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv `
  --out-dir data/external/clubelo
```

### 3) Enriquecer con ClubElo

```powershell
python scripts/enrich_matches_with_clubelo.py `
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv `
  --registry data/processed/entities/team_registry.csv `
  --clubelo-dir data/external/clubelo `
  --out-dir data/processed/enriched/epl_clubelo
```

### 4) Obtener xG opcional

Opción de investigación con Understat:

```powershell
python scripts/download_understat_xg.py `
  --league-season EPL:2021 EPL:2022 EPL:2023 EPL:2024 EPL:2025 `
  --out-dir data/external/xg/understat
```

Después:

```powershell
python scripts/enrich_matches_with_xg.py `
  --matches data/processed/enriched/epl_clubelo/canonical_matches_with_clubelo.csv `
  --xg data/external/xg/understat/understat_xg_matches.csv `
  --registry data/processed/entities/team_registry.csv `
  --provider understat `
  --out-dir data/processed/enriched/epl_clubelo_xg
```

xG es enriquecimiento opcional. Si no existe cobertura, el baseline sigue funcionando.

### 5) Construir snapshots enriquecidos

```powershell
python scripts/build_model_ready_dataset.py `
  --matches data/processed/enriched/epl_clubelo_xg/canonical_matches_with_xg.csv `
  --out-dir data/processed/model_ready_epl_enriched_v0496 `
  --dataset-name model_ready_epl_enriched_v0496
```


## Qué hace

- ELO propio.
- Poisson/Random Forest para lambdas de goles.
- Skellam/matriz de marcadores para 1X2, over/under, BTTS y marcador probable.
- Eventos de equipo: tiros, tiros a puerta, córners, faltas y tarjetas.
- Player props: tiros, tiros a puerta, faltas cometidas/recibidas, tarjetas, goles y asistencias.
- Sustituto+ en mercados de jugador.
- Value betting en paper mode: probabilidad modelo, probabilidad implícita, edge y expected return.
- Backtesting walk-forward.
- Adaptadores de datos para fuentes públicas.
- Control de calidad del dataset.

## Estado actual — v0.49.6

Mundialytics Betting Engine está consolidado como motor **simulator-first** con predicción probabilística, simulación Monte Carlo, líneas dinámicas, reportes estadísticos, evaluación offline y auditoría de datos.

La última fase implementada es **v0.49.6 — External Data Enrichment & Feature Expansion**. Esta fase documenta y prepara la arquitectura híbrida acordada:

```text
global Big 5 club model
+ league/context features
+ league-level diagnostics/calibration
+ team rolling features
+ internal Elo
+ optional ClubElo/external Elo
+ optional xG/event enrichments
```

También añade un contrato práctico de datos model-ready:

```text
canonical_matches.csv
→ model_ready_match_snapshots.csv
→ model_ready_feature_contract.csv
→ model_ready_snapshot_report.json
```

La fase anterior **v0.49.4 — Data Foundation and Match Dataset Treatment** priorizó la calidad y tratamiento de los datos antes de seguir tocando modelos: permite construir datasets canónicos multi-temporada/multi-liga, perfilar cobertura de features, detectar anomalías y generar un reporte de foundation integrado en la validación histórica.

Para continuar el proyecto desde un ZIP en un chat nuevo, empieza por:

```text
docs/PROJECT_CONTINUITY.md
docs/DECISIONS.md
docs/V0492_STATISTICAL_ENGINE_EVALUATION_SPEC.md
docs/V0493_STATISTICAL_MODEL_CALIBRATION_SPEC.md
docs/V0494_DATA_FOUNDATION_SPEC.md
docs/V0495_HYBRID_BIG5_DATA_MODEL_SPEC.md
```


## Build model-ready snapshots

After building a foundation dataset, create leakage-safe snapshots for hybrid Big 5 modelling:

```powershell
python scripts/build_model_ready_dataset.py `
  --matches data/processed/foundation_laliga_multi_season/canonical_matches.csv `
  --out-dir data/processed/model_ready_laliga_multi_season_v0495 `
  --dataset-name model_ready_laliga_multi_season_v0495
```

Outputs:

```text
model_ready_match_snapshots.csv
model_ready_feature_contract.csv
model_ready_snapshot_report.json
```

Use only columns marked as `feature` in `model_ready_feature_contract.csv` as model inputs. Columns marked as `target` are post-match labels for training/evaluation only.

```text
fixtures + lineups/squads + historical events
→ match predictions
→ scoreline probabilities
→ team/player event projections
→ dynamic market lines
→ tournament simulation
→ advanced match report
→ matchday summary rankings
→ tournament visual report
→ simulator contract/audit
→ optional paper value if odds are provided
→ CSV/HTML/audit outputs

evaluation offline:
match_predictions + actual_results
→ 1X2 metrics
→ calibration bins
→ goal errors
→ scoreline coverage
→ baseline comparison
→ simulation_evaluation_report.html
```

OddsPapi/RapidAPI queda como capa opcional y experimental para cuotas actuales o históricas por `fixtureId` conocido. El backfill histórico masivo por fecha no debe bloquear el motor estadístico.

Este proyecto **no es Betfair Exchange** y no implementa trading live, back/lay automático ni ejecución real de apuestas.






### v0.49.4 — Data foundation and match dataset treatment

Before trying to improve the model, the project now treats the match dataset as a first-class artifact.

Build a cleaned, profiled canonical dataset from one or many source files:

```powershell
python scripts/build_match_dataset.py `
  --source football-data-uk `
  --inputs "data/raw/football_data/2*/2*_SP1.csv" `
  --out-dir data/processed/foundation_laliga_multi_season `
  --dataset-name foundation_laliga_multi_season `
  --drop-incomplete-goals
```

Main outputs:

```text
canonical_matches.csv
match_dataset_foundation_report.json
match_dataset_feature_coverage.csv
match_dataset_quality_by_competition_season.csv
match_dataset_anomalies.csv
match_dataset_dropped_rows.csv
```

Then validate the cleaned dataset:

```powershell
python scripts/run_historical_validation.py `
  --matches data/processed/foundation_laliga_multi_season/canonical_matches.csv `
  --out-dir outputs/validation_laliga_foundation_v0494 `
  --min-train-matches 300 `
  --retrain-every 20 `
  --max-backtest-predictions 600 `
  --min-matches-ready 500 `
  --min-backtest-predictions-ready 200 `
  --model-types poisson random_forest_lambda
```

This phase does **not** claim better model metrics by itself. It gives the model cleaner, larger and better-audited data so future model changes can be judged honestly.


### v0.49.3 — Statistical model calibration foundation

The statistical engine now treats Elo and calibration as first-class model concerns.

Implemented outputs in historical validation:

```text
statistical_engine_line_calibration_<model_type>.csv
statistical_engine_calibration_layer_<model_type>.csv
statistical_engine_dixon_coles_scorelines_<model_type>.csv
```

Implemented model/evaluation improvements:

- detailed over/under total-goals calibration bins,
- BTTS calibration bins,
- offline calibration-layer diagnostics for 1X2, totals and BTTS,
- Dixon-Coles low-score scoreline diagnostics,
- recency sample weighting via `--time-decay-half-life-days`,
- rolling-feature shrinkage via `--rolling-shrinkage-prior-matches`,
- internal Elo features as default,
- optional external Elo / ClubElo feature columns when supplied by canonical data.

Important: calibration and Dixon-Coles outputs are statistical diagnostics. They are not staking or value-pick decisions.

Corners/cards should be modelled later as separate count processes, probably with Negative Binomial candidates when overdispersion is observed.

## Capas del sistema — decisión v0.49.2

Mundialytics separa explícitamente dos capas:

```text
Statistical Engine
→ predice fútbol y distribuciones:
  1X2, goles, marcador exacto, BTTS, córners, tarjetas, tiros,
  eventos de jugador, simulación de torneos y premios individuales.

Value Pick Engine
→ capa posterior y selectiva:
  pocas oportunidades de mercado cuando existan datos, calibración,
  edge y controles de calidad suficientes.
```

Reglas de producto:

- El motor estadístico se mejora con métricas estadísticas, no con ROI.
- Profit, yield o staking no deben dirigir el desarrollo del simulador.
- El value layer no significa apostar todos los partidos ni todos los mercados.
- Player props y premios individuales requieren elegibilidad actual, squads/lineups y contexto club/national seguro.
- Clubes y selecciones pueden coexistir en el proyecto, pero no se mezclan sin `team_scope` y contratos de datos explícitos.


## Evaluación offline del simulador — v0.48.4

Primero genera predicciones con el flujo estadístico:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_matchday_current `
  --n-simulations 50000 `
  --seed 42 `
  --clean-out-dir
```

Después evalúa esas predicciones contra resultados conocidos:

```powershell
python scripts/run_simulation_evaluation.py `
  --predictions outputs/statistical_matchday_current/match_predictions.csv `
  --actual-results data/sample/sample_actual_results_for_evaluation.csv `
  --scorelines outputs/statistical_matchday_current/scoreline_distribution.csv `
  --dynamic-lines outputs/statistical_matchday_current/dynamic_market_lines.csv `
  --out-dir outputs/simulation_evaluation_current `
  --evaluation-mode sample_smoke_evaluation `
  --clean-out-dir
```

Outputs principales:

```text
simulation_metrics.json
simulation_evaluation.csv
calibration_1x2.csv
goal_error_metrics.csv
scoreline_evaluation.csv
baseline_comparison.csv
line_evaluation.csv
simulation_evaluation_report.html
```

`data/sample/sample_actual_results_for_evaluation.csv` es solo un archivo de smoke test y está marcado como `sample_smoke_not_real`. Para evaluación real, usa resultados históricos o predicciones forward guardadas antes del kickoff.

Modos de evaluación:

```text
sample_smoke_evaluation  # solo comprueba que el flujo funciona
retrospective_backtest   # histórico reconstruido; revisar riesgo de leakage
forward_evaluation       # predicciones guardadas antes del partido
```

Si faltan resultados reales, el reporte se genera con `status = not_available`. El sistema no inventa métricas.

La guía de datos para la siguiente fase está en:

```text
docs/NEXT_DATA_FOUNDATION_REQUIREMENTS.md
```



## Validación histórica estadística — v0.49.2

`run_historical_validation.py` ahora añade evaluación estadística del motor para cada modelo:

```text
statistical_engine_<model_type>_summary.json
statistical_engine_goal_errors_<model_type>.csv
statistical_engine_goal_lines_<model_type>.csv
statistical_engine_scorelines_<model_type>.csv
```

Estos outputs miden:

- errores de goles esperados,
- calibración de over/under 0.5, 1.5, 2.5, 3.5 y 4.5,
- BTTS yes/no,
- probabilidad del marcador real,
- exact-score top-1/top-3/top-5 coverage.

En `operational_validation_report.json` aparecen bajo:

```text
backtests.<model_type>.statistical_engine_evaluation
```

El `historical_value_backtest`, si existe, es diagnóstico y no debe usarse como criterio principal de mejora del simulador.


## Auditoría offline de datos — v0.49.1

Ejecuta la auditoría con archivos locales. No descarga datos, no cambia modelos y no llama APIs:

```powershell
python scripts/run_data_audit.py `
  --fixtures data/input/fixtures.csv `
  --actual-results data/sample/sample_actual_results_for_evaluation.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --player-events data/sample/sample_player_events.csv `
  --odds data/input/odds.csv `
  --out-dir outputs/data_audit_current `
  --run-label sample_data_audit `
  --clean-out-dir
```

Outputs principales:

```text
data_audit_summary.json
data_audit_report.csv
coverage_report.csv
data_gaps_report.csv
entity_quality_report.csv
feature_availability_matrix.csv
next_data_requirements.csv
entity_guardrails_report.csv
squad_guardrails_report.csv
guardrail_summary.json
data_audit_report.html
```

La auditoría marca huecos como `not_available`, `coverage_gap`, `schema_gap`, `blocked`, `needs_review` o `unsafe_for_current_player_props`. Player props y premios como Golden Boot se mantienen conservadores: requieren squads/lineups actuales, historial de eventos de jugador, minutos esperados y progresión de equipo antes de inferencia actual.

## Instalación

```bash
cd mundialytics_betting_engine
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```


## Flujo diario MVP gratuito

Para obtener solo partidos del Mundial del día en hora americana, sin API-Football:

```powershell
python scripts/fetch_today_fixtures.py `
  --competition world_cup `
  --today `
  --timezone America/New_York `
  --provider auto `
  --out outputs/free_world_cup_today_et.csv `
  --raw-out outputs/free_world_cup_today_et_raw.json
```

`--provider auto` intenta SofaScore primero y ESPN como fallback. Ambas fuentes son gratuitas/keyless pero no oficiales, así que el engine guarda JSON raw para auditoría.

Si hay lineups disponibles en SofaScore:

```powershell
python scripts/fetch_fixture_lineups_free.py `
  --fixture-id <fixture_id> `
  --fixtures outputs/free_world_cup_today_et.csv `
  --out outputs/lineups_<fixture_id>.csv `
  --raw-out outputs/lineups_<fixture_id>_raw.json
```

Para estadísticas de equipo/partido:

```powershell
python scripts/build_team_match_stats.py `
  --player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --out outputs/team_match_stats.csv

python scripts/validate_team_props.py `
  --team-match-stats outputs/team_match_stats.csv `
  --strict

python scripts/predict_team_props.py `
  --team-match-stats outputs/team_match_stats.csv `
  --fixtures outputs/free_world_cup_today_et.csv `
  --out outputs/team_props_today.csv
```

Corners solo se ofrecen si la fuente histórica contiene corners reales. El sistema no los inventa.

Reporte visual HTML:

```powershell
python scripts/build_daily_report.py `
  --fixtures outputs/free_world_cup_today_et.csv `
  --team-props outputs/team_props_today.csv `
  --player-props outputs/matchday_analysis/safe_lineup_props.csv `
  --out outputs/daily_report.html
```

Ver detalles en `docs/V19_DAILY_MVP_FLOW.md`.

## Flujo recomendado v0.48.3 — simulador estadístico

El smoke path principal del simulador estadístico es:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_simulator_v0483 `
  --n-simulations 1000 `
  --seed 42 `
  --clean-out-dir
```

Para un reporte serio de torneo, usa una simulación grande:

```powershell
python scripts/run_statistical_matchday.py `
  --fixtures data/input/fixtures.csv `
  --lineups data/input/current_lineups.csv `
  --squads data/input/squads.csv `
  --odds data/input/odds.csv `
  --tournament-config data/input/tournament_config.csv `
  --historical-events data/sample/sample_player_events.csv `
  --out-dir outputs/statistical_simulator_v0483_50k `
  --n-simulations 50000 `
  --seed 42 `
  --clean-out-dir
```

`50,000` simulaciones reducen ruido Monte Carlo en probabilidades de torneo, pero no sustituyen buena calidad de datos, calibración ni lineups fiables.

Outputs principales:

```text
match_predictions.csv
scoreline_distribution.csv
team_stats_predictions.csv
player_event_predictions.csv
dynamic_market_lines.csv
betting_edges.csv
recommended_picks.csv
tournament_simulation.csv
tournament_details.csv
competition_summary.csv
matchday_summary.csv
matchday_summary.json
tournament_report.csv
tournament_report.json
daily_report.html
audit_report.json
simulation_contract_report.json
```

Para validar la ruta consolidada:

```bash
pytest tests/test_v047_statistical_matchday_smoke.py tests/test_v048_statistical_simulator_contract.py tests/test_v0481_advanced_match_report.py tests/test_v0482_matchday_summary.py tests/test_v0483_tournament_report.py
```

`daily_report.html` en v0.48.3 incluye:

```text
Executive Summary
Matchday Summary Rankings
Match Probabilities
Advanced Match Cards
Top Scorelines
Dynamic Goal Lines
Not Available Markets
Team Statistics
Player Statistics
Data Quality
Tournament Visual Report
Simulation Metadata
```

Ver:

```text
docs/V048_STATISTICAL_SIMULATOR_SPEC.md
docs/V0481_ADVANCED_MATCH_REPORT_SPEC.md
docs/V0482_MATCHDAY_SUMMARY_RANKINGS_SPEC.md
docs/V0483_TOURNAMENT_VISUAL_REPORT_SPEC.md
```


## Demo rápida

```bash
python scripts/run_demo.py
```

Genera:

```text
outputs/demo_daily_picks.csv
```

## Entrenar selecciones

```bash
python scripts/train_from_csv.py \
  --matches data/sample/sample_matches.csv \
  --model-out models/national_goal_model.pkl \
  --model-type poisson

python scripts/predict_fixtures.py \
  --bundle models/national_goal_model.pkl \
  --fixtures data/sample/sample_national_fixtures.csv \
  --out outputs/national_fixture_predictions.csv
```

## Entrenar clubes

```bash
python scripts/train_from_csv.py \
  --matches data/sample/sample_club_matches.csv \
  --model-out models/club_goal_model.pkl \
  --model-type poisson

python scripts/predict_fixtures.py \
  --bundle models/club_goal_model.pkl \
  --fixtures data/sample/sample_club_fixtures.csv \
  --out outputs/club_fixture_predictions.csv
```

## Convertir datos públicos a schema canónico

### Selecciones: international-results

```bash
python scripts/download_data_sources.py international-results \
  --out data/raw/international_results/results.csv

python scripts/build_dataset.py \
  --source international-results \
  --input data/raw/international_results/results.csv \
  --out data/processed/national_matches.csv
```

### Clubes: Football-Data.co.uk

```bash
python scripts/download_data_sources.py football-data-uk \
  --url https://www.football-data.co.uk/mmz4281/2526/E0.csv \
  --out data/raw/football_data_uk/2526_E0.csv

python scripts/build_dataset.py \
  --source football-data-uk \
  --input data/raw/football_data_uk/2526_E0.csv \
  --out data/processed/epl_2526.csv \
  --season 2025-2026
```


## Construir datasets de eventos/player props

### StatsBomb Open Data

```bash
python scripts/build_event_datasets.py statsbomb \
  --input data/raw/statsbomb/open-data/data/events \
  --competition "StatsBomb Open Data" \
  --team-scope club \
  --player-events-out data/processed/statsbomb_player_events.csv \
  --team-events-out data/processed/statsbomb_team_events.csv \
  --lineups-out data/processed/statsbomb_lineups.csv \
  --tactical-out data/processed/statsbomb_tactical_shifts.csv
```

### Wyscout public event dataset

```bash
python scripts/build_event_datasets.py wyscout \
  --events data/raw/wyscout/events_England.json \
  --matches data/raw/wyscout/matches_England.json \
  --players data/raw/wyscout/players.json \
  --teams data/raw/wyscout/teams.json \
  --competition "Premier League" \
  --season 2017-2018 \
  --team-scope club
```

Ver `docs/DATA_SOURCE_STRATEGY.md` y `docs/EVENT_DATA_PIPELINE.md`.

## Backtesting

```bash
python scripts/backtest_from_csv.py \
  --matches data/sample/sample_matches.csv \
  --out outputs/national_backtest.csv \
  --summary-out outputs/national_backtest_summary.json \
  --model-type random_forest_lambda
```

## Diagnóstico de datos

```bash
python scripts/diagnose_dataset.py \
  --matches data/sample/sample_matches.csv \
  --out outputs/sample_matches_diagnostic.json
```

## Interfaz Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Estructura

```text
src/mundialytics/
├── artifacts/            # bundles versionados con metadata
├── betting/              # odds, value, staking, market mapping
├── data/                 # schema, identity, loaders, adapters, quality
├── evaluation/           # RPS, Brier, log loss, backtesting
├── features/             # rolling features y player baselines
├── models/               # goles, eventos, minutos, Sustituto+
├── ratings/              # ELO
├── reports/              # daily picks y paper ledger
├── simulation/           # simulación de torneos
└── statistical_core/      # motor probabilístico, simulador, líneas dinámicas y reportes
```

## Estado real

Esto es una **base reproducible avanzada**, no un bot listo para apostar dinero real.

Listo:

- arquitectura,
- scopes club/national,
- entrenamiento/predicción,
- backtesting,
- fuentes/adaptadores,
- demo de props,
- tests.

Pendiente para producción:

- datos reales amplios,
- calibración seria,
- cuotas actuales reales,
- capa opcional de odds provider validada con snapshots reales,
- lineups/lesiones/minutos probables,
- paper tracking durante suficientes partidos.

Ver `docs/AGENT_ITERATION_REPORT.md` para el diagnóstico de la iteración v0.3.

## Paper tracking

```bash
python scripts/run_demo.py
python scripts/paper_track.py append \
  --picks outputs/demo_daily_picks.csv \
  --ledger outputs/paper_ledger.csv \
  --created-at 2026-06-15T16:45:00Z \
  --stake 1.0
```

## Auditoría v0.3.1

Se corrigieron regresiones detectadas en la revisión minuciosa:

- lineups y sustitutos ahora se normalizan igual que odds/player events;
- Sustituto+ ya encuentra reemplazos canónicos en la demo;
- `paper_track.py append` acepta `--created-at`;
- el ledger distingue `total_stake`, `open_stake` y `settled_stake`;
- `add_team_identity_columns` ya no puede truncar filas cuando falta `team_scope`;
- Football-Data.co.uk parsea fechas con `dayfirst=True`;
- el backtest walk-forward usa ELO/features actualizados hasta cada partido, incluso si el modelo se reentrena cada N partidos;
- los documentos ya no recomiendan usar un fixture mixto para modelos club/national.

## Auditoría v0.4-agent

Nueva tanda de mejoras orientada a producto reproducible:

- `reports/match_value.py`: compara predicciones 1X2 contra cuotas decimales.
- `scripts/value_from_predictions.py`: genera picks de partido desde `predict_fixtures.py` + odds.
- `scripts/build_odds_dataset.py`: extrae odds 1X2 de CSVs de Football-Data.co.uk.
- `scripts/value_backtest.py`: backtestea picks de valor contra resultados reales.
- `evaluation/readiness.py` y `scripts/quality_gate.py`: puerta de calidad para evitar fiarse de datasets pequeños/sucios.
- Backtest walk-forward ahora añade `picked_outcome`, `picked_probability`, `picked_correct` y fiabilidad por bins de confianza.
- `shrink_probability()` ya maneja `pd.NA` y fuerza cero sin errores.

Ejemplo de value betting 1X2 en paper mode:

```bash
python scripts/predict_fixtures.py \
  --bundle models/national_goal_model.pkl \
  --fixtures data/sample/sample_national_fixtures.csv \
  --out outputs/national_fixture_predictions.csv

python scripts/value_from_predictions.py \
  --predictions outputs/national_fixture_predictions.csv \
  --odds data/sample/sample_match_odds.csv \
  --out outputs/match_value_picks.csv
```

Puerta de calidad:

```bash
python scripts/diagnose_dataset.py \
  --matches data/sample/sample_matches.csv \
  --out outputs/sample_matches_diagnostic.json

python scripts/backtest_from_csv.py \
  --matches data/sample/sample_matches.csv \
  --out outputs/national_backtest.csv \
  --summary-out outputs/national_backtest_summary.json

python scripts/quality_gate.py \
  --data-report outputs/sample_matches_diagnostic.json \
  --backtest-summary outputs/national_backtest_summary.json \
  --out outputs/quality_gate.json
```

## One-command historical validation

To check whether the system is ready for paper mode on real data, use:

```bash
python scripts/run_historical_validation.py --matches data/sample/sample_matches.csv --out-dir outputs/validation_sample --min-train-matches 10 --min-matches-ready 10 --min-backtest-predictions-ready 5
```

For real datasets, see `docs/NEXT_VALIDATION_STEPS.md`. This script runs diagnostics, walk-forward backtests, readiness gates, optional historical odds value backtests, and trains a final model bundle.

## Validación operativa recomendada v0.8

No uses el histórico completo desde 1872 salvo que quieras hacer un benchmark largo. Para un modelo moderno de selecciones, el script filtra por defecto desde 2010 y limita el número de partidos evaluados. Esto evita ejecuciones eternas y usa datos más coherentes con el fútbol actual.

```powershell
python scripts/run_historical_validation.py `
  --source international-results `
  --input data/raw/international_results/results.csv `
  --out-dir outputs/validation_national_modern `
  --min-train-matches 1000 `
  --retrain-every 100 `
  --max-completed-matches 3000 `
  --max-backtest-predictions 1000 `
  --min-matches-ready 1500 `
  --min-backtest-predictions-ready 500
```

Esto valida tanto `poisson` como `random_forest_lambda` de forma acotada.

## Validación de player props

El histórico de resultados no valida props. Para props necesitas eventos. Una vez creado `player_events.csv` con StatsBomb o Wyscout:

```powershell
python scripts/validate_player_props.py `
  --player-events data/processed/wyscout_player_events.csv `
  --lineups data/processed/wyscout_lineups.csv `
  --out-dir outputs/validation_player_props `
  --min-train-matches 500 `
  --test-matches 300
```

Para comprobar que el script funciona sin descargar datos externos:

```powershell
python scripts/validate_player_props.py `
  --player-events data/sample/player_events_synthetic.csv `
  --out-dir outputs/validation_player_props_sample `
  --min-train-matches 20 `
  --test-matches 10
```


## Player props with real event data

For player props, do **not** use result-only datasets such as `international_results`. Use real event data. The recommended free source is StatsBomb Open Data, the same type of data used in scouting-style dashboards.

```powershell
python scripts/download_data_sources.py statsbomb-open-data --out data/raw/statsbomb/open-data/data
python scripts/run_player_props_pipeline.py `
  --statsbomb-data data/raw/statsbomb/open-data/data `
  --team-scope club `
  --out-dir outputs/player_props_statsbomb `
  --min-matches 50 `
  --min-player-rows 500 `
  --min-train-matches 50 `
  --test-matches 300
```

This builds `player_events`, `team_events`, `lineups`, `tactical_shifts`, checks event coverage strictly, and validates markets such as 1+ shot, 1+ shot on target, 1+ foul committed and 1+ yellow card. See `docs/STATS_BOMB_PLAYER_PROPS_OPERATION.md`.


### Player props calibration

After running the StatsBomb player-props pipeline, calibrate probabilities by market:

```powershell
python scripts/calibrate_player_props.py `
  --predictions outputs/player_props_statsbomb_national/validation/player_props_backtest_predictions.csv `
  --out-dir outputs/player_props_statsbomb_national/calibration `
  --calibration-fraction 0.5 `
  --min-market-rows 500
```

Or run it inside the full event pipeline:

```powershell
python scripts/run_player_props_pipeline.py `
  --statsbomb-data data/raw/statsbomb/open-data/data `
  --team-scope national `
  --out-dir outputs/player_props_statsbomb_national `
  --min-matches 50 `
  --min-player-rows 500 `
  --min-train-matches 50 `
  --test-matches 300 `
  --run-calibration `
  --min-calibration-market-rows 500
```

See `docs/PROP_CALIBRATION.md`.


### v0.11 safe player-prop operation

For player props, do not generate candidates from historical players. Use `scripts/run_safe_props_for_lineups.py` with a current lineup CSV. See `docs/SAFE_LINEUP_PROPS.md`.


### v0.15 domain labels

Player-prop datasets now include objective competition labels: `team_type`, `competition_context`, and `gender`, while keeping `team_scope` as a backward-compatible alias (`club` or `national`). This fixes the previous ambiguity where domestic club leagues could be mislabeled as national-team data. See `docs/V15_COMPETITION_TAXONOMY.md`.


### v0.16 highlights

- Hierarchical player-prop calibration: competition → domain context → team type/gender → market fallback.
- Optional club-to-national player evidence via `--feature-player-events`, with temporal cutoff to avoid leakage.
- Prop outputs now expose `club_minutes_sample`, `national_minutes_sample`, `cross_context_feature_used`, `calibration_level`, and `calibration_group_key`.
- See `docs/V16_HIERARCHICAL_CALIBRATION_AND_CROSS_CONTEXT.md` and `docs/V16_AUDIT_REPORT.md`.

### v0.17 player-props finalization

After running a clean props rebuild, the pipeline now writes `player_props_policy.json`. Use it during matchday safe inference:

```powershell
python scripts/run_safe_props_for_lineups.py `
  --lineups data/templates/current_lineups_template.csv `
  --player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --calibration-predictions outputs/player_props_national_men_v16/validation/player_props_backtest_predictions.csv `
  --calibration-policy outputs/player_props_national_men_v16/player_props_policy.json `
  --out outputs/safe_lineup_props.csv `
  --strict-lineup-contract
```

The policy chooses simple vs adaptive hierarchical calibration per market and preserves safety caps/floors for paper-mode use.


### Player identity diagnosis

Before trusting matchday player props, especially for national-team squads, run:

```powershell
python scripts/diagnose_player_identity.py `
  --player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --players "Federico Valverde" "Alvaro Morata" "Lamine Yamal"
```

`SAFE_LINEUP_PROPS` outputs include resolved player identity columns. Treat
`player_match_status != matched` or `sample_size = 0` for expected starters as a
data-quality warning, not a betting signal.


### v0.18 Provider Identity Layer

For the free MVP, use API-Football/API-Sports as the operational provider for current fixtures and lineups, and StatsBomb Open Data as the historical event source. The new provider identity layer maps `api_football:<provider_player_id>` to the historical `player_id_global` used by player-props models.

Key commands:

```powershell
$env:API_FOOTBALL_KEY="YOUR_KEY"
python scripts/fetch_api_football_lineups.py --fixture-id 123456 --date 2026-06-26 --competition "FIFA World Cup" --out outputs/api_football_current_lineups.csv
python scripts/build_provider_identity_map.py --provider-players outputs/api_football_current_lineups.csv --historical-player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv --out data/identity/player_identity_map.csv
python scripts/run_safe_props_for_lineups.py --lineups outputs/api_football_current_lineups.csv --identity-map data/identity/player_identity_map.csv --player-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv --calibration-policy outputs/player_props_national_men_v16/player_props_policy.json --out outputs/safe_lineup_props_provider.csv --strict-lineup-contract
```

See `docs/V18_PROVIDER_IDENTITY_LAYER.md`.


## v0.18.1 — Today fixtures command

Added a real fixture discovery step so examples no longer require an invented `fixture_id`. Use:

```powershell
$env:API_FOOTBALL_KEY="YOUR_KEY"
python scripts/fetch_today_fixtures.py `
  --timezone America/New_York `
  --out outputs/api_football_today_fixtures_et.csv `
  --raw-out outputs/api_football_today_fixtures_et.json
```

The command prints a compact table with real `fixture_id` values. Use one of those IDs in `fetch_api_football_lineups.py`. See `docs/V18_1_TODAY_FIXTURES.md`.

### v0.18.2 — only World Cup fixtures

Use this before fetching lineups; do not use placeholder fixture IDs.

```powershell
$env:API_FOOTBALL_KEY="TU_KEY"

python scripts/fetch_world_cup_fixtures.py `
  --today `
  --timezone America/New_York `
  --out outputs/api_football_world_cup_today_et.csv `
  --raw-out outputs/api_football_world_cup_today_et.json
```

For an exact date in US Eastern time:

```powershell
python scripts/fetch_world_cup_fixtures.py `
  --date 2026-06-17 `
  --timezone America/New_York `
  --out outputs/api_football_world_cup_2026-06-17_et.csv
```

The command uses API-Football `league=1` and `season=2026`, and post-filters kickoff dates in the requested timezone.

## v0.18.3 Free current World Cup fixtures

API-Football free plans can reject current World Cup seasons with an error like
`Free plans do not have access to this season`. For the free MVP, use the
SofaScore public scheduled-events endpoint for today's fixture list and keep
API-Football as optional for provider IDs/lineups if your plan allows it.

Key-free World Cup fixtures in US Eastern time:

```powershell
python scripts/fetch_world_cup_fixtures_free.py `
  --today `
  --timezone America/New_York `
  --out outputs/sofascore_world_cup_today_et.csv `
  --raw-out outputs/sofascore_world_cup_today_et.json
```

Use exact date to avoid timezone ambiguity:

```powershell
python scripts/fetch_world_cup_fixtures_free.py `
  --date 2026-06-17 `
  --timezone America/New_York `
  --out outputs/sofascore_world_cup_2026-06-17_et.csv
```

Generic SofaScore fixture fetcher:

```powershell
python scripts/fetch_sofascore_fixtures.py `
  --today `
  --timezone America/New_York `
  --world-cup
```

This source is free and keyless but unofficial. Cache `--raw-out`, keep request
volume low, and treat it as a fixture discovery layer rather than a guaranteed
contracted data feed.

## v0.36 market model audit

Before moving to paper betting, use these scripts to separate model quality from betting value:

```powershell
python scripts/audit_market_coverage.py `
  --historical-events outputs/player_props_statsbomb_clean_v15/statsbomb_player_events_clean.csv `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --out-dir outputs/market_coverage_audit_current

python scripts/backtest_pick_policy.py `
  --match-backtest outputs/evaluation_current/match_backtest_predictions.csv `
  --out-dir outputs/pick_policy_backtest_current `
  --min-picks 30 `
  --write-odds-template
```

New outputs:

- `market_coverage_audit.csv`: which markets are trainable from the current data.
- `market_model_performance.csv`: aggregate predictive performance by market/split.
- `market_line_performance.csv`: line-level predictive performance.
- `market_threshold_performance.csv`: high-confidence performance by market and probability threshold.
- `market_model_takeaways.json`: conservative status summary.

Rules:

- Corners stay `not_available` until a real corners target exists.
- Goalkeeper saves stay `not_available` until a real saves/goalkeeper_saves target exists.
- No silent proxies for corners or saves.

## v0.40 odds-ready layer

The project now includes a provider-agnostic odds contract so historical odds APIs can be plugged in later without changing the model layer.

Create a compact odds shortlist/template:

```powershell
python scripts/build_odds_ready_shortlist.py `
  --line-signals outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv `
  --decision-matrix outputs/market_distribution_lab_current_v0391_clean/market_side_decision_matrix.csv `
  --out-dir outputs/odds_ready_current `
  --decisions candidate `
  --min-model-probability 0.52 `
  --min-fair-odds 1.25 `
  --max-fair-odds 3.50
```

Then, once a historical odds API/export is mapped into the universal schema:

```powershell
python scripts/calculate_value_edges_from_odds.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --historical-odds data/processed/historical_odds_input.csv `
  --out-dir outputs/value_edges_current
```

See `docs/ODDS_READY_CONTRACT.md`.

## v0.41 odds readiness and coverage audit

v0.41 adds provider-shopping and historical odds coverage checks without fetching odds or changing model predictions.

```powershell
python scripts/check_odds_readiness.py `
  --odds-template outputs/odds_ready_current/odds_needed_template.csv `
  --line-signals outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv `
  --decision-matrix outputs/market_distribution_lab_current_v0391_clean/market_side_decision_matrix.csv `
  --out-dir outputs/odds_readiness_current
```

After mapped historical odds exist:

```powershell
python scripts/audit_historical_odds_coverage.py `
  --model-lines outputs/odds_ready_current/model_market_lines.csv `
  --historical-odds data/processed/historical_odds_input.csv `
  --out-dir outputs/historical_odds_coverage_current
```

See `docs/V041_ODDS_READINESS_AND_COVERAGE_AUDIT.md`.

### v0.49.7 ClubElo download

Use the fast team-history mode by default:

```powershell
python scripts/download_clubelo.py `
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv `
  --registry data/processed/entities/team_registry.csv `
  --out-dir data/external/clubelo
```

Then enrich:

```powershell
python scripts/enrich_matches_with_clubelo.py `
  --matches data/processed/foundation_epl_multi_season/canonical_matches.csv `
  --registry data/processed/entities/team_registry.csv `
  --clubelo-dir data/external/clubelo `
  --out-dir data/processed/enriched/epl_clubelo `
  --dataset-name epl_clubelo
```

Legacy daily snapshots remain available with `--mode daily-snapshot`, but they are slower on multi-season datasets.

## v0.50.0 advanced football data layer

Mundialytics now has an advanced data acquisition layer. The goal is to extract maximum value from free/semi-free football data without tying the engine to one brittle provider.

Provider strategy:

```text
FBref/soccerdata/worldfootballR exports
+ Kaggle Understat backfill
+ RapidAPI xG exports
+ StatsBomb Open Data
+ manual/provider CSV import
→ canonical advanced match/player/shot contracts
→ provider-priority merge
→ advanced coverage audit
→ model-ready prior rolling features
```

Main scripts:

```powershell
python scripts/download_fbref_advanced.py --league "ENG-Premier League" --season 2021 2022 2023 2024 2025
python scripts/import_advanced_csv.py --input provider_export.csv --provider fbref
python scripts/import_kaggle_understat.py --input kaggle_understat_games.csv
python scripts/import_statsbomb_open_advanced.py --data-dir data/raw/statsbomb/open-data/data
python scripts/merge_advanced_sources.py --source fbref=... kaggle_understat=... statsbomb_open_data=...
python scripts/enrich_matches_with_advanced_stats.py --matches canonical_matches.csv --advanced canonical_advanced_match_stats.csv
python scripts/audit_advanced_data_coverage.py --matches canonical_matches_with_advanced_stats.csv
```

Hard rule: xG, xA, possession, progression and defensive actions from the current match are post-match observations. They can be targets/diagnostics, but the model may only consume prior rolling versions built from earlier matches.
