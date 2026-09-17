# Diseño del modelo


## Data foundation before model improvement

The statistical model should not be improved blindly. Before changing model families or parameters, build and inspect a data-foundation dataset:

```text
raw provider files
→ canonical_matches_raw_combined.csv
→ canonical_matches.csv
→ feature coverage report
→ competition/season quality report
→ anomaly report
→ historical validation
```

The model may only rely on features that are actually covered in the selected dataset. For example:

- no corners model without corners coverage,
- no cards model without cards coverage,
- no shot model without shots and shots-on-target coverage,
- no external Elo/ClubElo assumptions unless coverage is visible,
- no club/national mixing.

This is now implemented in `v0.49.4` through `scripts/build_match_dataset.py` and `src/mundialytics/data_quality/match_dataset_foundation.py`.


## v0.49.5 Hybrid Big 5 data model direction

Accepted direction for club football:

```text
global Big 5 club model
+ league/context features
+ league-level diagnostics/calibration
+ team rolling features
+ internal Elo
+ optional ClubElo/external Elo
+ optional xG/event enrichments
```

Rationale:

- Training with all Big 5 leagues adds volume and robustness.
- League behaviour must still be explicit because local validation showed different goal behaviour by league.
- The Premier League foundation run overestimated total goals more than LaLiga, so a blindly pooled model is unsafe.
- Fully separate league models remain possible later, but they are not the default first architecture.

The model input contract is now:

```text
canonical_matches.csv
→ model_ready_match_snapshots.csv
→ model_ready_feature_contract.csv
```

`model_ready_match_snapshots.csv` contains one row per match with identity, pre-match features and post-match targets.
`model_ready_feature_contract.csv` marks every column as `identity`, `feature` or `target`.

Hard rule:

```text
Only columns marked as pre-match features may be passed into a model.
Targets are labels only.
```

### Data levels

The statistical engine should support three data levels:

```text
Level 1: goals + shots + shots on target + corners + fouls/cards + internal Elo
Level 2: Level 1 + ClubElo/external Elo
Level 3: Level 2 + xG/xA/event data
```

xG should improve the model when available, but it must not be mandatory for baseline operation.

### Feature families for the hybrid club model

Identity/context:

```text
competition
season
stage
team_scope
neutral
league/competition indicators
season phase, when implemented
```

Elo/strength:

```text
home_elo_pre
away_elo_pre
elo_diff_pre
expected_home_score_elo_pre
home_clubelo / away_clubelo, when enriched
```

League prior rates:

```text
league_goal_rate_pre
league_home_goal_rate_pre
league_away_goal_rate_pre
league_draw_rate_pre
league_btts_rate_pre
league_over25_rate_pre
```

Team rolling form/stat features:

```text
rolling goals for/against
rolling xG for/against, when available
rolling shots and shots-on-target for/against
rolling corners for/against
rolling fouls and yellow cards for/against
rolling goal/xG/shot differential
```

### Club vs national model use

Club data should be used heavily for player/event evidence:

```text
minutes, goals, shots, SOT, xG/xA, role, player form, props and awards
```

National-team results should remain national-first:

```text
national Elo, national results, competition/friendly/tournament context, neutral venue
```

Do not mix club and national rows in one base model unless a later explicit cross-context model is designed and validated.


## Principio de capas — motor estadístico vs value picks

Mundialytics debe separar explícitamente dos responsabilidades:

```text
Statistical Engine
→ predice fútbol y distribuciones: 1X2, goles, marcador exacto, córners, tarjetas, tiros,
  eventos de jugador, simulación de torneos y premios individuales.

Value Pick Engine
→ capa posterior y selectiva que busca pocas oportunidades de mercado cuando hay datos,
  calibración y edge suficiente.
```

Reglas:

- El motor estadístico se evalúa con métricas estadísticas, no con profit.
- ROI, yield o staking no deben dirigir el desarrollo del simulador.
- Los value picks no significan apostar todos los partidos ni todos los mercados.
- La capa value debe ser selectiva, normalmente con pocas señales, y validarse aparte.
- Clubes y selecciones pueden aportar evidencia, pero siempre con `team_scope`, contexto y cortes temporales explícitos.
- Para player props, la evidencia de club puede servir como baseline de jugador, pero la inferencia de selección exige elegibilidad actual, squad/lineup y contexto nacional.

## Capa 1: ELO

Calcula fuerza relativa antes de cada partido.

Features principales:

- `home_elo`
- `away_elo`
- `elo_diff`
- `expected_home_score_elo`

## Capa 2: goles esperados

Se generan dos lambdas:

```text
home_goals ~ Poisson(lambda_home)
away_goals ~ Poisson(lambda_away)
```

El MVP usa `PoissonRegressor`. La versión ML puede usar `RandomForestRegressor` para predecir lambdas y luego convertirlas en Poisson.

## Capa 3: resultado

Con Skellam:

```text
home_goals - away_goals ~ Skellam(lambda_home, lambda_away)
```

Outputs:

- Home win
- Draw
- Away win
- Over 2.5
- BTTS
- Marcador exacto

## Capa 4: eventos de equipo

Modelo de conteos esperados:

- shots
- shots on target
- corners
- fouls
- yellow cards

## Capa 5: eventos de jugador

Modelo transparente inicial:

```text
expected_count = player_rate_per90 × expected_minutes / 90 × context_multiplier
```

Después se transforma a probabilidad de línea:

```text
P(eventos >= k) = 1 - PoissonCDF(k - 1, expected_count)
```

## Capa 6: Sustituto+

Para mercados de umbral:

```text
P(final) = P(original) + (1 - P(original)) × P(entra_sustituto) × P(sustituto_cumple)
```

## Capa 7: value betting (inactiva)

Capa construida mientras la pregunta de apuestas seguía abierta. Se conserva
porque el benchmark contra el cierre de Bet365 depende del de-vigging, pero no
forma parte del producto: el motor es una herramienta de analítica. Ver `README.md`.

```text
implied_probability = 1 / decimal_odds
edge = model_probability - implied_probability
expected_return = model_probability × net_win - (1 - model_probability)
```

En modo Exchange se aplica comisión sobre la ganancia neta.

## Validación recomendada

- RPS para 1X2.
- Log loss.
- Brier score.
- Calibration curve.
- MAE/RMSE de goles/eventos.
- Paper ROI.
- Closing line value si se guardan cuotas temporales.


## Evaluación estadística por capa

La evaluación principal del motor debe mirar calidad predictiva, no rentabilidad.

### Resultado 1X2

- Log loss.
- RPS.
- Brier multiclass.
- Calibration/reliability bins.
- Accuracy del outcome con máxima probabilidad como métrica secundaria.

### Goles y marcador

- MAE/RMSE de goles locales, visitantes y totales.
- Probabilidad asignada al marcador real.
- Scoreline log loss.
- Top-1, top-3 y top-5 exact-score coverage.
- Calibración de líneas: over/under 0.5, 1.5, 2.5, 3.5, 4.5.
- BTTS yes/no calibration.

### Eventos de equipo

Para córners, tarjetas, tiros, tiros a puerta, faltas y saves:

- MAE/RMSE.
- Calibration por líneas de mercado habituales.
- Error por equipo, liga, competición y home/away.
- Segmentos por favorito/no favorito y estilo de partido.

### Eventos de jugador y premios

Para player props y premios individuales:

- Minutos esperados vs minutos reales.
- Goles, tiros, tiros a puerta, asistencias, tarjetas, saves.
- Probabilidades por jugador condicionadas a expected minutes.
- Current squad eligibility obligatoria para inferencia actual.
- Evaluación separada para club evidence y national context.

### Value pick layer

La capa value se evalúa después y por separado:

- Pocos picks, no señales masivas por partido.
- ROI/CLV/yield solo cuando hay histórico de mercado suficiente.
- Deduplicación por partido/mercado/bookmaker.
- Filtros de edge y calidad conservadores.


## v0.49.3 — Mejoras estadísticas aceptadas para el motor

Estas decisiones son parte del contrato del motor estadístico.

### 1. ELO es una feature central, no un extra

El proyecto ya incluye `src/mundialytics/ratings/elo.py` y la validación histórica usa Elo interno en walk-forward antes de predecir cada bloque.

Features base:

```text
team_elo
opponent_elo
elo_diff
```

También se soportan columnas externas de Elo/ClubElo en datasets canónicos:

```text
home_external_elo
away_external_elo
home_clubelo
away_clubelo
home_elo
away_elo
```

Cuando existen, se transforman en:

```text
external_team_elo
external_opponent_elo
external_elo_diff
```

Regla: Elo debe usarse como prior/feature de fuerza, no como sustituto completo del modelo de goles.

### 2. Diagnóstico detallado por líneas

El motor debe evaluar no solo 1X2, sino también la calidad estadística de líneas:

```text
total goals over/under 0.5, 1.5, 2.5, 3.5, 4.5
BTTS yes/no
```

Output:

```text
statistical_engine_line_calibration_<model_type>.csv
```

### 3. Calibration layer

La calibración debe ser una capa explícita posterior al modelo base.

Targets iniciales:

```text
1X2
total goals
BTTS
```

La calibración se evalúa con holdout temporal. Cuando no hay suficiente muestra, el sistema debe marcar `identity_insufficient_calibration_data`, no inventar mejora.

### 4. Dixon-Coles para marcadores bajos

El Poisson independiente sigue siendo baseline, pero debe evaluarse un ajuste Dixon-Coles para:

```text
0-0
1-0
0-1
1-1
```

Objetivo: mejorar scoreline log loss, top-k coverage y distribución de marcadores bajos.

### 5. Time decay y shrinkage

El modelo de goles debe estabilizar fuerza de equipo con:

```text
time_decay_half_life_days
rolling_shrinkage_prior_matches
```

Motivo:

- equipos cambian plantilla/entrenador/estilo,
- recién ascendidos y selecciones con poca muestra no deben producir features extremas,
- datos recientes deben pesar más que datos antiguos.

### 6. Córners y tarjetas como count models separados

Córners, tarjetas, tiros y saves no deben modelarse como simples derivados de goles.

Diseño recomendado:

```text
corners_model
cards_model
shots_model
saves_model
```

Distribuciones candidatas:

```text
Poisson baseline
Negative Binomial cuando haya overdispersion
hierarchical count model a medio plazo
```

Métricas:

```text
MAE/RMSE
calibración por líneas habituales
over/under 8.5, 9.5, 10.5, 11.5 para córners
over/under líneas de tarjetas
segmentos por liga/equipo/árbitro/home-away
```

Esta parte está planificada después de estabilizar goles, scorelines y calibración.

## v0.49.6 Enriched data model

The hybrid club model can now be fed by enriched model-ready snapshots:

```text
canonical_matches.csv
→ canonical_matches_with_clubelo.csv
→ canonical_matches_with_xg.csv
→ model_ready_match_snapshots.csv
```

New model-ready features include:

```text
calendar:
- rest days
- rest-day difference
- season progress

external strength:
- ClubElo/external Elo before match
- rating differences
- availability flags

provider xG:
- post-match home_xg/away_xg as targets/observations
- prior rolling xG as allowed model features

conversion/pressure:
- rolling shot conversion
- rolling SOT rate
- rolling SOT conversion
- rolling xG per shot
- goals minus xG
```

Hard leakage rule:

```text
Current-match xG must never be used as an input feature for that same match.
Only rolling xG from prior matches may be consumed by the model.
```

Model improvement should now be evaluated as:

```text
Level 1 Football-Data baseline
vs
Level 2 + ClubElo
vs
Level 3 + xG
```

Metrics must be reported globally and by league.


## v0.49.8 xG ingestion fallback

xG remains a Level 3 enrichment. It should improve the model when coverage exists, but it must not be mandatory.

Direct Understat scraping is best-effort only. When blocked, import a provider/manual CSV through:

```bash
python scripts/import_xg_csv.py --input <provider_xg.csv> --provider <provider_name>
```

Only rolling prior xG features may be used as model inputs. Current-match xG is a post-match observation/target and must not be passed as a pre-match feature.

## v0.50.0 advanced data model implications

Advanced provider data should improve the model through prior rolling features, not same-match leakage.

Use advanced data as follows:

```text
current-match xG/xA/possession/progression/defence = target or diagnostic
prior rolling xG/xA/progression/defence/keeper metrics = model feature
```

This supports the hybrid Big 5 design:

```text
global Big 5 model
+ league features
+ team rolling advanced features
+ internal Elo and ClubElo
+ provider coverage flags
+ by-league diagnostics/calibration
```

The next modelling work should evaluate whether the advanced features improve goal calibration, scoreline log loss, 1X2 log loss/RPS and future corner/card/player models.
