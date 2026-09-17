# Agent Iteration Report — v0.3

## Estado alcanzado

El proyecto ya no es solo una demo. Ahora tiene un flujo reproducible mínimo para:

1. convertir fuentes externas a un schema canónico,
2. entrenar modelos separados de clubes y selecciones,
3. guardar bundles con metadatos,
4. bloquear predicciones con scope incompatible,
5. predecir fixtures futuros,
6. ejecutar backtesting walk-forward,
7. generar props en paper mode con ajuste de incertidumbre,
8. auditar calidad de datos.

## Cambios críticos realizados

### 1. Scopes estrictos

Antes era posible entrenar un modelo de clubes y predecir selecciones por error. Ahora:

- `train_from_csv.py` infiere un único `team_scope` explícito.
- `ModelBundle` guarda `model_scope`.
- `predict_fixtures.py` valida que los fixtures coincidan con el scope del modelo.
- Si hay mezcla club/national, el flujo falla con un error claro.

### 2. Identidades canónicas

Añadido `src/mundialytics/data/identity.py`:

- normalización de nombres de equipos,
- alias básicos,
- `team_id`,
- `player_id_global`,
- `player_context_id`.

Esto evita que `Spain`, `España`, `ESP` o `Real Madrid CF` creen entidades distintas.

### 3. Mismo jugador, distinto contexto

El proyecto ya distingue:

```text
player_federico_valverde__club_real_madrid__competition_laliga
player_federico_valverde__national_uruguay__competition_world_cup
```

La idea correcta es usar el perfil global del jugador como baseline, pero ajustar por contexto de club/selección, rol, equipo, competición, minutos y rival.

### 4. Adaptadores de datos

Añadidos adaptadores:

- `football_data_uk.py`
- `international_results.py`
- `openfootball.py`
- `clubelo.py`
- `statsbomb.py`

También añadido `scripts/build_dataset.py` para convertir fuentes a CSV canónico.

### 5. Backtesting walk-forward

Añadido `src/mundialytics/evaluation/backtest_runner.py` y `scripts/backtest_from_csv.py`.

El backtest usa ventana expansiva: cada partido se predice usando solo partidos anteriores. Métricas:

- RPS,
- Brier multiclass,
- log loss,
- accuracy de clase máxima.

### 6. Control de incertidumbre en props

El demo anterior producía EVs demasiado optimistas. Ahora `value.py` incluye shrinkage empírico:

```text
model_probability_adjusted = shrink(raw_probability, sample_size, market_prior)
```

Los picks conservan `model_probability` y `model_probability_adjusted` para auditar.

### 7. Data quality

Añadido:

- `src/mundialytics/data/quality.py`
- `scripts/diagnose_dataset.py`

Detecta duplicados, scopes mezclados, goles faltantes, goles negativos, rango temporal y competiciones.

## Pruebas ejecutadas

```bash
python -m pytest -q
# 10 passed

python scripts/train_from_csv.py --matches data/sample/sample_matches.csv --model-out models/national_goal_model.pkl --model-type poisson
python scripts/train_from_csv.py --matches data/sample/sample_club_matches.csv --model-out models/club_goal_model.pkl --model-type poisson

python scripts/predict_fixtures.py --bundle models/national_goal_model.pkl --fixtures data/sample/sample_national_fixtures.csv --out outputs/national_fixture_predictions_v03.csv
python scripts/predict_fixtures.py --bundle models/club_goal_model.pkl --fixtures data/sample/sample_club_fixtures.csv --out outputs/club_fixture_predictions_v03.csv

python scripts/backtest_from_csv.py --matches data/sample/sample_matches.csv --out outputs/national_backtest_rf_v03.csv --summary-out outputs/national_backtest_rf_summary_v03.json --model-type random_forest_lambda

python scripts/diagnose_dataset.py --matches data/sample/sample_matches.csv --out outputs/sample_matches_diagnostic.json
python scripts/run_demo.py
```

Scope-blocking probado:

```bash
python scripts/predict_fixtures.py --bundle models/club_goal_model.pkl --fixtures data/sample/sample_national_fixtures.csv
# Error esperado: Fixture scope mismatch
```

## Limitaciones honestas

- Los datos sample son pequeños: no sirven para apostar dinero real.
- Los player props necesitan eventos reales y mucha calibración.
- Betfair API todavía no está conectada con autenticación real.
- Las alineaciones probables reales todavía no se descargan automáticamente.
- El módulo StatsBomb agrega eventos desde JSON, pero falta pipeline completo de competiciones/matches/eventos.

## Siguiente iteración recomendada

1. Descargar `international_results` real y entrenar modelo nacional amplio.
2. Descargar Football-Data.co.uk de varias temporadas y entrenar modelo de clubes amplio.
3. Crear backtest por temporada para clubes y leave-one-tournament-out para selecciones.
4. Convertir StatsBomb Open Data a player events reales.
5. Añadir paper ledger diario con cuotas manuales/API.
