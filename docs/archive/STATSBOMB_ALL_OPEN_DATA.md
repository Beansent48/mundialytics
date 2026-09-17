# StatsBomb Open Data: descarga completa

Para entrenar corners y paradas reales por portero desde StatsBomb raw events, no hace falta elegir una competición manualmente.

## Dry run

```powershell
python scripts/download_statsbomb_open_data_events.py `
  --all-competitions `
  --dry-run `
  --out-dir data/raw/statsbomb/events
```

## Descargar todo

```powershell
python scripts/download_statsbomb_open_data_events.py `
  --all-competitions `
  --skip-existing `
  --sleep 0.10 `
  --out-dir data/raw/statsbomb/events
```

Atajo equivalente:

```powershell
python scripts/download_all_statsbomb_open_data.py `
  --out-dir data/raw/statsbomb/events `
  --sleep 0.10
```

## Construir estadísticas reales desde raw events

```powershell
python scripts/build_statsbomb_raw_extra_stats.py `
  --event-json-dir data/raw/statsbomb/events `
  --out data/processed/statsbomb_raw_extra_match_stats.csv

python scripts/build_statsbomb_raw_goalkeeper_stats.py `
  --event-json-dir data/raw/statsbomb/events `
  --out data/processed/goalkeeper_match_stats.csv
```

## Combinar con Football-Data

```powershell
python scripts/combine_team_match_market_stats.py `
  --csv data/processed/team_match_market_stats.csv `
  --csv data/processed/statsbomb_raw_extra_match_stats.csv `
  --out data/processed/team_match_market_stats_combined.csv
```

## Backtest de líneas

```powershell
python scripts/build_event_line_backtest.py `
  --team-match-stats data/processed/team_match_market_stats_combined.csv `
  --goalkeeper-match-stats data/processed/goalkeeper_match_stats.csv `
  --out-dir outputs/event_line_backtest_current
```

Notas:

- Usa `--skip-existing` para reanudar si se corta.
- Usa `--max-matches 50` solo para pruebas rápidas.
- Usa `--min-season-year 2010` si quieres excluir temporadas antiguas del raw event set.
- No mezcles competiciones sin conservar contexto: los CSVs mantienen `competition`, `season`, `data_source` y `data_quality_flag` para que el entrenamiento pueda segmentar.
