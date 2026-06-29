# v0.34.2 Console Pick-Oriented Viewer

This version updates `scripts/show_matchday_console.py` so the console view shows both over and under candidates by default.

Important distinction:

- Statistical signal: model probability, fair odds and evidence support a side of a line.
- Real pick: statistical probability is compared with bookmaker odds and validated through paper tracking/backtesting.

Without odds, the console output is a shortlist of candidates to inspect. It is not a real betting recommendation.

## Useful commands

```powershell
python scripts/show_matchday_console.py `
  --out-dir outputs/statistical_matchday_today `
  --team curacao `
  --team ecuador `
  --team tunisia `
  --team japan `
  --top-lines 12 `
  --top-players 12
```

Show only over player props:

```powershell
python scripts/show_matchday_console.py `
  --out-dir outputs/statistical_matchday_today `
  --team japan `
  --overs-only-player-props
```

Show extreme high-probability lines too:

```powershell
python scripts/show_matchday_console.py `
  --out-dir outputs/statistical_matchday_today `
  --team japan `
  --include-extreme-lines
```

Tune thresholds:

```powershell
python scripts/show_matchday_console.py `
  --out-dir outputs/statistical_matchday_today `
  --team japan `
  --min-line-prob 0.55 `
  --min-player-prob 0.50
```
