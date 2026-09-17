# v0.36.1 Market-side pick policy audit

This release fixes pick-policy evaluation so that side-specific selections are treated as first-class candidates:

- goals_over and goals_under are evaluated separately.
- btts_yes and btts_no are evaluated separately.
- 1x2 home/draw/away can be evaluated separately.
- The global selected policy can still include both over and under when the learned policy allows `goals`.
- New outputs:
  - `market_selection_performance.csv`
  - `market_selection_threshold_performance.csv`
  - `pick_policy_best_by_signal_group.csv`
  - `pick_policy_best_by_signal_group.json`

Important: this remains signal-only unless historical bookmaker odds are supplied. It measures calibration/hit rate, not ROI.
