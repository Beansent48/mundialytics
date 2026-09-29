# Team news experiments (2026-09-29)

Question: what else about *who plays* improves the 1X2, and how much of it can be known
before the lineups? Everything is scored the same way: the weekly-refit walk-forward
(`scripts/squad_value/wf_refit.py`) with the deployed squad-value shift as the base
(RPS 0.19941 on 10,403 matches, 2020/21–2025/26). Each signal enters as a total-goals-
preserving tilt, s = Σ θ_k (x_home,k − x_away,k), λh·e^{s/2}, λa·e^{−s/2}. Seasons are
left out one at a time (LOSO), and the bar is 5 of 6 seasons better with O/U 2.5 not worse.

```
python scripts/lineup_pass/build_tm_absence.py     # TM lineups + values -> work/tm_absence.csv
python scripts/lineup_pass/build_suspensions.py    # bans from cards, "still out" -> work/suspensions.csv
python scripts/elo_blend/build_clubelo_pre.py      # ClubElo the day before -> work/clubelo_pre.csv
python scripts/lineup_pass/eval_signals.py [SPEC ...] [--fit-all]
```

Data: the Kaggle Transfermarkt dump (git-ignored under `data/external/transfermarkt/`):
lineups with the whole matchday squad (starters and bench, 10,950 Big Five matches), card
events, and valuations read strictly before each match date.

## Results (dRPS per match, LOSO)

| Signal | dRPS | seasons | Known |
|---|---|---|---|
| Regulars not starting (the first deployed lineup pass, Understat) | −0.00052 | 6/6 | ~1h before |
| same, Transfermarkt lineups | −0.00055 | 6/6 | ~1h before |
| same, weighted by market value | −0.00069 | 5/6 | ~1h before |
| log value of today's XI vs the regular XI | −0.00032 | 4/6 | ~1h before |
| **Regulars not in the matchday squad, by value** (deployed) | **−0.00118** | **6/6** | ~1h before |
| same, by count | −0.00107 | 5/6 | ~1h before |
| Regulars on the bench | +0.00004 | 0/6 | ~1h before |
| **Out of the previous squad, no ban there or now** (deployed, morning) | **−0.00026** | **6/6** | days before |
| same, bans not excluded | −0.00020 | 5/6 | days before |
| out of the last two squads | −0.00012 | 4/6 | days before |
| Bans predicted from cards (count / value) | 0.00000 | 3/6, 4/6 | days before |
| ClubElo difference | −0.00001 | 4/6 | days before |
| ClubElo + the model's own stretch | −0.00004 | 2/6 | days before |
| ClubElo frozen at 1 August + stretch | +0.00011 | 0/6 | days before |

What it says:

- **The lineup signal was diluted by rotation.** A regular on the bench carries nothing:
  strong sides rest players when they can afford to, and the market already expects it.
  A regular missing from the whole squad is injured, suspended or out of favour, and
  that is the news. Once the squads are out, the morning signal adds nothing on top
  (θ −0.08 jointly).
- **The morning version recovers about a fifth of it.** "Out of the last squad, not for a
  ban" is right 60% of the time and covers 41% of the regulars who turn out to be
  missing. The ban exclusion matters: a banned player is back next week.
- **Bans on their own do nothing**, even though the rules reproduce what happened (83% of
  predicted bans are indeed out of the squad): one player out of eleven is a small tilt.
- **ClubElo adds nothing.** The engine already holds what a results-based rating knows.

Deployed constants, fitted on all six seasons: `lineup_shift` θ −0.592 (value share not in
the squad), `morning_shift` θ −0.3928 (share likely still out). Production replay against
Bet365 closing on 10,080 matches: 0.1983 → **0.1976** (Bet365 0.1946).

Production reads the same things from ESPN (the current season's rosters hold the whole
matchday squad and the cards) and values from the weekly live Transfermarkt read
(`data/processed/squad_player_values.csv`; 97.8% of the regulars matched by name, the rest
count at the team's median). `still_out` on Transfermarkt data reproduces the experiment's
signal on 97.2% of 2023/24 team-matches; the gap is how each source records cards.
