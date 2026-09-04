# European fixture coverage

The European competition layer prices a match from the two clubs' ClubElo
ratings, so a fixture is unusable when either side has no ClubElo history on
disk. This note records what the gap actually is, because the first diagnosis
was wrong.

Run the audit any time:

```bash
python scripts/audit_european_elo_coverage.py
```

## Where it stands

| | |
|---|---|
| Club appearances in the UEFA fixture files | 3,450 |
| Priceable — ClubElo history held | 3,022 (87.6%) |
| Blocked | 428, across 46 clubs |
| Fixtures with **both** sides priceable | 84% |

## It is not an alias problem

The obvious first guess — that the blocked clubs are spelling mismatches
between the fixture source and ClubElo, fixable with a mapping — is wrong. It
was checked against the 220 club histories actually on disk, and **none of the
46 blocked clubs is present under any spelling**. Slavia Praha, Qarabag,
Ludogorets, AZ Alkmaar, APOEL, FCSB, Pafos, Omonia, Lech Poznan and the rest are
simply not downloaded.

Fuzzy matching is what makes this trap dangerous. Asked for near-matches it
proposes `Union SG → Udinese`, `Ludogorets → Lorient`, `Athletic Club →
Atletico`, `Pafos → PAOK`, `Omonoia → Monza`. Every one of those is a different
club, and accepting them would price real European fixtures with another team's
rating — silently, and worse than dropping the match. **Do not auto-map by
string distance here.**

There *was* one genuine resolution bug, but on our side: the evaluation script
pre-normalised club names before handing them to `make_resolver`, which strips
the spaces its alias table is keyed on. Passing raw names lifted fixture
coverage from 80% to 84% and improved the held-out RPS from 0.2104 to 0.2044.
`make_resolver` normalises internally — always give it the raw name.

## Fixing it

31 of the 46 blocked clubs appear in the latest ClubElo daily snapshot, so for
those the fix is downloading their history. The remaining 15 are not in the
snapshot either and need checking against ClubElo's own naming.

This is blocked on the upstream API:

```
$ curl -s -o /dev/null -w "%{http_code}" http://api.clubelo.com/Slavia
502
```

That is the same outage that prompted the local ClubElo roll-forward
(`src/mundialytics/ratings/clubelo_local.py`). When the API recovers,
`scripts/download_clubelo.py --mode team-history` takes a matches CSV with
`date`, `home_team` and `away_team` columns; the UEFA files under
`data/external/uefa/` need reshaping to that schema first, since they ship as
`Date` / `Home Team` / `Away Team` with a combined `Result`.

Re-run the audit afterwards to see coverage move, then
`scripts/evaluate_european_layer.py` to re-score on the larger sample.

## Why the blocked clubs skew small

They are mostly champions of smaller associations — Cyprus, Israel, Bulgaria,
Czechia, the Nordics, the Caucasus — which is exactly the population the Swiss
league phase added. That biases the measured sample toward matches between clubs
big enough to be in the downloaded set, so the reported RPS is measured on a
slightly easier, more predictable slice than the full competition.
