# Bookmaker-compatible event rules

This project separates three concepts:

1. **official provider statistic**: exactly the value used by the bookmaker/provider.
2. **bookmaker-compatible reconstruction**: a transparent rule built from open event data to mimic common settlement rules.
3. **proxy**: a weaker approximation. Proxies must not be marketed as official betting stats.

## Shots on target

For bookmaker-style player shots-on-target markets, use the Opta-like rule:

Count as shot on target:
- goal;
- shot saved by the goalkeeper that was on target;
- last-line defensive block that prevents a goal when no goalkeeper/defender remains behind the blocker, if the source allows this to be identified.

Do not count:
- ordinary blocked shots with defenders/goalkeeper still behind the blocker;
- shots hitting the post/bar unless they go in and are awarded as a goal;
- crosses/passes unless the provider tags them as deliberate shots.

Open-data mapping used by the adapters:

- StatsBomb: `Shot` with outcome `Goal`, `Saved`, or `Saved to Post`. Ordinary `Blocked` is not counted because open StatsBomb events alone do not reliably distinguish last-line blocks.
- Wyscout public data: `Shot` with goal tag `101` or accurate tag `1801`. This is `source_defined_sot_like`; it should be validated against the exact competition/provider before real-money use.

## Fouls and cards

- StatsBomb: `Foul Committed`, `Foul Won`, card field under `foul_committed.card`.
- Wyscout: `Foul` is treated as committed by the event player; yellow/red cards are inferred from card tags. Fouls drawn are not always directly recoverable from the public event row, so the adapter does not invent them.

## Sustituto+

Sustituto+/Safe Sub style markets are handled as a market adjustment, not as a raw event stat. The model needs:

- projected starter;
- expected minutes;
- substitution probability/minute;
- likely replacement;
- event probability for the original player and replacement.

The current formula is a transparent first-order approximation:

`P(original OR replacement) = P(original) + P(not original) * P(replacement enters) * P(replacement hits line)`

Keep this in paper mode until the exact operator market rules are confirmed for the selected market.
