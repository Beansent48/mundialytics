#!/usr/bin/env python3
"""Find and remove duplicate player identities created by
scripts/merge_fbref_recent_players.py: StatsBomb uses full legal names
("Kylian Mbappé Lottin") while FBref uses common display names ("Kylian
Mbappé"), so a player active in both eras ended up with TWO separate rows
in player_profiles_with_positions.csv instead of one.

Matching strategy (no fuzzy-string-distance library needed -- exact token
containment is enough here since the two naming conventions differ by
DROPPED tokens, not typos):
  1. Exact lookup against data/identity/player_display_names.csv (58
     hand-curated legal-name -> display-name pairs) -- highest confidence,
     checked first.
  2. Token-subset matching: normalize both names (lowercase, strip accents),
     split into tokens, and treat an FBref name as a match for a StatsBomb
     name if EVERY token in the (shorter) FBref name appears in the
     (longer) StatsBomb name's token set. Candidates are narrowed via an
     inverted token index (only StatsBomb players sharing at least one
     token are considered) rather than an O(n*m) full scan.
  3. Disambiguation: if more than one StatsBomb candidate matches an FBref
     name's tokens, prefer one whose canonical team matches the FBref
     player's team; if that still doesn't resolve to exactly one, skip
     (leave both rows rather than risk merging two different real people).

Resolution: when a confident match is found, the ORIGINAL StatsBomb row is
kept as-is (it has real defensive-quality/creation stats the FBref-only row
lacks) and the duplicate FBref-sourced row is dropped -- this is an identity
fix, not a stat blend; see the module docstring in
merge_fbref_recent_players.py for why blending the two stat sets isn't
attempted here.

Run after merge_fbref_recent_players.py:
    python scripts/reconcile_fbref_duplicate_identities.py
"""
from __future__ import annotations

import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.identity.normalization import canonical_team_name

CAREER_PATH = ROOT / "data/processed/player_profiles_with_positions.csv"
DISPLAY_NAMES_PATH = ROOT / "data/identity/player_display_names.csv"
N_ORIGINAL_ROWS = 9885  # row count before merge_fbref_recent_players.py appended its 4674 new rows


def _normalize(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower()


def _tokens(name: str) -> frozenset[str]:
    return frozenset(_normalize(name).split())


def main() -> None:
    df = pd.read_csv(CAREER_PATH)
    if len(df) <= N_ORIGINAL_ROWS:
        print(f"File has {len(df)} rows, expected > {N_ORIGINAL_ROWS} -- "
              "has merge_fbref_recent_players.py already run? Aborting.")
        return

    original = df.iloc[:N_ORIGINAL_ROWS].copy()
    fbref_added = df.iloc[N_ORIGINAL_ROWS:].copy()
    print(f"Original (StatsBomb-era) rows: {len(original)}")
    print(f"FBref-added rows to check for duplicates: {len(fbref_added)}")

    # Pass 1: exact lookup via the hand-curated display-name map.
    display_map = {}
    if DISPLAY_NAMES_PATH.exists():
        dn = pd.read_csv(DISPLAY_NAMES_PATH)
        display_map = dict(zip(dn["display_name"].str.lower(), dn["player"]))

    original_names = set(original["player"])
    original_team_by_name = dict(zip(original["player"], original["team"].map(canonical_team_name)))

    # Pass 2: token-subset index over original (StatsBomb) player names.
    token_index: dict[str, list[str]] = defaultdict(list)
    original_tokens: dict[str, frozenset[str]] = {}
    for player in original_names:
        toks = _tokens(player)
        original_tokens[player] = toks
        for t in toks:
            token_index[t].append(player)

    matched_rows = []
    ambiguous = []
    unmatched_count = 0

    for _, row in fbref_added.iterrows():
        fb_name = row["player"]
        fb_team = canonical_team_name(row["team"])
        match: str | None = None

        exact = display_map.get(fb_name.lower())
        if exact and exact in original_names:
            match = exact
        else:
            fb_toks = _tokens(fb_name)
            candidate_lists = [set(token_index[t]) for t in fb_toks if t in token_index]
            candidates = set.intersection(*candidate_lists) if candidate_lists else set()
            candidates = {c for c in candidates if fb_toks.issubset(original_tokens[c])}
            if len(candidates) == 1:
                match = next(iter(candidates))
            elif len(candidates) > 1:
                team_matches = [c for c in candidates if original_team_by_name.get(c) == fb_team]
                if len(team_matches) == 1:
                    match = team_matches[0]
                else:
                    ambiguous.append((fb_name, sorted(candidates)))

        if match:
            matched_rows.append((fb_name, match))
        else:
            unmatched_count += 1

    matched_fb_names = {fb for fb, _ in matched_rows}
    kept_fbref = fbref_added[~fbref_added["player"].isin(matched_fb_names)]

    print(f"\nResolved as duplicates (dropped, original StatsBomb row kept): {len(matched_rows)}")
    for fb_name, sb_name in matched_rows[:20]:
        print(f"  {fb_name!r} -> {sb_name!r}")
    if len(matched_rows) > 20:
        print(f"  ... and {len(matched_rows) - 20} more")

    print(f"\nAmbiguous (multiple candidates, left unmerged): {len(ambiguous)}")
    for fb_name, cands in ambiguous[:10]:
        print(f"  {fb_name!r} -> candidates: {cands}")

    print(f"\nGenuinely new players kept: {len(kept_fbref)}")

    result = pd.concat([original, kept_fbref], ignore_index=True)
    result.to_csv(CAREER_PATH, index=False)
    print(f"\nWrote {len(result)} rows -> {CAREER_PATH} (was {len(df)})")


if __name__ == "__main__":
    main()
