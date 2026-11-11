"""Copy the measured serving bundle somewhere, preserving its paths.

`scripts/measure_serving_bundle.py` records which files the API opens while
every endpoint is called. This takes that manifest and makes the directory a
deployment actually mounts -- 227 MB of the 26.5 GB tree as measured on
2026-11-02 -- so seeding a server, or testing that the bundle is *sufficient*,
is one command:

    python scripts/measure_serving_bundle.py --out outputs/serving_bundle
    python scripts/build_serving_bundle.py --to outputs/bundle_data
    DATA_DIR=outputs/bundle_data docker compose up -d --wait api

That last line is the test the bundle exists for: the stack serving from nothing
but the measured files is what proves the list is complete. A missing file shows
up as a 500 on the page that needs it, not as a silent fallback.

Only `data/` is copied. The read set also contains `api/*.py`, `src/**` and
`web/data/*`, which are code and travel in the images, and
`data/processed/cache/` is skipped on purpose: the fitted-engine joblib is keyed
on a hash of the engine source, and the container's copy of that source is not
byte-identical to a Windows checkout's, so a copied cache would never be hit.
The container pays one cold fit on its first boot instead -- which is what the
healthcheck's `start_period` is for.

    --to DIR        where to write (required)
    --manifest P    manifest.json from the measure script
    --clean         delete DIR first
    --dry-run       print the plan and the size, copy nothing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Read while serving but deliberately left out of a bundle.
SKIP_PREFIXES = ("data/processed/cache/",)

# Needed to serve correctly and NOT in the read set, which is a distinction the
# first version of this script got wrong. Found on 2026-11-03 by running the
# container: the API came up and printed "market level unavailable, lambdas left
# alone", which is the fail-soft branch, so it served every match with no market
# shift at all -- probabilities quietly different from the deployed ones, out of
# a bundle that looked complete.
#
# The chain: `api/engine._market_levels` memoises the level on disk under
# `data/processed/cache/` (excluded above, for good reason -- its key includes a
# hash of the engine source, which differs in a container), and its key is built
# by *stat-ing* every football-data CSV through a glob. A glob is `os.scandir`,
# not `open`, so with the memo warm the measurement neither opened those files
# nor needed them. In the container the memo missed, the level recomputed, and
# its inputs were not there.
#
# 41 MB and 201 files, against 194.5 MB of measured reads. The measurement
# script now reports directories it listed without opening (`walked` in the
# manifest) so the next gap of this shape shows up before a container does.
ALWAYS_INCLUDE = (
    "data/raw/football_data",
    # And the same shape of gap one layer along, found the same day and
    # corroborated by a parallel session that measured the bundle *cold*
    # (439 files / 356 MB, branch `claude/settlement-audit`): with the props
    # joblib warm, the files that BUILT it are never opened.
    #
    # `understat_shots.csv` is 121 MB and `PlayerPropsModel.fit` takes it as
    #   `if shots is None and shots_path is not None and Path(shots_path).exists()`
    # — so its absence is tolerated in silence, and the penalty information it
    # carries (the pen-taker split) simply is not in the model. Verified in the
    # container on 2026-11-03: a cold refit without it returned a working
    # `PlayerPropsModel`, every price plausible, the feature gone. The 2026-10-21
    # plan called this file a props-fit input and it was right; the warm
    # measurement is what said otherwise.
    "data/external/advanced/understat/understat_shots.csv",
)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--to", type=Path, required=True, help="destination directory")
    ap.add_argument("--manifest", type=Path,
                    default=Path("outputs/serving_bundle/manifest.json"))
    ap.add_argument("--clean", action="store_true", help="delete the destination first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if not manifest.exists():
        print(f"no manifest at {manifest}\n"
              f"run: python scripts/measure_serving_bundle.py --out "
              f"{manifest.parent.relative_to(ROOT) if manifest.is_relative_to(ROOT) else manifest.parent}",
              file=sys.stderr)
        return 1
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("failures"):
        print("warning: the manifest records failures, so the read set may be "
              "incomplete:", file=sys.stderr)
        for f in data["failures"]:
            print(f"  {f}", file=sys.stderr)

    wanted = [rel for rel in data.get("files", {})
              if rel.startswith("data/") and not rel.startswith(SKIP_PREFIXES)]
    for extra in ALWAYS_INCLUDE:
        base = ROOT / extra
        # Entries are directories or single files. The first version only
        # handled directories and skipped `understat_shots.csv` with a warning
        # that read like a missing-file problem -- so the gap it was added to
        # close stayed open for another cold start.
        if base.is_file():
            wanted.append(base.relative_to(ROOT).as_posix())
        elif base.is_dir():
            wanted += [f.relative_to(ROOT).as_posix() for f in sorted(base.rglob("*"))
                       if f.is_file()]
        else:
            print(f"warning: {extra} is not in this checkout", file=sys.stderr)
    wanted = sorted(set(wanted))
    missing = [rel for rel in wanted if not (ROOT / rel).is_file()]
    total = sum((ROOT / rel).stat().st_size for rel in wanted if (ROOT / rel).is_file())

    dest = args.to if args.to.is_absolute() else ROOT / args.to
    print(f"{len(wanted)} files, {human(total)}  ->  {dest}")
    if missing:
        print(f"{len(missing)} are missing from this checkout:", file=sys.stderr)
        for rel in missing[:10]:
            print(f"  {rel}", file=sys.stderr)
    if args.dry_run:
        return 1 if missing else 0

    if args.clean and dest.exists():
        shutil.rmtree(dest)
    copied = 0
    for rel in wanted:
        src = ROOT / rel
        if not src.is_file():
            continue
        # `data/` is stripped: the destination IS the data directory, so it can
        # be handed straight to DATA_DIR or rsynced onto a server's volume.
        out = dest / Path(rel).relative_to("data")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        copied += 1
    print(f"copied {copied} files")

    # The two places the serving path writes that are not in the read set: the
    # fit cache and the run logs. Created empty so a read-only source tree does
    # not become a container that cannot start.
    for d in ("processed/cache", "processed/logs/runs", "processed/competition_cache"):
        (dest / d).mkdir(parents=True, exist_ok=True)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
