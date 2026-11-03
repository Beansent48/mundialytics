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
