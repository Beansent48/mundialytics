from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP_URL = "https://github.com/statsbomb/open-data/archive/refs/heads/master.zip"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _copytree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download or install StatsBomb Open Data into data/raw/statsbomb/open-data/data."
    )
    p.add_argument("--url", default=DEFAULT_ZIP_URL)
    p.add_argument("--zip", default=None, help="Optional local StatsBomb open-data zip already downloaded from GitHub.")
    p.add_argument("--out", default="data/raw/statsbomb/open-data/data", help="Output directory that should contain competitions.json, events/, matches/, lineups/...")
    p.add_argument("--keep-zip", default=None, help="Optional path to keep downloaded zip.")
    args = p.parse_args()

    out = _resolve(args.out)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = _resolve(args.zip) if args.zip else tmp / "statsbomb_open_data.zip"
        if args.zip is None:
            print(f"Downloading StatsBomb Open Data zip from {args.url} ...")
            try:
                urllib.request.urlretrieve(args.url, zip_path)
            except Exception as exc:
                raise SystemExit(
                    "Could not download StatsBomb Open Data. Download manually from "
                    "https://github.com/statsbomb/open-data/archive/refs/heads/master.zip "
                    f"and rerun with --zip path\\to\\master.zip. Original error: {exc}"
                ) from exc
        if args.keep_zip:
            keep = _resolve(args.keep_zip)
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zip_path, keep)
            print(f"Kept zip -> {keep}")
        extract_dir = tmp / "extract"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        candidates = list(extract_dir.glob("*/data")) + list(extract_dir.glob("data"))
        data_dir = next((c for c in candidates if (c / "events").exists() and (c / "matches").exists()), None)
        if data_dir is None:
            raise SystemExit("Could not find StatsBomb data/events and data/matches in extracted zip.")
        _copytree(data_dir, out)
    print(f"StatsBomb Open Data ready at: {out}")
    print("Next:")
    print("  python scripts/build_event_datasets.py statsbomb --input data/raw/statsbomb/open-data/data --team-scope club")
    print("  python scripts/diagnose_event_data.py --player-events data/processed/statsbomb_player_events.csv --lineups data/processed/statsbomb_lineups.csv --strict")


if __name__ == "__main__":
    main()
