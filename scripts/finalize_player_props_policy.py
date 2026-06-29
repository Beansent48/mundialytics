from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.evaluation.player_prop_policy import build_player_prop_policy


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Create final player-prop calibration/safety policy from temporal validation reports.")
    p.add_argument("--temporal-report", required=True, help="calibration_temporal_check/temporal_calibration_report.json")
    p.add_argument("--hierarchical-report", default=None, help="Optional hierarchical_temporal_calibration_report.json")
    p.add_argument("--out", required=True, help="Output policy JSON")
    args = p.parse_args()
    temporal_path = _resolve(args.temporal_report)
    hier_path = _resolve(args.hierarchical_report)
    out_path = _resolve(args.out)
    assert temporal_path is not None and out_path is not None
    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    hierarchical = json.loads(hier_path.read_text(encoding="utf-8")) if hier_path and hier_path.exists() else None
    policy = build_player_prop_policy(temporal, hierarchical)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(policy, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
