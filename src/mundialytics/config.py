from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config.

    Parameters
    ----------
    path:
        YAML path. If omitted, loads config/default.yaml from project root.
    """
    cfg_path = Path(path) if path else PROJECT_ROOT / "config" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    sample: Path = PROJECT_ROOT / "data" / "sample"
    outputs: Path = PROJECT_ROOT / "outputs"


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
