from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mundialytics.data.schema import infer_single_scope, validate_fixture_scope


@dataclass
class ModelBundle:
    """Versioned artifact for trained prediction models.

    Storing metadata next to the models prevents silent misuse: a club model
    should not predict national-team fixtures, and old data snapshots should be
    auditable.
    """

    goal_model: Any
    elo_rater: Any
    training_frame: pd.DataFrame
    model_scope: str
    model_type: str
    created_at_utc: str
    training_start_date: str | None
    training_end_date: str | None
    teams_seen: list[str] = field(default_factory=list)
    competitions_seen: list[str] = field(default_factory=list)
    data_source: str = "unknown"
    code_version: str = "0.8-operational-runtime-events"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict would recurse into sklearn objects poorly for display, but joblib
        # can still dump this dict. Keep objects explicit.
        d["goal_model"] = self.goal_model
        d["elo_rater"] = self.elo_rater
        d["training_frame"] = self.training_frame
        return d

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model_scope": self.model_scope,
            "model_type": self.model_type,
            "created_at_utc": self.created_at_utc,
            "training_start_date": self.training_start_date,
            "training_end_date": self.training_end_date,
            "n_training_rows": int(len(self.training_frame)),
            "n_teams_seen": int(len(self.teams_seen)),
            "competitions_seen": self.competitions_seen,
            "data_source": self.data_source,
            "code_version": self.code_version,
        }

    def validate_fixtures(self, fixtures: pd.DataFrame) -> None:
        validate_fixture_scope(fixtures, self.model_scope)


def _date_str(series: pd.Series, fn: str) -> str | None:
    if series.empty:
        return None
    dt = getattr(pd.to_datetime(series), fn)()
    return pd.Timestamp(dt).date().isoformat()


def create_model_bundle(goal_model: Any, elo_rater: Any, training_frame: pd.DataFrame, matches: pd.DataFrame, model_type: str, data_source: str = "unknown") -> ModelBundle:
    scope = infer_single_scope(matches)
    teams = sorted(set(matches["home_team"].astype(str)).union(matches["away_team"].astype(str)))
    competitions = sorted(set(matches.get("competition", pd.Series(["unknown"])).fillna("unknown").astype(str)))
    return ModelBundle(
        goal_model=goal_model,
        elo_rater=elo_rater,
        training_frame=training_frame,
        model_scope=scope,
        model_type=model_type,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_start_date=_date_str(matches["date"], "min"),
        training_end_date=_date_str(matches["date"], "max"),
        teams_seen=teams,
        competitions_seen=competitions,
        data_source=data_source,
    )


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.to_dict(), path)
    return path


def load_model_bundle(path: str | Path) -> ModelBundle:
    raw = joblib.load(path)
    if isinstance(raw, ModelBundle):
        return raw
    # Backward compatibility with old dict bundles.
    if isinstance(raw, dict):
        metadata = raw.get("metadata", {})
        return ModelBundle(
            goal_model=raw["goal_model"],
            elo_rater=raw["elo_rater"],
            training_frame=raw["training_frame"],
            model_scope=raw.get("model_scope") or metadata.get("model_scope", "unknown"),
            model_type=raw.get("model_type") or metadata.get("model_type", "unknown"),
            created_at_utc=raw.get("created_at_utc") or metadata.get("created_at_utc", "unknown"),
            training_start_date=raw.get("training_start_date") or metadata.get("training_start_date"),
            training_end_date=raw.get("training_end_date") or metadata.get("training_end_date"),
            teams_seen=raw.get("teams_seen") or metadata.get("teams_seen", []),
            competitions_seen=raw.get("competitions_seen") or metadata.get("competitions_seen", []),
            data_source=raw.get("data_source") or metadata.get("data_source", "unknown"),
            code_version=raw.get("code_version") or metadata.get("code_version", "legacy"),
        )
    raise TypeError(f"Unsupported bundle type: {type(raw)!r}")
