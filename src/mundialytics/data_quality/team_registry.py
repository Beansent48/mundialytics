from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mundialytics.identity.normalization import canonical_team_name


TEAM_REGISTRY_VERSION = "v0.49.6_team_registry"

TEAM_REGISTRY_COLUMNS = [
    "canonical_team_id",
    "canonical_team_name",
    "team_scope",
    "country",
    "primary_competition",
    "football_data_name",
    "clubelo_name",
    "understat_name",
    "statsbomb_name",
    "active_from",
    "active_to",
    "alias_status",
    "notes",
]


# Small seed map for common provider naming differences. The generated registry
# is intentionally editable by the user; these defaults only reduce manual work.
PROVIDER_ALIAS_SEEDS: dict[str, dict[str, str]] = {
    "arsenal": {"clubelo_name": "Arsenal", "understat_name": "Arsenal"},
    "aston villa": {"clubelo_name": "Aston Villa", "understat_name": "Aston Villa"},
    "bournemouth": {"clubelo_name": "Bournemouth", "understat_name": "Bournemouth"},
    "brentford": {"clubelo_name": "Brentford", "understat_name": "Brentford"},
    "brighton": {"clubelo_name": "Brighton", "understat_name": "Brighton"},
    "chelsea": {"clubelo_name": "Chelsea", "understat_name": "Chelsea"},
    "crystal palace": {"clubelo_name": "Crystal Palace", "understat_name": "Crystal Palace"},
    "everton": {"clubelo_name": "Everton", "understat_name": "Everton"},
    "fulham": {"clubelo_name": "Fulham", "understat_name": "Fulham"},
    "leeds": {"clubelo_name": "Leeds", "understat_name": "Leeds"},
    "leicester": {"clubelo_name": "Leicester", "understat_name": "Leicester"},
    "liverpool": {"clubelo_name": "Liverpool", "understat_name": "Liverpool"},
    "man city": {"clubelo_name": "Man City", "understat_name": "Manchester City"},
    "man united": {"clubelo_name": "Man United", "understat_name": "Manchester United"},
    "newcastle": {"clubelo_name": "Newcastle", "understat_name": "Newcastle United"},
    "nottingham forest": {"clubelo_name": "Nottingham Forest", "understat_name": "Nottingham Forest"},
    "southampton": {"clubelo_name": "Southampton", "understat_name": "Southampton"},
    "tottenham": {"clubelo_name": "Tottenham", "understat_name": "Tottenham"},
    "west ham": {"clubelo_name": "West Ham", "understat_name": "West Ham"},
    "wolves": {"clubelo_name": "Wolves", "understat_name": "Wolverhampton Wanderers"},

    "ath bilbao": {"clubelo_name": "Athletic", "understat_name": "Athletic Club"},
    "athletic club": {"clubelo_name": "Athletic", "understat_name": "Athletic Club"},
    "ath madrid": {"clubelo_name": "Atletico", "understat_name": "Atletico Madrid"},
    "atletico madrid": {"clubelo_name": "Atletico", "understat_name": "Atletico Madrid"},
    "barcelona": {"clubelo_name": "Barcelona", "understat_name": "Barcelona"},
    "betis": {"clubelo_name": "Betis", "understat_name": "Real Betis"},
    "celta": {"clubelo_name": "Celta", "understat_name": "Celta Vigo"},
    "getafe": {"clubelo_name": "Getafe", "understat_name": "Getafe"},
    "girona": {"clubelo_name": "Girona", "understat_name": "Girona"},
    "mallorca": {"clubelo_name": "Mallorca", "understat_name": "Mallorca"},
    "osasuna": {"clubelo_name": "Osasuna", "understat_name": "Osasuna"},
    "real madrid": {"clubelo_name": "Real Madrid", "understat_name": "Real Madrid"},
    "sevilla": {"clubelo_name": "Sevilla", "understat_name": "Sevilla"},
    "sociedad": {"clubelo_name": "Real Sociedad", "understat_name": "Real Sociedad"},
    "valencia": {"clubelo_name": "Valencia", "understat_name": "Valencia"},
    "villareal": {"clubelo_name": "Villarreal", "understat_name": "Villarreal"},
    "villarreal": {"clubelo_name": "Villarreal", "understat_name": "Villarreal"},

    "ac milan": {"clubelo_name": "Milan", "understat_name": "AC Milan"},
    "atalanta": {"clubelo_name": "Atalanta", "understat_name": "Atalanta"},
    "bologna": {"clubelo_name": "Bologna", "understat_name": "Bologna"},
    "empoli": {"clubelo_name": "Empoli", "understat_name": "Empoli"},
    "fiorentina": {"clubelo_name": "Fiorentina", "understat_name": "Fiorentina"},
    "inter": {"clubelo_name": "Inter", "understat_name": "Inter"},
    "juventus": {"clubelo_name": "Juventus", "understat_name": "Juventus"},
    "lazio": {"clubelo_name": "Lazio", "understat_name": "Lazio"},
    "napoli": {"clubelo_name": "Napoli", "understat_name": "Napoli"},
    "roma": {"clubelo_name": "Roma", "understat_name": "Roma"},
    "sassuolo": {"clubelo_name": "Sassuolo", "understat_name": "Sassuolo"},
    "torino": {"clubelo_name": "Torino", "understat_name": "Torino"},
    "udinese": {"clubelo_name": "Udinese", "understat_name": "Udinese"},

    "augsburg": {"clubelo_name": "Augsburg", "understat_name": "Augsburg"},
    "bayern munich": {"clubelo_name": "Bayern", "understat_name": "Bayern Munich"},
    "dortmund": {"clubelo_name": "Dortmund", "understat_name": "Borussia Dortmund"},
    "ein frankfurt": {"clubelo_name": "Eintracht Frankfurt", "understat_name": "Eintracht Frankfurt"},
    "freiburg": {"clubelo_name": "Freiburg", "understat_name": "Freiburg"},
    "hoffenheim": {"clubelo_name": "Hoffenheim", "understat_name": "Hoffenheim"},
    "leverkusen": {"clubelo_name": "Leverkusen", "understat_name": "Bayer Leverkusen"},
    "mainz": {"clubelo_name": "Mainz", "understat_name": "Mainz 05"},
    "mgladbach": {"clubelo_name": "Gladbach", "understat_name": "Borussia M.Gladbach"},
    "rb leipzig": {"clubelo_name": "RB Leipzig", "understat_name": "RasenBallsport Leipzig"},
    "stuttgart": {"clubelo_name": "Stuttgart", "understat_name": "VfB Stuttgart"},
    "union berlin": {"clubelo_name": "Union Berlin", "understat_name": "Union Berlin"},
    "wolfsburg": {"clubelo_name": "Wolfsburg", "understat_name": "Wolfsburg"},

    "angers": {"clubelo_name": "Angers", "understat_name": "Angers"},
    "bordeaux": {"clubelo_name": "Bordeaux", "understat_name": "Bordeaux"},
    "lens": {"clubelo_name": "Lens", "understat_name": "Lens"},
    "lille": {"clubelo_name": "Lille", "understat_name": "Lille"},
    "lyon": {"clubelo_name": "Lyon", "understat_name": "Lyon"},
    "marseille": {"clubelo_name": "Marseille", "understat_name": "Marseille"},
    "monaco": {"clubelo_name": "Monaco", "understat_name": "Monaco"},
    "montpellier": {"clubelo_name": "Montpellier", "understat_name": "Montpellier"},
    "nantes": {"clubelo_name": "Nantes", "understat_name": "Nantes"},
    "nice": {"clubelo_name": "Nice", "understat_name": "Nice"},
    "paris sg": {"clubelo_name": "Paris SG", "understat_name": "Paris Saint Germain"},
    "psg": {"clubelo_name": "Paris SG", "understat_name": "Paris Saint Germain"},
    "rennes": {"clubelo_name": "Rennes", "understat_name": "Rennes"},
    "st etienne": {"clubelo_name": "Saint-Etienne", "understat_name": "Saint-Etienne"},
    "strasbourg": {"clubelo_name": "Strasbourg", "understat_name": "Strasbourg"},
    "toulouse": {"clubelo_name": "Toulouse", "understat_name": "Toulouse"},
}


@dataclass(frozen=True)
class TeamRegistryOutputs:
    registry: pd.DataFrame
    summary: dict[str, Any]


def normalize_provider_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def team_id_from_name(name: Any) -> str:
    norm = normalize_provider_name(name)
    return norm.replace(" ", "_") if norm else "unknown_team"


def title_from_canonical(name: str) -> str:
    return " ".join(part.capitalize() for part in str(name).split())


def _date_min_max(values: pd.Series) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.notna().any():
        return str(dates.min().date()), str(dates.max().date())
    return None, None


def build_team_registry(matches: pd.DataFrame, *, dataset_name: str = "team_registry") -> TeamRegistryOutputs:
    required = {"home_team", "away_team"}
    missing = sorted(required - set(matches.columns))
    if missing:
        summary = {
            "version": TEAM_REGISTRY_VERSION,
            "dataset_name": dataset_name,
            "status": "blocked",
            "missing_required_columns": missing,
            "input_rows": int(len(matches)),
            "registry_rows": 0,
        }
        return TeamRegistryOutputs(pd.DataFrame(columns=TEAM_REGISTRY_COLUMNS), summary)

    df = matches.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    for col in ["competition", "team_scope"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    rows: list[dict[str, Any]] = []
    teams = sorted(set(df["home_team"].dropna().astype(str)) | set(df["away_team"].dropna().astype(str)))
    for team in teams:
        canonical = canonical_team_name(team)
        team_filter = (df["home_team"].astype(str) == team) | (df["away_team"].astype(str) == team)
        g = df.loc[team_filter]
        active_from, active_to = _date_min_max(g["date"])
        competitions = g["competition"].dropna().astype(str)
        primary_competition = competitions.mode().iloc[0] if not competitions.empty else "unknown"
        scope_values = g["team_scope"].dropna().astype(str)
        team_scope = scope_values.mode().iloc[0] if not scope_values.empty else "unknown"

        aliases = PROVIDER_ALIAS_SEEDS.get(canonical, {})
        default_name = title_from_canonical(canonical)
        rows.append({
            "canonical_team_id": team_id_from_name(canonical),
            "canonical_team_name": default_name,
            "team_scope": team_scope,
            "country": "",
            "primary_competition": primary_competition,
            "football_data_name": team,
            "clubelo_name": aliases.get("clubelo_name", default_name),
            "understat_name": aliases.get("understat_name", default_name),
            "statsbomb_name": aliases.get("statsbomb_name", default_name),
            "active_from": active_from,
            "active_to": active_to,
            "alias_status": "seeded" if aliases else "generated_review_needed",
            "notes": "Review provider aliases before external joins." if not aliases else "",
        })

    registry = pd.DataFrame(rows, columns=TEAM_REGISTRY_COLUMNS)
    summary = {
        "version": TEAM_REGISTRY_VERSION,
        "dataset_name": dataset_name,
        "status": "ok",
        "input_rows": int(len(matches)),
        "registry_rows": int(len(registry)),
        "generated_review_needed": int((registry["alias_status"] == "generated_review_needed").sum()),
        "seeded_alias_rows": int((registry["alias_status"] == "seeded").sum()),
        "team_scopes": sorted(registry["team_scope"].dropna().astype(str).unique().tolist()),
        "principle": "team_registry_is_editable_source_of_truth_for_provider_aliases",
    }
    return TeamRegistryOutputs(registry, summary)


def load_team_registry(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in TEAM_REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[TEAM_REGISTRY_COLUMNS].copy()


def provider_alias_map(registry: pd.DataFrame, provider_column: str) -> dict[str, str]:
    if provider_column not in registry.columns:
        return {}
    mapping: dict[str, str] = {}
    for _, row in registry.iterrows():
        canonical = normalize_provider_name(row.get("football_data_name") or row.get("canonical_team_name"))
        provider = str(row.get(provider_column) or "").strip()
        if canonical and provider:
            mapping[canonical] = provider
        canonical_id = normalize_provider_name(row.get("canonical_team_id"))
        if canonical_id and provider:
            mapping[canonical_id] = provider
    return mapping
