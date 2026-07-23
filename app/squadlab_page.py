"""
SquadLab page — shown inside the main Streamlit app.
Two modes:
  Draft    : pick a competition, your club replaces the league's current
             last-place team, draft 11 players from 5 candidates per slot
             (each pick is removed from the global pool — no duplicates,
             no player available to a club after being drafted away),
             then play a full season match-by-match (results, standings,
             top scorer/assist board) plus a Monte Carlo odds layer.
  Sandbox  : pick any 11 players freely, same season + Monte Carlo engine
             against a reference competition's real clubs.
"""
from __future__ import annotations
import hashlib
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mundialytics.identity.display_names import display_name as _disp
from mundialytics.statistical_core.player_strength import PlayerStrengthModel
from mundialytics.statistical_core.schemas import canonical_name
from mundialytics.statistical_core.squadlab.calendar import generate_double_round_robin
from mundialytics.statistical_core.squadlab.lambda_source import RealTeamLambdaSource, SeasonLambdaSource
from mundialytics.statistical_core.squadlab.season_simulator import (
    MatchResult, SeasonOrchestrator, SeasonResult, table_through_matchday,
)
from mundialytics.statistical_core.squadlab.squad_lambda_model import SquadLambdaModel

SQUAD_TEAM_NAME = "Tu Equipo"

# Capped per explicit user request ("para no petar mucho") — no configurable
# up-to-1M option anymore, 100k is fast enough (~seconds) to just always run.
MC_N_SIMS = 100_000
LIVE_HALF_TICKS = 10
LIVE_TICK_SECONDS = 1.0

POSITIONS_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

POSITION_SLOTS = {
    "Goalkeeper": 1,
    "Defender":   4,
    "Midfielder": 3,
    "Forward":    3,
}

# Player-profiles CSV and match-results CSV spell competition names differently.
PLAYER_COMP_MAP = {
    "LaLiga":         "La Liga",
    "Premier League": "Premier League",
    "Serie A":        "Serie A",
    "Bundesliga":     "1. Bundesliga",
    "Ligue 1":        "Ligue 1",
}
MATCH_COMP_MAP = {
    "LaLiga":         "LaLiga",
    "Premier League": "Premier League",
    "Serie A":        "Serie A",
    "Bundesliga":     "Bundesliga",
    "Ligue 1":        "Ligue 1",
}
DRAFT_COMPETITIONS = list(PLAYER_COMP_MAP.keys())

# Pitch coordinates (% of width/height) per formation, attacking toward y=0.
FORMATIONS = {
    "4-3-3": {
        "Goalkeeper": [(50, 92)],
        "Defender":   [(15, 72), (38, 76), (62, 76), (85, 72)],
        "Midfielder": [(25, 50), (50, 45), (75, 50)],
        "Forward":    [(20, 18), (50, 12), (80, 18)],
    },
    "4-4-2": {
        "Goalkeeper": [(50, 92)],
        "Defender":   [(15, 72), (38, 76), (62, 76), (85, 72)],
        "Midfielder": [(15, 48), (38, 44), (62, 44), (85, 48)],
        "Forward":    [(35, 16), (65, 16)],
    },
    "4-2-3-1": {
        "Goalkeeper": [(50, 92)],
        "Defender":   [(15, 72), (38, 76), (62, 76), (85, 72)],
        "Midfielder": [(35, 58), (65, 58), (15, 38), (50, 34), (85, 38)],
        "Forward":    [(50, 14)],
    },
    "3-5-2": {
        "Goalkeeper": [(50, 92)],
        "Defender":   [(25, 75), (50, 80), (75, 75)],
        "Midfielder": [(10, 50), (30, 42), (50, 38), (70, 42), (90, 50)],
        "Forward":    [(35, 16), (65, 16)],
    },
}
DEFAULT_FORMATION = "4-3-3"


@st.cache_resource(show_spinner="Cargando perfiles de jugadores...")
def load_strength_model() -> PlayerStrengthModel:
    m = PlayerStrengthModel()
    m.fit()
    return m


def render_player_card(p, compact: bool = False) -> str:
    """HTML card for a player."""
    off_bar = int(p.offensive_strength * 0.9)
    def_bar = int(p.defensive_strength * 0.9)
    ov_color = "#16a34a" if p.overall >= 70 else ("#2563eb" if p.overall >= 50 else "#9ca3af")
    if compact:
        return (
            f'<div style="background:var(--secondary-background-color);border-radius:8px;padding:8px 10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-weight:500;font-size:12px">{p.player.split()[0]}</span>'
            f'<span style="font-weight:700;color:{ov_color};font-size:14px">{p.overall:.0f}</span>'
            f'</div>'
            f'<div style="font-size:10px;color:#9ca3af">{p.team.title()} · {p.position[:3]}</div>'
            f'</div>'
        )
    return (
        f'<div style="background:var(--secondary-background-color);border-radius:10px;padding:12px 14px;'
        f'margin-bottom:8px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        f'<div>'
        f'<div style="font-weight:500;font-size:13px">{p.player}</div>'
        f'<div style="font-size:11px;color:#9ca3af">{p.team.title()} · {p.competition}</div>'
        f'</div>'
        f'<div style="font-size:22px;font-weight:700;color:{ov_color}">{p.overall:.0f}</div>'
        f'</div>'
        f'<div style="margin-top:8px">'
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<span style="font-size:10px;color:#9ca3af;width:44px">Ataque</span>'
        f'<div style="flex:1;background:#e5e7eb33;border-radius:2px;height:8px">'
        f'<div style="width:{off_bar}%;height:8px;background:#3b82f6;border-radius:2px"></div>'
        f'</div><span style="font-size:10px;width:28px;text-align:right">{p.offensive_strength:.0f}</span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<span style="font-size:10px;color:#9ca3af;width:44px">Defensa</span>'
        f'<div style="flex:1;background:#e5e7eb33;border-radius:2px;height:8px">'
        f'<div style="width:{def_bar}%;height:8px;background:#10b981;border-radius:2px"></div>'
        f'</div><span style="font-size:10px;width:28px;text-align:right">{p.defensive_strength:.0f}</span>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;gap:10px;margin-top:8px;font-size:11px;color:#9ca3af">'
        f'<span>xG {p.xg_per_match:.2f}</span>'
        f'<span>Goles {p.goals_per_match:.2f}</span>'
        f'<span>Tackl {p.tackles_per_match:.2f}</span>'
        f'<span>n={p.matches}</span>'
        f'</div>'
        f'</div>'
    )


def team_strength_visual(strength: dict, team_name: str) -> go.Figure:
    categories = ["Ataque", "Defensa", "xG base"]
    vals = [strength["attack_index"], strength["defense_index"],
            min(100, strength["xg_per_match"] / 3.0 * 100)]
    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    fig = go.Figure(go.Bar(
        x=vals, y=categories, orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=f"Fuerza del equipo — {team_name}",
        height=180, margin=dict(l=70, r=50, t=40, b=10),
        xaxis=dict(range=[0, 110], showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_season_orchestrator(
    model: PlayerStrengthModel, engine, squad: list, real_teams: list[str],
    competition: str, squad_team_name: str = SQUAD_TEAM_NAME,
) -> SeasonOrchestrator:
    """Wires a drafted/sandbox squad into the same Poisson/Dixon-Coles
    machinery real teams already use (see squadlab/lambda_source.py) and
    builds a full double round-robin calendar against the given real
    opponents. Reused by both Draft and Sandbox — this IS the mechanism
    that makes Sandbox's Monte Carlo layer "fall out almost for free" once
    the Draft narrative engine exists."""
    real_source = RealTeamLambdaSource(engine)
    bridge = SquadLambdaModel(model)
    lambda_source = SeasonLambdaSource(squad_team_name, squad, bridge, real_source, engine.ad_model_)
    fixtures = generate_double_round_robin([squad_team_name] + real_teams)
    # FULL-LEAGUE narrative: give every real club a best-XI too, so every match
    # (not just ours) produces scorers/assists/cards/ratings -> a real league-wide
    # top-scorer race. The orchestrator already keys rosters by team.
    rosters: dict[str, list] = {squad_team_name: squad}
    # signed players leave their original club (the draft's own rule) — without
    # this a player sits in TWO rosters and his goals are double-counted
    rosters.update(build_real_team_rosters(model, real_teams,
                                           exclude={p.player for p in squad}))
    return SeasonOrchestrator(
        lambda_source, fixtures, squad_roster=rosters, competition=competition,
    )


def build_real_team_rosters(model: PlayerStrengthModel, real_teams: list[str],
                            exclude: set[str] | None = None) -> dict[str, list]:
    """Best XI (1-4-3-3 shape) per real club from the player pool, so the whole
    league gets player-level events. `exclude` drops players already signed by
    the user's squad — a player must never appear in two rosters (that would
    double-count his goals). Clubs with too little coverage are skipped (their
    matches simply keep scoreline-only detail, as before)."""
    exclude = exclude or set()
    by_team: dict[str, list] = {}
    for p in model.profiles_.values():
        if p.player in exclude:
            continue
        by_team.setdefault(canonical_name(p.team), []).append(p)
    slots = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}
    out: dict[str, list] = {}
    for team in real_teams:
        pool = by_team.get(canonical_name(team), [])
        if not pool:
            continue
        xi: list = []
        for pos, n in slots.items():
            cands = sorted([p for p in pool if p.position == pos],
                           key=lambda p: -p.overall)[:n]
            xi += cands
        if len(xi) >= 8:          # enough to attribute events meaningfully
            out[team] = xi
    return out


def render_standings_table(table_df: pd.DataFrame, squad_team_name: str = SQUAD_TEAM_NAME,
                           title: str = "📊 Clasificación") -> None:
    st.markdown(f"#### {title}")
    table_display = table_df.copy()
    table_display.insert(0, "pos", range(1, len(table_display) + 1))
    table_display["team"] = table_display["team"].apply(
        lambda t: f"⭐ {t}" if t == squad_team_name else t.title()
    )
    st.dataframe(table_display.rename(columns={
        "pos": "#", "team": "Equipo", "played": "PJ", "pts": "Pts",
        "gf": "GF", "ga": "GC", "gd": "DG",
    }), use_container_width=True, hide_index=True)


def render_season_result(result: SeasonResult, squad_team_name: str = SQUAD_TEAM_NAME) -> None:
    render_standings_table(result.table, squad_team_name, title="📊 Clasificación final")

    if not result.player_season_tallies.empty:
        st.markdown("#### ⚽ Máximos goleadores de tu plantilla")
        tallies = result.player_season_tallies.head(10).copy()
        st.dataframe(tallies.rename(columns={
            "player": "Jugador", "position": "Pos", "goals": "Goles",
            "assists": "Asist.", "yellow_cards": "TA", "matches": "PJ", "avg_rating": "Rating medio",
        })[["Jugador", "Pos", "Goles", "Asist.", "TA", "PJ", "Rating medio"]],
        use_container_width=True, hide_index=True)

    squad_matches = [m for m in result.matches if m.home == squad_team_name or m.away == squad_team_name]
    with st.expander(f"Ver los {len(squad_matches)} partidos de tu equipo"):
        for m in squad_matches:
            home_label = "⭐ " + m.home if m.home == squad_team_name else m.home.title()
            away_label = "⭐ " + m.away if m.away == squad_team_name else m.away.title()
            st.markdown(f"J{m.matchday}: {home_label} **{m.home_goals}-{m.away_goals}** {away_label}")


def render_monte_carlo_result(mc: pd.DataFrame, n_sims: int, squad_team_name: str = SQUAD_TEAM_NAME) -> None:
    squad_row = mc[mc["team"] == squad_team_name]
    if not squad_row.empty:
        r = squad_row.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("% Campeón", f"{r['p_champion']:.1%}")
        c2.metric("% Top 4", f"{r['p_top4']:.1%}")
        c3.metric("% Descenso", f"{r['p_relegation']:.1%}")
        c4.metric("Puntos medios", f"{r['avg_pts']:.1f}")
    st.caption(f"Basado en {n_sims:,} simulaciones Monte Carlo (mismo motor que la temporada narrativa).")
    mc_display = mc.copy()
    mc_display["team"] = mc_display["team"].apply(lambda t: f"⭐ {t}" if t == squad_team_name else t.title())
    for col in ["p_champion", "p_top2", "p_top4", "p_relegation"]:
        mc_display[col] = (mc_display[col] * 100).round(1)
    st.dataframe(mc_display.rename(columns={
        "team": "Equipo", "p_champion": "% Campeón", "p_top2": "% Top 2", "p_top4": "% Top 4",
        "p_relegation": "% Descenso", "avg_pts": "Pts medios", "avg_goals": "Goles medios",
    }), use_container_width=True, hide_index=True)


# ── Background Monte Carlo (runs while the user watches the live season) ────

def _run_monte_carlo_background(orchestrator: SeasonOrchestrator, holder: dict, n_sims: int) -> None:
    try:
        holder["result"] = orchestrator.run_monte_carlo(n_sims=n_sims)
    except Exception as exc:  # surfaced in the UI via holder["error"], not swallowed
        holder["error"] = str(exc)
    finally:
        holder["done"] = True


def start_monte_carlo_background(orchestrator: SeasonOrchestrator, n_sims: int = MC_N_SIMS) -> dict:
    """Kicks off run_monte_carlo() on a background thread so the user can
    watch the live matchday-by-matchday playback instead of staring at a
    spinner. The orchestrator's lambda cache is already warm by the time
    this is called (play_once() just ran), so the thread only does
    read-only array/model access — no shared-state mutation races with the
    main thread."""
    holder: dict = {"done": False, "result": None, "error": None}
    thread = threading.Thread(target=_run_monte_carlo_background, args=(orchestrator, holder, n_sims), daemon=True)
    thread.start()
    holder["thread"] = thread
    return holder


def render_monte_carlo_status(holder: dict | None, n_sims: int = MC_N_SIMS,
                              squad_team_name: str = SQUAD_TEAM_NAME) -> None:
    st.markdown("### 🎲 Probabilidades (Monte Carlo)")
    if holder is None:
        return
    if not holder["done"]:
        st.info(f"⏳ Calculando {n_sims:,} simulaciones en segundo plano — aparecerán solas "
               "al pasar de jornada (o pulsa cualquier botón para comprobar).")
        return
    if holder.get("error"):
        st.error(f"Error calculando probabilidades: {holder['error']}")
        return
    render_monte_carlo_result(holder["result"], n_sims, squad_team_name)


# ── Live matchday playback ───────────────────────────────────────────────────

def _stat_grid_html(events: list[tuple]) -> str:
    boxes = []
    for row in events:
        label, hv, av = row[0], row[1], row[2]
        dec = 2 if (len(row) > 3 and row[3]) else 0
        boxes.append(
            '<div style="flex:1;text-align:center;background:var(--secondary-background-color);'
            'border-radius:8px;padding:6px 4px">'
            f'<div style="font-size:10px;color:#9ca3af">{label}</div>'
            f'<div style="font-weight:700;font-size:1.0rem">{hv:.{dec}f} '
            f'<span style="color:#9ca3af;font-weight:400;font-size:.8rem">–</span> {av:.{dec}f}</div>'
            '</div>'
        )
    return f'<div style="display:flex;gap:6px;margin-top:6px">{"".join(boxes)}</div>'


def render_other_results(other_matches: list[MatchResult], squad_team_name: str = SQUAD_TEAM_NAME) -> None:
    st.markdown("#### 📰 Otros resultados de la jornada")
    for m in other_matches:
        st.markdown(f"{m.home.title()} **{m.home_goals}-{m.away_goals}** {m.away.title()}")


def play_live_match(match: MatchResult, squad_team_name: str = SQUAD_TEAM_NAME,
                    picks: dict | None = None, coords: dict | None = None) -> None:
    """Blocks for ~21s (two 10-tick, 10s halves + a half-time beat),
    progressively revealing the already-simulated result: a minute clock,
    goal/card events surfacing at pseudo-random minutes, and stats growing
    toward their true final values. Nothing here changes the result —
    it's a presentation-layer replay of what SeasonOrchestrator already
    computed, the same way a video game "simulates" a match by animating a
    pre-determined outcome.
    """
    squad_is_home = match.home == squad_team_name
    home_label = ("⭐ " + match.home) if squad_is_home else match.home.title()
    away_label = match.away.title() if squad_is_home else ("⭐ " + match.away)

    rng = np.random.default_rng(abs(hash((match.matchday, match.home, match.away))) % (2**32))
    timeline: list[tuple[int, str, str]] = []  # (minute, side, description)

    def _add_goals(goal_events: list[tuple[str, str | None]] | None, side: str) -> None:
        for scorer, assister in (goal_events or []):
            minute = int(rng.integers(1, 91))
            desc = f"⚽ Gol de {_disp(scorer)}" + (f" (asist. {_disp(assister)})" if assister else "")
            timeline.append((minute, side, desc))

    def _add_cards(card_players: list[str] | None, side: str) -> None:
        for player in (card_players or []):
            minute = int(rng.integers(1, 91))
            timeline.append((minute, side, f"🟨 Amarilla a {_disp(player)}"))

    _add_goals(match.home_goal_events, "home")
    _add_goals(match.away_goal_events, "away")
    _add_cards(match.home_card_players, "home")
    _add_cards(match.away_card_players, "away")

    # Real-team opponents have no player-level attribution (out of scope —
    # see season_simulator.py) — represent their goals generically so the
    # scoreline still updates live, without inventing a scorer's name.
    tracked_home_goals = sum(1 for _, side, d in timeline if side == "home" and d.startswith("⚽"))
    tracked_away_goals = sum(1 for _, side, d in timeline if side == "away" and d.startswith("⚽"))
    for _ in range(match.home_goals - tracked_home_goals):
        timeline.append((int(rng.integers(1, 91)), "home", f"⚽ Gol de {match.home.title()}"))
    for _ in range(match.away_goals - tracked_away_goals):
        timeline.append((int(rng.integers(1, 91)), "away", f"⚽ Gol de {match.away.title()}"))
    timeline.sort(key=lambda x: x[0])

    score_ph = st.empty()
    clock_ph = st.empty()
    feed_ph = st.empty()
    stats_ph = st.empty()

    home_score = away_score = 0
    revealed: list[str] = []
    idx = 0

    def _reveal_up_to(virtual_minute: int) -> None:
        nonlocal idx, home_score, away_score
        while idx < len(timeline) and timeline[idx][0] <= virtual_minute:
            minute, side, desc = timeline[idx]
            if desc.startswith("⚽"):
                if side == "home":
                    home_score += 1
                else:
                    away_score += 1
            revealed.append(f"{minute}' {desc}")
            idx += 1

    def _render(minute_label: str, fraction: float) -> None:
        score_ph.markdown(f"### {home_label}&nbsp;&nbsp;**{home_score} - {away_score}**&nbsp;&nbsp;{away_label}")
        clock_ph.markdown(f"**⏱️ Minuto {minute_label}**")
        feed_ph.markdown("<br>".join(reversed(revealed[-6:])) or "_Sin novedades todavía..._",
                         unsafe_allow_html=True)
        stats_ph.markdown(_stat_grid_html([
            ("xG",        match.home_xg * fraction,           match.away_xg * fraction, True),
            ("Disparos",  match.home_shots * fraction,        match.away_shots * fraction),
            ("A puerta",  match.home_sot * fraction,          match.away_sot * fraction),
            ("Córners",   match.home_corners * fraction,      match.away_corners * fraction),
            ("Amarillas", match.home_yellow_cards * fraction, match.away_yellow_cards * fraction),
        ]), unsafe_allow_html=True)

    for tick in range(LIVE_HALF_TICKS):
        minute = min(45, int((tick + 1) * 45 / LIVE_HALF_TICKS))
        _reveal_up_to(minute)
        _render(f"{minute}'", minute / 90)
        time.sleep(LIVE_TICK_SECONDS)

    score_ph.markdown(f"### {home_label}&nbsp;&nbsp;**{home_score} - {away_score}**&nbsp;&nbsp;{away_label}")
    clock_ph.info("🟨 Descanso")
    time.sleep(LIVE_TICK_SECONDS)

    for tick in range(LIVE_HALF_TICKS):
        minute = min(90, 45 + int((tick + 1) * 45 / LIVE_HALF_TICKS))
        _reveal_up_to(minute)
        _render(f"{minute}'", minute / 90)
        time.sleep(LIVE_TICK_SECONDS)

    clock_ph.markdown("**⏱️ Final del partido**")

    # ── post-match Sofascore-style rating pitch ────────────────────────────────
    squad_events = match.home_events if squad_is_home else match.away_events
    if squad_events and picks and coords:
        st.markdown("#### 📋 Valoraciones del partido")
        st.markdown(match_pitch_svg(squad_events, picks, coords), unsafe_allow_html=True)
        best = max(squad_events.values(), key=lambda e: e.rating)
        st.caption(f"⭐ MVP: **{_disp(best.player)}** ({best.rating:.1f}) · "
                   "valoración 0-10 por goles, asistencias, portería a cero y goles encajados. "
                   f"xG del partido: {(match.home_xg if squad_is_home else match.away_xg):.2f}.")


def render_matchday_summary(season_result: SeasonResult, matchday: int,
                            squad_team_name: str = SQUAD_TEAM_NAME) -> None:
    matchday_matches = [m for m in season_result.matches if m.matchday == matchday]
    squad_match = next(m for m in matchday_matches if m.home == squad_team_name or m.away == squad_team_name)
    other_matches = [m for m in matchday_matches if m is not squad_match]

    home_label = f"⭐ {squad_match.home}" if squad_match.home == squad_team_name else squad_match.home.title()
    away_label = f"⭐ {squad_match.away}" if squad_match.away == squad_team_name else squad_match.away.title()
    st.markdown(f"### {home_label} {squad_match.home_goals} - {squad_match.away_goals} {away_label}")
    st.markdown("#### 📈 Estadísticas del partido")
    st.markdown(_stat_grid_html([
        ("Disparos",  squad_match.home_shots,        squad_match.away_shots),
        ("A puerta",  squad_match.home_sot,          squad_match.away_sot),
        ("Córners",   squad_match.home_corners,      squad_match.away_corners),
        ("Amarillas", squad_match.home_yellow_cards, squad_match.away_yellow_cards),
    ]), unsafe_allow_html=True)

    if other_matches:
        render_other_results(other_matches, squad_team_name)

    table_so_far = table_through_matchday(season_result.matches, matchday)
    render_standings_table(table_so_far, squad_team_name, title=f"Clasificación tras la jornada {matchday}")


# ── Draft mode helpers ──────────────────────────────────────────────────────
def compute_standings(df_clubs: pd.DataFrame, comp_id: str, season: str) -> pd.DataFrame:
    """Current league table from played matches (works mid-season too)."""
    mask = (df_clubs["competition"] == comp_id) & (df_clubs["season"] == season)
    df = df_clubs[mask].dropna(subset=["home_goals", "away_goals"])
    if df.empty:
        return pd.DataFrame()

    rows: dict[str, dict] = {}
    for _, m in df.iterrows():
        h, a, hg, ag = m["home_team"], m["away_team"], m["home_goals"], m["away_goals"]
        for t in (h, a):
            rows.setdefault(t, {"team": t, "played": 0, "pts": 0, "gf": 0, "ga": 0})
        rows[h]["played"] += 1; rows[a]["played"] += 1
        rows[h]["gf"] += hg; rows[h]["ga"] += ag
        rows[a]["gf"] += ag; rows[a]["ga"] += hg
        if hg > ag: rows[h]["pts"] += 3
        elif hg < ag: rows[a]["pts"] += 3
        else: rows[h]["pts"] += 1; rows[a]["pts"] += 1

    table = pd.DataFrame(rows.values())
    table["gd"] = table["gf"] - table["ga"]
    return table.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)


POS_ABBR = {"Goalkeeper": "POR", "Defender": "DEF", "Midfielder": "MED", "Forward": "DEL"}

# Card tiers by our own overall rating. `variant` allows future special editions
# (award cards, memorable-match cards) to reuse the same renderer.
CARD_TIERS = {
    "elite":  {"bg": ("#1b1147", "#3b1d6e"), "ink": "#ffd75e", "line": "#ffd75e", "sub": "#e9d8ff"},
    "gold":   {"bg": ("#6b5411", "#c9a227"), "ink": "#fff6d0", "line": "#ffe9a3", "sub": "#f3e3ae"},
    "silver": {"bg": ("#3b4450", "#8c98a6"), "ink": "#ffffff", "line": "#dfe6ee", "sub": "#e3e9f0"},
    "bronze": {"bg": ("#4a2f1b", "#a97142"), "ink": "#ffeeda", "line": "#f0cba5", "sub": "#f0d7bd"},
}


def _card_tier(overall: float, variant: str | None = None) -> dict:
    if variant and variant in CARD_TIERS:
        return CARD_TIERS[variant]
    if overall >= 90:
        return CARD_TIERS["elite"]
    if overall >= 85:
        return CARD_TIERS["gold"]
    if overall >= 78:
        return CARD_TIERS["silver"]
    return CARD_TIERS["bronze"]


def player_card_svg(p, selected: bool = False, variant: str | None = None,
                    width: int = 150) -> str:
    """FUT-style player card using OUR ratings only (overall + role + ATA/DEF),
    never FIFA-style pace/shooting/passing splits."""
    w, h = 150, 214
    t = _card_tier(p.overall, variant)
    uid = abs(hash((p.player, variant or "", selected))) % 10**7
    is_gk = p.position == "Goalkeeper"
    # GKs: our defensive axis for them lives in gk_strength
    off = p.offensive_strength
    dfn = p.gk_strength if (is_gk and getattr(p, "gk_strength", 0)) else p.defensive_strength
    glow = ('filter="url(#glow%d)"' % uid) if selected else ""
    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
         f'style="width:100%;max-width:{width}px;display:block;margin:0 auto">',
         '<defs>',
         f'<linearGradient id="g{uid}" x1="0" y1="0" x2="0.4" y2="1">'
         f'<stop offset="0%" stop-color="{t["bg"][1]}"/>'
         f'<stop offset="100%" stop-color="{t["bg"][0]}"/></linearGradient>',
         f'<filter id="glow{uid}" x="-30%" y="-30%" width="160%" height="160%">'
         f'<feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#3b82f6" flood-opacity="0.95"/>'
         f'</filter>', '</defs>']
    # card body (hexagon-ish FUT silhouette via rounded rect + notched top)
    s.append(f'<path d="M8 14 Q8 4 20 4 L130 4 Q142 4 142 14 L142 176 '
             f'Q142 190 128 196 L79 210 Q75 211 71 210 L22 196 Q8 190 8 176 Z" '
             f'fill="url(#g{uid})" stroke="{t["line"]}" stroke-width="{3 if selected else 1.5}" '
             f'{glow}/>')
    # overall + position + role (left column)
    s.append(f'<text x="30" y="46" text-anchor="middle" font-size="30" font-weight="800" '
             f'fill="{t["ink"]}">{p.overall:.0f}</text>')
    s.append(f'<text x="30" y="62" text-anchor="middle" font-size="11" font-weight="700" '
             f'fill="{t["sub"]}">{POS_ABBR.get(p.position, "—")}</text>')
    role = (getattr(p, "role", "") or "").upper()[:9]
    if role:
        s.append(f'<text x="30" y="77" text-anchor="middle" font-size="7.5" font-weight="600" '
                 f'fill="{t["sub"]}" opacity="0.85">{role}</text>')
    s.append(f'<line x1="52" y1="26" x2="52" y2="78" stroke="{t["line"]}" stroke-width="1" opacity=".5"/>')
    # crest-ish emblem (no photos available -> typographic monogram)
    s.append(f'<circle cx="99" cy="52" r="26" fill="#00000033" stroke="{t["line"]}" '
             f'stroke-width="1" opacity=".8"/>')
    s.append(f'<text x="99" y="62" text-anchor="middle" font-size="24" font-weight="800" '
             f'fill="{t["ink"]}" opacity=".92">{_disp(p.player)[:1].upper()}</text>')
    # name
    s.append(f'<text x="75" y="112" text-anchor="middle" font-size="13" font-weight="800" '
             f'fill="{t["ink"]}" letter-spacing="0.5">{_disp(p.player)[:14].upper()}</text>')
    s.append(f'<line x1="28" y1="122" x2="122" y2="122" stroke="{t["line"]}" stroke-width="1" opacity=".55"/>')
    # OUR two stats only
    s.append(f'<text x="52" y="146" text-anchor="middle" font-size="18" font-weight="800" '
             f'fill="{t["ink"]}">{off:.0f}</text>')
    s.append(f'<text x="52" y="159" text-anchor="middle" font-size="8" font-weight="700" '
             f'fill="{t["sub"]}">ATAQUE</text>')
    s.append(f'<text x="98" y="146" text-anchor="middle" font-size="18" font-weight="800" '
             f'fill="{t["ink"]}">{dfn:.0f}</text>')
    s.append(f'<text x="98" y="159" text-anchor="middle" font-size="8" font-weight="700" '
             f'fill="{t["sub"]}">{"PARADAS" if is_gk else "DEFENSA"}</text>')
    # team
    s.append(f'<text x="75" y="181" text-anchor="middle" font-size="8.5" font-weight="600" '
             f'fill="{t["sub"]}" opacity=".9">{str(p.team).title()[:18]}</text>')
    s.append('</svg>')
    return "".join(s)


def _rating_color(r: float) -> str:
    """Sofascore-style rating color: red (poor) -> amber -> green (great)."""
    if r >= 8.5:
        return "#137a3c"
    if r >= 7.5:
        return "#22a94f"
    if r >= 7.0:
        return "#63b544"
    if r >= 6.5:
        return "#c99a1e"
    if r >= 6.0:
        return "#dd8a2f"
    return "#d64545"


def match_pitch_svg(events: dict, picks: dict, coords: dict) -> str:
    """Sofascore-style post-match pitch: each player positioned by formation,
    with a colored match-rating badge and goal/assist/card icons."""
    w, h = 470, 620
    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
         f'style="width:100%;max-width:470px;display:block;margin:0 auto">']
    s += _pitch_base_svg(w, h)

    for pos, pos_coords in coords.items():
        for slot, (xp, yp) in enumerate(pos_coords):
            profile = picks.get(f"{pos}_{slot}")
            if not profile:
                continue
            cx, cy = xp / 100 * w, yp / 100 * h
            ev = events.get(profile.player)
            rating = getattr(ev, "rating", 6.5) if ev else 6.5
            rc = _rating_color(rating)
            # player disc
            s.append(f'<circle cx="{cx}" cy="{cy}" r="19" fill="#0b1220" stroke="#e2e8f0" stroke-width="1.5"/>')
            s.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="12" '
                     f'font-weight="700" fill="#e2e8f0">{_disp(profile.player)[:3].upper()}</text>')
            # rating badge (bottom-right of disc)
            bx, by = cx + 9, cy + 9
            s.append(f'<rect x="{bx-1}" y="{by-1}" width="30" height="17" rx="4" fill="{rc}" '
                     f'stroke="#0b1220" stroke-width="1"/>')
            s.append(f'<text x="{bx+14}" y="{by+12}" text-anchor="middle" font-size="11" '
                     f'font-weight="800" fill="white">{rating:.1f}</text>')
            # event icons (top-right of disc): goals, assist, card
            icons = ""
            if ev:
                icons += "⚽" * min(getattr(ev, "goals", 0), 3)
                if getattr(ev, "assists", 0):
                    icons += "🅰️"
                if getattr(ev, "yellow_cards", 0):
                    icons += "🟨"
            if icons:
                s.append(f'<text x="{cx-11}" y="{cy-13}" text-anchor="end" font-size="12">{icons}</text>')
            # name below
            s.append(f'<text x="{cx}" y="{cy+34}" text-anchor="middle" font-size="11" '
                     f'font-weight="600" fill="white" style="text-shadow:0 1px 3px #000000cc">'
                     f'{_disp(profile.player)[:13]}</text>')
    s.append('</svg>')
    return "".join(s)


def _pitch_base_svg(w: int, h: int) -> list[str]:
    """Shared, properly-marked football pitch (mowing stripes, boxes, arcs)."""
    L = "#ffffff70"
    s = [f'<defs><linearGradient id="turf" x1="0" y1="0" x2="0" y2="1">'
         f'<stop offset="0%" stop-color="#1a7f42"/><stop offset="100%" stop-color="#125f30"/>'
         f'</linearGradient>'
         f'<radialGradient id="vig" cx="50%" cy="45%" r="75%">'
         f'<stop offset="60%" stop-color="#00000000"/><stop offset="100%" stop-color="#00000055"/>'
         f'</radialGradient></defs>',
         f'<rect width="{w}" height="{h}" fill="url(#turf)" rx="14"/>']
    for i in range(9):                                    # mowing stripes
        if i % 2 == 0:
            s.append(f'<rect x="0" y="{i*h/9:.1f}" width="{w}" height="{h/9:.1f}" fill="#ffffff0a"/>')
    m = 12
    s.append(f'<rect x="{m}" y="{m}" width="{w-2*m}" height="{h-2*m}" fill="none" stroke="{L}" stroke-width="2" rx="2"/>')
    s.append(f'<line x1="{m}" y1="{h/2}" x2="{w-m}" y2="{h/2}" stroke="{L}" stroke-width="2"/>')
    s.append(f'<circle cx="{w/2}" cy="{h/2}" r="54" fill="none" stroke="{L}" stroke-width="2"/>')
    s.append(f'<circle cx="{w/2}" cy="{h/2}" r="3" fill="{L}"/>')
    for top in (True, False):
        by = m if top else h - m - 66            # 18-yard box
        s.append(f'<rect x="{w/2-92}" y="{by}" width="184" height="66" fill="none" stroke="{L}" stroke-width="2"/>')
        sy = m if top else h - m - 26            # 6-yard box
        s.append(f'<rect x="{w/2-46}" y="{sy}" width="92" height="26" fill="none" stroke="{L}" stroke-width="2"/>')
        py = m + 46 if top else h - m - 46       # penalty spot
        s.append(f'<circle cx="{w/2}" cy="{py}" r="2.5" fill="{L}"/>')
        # penalty arc (drawn outside the box)
        ay = m + 66 if top else h - m - 66
        sweep = 1 if top else 0
        s.append(f'<path d="M {w/2-30} {ay} A 32 32 0 0 {sweep} {w/2+30} {ay}" fill="none" stroke="{L}" stroke-width="2"/>')
    for cx0, cy0, sw in ((m, m, 1), (w-m, m, 0), (m, h-m, 0), (w-m, h-m, 1)):   # corner arcs
        s.append(f'<path d="M {cx0} {cy0+(9 if cy0==m else -9)} A 9 9 0 0 {sw} {cx0+(9 if cx0==m else -9)} {cy0}" '
                 f'fill="none" stroke="{L}" stroke-width="1.5"/>')
    s.append(f'<rect width="{w}" height="{h}" fill="url(#vig)" rx="14" pointer-events="none"/>')
    return s


def pitch_svg(picks: dict, coords: dict) -> str:
    """Draft pitch: each filled slot shows a mini player card (our rating)."""
    w, h = 470, 620
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
           f'style="width:100%;max-width:470px;display:block;margin:0 auto">']
    svg += _pitch_base_svg(w, h)
    for pos, pos_coords in coords.items():
        for slot, (xp, yp) in enumerate(pos_coords):
            profile = picks.get(f"{pos}_{slot}")
            cx, cy = xp / 100 * w, yp / 100 * h
            if not profile:
                svg.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="#00000040" stroke="#ffffff77" '
                           f'stroke-width="1.5" stroke-dasharray="4 3"/>')
                svg.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="15" '
                           f'fill="#ffffffaa">+</text>')
                continue
            t = _card_tier(profile.overall)
            # mini card chip
            svg.append(f'<rect x="{cx-20}" y="{cy-24}" width="40" height="46" rx="6" '
                       f'fill="{t["bg"][1]}" stroke="{t["line"]}" stroke-width="1.5" opacity="0.97"/>')
            svg.append(f'<text x="{cx}" y="{cy-6}" text-anchor="middle" font-size="16" '
                       f'font-weight="800" fill="{t["ink"]}">{profile.overall:.0f}</text>')
            svg.append(f'<text x="{cx}" y="{cy+7}" text-anchor="middle" font-size="7" '
                       f'font-weight="700" fill="{t["sub"]}">{POS_ABBR.get(profile.position, "")}</text>')
            svg.append(f'<text x="{cx}" y="{cy+18}" text-anchor="middle" font-size="6.5" '
                       f'fill="{t["sub"]}" opacity=".9">{(getattr(profile,"role","") or "")[:10].upper()}</text>')
            svg.append(f'<text x="{cx}" y="{cy+36}" text-anchor="middle" font-size="10.5" '
                       f'font-weight="700" fill="white" style="text-shadow:0 1px 3px #000000cc">'
                       f'{_disp(profile.player)[:12]}</text>')
    svg.append('</svg>')
    return "".join(svg)


def seeded_candidates(pool: list, pos: str, seed_str: str, excluded: set,
                      current: str | None, model: PlayerStrengthModel,
                      n: int = 5, shortlist_size: int = 15) -> list:
    """Pick a random-but-stable sample of n candidates for one slot.

    Drawing from a wider shortlist (top `shortlist_size` by overall) and
    seeding the RNG per slot/reroll means each position slot — including
    siblings like Defender #1..#4 — gets a different set of 5 instead of
    always showing the same top-5 overall players.
    """
    cands = [p for p in pool if p.position == pos and p.player not in excluded and p.matches >= 3]
    cands = sorted(cands, key=lambda p: -p.overall)[:shortlist_size]
    if len(cands) <= n:
        sample = cands
    else:
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        sample = random.Random(seed).sample(cands, n)
    sample = sorted(sample, key=lambda p: -p.overall)
    if current:
        cp = model.get(current)
        if cp and cp.player not in [s.player for s in sample]:
            sample = [cp] + sample[:n - 1]
    return sample


def next_empty_slot(coords: dict, picks: dict) -> str | None:
    for pos in POSITIONS_ORDER:
        for slot in range(len(coords.get(pos, []))):
            key = f"{pos}_{slot}"
            if key not in picks:
                return key
    return None


def render_draft_mode(model: PlayerStrengthModel, df_clubs: pd.DataFrame, engine=None):
    st.markdown("### 🎮 Draft de temporada")
    st.caption("Elige una competición: tu club sustituye al colista actual. Cada jugador fichado deja "
              "de estar disponible para su equipo de origen — sin duplicados. Pulsa una posición para "
              "ver candidatos.")

    col_comp, col_form = st.columns([2, 1])
    with col_comp:
        comp_d = st.selectbox("Competición", DRAFT_COMPETITIONS, key="draft_comp_v2")
    with col_form:
        formation_name = st.selectbox("Formación", list(FORMATIONS.keys()), key="draft_formation")
    coords = FORMATIONS[formation_name]

    # Reset draft state on competition or formation change
    active_key = (comp_d, formation_name)
    if st.session_state.get("draft_active_comp") != active_key:
        st.session_state.draft_picks = {}
        st.session_state.draft_excluded = set()
        st.session_state.draft_active_slot = None
        st.session_state.draft_reroll = {}
        st.session_state.draft_token = {}
        st.session_state.draft_active_comp = active_key
    st.session_state.setdefault("draft_token", {})

    comp_match_id = MATCH_COMP_MAP[comp_d]
    seasons = sorted(df_clubs.loc[df_clubs["competition"] == comp_match_id, "season"].unique(), reverse=True)
    if not seasons:
        st.warning("Sin datos de calendario para esta competición.")
        return
    season_d = seasons[0]
    table = compute_standings(df_clubs, comp_match_id, season_d)
    if table.empty:
        st.warning("Sin partidos disputados todavía esta temporada.")
        return

    last_row = table.iloc[-1]
    last_team = last_row["team"]
    st.info(f"🔻 Tu club sustituye a **{last_team.title()}**, colista de {comp_d} {season_d} "
           f"({int(last_row['pts'])} pts en {int(last_row['played'])} partidos).")

    player_comp = PLAYER_COMP_MAP[comp_d]
    pool = [p for p in model.profiles_.values() if p.competition == player_comp]
    if not pool:
        st.warning(f"No hay jugadores con perfil para {comp_d} en los datos disponibles.")
        return

    picks_resolved = {k: model.get(v) for k, v in st.session_state.draft_picks.items() if model.get(v)}

    col_pitch, col_pick = st.columns([2, 3])

    with col_pitch:
        st.markdown(pitch_svg(picks_resolved, coords), unsafe_allow_html=True)
        n_picked = len(picks_resolved)
        st.markdown(f"<div style='text-align:center;font-weight:600;margin-top:6px'>"
                   f"{n_picked}/11 fichados</div>", unsafe_allow_html=True)
        if picks_resolved:
            strength = model.team_strength(list(picks_resolved.values()))
            st.plotly_chart(team_strength_visual(strength, "Tu equipo"), use_container_width=True)
        if n_picked == 11:
            st.success("Plantilla completa. Simula la temporada más abajo.")
            if st.button("🔄 Reiniciar draft", key="draft_reset"):
                st.session_state.draft_picks = {}
                st.session_state.draft_excluded = set()
                st.session_state.draft_active_slot = None
                st.session_state.draft_reroll = {}
                for k in ("draft_orch_key", "draft_orchestrator", "draft_season_result",
                         "draft_playback_matchday", "draft_watched_matchdays",
                         "draft_playback_done", "draft_mc_holder"):
                    st.session_state.pop(k, None)
                st.rerun()

    with col_pick:
        st.caption("El número en cada candidato es su **valoración global (0-100)**: una nota absoluta de "
                  "ataque y defensa calculada a partir de sus estadísticas reales por partido (goles, "
                  "asistencias, xG, entradas, presiones...), no un ranking relativo a otros jugadores.")

        st.markdown("**Elige una posición:**")
        for pos in POSITIONS_ORDER:
            pos_coords = coords.get(pos, [])
            if not pos_coords:
                continue
            st.markdown(f"<div style='font-size:11px;color:#9ca3af;margin-top:6px'>{pos}</div>",
                       unsafe_allow_html=True)
            slot_cols = st.columns(len(pos_coords))
            for slot, sc in enumerate(slot_cols):
                key = f"{pos}_{slot}"
                current = st.session_state.draft_picks.get(key)
                is_active = st.session_state.draft_active_slot == key
                label = (_disp(current)[:10] if current else f"#{slot + 1}")
                if sc.button(("🟢 " if is_active else ("✅ " if current else "")) + label,
                           key=f"slotbtn_{comp_d}_{formation_name}_{key}",
                           use_container_width=True):
                    st.session_state.draft_active_slot = None if is_active else key
                    if not is_active:   # opening a slot -> fresh random candidate draw
                        st.session_state.draft_token[key] = random.randrange(2**31)
                    st.rerun()

        active_slot = st.session_state.draft_active_slot
        if active_slot:
            pos = active_slot.rsplit("_", 1)[0]
            current = st.session_state.draft_picks.get(active_slot)
            # genuine per-view randomness: a token generated when the slot is
            # opened (and bumped on reroll), not a seed tied to slot identity
            token = st.session_state.draft_token.setdefault(active_slot, random.randrange(2**31))
            seed_str = f"{token}"
            candidates = seeded_candidates(pool, pos, seed_str, st.session_state.draft_excluded,
                                          current, model)

            st.markdown("---")
            st.markdown(f"<div style='text-align:center;font-size:1.1rem;font-weight:800;"
                        f"letter-spacing:.06em;color:#d7e3f4'>ELIGE UN JUGADOR</div>"
                        f"<div style='text-align:center;font-size:.75rem;color:#8b9bb4;"
                        f"margin-bottom:6px'>{pos} · slot #{int(active_slot.rsplit('_',1)[1]) + 1}</div>",
                        unsafe_allow_html=True)
            if not candidates:
                st.caption("Sin candidatos disponibles en esta competición.")
            else:
                cand_cols = st.columns(len(candidates))
                for cc, cand in zip(cand_cols, candidates):
                    is_current = current == cand.player
                    cc.markdown(player_card_svg(cand, selected=is_current), unsafe_allow_html=True)
                    if cc.button("Elegido" if is_current else "Elegir",
                                key=f"pick_{comp_d}_{formation_name}_{active_slot}_{cand.player}",
                                disabled=is_current):
                        if current:
                            st.session_state.draft_excluded.discard(current)
                        st.session_state.draft_picks[active_slot] = cand.player
                        st.session_state.draft_excluded.add(cand.player)
                        st.session_state.draft_active_slot = next_empty_slot(coords, st.session_state.draft_picks)
                        st.rerun()
                if st.button("🎲 Ver otros 5 candidatos", key=f"reroll_{comp_d}_{formation_name}_{active_slot}"):
                    st.session_state.draft_token[active_slot] = random.randrange(2**31)
                    st.rerun()
        else:
            st.caption("👆 Pulsa una posición arriba para ver sus 5 candidatos.")

    if len(picks_resolved) < 11:
        return

    st.markdown("---")
    st.markdown("### 🏆 Temporada en directo")
    if engine is None:
        st.warning("Motor de predicción no disponible; no se puede simular la temporada.")
        return

    squad_11 = list(picks_resolved.values())
    real_opponents = [canonical_name(t) for t in table["team"].tolist() if t != last_team]
    squad_key = (comp_d, formation_name, tuple(sorted(st.session_state.draft_picks.items())))

    if st.session_state.get("draft_orch_key") != squad_key:
        st.session_state.draft_orch_key = squad_key
        st.session_state.draft_orchestrator = _build_season_orchestrator(
            model, engine, squad_11, real_opponents, competition=comp_match_id,
        )
        for k in ("draft_season_result", "draft_playback_matchday", "draft_watched_matchdays",
                 "draft_playback_done", "draft_mc_holder"):
            st.session_state.pop(k, None)

    orchestrator: SeasonOrchestrator = st.session_state.draft_orchestrator

    if "draft_season_result" not in st.session_state:
        st.caption("Juega tu jornada a jornada, en directo — mientras tanto calculamos "
                  f"{MC_N_SIMS:,} simulaciones Monte Carlo en segundo plano.")
        if st.button("🎬 Empezar temporada en directo", key="draft_start_live", use_container_width=True):
            with st.spinner("Preparando la temporada..."):
                st.session_state.draft_season_result = orchestrator.play_once(narrative=True)
            st.session_state.draft_playback_matchday = 1
            st.session_state.draft_watched_matchdays = set()
            st.session_state.draft_playback_done = False
            st.session_state.draft_mc_holder = start_monte_carlo_background(orchestrator, MC_N_SIMS)
            st.rerun()
        return

    season_result: SeasonResult = st.session_state.draft_season_result
    total_matchdays = max(m.matchday for m in season_result.matches)

    if not st.session_state.get("draft_playback_done", False):
        matchday = st.session_state.draft_playback_matchday
        st.markdown(f"#### 🗓️ Jornada {matchday} de {total_matchdays}")

        matchday_matches = [m for m in season_result.matches if m.matchday == matchday]
        squad_match = next(m for m in matchday_matches if m.home == SQUAD_TEAM_NAME or m.away == SQUAD_TEAM_NAME)

        already_watched = matchday in st.session_state.draft_watched_matchdays
        if not already_watched:
            home_label = "⭐ " + squad_match.home if squad_match.home == SQUAD_TEAM_NAME else squad_match.home.title()
            away_label = "⭐ " + squad_match.away if squad_match.away == SQUAD_TEAM_NAME else squad_match.away.title()
            st.markdown(f"**{home_label} vs {away_label}**")
            wc1, wc2 = st.columns([2, 1])
            play_clicked = wc1.button("▶️ Reproducir partido en directo", key=f"draft_play_{matchday}",
                                      use_container_width=True)
            skip_anim = wc2.button("⏭️ Saltar animación", key=f"draft_skip_anim_{matchday}",
                                   use_container_width=True)
            if play_clicked:
                play_live_match(squad_match, SQUAD_TEAM_NAME, picks_resolved, coords)
                st.session_state.draft_watched_matchdays.add(matchday)
                already_watched = True
            elif skip_anim:
                st.session_state.draft_watched_matchdays.add(matchday)
                already_watched = True

        if already_watched:
            render_matchday_summary(season_result, matchday, SQUAD_TEAM_NAME)
            nb1, nb2 = st.columns([2, 1])
            if matchday < total_matchdays:
                if nb1.button("▶️ Siguiente jornada", key=f"draft_next_{matchday}", use_container_width=True):
                    st.session_state.draft_playback_matchday += 1
                    st.rerun()
            else:
                if nb1.button("🏁 Ver resumen final de temporada", key="draft_finish", use_container_width=True):
                    st.session_state.draft_playback_done = True
                    st.rerun()
            if nb2.button("⏭️ Saltar al resultado final", key=f"draft_skip_all_{matchday}", use_container_width=True):
                st.session_state.draft_playback_done = True
                st.rerun()

    if st.session_state.get("draft_playback_done", False):
        st.markdown("---")
        render_season_result(season_result)

    st.markdown("---")
    render_monte_carlo_status(st.session_state.get("draft_mc_holder"), MC_N_SIMS)


def render_sandbox_mode(model: PlayerStrengthModel, engine=None, df_clubs: pd.DataFrame | None = None):
    st.markdown("### 🔬 Construye tu equipo ideal")
    st.caption("Mezcla jugadores de cualquier época o liga — experimento estadístico sin restricciones, "
              "pensado para escenarios tipo 'el Mundial si España tuviera a Messi'.")

    all_players = sorted(model.profiles_.keys())

    col_setup, col_squad = st.columns([2, 3])
    with col_setup:
        competition_filter = st.selectbox(
            "Filtrar por competición", ["Todas"] + DRAFT_COMPETITIONS, key="sb_comp")
        search_q = st.text_input("Buscar jugador", placeholder="Ej: Messi, Ronaldo...", key="sb_search")

        if search_q:
            comp_f = None if competition_filter == "Todas" else PLAYER_COMP_MAP[competition_filter]
            results = model.search(search_q, competition=comp_f, top_n=8)
            if results:
                st.markdown("**Resultados:**")
                for p in results:
                    st.markdown(render_player_card(p, compact=True), unsafe_allow_html=True)
            else:
                st.caption("Sin resultados.")

    squad_selected: list = []
    with col_squad:
        st.markdown("#### Alineación (4-3-3)")
        for pos, n_slots in POSITION_SLOTS.items():
            st.markdown(f"**{pos}** ({n_slots})")
            cols = st.columns(n_slots)
            for i, col in enumerate(cols):
                key = f"sb_{pos}_{i}"
                if competition_filter == "Todas":
                    candidates = [name for name, p in model.profiles_.items() if p.position == pos]
                else:
                    comp_f = PLAYER_COMP_MAP[competition_filter]
                    candidates = [name for name, p in model.profiles_.items()
                                if p.position == pos and p.competition == comp_f]
                candidates = sorted(candidates)
                if key not in st.session_state and candidates:
                    st.session_state[key] = candidates[0]
                sel = col.selectbox(f"#{i+1}", candidates, key=key)
                if sel and sel in model.profiles_:
                    squad_selected.append(model.profiles_[sel])

        if len(squad_selected) >= 4:
            strength = model.team_strength(squad_selected)
            st.markdown("---")
            st.plotly_chart(team_strength_visual(strength, "Tu equipo"), use_container_width=True)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Índice ataque", f"{strength['attack_index']:.0f}/100")
            mc2.metric("Índice defensa", f"{strength['defense_index']:.0f}/100")
            mc3.metric("xGoals estimados", f"{strength['xg_per_match']:.2f}/partido")

    if len(squad_selected) < 8:
        return

    st.markdown("---")
    st.markdown("### 🏆 Simulación de temporada")
    if engine is None or df_clubs is None:
        st.warning("Motor de predicción o datos de calendario no disponibles; no se puede simular la temporada.")
        return

    st.caption("Tu equipo se une como un club más a la competición elegida (no sustituye a nadie) "
              "y juega la temporada completa contra sus clubes reales.")
    default_idx = DRAFT_COMPETITIONS.index(competition_filter) if competition_filter in DRAFT_COMPETITIONS else 0
    ref_comp = st.selectbox("Competición de referencia (rivales reales)", DRAFT_COMPETITIONS,
                            index=default_idx, key="sb_ref_comp")
    comp_match_id = MATCH_COMP_MAP[ref_comp]

    seasons = sorted(df_clubs.loc[df_clubs["competition"] == comp_match_id, "season"].unique(), reverse=True)
    if not seasons:
        st.warning("Sin datos de calendario para esta competición.")
        return
    ref_teams_df = df_clubs[(df_clubs["competition"] == comp_match_id) & (df_clubs["season"] == seasons[0])]
    real_opponents = sorted(set(ref_teams_df["home_team"].map(canonical_name))
                            | set(ref_teams_df["away_team"].map(canonical_name)))
    real_opponents = [t for t in real_opponents if t in engine.ad_model_.team_index_][:19]
    if len(real_opponents) < 3:
        st.warning("No hay suficientes clubes reales reconocidos por el motor para esta competición.")
        return

    squad_key = (ref_comp, tuple(sorted(p.player for p in squad_selected)))
    if st.session_state.get("sb_orch_key") != squad_key:
        st.session_state.sb_orch_key = squad_key
        st.session_state.sb_orchestrator = _build_season_orchestrator(
            model, engine, squad_selected, real_opponents, competition=comp_match_id,
        )
        st.session_state.pop("sb_season_result", None)
        st.session_state.pop("sb_mc_result", None)

    b1, b2, b3 = st.columns([1, 1, 1])
    if b1.button("▶️ Simular temporada", key="sb_sim_season", use_container_width=True):
        with st.spinner("Jugando la temporada completa, partido a partido..."):
            st.session_state.sb_season_result = st.session_state.sb_orchestrator.play_once(narrative=True)
    n_sims = b2.selectbox("Simulaciones Monte Carlo", [10_000, 50_000, MC_N_SIMS],
                          index=1, key="sb_mc_n")
    if b3.button("🎲 Calcular probabilidades", key="sb_mc", use_container_width=True):
        with st.spinner(f"Corriendo {n_sims:,} simulaciones..."):
            st.session_state.sb_mc_result = st.session_state.sb_orchestrator.run_monte_carlo(n_sims=n_sims)

    if "sb_season_result" in st.session_state:
        render_season_result(st.session_state.sb_season_result)
    if "sb_mc_result" in st.session_state:
        render_monte_carlo_result(st.session_state.sb_mc_result, n_sims)


def render(engine=None, df_clubs: pd.DataFrame | None = None):
    """Main SquadLab render function — called from streamlit_app.py."""
    st.title("🧪 SquadLab")

    sq_model = load_strength_model()
    if not sq_model.profiles_:
        st.warning("Perfiles de jugadores no disponibles. Asegúrate de que "
                  "data/processed/player_profiles_with_positions.csv existe.")
        return

    mode = st.radio("Modo", ["🎮 Draft — temporada FM-style", "🔬 Sandbox — equipo libre"], horizontal=True)

    if "Draft" in mode:
        if df_clubs is None:
            st.warning("Datos de calendario no disponibles para calcular el colista.")
            return
        render_draft_mode(sq_model, df_clubs, engine)
    else:
        render_sandbox_mode(sq_model, engine, df_clubs)
