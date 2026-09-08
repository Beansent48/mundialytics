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
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mundialytics.identity.display_names import display_name as _display_name_raw
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
    """HTML card for a player.

    An ICONO is simply a player no big-five club has today — retired, or gone to
    a league we do not track. The pool deliberately keeps them (that is the
    point of a sandbox), so the card says which era it is offering rather than
    letting a 2011 Barcelona side pass for this season's. Rarity and draw odds
    are a separate, later decision; this only names the set.
    """
    off_bar = int(p.offensive_strength * 0.9)
    def_bar = int(p.defensive_strength * 0.9)
    ov_color = "#16a34a" if p.overall >= 70 else ("#2563eb" if p.overall >= 50 else "#9ca3af")
    icon = getattr(p, "is_icon", False)
    icon_tag = ('<span style="font-size:9px;font-weight:700;letter-spacing:.5px;'
                'color:#b45309;background:#fbbf2433;border:1px solid #fbbf2455;'
                'border-radius:4px;padding:1px 4px;margin-left:6px">ICONO</span>') if icon else ""
    where = (p.current_team or p.team).title() if not icon else p.team.title()
    if compact:
        return (
            f'<div style="background:var(--secondary-background-color);border-radius:8px;padding:8px 10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-weight:500;font-size:12px">{p.player.split()[0]}{icon_tag}</span>'
            f'<span style="font-weight:700;color:{ov_color};font-size:14px">{p.overall:.0f}</span>'
            f'</div>'
            f'<div style="font-size:10px;color:#9ca3af">{where} · {p.position[:3]}</div>'
            f'</div>'
        )
    return (
        f'<div style="background:var(--secondary-background-color);border-radius:10px;padding:12px 14px;'
        f'margin-bottom:8px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        f'<div>'
        f'<div style="font-weight:500;font-size:13px">{p.player}{icon_tag}</div>'
        f'<div style="font-size:11px;color:#9ca3af">{where} · {p.competition}</div>'
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
    # Drafted players are CLONES: same stats, distinct identity. The real player
    # keeps playing for his club, and both appear separately in the scorer race
    # (without cloning, one name in two rosters double-counts his goals).
    squad_clones = [clone_for_squad(p, squad_team_name) for p in squad]
    rosters: dict[str, list] = {squad_team_name: squad_clones}
    rosters.update(build_real_team_rosters(model, real_teams))
    return SeasonOrchestrator(
        lambda_source, fixtures, squad_roster=rosters, competition=competition,
    )


def build_real_team_rosters(model: PlayerStrengthModel, real_teams: list[str]) -> dict[str, list]:
    """Best XI (1-4-3-3 shape) per real club, from today's squads.

    Delegates to the serving helper the API already uses: two front ends that
    disagree about who plays for Barcelona would narrate two different seasons.
    See mundialytics.serving.squad_roster for why the club a profile is filed
    under is not the club its player is at.
    """
    from mundialytics.serving.squad_roster import real_team_rosters
    return real_team_rosters(model, real_teams)


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

# A drafted player is a CLONE: identical stats, distinct identity. The original
# keeps playing for his real club, so both compete separately in the scorer
# race. The mark makes the season-tally key unique (tallies are keyed by name)
# and is stripped for display.
SQUAD_CLONE_MARK = " ✦"


def clone_for_squad(profile, squad_team_name: str):
    """Same-stats copy of a player under the user's club and a unique identity."""
    import dataclasses
    if str(profile.player).endswith(SQUAD_CLONE_MARK):
        return profile
    return dataclasses.replace(profile,
                               player=f"{profile.player}{SQUAD_CLONE_MARK}",
                               team=squad_team_name)


def _strip_clone(name: str) -> str:
    return str(name).replace(SQUAD_CLONE_MARK, "")


def _disp(name: str | None) -> str:
    """Short display name, keeping whatever mark the card appended.

    Three marks are in play — " ✦" for a drafted clone, " ★" for an icon and
    " 15/16" for a prime — and all three are the tail of the string, which is
    exactly what `display_name` keeps when it shortens. Every one of them came
    back AS the name until it was split off first.
    """
    if not name:
        return ""
    from mundialytics.statistical_core.squadlab.cards import split_card_mark

    raw = str(name)
    clone = SQUAD_CLONE_MARK if raw.endswith(SQUAD_CLONE_MARK) else ""
    base, mark = split_card_mark(_strip_clone(raw))
    return _display_name_raw(base) + mark + clone


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
            # picks hold the ORIGINAL profile; season events are keyed by the
            # drafted clone's unique name — try both.
            ev = events.get(profile.player) or events.get(f"{profile.player}{SQUAD_CLONE_MARK}")
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


HIST_CSV = ROOT / "data/processed/historical_teams.csv"


@st.cache_data(show_spinner=False)
def load_historical_teams() -> pd.DataFrame:
    return pd.read_csv(HIST_CSV) if HIST_CSV.exists() else pd.DataFrame()


# Merge the ways one club is spelled across the two data sources: a curated
# side's short label ("Milan", "Inter", "Bayern") vs the season file's full name
# ("ac milan", "inter milan", "bayern munich"), plus accents.
_CLUB_ALIASES = {
    "ac milan": "milan", "inter milan": "inter", "internazionale": "inter",
    "bayern munich": "bayern", "fc bayern": "bayern", "atletico de madrid": "atletico madrid",
    "manchester utd": "manchester united", "man city": "manchester city",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur",
    "paris saint-germain": "paris saint germain", "psg": "paris saint germain",
    "borussia dortmund": "dortmund", "bvb": "dortmund", "fc porto": "porto",
    "afc ajax": "ajax", "as roma": "roma", "ssc napoli": "napoli",
}


def _canon_club(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = n.lower().strip()
    return _CLUB_ALIASES.get(n, n)


# Best season of each true giant makes the "leyendas" bracket, so it isn't
# padded by full-coverage 2014/15 mid-table sides (Empoli, Guingamp…) that only
# rank high because StatsBomb released their whole season.
_GIANT_CLUBS = {
    "barcelona", "real madrid", "paris saint germain", "bayern", "manchester city",
    "manchester united", "liverpool", "chelsea", "arsenal", "juventus", "milan",
    "inter", "napoli", "roma", "atletico madrid", "dortmund", "sevilla", "valencia",
    "tottenham hotspur", "bayer leverkusen", "ajax", "porto", "lyon", "as monaco",
    "atalanta", "villarreal", "real sociedad", "rb leipzig",
}


def _one_per_club(cat: pd.DataFrame) -> pd.DataFrame:
    """Keep each club's single strongest side.

    StatsBomb released every Barça match, so the raw top-20 is ~19 Barcelona
    seasons — a 'best historic teams' draw of 19 identical Barças. Collapsing to
    one entry per club turns it into a varied bracket: Barça, Madrid, Bayern,
    Milan, Ajax, Porto…"""
    out = cat.sort_values("strength", ascending=False).copy()
    out["_club"] = out["team"].map(_canon_club)
    return (out.drop_duplicates(subset="_club", keep="first")
               .drop(columns="_club").reset_index(drop=True))


def _iconic_pool(cat: pd.DataFrame) -> pd.DataFrame:
    """Marquee bracket: the 13 curated iconic sides + each giant's best season.

    A curated side (Milan 2007, Bayern 2013…) wins its club over the season
    version (ac milan 2015) even when weaker on paper — it's the one people mean.
    """
    out = cat.copy()
    out["_club"] = out["team"].map(_canon_club)
    keep = out[(out["kind"] == "curated") | (out["_club"].isin(_GIANT_CLUBS))]
    # curated first so it wins the drop_duplicates collision, then by strength
    keep = keep.assign(_pri=(keep["kind"] != "curated").astype(int))
    keep = keep.sort_values(["_pri", "strength"], ascending=[True, False])
    keep = keep.drop_duplicates(subset="_club", keep="first")
    return keep.sort_values("strength", ascending=False).drop(columns=["_club", "_pri"]).reset_index(drop=True)


def _tie_row_html(tie: dict, hi: str | None = None) -> str:
    a, b, w = tie["team_a"], tie["team_b"], tie["winner"]
    def nm(x):
        strong = "font-weight:800;color:#e2e8f0" if x == w else "color:#9aa9bf"
        mark = " ⭐" if hi and x == hi else ""
        return f'<span style="{strong}">{x.title()}{mark}</span>'
    note = f' <span style="color:#f59e0b;font-size:.7rem">{tie["note"]}</span>' if tie["note"] else ""
    legs = f'{tie["leg1"]}' + (f' · {tie["leg2"]}' if tie.get("leg2") else "")
    return (f'<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;'
            f'background:#151f30;border:1px solid #243350;border-radius:8px;margin:3px 0">'
            f'<div style="flex:1;text-align:right;font-size:.86rem">{nm(a)}</div>'
            f'<div style="min-width:56px;text-align:center;font-weight:800;color:#3b82f6">'
            f'{tie["agg"]}{note}</div>'
            f'<div style="flex:1;font-size:.86rem">{nm(b)}</div>'
            f'<div style="width:82px;text-align:right;color:#64748b;font-size:.68rem">{legs}</div>'
            f'</div>')


def render_historic_champions() -> None:
    st.markdown("### 🏛️ Champions histórica")
    st.caption("Una Champions con equipos de cualquier época: el Barça de Guardiola, el "
               "Madrid de CR7, el Milan de Kaká, el Ajax de De Jong y De Ligt, el Porto de "
               "Mourinho… Se enfrentan en el formato actual (fase liga de 36 + playoff + "
               "eliminatorias a doble partido + final a un solo encuentro).")

    cat = load_historical_teams()
    if cat.empty:
        st.warning("Falta el catálogo histórico. Genera con: "
                   "`python scripts/build_historical_teams.py`")
        return
    try:
        from mundialytics.statistical_core.competition.european import (
            EuropeanTournament, load_calibration)
        calib = load_calibration(ROOT)
    except Exception as exc:
        st.warning(f"Simulador europeo no disponible: {exc}")
        return

    clubs = _one_per_club(cat)   # one entry per club → varied brackets
    icons = _iconic_pool(cat)    # curated giants + each giant's best season

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        pool_mode = st.selectbox(
            "¿Qué equipos entran?",
            ["🏆 Leyendas (equipos icónicos)",
             "🎲 Sorteo variado (un club, un equipo)",
             "🎰 Cualquier época (todas las versiones)",
             "✍️ Elegir a mano"],
            key="hist_pool")
    with c2:
        seed = st.number_input("Semilla", 1, 9999, 7, key="hist_seed",
                               help="Cambia la semilla para otro sorteo y otro desarrollo")
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        go = st.button("⚽ Jugar la Champions", use_container_width=True, key="hist_go")

    if pool_mode.startswith("✍️"):
        opts = cat["label"].tolist()
        default = icons.head(36)["label"].tolist()
        chosen = st.multiselect("Equipos (elige 36; menos también vale, mínimo 8)",
                                opts, default=default, key="hist_manual")
        sel = cat[cat["label"].isin(chosen)]
    elif pool_mode.startswith("🏆"):
        sel = icons.head(36)
    elif pool_mode.startswith("🎲"):
        sel = clubs.head(64).sample(min(36, len(clubs)), random_state=int(seed))
    else:
        sel = cat.sample(min(36, len(cat)), random_state=int(seed))

    fav = sel.nlargest(1, "strength")
    st.caption(f"{len(sel)} equipos · favorito: **{fav.iloc[0]['label'].title()}** "
               f"(elo {fav.iloc[0]['elo']:.0f})" if len(sel) else "")

    if go and len(sel) >= 8:
        elo = {r.label: float(r.elo) for r in sel.itertuples(index=False)}
        rng = np.random.default_rng(int(seed) * 1000 + 7)
        tour = EuropeanTournament("champions", elo, calib, rng=rng)
        with st.spinner("Jugando la fase liga y las eliminatorias..."):
            st.session_state["hist_result"] = tour.play_single()
            st.session_state["hist_probs"] = tour.simulate(600)

    res = st.session_state.get("hist_result")
    if not res:
        st.info("Elige los equipos y pulsa **Jugar la Champions**.")
        return

    champ = res["champion"]
    st.markdown(
        f'<div style="text-align:center;padding:16px;margin:8px 0;border-radius:14px;'
        f'background:linear-gradient(140deg,#1b1147,#3b1d6e);border:2px solid #ffd75e">'
        f'<div style="font-size:.8rem;letter-spacing:.2em;color:#e9d8ff">CAMPEÓN DE EUROPA</div>'
        f'<div style="font-size:1.9rem;font-weight:900;color:#ffd75e">{champ.title()}</div>'
        f'<div style="font-size:.8rem;color:#e9d8ff">venció a {res["runner_up"].title()} '
        f'en la final</div></div>', unsafe_allow_html=True)

    st.markdown("#### 🗝️ Camino al título")
    for key, label in [("final", "Final"), ("sf", "Semifinales"), ("qf", "Cuartos"),
                       ("r16", "Octavos"), ("playoff", "Playoff")]:
        ties = res["rounds"].get(key, [])
        if not ties:
            continue
        with st.expander(label, expanded=key in ("final", "sf", "qf")):
            for tie in ties:
                st.markdown(_tie_row_html(tie, hi=champ), unsafe_allow_html=True)

    with st.expander("📊 Fase liga (clasificación final)"):
        tbl = res["table"].copy()
        tbl["team"] = tbl["team"].str.title()
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=420)

    probs = st.session_state.get("hist_probs")
    if probs is not None and len(probs):
        st.markdown("#### 🎯 ¿Quién era favorito? (600 simulaciones del mismo cuadro)")
        top = probs.head(10)[["team", "elo", "p_top8", "p_champion"]].copy()
        top["team"] = top["team"].str.title()
        top[["p_top8", "p_champion"]] = (top[["p_top8", "p_champion"]] * 100).round(1)
        top = top.rename(columns={"team": "Equipo", "elo": "Elo", "p_top8": "Top 8",
                                  "p_champion": "Campeón"})
        st.dataframe(top, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in ["Top 8", "Campeón"]})
        row = probs[probs["team"] == champ]
        if len(row):
            st.caption(f"El campeón partía con un **{row.iloc[0]['p_champion']:.1%}** de "
                       "probabilidad — así que fue lo esperado o una sorpresa, según el número.")


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

    mode = st.radio("Modo", ["🎮 Champions Draft", "🔬 Sandbox — equipo libre",
                             "🏛️ Champions histórica"], horizontal=True)

    if "Draft" in mode:
        import squadlab_draft
        squadlab_draft.render()
    elif "Champions" in mode:
        render_historic_champions()
    else:
        render_sandbox_mode(sq_model, engine, df_clubs)
