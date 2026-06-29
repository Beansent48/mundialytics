"""
SquadLab page — shown inside the main Streamlit app.
Two modes:
  Sandbox  : pick any 11 players freely, simulate match/tournament
  Draft    : 5 players shown per position slot, user picks one each round
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mundialytics.statistical_core.player_strength import PlayerStrengthModel

POSITIONS_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

POSITION_SLOTS = {
    "Goalkeeper": 1,
    "Defender":   4,
    "Midfielder": 3,
    "Forward":    3,
}

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
            f'<div style="background:var(--surface-1);border-radius:8px;padding:8px 10px;'
            f'border:0.5px solid var(--border)">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-weight:500;font-size:12px">{p.player.split()[0]}</span>'
            f'<span style="font-weight:700;color:{ov_color};font-size:14px">{p.overall:.0f}</span>'
            f'</div>'
            f'<div style="font-size:10px;color:var(--text-muted)">{p.team.title()} · {p.position[:3]}</div>'
            f'</div>'
        )
    return (
        f'<div style="background:var(--surface-1);border-radius:10px;padding:12px 14px;'
        f'border:0.5px solid var(--border);margin-bottom:8px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        f'<div>'
        f'<div style="font-weight:500;font-size:13px">{p.player}</div>'
        f'<div style="font-size:11px;color:var(--text-muted)">{p.team.title()} · {p.competition}</div>'
        f'</div>'
        f'<div style="font-size:22px;font-weight:700;color:{ov_color}">{p.overall:.0f}</div>'
        f'</div>'
        f'<div style="margin-top:8px">'
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<span style="font-size:10px;color:var(--text-muted);width:44px">Ataque</span>'
        f'<div style="flex:1;background:var(--surface-2);border-radius:2px;height:8px">'
        f'<div style="width:{off_bar}%;height:8px;background:#3b82f6;border-radius:2px"></div>'
        f'</div><span style="font-size:10px;width:28px;text-align:right">{p.offensive_strength:.0f}</span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<span style="font-size:10px;color:var(--text-muted);width:44px">Defensa</span>'
        f'<div style="flex:1;background:var(--surface-2);border-radius:2px;height:8px">'
        f'<div style="width:{def_bar}%;height:8px;background:#10b981;border-radius:2px"></div>'
        f'</div><span style="font-size:10px;width:28px;text-align:right">{p.defensive_strength:.0f}</span>'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;gap:10px;margin-top:8px;font-size:11px;color:var(--text-muted)">'
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


def simulate_match_with_squad(
    squad_home: list,
    squad_away: list,
    model: PlayerStrengthModel,
    n_sims: int = 50_000,
    neutral: bool = True,
) -> dict:
    """Simple Poisson match simulation from squad strength."""
    from scipy.stats import poisson
    home_str = model.team_strength(squad_home)
    away_str = model.team_strength(squad_away)

    lh_base = home_str["xg_per_match"]
    la_base = away_str["xg_per_match"]

    # Defense adjustment: opponent's defense index lowers attacking lambda
    def_ratio_h = away_str["defense_index"] / 50.0
    def_ratio_a = home_str["defense_index"] / 50.0
    lh = float(np.clip(lh_base / (def_ratio_h ** 0.5), 0.3, 4.0))
    la = float(np.clip(la_base / (def_ratio_a ** 0.5), 0.3, 4.0))

    if not neutral:
        lh *= 1.12  # home advantage ~12%

    rng = np.random.default_rng(42)
    hg = rng.poisson(lh, n_sims)
    ag = rng.poisson(la, n_sims)
    p_home = float((hg > ag).mean())
    p_draw = float((hg == ag).mean())
    p_away = float((ag > hg).mean())

    # Score matrix
    goals = np.arange(7)
    matrix = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "lambda_home": lh, "lambda_away": la,
        "p_btts": float(1 - poisson.cdf(0, lh) - poisson.cdf(0, la) + poisson.cdf(0, lh) * poisson.cdf(0, la)),
        "p_over25": float(1 - sum(poisson.pmf(k, lh + la) for k in range(3))),
        "score_matrix": matrix,
        "home_attack": home_str["attack_index"],
        "home_defense": home_str["defense_index"],
        "away_attack": away_str["attack_index"],
        "away_defense": away_str["defense_index"],
    }


def render(engine=None):
    """Main SquadLab render function — called from streamlit_app.py."""
    st.title("🧪 SquadLab")

    sq_model = load_strength_model()

    if not sq_model.profiles_:
        st.warning("Perfiles de jugadores no disponibles. Asegúrate de que data/processed/player_profiles_with_positions.csv existe.")
        return

    mode = st.radio("Modo", ["🔬 Sandbox — elige tu equipo libre", "🎲 Draft — 5 candidatos por posición"], horizontal=True)

    # ── SANDBOX MODE ──────────────────────────────────────────────────────────
    if "Sandbox" in mode:
        st.markdown("### Construye tu equipo ideal")
        st.caption("Mezcla jugadores de cualquier época o liga — es un experimento estadístico.")

        all_players = sorted(sq_model.profiles_.keys())

        col_setup, col_squad = st.columns([2, 3])
        with col_setup:
            competition_filter = st.selectbox(
                "Filtrar por competición", ["Todas", "La Liga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"],
                key="sb_comp")
            search_q = st.text_input("Buscar jugador", placeholder="Ej: Messi, Ronaldo...", key="sb_search")

            if search_q:
                comp_f = None if competition_filter == "Todas" else competition_filter
                results = sq_model.search(search_q, competition=comp_f, top_n=8)
                if results:
                    st.markdown("**Resultados:**")
                    for p in results:
                        st.markdown(render_player_card(p, compact=True), unsafe_allow_html=True)
                else:
                    st.caption("Sin resultados.")

        with col_squad:
            st.markdown("#### Alineación (4-3-3)")
            squad_selected: list = []

            for pos, n_slots in POSITION_SLOTS.items():
                st.markdown(f"**{pos}** ({n_slots})")
                cols = st.columns(n_slots)
                for i, col in enumerate(cols):
                    key = f"sb_{pos}_{i}"
                    # Filter candidates by position
                    if competition_filter == "Todas":
                        candidates = [name for name, p in sq_model.profiles_.items() if p.position == pos]
                    else:
                        candidates = [name for name, p in sq_model.profiles_.items()
                                       if p.position == pos and competition_filter.lower() in p.competition.lower()]
                    candidates = sorted(candidates)
                    default_idx = 0
                    if key not in st.session_state and candidates:
                        st.session_state[key] = candidates[0]
                    sel = col.selectbox(f"#{i+1}", candidates, key=key)
                    if sel and sel in sq_model.profiles_:
                        squad_selected.append(sq_model.profiles_[sel])

            if len(squad_selected) >= 4:
                strength = sq_model.team_strength(squad_selected)
                st.markdown("---")
                st.plotly_chart(team_strength_visual(strength, "Tu equipo"), use_container_width=True)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Índice ataque", f"{strength['attack_index']:.0f}/100")
                mc2.metric("Índice defensa", f"{strength['defense_index']:.0f}/100")
                mc3.metric("xGoals estimados", f"{strength['xg_per_match']:.2f}/partido")

    # ── DRAFT MODE ────────────────────────────────────────────────────────────
    else:
        st.markdown("### Draft de jugadores")
        st.caption("Se muestran 5 candidatos por posición ordenados por rating. Elige uno de cada.")

        competition_d = st.selectbox("Liga de referencia",
                                      ["La Liga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"],
                                      key="draft_comp")

        draft_squad: list = []
        tabs = st.tabs([f"{pos} ({n})" for pos, n in POSITION_SLOTS.items()])

        for tab, (pos, n_slots) in zip(tabs, POSITION_SLOTS.items()):
            with tab:
                top_players = sq_model.top_by_position(pos, competition=competition_d, n=5)
                if not top_players:
                    st.caption(f"Sin datos de {pos} en {competition_d}")
                    continue

                st.markdown(f"**Elige {n_slots} {pos}(s):**")
                cols = st.columns(len(top_players))
                for i, (col, candidate) in enumerate(zip(cols, top_players)):
                    ov_c = "#16a34a" if candidate.overall >= 70 else "#2563eb"
                    col.markdown(
                        f'<div style="background:var(--surface-1);border-radius:8px;padding:8px;'
                        f'border:0.5px solid var(--border);text-align:center">'
                        f'<div style="font-weight:700;font-size:18px;color:{ov_c}">{candidate.overall:.0f}</div>'
                        f'<div style="font-size:11px;font-weight:500">{candidate.player.split()[0]}</div>'
                        f'<div style="font-size:10px;color:var(--text-muted)">{candidate.team.title()}</div>'
                        f'<div style="font-size:10px;color:var(--text-muted)">xG {candidate.xg_per_match:.2f} | Tackl {candidate.tackles_per_match:.2f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                selected_names = [p.player for p in top_players]
                picks = []
                for slot in range(n_slots):
                    key = f"draft_{pos}_{slot}"
                    default = min(slot, len(selected_names) - 1)
                    pick = st.selectbox(f"{pos} #{slot+1}", selected_names, index=default, key=key)
                    p = sq_model.get(pick)
                    if p:
                        picks.append(p)
                        draft_squad.append(p)

        if len(draft_squad) >= 8:
            st.markdown("---")
            st.markdown("### Tu equipo draft")
            strength_d = sq_model.team_strength(draft_squad)
            st.plotly_chart(team_strength_visual(strength_d, "Equipo draft"), use_container_width=True)

            dcols = st.columns(3)
            dcols[0].metric("Índice ataque", f"{strength_d['attack_index']:.0f}/100")
            dcols[1].metric("Índice defensa", f"{strength_d['defense_index']:.0f}/100")
            dcols[2].metric("xGoals estimados", f"{strength_d['xg_per_match']:.2f}/partido")

    # ── MATCH SIMULATION ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Simular partido con tu equipo")

    if "squad_selected" in dir() and len(squad_selected) >= 8:
        home_squad = squad_selected
    elif "draft_squad" in dir() and len(draft_squad) >= 8:
        home_squad = draft_squad
    else:
        home_squad = []

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("**Tu equipo** (de arriba)")
        if not home_squad:
            st.caption("Construye tu equipo primero.")
    with col_sim2:
        st.markdown("**Rival**")
        rival_option = st.radio("Rival", ["Equipo real", "Custom"], horizontal=True, key="rival_opt")
        if rival_option == "Equipo real" and engine is not None:
            rival_teams = sorted(engine.ad_model_.team_index_.keys())
            rival_team = st.selectbox("Equipo rival", rival_teams, key="rival_team")
            rival_comp = st.selectbox("Competición rival", ["LaLiga","Premier League","Serie A","Bundesliga","Ligue 1"], key="rival_comp_sel")
        else:
            rival_team = "Selección genérica"
            rival_comp = "Other"

    n_sims_sl = st.select_slider("Simulaciones", [1_000, 10_000, 50_000], value=10_000, key="sl_n")

    if home_squad and st.button("Simular partido", key="sl_sim"):
        # Build away squad from top players of rival team
        away_squad_candidates = [p for p in sq_model.profiles_.values()
                                   if p.team.lower() == rival_team.lower() or
                                   (rival_team in sq_model.profiles_ and sq_model.profiles_[rival_team].team.lower() == p.team.lower())]
        away_squad_candidates = sorted(away_squad_candidates, key=lambda p: -p.overall)[:11]

        if len(away_squad_candidates) < 3:
            # Fallback: top players from first competition by overall
            away_squad_candidates = sorted(sq_model.profiles_.values(), key=lambda p: -p.overall)[:11]

        with st.spinner("Simulando..."):
            result = simulate_match_with_squad(home_squad, away_squad_candidates, sq_model, n_sims=n_sims_sl, neutral=False)

        # Show result
        home_label = "Tu equipo"
        away_label = rival_team.title()
        st.markdown(f"#### {home_label} vs {away_label}")

        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric(home_label, f"{result['p_home']:.1%}")
        rc2.metric("Empate", f"{result['p_draw']:.1%}")
        rc3.metric(away_label, f"{result['p_away']:.1%}")
        rc4.metric("xG", f"{result['lambda_home']:.2f}–{result['lambda_away']:.2f}")
        rc5.metric("BTTS", f"{result['p_btts']:.1%}")

        # Score matrix
        matrix = result["score_matrix"]
        fig_mat = go.Figure(go.Heatmap(
            z=matrix * 100,
            x=[str(i) for i in range(7)],
            y=[str(i) for i in range(7)],
            text=[[f"{v:.1f}%" for v in row] for row in matrix * 100],
            texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            hovertemplate=f"{home_label} %{{y}}–%{{x}} {away_label}: %{{z:.2f}}%<extra></extra>",
        ))
        fig_mat.update_layout(
            xaxis_title=f"Goles {away_label}", yaxis_title=f"Goles {home_label}",
            yaxis=dict(autorange="reversed"), height=280,
            margin=dict(l=50, r=10, t=10, b=50),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mat, use_container_width=True)

        st.markdown(f"**En 100k simulaciones:** tu equipo gana el **{result['p_home']:.0%}** de las veces — {result['p_home']*100:.0f} de cada 100 partidos.")
