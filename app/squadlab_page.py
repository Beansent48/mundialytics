"""
SquadLab page — shown inside the main Streamlit app.
Two modes:
  Draft    : pick a competition, your club replaces the league's current
             last-place team, draft 11 players from 5 candidates per slot
             (each pick is removed from the global pool — no duplicates,
             no player available to a club after being drafted away).
  Sandbox  : pick any 11 players freely, quick single-match simulation.

Season-long match-by-match simulation (Football-Manager style results,
lineups, ratings) and the 1M-sim Monte Carlo award/odds layer are the
next phase — this file currently builds the draft + squad foundation.
"""
from __future__ import annotations
import hashlib
import random
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

    def_ratio_h = away_str["defense_index"] / 50.0
    def_ratio_a = home_str["defense_index"] / 50.0
    lh = float(np.clip(lh_base / (def_ratio_h ** 0.5), 0.3, 4.0))
    la = float(np.clip(la_base / (def_ratio_a ** 0.5), 0.3, 4.0))

    if not neutral:
        lh *= 1.12

    rng = np.random.default_rng(42)
    hg = rng.poisson(lh, n_sims)
    ag = rng.poisson(la, n_sims)
    p_home = float((hg > ag).mean())
    p_draw = float((hg == ag).mean())
    p_away = float((ag > hg).mean())

    goals = np.arange(7)
    matrix = np.outer(poisson.pmf(goals, lh), poisson.pmf(goals, la))

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "lambda_home": lh, "lambda_away": la,
        "p_btts": float(1 - poisson.cdf(0, lh) - poisson.cdf(0, la) + poisson.cdf(0, lh) * poisson.cdf(0, la)),
        "p_over25": float(1 - sum(poisson.pmf(k, lh + la) for k in range(3))),
        "score_matrix": matrix,
        "home_attack": home_str["attack_index"], "home_defense": home_str["defense_index"],
        "away_attack": away_str["attack_index"], "away_defense": away_str["defense_index"],
    }


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


def pitch_svg(picks: dict, coords: dict) -> str:
    w, h = 460, 600
    svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
           f'style="width:100%;max-width:460px;display:block;margin:0 auto">']
    svg.append(f'<rect width="{w}" height="{h}" fill="#1f7a3d" rx="12"/>')
    svg.append(f'<rect x="8" y="8" width="{w-16}" height="{h-16}" fill="none" stroke="#ffffff66" stroke-width="2"/>')
    svg.append(f'<circle cx="{w/2}" cy="{h/2}" r="48" fill="none" stroke="#ffffff66" stroke-width="2"/>')
    svg.append(f'<line x1="8" y1="{h/2}" x2="{w-8}" y2="{h/2}" stroke="#ffffff66" stroke-width="2"/>')
    svg.append(f'<rect x="{w/2-85}" y="8" width="170" height="58" fill="none" stroke="#ffffff66" stroke-width="2"/>')
    svg.append(f'<rect x="{w/2-85}" y="{h-66}" width="170" height="58" fill="none" stroke="#ffffff66" stroke-width="2"/>')

    for pos, pos_coords in coords.items():
        for slot, (xp, yp) in enumerate(pos_coords):
            key = f"{pos}_{slot}"
            profile = picks.get(key)
            cx, cy = xp / 100 * w, yp / 100 * h
            if profile:
                label = profile.player.split()[-1][:11]
                ov = f"{profile.overall:.0f}"
                color = "#16a34a" if profile.overall >= 70 else "#2563eb"
            else:
                label, ov, color = "vacío", "", "#9ca3af99"
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="22" fill="{color}" stroke="white" stroke-width="2"/>')
            if ov:
                svg.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="13" '
                          f'font-weight="700" fill="white">{ov}</text>')
            svg.append(f'<text x="{cx}" y="{cy+38}" text-anchor="middle" font-size="11" '
                      f'fill="white" style="text-shadow:0 1px 2px #00000088">{label}</text>')
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


def render_draft_mode(model: PlayerStrengthModel, df_clubs: pd.DataFrame):
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
        st.session_state.draft_active_comp = active_key

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
            st.success("Plantilla completa. La simulación de temporada partido a partido "
                      "(resultados, alineaciones, ratings y el Monte Carlo de premios) "
                      "llega en la siguiente fase.")
            if st.button("🔄 Reiniciar draft", key="draft_reset"):
                st.session_state.draft_picks = {}
                st.session_state.draft_excluded = set()
                st.session_state.draft_active_slot = None
                st.session_state.draft_reroll = {}
                st.rerun()

    with col_pick:
        st.caption("El número en cada candidato es su **valoración global (0-100)**: combina su percentil "
                  "de ataque y de defensa frente a otros jugadores de su misma posición, a partir de sus "
                  "estadísticas reales por partido (goles, asistencias, tiros, entradas, presiones...).")

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
                label = (current.split()[-1][:10] if current else f"#{slot + 1}")
                if sc.button(("🟢 " if is_active else ("✅ " if current else "")) + label,
                           key=f"slotbtn_{comp_d}_{formation_name}_{key}",
                           use_container_width=True):
                    st.session_state.draft_active_slot = None if is_active else key
                    st.rerun()

        active_slot = st.session_state.draft_active_slot
        if active_slot:
            pos = active_slot.rsplit("_", 1)[0]
            current = st.session_state.draft_picks.get(active_slot)
            reroll_n = st.session_state.draft_reroll.get(active_slot, 0)
            seed_str = f"{comp_d}|{formation_name}|{active_slot}|{reroll_n}"
            candidates = seeded_candidates(pool, pos, seed_str, st.session_state.draft_excluded,
                                          current, model)

            st.markdown("---")
            st.markdown(f"**Candidatos a {pos} — slot #{int(active_slot.rsplit('_',1)[1]) + 1}**")
            if not candidates:
                st.caption("Sin candidatos disponibles en esta competición.")
            else:
                cand_cols = st.columns(len(candidates))
                for cc, cand in zip(cand_cols, candidates):
                    is_current = current == cand.player
                    ov_c = "#16a34a" if cand.overall >= 70 else "#2563eb"
                    border = "border:2px solid #16a34a;" if is_current else ""
                    cc.markdown(
                        f'<div style="background:var(--secondary-background-color);border-radius:8px;'
                        f'padding:6px;text-align:center;font-size:11px;{border}">'
                        f'<div style="font-weight:700;font-size:16px;color:{ov_c}">{cand.overall:.0f}</div>'
                        f'<div style="font-weight:500">{cand.player.split()[-1][:10]}</div>'
                        f'<div style="color:#9ca3af;font-size:9px">{cand.team.title()[:14]}</div>'
                        f'</div>', unsafe_allow_html=True)
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
                    st.session_state.draft_reroll[active_slot] = reroll_n + 1
                    st.rerun()
        else:
            st.caption("👆 Pulsa una posición arriba para ver sus 5 candidatos.")


def render_sandbox_mode(model: PlayerStrengthModel, engine=None):
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
    st.markdown("### Simulación rápida de partido")
    st.caption("Vista previa de un único partido. La simulación de torneo completo (1M sims, "
              "% de campeón, bota de oro, etc.) llega en la siguiente fase.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("**Tu equipo** (de arriba)")
    with col_sim2:
        st.markdown("**Rival**")
        if engine is not None:
            rival_teams = sorted(engine.ad_model_.team_index_.keys())
            rival_team = st.selectbox("Equipo rival", rival_teams, key="rival_team")
        else:
            rival_team = "Selección genérica"

    if st.button("Simular partido", key="sl_sim"):
        away_squad_candidates = [p for p in model.profiles_.values() if p.team.lower() == rival_team.lower()]
        away_squad_candidates = sorted(away_squad_candidates, key=lambda p: -p.overall)[:11]
        if len(away_squad_candidates) < 3:
            away_squad_candidates = sorted(model.profiles_.values(), key=lambda p: -p.overall)[:11]

        with st.spinner("Simulando..."):
            result = simulate_match_with_squad(squad_selected, away_squad_candidates, model,
                                               n_sims=50_000, neutral=False)

        home_label, away_label = "Tu equipo", rival_team.title()
        st.markdown(f"#### {home_label} vs {away_label}")

        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc1.metric(home_label, f"{result['p_home']:.1%}")
        rc2.metric("Empate", f"{result['p_draw']:.1%}")
        rc3.metric(away_label, f"{result['p_away']:.1%}")
        rc4.metric("xG", f"{result['lambda_home']:.2f}–{result['lambda_away']:.2f}")
        rc5.metric("BTTS", f"{result['p_btts']:.1%}")

        matrix = result["score_matrix"]
        fig_mat = go.Figure(go.Heatmap(
            z=matrix * 100,
            x=[str(i) for i in range(7)], y=[str(i) for i in range(7)],
            text=[[f"{v:.1f}%" for v in row] for row in matrix * 100],
            texttemplate="%{text}", colorscale="Blues", showscale=False,
            hovertemplate=f"{home_label} %{{y}}–%{{x}} {away_label}: %{{z:.2f}}%<extra></extra>",
        ))
        fig_mat.update_layout(
            xaxis_title=f"Goles {away_label}", yaxis_title=f"Goles {home_label}",
            yaxis=dict(autorange="reversed"), height=280,
            margin=dict(l=50, r=10, t=10, b=50),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mat, use_container_width=True)
        st.markdown(f"**En 50k simulaciones:** tu equipo gana el **{result['p_home']:.0%}** de las veces.")


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
        render_draft_mode(sq_model, df_clubs)
    else:
        render_sandbox_mode(sq_model, engine)
