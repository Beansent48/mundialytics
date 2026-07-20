"""
Competition forecast page — "league from the current point".

Renders a cached full-season snapshot bundle (see
statistical_core/competition/forecast_cache): the matchday slider scrubs the whole
season instantly (table, title/top-4/relegation probabilities, position matrix all
update from cache), plus the matchday-by-matchday probability evolution and
upcoming fixtures with 1X2 predictions.

Reading is instant (JSON cache). Only the explicit "recalcular" button computes —
the normal page load never trains anything, matching the daily-traffic design.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT / "data/processed/foundation_big5_multi_season.csv"
CACHE_DIR = ROOT / "data/processed/competition_cache"

LEAGUES = ["LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"]
SERIES_HEX = ["#2a78d6", "#4a3aa7", "#e34948", "#1baf7a", "#eda100", "#e87ba4"]


@st.cache_data(show_spinner=False)
def _foundation() -> pd.DataFrame:
    return pd.read_csv(FOUNDATION, low_memory=False)


def _seasons_for(comp: str) -> list[str]:
    df = _foundation()
    return list(sorted(df.loc[df["competition"] == comp, "season"].unique(), reverse=True))


def _title(name: str) -> str:
    return str(name).title()


# ── Renderers (all read a single snapshot dict) ─────────────────────────────────

def _render_standings(snap: dict) -> None:
    df = pd.DataFrame(snap["standings"])
    if df.empty:
        return
    df = df[["rank", "team", "played", "won", "drawn", "lost", "goals_for", "goals_against", "goal_diff", "points"]]
    df["team"] = df["team"].map(_title)
    df.columns = ["#", "Equipo", "PJ", "G", "E", "P", "GF", "GC", "DG", "Pts"]
    st.dataframe(df, hide_index=True, use_container_width=True, height=min(38 * len(df) + 40, 760))


def _prob_bar(snap: dict) -> None:
    tp = pd.DataFrame(snap["forecast"]["team_probs"]).copy()
    if tp.empty or snap.get("n_remaining", 0) == 0:
        st.info("Temporada completa — no hay partidos pendientes que simular.")
        return
    tp["team"] = tp["team"].map(_title)
    tp = tp.sort_values("p_champion", ascending=True)
    fig = go.Figure(go.Bar(
        x=(tp["p_champion"] * 100).round(1), y=tp["team"], orientation="h", marker_color="#2a78d6",
        text=[f"{v:.0f}%" if v >= 1 else "" for v in tp["p_champion"] * 100], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra>Campeón</extra>",
    ))
    fig.update_layout(height=max(28 * len(tp), 200), margin=dict(l=8, r=8, t=8, b=8),
                      xaxis_title="Probabilidad de ser campeón (%)", yaxis_title=None,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def _position_heatmap(snap: dict) -> None:
    pm = snap["forecast"]["position_matrix"]
    teams = [_title(t) for t in pm["teams"]]
    z = [[round(v * 100, 1) for v in row] for row in pm["values"]]
    if not z:
        return
    fig = go.Figure(go.Heatmap(z=z, x=pm["positions"], y=teams, colorscale="Blues", zmin=0,
                               hovertemplate="%{y} — %{x}º: %{z:.1f}%<extra></extra>", colorbar=dict(title="%")))
    fig.update_layout(height=max(26 * len(teams), 300), margin=dict(l=8, r=8, t=8, b=8),
                      xaxis_title="Posición final", yaxis=dict(autorange="reversed"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def _evolution_chart(bundle: dict, highlight_md: int) -> None:
    from mundialytics.statistical_core.competition import forecast_cache as fc
    tl = pd.DataFrame(fc.build_timeline(bundle))
    if tl.empty or tl["matchday"].nunique() < 2:
        st.caption("Aún no hay suficientes jornadas para dibujar la evolución.")
        return
    last_md = tl["matchday"].max()
    top_teams = (tl[tl["matchday"] == last_md].sort_values("p_champion", ascending=False)["team"].head(5).tolist())
    fig = go.Figure()
    for i, team in enumerate(top_teams):
        sub = tl[tl["team"] == team].sort_values("matchday")
        fig.add_trace(go.Scatter(x=sub["matchday"], y=(sub["p_champion"] * 100).round(1),
                                 mode="lines+markers", name=_title(team),
                                 line=dict(color=SERIES_HEX[i % len(SERIES_HEX)], width=2)))
    fig.add_vline(x=highlight_md, line_dash="dot", line_color="#9ca3af")
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="Jornada",
                      yaxis_title="Prob. de campeón (%)", yaxis_range=[0, 100],
                      legend=dict(orientation="h", y=-0.2),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def _upcoming_fixtures(snap: dict, n: int = 12) -> None:
    rem = pd.DataFrame(snap["fixtures"]["remaining"])
    if rem.empty:
        st.caption("Sin partidos pendientes.")
        return
    for r in rem.head(n).itertuples(index=False):
        c1, c2, c3 = st.columns([3, 4, 3])
        c1.markdown(f"<div style='text-align:right;font-weight:600'>{_title(r.home_team)}</div>", unsafe_allow_html=True)
        c2.markdown(
            f"<div style='text-align:center;font-size:.85rem;color:#9ca3af'>"
            f"<b style='color:#16a34a'>{r.p_home*100:.0f}%</b> &nbsp;·&nbsp; {r.p_draw*100:.0f}% &nbsp;·&nbsp; "
            f"<b style='color:#dc2626'>{r.p_away*100:.0f}%</b><br>"
            f"<span style='font-size:.72rem'>λ {r.lambda_home:.2f} – {r.lambda_away:.2f}</span></div>",
            unsafe_allow_html=True)
        c3.markdown(f"<div style='text-align:left;font-weight:600'>{_title(r.away_team)}</div>", unsafe_allow_html=True)


# ── Entry point ─────────────────────────────────────────────────────────────────

def render() -> None:
    from mundialytics.statistical_core.competition import forecast_cache as fc

    st.title("📊  Pronóstico de liga (desde el punto actual)")
    st.caption("Arrastra la jornada para ver cómo evolucionan las probabilidades de título, Champions y descenso "
               "a lo largo de la temporada. Lectura instantánea desde caché.")

    c1, c2 = st.columns(2)
    comp = c1.selectbox("Competición", LEAGUES, key="cf_comp")
    season = c2.selectbox("Temporada", _seasons_for(comp), key="cf_season")

    bundle = fc.load_bundle(comp, season, cache_dir=CACHE_DIR)

    if bundle is None or bundle.get("meta", {}).get("schema") != fc.BUNDLE_SCHEMA:
        st.info(f"No hay pronóstico cacheado para **{comp} {season}**. "
                "Genéralo (se guardará para próximas visitas — la lectura luego es instantánea).")
        if st.button("🔄 Generar pronóstico de temporada", type="primary"):
            with st.spinner(f"Simulando {comp} {season} jornada a jornada… (una sola vez)"):
                bundle = fc.get_or_build(comp, season, _foundation(), timeline_step=5,
                                         n_sims=10000, cache_dir=CACHE_DIR)
            st.rerun()
        return

    mds = fc.available_matchdays(bundle)
    default_md = bundle["meta"].get("current_matchday", max(mds))
    md_choice = st.slider("Jornada", min(mds), max(mds), default_md, key="cf_md")
    used_md, snap = fc.snapshot_for(bundle, md_choice)
    if used_md != md_choice:
        st.caption(f"Mostrando la jornada cacheada más cercana: **{used_md}**.")

    meta = bundle["meta"]
    st.caption(f"{meta['n_sims']:,} simulaciones · {meta['model_note']}")

    tp = pd.DataFrame(snap["forecast"]["team_probs"])
    lead = pd.DataFrame(snap["standings"]).iloc[0]
    m = st.columns(3)
    m[0].metric("Líder en jornada " + str(used_md), _title(lead["team"]), f"{int(lead['points'])} pts")
    if not tp.empty and snap.get("n_remaining", 0) > 0:
        fav = tp.sort_values("p_champion", ascending=False).iloc[0]
        m[1].metric("Favorito al título", _title(fav["team"]), f"{fav['p_champion']*100:.0f}%")
    m[2].metric("Partidos restantes", f"{snap['fingerprint']['n_played']} jugados",
                f"{snap.get('n_remaining', 0)} por jugar")

    tabs = st.tabs(["🏆 Probabilidades", "🎲 Matriz de posiciones", "📈 Evolución", "🗓️ Próximos", "📋 Clasificación"])
    with tabs[0]:
        _prob_bar(snap)
    with tabs[1]:
        st.caption("Cada fila es un equipo; cada columna, una posición final. Más oscuro = más probable.")
        _position_heatmap(snap)
    with tabs[2]:
        st.caption("Probabilidad de título jornada a jornada (leakage-free en cada punto). La línea punteada marca la jornada seleccionada.")
        _evolution_chart(bundle, used_md)
    with tabs[3]:
        _upcoming_fixtures(snap)
    with tabs[4]:
        _render_standings(snap)
