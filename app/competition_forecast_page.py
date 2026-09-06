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

import mv_ui
from mv_ui import chart

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT / "data/processed/foundation_big5_multi_season.csv"
CACHE_DIR = ROOT / "data/processed/competition_cache"

LEAGUES = ["LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"]
# one palette for the whole app — this page used to run its own, so the same
# team was a different blue here than on every other screen
SERIES_HEX = mv_ui.SERIES


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
        x=(tp["p_champion"] * 100).round(1), y=tp["team"], orientation="h",
        marker_color=mv_ui.C["blue"],
        text=[f"{v:.0f}%" if v >= 1 else "" for v in tp["p_champion"] * 100], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra>Campeón</extra>",
    ))
    fig.update_layout(height=max(28 * len(tp), 200), margin=dict(l=8, r=8, t=8, b=8),
                      xaxis_title="Probabilidad de ser campeón (%)", yaxis_title=None,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    chart(fig)


def _position_heatmap(snap: dict) -> None:
    pm = snap["forecast"]["position_matrix"]
    teams = [_title(t) for t in pm["teams"]]
    z = [[round(v * 100, 1) for v in row] for row in pm["values"]]
    if not z:
        return
    fig = go.Figure(go.Heatmap(z=z, x=pm["positions"], y=teams, colorscale=mv_ui.HEAT, zmin=0,
                               xgap=1, ygap=1,
                               hovertemplate="%{y} — %{x}º: %{z:.1f}%<extra></extra>",
                               colorbar=dict(title="%", outlinewidth=0,
                                             tickfont=dict(color=mv_ui.C["dim"], size=10))))
    fig.update_layout(height=max(26 * len(teams), 300), margin=dict(l=8, r=8, t=8, b=8),
                      xaxis_title="Posición final", yaxis=dict(autorange="reversed"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    chart(fig)


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
    fig.add_vline(x=highlight_md, line_dash="dot", line_color=mv_ui.C["dim"])
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="Jornada",
                      yaxis_title="Prob. de campeón (%)", yaxis_range=[0, 100],
                      legend=dict(orientation="h", y=-0.2),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    chart(fig)


def _upcoming_fixtures(snap: dict, n: int = 12) -> None:
    rem = pd.DataFrame(snap["fixtures"]["remaining"])
    if rem.empty:
        st.caption("Sin partidos pendientes.")
        return
    # same fixture row as the Jornada page — one component, one look
    for r in rem.head(n).itertuples(index=False):
        st.markdown(
            mv_ui.fixture_row(_title(r.home_team), _title(r.away_team),
                              date=getattr(r, "date", None),
                              probs=(r.p_home, r.p_draw, r.p_away),
                              note=f"λ {r.lambda_home:.2f} – {r.lambda_away:.2f}"),
            unsafe_allow_html=True)


# ── Entry point ─────────────────────────────────────────────────────────────────

def render() -> None:
    from mundialytics.statistical_core.competition import forecast_cache as fc

    st.title("📊  Pronóstico de liga (desde el punto actual)")
    st.caption("Arrastra la jornada para ver cómo evolucionan las probabilidades de título, Champions y descenso "
               "a lo largo de la temporada. Lectura instantánea desde caché.")

    c1, c2 = st.columns(2)
    comp = c1.selectbox("Competición", LEAGUES, key="cf_comp")
    season = c2.selectbox("Temporada", _seasons_for(comp), key="cf_season")

    # A season still being played cannot be read from the bundle: build_bundle
    # cuts a COMPLETED season at a series of matchdays, and the foundation holds
    # only played matches, so every cut on an in-progress season leaves nothing
    # to forecast. LaLiga 2026/27 came back as "matchday 37 of 38" after three
    # rounds. The live path takes the real remaining calendar instead.
    if _is_in_progress(comp, season):
        _render_live(comp, season)
        return

    bundle = fc.load_bundle(comp, season, cache_dir=CACHE_DIR)

    if bundle is None or bundle.get("meta", {}).get("schema") != fc.BUNDLE_SCHEMA:
        # built on first sight rather than on a button: the result is written to
        # CACHE_DIR, so this cost is paid once per season, ever
        with st.spinner(f"Generando el pronóstico de {comp} {season} jornada a jornada… "
                        "(una sola vez; luego la lectura es instantánea)"):
            try:
                bundle = fc.get_or_build(comp, season, _foundation(), timeline_step=5,
                                         n_sims=10000, cache_dir=CACHE_DIR)
            except Exception as exc:
                st.warning(f"No se pudo generar el pronóstico: {str(exc)[:140]}")
                return

    mds = fc.available_matchdays(bundle)
    default_md = bundle["meta"].get("current_matchday", max(mds))
    md_choice = st.slider("Jornada", min(mds), max(mds), default_md, key="cf_md")
    used_md, snap = fc.snapshot_for(bundle, md_choice)
    if used_md != md_choice:
        st.caption(f"Mostrando la jornada cacheada más cercana: **{used_md}**.")

    meta = bundle["meta"]
    st.caption(f"{mv_ui.num(meta['n_sims'])} simulaciones · {meta['model_note']}")

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
        st.caption("Cada fila es un equipo; cada columna, una posición final. "
                   "Más azul = más probable.")
        _position_heatmap(snap)
    with tabs[2]:
        st.caption("Probabilidad de título jornada a jornada (leakage-free en cada punto). La línea punteada marca la jornada seleccionada.")
        _evolution_chart(bundle, used_md)
    with tabs[3]:
        _upcoming_fixtures(snap)
    with tabs[4]:
        _render_standings(snap)


def _is_in_progress(comp: str, season: str) -> bool:
    """True when this league-season still has matches to play.

    Judged on the calendar rather than the clock: a season is in progress when
    the foundation holds fewer matches than a full double round-robin.
    """
    df = _foundation()
    sub = df[(df["competition"] == comp) & (df["season"] == season)]
    if sub.empty:
        return False
    teams = len(set(sub["home_team"]) | set(sub["away_team"]))
    return len(sub) < teams * (teams - 1)


def _live_cache_path(comp: str, season: str) -> Path:
    slug = f"{comp}_{season}".lower().replace(" ", "-").replace("/", "-")
    return CACHE_DIR / f"live_{slug}.json"


def _live_snapshot(comp: str, season: str) -> dict | None:
    """The live forecast for a season in progress, read from disk when it is current.

    The page used to open on a button because the simulation costs ~20s. A page
    that shows nothing until you ask it to work is not a page, so the snapshot is
    written to disk and re-read; it is recomputed only when new results have
    landed, which the played-match count detects. First visit after a matchday
    pays once, every visit after it is instant.
    """
    import json

    from mundialytics.statistical_core.competition import forecast_cache as fc

    found = _foundation()
    n_played = int(((found["competition"] == comp) & (found["season"] == season)).sum())
    path = _live_cache_path(comp, season)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("_n_played") == n_played:
                return cached
        except Exception:
            pass

    with st.spinner(f"Simulando lo que queda de {comp} {season}… (se guarda para "
                    "las próximas visitas)"):
        try:
            snap = fc.build_live_snapshot(comp, season, found, n_sims=10000, root=ROOT)
        except ValueError as exc:
            st.warning(str(exc))
            return None
        except Exception as exc:
            st.warning(f"No se pudo calcular: {str(exc)[:140]}")
            return None
    snap["_n_played"] = n_played
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, default=str), encoding="utf-8")
    except OSError:
        pass
    return snap


def _render_live(comp: str, season: str) -> None:
    """Forecast an in-progress season from where it actually stands."""
    # imported here, not taken from the caller: `fc` is a local import inside
    # render(), so referencing it from this function raised NameError and the
    # button reported "no se pudo calcular" for every league
    from mundialytics.statistical_core.competition import forecast_cache as fc

    snap = _live_snapshot(comp, season)
    if snap is None:
        return

    tp = pd.DataFrame(snap["forecast"]["team_probs"])
    st.caption(f"Jornada **{snap['matchday']}** · quedan **{snap['n_remaining']}** partidos "
               "· 10.000 simulaciones desde el estado real")

    stand = pd.DataFrame(snap["standings"])
    if not stand.empty:
        lead = stand.iloc[0]
        c = st.columns(3)
        c[0].metric("Líder", _title(lead["team"]), f"{int(lead['points'])} pts")
        if "p_champion" in tp.columns:
            # the headline the cached page leads with, and the one a reader
            # actually wants from a season in progress
            fav = tp.nlargest(1, "p_champion").iloc[0]
            c[1].metric("Favorito al título", _title(fav["team"]), f"{fav['p_champion']:.0%}")
        if "p_relegation" in tp.columns:
            rel = tp.nlargest(1, "p_relegation").iloc[0]
            c[2].metric("Más probable descenso", _title(rel["team"]), f"{rel['p_relegation']:.0%}")

    show = tp.copy()
    if "team" in show.columns:
        show["team"] = show["team"].map(_title)
    pct = [c for c in show.columns if c.startswith("p_")]
    for c in pct:
        show[c] = (show[c] * 100).round(1)
    # the forecaster emits exp_points/p_champion, not expected_points/p_title:
    # the old names matched nothing, so the table rendered raw column keys
    ren = {"team": "Equipo", "exp_points": "Puntos esp.", "p_top4": "Champions",
           "p_relegation": "Descenso", "p_champion": "Título", "p_top2": "Top 2",
           "exp_rank": "Puesto esp."}
    show = show.rename(columns={k: v for k, v in ren.items() if k in show.columns})
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={ren[c]: st.column_config.NumberColumn(format="%.1f%%")
                                for c in pct if c in ren})
