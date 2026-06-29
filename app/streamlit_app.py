"""
Mundialytics — Prediction Engine (Bloque 2)
Menu: ⚽ Partido | 🏆 Competición | 🥇 Premios Individuales
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from mundialytics.statistical_core.prediction_engine import PredictionEngine
from mundialytics.ratings.elo import EloRater, EloConfig


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundialytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 0.8rem; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 1.9rem; font-weight: 700; }
    .metric-home  { color: #3b82f6; }
    .metric-draw  { color: #6b7280; }
    .metric-away  { color: #ef4444; }
    .section-title { font-size: 1.1rem; font-weight: 600; margin: 1.2rem 0 0.5rem; }
    .pill {
        display: inline-block; padding: 2px 10px;
        border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    }
    .pill-green { background: #d1fae5; color: #065f46; }
    .pill-blue  { background: #dbeafe; color: #1e40af; }
    .pill-red   { background: #fee2e2; color: #991b1b; }
    .pill-gray  { background: #f3f4f6; color: #374151; }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos...")
def load_engine() -> tuple[PredictionEngine, list[str]]:
    df = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_goals", "away_goals"]).sort_values("date")

    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    elo_hist = elo.fit(df)

    engine = PredictionEngine(blend_weight_gl=0.60, ad_rho=-0.07)
    engine.fit(df, elo_history=pd.DataFrame(elo.history))

    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    return engine, teams


engine, ALL_TEAMS = load_engine()

COMPETITIONS = ["LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1",
                "World Cup", "UEFA Euro", "Champions League", "Copa del Rey", "Other"]

WC_GROUPS_2022 = {
    "A": ["qatar", "ecuador", "senegal", "netherlands"],
    "B": ["england", "iran", "usa", "wales"],
    "C": ["argentina", "saudi arabia", "mexico", "poland"],
    "D": ["france", "australia", "denmark", "tunisia"],
    "E": ["spain", "costa rica", "germany", "japan"],
    "F": ["belgium", "canada", "morocco", "croatia"],
    "G": ["brazil", "serbia", "switzerland", "cameroon"],
    "H": ["portugal", "ghana", "uruguay", "south korea"],
}

EURO_GROUPS_2024 = {
    "A": ["germany", "scotland", "hungary", "switzerland"],
    "B": ["spain", "croatia", "italy", "albania"],
    "C": ["slovenia", "denmark", "serbia", "england"],
    "D": ["poland", "netherlands", "austria", "france"],
    "E": ["belgium", "slovakia", "romania", "ukraine"],
    "F": ["turkey", "georgia", "portugal", "czech republic"],
}


# ── Helper visuals ─────────────────────────────────────────────────────────────

def prob_bar(p_home: float, p_draw: float, p_away: float, home: str, away: str):
    """Horizontal stacked bar for 1X2 probabilities."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[p_home], y=[""], orientation="h",
        name=home.title(), marker_color="#3b82f6",
        text=f"{p_home:.1%}", textposition="inside",
        hovertemplate=f"{home.title()}: {p_home:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[p_draw], y=[""], orientation="h",
        name="Empate", marker_color="#9ca3af",
        text=f"{p_draw:.1%}", textposition="inside",
        hovertemplate=f"Empate: {p_draw:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[p_away], y=[""], orientation="h",
        name=away.title(), marker_color="#ef4444",
        text=f"{p_away:.1%}", textposition="inside",
        hovertemplate=f"{away.title()}: {p_away:.1%}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack", showlegend=True,
        height=70, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def score_matrix_heatmap(matrix: pd.DataFrame, home: str, away: str) -> go.Figure:
    """Plotly heatmap of scoreline probabilities (%)."""
    max_show = 7
    mat = matrix.iloc[:max_show, :max_show] * 100
    z = mat.values
    x_labels = [str(i) for i in mat.columns]
    y_labels = [str(i) for i in mat.index]

    text = [[f"{v:.1f}%" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=False,
        hoverongaps=False,
        hovertemplate=f"{home.title()} %{{y}} – %{{x}} {away.title()}: %{{z:.2f}}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=f"Goles {away.title()}",
        yaxis_title=f"Goles {home.title()}",
        yaxis=dict(autorange="reversed"),
        height=360,
        margin=dict(l=60, r=20, t=20, b=60),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def event_gauge(label: str, home_val: float, away_val: float, home: str, away: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name=home.title(), x=[home_val], y=[label], orientation="h",
                          marker_color="#3b82f6", text=f"{home_val:.1f}", textposition="inside"))
    fig.add_trace(go.Bar(name=away.title(), x=[away_val], y=[label], orientation="h",
                          marker_color="#ef4444", text=f"{away_val:.1f}", textposition="inside"))
    fig.update_layout(barmode="group", height=80, showlegend=False,
                       margin=dict(l=0,r=0,t=0,b=0),
                       xaxis=dict(showgrid=False), yaxis=dict(showticklabels=False),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def tournament_bar(df: pd.DataFrame, col: str, title: str, color: str = "#3b82f6") -> go.Figure:
    top = df.head(16).sort_values(col)
    fig = go.Figure(go.Bar(
        y=top["team"].str.title(), x=top[col],
        orientation="h", marker_color=color,
        text=[f"{v:.1%}" for v in top[col]], textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(300, len(top) * 28),
        margin=dict(l=120, r=80, t=40, b=10),
        xaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚽ Mundialytics")
st.sidebar.markdown("Motor estadístico de predicción")
st.sidebar.divider()

page = st.sidebar.radio(
    "Modo",
    ["⚽  Partido", "🏆  Competición", "🥇  Premios Individuales"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Datos: Big5 2021-2026 · Modelo: Poisson GLM + MLE Attack/Defense + ELO")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PARTIDO
# ══════════════════════════════════════════════════════════════════════════════
if page == "⚽  Partido":
    st.title("⚽ Predicción de partido")

    # ── Team selectors ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([3, 1, 3, 2])
    with col1:
        home_default = ALL_TEAMS.index("barcelona") if "barcelona" in ALL_TEAMS else 0
        home = st.selectbox("Equipo local", ALL_TEAMS, index=home_default, key="home_team")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;margin-top:6px'>vs</h3>", unsafe_allow_html=True)
    with col3:
        away_default = ALL_TEAMS.index("real madrid") if "real madrid" in ALL_TEAMS else 1
        away = st.selectbox("Equipo visitante", ALL_TEAMS, index=away_default, key="away_team")
    with col4:
        competition = st.selectbox("Competición", COMPETITIONS, key="competition")
        neutral = st.checkbox("Campo neutro", value=False)

    if home == away:
        st.warning("Selecciona dos equipos distintos.")
        st.stop()

    pred = engine.predict_match(home, away, competition=competition, neutral=neutral)

    # ── 1X2 big bar ────────────────────────────────────────────────────────
    st.markdown("### Resultado")
    st.plotly_chart(prob_bar(pred.p_home_win, pred.p_draw, pred.p_away_win, home, away),
                    use_container_width=True)

    # ── Key metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (home.title(), f"{pred.p_home_win:.1%}", "metric-home"),
        ("Empate",      f"{pred.p_draw:.1%}",    "metric-draw"),
        (away.title(),  f"{pred.p_away_win:.1%}", "metric-away"),
        ("xGoals",      f"{pred.lambda_home:.2f} – {pred.lambda_away:.2f}", ""),
        ("BTTS",        f"{pred.p_btts:.1%}",     ""),
        ("Over 2.5",    f"{pred.p_over_25:.1%}",  ""),
    ]
    for col, (label, value, cls) in zip([c1,c2,c3,c4,c5,c6], metrics):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Score matrix + top scorelines ──────────────────────────────────────
    col_mat, col_scores = st.columns([3, 2])
    with col_mat:
        st.markdown('<div class="section-title">Matriz de resultados (%)</div>', unsafe_allow_html=True)
        st.plotly_chart(score_matrix_heatmap(pred.score_matrix, home, away), use_container_width=True)

    with col_scores:
        st.markdown('<div class="section-title">Resultados más probables</div>', unsafe_allow_html=True)
        scores_df = pd.DataFrame(pred.top_scorelines[:10])
        scores_df["prob_pct"] = scores_df["probability"].apply(lambda x: f"{x:.1%}")
        scores_df["bar"] = scores_df["probability"]
        for _, row in scores_df.iterrows():
            pct = row["probability"]
            bar_width = int(pct * 200)
            label = row["score"]
            st.markdown(
                f'<div style="display:flex;align-items:center;margin:3px 0;gap:8px">'
                f'<span style="width:40px;font-weight:600;font-size:1rem">{label}</span>'
                f'<div style="flex:1;background:#e5e7eb;border-radius:4px;height:18px">'
                f'<div style="width:{bar_width}px;max-width:100%;background:#3b82f6;'
                f'border-radius:4px;height:18px"></div></div>'
                f'<span style="width:44px;text-align:right;font-size:0.85rem;color:#6b7280">'
                f'{pct:.1%}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Over/Under lines ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">Mercados de goles</div>', unsafe_allow_html=True)
    oc1, oc2, oc3, oc4, oc5 = st.columns(5)
    ou_data = [
        ("Over 1.5", pred.p_over_15), ("Under 2.5", pred.p_under_25),
        ("Over 2.5", pred.p_over_25), ("Over 3.5", pred.p_over_35),
        ("BTTS", pred.p_btts),
    ]
    for col, (label, val) in zip([oc1,oc2,oc3,oc4,oc5], ou_data):
        color = "#16a34a" if val >= 0.55 else ("#dc2626" if val <= 0.45 else "#2563eb")
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color}">{val:.1%}</div></div>',
            unsafe_allow_html=True,
        )

    # ── Team event predictions ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Estadísticas esperadas por equipo</div>', unsafe_allow_html=True)
    ev_data = [
        ("Disparos",    pred.expected_shots_home,   pred.expected_shots_away),
        ("A puerta",    pred.expected_sot_home,     pred.expected_sot_away),
        ("Córners",     pred.expected_corners_home, pred.expected_corners_away),
        ("Faltas",      pred.expected_fouls_home,   pred.expected_fouls_away),
        ("Amarillas",   pred.expected_yellows_home, pred.expected_yellows_away),
    ]
    ev_cols = st.columns(len(ev_data))
    for col, (label, h_val, a_val) in zip(ev_cols, ev_data):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'background:var(--secondary-background-color);border-radius:8px;padding:8px 12px">'
                f'<span style="color:#3b82f6;font-weight:700;font-size:1.1rem">{h_val:.1f}</span>'
                f'<span style="color:#9ca3af;font-size:0.8rem">vs</span>'
                f'<span style="color:#ef4444;font-weight:700;font-size:1.1rem">{a_val:.1f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── ELO ratings ────────────────────────────────────────────────────────
    with st.expander("ℹ️ Fuerza de equipo (ELO)"):
        h_params = engine.ad_model_.team_params()
        h_row = h_params[h_params["team"] == home]
        a_row = h_params[h_params["team"] == away]
        c_h, c_a = st.columns(2)
        if not h_row.empty:
            r = h_row.iloc[0]
            c_h.markdown(f"**{home.title()}** — Ataque: `{r['attack']:+.3f}` · Defensa: `{r['defense']:+.3f}` · Fuerza: `{r['strength']:.3f}`")
        if not a_row.empty:
            r = a_row.iloc[0]
            c_a.markdown(f"**{away.title()}** — Ataque: `{r['attack']:+.3f}` · Defensa: `{r['defense']:+.3f}` · Fuerza: `{r['strength']:.3f}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: COMPETICIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆  Competición":
    st.title("🏆 Simulación de competición")

    mode = st.radio("Tipo", ["Liga", "Torneo (grupos + eliminatorias)"], horizontal=True)

    # ── LEAGUE MODE ────────────────────────────────────────────────────────
    if mode == "Liga":
        st.markdown("#### Selección de equipos")
        col_l, col_r = st.columns([2, 3])
        with col_l:
            competition_l = st.selectbox("Competición", COMPETITIONS, key="league_comp")
            n_sims_l = st.select_slider("Simulaciones", [1000, 10_000, 50_000, 100_000], value=10_000)
            home_away = st.checkbox("Partidos ida y vuelta", value=True)
            available = [t for t in ALL_TEAMS]
            default_liga = [t for t in ["real madrid","barcelona","atletico madrid","athletic bilbao",
                              "villarreal","real sociedad","girona","rayo vallecano",
                              "sevilla","betis"] if t in available][:8]
            selected_teams = st.multiselect("Equipos", available, default=default_liga, key="league_teams")

        with col_r:
            if len(selected_teams) >= 2:
                with st.spinner(f"Simulando {n_sims_l:,} temporadas..."):
                    result_l = engine.simulate_league(
                        selected_teams, n_sims=n_sims_l,
                        competition=competition_l, home_away=home_away,
                    )
                df_l = result_l.team_stats
                st.markdown(f"#### Resultados ({n_sims_l:,} simulaciones)")
                # Table
                display_cols = ["team", "p_win", "p_top2", "p_top4", "avg_pts", "avg_goals"]
                display_cols = [c for c in display_cols if c in df_l.columns]
                formatted = df_l[display_cols].copy()
                formatted["team"] = formatted["team"].str.title()
                pct_cols = [c for c in ["p_win","p_top2","p_top4"] if c in formatted.columns]
                for c in pct_cols:
                    formatted[c] = formatted[c].apply(lambda x: f"{x:.1%}")
                if "avg_pts" in formatted.columns:
                    formatted["avg_pts"] = formatted["avg_pts"].apply(lambda x: f"{x:.1f}")
                if "avg_goals" in formatted.columns:
                    formatted["avg_goals"] = formatted["avg_goals"].apply(lambda x: f"{x:.1f}")
                st.dataframe(formatted, hide_index=True, use_container_width=True)

                if "p_win" in df_l.columns:
                    st.plotly_chart(tournament_bar(df_l, "p_win", "% Campeón de liga"), use_container_width=True)
            else:
                st.info("Selecciona al menos 2 equipos.")

    # ── TOURNAMENT MODE ────────────────────────────────────────────────────
    else:
        col_t1, col_t2 = st.columns([2, 3])
        with col_t1:
            preset = st.selectbox("Plantilla", ["Personalizado", "World Cup 2022", "Euro 2024"], key="preset")
            competition_t = st.selectbox("Competición (modelo)", COMPETITIONS, key="tourn_comp")
            n_sims_t = st.select_slider("Simulaciones", [1_000, 10_000, 50_000, 100_000], value=10_000, key="nsims_t")
            bracket_fmt = st.selectbox("Formato bracket", ["auto","wc","euro","sequential"])
            neutral_t = st.checkbox("Campo neutro", value=True, key="neutral_t")

        if preset == "World Cup 2022":
            groups_raw = WC_GROUPS_2022
        elif preset == "Euro 2024":
            groups_raw = EURO_GROUPS_2024
        else:
            with col_t1:
                st.markdown("#### Grupos (un equipo por línea, grupos separados por línea vacía)")
                raw_text = st.text_area(
                    "Equipos",
                    value="spain\nfrance\ngermany\nportugal\n\nengland\nnetherlands\nbelgium\nitaly",
                    height=200, key="groups_text"
                )
                # Parse
                groups_raw = {}
                letter = "A"
                current = []
                for line in raw_text.strip().split("\n"):
                    line = line.strip().lower()
                    if line == "":
                        if current:
                            groups_raw[letter] = current
                            letter = chr(ord(letter) + 1)
                            current = []
                    elif line:
                        current.append(line)
                if current:
                    groups_raw[letter] = current

        # Normalize team names
        groups = {g: [t for t in teams if t in ALL_TEAMS] for g, teams in groups_raw.items()}
        groups = {g: teams for g, teams in groups.items() if len(teams) >= 2}

        with col_t2:
            if groups:
                st.markdown("#### Grupos configurados")
                for g, teams in groups.items():
                    st.markdown(f"**Grupo {g}**: " + ", ".join(t.title() for t in teams))

                with st.spinner(f"Simulando torneo ({n_sims_t:,} repeticiones)..."):
                    result_t = engine.simulate_tournament(
                        groups, knockout_slots=2, n_sims=n_sims_t,
                        competition=competition_t, neutral=neutral_t,
                        bracket_format=bracket_fmt,
                    )
                df_t = result_t.team_stats

                st.markdown(f"#### Resultados — {n_sims_t:,} simulaciones")
                # Format table
                show_cols = ["team","p_win","p_final","p_semis","p_quarters","p_advance_groups","avg_goals"]
                show_cols = [c for c in show_cols if c in df_t.columns]
                fmt = df_t[show_cols].copy()
                fmt["team"] = fmt["team"].str.title()
                pct_c = [c for c in show_cols if c.startswith("p_")]
                for c in pct_c:
                    fmt[c] = fmt[c].apply(lambda x: f"{x:.1%}")
                st.dataframe(fmt, hide_index=True, use_container_width=True)

                tab1, tab2, tab3 = st.tabs(["🏆 Campeón", "🎯 Llegar a final", "📈 Pasar grupos"])
                with tab1:
                    if "p_win" in df_t.columns:
                        st.plotly_chart(tournament_bar(df_t, "p_win", "% Campeón", "#f59e0b"), use_container_width=True)
                with tab2:
                    if "p_final" in df_t.columns:
                        st.plotly_chart(tournament_bar(df_t, "p_final", "% Llegar a la final", "#8b5cf6"), use_container_width=True)
                with tab3:
                    if "p_advance_groups" in df_t.columns:
                        st.plotly_chart(tournament_bar(df_t, "p_advance_groups", "% Pasar fase de grupos", "#10b981"), use_container_width=True)
            else:
                st.info("Ninguno de los equipos introducidos está en el dataset. Usa nombres en minúsculas como 'barcelona', 'real madrid'.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREMIOS INDIVIDUALES
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🥇 Premios Individuales")
    st.info("Para simular la Bota de Oro necesitas configurar una competición primero. "
            "Este módulo usa las tasas de gol por partido del PlayerProfileModel.")

    # ── Team goal leaders ──────────────────────────────────────────────────
    st.markdown("### Clasificación de equipos por xGoals esperados (partido)")
    col_gi, col_gf = st.columns([2, 3])
    with col_gi:
        comp_award = st.selectbox("Competición", COMPETITIONS, key="award_comp")
        selected_award_teams = st.multiselect(
            "Equipos a comparar",
            ALL_TEAMS,
            default=[t for t in ["real madrid","barcelona","man city","liverpool",
                                  "paris sg","inter","arsenal","juventus",
                                  "bayern munich","dortmund"] if t in ALL_TEAMS],
            key="award_teams"
        )

    with col_gf:
        if selected_award_teams:
            rows = []
            for team in selected_award_teams:
                # Average expected goals home and away
                avg_lh, avg_la, n = 0.0, 0.0, 0
                for opp in selected_award_teams:
                    if opp != team:
                        pred_tmp = engine.predict_match(team, opp, competition=comp_award, neutral=True)
                        avg_lh += pred_tmp.lambda_home
                        n += 1
                if n > 0:
                    avg_lh /= n
                ad_row = engine.ad_model_.team_params()
                ad_row = ad_row[ad_row["team"] == team]
                att = float(ad_row["attack"].iloc[0]) if not ad_row.empty else 0.0
                dfs = float(ad_row["defense"].iloc[0]) if not ad_row.empty else 0.0
                rows.append({
                    "Equipo": team.title(),
                    "xG medio/partido": round(avg_lh, 2),
                    "Ataque (MLE)": round(att, 3),
                    "Defensa (MLE)": round(dfs, 3),
                    "Fuerza total": round(att + dfs, 3),
                })
            df_awards = pd.DataFrame(rows).sort_values("xG medio/partido", ascending=False)
            st.dataframe(df_awards, hide_index=True, use_container_width=True)

            fig_a = go.Figure()
            top_a = df_awards.head(12)
            fig_a.add_trace(go.Bar(
                y=top_a["Equipo"], x=top_a["Ataque (MLE)"],
                name="Ataque", orientation="h", marker_color="#3b82f6",
            ))
            fig_a.add_trace(go.Bar(
                y=top_a["Equipo"], x=top_a["Defensa (MLE)"],
                name="Defensa", orientation="h", marker_color="#10b981",
            ))
            fig_a.update_layout(
                barmode="stack", title="Ataque + Defensa por equipo (parámetros MLE)",
                height=max(300, len(top_a)*28),
                margin=dict(l=120,r=40,t=50,b=20),
                xaxis_title="Parámetro (log-scale, relativo al promedio de su liga)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_a, use_container_width=True)

    # ── Player profile quick lookup ────────────────────────────────────────
    st.divider()
    st.markdown("### Perfil de jugador")
    st.caption("Tasas por partido basadas en StatsBomb open data (posiciones de GitHub lineups)")

    try:
        profiles_df = pd.read_csv(ROOT / "data/processed/player_profiles_with_positions.csv")
        all_players_p = sorted(profiles_df["player"].dropna().unique().tolist())
        player_search = st.selectbox("Buscar jugador", all_players_p,
                                      index=all_players_p.index("Lionel Andrés Messi Cuccittini")
                                      if "Lionel Andrés Messi Cuccittini" in all_players_p else 0,
                                      key="player_lookup")
        row_p = profiles_df[profiles_df["player"] == player_search]
        if not row_p.empty:
            r = row_p.iloc[0]
            pc1, pc2, pc3 = st.columns(3)
            pc1.markdown(f"**Equipo**: {str(r.get('team_c','')).title()}")
            pc1.markdown(f"**Competición**: {r.get('competition_c','')}")
            pc2.markdown(f"**Posición**: {r.get('position_group','')}")
            pc2.markdown(f"**Partidos**: {int(r.get('matches',0))}")

            stat_cols = [c for c in profiles_df.columns if c.endswith("_per_match")]
            stat_names = {
                "shots_per_match": "Disparos",
                "sot_per_match": "A puerta",
                "goals_per_match": "Goles",
                "assists_per_match": "Asistencias",
                "tackles_per_match": "Entradas",
                "fouls_per_match": "Faltas",
                "yellow_cards_per_match": "Amarillas",
                "pressures_per_match": "Presiones",
            }
            stat_data = [(stat_names.get(c, c), float(r.get(c, 0))) for c in stat_cols if c in r.index]
            stat_data.sort(key=lambda x: -x[1])
            fig_p = go.Figure(go.Bar(
                x=[s[0] for s in stat_data], y=[s[1] for s in stat_data],
                marker_color="#3b82f6",
                text=[f"{s[1]:.2f}" for s in stat_data], textposition="outside",
            ))
            fig_p.update_layout(
                title=f"Estadísticas por partido — {player_search.split()[0]}",
                height=280, margin=dict(l=20,r=20,t=50,b=20),
                yaxis_title="Por partido",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_p, use_container_width=True)
    except FileNotFoundError:
        st.warning("Perfil de jugadores no disponible. Ejecuta el fetch de posiciones primero.")
