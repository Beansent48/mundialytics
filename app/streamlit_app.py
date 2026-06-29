"""
Mundialytics — Prediction Engine · Block 2 + SquadLab
Menu: ⚽ Partido | 🗓️ Jornada | 🏆 Competición | 🥇 Premios | 🧪 SquadLab
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

from mundialytics.statistical_core.prediction_engine import PredictionEngine
from mundialytics.statistical_core.engine_utils import (
    load_clubs_data, load_international_data,
    get_h2h, get_form, h2h_summary, bracket_html,
)
from mundialytics.ratings.elo import EloRater, EloConfig

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mundialytics", page_icon="⚽", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
    .metric-card{background:var(--secondary-background-color);border-radius:10px;
        padding:14px 18px;text-align:center;height:90px;display:flex;
        flex-direction:column;justify-content:center}
    .metric-label{font-size:.75rem;color:#888;margin-bottom:3px}
    .metric-value{font-size:1.75rem;font-weight:700;line-height:1}
    .form-badge{display:inline-block;width:26px;height:26px;border-radius:50%;
        font-weight:700;font-size:.8rem;text-align:center;line-height:26px;margin:1px}
    .form-W{background:#16a34a;color:#fff}
    .form-D{background:#9ca3af;color:#fff}
    .form-L{background:#dc2626;color:#fff}
    h3{margin-top:1.2rem!important}
    div[data-testid="stSidebarNav"]{display:none}
</style>""", unsafe_allow_html=True)

# ── Tournament presets ─────────────────────────────────────────────────────────
WC_2022 = {
    "A": ["qatar","ecuador","senegal","netherlands"],
    "B": ["england","iran","usa","wales"],
    "C": ["argentina","saudi arabia","mexico","poland"],
    "D": ["france","australia","denmark","tunisia"],
    "E": ["spain","costa rica","germany","japan"],
    "F": ["belgium","canada","morocco","croatia"],
    "G": ["brazil","serbia","switzerland","cameroon"],
    "H": ["portugal","ghana","uruguay","south korea"],
}
EURO_2024 = {
    "A": ["germany","scotland","hungary","switzerland"],
    "B": ["spain","croatia","italy","albania"],
    "C": ["slovenia","denmark","serbia","england"],
    "D": ["poland","netherlands","austria","france"],
    "E": ["belgium","slovakia","romania","ukraine"],
    "F": ["turkey","georgia","portugal","czech republic"],
}
COPA_2024 = {
    "A": ["argentina","peru","chile","canada"],
    "B": ["ecuador","venezuela","mexico","jamaica"],
    "C": ["united states","uruguay","panama","bolivia"],
    "D": ["brazil","colombia","paraguay","costa rica"],
}
CA2025_QUALIF = {
    "A": ["spain","france","germany","portugal"],
    "B": ["england","netherlands","belgium","italy"],
}


# ── Engine loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Cargando modelos de clubes...")
def load_club_engine():
    df = load_clubs_data()
    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    elo_hist = elo.fit(df)
    engine = PredictionEngine(blend_weight_gl=0.60, ad_rho=-0.07)
    engine.fit(df, elo_history=pd.DataFrame(elo.history))
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return engine, teams, df


@st.cache_resource(show_spinner="🌍  Cargando modelos de selecciones...")
def load_intl_engine():
    df = load_international_data(min_year=2010)
    elo = EloRater(EloConfig(season_reset_fraction=0.35, k_base=28.0))
    elo_hist = elo.fit(df)
    engine = PredictionEngine(blend_weight_gl=0.45, ad_rho=-0.06)
    engine.fit(df, elo_history=pd.DataFrame(elo.history))
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return engine, teams, df


engine_clubs, CLUB_TEAMS, df_clubs = load_club_engine()
engine_intl,  INTL_TEAMS,  df_intl  = load_intl_engine()

ALL_TEAMS_COMBINED = sorted(set(CLUB_TEAMS) | set(INTL_TEAMS))
COMPETITIONS = ["LaLiga","Premier League","Serie A","Bundesliga","Ligue 1",
                "World Cup","UEFA Euro","Copa América","Champions League","Other"]


# ── Helper functions ───────────────────────────────────────────────────────────
def pick_engine(competition: str) -> tuple[PredictionEngine, list[str], pd.DataFrame]:
    intl_kw = {"world cup","euro","copa","nations","international","afcon","afc","concacaf"}
    if any(k in competition.lower() for k in intl_kw):
        return engine_intl, INTL_TEAMS, df_intl
    return engine_clubs, CLUB_TEAMS, df_clubs


def resolve_team(name: str, teams: list[str]) -> str:
    """Return closest matching team name or original if not found."""
    if name in teams:
        return name
    lower = name.lower()
    matches = [t for t in teams if lower in t or t in lower]
    return matches[0] if matches else name


def predict_safe(engine: PredictionEngine, home: str, away: str,
                 competition: str, neutral: bool):
    try:
        return engine.predict_match(home, away, competition=competition, neutral=neutral)
    except Exception:
        return None


def form_badges(form_list: list[dict]) -> str:
    html = ""
    for f in form_list:
        r = f["result"]
        html += f'<span class="form-badge form-{r}" title="{f["score"]} vs {f["opponent"].title()}">{r}</span>'
    return html


def metric_card(label: str, value: str, color: str = "") -> str:
    style = f"color:{color}" if color else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="{style}">{value}</div></div>')


def prob_bar_chart(p_home: float, p_draw: float, p_away: float,
                   home: str, away: str) -> go.Figure:
    fig = go.Figure()
    for name, val, color in [
        (home.title(), p_home, "#3b82f6"),
        ("Empate", p_draw, "#9ca3af"),
        (away.title(), p_away, "#ef4444"),
    ]:
        fig.add_trace(go.Bar(
            x=[val], y=[""], orientation="h", name=name,
            marker_color=color, text=f"{val:.1%}", textposition="inside",
            hovertemplate=f"{name}: {val:.1%}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=62, showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.1,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def score_heatmap(matrix: pd.DataFrame, home: str, away: str) -> go.Figure:
    n = min(7, len(matrix))
    mat = matrix.iloc[:n, :n] * 100
    fig = go.Figure(go.Heatmap(
        z=mat.values,
        x=[str(i) for i in mat.columns],
        y=[str(i) for i in mat.index],
        text=[[f"{v:.1f}%" for v in row] for row in mat.values],
        texttemplate="%{text}", colorscale="Blues", showscale=False,
        hovertemplate=f"{home.title()} %{{y}}–%{{x}} {away.title()}: %{{z:.2f}}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=f"⚽ {away.title()}", yaxis_title=f"⚽ {home.title()}",
        yaxis=dict(autorange="reversed"), height=340,
        margin=dict(l=55, r=10, t=10, b=55),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def tourn_bar(df: pd.DataFrame, col: str, title: str, color="#3b82f6") -> go.Figure:
    top = df[df[col].notna()].head(20).sort_values(col)
    fig = go.Figure(go.Bar(
        y=top["team"].str.title(), x=top[col], orientation="h",
        marker_color=color, text=[f"{v:.1%}" for v in top[col]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(280, len(top) * 26),
        margin=dict(l=120, r=80, t=40, b=10),
        xaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚽ Mundialytics")
st.sidebar.caption("Motor estadístico de predicción")
st.sidebar.divider()
page = st.sidebar.radio("", [
    "⚽  Partido",
    "🗓️  Jornada",
    "🏆  Competición",
    "🥇  Premios Individuales",
    "🧪  SquadLab",
], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Big5 2021-26 · Selecciones 2010-26\nPoisson GLM · MLE Attack/Defense · ELO")


# ══════════════════════════════════════════════════════════════════════════════
#  PARTIDO
# ══════════════════════════════════════════════════════════════════════════════
if page == "⚽  Partido":
    st.title("⚽  Predicción de partido")

    c1, c2, c3, c4 = st.columns([3, 1, 3, 2])
    with c1:
        hi = ALL_TEAMS_COMBINED.index("barcelona") if "barcelona" in ALL_TEAMS_COMBINED else 0
        home = st.selectbox("Equipo local", ALL_TEAMS_COMBINED, index=hi, key="p_home")
    with c2:
        st.markdown("<br><h3 style='text-align:center;margin:6px 0'>vs</h3>",
                    unsafe_allow_html=True)
    with c3:
        ai = ALL_TEAMS_COMBINED.index("real madrid") if "real madrid" in ALL_TEAMS_COMBINED else 1
        away = st.selectbox("Equipo visitante", ALL_TEAMS_COMBINED, index=ai, key="p_away")
    with c4:
        competition = st.selectbox("Competición", COMPETITIONS, key="p_comp")
        neutral = st.checkbox("Campo neutro", value=False, key="p_neutral")

    if home == away:
        st.warning("Selecciona dos equipos distintos.")
        st.stop()

    eng, eng_teams, df_hist = pick_engine(competition)
    pred = predict_safe(eng, home, away, competition, neutral)
    if pred is None:
        st.error("Error al generar predicción."); st.stop()

    # Fallback warning
    if home not in eng_teams or away not in eng_teams:
        missing = [t.title() for t in [home, away] if t not in eng_teams]
        st.info(f"ℹ️ {', '.join(missing)} no está en el dataset de entrenamiento. Se usa prior global.")

    # ── Prob bar ──────────────────────────────────────────────────────────
    st.markdown("### Resultado")
    st.plotly_chart(prob_bar_chart(pred.p_home_win, pred.p_draw, pred.p_away_win, home, away),
                    use_container_width=True)

    # ── Metric cards ──────────────────────────────────────────────────────
    cols = st.columns(6)
    data = [
        (home.title(),  f"{pred.p_home_win:.1%}", "#3b82f6"),
        ("Empate",      f"{pred.p_draw:.1%}",     "#6b7280"),
        (away.title(),  f"{pred.p_away_win:.1%}",  "#ef4444"),
        ("xGoals",      f"{pred.lambda_home:.2f} – {pred.lambda_away:.2f}", ""),
        ("BTTS",        f"{pred.p_btts:.1%}", "#16a34a" if pred.p_btts > 0.55 else ""),
        ("Over 2.5",    f"{pred.p_over_25:.1%}", "#16a34a" if pred.p_over_25 > 0.55 else ""),
    ]
    for col, (label, value, color) in zip(cols, data):
        col.markdown(metric_card(label, value, color), unsafe_allow_html=True)

    st.markdown("")

    # ── Score matrix + scorelines ──────────────────────────────────────────
    col_heat, col_scores = st.columns([3, 2])
    with col_heat:
        st.markdown("### Matriz de resultados (%)")
        st.plotly_chart(score_heatmap(pred.score_matrix, home, away), use_container_width=True)

    with col_scores:
        st.markdown("### Resultados más probables")
        for item in pred.top_scorelines[:8]:
            pct = item["probability"]
            bar = int(pct * 220)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                f'<span style="width:36px;font-weight:700;font-size:.95rem">{item["score"]}</span>'
                f'<div style="flex:1;background:#e5e7eb;border-radius:4px;height:16px">'
                f'<div style="width:{bar}px;max-width:100%;background:#3b82f6;border-radius:4px;height:16px"></div>'
                f'</div><span style="width:40px;text-align:right;color:#6b7280;font-size:.8rem">{pct:.1%}</span>'
                f'</div>', unsafe_allow_html=True)

    # ── Over/Under markets ────────────────────────────────────────────────
    st.markdown("### Mercados de goles")
    ocols = st.columns(5)
    ou = [("Over 1.5", pred.p_over_15), ("Under 2.5", pred.p_under_25),
          ("Over 2.5", pred.p_over_25), ("Over 3.5", pred.p_over_35),
          ("BTTS",     pred.p_btts)]
    for col, (label, val) in zip(ocols, ou):
        c = "#16a34a" if val >= 0.55 else ("#dc2626" if val <= 0.45 else "#2563eb")
        col.markdown(metric_card(label, f"{val:.1%}", c), unsafe_allow_html=True)

    # ── Team events ───────────────────────────────────────────────────────
    st.markdown("### Estadísticas esperadas")
    ev_cols = st.columns(5)
    events = [
        ("Disparos",  pred.expected_shots_home,   pred.expected_shots_away),
        ("A puerta",  pred.expected_sot_home,     pred.expected_sot_away),
        ("Córners",   pred.expected_corners_home, pred.expected_corners_away),
        ("Faltas",    pred.expected_fouls_home,   pred.expected_fouls_away),
        ("Amarillas", pred.expected_yellows_home, pred.expected_yellows_away),
    ]
    for col, (label, hv, av) in zip(ev_cols, events):
        col.markdown(f"**{label}**")
        col.markdown(
            f'<div style="display:flex;justify-content:space-between;background:'
            f'var(--secondary-background-color);border-radius:8px;padding:8px 12px">'
            f'<span style="color:#3b82f6;font-weight:700;font-size:1.05rem">{hv:.1f}</span>'
            f'<span style="color:#9ca3af;font-size:.8rem">vs</span>'
            f'<span style="color:#ef4444;font-weight:700;font-size:1.05rem">{av:.1f}</span></div>',
            unsafe_allow_html=True)

    # ── H2H + Form ────────────────────────────────────────────────────────
    st.markdown("---")
    col_h2h, col_form = st.columns(2)

    with col_h2h:
        st.markdown("### Historial H2H")
        summary = h2h_summary(home, away, df_hist)
        if summary["total"] > 0:
            w, d, l, tot = summary["w"], summary["d"], summary["l"], summary["total"]
            st.markdown(
                f'<div style="display:flex;gap:16px;margin-bottom:12px">'
                f'<span style="color:#3b82f6;font-weight:700;font-size:1.3rem">{w}V</span>'
                f'<span style="color:#6b7280;font-weight:700;font-size:1.3rem">{d}E</span>'
                f'<span style="color:#ef4444;font-weight:700;font-size:1.3rem">{l}D</span>'
                f'<span style="color:#9ca3af;font-size:.85rem;align-self:flex-end">'
                f'({tot} partidos)</span></div>',
                unsafe_allow_html=True)
            h2h_df = get_h2h(home, away, df_hist, n=6)
            for _, row in h2h_df.iterrows():
                res = row["result_t1"]
                c = "#16a34a" if res=="W" else ("#9ca3af" if res=="D" else "#ef4444")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                    f'border-bottom:1px solid #f3f4f6;font-size:.85rem">'
                    f'<span style="color:#6b7280">{str(row["date"])[:10]}</span>'
                    f'<span>{row["home_team"].title()} {row["score"]} {row["away_team"].title()}</span>'
                    f'<span style="color:{c};font-weight:700">{res}</span></div>',
                    unsafe_allow_html=True)
        else:
            st.caption("Sin historial disponible entre estos equipos.")

    with col_form:
        st.markdown(f"### Forma reciente")
        for team, color in [(home, "#3b82f6"), (away, "#ef4444")]:
            form = get_form(team, df_hist, n=5)
            badges = form_badges(form)
            st.markdown(
                f'<div style="margin-bottom:12px">'
                f'<span style="font-weight:600;color:{color}">{team.title()}</span> '
                f'<span style="font-size:.75rem;color:#9ca3af">(últimos {len(form)})</span><br>'
                f'<div style="margin-top:5px">{badges}</div></div>',
                unsafe_allow_html=True)
            if form:
                for f in form[:3]:
                    r_c = "#16a34a" if f["result"]=="W" else ("#9ca3af" if f["result"]=="D" else "#ef4444")
                    st.markdown(
                        f'<div style="font-size:.8rem;color:#6b7280;padding:1px 0">'
                        f'{f["date"]} · {f["result"]} {f["score"]} vs {f["opponent"].title()}'
                        f' <span style="color:#9ca3af">({f["home_away"]})</span></div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  JORNADA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗓️  Jornada":
    st.title("🗓️  Predicción de jornada")
    st.caption("Introduce los partidos de la jornada y obtén todas las predicciones de una vez.")

    col_in, col_out = st.columns([2, 3])
    with col_in:
        competition_j = st.selectbox("Competición", COMPETITIONS, key="j_comp")
        neutral_j = st.checkbox("Campo neutro", value=False, key="j_neutral")
        fixtures_raw = st.text_area(
            "Partidos (local vs visitante, uno por línea)",
            value="real madrid vs barcelona\nman city vs arsenal\nparis sg vs marseille\ninter vs juventus\nbayern munich vs dortmund",
            height=220, key="j_fixtures",
        )

    with col_out:
        fixtures = []
        for line in fixtures_raw.strip().split("\n"):
            line = line.strip().lower()
            if " vs " in line:
                h, a = line.split(" vs ", 1)
                fixtures.append((h.strip(), a.strip()))

        if not fixtures:
            st.info("Introduce partidos en el formato 'local vs visitante'.")
        else:
            eng_j, eng_teams_j, _ = pick_engine(competition_j)
            rows = []
            for home_j, away_j in fixtures:
                pred_j = predict_safe(eng_j, home_j, away_j, competition_j, neutral_j)
                if pred_j:
                    rows.append({
                        "Partido": f"{home_j.title()} vs {away_j.title()}",
                        "Local %": f"{pred_j.p_home_win:.1%}",
                        "Empate %": f"{pred_j.p_draw:.1%}",
                        "Visitante %": f"{pred_j.p_away_win:.1%}",
                        "xG L": f"{pred_j.lambda_home:.2f}",
                        "xG V": f"{pred_j.lambda_away:.2f}",
                        "O2.5": f"{pred_j.p_over_25:.1%}",
                        "BTTS": f"{pred_j.p_btts:.1%}",
                        "Más probable": pred_j.top_scorelines[0]["score"] if pred_j.top_scorelines else "—",
                    })

            if rows:
                st.markdown(f"### {len(rows)} partido{'s' if len(rows)>1 else ''}")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                # Visual summary
                st.markdown("### % Victoria local por partido")
                df_j = pd.DataFrame(rows)
                home_probs = [float(r.split("%")[0]) / 100 for r in df_j["Local %"]]
                matches_labels = [r["Partido"].split(" vs ")[0] + " H" for r in rows]
                fig_j = go.Figure(go.Bar(
                    x=matches_labels, y=home_probs,
                    marker_color=["#16a34a" if p > 0.5 else ("#9ca3af" if p > 0.38 else "#dc2626")
                                  for p in home_probs],
                    text=[f"{p:.0%}" for p in home_probs], textposition="outside",
                ))
                fig_j.update_layout(
                    height=300, yaxis=dict(tickformat=".0%", range=[0, 1]),
                    margin=dict(l=20, r=20, t=20, b=60),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_j, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  COMPETICIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆  Competición":
    st.title("🏆  Simulación de competición")

    mode = st.radio("Tipo", ["Liga", "Torneo (grupos + eliminatorias)"], horizontal=True, key="comp_mode")

    # ── LIGA ──────────────────────────────────────────────────────────────
    if mode == "Liga":
        col_l1, col_l2 = st.columns([2, 3])
        with col_l1:
            competition_l = st.selectbox("Competición", COMPETITIONS, key="liga_comp")
            eng_l, teams_l, _ = pick_engine(competition_l)
            n_sims_l = st.select_slider("Simulaciones", [1_000, 10_000, 50_000, 100_000], value=10_000, key="liga_n")
            home_away_l = st.checkbox("Ida y vuelta", value=True, key="liga_ha")

            # Smart defaults based on competition
            if "laliga" in competition_l.lower() or "la liga" in competition_l.lower():
                default_t = [t for t in ["real madrid","barcelona","atletico madrid","athletic bilbao",
                                          "villarreal","real sociedad","girona","rayo vallecano"] if t in teams_l]
            elif "premier" in competition_l.lower():
                default_t = [t for t in ["man city","arsenal","liverpool","chelsea",
                                          "tottenham","man united","newcastle","aston villa"] if t in teams_l]
            elif "bundesliga" in competition_l.lower():
                default_t = [t for t in ["bayern munich","dortmund","leverkusen","rb leipzig",
                                          "frankfurt","stuttgart","freiburg","wolfsburg"] if t in teams_l]
            elif "ligue" in competition_l.lower():
                default_t = [t for t in ["paris sg","marseille","lyon","monaco",
                                          "lens","lille","nice","rennes"] if t in teams_l]
            elif "serie" in competition_l.lower():
                default_t = [t for t in ["inter","juventus","milan","napoli",
                                          "roma","lazio","atalanta","bologna"] if t in teams_l]
            else:
                default_t = teams_l[:8]

            selected_l = st.multiselect("Equipos", teams_l, default=default_t[:8], key="liga_teams")

        with col_l2:
            if len(selected_l) >= 2:
                with st.spinner(f"Simulando {n_sims_l:,} temporadas..."):
                    res_l = engine_clubs.simulate_league(
                        selected_l, n_sims=n_sims_l, competition=competition_l, home_away=home_away_l
                    )
                df_l = res_l.team_stats.copy()
                df_l["team"] = df_l["team"].str.title()

                st.markdown(f"### Resultados — {n_sims_l:,} simulaciones")
                show_l = [c for c in ["team","p_win","p_top2","p_top4","avg_pts","avg_goals"] if c in df_l.columns]
                fmt_l = df_l[show_l].copy()
                for c in ["p_win","p_top2","p_top4"]:
                    if c in fmt_l: fmt_l[c] = fmt_l[c].apply(lambda x: f"{x:.1%}")
                if "avg_pts" in fmt_l: fmt_l["avg_pts"] = fmt_l["avg_pts"].apply(lambda x: f"{x:.1f}")
                if "avg_goals" in fmt_l: fmt_l["avg_goals"] = fmt_l["avg_goals"].apply(lambda x: f"{x:.1f}")
                fmt_l.columns = [c.replace("_"," ").title() for c in fmt_l.columns]
                st.dataframe(fmt_l, hide_index=True, use_container_width=True)

                if "p_win" in res_l.team_stats.columns:
                    st.plotly_chart(tourn_bar(res_l.team_stats, "p_win", "% Campeón de liga", "#f59e0b"),
                                    use_container_width=True)
            else:
                st.info("Selecciona al menos 2 equipos.")

    # ── TORNEO ────────────────────────────────────────────────────────────
    else:
        col_t1, col_t2 = st.columns([2, 3])
        with col_t1:
            preset = st.selectbox("Plantilla", [
                "Personalizado", "World Cup 2022", "Euro 2024", "Copa América 2024"
            ], key="preset")
            competition_t = st.selectbox("Competición (modelo)", COMPETITIONS, key="t_comp")
            eng_t, teams_t, _ = pick_engine(competition_t)
            n_sims_t = st.select_slider("Simulaciones", [1_000, 10_000, 50_000, 100_000], value=10_000, key="t_n")
            bracket_fmt = st.selectbox("Formato bracket", ["auto","wc","euro","sequential"], key="t_fmt")
            neutral_t = st.checkbox("Campo neutro", value=True, key="t_neutral")

        if preset == "World Cup 2022":     groups_raw = WC_2022
        elif preset == "Euro 2024":        groups_raw = EURO_2024
        elif preset == "Copa América 2024": groups_raw = COPA_2024
        else:
            with col_t1:
                raw_txt = st.text_area(
                    "Grupos (equipo por línea, línea vacía = nuevo grupo)",
                    value="spain\nfrance\ngermany\nportugal\n\nengland\nnetherlands\nbelgium\nitaly",
                    height=200, key="t_custom")
                groups_raw = {}
                letter, cur = "A", []
                for line in raw_txt.strip().split("\n"):
                    line = line.strip().lower()
                    if not line:
                        if cur: groups_raw[letter] = cur; letter = chr(ord(letter)+1); cur = []
                    else:
                        cur.append(line)
                if cur: groups_raw[letter] = cur

        groups = {g: [t for t in teams if t in teams_t] for g, teams in groups_raw.items()}
        groups = {g: t for g, t in groups.items() if len(t) >= 2}

        with col_t2:
            if not groups:
                st.warning("Ningún equipo del preset está en el dataset seleccionado.")
                missing = [t for teams in groups_raw.values() for t in teams if t not in teams_t]
                if missing:
                    st.caption(f"Equipos no encontrados: {', '.join(sorted(set(missing))[:15])}")
            else:
                # Show groups
                for g, tms in groups.items():
                    st.markdown(f"**Grupo {g}**: " + "  ·  ".join(t.title() for t in tms))

                with st.spinner(f"Simulando torneo ({n_sims_t:,} veces)..."):
                    res_t = eng_t.simulate_tournament(
                        groups, knockout_slots=2, n_sims=n_sims_t,
                        competition=competition_t, neutral=neutral_t,
                        bracket_format=bracket_fmt,
                    )
                df_t = res_t.team_stats

                st.markdown(f"### Resultados — {n_sims_t:,} simulaciones")
                # Bracket-style table
                n_groups = len(groups)
                st.markdown(bracket_html(df_t, n_groups), unsafe_allow_html=True)

                st.markdown("---")
                tab1, tab2, tab3, tab4 = st.tabs(["🏆 Campeón", "🎯 Final", "🏅 Semifinal", "📈 Grupos"])
                with tab1:
                    if "p_win" in df_t: st.plotly_chart(tourn_bar(df_t, "p_win", "% Campeón", "#f59e0b"), use_container_width=True)
                with tab2:
                    if "p_final" in df_t: st.plotly_chart(tourn_bar(df_t, "p_final", "% Llegar a la Final", "#8b5cf6"), use_container_width=True)
                with tab3:
                    if "p_semis" in df_t: st.plotly_chart(tourn_bar(df_t, "p_semis", "% Llegar a Semifinales", "#3b82f6"), use_container_width=True)
                with tab4:
                    if "p_advance_groups" in df_t: st.plotly_chart(tourn_bar(df_t, "p_advance_groups", "% Pasar Fase de Grupos", "#10b981"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PREMIOS INDIVIDUALES
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🥇  Premios Individuales")

    tab_gb, tab_profile, tab_teams = st.tabs(["🥾 Bota de Oro", "👤 Perfil de jugador", "📊 Ranking de equipos"])

    # ── BOTA DE ORO ───────────────────────────────────────────────────────
    with tab_gb:
        st.markdown("### Simulación de Bota de Oro")
        st.caption("Usa las tasas de gol por partido del PlayerProfileModel combinadas con la simulación de torneo.")

        col_gb1, col_gb2 = st.columns([2, 3])
        with col_gb1:
            competition_gb = st.selectbox("Competición", COMPETITIONS, key="gb_comp")
            eng_gb, teams_gb, _ = pick_engine(competition_gb)
            n_sims_gb = st.select_slider("Simulaciones", [1_000, 5_000, 10_000, 50_000], value=5_000, key="gb_n")
            preset_gb = st.selectbox("Torneo", ["World Cup 2022","Euro 2024","Copa América 2024","Personalizado"], key="gb_preset")

            # Player goal rates
            try:
                profiles = pd.read_csv(ROOT / "data/processed/player_profiles_with_positions.csv")
                all_players = sorted(profiles["player"].dropna().unique())
                default_players = [p for p in [
                    "Lionel Andrés Messi Cuccittini", "Kylian Mbappé Lottin",
                    "Cristiano Ronaldo dos Santos Aveiro", "Neymar Jr",
                ] if p in all_players]
                selected_players = st.multiselect("Jugadores a seguir", all_players,
                                                   default=default_players[:4], key="gb_players")
            except FileNotFoundError:
                st.warning("Ejecuta primero el fetch de posiciones.")
                selected_players = []

        with col_gb2:
            if not selected_players:
                st.info("Selecciona jugadores para la simulación.")
            else:
                if preset_gb == "World Cup 2022":   groups_gb = WC_2022
                elif preset_gb == "Euro 2024":       groups_gb = EURO_2024
                elif preset_gb == "Copa América 2024": groups_gb = COPA_2024
                else:
                    groups_gb = {"A": ["spain","france","germany","portugal"],
                                  "B": ["england","netherlands","belgium","italy"]}

                groups_gb_f = {g: [t for t in ts if t in teams_gb] for g, ts in groups_gb.items()}
                groups_gb_f = {g: t for g, t in groups_gb_f.items() if len(t) >= 2}

                # Build player_goals dict
                player_goals_map = {}
                profiles_local = pd.read_csv(ROOT / "data/processed/player_profiles_with_positions.csv")
                for player_name in selected_players:
                    row = profiles_local[profiles_local["player"] == player_name]
                    if row.empty: continue
                    r = row.iloc[0]
                    team = str(r.get("team_c", "")).lower()
                    goals_pm = float(r.get("goals_per_match", r.get("goals_per90", 0.15)))
                    player_goals_map[player_name] = {team: goals_pm}

                if groups_gb_f and player_goals_map:
                    with st.spinner("Simulando Bota de Oro..."):
                        res_gb = eng_gb.simulate_tournament(
                            groups_gb_f, n_sims=n_sims_gb,
                            competition=competition_gb, neutral=True,
                            player_goals=player_goals_map,
                        )

                    if not res_gb.golden_boot.empty:
                        gb_df = res_gb.golden_boot.copy()
                        gb_df["player"] = gb_df["player"].str.split().str[0]  # first name only
                        st.markdown("#### 🥾 Bota de Oro — ranking")
                        st.dataframe(gb_df.head(10), hide_index=True, use_container_width=True)

                        fig_gb = go.Figure(go.Bar(
                            y=gb_df.head(8)["player"][::-1],
                            x=gb_df.head(8)["avg_goals_tournament"][::-1],
                            orientation="h", marker_color="#f59e0b",
                            text=[f"{v:.2f}" for v in gb_df.head(8)["avg_goals_tournament"][::-1]],
                            textposition="outside",
                        ))
                        fig_gb.update_layout(
                            title="Goles medios en el torneo",
                            height=280, margin=dict(l=100,r=60,t=40,b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_gb, use_container_width=True)
                    else:
                        st.info("Sin datos de Bota de Oro suficientes.")
                else:
                    st.warning("Equipos del torneo no encontrados en el motor seleccionado.")

    # ── PERFIL DE JUGADOR ─────────────────────────────────────────────────
    with tab_profile:
        st.markdown("### 👤 Perfil estadístico de jugador")
        try:
            profiles_p = pd.read_csv(ROOT / "data/processed/player_profiles_with_positions.csv")
            all_p = sorted(profiles_p["player"].dropna().unique())
            idx_def = all_p.index("Lionel Andrés Messi Cuccittini") if "Lionel Andrés Messi Cuccittini" in all_p else 0
            player_sel = st.selectbox("Jugador", all_p, index=idx_def, key="profile_player")
            row_p = profiles_p[profiles_p["player"] == player_sel]

            if not row_p.empty:
                r = row_p.iloc[0]
                c_info, c_chart = st.columns([1, 2])
                with c_info:
                    st.markdown(f"**Equipo:** {str(r.get('team_c','')).title()}")
                    st.markdown(f"**Posición:** {r.get('position_group','—')}")
                    st.markdown(f"**Competición:** {r.get('competition_c','—')}")
                    st.markdown(f"**Partidos:** {int(r.get('matches', 0))}")
                    st.markdown(f"**Confianza:** {min(100, int(r.get('matches',0)/50*100))}%")
                with c_chart:
                    stat_map = {
                        "shots_per_match":"Disparos","sot_per_match":"A puerta",
                        "goals_per_match":"Goles","assists_per_match":"Asistencias",
                        "tackles_per_match":"Entradas","fouls_per_match":"Faltas",
                        "yellow_cards_per_match":"Amarillas","pressures_per_match":"Presiones",
                    }
                    vals = [(label, float(r.get(col, 0))) for col, label in stat_map.items() if col in r.index]
                    vals_sorted = sorted(vals, key=lambda x: -x[1])
                    fig_p = go.Figure(go.Bar(
                        x=[v[0] for v in vals_sorted], y=[v[1] for v in vals_sorted],
                        marker_color="#3b82f6",
                        text=[f"{v[1]:.2f}" for v in vals_sorted], textposition="outside",
                    ))
                    fig_p.update_layout(
                        height=260, margin=dict(l=10,r=10,t=10,b=50),
                        yaxis_title="Por partido",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

                # Compare with position median
                pos = str(r.get("position_group","Unknown"))
                pos_median = profiles_p[profiles_p["position_group"] == pos]
                if not pos_median.empty:
                    st.markdown(f"**vs mediana de {pos}s:**")
                    compare_cols = [c for c in stat_map.keys() if c in profiles_p.columns]
                    compare_rows = []
                    for col in compare_cols:
                        player_val = float(r.get(col, 0))
                        median_val = float(pos_median[col].median())
                        if median_val > 0:
                            pct_diff = (player_val - median_val) / median_val
                            arrow = "↑" if pct_diff > 0.05 else ("↓" if pct_diff < -0.05 else "→")
                            compare_rows.append({
                                "Estadística": stat_map[col],
                                "Jugador": f"{player_val:.2f}",
                                "Mediana pos.": f"{median_val:.2f}",
                                "Diferencia": f"{arrow} {pct_diff:+.0%}",
                            })
                    if compare_rows:
                        st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)
        except FileNotFoundError:
            st.warning("Perfil de jugadores no disponible.")

    # ── RANKING DE EQUIPOS ────────────────────────────────────────────────
    with tab_teams:
        st.markdown("### 📊 Ranking de equipos por parámetros MLE")
        comp_rank = st.selectbox("Competición", COMPETITIONS, key="rank_comp")
        eng_rank, _, _ = pick_engine(comp_rank)

        params = eng_rank.ad_model_.team_params().copy()
        params["team"] = params["team"].str.title()
        params_show = params[["team","attack","defense","strength","attack_balance","matches"]].copy()
        params_show.columns = ["Equipo","Ataque","Defensa","Fuerza total","Balance ofensivo","Partidos"]

        st.dataframe(params_show.head(30), hide_index=True, use_container_width=True)

        fig_rank = go.Figure()
        top_r = params.head(20)
        fig_rank.add_trace(go.Bar(y=top_r["team"][::-1], x=top_r["attack"][::-1],
                                   name="Ataque", orientation="h", marker_color="#3b82f6"))
        fig_rank.add_trace(go.Bar(y=top_r["team"][::-1], x=top_r["defense"][::-1],
                                   name="Defensa", orientation="h", marker_color="#10b981"))
        fig_rank.update_layout(
            barmode="stack", title="Ataque + Defensa por equipo (parámetros MLE, escala logarítmica)",
            height=max(350, len(top_r)*22),
            margin=dict(l=120,r=40,t=50,b=20),
            xaxis_title="Parámetro MLE (relativo a la media de su liga)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rank, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SQUADLAB
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  SquadLab":
    import importlib, app.squadlab_page as sl_page
    importlib.reload(sl_page)
    sl_page.render(engine=engine_clubs)
