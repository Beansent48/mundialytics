"""
Mundialytics — Prediction Engine · Block 2 + SquadLab
Menu: 🗓️ Jornada | 🏆 Competición | 🥇 Premios | 🧪 SquadLab
"""
from __future__ import annotations
import sys
import importlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

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
    .fixture-row{padding:10px 14px;border-radius:8px;margin:4px 0;
        background:var(--secondary-background-color)}
    .fixture-row:hover{background:#e5e7eb22}
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
CHAMPIONS_LEAGUE_TOP = [
    "real madrid","barcelona","man city","bayern munich","liverpool","arsenal",
    "inter","paris sg","dortmund","atletico madrid","juventus","leverkusen",
    "milan","napoli","chelsea","aston villa","leipzig","benfica",
]

# ── Competition catalogue (knows its own format — no user prompt needed) ───────
COMP_CONFIG = {
    "LaLiga":           {"type": "liga",   "engine": "clubs", "comp_id": "LaLiga"},
    "Premier League":   {"type": "liga",   "engine": "clubs", "comp_id": "Premier League"},
    "Serie A":          {"type": "liga",   "engine": "clubs", "comp_id": "Serie A"},
    "Bundesliga":       {"type": "liga",   "engine": "clubs", "comp_id": "Bundesliga"},
    "Ligue 1":          {"type": "liga",   "engine": "clubs", "comp_id": "Ligue 1"},
    "Champions League": {"type": "torneo", "engine": "clubs", "groups": None, "teams": CHAMPIONS_LEAGUE_TOP},
    "World Cup":        {"type": "torneo", "engine": "intl",  "groups": WC_2022},
    "UEFA Euro":        {"type": "torneo", "engine": "intl",  "groups": EURO_2024},
    "Copa América":     {"type": "torneo", "engine": "intl",  "groups": COPA_2024},
}
COMPETITIONS = list(COMP_CONFIG.keys())
LEAGUE_COMPETITIONS = [c for c, cfg in COMP_CONFIG.items() if cfg["type"] == "liga"]
TOURNAMENT_COMPETITIONS = [c for c, cfg in COMP_CONFIG.items() if cfg["type"] == "torneo"]


# ── Engine loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Cargando modelos de clubes...")
def load_club_engine():
    df = load_clubs_data()
    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    elo_hist = elo.fit(df)
    # blend_weight_gl 0.30 (was 0.60): 8-fold backtest — goals-AttackDefense
    # deserves ~70%, GL ~30% (0.30 beat 0.60 in 8/8 folds). See project_xg_modeling_findings.
    # sharpen_gamma_1x2=1.2: LOFO-validated 1X2 calibration (RPS/LL/ECE all improve).
    engine = PredictionEngine(blend_weight_gl=0.30, ad_rho=-0.07, sharpen_gamma_1x2=1.3,
                              rescale_lambda_to_goals=True, outcome_rho=-0.17,
                              xg_rate_kwargs={"use_ewma": True})
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


@st.cache_resource(show_spinner="🎯  Cargando modelos de props...")
def load_props_models():
    """Team-event + player-prop models (clubs only). Fails soft: (None, None)."""
    try:
        from mundialytics.props import PlayerPropsModel, TeamPropsModel
        tp = TeamPropsModel().fit(df_clubs, root=ROOT)
    except Exception:
        return None, None
    try:
        pmp = ROOT / "data/external/advanced/understat/understat_player_match.csv"
        tmx = (pd.read_csv(ROOT / "data/processed/understat_team_match_xg.csv")
               [["provider_match_id", "date"]]
               .rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id"))
        pm = pd.read_csv(pmp).merge(tmx, on="game_id", how="left")
        pp = PlayerPropsModel().fit(pm)
    except Exception:
        pp = None
    return tp, pp


# ── Helper functions ───────────────────────────────────────────────────────────
def pick_engine(competition: str) -> tuple[PredictionEngine, list[str], pd.DataFrame]:
    cfg = COMP_CONFIG.get(competition)
    if cfg and cfg["engine"] == "intl":
        return engine_intl, INTL_TEAMS, df_intl
    if cfg and cfg["engine"] == "clubs":
        return engine_clubs, CLUB_TEAMS, df_clubs
    intl_kw = {"world cup","euro","copa","nations","international","afcon","afc","concacaf"}
    if any(k in competition.lower() for k in intl_kw):
        return engine_intl, INTL_TEAMS, df_intl
    return engine_clubs, CLUB_TEAMS, df_clubs


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


def render_stat_grid(events: list[tuple[str, float, float]]) -> None:
    """Shared 'home vs away' stat-card row, used for both real and predicted stats."""
    ev_cols = st.columns(len(events))
    for col, (label, hv, av) in zip(ev_cols, events):
        col.markdown(f"**{label}**")
        col.markdown(
            f'<div style="display:flex;justify-content:space-between;background:'
            f'var(--secondary-background-color);border-radius:8px;padding:8px 12px">'
            f'<span style="color:#3b82f6;font-weight:700;font-size:1.05rem">{hv:.1f}</span>'
            f'<span style="color:#9ca3af;font-size:.8rem">vs</span>'
            f'<span style="color:#ef4444;font-weight:700;font-size:1.05rem">{av:.1f}</span></div>',
            unsafe_allow_html=True)


MARKET_ES = {"corners": "Córners", "yellows": "Amarillas", "fouls": "Faltas",
             "shots": "Disparos", "sot": "A puerta"}


def render_props_section(home: str, away: str, pred) -> None:
    """O/U team-event markets + per-player props (validated walk-forward models)."""
    tp, pp = load_props_models()
    if tp is None:
        return
    referee = None
    if tp.known_referees and tp.is_epl_fixture(home, away):
        referee = st.selectbox(
            "Árbitro (opcional — mejora tarjetas y faltas)",
            ["—"] + tp.known_referees, key=f"ref_{home}_{away}")
        referee = None if referee == "—" else referee
    fx = tp.predict_fixture(home, away, referee=referee)
    players = pd.DataFrame()
    if pp is not None:
        try:
            players = pp.predict_fixture(home, away, lam_home=pred.lambda_home,
                                         lam_away=pred.lambda_away)
        except Exception:
            players = pd.DataFrame()
    if not fx and players.empty:
        return

    st.markdown("### 🎯 Props del partido")
    if fx:
        rows = []
        for mk, d in fx.items():
            r = {"Mercado": MARKET_ES.get(mk, mk), "λ Local": d["lambda_home"],
                 "λ Visitante": d["lambda_away"], "λ Total": d["lambda_total"]}
            for ln, p in d["over"].items():
                r[f"Over {ln}"] = f"{p:.0%}"
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Prob. de superar cada línea en el TOTAL del partido "
                   "(binomial negativa con sobre-dispersión medida por mercado).")

    if not players.empty:
        st.markdown("#### Props de jugadores")
        view = players[players["exp_min"] >= 30].copy()
        view["Equipo"] = np.where(view["side"] == "home", home.title(), away.title())
        cols = {"player": "Jugador", "Equipo": "Equipo", "exp_min": "Min esp.",
                "p_anytime_scorer": "Gol (anytime)", "p_2plus_goals": "2+ goles",
                "p_shots_over_1_5": "+1.5 tiros", "p_shots_over_2_5": "+2.5 tiros",
                "p_assist": "Asistencia", "p_yellow": "Amarilla"}
        tbl = view[list(cols)].rename(columns=cols)
        for c in ["Gol (anytime)", "2+ goles", "+1.5 tiros", "+2.5 tiros", "Asistencia", "Amarilla"]:
            tbl[c] = (tbl[c] * 100).round(1)
        st.dataframe(
            tbl, hide_index=True, use_container_width=True, height=420,
            column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                           for c in ["Gol (anytime)", "2+ goles", "+1.5 tiros",
                                     "+2.5 tiros", "Asistencia", "Amarilla"]})
        st.caption("Probabilidades condicionadas a que el jugador juegue (convención "
                   "bookmaker). Min esp. = minutos esperados si juega; ratios de "
                   "carrera + forma reciente, ajustados al contexto del partido.")


def render_real_match_stats(row: dict, home: str, away: str) -> None:
    """Actual historical stats for an already-played match — no model involved."""
    st.markdown(f"## ⚽ {home.title()} {int(row['home_goals'])} – {int(row['away_goals'])} {away.title()}")
    st.caption(f"Resultado y estadísticas reales · {str(row.get('date', ''))[:10]}")

    st.markdown("### Estadísticas reales del partido")
    events = [
        ("Disparos",  row.get("home_shots"),  row.get("away_shots")),
        ("A puerta",  row.get("home_sot"),    row.get("away_sot")),
        ("Córners",   row.get("home_corners"), row.get("away_corners")),
        ("Faltas",    row.get("home_fouls"),  row.get("away_fouls")),
        ("Amarillas", row.get("home_yellow_cards"), row.get("away_yellow_cards")),
    ]
    events = [(label, float(hv) if pd.notna(hv) else 0.0, float(av) if pd.notna(av) else 0.0)
              for label, hv, av in events]
    render_stat_grid(events)


def render_partido_detail(home: str, away: str, competition: str, neutral: bool,
                          actual_result: str | None = None, row: dict | None = None):
    """Full match-prediction breakdown, reused by the Jornada flow.

    When `row` is given (the match already has a real result), the real
    historical stats are shown by default instead of the model's
    predicted ones — the prediction breakdown becomes an opt-in toggle,
    kept around for the user's own testing/comparison, not the default view.
    """
    if row is not None:
        render_real_match_stats(row, home, away)
        show_prediction = st.checkbox(
            "🔮 Ver predicción del modelo (uso de pruebas)", key=f"pred_toggle_{home}_{away}")
        if not show_prediction:
            return
        st.markdown("---")
        st.caption("Predicción del modelo para este partido (no es lo que ocurrió realmente).")

    eng, eng_teams, df_hist = pick_engine(competition)
    pred = predict_safe(eng, home, away, competition, neutral)
    if pred is None:
        st.error("Error al generar predicción para este partido.")
        return

    if row is None:
        st.markdown(f"## ⚽ {home.title()} vs {away.title()}")
        if actual_result:
            st.markdown(f"**Resultado real:** {actual_result}")

    if home not in eng_teams or away not in eng_teams:
        missing = [t.title() for t in [home, away] if t not in eng_teams]
        st.info(f"ℹ️ {', '.join(missing)} no está en el dataset de entrenamiento. Se usa prior global.")

    st.markdown("### Resultado")
    st.plotly_chart(prob_bar_chart(pred.p_home_win, pred.p_draw, pred.p_away_win, home, away),
                    use_container_width=True)

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

    st.markdown("### Mercados de goles")
    ocols = st.columns(5)
    ou = [("Over 1.5", pred.p_over_15), ("Under 2.5", pred.p_under_25),
          ("Over 2.5", pred.p_over_25), ("Over 3.5", pred.p_over_35),
          ("BTTS",     pred.p_btts)]
    for col, (label, val) in zip(ocols, ou):
        c = "#16a34a" if val >= 0.55 else ("#dc2626" if val <= 0.45 else "#2563eb")
        col.markdown(metric_card(label, f"{val:.1%}", c), unsafe_allow_html=True)

    st.markdown("### Estadísticas esperadas")
    render_stat_grid([
        ("Disparos",  pred.expected_shots_home,   pred.expected_shots_away),
        ("A puerta",  pred.expected_sot_home,     pred.expected_sot_away),
        ("Córners",   pred.expected_corners_home, pred.expected_corners_away),
        ("Faltas",    pred.expected_fouls_home,   pred.expected_fouls_away),
        ("Amarillas", pred.expected_yellows_home, pred.expected_yellows_away),
    ])

    if eng is engine_clubs:
        render_props_section(home, away, pred)

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
        st.markdown("### Forma reciente")
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


def get_round_fixtures(competition: str, season: str, round_num: int, df: pd.DataFrame):
    """Derive a matchday's fixtures from chronologically-ordered season data."""
    cfg = COMP_CONFIG.get(competition, {})
    comp_id = cfg.get("comp_id", competition)
    mask = (df["competition"] == comp_id) & (df["season"] == season)
    df_season = df[mask].copy().sort_values("date").reset_index(drop=True)
    if df_season.empty:
        return pd.DataFrame(), 0
    n_teams = len(set(df_season["home_team"]) | set(df_season["away_team"]))
    n_per_round = max(1, n_teams // 2)
    total_rounds = max(1, -(-len(df_season) // n_per_round))
    start = (round_num - 1) * n_per_round
    return df_season.iloc[start:start + n_per_round], total_rounds


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚽ Mundialytics")
st.sidebar.caption("Motor estadístico de predicción")
st.sidebar.divider()
page = st.sidebar.radio("", [
    "🗓️  Jornada",
    "🏆  Competición",
    "📊  Pronóstico de liga",
    "🥇  Premios Individuales",
    "🧪  SquadLab",
], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Big5 2021-26 · Selecciones 2010-26\nPoisson GLM · MLE Attack/Defense · ELO")


# ══════════════════════════════════════════════════════════════════════════════
#  JORNADA  (competición → jornada → partido, todo encadenado)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🗓️  Jornada":
    st.title("🗓️  Jornada")
    st.caption("Elige una competición y una jornada para ver todos sus partidos y análisis.")

    if "j_selected" not in st.session_state:
        st.session_state.j_selected = None

    col_j1, col_j2, col_j3 = st.columns([2, 2, 1])
    with col_j1:
        comp_j = st.selectbox("Competición", LEAGUE_COMPETITIONS, key="j_comp",
                              on_change=lambda: st.session_state.update(j_selected=None))
    with col_j2:
        comp_id_j = COMP_CONFIG[comp_j]["comp_id"]
        seasons_j = sorted(df_clubs.loc[df_clubs["competition"] == comp_id_j, "season"].unique(), reverse=True)
        season_j = st.selectbox("Temporada", seasons_j, key="j_season",
                                on_change=lambda: st.session_state.update(j_selected=None))
    with col_j3:
        neutral_j = st.checkbox("Campo neutro", value=False, key="j_neutral")

    df_round, total_rounds = get_round_fixtures(comp_j, season_j, 1, df_clubs)
    if total_rounds == 0:
        st.warning("No hay datos disponibles para esta competición/temporada.")
        st.stop()

    jornada_num = st.slider("Jornada", 1, total_rounds, total_rounds, key="j_num",
                            on_change=lambda: st.session_state.update(j_selected=None))
    df_round, _ = get_round_fixtures(comp_j, season_j, jornada_num, df_clubs)

    if df_round.empty:
        st.info("Sin partidos para esta jornada.")
        st.stop()

    date_min, date_max = df_round["date"].min(), df_round["date"].max()
    st.markdown(f"### Jornada {jornada_num} de {total_rounds} "
               f"<span style='color:#9ca3af;font-size:.9rem'>"
               f"({date_min.strftime('%d %b')} – {date_max.strftime('%d %b %Y')})</span>",
               unsafe_allow_html=True)

    for i, (_, row) in enumerate(df_round.iterrows()):
        home, away = row["home_team"], row["away_team"]
        played = pd.notna(row["home_goals"]) and pd.notna(row["away_goals"])
        result_str = f"{int(row['home_goals'])} – {int(row['away_goals'])}" if played else "vs"

        c_home, c_score, c_away, c_btn = st.columns([3, 1, 3, 2])
        c_home.markdown(f"<div style='text-align:right;font-weight:600;padding-top:6px'>{home.title()}</div>",
                        unsafe_allow_html=True)
        c_score.markdown(f"<div style='text-align:center;font-weight:700;padding-top:6px;font-size:1.05rem'>{result_str}</div>",
                         unsafe_allow_html=True)
        c_away.markdown(f"<div style='text-align:left;font-weight:600;padding-top:6px'>{away.title()}</div>",
                        unsafe_allow_html=True)
        btn_label = "Ver resultado y análisis" if played else "Ver predicción"
        if c_btn.button(btn_label, key=f"j_btn_{i}"):
            st.session_state.j_selected = {
                "home": home, "away": away,
                "result": result_str if played else None,
                "row": row.to_dict() if played else None,
            }

    sel = st.session_state.j_selected
    if sel:
        st.markdown("---")
        render_partido_detail(sel["home"], sel["away"], comp_j, neutral_j,
                              actual_result=sel["result"], row=sel.get("row"))


# ══════════════════════════════════════════════════════════════════════════════
#  COMPETICIÓN  (formato auto-detectado, sin elegir liga/torneo a mano)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆  Competición":
    st.title("🏆  Simulación de competición")

    competition_c = st.selectbox("Competición", COMPETITIONS, key="comp_select")
    cfg = COMP_CONFIG[competition_c]
    eng_c, teams_c, _ = pick_engine(competition_c)

    # ── LIGA (formato detectado automáticamente) ────────────────────────────
    if cfg["type"] == "liga":
        comp_id_c = cfg["comp_id"]
        league_teams = sorted(set(df_clubs.loc[df_clubs["competition"] == comp_id_c, "home_team"]) |
                              set(df_clubs.loc[df_clubs["competition"] == comp_id_c, "away_team"]))
        league_teams = [t for t in league_teams if t in teams_c]

        col_l1, col_l2 = st.columns([2, 3])
        with col_l1:
            n_sims_l = st.select_slider("Simulaciones", [1_000, 10_000, 50_000, 100_000], value=10_000, key="liga_n")
            home_away_l = st.checkbox("Ida y vuelta", value=True, key="liga_ha")
            st.caption(f"{len(league_teams)} equipos de {competition_c} cargados automáticamente.")

        with col_l2:
            if len(league_teams) >= 2:
                with st.spinner(f"Simulando {n_sims_l:,} temporadas..."):
                    res_l = engine_clubs.simulate_league(
                        league_teams, n_sims=n_sims_l, competition=competition_c, home_away=home_away_l
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
                    st.plotly_chart(tourn_bar(res_l.team_stats, "p_win", f"% Campeón de {competition_c}", "#f59e0b"),
                                    use_container_width=True)
            else:
                st.warning("No se encontraron suficientes equipos para esta liga en el dataset.")

    # ── TORNEO (grupos conocidos automáticamente) ────────────────────────────
    else:
        col_t1, col_t2 = st.columns([2, 3])
        with col_t1:
            n_sims_t = st.select_slider("Simulaciones", [1_000, 10_000, 50_000, 100_000], value=10_000, key="t_n")
            bracket_fmt = st.selectbox("Formato bracket", ["auto","wc","euro","sequential"], key="t_fmt")
            neutral_t = st.checkbox("Campo neutro", value=True, key="t_neutral")

        groups_raw = cfg.get("groups")
        if groups_raw is None:
            # No fixed groups (e.g. Champions League) — seed a single open pool
            flat_teams = [t for t in cfg.get("teams", []) if t in teams_c]
            groups_raw = {chr(65+i//4): flat_teams[i:i+4] for i in range(0, len(flat_teams), 4)}

        groups = {g: [t for t in teams if t in teams_c] for g, teams in groups_raw.items()}
        groups = {g: t for g, t in groups.items() if len(t) >= 2}

        with col_t2:
            if not groups:
                st.warning("Ningún equipo de esta competición está en el dataset seleccionado.")
            else:
                for g, tms in groups.items():
                    st.markdown(f"**Grupo {g}**: " + "  ·  ".join(t.title() for t in tms))

                with st.spinner(f"Simulando {competition_c} ({n_sims_t:,} veces)..."):
                    res_t = eng_c.simulate_tournament(
                        groups, knockout_slots=2, n_sims=n_sims_t,
                        competition=competition_c, neutral=neutral_t,
                        bracket_format=bracket_fmt,
                    )
                df_t = res_t.team_stats

                st.markdown(f"### Resultados — {n_sims_t:,} simulaciones")
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
elif page == "🥇  Premios Individuales":
    st.title("🥇  Premios Individuales")

    tab_gb, tab_profile, tab_teams = st.tabs(["🥾 Máximo goleador", "👤 Perfil de jugador", "📊 Ranking de equipos"])

    # ── MÁXIMO GOLEADOR — predicción determinista (sin nº de simulaciones) ──
    with tab_gb:
        st.markdown("### Predicción de máximo goleador")
        st.caption("Calculado a partir de las tasas de gol esperadas (xG) del jugador frente al calendario del torneo.")

        col_gb1, col_gb2 = st.columns([2, 3])
        with col_gb1:
            preset_gb = st.selectbox("Torneo", ["World Cup 2022","Euro 2024","Copa América 2024"], key="gb_preset")
            comp_map_gb = {"World Cup 2022": "World Cup", "Euro 2024": "UEFA Euro", "Copa América 2024": "Copa América"}
            competition_gb = comp_map_gb[preset_gb]
            eng_gb, teams_gb, _ = pick_engine(competition_gb)

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
                st.info("Selecciona jugadores para la predicción.")
            else:
                groups_gb = {"World Cup 2022": WC_2022, "Euro 2024": EURO_2024,
                            "Copa América 2024": COPA_2024}[preset_gb]
                groups_gb_f = {g: [t for t in ts if t in teams_gb] for g, ts in groups_gb.items()}
                groups_gb_f = {g: t for g, t in groups_gb_f.items() if len(t) >= 2}

                profiles_local = pd.read_csv(ROOT / "data/processed/player_profiles_with_positions.csv")
                player_goals_map = {}
                for player_name in selected_players:
                    row = profiles_local[profiles_local["player"] == player_name]
                    if row.empty: continue
                    r = row.iloc[0]
                    team = str(r.get("team", "")).lower()
                    goals_pm = float(r.get("goals_per_match", 0.15))
                    player_goals_map[player_name] = {team: goals_pm}

                if groups_gb_f and player_goals_map:
                    with st.spinner("Calculando predicción de máximo goleador..."):
                        res_gb = eng_gb.simulate_tournament(
                            groups_gb_f, n_sims=20_000,
                            competition=competition_gb, neutral=True,
                            player_goals=player_goals_map,
                        )

                    if not res_gb.golden_boot.empty:
                        gb_df = res_gb.golden_boot.copy()
                        gb_df["player"] = gb_df["player"].str.split().str[0]
                        st.markdown("#### 🥾 Máximo goleador — predicción")
                        st.dataframe(gb_df.head(10), hide_index=True, use_container_width=True)

                        fig_gb = go.Figure(go.Bar(
                            y=gb_df.head(8)["player"][::-1],
                            x=gb_df.head(8)["avg_goals_tournament"][::-1],
                            orientation="h", marker_color="#f59e0b",
                            text=[f"{v:.2f}" for v in gb_df.head(8)["avg_goals_tournament"][::-1]],
                            textposition="outside",
                        ))
                        fig_gb.update_layout(
                            title="Goles esperados en el torneo",
                            height=280, margin=dict(l=100,r=60,t=40,b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig_gb, use_container_width=True)
                    else:
                        st.info("Sin datos suficientes para calcular el máximo goleador.")
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
                    st.markdown(f"**Equipo:** {str(r.get('team','')).title()}")
                    st.markdown(f"**Posición:** {r.get('position','—')}")
                    st.markdown(f"**Competición:** {r.get('competition','—')}")
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
                pos = str(r.get("position","Unknown"))
                pos_median = profiles_p[profiles_p["position"] == pos]
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
elif page == "📊  Pronóstico de liga":
    spec = importlib.util.spec_from_file_location(
        "competition_forecast_page", Path(__file__).parent / "competition_forecast_page.py")
    cf_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf_page)
    cf_page.render()

elif page == "🧪  SquadLab":
    spec = importlib.util.spec_from_file_location(
        "squadlab_page", Path(__file__).parent / "squadlab_page.py")
    sl_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sl_page)
    sl_page.render(engine=engine_clubs, df_clubs=df_clubs)
