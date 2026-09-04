"""
Mundialytics — Prediction Engine · Block 2 + SquadLab
Menu: 🗓️ Jornada | 🏆 Competición | 📊 Pronóstico de liga | 🎯 Props | 🥇 Premios | 🧪 SquadLab
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
    /* ── Mundialytics design system ─────────────────────────────────────── */
    :root{
        --mv-navy:#0b1220; --mv-panel:#151f30; --mv-panel-2:#1c2942;
        --mv-line:#243350; --mv-blue:#3b82f6; --mv-blue-soft:#60a5fa;
        --mv-green:#10b981; --mv-amber:#f59e0b; --mv-red:#ef4444;
        --mv-text:#e2e8f0; --mv-muted:#8b9bb4;
    }
    html,body,[class*="css"]{font-feature-settings:"tnum" 1;}
    /* headings: tighter, brand weight */
    h1{font-weight:800!important;letter-spacing:-.02em}
    h2,h3{font-weight:700!important;letter-spacing:-.01em}
    h3{margin-top:1.3rem!important}
    a{color:var(--mv-blue-soft)}

    /* metric cards */
    .metric-card{background:linear-gradient(160deg,var(--mv-panel),var(--mv-panel-2));
        border:1px solid var(--mv-line);border-radius:12px;padding:14px 16px;
        text-align:center;height:92px;display:flex;flex-direction:column;
        justify-content:center;transition:border-color .15s}
    .metric-card:hover{border-color:var(--mv-blue)}
    .metric-label{font-size:.72rem;color:var(--mv-muted);margin-bottom:4px;
        text-transform:uppercase;letter-spacing:.04em;font-weight:600}
    .metric-value{font-size:1.7rem;font-weight:800;line-height:1;color:var(--mv-text)}

    /* form badges */
    .form-badge{display:inline-block;width:26px;height:26px;border-radius:50%;
        font-weight:700;font-size:.8rem;text-align:center;line-height:26px;margin:1px}
    .form-W{background:var(--mv-green);color:#04120c}
    .form-D{background:#64748b;color:#fff}
    .form-L{background:var(--mv-red);color:#fff}

    .fixture-row{padding:10px 14px;border-radius:8px;margin:4px 0;
        background:var(--mv-panel);border:1px solid transparent}
    .fixture-row:hover{border-color:var(--mv-line)}

    /* buttons: brand primary */
    .stButton>button{border-radius:9px;border:1px solid var(--mv-line);
        font-weight:600;transition:all .15s}
    .stButton>button:hover{border-color:var(--mv-blue);color:var(--mv-blue-soft)}
    .stDownloadButton>button{border-radius:9px;font-weight:600}

    /* dataframes + tables: softer grid */
    [data-testid="stDataFrame"]{border:1px solid var(--mv-line);border-radius:10px}

    /* sidebar radio -> nav pills */
    section[data-testid="stSidebar"]{background:#0e1626;border-right:1px solid var(--mv-line)}
    section[data-testid="stSidebar"] .stRadio label{padding:2px 0}
    div[data-testid="stSidebarNav"]{display:none}

    /* brand wordmark */
    .mv-brand{display:flex;align-items:center;gap:11px;padding:2px 2px 4px}
    .mv-logo{width:38px;height:38px;border-radius:10px;flex:0 0 38px;
        background:linear-gradient(140deg,var(--mv-blue),#1e40af);
        display:flex;align-items:center;justify-content:center;
        font-weight:900;font-size:20px;color:#fff;
        box-shadow:0 3px 12px rgba(59,130,246,.35)}
    .mv-name{font-size:1.32rem;font-weight:800;letter-spacing:-.03em;
        line-height:1;color:var(--mv-text)}
    .mv-name .mv-accent{color:var(--mv-blue-soft)}
    .mv-tag{font-size:.62rem;color:var(--mv-muted);letter-spacing:.16em;
        text-transform:uppercase;margin-top:3px;font-weight:600}
</style>""", unsafe_allow_html=True)


def brand_header() -> str:
    return ('<div class="mv-brand">'
            '<div class="mv-logo">M</div>'
            '<div><div class="mv-name">Mundia<span class="mv-accent">lytics</span></div>'
            '<div class="mv-tag">Precision Football Models</div></div></div>')

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
    # Champions/Europa/Conference viven en la página 🏆 Europa (formato suizo real
    # + ClubElo cross-league); el viejo torneo de 18 equipos big-5 queda retirado.
    "World Cup":        {"type": "torneo", "engine": "intl",  "groups": WC_2022},
    "UEFA Euro":        {"type": "torneo", "engine": "intl",  "groups": EURO_2024},
    "Copa América":     {"type": "torneo", "engine": "intl",  "groups": COPA_2024},
}
COMPETITIONS = list(COMP_CONFIG.keys())
LEAGUE_COMPETITIONS = [c for c, cfg in COMP_CONFIG.items() if cfg["type"] == "liga"]
TOURNAMENT_COMPETITIONS = [c for c, cfg in COMP_CONFIG.items() if cfg["type"] == "torneo"]


# ── Model cache keys ────────────────────────────────────────────────────────────
def _code_fingerprint(*rel_paths: str) -> str:
    """8-char hash of the given source files' bytes. Folded into cache keys so a
    fitted model self-invalidates when its code changes — no manual version bump
    (the old PROPS/ENGINE_CACHE_VERSION footgun: forget to bump -> stale models
    served silently)."""
    import hashlib
    h = hashlib.sha256()
    for rel in sorted(rel_paths):
        p = ROOT / rel
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:8]


_ENGINE_FP = _code_fingerprint(
    "src/mundialytics/statistical_core/prediction_engine.py",
    "src/mundialytics/statistical_core/attack_defense_model.py",
    "src/mundialytics/statistical_core/distributions.py",
    "src/mundialytics/models/goal_model.py",
    "src/mundialytics/models/xg_rate_model.py",
    "src/mundialytics/ratings/elo.py",
)
_PROPS_FP = _code_fingerprint(
    "src/mundialytics/props/team_props.py",
    "src/mundialytics/props/player_props.py",
)


# ── Engine loading ─────────────────────────────────────────────────────────────
def _engine_cache_path(tag: str, df: pd.DataFrame) -> Path:
    cache_dir = ROOT / "data/processed/cache"
    key = f"{tag}_{len(df)}_{str(df['date'].max())[:10]}_{_ENGINE_FP}"
    return cache_dir / f"engine_{key}.joblib"


def _cached_engine_fit(tag: str, df: pd.DataFrame, build):
    """Disk-cache a fitted engine (fit ~2-3 min -> load <1s). Key = data + version."""
    import joblib
    cache_f = _engine_cache_path(tag, df)
    if cache_f.exists():
        try:
            return joblib.load(cache_f)
        except Exception:
            pass
    engine = build(df)
    try:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(engine, cache_f, compress=3)
        for old in cache_f.parent.glob(f"engine_{tag}_*.joblib"):
            if old != cache_f:
                old.unlink(missing_ok=True)
    except Exception:
        pass
    return engine


@st.cache_resource(show_spinner="⚙️  Cargando modelos de clubes...")
def load_club_engine():
    df = load_clubs_data()

    def build(d):
        elo = EloRater(EloConfig(season_reset_fraction=0.40))
        elo.fit(d)
        # blend_weight_gl 0.30 (was 0.60): 8-fold backtest — goals-AttackDefense
        # deserves ~70%, GL ~30% (0.30 beat 0.60 in 8/8 folds). See project_xg_modeling_findings.
        # sharpen_gamma_1x2: LOFO-validated 1X2 calibration (RPS/LL/ECE all improve).
        eng = PredictionEngine(blend_weight_gl=0.30, ad_rho=-0.07, sharpen_gamma_1x2=1.3,
                               rescale_lambda_to_goals=True, outcome_rho=-0.17,
                               xg_rate_kwargs={"use_ewma": True})
        eng.fit(d, elo_history=pd.DataFrame(elo.history))
        return eng

    engine = _cached_engine_fit("clubs", df, build)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return engine, teams, df


@st.cache_resource(show_spinner="🌍  Cargando modelos de selecciones...")
def load_intl_engine():
    df = load_international_data(min_year=2010)

    def build(d):
        elo = EloRater(EloConfig(season_reset_fraction=0.35, k_base=28.0))
        elo.fit(d)
        eng = PredictionEngine(blend_weight_gl=0.45, ad_rho=-0.06)
        eng.fit(d, elo_history=pd.DataFrame(elo.history))
        return eng

    engine = _cached_engine_fit("intl", df, build)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return engine, teams, df


engine_clubs, CLUB_TEAMS, df_clubs = load_club_engine()
engine_intl,  INTL_TEAMS,  df_intl  = load_intl_engine()

ALL_TEAMS_COMBINED = sorted(set(CLUB_TEAMS) | set(INTL_TEAMS))


@st.cache_resource(show_spinner="🎯  Cargando modelos de props...")
def load_props_models():
    """Team-event + player-prop models (clubs only). Fails soft: (None, None).
    Fitted objects are disk-cached (fit costs ~45s: 2 AD-MLE fits + Platt
    walk-forward); the key auto-invalidates on data OR props-code change."""
    import joblib
    cache_dir = ROOT / "data/processed/cache"
    key = f"{len(df_clubs)}_{str(df_clubs['date'].max())[:10]}_{_PROPS_FP}"
    cache_f = cache_dir / f"props_models_{key}.joblib"
    if cache_f.exists():
        try:
            return joblib.load(cache_f)
        except Exception:
            pass
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
        pp = PlayerPropsModel().fit(
            pm, shots_path=ROOT / "data/external/advanced/understat/understat_shots.csv")
    except Exception:
        pp = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump((tp, pp), cache_f, compress=3)
        for old in cache_dir.glob("props_models_*.joblib"):
            if old != cache_f:
                old.unlink(missing_ok=True)
    except Exception:
        pass
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
        ("Empate", p_draw, "#64748b"),
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
             "shots": "Disparos", "sot": "A puerta", "booking_pts": "Booking pts (10A/25R)",
             "ht_1x2": "1X2 al descanso", "ht_goles": "Goles al descanso",
             "ht_ft": "Descanso/Final"}


def make_match_card(home: str, away: str, ph: float, pdr: float, pa: float,
                    po25: float, lh: float, la: float, sub: str = "") -> bytes:
    """Shareable branded PNG (1200x675) for a match prediction."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")
    ax.text(0.5, 0.93, "MUNDIALYTICS", ha="center", fontsize=15, color="#64748b",
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.80, f"{home.title()}  vs  {away.title()}", ha="center", fontsize=30,
            color="white", fontweight="bold", transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.72, sub, ha="center", fontsize=13, color="#94a3b8", transform=ax.transAxes)
    labels = [home.title()[:14], "Empate", away.title()[:14]]
    vals = [ph, pdr, pa]
    colors = ["#3b82f6", "#6b7280", "#ef4444"]
    x0 = 0.10
    for i, (lab, v, col) in enumerate(zip(labels, vals, colors)):
        y = 0.55 - i * 0.13
        ax.text(x0, y + 0.02, lab, fontsize=15, color="#cbd5e1", transform=ax.transAxes)
        ax.barh([y], [v * 0.55], left=0.30, height=0.07, color=col,
                transform=ax.transAxes, zorder=3)
        ax.barh([y], [0.55], left=0.30, height=0.07, color="#1e293b",
                transform=ax.transAxes, zorder=2)
        ax.text(0.30 + v * 0.55 + 0.015, y + 0.015, f"{v:.0%}", fontsize=16, color="white",
                fontweight="bold", transform=ax.transAxes)
    ax.text(0.10, 0.10, f"Goles esperados  {lh:.2f} – {la:.2f}", fontsize=14,
            color="#cbd5e1", transform=ax.transAxes)
    ax.text(0.60, 0.10, f"Over 2.5:  {po25:.0%}", fontsize=14, color="#cbd5e1",
            transform=ax.transAxes)
    ax.text(0.5, 0.02, "probabilidades calibradas · validación out-of-sample", ha="center",
            fontsize=10, color="#475569", transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_tournament_card(res_df: pd.DataFrame, title: str) -> bytes:
    """Shareable branded PNG: top-8 champion probabilities."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = res_df.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.barh(top["team"], top["p_champion"] * 100, color="#3b82f6")
    for i, (t, v) in enumerate(zip(top["team"], top["p_champion"])):
        ax.text(v * 100 + 0.5, i, f"{v:.1%}", va="center", color="white",
                fontsize=13, fontweight="bold")
    ax.set_title(f"¿Quién gana la {title}?", color="white", fontsize=22,
                 fontweight="bold", pad=18)
    ax.tick_params(colors="#cbd5e1", labelsize=13)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.text(0.99, -0.08, "MUNDIALYTICS", ha="right", fontsize=12, color="#64748b",
            fontweight="bold", transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
PLAYER_COLS_ES = {"player": "Jugador", "Equipo": "Equipo", "exp_min": "Min esp.",
                  "p_anytime_scorer": "Gol", "p_2plus_goals": "2+ goles",
                  "p_shots_over_1_5": "+1.5 tiros", "p_shots_over_2_5": "+2.5 tiros",
                  "p_assist": "Asistencia", "p_yellow": "Amarilla"}


def render_player_props_table(players: pd.DataFrame, home: str, away: str,
                              height: int = 400) -> None:
    """Shared per-player props table (match detail + Props page)."""
    vw = players[players["exp_min"] >= 30].copy()
    if vw.empty:
        return
    vw["Equipo"] = np.where(vw["side"] == "home", home.title(), away.title())
    tbl = vw[list(PLAYER_COLS_ES)].rename(columns=PLAYER_COLS_ES)
    pct = [c for c in tbl.columns if c not in ("Jugador", "Equipo", "Min esp.")]
    for c in pct:
        tbl[c] = (tbl[c] * 100).round(1)
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=height,
                 column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in pct})


PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"
LOG_KEYS = ["season", "jornada", "partido", "mercado", "ambito", "linea"]


def log_round_predictions(df_round: pd.DataFrame, comp: str, season: str,
                          jornada: int, tp, pp) -> int:
    """Register every prediction for the round (props + 1X2/O2.5) with a serve
    timestamp. Append-only; duplicates (same season/round/match/market/line)
    keep the FIRST serve. Returns rows newly logged."""
    rows = []
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    for r in df_round.itertuples():
        pr = predict_safe(engine_clubs, r.home_team, r.away_team, comp, False)
        label = f"{r.home_team} vs {r.away_team}"
        base = dict(logged_at=now, season=season, jornada=jornada, partido=label,
                    fecha=str(r.date)[:10], home=r.home_team, away=r.away_team)
        if pr is not None:
            trio = {"1": pr.p_home_win, "X": pr.p_draw, "2": pr.p_away_win}
            pick = max(trio, key=trio.get)
            rows.append({**base, "mercado": "1X2", "ambito": "Total", "linea": "",
                         "prob": round(trio[pick], 4), "seleccion": pick})
            rows.append({**base, "mercado": "Goles", "ambito": "Total", "linea": 2.5,
                         "prob": round(pr.p_over_25, 4),
                         "seleccion": "OVER" if pr.p_over_25 >= 0.5 else "UNDER"})
        fx = tp.predict_fixture(r.home_team, r.away_team,
                                lam_home=pr.lambda_home if pr else None,
                                lam_away=pr.lambda_away if pr else None)
        for mk, d in fx.items():
            for ln, p in d.get("over", {}).items():
                rows.append({**base, "mercado": mk, "ambito": "Total", "linea": ln,
                             "prob": p, "seleccion": "OVER" if p >= 0.5 else "UNDER"})
            for skey, amb in [("over_home", "Local"), ("over_away", "Visitante")]:
                for ln, p in d.get(skey, {}).items():
                    rows.append({**base, "mercado": mk, "ambito": amb, "linea": ln,
                                 "prob": p, "seleccion": "OVER" if p >= 0.5 else "UNDER"})
    new = pd.DataFrame(rows)
    if new.empty:
        return 0
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    if PRED_LOG.exists():
        old = pd.read_csv(PRED_LOG)
        comb = pd.concat([old, new], ignore_index=True)
    else:
        old = pd.DataFrame()
        comb = new
    # linea="" (the 1X2 rows) round-trips through CSV as NaN; a bare
    # astype(str) makes that "nan" for stored rows and "" for fresh ones, so
    # they never dedupe and every re-log appends a duplicate per fixture.
    comb["linea"] = comb["linea"].fillna("").astype(str).replace("nan", "")
    comb = comb.drop_duplicates(subset=LOG_KEYS, keep="first")  # NOT + seleccion:
    # including the pick in the key let the SAME line survive twice once its
    # probability crossed 0.5 between runs, logging both OVER and UNDER for it.
    comb.to_csv(PRED_LOG, index=False)
    return len(comb) - len(old)


def append_predictions(rows: list[dict]) -> int:
    """Append rows to the prediction log, keeping the FIRST serve of each key."""
    new = pd.DataFrame(rows)
    if new.empty:
        return 0
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(PRED_LOG) if PRED_LOG.exists() else pd.DataFrame()
    comb = pd.concat([old, new], ignore_index=True) if len(old) else new
    # linea="" (the 1X2 rows) round-trips through CSV as NaN; a bare
    # astype(str) makes that "nan" for stored rows and "" for fresh ones, so
    # they never dedupe and every re-log appends a duplicate per fixture.
    comb["linea"] = comb["linea"].fillna("").astype(str).replace("nan", "")
    comb = comb.drop_duplicates(subset=LOG_KEYS, keep="first")  # NOT + seleccion:
    # including the pick in the key let the SAME line survive twice once its
    # probability crossed 0.5 between runs, logging both OVER and UNDER for it.
    comb.to_csv(PRED_LOG, index=False)
    return len(comb) - len(old)


@st.cache_data(show_spinner=False)
def _european_results() -> pd.DataFrame:
    """Real European results from the cached season CSVs, ClubElo names."""
    try:
        from mundialytics.statistical_core.competition.european import (
            fetch_current_elo, make_resolver)
        resolver = make_resolver(list(fetch_current_elo(ROOT)))
    except Exception:
        return pd.DataFrame()
    rows = []
    for p in (ROOT / "data/external/uefa").glob("raw_*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        res = df["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
        df["hg"] = pd.to_numeric(res[0], errors="coerce")
        df["ag"] = pd.to_numeric(res[1], errors="coerce")
        df["fecha"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce",
                                     format="mixed").dt.strftime("%Y-%m-%d")
        df["home"] = df["Home Team"].astype(str).map(resolver)
        df["away"] = df["Away Team"].astype(str).map(resolver)
        rows.append(df.dropna(subset=["home", "away", "hg", "fecha"])[
            ["fecha", "home", "away", "hg", "ag"]])
    if not rows:
        return pd.DataFrame()
    return (pd.concat(rows, ignore_index=True)
            .drop_duplicates(subset=["fecha", "home", "away"])
            .rename(columns={"hg": "home_goals", "ag": "away_goals"}))


EVENT_COLS = {"corners": ("home_corners", "away_corners"),
              "yellows": ("home_yellow_cards", "away_yellow_cards"),
              "fouls": ("home_fouls", "away_fouls"),
              "shots": ("home_shots", "away_shots"),
              "sot": ("home_sot", "away_sot")}


PLAYER_MARKETS_ES = {"jug_goleador": "Goleador", "jug_2goles": "2+ goles",
                     "jug_tiros": "Tiros jugador", "jug_asistencia": "Asistencia",
                     "jug_amarilla": "Amarilla jugador"}


@st.cache_data(show_spinner=False)
def _player_match_actuals() -> pd.DataFrame:
    """(fecha, equipo, jugador) -> what the player actually did.

    Built from the Understat player-match file, keyed onto football-data team
    names via understat_team_match_xg. Without this the jug_* markets would be
    logged and never scored -- exactly the booking-points failure, where a market
    was priced and served for months while the evaluator silently skipped it.
    """
    pm = ROOT / "data/external/advanced/understat/understat_player_match.csv"
    tm = ROOT / "data/processed/understat_team_match_xg.csv"
    if not pm.exists() or not tm.exists():
        return pd.DataFrame()
    try:
        p = pd.read_csv(pm, usecols=["game_id", "team", "player", "minutes",
                                     "goals", "shots", "assists", "yellow_cards"],
                        low_memory=False)
        g = pd.read_csv(tm, usecols=["provider_match_id", "date", "home_team",
                                     "away_team", "home_team_fd", "away_team_fd"])
    except Exception:
        return pd.DataFrame()
    g = g.rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id")
    g["fecha"] = pd.to_datetime(g["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    m = p.merge(g, on="game_id", how="inner")
    # the player file names teams the Understat way; the log uses football-data's
    m["equipo"] = np.where(m["team"] == m["home_team"], m["home_team_fd"],
                           np.where(m["team"] == m["away_team"], m["away_team_fd"], None))
    m = m.dropna(subset=["equipo", "fecha"])
    return m[["fecha", "equipo", "player", "minutes", "goals", "shots",
              "assists", "yellow_cards"]]


@st.cache_data(show_spinner=False)
def pending_predictions() -> pd.DataFrame:
    """Logged predictions whose match has not been played yet.

    Worth surfacing: a public commitment made before kickoff is the part that
    makes the track record credible, and it is visible from the moment the round
    is logged rather than only once results land.
    """
    if not PRED_LOG.exists():
        return pd.DataFrame()
    try:
        log = pd.read_csv(PRED_LOG)
    except Exception:
        return pd.DataFrame()
    if log.empty:
        return pd.DataFrame()
    played = set(df_clubs["home_team"].astype(str) + "|" + df_clubs["away_team"].astype(str)
                 + "|" + df_clubs["date"].astype(str).str[:10])
    key = log["home"].astype(str) + "|" + log["away"].astype(str) + "|" + log["fecha"].astype(str)
    return log[~key.isin(played)]


def evaluate_prediction_log(_df_clubs_len: int) -> pd.DataFrame:
    """Join the served-predictions log with real results -> hit per prediction."""
    if not PRED_LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(PRED_LOG)
    extra = [c for c in ("home_red_cards", "away_red_cards",
                         "home_goals_ht", "away_goals_ht") if c in df_clubs.columns]
    res = df_clubs[["home_team", "away_team", "date", "home_goals", "away_goals"]
                   + [c for cc in EVENT_COLS.values() for c in cc] + extra].copy()
    res["fecha"] = res["date"].astype(str).str[:10]
    m = log.merge(res.rename(columns={"home_team": "home", "away_team": "away"}),
                  on=["home", "away", "fecha"], how="left")
    # European rows won't match the domestic foundation — fill from the UEFA results
    eu_res = _european_results()
    if len(eu_res):
        miss = m["home_goals"].isna()
        fill = (m.loc[miss, ["home", "away", "fecha"]]
                .merge(eu_res, on=["home", "away", "fecha"], how="left"))
        m.loc[miss, "home_goals"] = fill["home_goals"].to_numpy()
        m.loc[miss, "away_goals"] = fill["away_goals"].to_numpy()
    # player markets settle from a different source (Understat player-match),
    # so they are joined separately rather than through the team-level frame
    pl = _player_match_actuals()
    plk = {}
    if len(pl):
        for r in pl.itertuples(index=False):
            plk[(r.fecha, str(r.equipo), str(r.player))] = r

    m = m.dropna(subset=["home_goals"])
    out = []
    for r in m.itertuples(index=False):
        mk = r.mercado
        if mk == "1X2":
            real = "1" if r.home_goals > r.away_goals else ("X" if r.home_goals == r.away_goals else "2")
            hit = float(r.seleccion == real)
        elif mk == "Goles":
            over = (r.home_goals + r.away_goals) > float(r.linea)
            hit = float(over == (r.seleccion == "OVER"))
        elif mk in EVENT_COLS:
            hc, ac = EVENT_COLS[mk]
            hv, av = getattr(r, hc), getattr(r, ac)
            if pd.isna(hv):
                continue
            actual = {"Total": hv + av, "Local": hv, "Visitante": av}[r.ambito]
            over = actual > float(r.linea)
            hit = float(over == (r.seleccion == "OVER"))
        elif mk.startswith("jug_"):
            # `ambito` carries the player for these markets
            act = (plk.get((r.fecha, str(r.home), str(r.ambito)))
                   or plk.get((r.fecha, str(r.away), str(r.ambito))))
            if act is None:
                continue          # player data for this match not published yet
            if mk == "jug_goleador":
                happened = act.goals >= 1
            elif mk == "jug_2goles":
                happened = act.goals >= 2
            elif mk == "jug_tiros":
                happened = act.shots > float(r.linea)
            elif mk == "jug_asistencia":
                happened = act.assists >= 1
            else:
                happened = act.yellow_cards >= 1
            hit = float(bool(happened) == (r.seleccion == "SI"))
        elif mk in ("ht_1x2", "ht_goles", "ht_ft"):
            # Half-time markets, settled from home/away_goals_ht (in the
            # foundation since 2026-09-03). See props/half_time.py.
            hh = getattr(r, "home_goals_ht", float("nan"))
            ah = getattr(r, "away_goals_ht", float("nan"))
            if pd.isna(hh) or pd.isna(ah):
                continue
            ht_res = "1" if hh > ah else ("X" if hh == ah else "2")
            if mk == "ht_1x2":
                hit = float(r.seleccion == ht_res)
            elif mk == "ht_goles":
                over = (hh + ah) > float(r.linea)
                hit = float(over == (r.seleccion == "OVER"))
            else:
                ft_res = ("1" if r.home_goals > r.away_goals
                          else ("X" if r.home_goals == r.away_goals else "2"))
                hit = float(r.seleccion == f"{ht_res}/{ft_res}")
        elif mk == "booking_pts":
            # 10 per yellow + 25 per red, the standard scale the model prices on.
            # This branch used to be a bare `continue` because reds were not in
            # the foundation, so 7% of every logged round was silently
            # unscoreable. home/away_red_cards are now carried through.
            hy, ay = r.home_yellow_cards, r.away_yellow_cards
            hr = getattr(r, "home_red_cards", float("nan"))
            ar = getattr(r, "away_red_cards", float("nan"))
            if pd.isna(hy) or pd.isna(hr):
                continue
            actual = 10.0 * (hy + ay) + 25.0 * (hr + ar)
            over = actual > float(r.linea)
            hit = float(over == (r.seleccion == "OVER"))
        else:
            continue
        # For pick-one markets the stored prob IS the selection's probability;
        # for over/under it is the OVER probability, so the confidence in an
        # UNDER pick is its complement.
        pick_one = mk in ("1X2", "ht_1x2", "ht_ft")
        conf = r.prob if pick_one else max(r.prob, 1 - r.prob)
        out.append({"mercado": {**MARKET_ES, **PLAYER_MARKETS_ES}.get(mk, mk), "ambito": r.ambito,
                    "lado": r.seleccion, "confianza": conf, "acierto": hit,
                    "jornada": r.jornada, "season": r.season})
    return pd.DataFrame(out)


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
    fx = tp.predict_fixture(home, away, referee=referee,
                            lam_home=pred.lambda_home, lam_away=pred.lambda_away)
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

    def _cell(p: float) -> str:
        """Dominant side per line — unders are picks too, not just overs."""
        return f"O {p:.0%}" if p >= 0.5 else f"U {1 - p:.0%}"

    if fx:
        rows = []
        for mk, d in fx.items():
            r = {"Mercado": MARKET_ES.get(mk, mk),
                 "λ Local": d.get("lambda_home", d.get("lambda_yellows", "")),
                 "λ Visitante": d.get("lambda_away", d.get("lambda_reds", "")),
                 "λ Total": d.get("lambda_total", "")}
            for ln, p in d["over"].items():
                r[f"Línea {ln}"] = _cell(p)
            rows.append(r)
        st.dataframe(pd.DataFrame(rows).fillna(""), hide_index=True, use_container_width=True)
        st.caption("O = prob. de superar la línea, U = de quedarse corto (se muestra el lado "
                   "dominante) · booking pts = 10·amarilla + 25·roja.")

        side_rows = []
        for mk, d in fx.items():
            if "over_home" not in d:
                continue
            for key, team_name in [("over_home", home.title()), ("over_away", away.title())]:
                r = {"Mercado": MARKET_ES.get(mk, mk), "Equipo": team_name}
                for ln, p in d[key].items():
                    r[f"Línea {ln}"] = _cell(p)
                side_rows.append(r)
        if side_rows:
            st.markdown("**Líneas por equipo**")
            st.dataframe(pd.DataFrame(side_rows).fillna(""), hide_index=True,
                         use_container_width=True)

    if not players.empty:
        st.markdown("#### Props de jugadores")
        render_player_props_table(players, home, away, height=420)
        st.caption("Probabilidades condicionadas a que el jugador juegue. "
                   "Min esp. = minutos esperados si juega.")


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

    st.download_button(
        "📸 Tarjeta para compartir",
        make_match_card(home, away, pred.p_home_win, pred.p_draw, pred.p_away_win,
                        pred.p_over_25, pred.lambda_home, pred.lambda_away, sub=competition),
        file_name=f"{home}_{away}_prediccion.png".replace(" ", "_"), mime="image/png",
        key=f"card_{home}_{away}")

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
                f'<div style="flex:1;background:#243350;border-radius:4px;height:16px">'
                f'<div style="width:{bar}px;max-width:100%;background:#3b82f6;border-radius:4px;height:16px"></div>'
                f'</div><span style="width:40px;text-align:right;color:#8b9bb4;font-size:.8rem">{pct:.1%}</span>'
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
st.sidebar.markdown(brand_header(), unsafe_allow_html=True)
st.sidebar.divider()
page = st.sidebar.radio("", [
    "🗓️  Jornada",
    "🏆  Competición",
    "📊  Pronóstico de liga",
    "🎯  Props",
    "🏆  Europa",
    "📈  Resultados",
    "🥇  Premios Individuales",
    "🧪  SquadLab",
], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(
    "<div style='font-size:.68rem;color:#8b9bb4;line-height:1.5'>"
    "Big5 2021-26 · Selecciones 2010-26 · Europa (ClubElo)<br>"
    "<span style='color:#10b981'>●</span> 1X2 a 4% del cierre Bet365 · "
    "props 5/5 folds validados</div>", unsafe_allow_html=True)


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
elif page == "🏆  Europa":
    st.title("🏆  Competiciones europeas")
    st.caption("Champions, Europa y Conference League: probabilidades de cada ronda "
               "simulando el formato completo (fase liga + playoff + eliminatorias).")

    try:
        from mundialytics.statistical_core.competition.european import (
            FORMATS, KO_ROUNDS, EuropeanTournament, fetch_current_elo,
            fetch_season_fixtures, load_calibration, make_resolver, normalize_club,
            parse_fixturedownload)
        calib_eu = load_calibration(ROOT)
        elo_all = fetch_current_elo(ROOT)
    except Exception as exc:
        st.warning(f"Capa europea no disponible: {exc}")
        st.stop()

    COMP_EU = {"Champions League": "champions", "Europa League": "europa",
               "Conference League": "conference"}
    comp_eu_label = st.selectbox("Competición", list(COMP_EU), key="eu_comp")
    comp_eu = COMP_EU[comp_eu_label]

    resolver_eu = make_resolver(list(elo_all))
    elo_by_norm = {normalize_club(k): v for k, v in elo_all.items()}
    today_eu = pd.Timestamp.today()
    auto_yr = today_eu.year if today_eu.month >= 7 else today_eu.year - 1
    from mundialytics.statistical_core.competition.european import FD_SLUG as _FD_SLUG
    import re as _re
    cached_yrs = {int(m.group(1)) for p in (ROOT / "data/external/uefa").glob(
        f"raw_{_FD_SLUG[comp_eu]}_*.csv") if (m := _re.search(r"_(\d{4})\.csv$", p.name))}
    yrs_eu = sorted(cached_yrs | {auto_yr}, reverse=True)
    season_yr = st.selectbox("Temporada", yrs_eu, key="eu_season",
                             format_func=lambda y: f"{y}/{str(y + 1)[2:]}"
                             + ("  (próxima)" if y == auto_yr and y not in cached_yrs else ""))
    raw_eu = fetch_season_fixtures(ROOT, comp_eu, season_yr)

    league_eu, ko_eu = (None, None)
    if raw_eu is not None:
        league_eu, ko_eu = parse_fixturedownload(raw_eu, resolver_eu)
        teams_real = sorted(set(league_eu.home) | set(league_eu.away))
        teams_eu = {t: elo_by_norm.get(normalize_club(t)) for t in teams_real}
        teams_eu = {t: e for t, e in teams_eu.items() if e is not None}
        played_lg = int(league_eu["home_goals"].notna().sum())
        ko_played = int(ko_eu["hg"].notna().sum()) if len(ko_eu) else 0
        if ko_played:
            cur = next((r for r in reversed(KO_ROUNDS)
                        if len(ko_eu[(ko_eu["round"] == r) & ko_eu["hg"].notna()])), "playoff")
            estado = {"playoff": "Playoff", "r16": "Octavos", "qf": "Cuartos",
                      "sf": "Semifinales", "final": "Final"}[cur]
            st.success(f"Temporada {season_yr}/{str(season_yr+1)[2:]} REAL cargada — "
                       f"eliminatorias en curso ({estado}). Las probabilidades parten "
                       f"del estado actual del torneo.")
        else:
            st.success(f"Temporada {season_yr}/{str(season_yr+1)[2:]} REAL cargada — fase liga "
                       f"{played_lg}/{len(league_eu)} partidos jugados.")
    else:
        tier = {"champions": (0, 36), "europa": (36, 72), "conference": (72, 108)}[comp_eu]
        elo_sorted = sorted(elo_all.items(), key=lambda kv: -kv[1])
        teams_eu = dict(elo_sorted[tier[0]:tier[1]])
        st.info(f"⚠️ La temporada {season_yr}/{str(season_yr+1)[2:]} aún no está publicada — "
                "participantes ESTIMADOS por ranking Elo y modo PRE-SORTEO (cada simulación "
                "sortea una fase liga válida por bombos). En cuanto exista el calendario "
                "oficial, se cargará automáticamente.")

    if st.button("🎲 Simular torneo (2.000 iteraciones)", key="eu_sim"):
        with st.spinner("Simulando desde el estado actual..."):
            tour = EuropeanTournament(comp_eu, teams_eu, calib_eu, league_eu, ko_eu)
            st.session_state[f"eu_res_{comp_eu}"] = tour.simulate(2000)

    res_eu = st.session_state.get(f"eu_res_{comp_eu}")
    if res_eu is not None:
        top12 = res_eu.head(12)
        figeu = go.Figure(go.Bar(
            x=top12["p_champion"] * 100, y=top12["team"], orientation="h",
            marker_color="#3b82f6", text=[f"{v:.1%}" for v in top12["p_champion"]],
            textposition="outside"))
        figeu.update_layout(height=360, margin=dict(l=0, r=40, t=10, b=0),
                            yaxis=dict(autorange="reversed"), xaxis_title="P(campeón) %",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.markdown(f"### ¿Quién gana la {comp_eu_label}?")
        st.plotly_chart(figeu, use_container_width=True)
        st.download_button("📸 Tarjeta para compartir",
                           make_tournament_card(res_eu, comp_eu_label),
                           file_name=f"{comp_eu}_campeon.png", mime="image/png",
                           key=f"eu_card_{comp_eu}")

        tbl_eu = res_eu.rename(columns={
            "team": "Equipo", "elo": "Elo", "p_top24": "Pasar fase (Top 24)",
            "p_top8": "Top 8 directo", "p_playoff": "Playoff",
            "p_r16": "Octavos", "p_qf": "Cuartos", "p_sf": "Semis",
            "p_final": "Final", "p_champion": "Campeón"})
        pct_eu = ["Pasar fase (Top 24)", "Top 8 directo", "Playoff", "Octavos",
                  "Cuartos", "Semis", "Final", "Campeón"]
        for c in pct_eu:
            tbl_eu[c] = (tbl_eu[c] * 100).round(1)
        st.dataframe(tbl_eu, hide_index=True, use_container_width=True, height=520,
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in pct_eu})
        st.caption("Fuerzas: ClubElo (escala única europea) con mapping Elo→goles calibrado "
                   "sobre 20.000 partidos propios. Eliminatorias a doble partido con "
                   "prórroga y penaltis; final única en campo neutral.")

    # ── partidos de la jornada europea ─────────────────────────────────────────
    if raw_eu is not None and league_eu is not None and len(league_eu):
        from mundialytics.statistical_core.distributions import outcome_probabilities as _op_eu
        st.markdown("---")
        st.markdown("### Partidos de la fase liga")
        rr = raw_eu.copy()
        rr["rnum"] = pd.to_numeric(rr["Round Number"], errors="coerce")
        lg_rows = rr[rr["rnum"].notna()].copy()
        lg_rows["played"] = lg_rows["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+")
        rounds = sorted(lg_rows["rnum"].astype(int).unique())
        pend_rounds = [r for r in rounds if not lg_rows[lg_rows.rnum == r]["played"].all()]
        default_r = pend_rounds[0] if pend_rounds else rounds[-1]
        rnd_sel = st.selectbox("Jornada", rounds, index=rounds.index(default_r), key="eu_round")
        sel = lg_rows[lg_rows.rnum == rnd_sel]
        pend_count = 0
        for _, r in sel.iterrows():
            h_lbl, a_lbl = str(r["Home Team"]), str(r["Away Team"])
            h_ce, a_ce = resolver_eu(h_lbl), resolver_eu(a_lbl)
            c1, c2, c3 = st.columns([3, 2, 3])
            c1.markdown(f"<div style='text-align:right;font-weight:600;padding-top:4px'>{h_lbl}</div>",
                        unsafe_allow_html=True)
            if r["played"]:
                c2.markdown(f"<div style='text-align:center;font-weight:700'>{r['Result']}</div>",
                            unsafe_allow_html=True)
                c3.markdown(f"<div style='font-weight:600;padding-top:4px'>{a_lbl}</div>",
                            unsafe_allow_html=True)
            else:
                eh = elo_by_norm.get(normalize_club(h_ce)) if h_ce else None
                ea = elo_by_norm.get(normalize_club(a_ce)) if a_ce else None
                if eh and ea:
                    d400 = (eh - ea) / 400.0
                    lh_m = float(np.exp(calib_eu["c"] + calib_eu["hfa"] + calib_eu["b"] * d400))
                    la_m = float(np.exp(calib_eu["c"] - calib_eu["b"] * d400))
                    p = _op_eu(lh_m, la_m, dixon_coles_rho=-0.07)
                    c2.markdown(f"<div style='text-align:center;font-size:.9rem'>"
                                f"<b>{p['p_home_win']:.0%}</b> · {p['p_draw']:.0%} · "
                                f"<b>{p['p_away_win']:.0%}</b><br>"
                                f"<span style='color:#9ca3af;font-size:.75rem'>O2.5 "
                                f"{p['p_over_25']:.0%}</span></div>", unsafe_allow_html=True)
                    pend_count += 1
                else:
                    c2.markdown("<div style='text-align:center;color:#9ca3af'>sin Elo</div>",
                                unsafe_allow_html=True)
                c3.markdown(f"<div style='font-weight:600;padding-top:4px'>{a_lbl}</div>",
                            unsafe_allow_html=True)
        if pend_count:
            st.caption(f"{pend_count} partidos pendientes con predicción (1 · X · 2, Elo del día).")
            if st.button("📌 Registrar predicciones de esta jornada europea", key="eu_log_btn"):
                now_iso = pd.Timestamp.now().isoformat(timespec="seconds")
                rows_log = []
                for _, r in sel.iterrows():
                    if r["played"]:
                        continue
                    h_ce3, a_ce3 = resolver_eu(str(r["Home Team"])), resolver_eu(str(r["Away Team"]))
                    eh3 = elo_by_norm.get(normalize_club(h_ce3)) if h_ce3 else None
                    ea3 = elo_by_norm.get(normalize_club(a_ce3)) if a_ce3 else None
                    if not (eh3 and ea3):
                        continue
                    d43 = (eh3 - ea3) / 400.0
                    lh3 = float(np.exp(calib_eu["c"] + calib_eu["hfa"] + calib_eu["b"] * d43))
                    la3 = float(np.exp(calib_eu["c"] - calib_eu["b"] * d43))
                    p3 = _op_eu(lh3, la3, dixon_coles_rho=-0.07)
                    fecha3 = str(pd.to_datetime(r["Date"], dayfirst=True, errors="coerce",
                                                format="mixed"))[:10]
                    base3 = dict(logged_at=now_iso, season=f"EU {season_yr}-{season_yr + 1}",
                                 jornada=f"{comp_eu_label} J{rnd_sel}",
                                 partido=f"{h_ce3} vs {a_ce3}", fecha=fecha3,
                                 home=h_ce3, away=a_ce3)
                    trio3 = {"1": p3["p_home_win"], "X": p3["p_draw"], "2": p3["p_away_win"]}
                    pick3 = max(trio3, key=trio3.get)
                    rows_log.append({**base3, "mercado": "1X2", "ambito": "Total", "linea": "",
                                     "prob": round(trio3[pick3], 4), "seleccion": pick3})
                    rows_log.append({**base3, "mercado": "Goles", "ambito": "Total", "linea": 2.5,
                                     "prob": round(p3["p_over_25"], 4),
                                     "seleccion": "OVER" if p3["p_over_25"] >= 0.5 else "UNDER"})
                n_new = append_predictions(rows_log)
                evaluate_prediction_log.clear()
                st.success(f"{n_new} predicciones europeas registradas — track record en 📈 Resultados.")

    # ── análisis completo de un partido europeo ────────────────────────────────
    if raw_eu is not None and league_eu is not None and len(league_eu):
        from mundialytics.statistical_core.competition.club_aliases import CLUBELO_TO_FD
        from mundialytics.statistical_core.competition.european import (
            load_event_calibration, predict_euro_events)
        from mundialytics.props.team_props import SIDE_LINES as _SIDE_EU

        st.markdown("---")
        st.markdown("### 🔬 Análisis completo de un partido")
        all_fx = raw_eu.copy()
        fx_labels = [f"{r['Home Team']} vs {r['Away Team']} ({r['Round Number']})"
                     for _, r in all_fx.iterrows()]
        pick_eu = st.selectbox("Partido", fx_labels, key="eu_match_pick")
        row_eu = all_fx.iloc[fx_labels.index(pick_eu)]
        h_lbl2, a_lbl2 = str(row_eu["Home Team"]), str(row_eu["Away Team"])
        h_ce2, a_ce2 = resolver_eu(h_lbl2), resolver_eu(a_lbl2)
        eh2 = elo_by_norm.get(normalize_club(h_ce2)) if h_ce2 else None
        ea2 = elo_by_norm.get(normalize_club(a_ce2)) if a_ce2 else None
        if not (eh2 and ea2):
            st.warning("Sin Elo para alguno de los equipos.")
        else:
            from mundialytics.statistical_core.distributions import outcome_probabilities as _op2
            d400_2 = (eh2 - ea2) / 400.0
            lh2 = float(np.exp(calib_eu["c"] + calib_eu["hfa"] + calib_eu["b"] * d400_2))
            la2 = float(np.exp(calib_eu["c"] - calib_eu["b"] * d400_2))
            p2 = _op2(lh2, la2, dixon_coles_rho=-0.07)
            st.plotly_chart(prob_bar_chart(p2["p_home_win"], p2["p_draw"], p2["p_away_win"],
                                           h_lbl2, a_lbl2), use_container_width=True)
            mc = st.columns(5)
            for col, (lab, val) in zip(mc, [
                    ("xGoals", f"{lh2:.2f} – {la2:.2f}"),
                    ("Over 1.5", f"{p2['p_over_15']:.0%}"), ("Over 2.5", f"{p2['p_over_25']:.0%}"),
                    ("Over 3.5", f"{p2['p_over_35']:.0%}"), ("BTTS", f"{p2['p_btts']:.0%}")]):
                col.markdown(metric_card(lab, val), unsafe_allow_html=True)
            st.download_button(
                "📸 Tarjeta para compartir",
                make_match_card(h_lbl2, a_lbl2, p2["p_home_win"], p2["p_draw"],
                                p2["p_away_win"], p2["p_over_25"], lh2, la2, sub=comp_eu_label),
                file_name=f"{h_lbl2}_{a_lbl2}.png".replace(" ", "_"), mime="image/png",
                key="eu_match_card")

            # team props: full domestic model if both clubs are big-5, Elo-events otherwise
            fd_h, fd_a = CLUBELO_TO_FD.get(h_ce2), CLUBELO_TO_FD.get(a_ce2)
            tp_eu, pp_eu = load_props_models()
            fx_props = None
            if fd_h and fd_a and tp_eu is not None:
                fx_props = tp_eu.predict_fixture(fd_h, fd_a, lam_home=lh2, lam_away=la2)
                src_props = "modelo completo (ambos clubes big-5)"
            if not fx_props:
                ev_cal = load_event_calibration(ROOT)
                if ev_cal:
                    fx_props = predict_euro_events(eh2, ea2, ev_cal, _SIDE_EU)
                    src_props = "mapping Elo→eventos (calibrado en 17k partidos)"
            if fx_props:
                st.markdown(f"#### Props del partido · <span style='color:#9ca3af;font-size:.8rem'>"
                            f"{src_props}</span>", unsafe_allow_html=True)
                rows_p = []
                for mk, dd in fx_props.items():
                    if "over" not in dd:
                        continue
                    r_ = {"Mercado": MARKET_ES.get(mk, mk),
                          "λ Local": dd.get("lambda_home", ""), "λ Visit": dd.get("lambda_away", ""),
                          "λ Total": dd.get("lambda_total", "")}
                    for ln, pv in dd["over"].items():
                        r_[f"Línea {ln}"] = f"O {pv:.0%}" if pv >= 0.5 else f"U {1-pv:.0%}"
                    rows_p.append(r_)
                st.dataframe(pd.DataFrame(rows_p).fillna(""), hide_index=True,
                             use_container_width=True)

            # player props: big-5 covered sides only
            if pp_eu is not None:
                shown = False
                for fd_t, lam_t, lbl_t in [(fd_h, lh2, h_lbl2), (fd_a, la2, a_lbl2)]:
                    if not fd_t:
                        continue
                    try:
                        pl = pp_eu.team_players_for_lambda(fd_t, lam_t)
                    except Exception:
                        continue
                    if pl is None or pl.empty:
                        continue
                    if not shown:
                        st.markdown("#### Props de jugadores")
                        shown = True
                    st.markdown(f"**{lbl_t}**")
                    pl = pl.assign(side="home")
                    render_player_props_table(pl, lbl_t.lower(), lbl_t.lower(), height=300)
                if shown:
                    st.caption("Solo clubes de las 5 grandes ligas (cobertura de datos de "
                               "jugadores); ratios domésticos + contexto del partido europeo.")

elif page == "📈  Resultados":
    st.title("📈  Resultados y fiabilidad")
    st.caption("Validación fuera de muestra, comparación con el mercado y track record en vivo.")

    # ── hero ───────────────────────────────────────────────────────────────────
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.markdown(metric_card("Partidos de validación", "10.400+"), unsafe_allow_html=True)
    hc2.markdown(metric_card("vs cierre de Bet365 (1X2)", "4.1%", "#16a34a"), unsafe_allow_html=True)
    hc3.markdown(metric_card("Mercados cubiertos", "30+"), unsafe_allow_html=True)
    hc4.markdown(metric_card("Grandes ligas", "5"), unsafe_allow_html=True)
    st.caption("Toda la validación es temporal y fuera de muestra: cada temporada se predice "
               "solo con información anterior a ella. Ninguna cuota entra jamás en los modelos.")

    # ── benchmark vs mercado ───────────────────────────────────────────────────
    st.markdown("### Frente al mercado")
    st.markdown("Distancia de nuestras probabilidades 1X2 al **cierre de Bet365** — la "
                "referencia más exigente que existe — sobre 10.080 partidos (2020–2026). "
                "Los mejores modelos académicos publicados se sitúan entre 0.005 y 0.012; "
                "el nuestro: **0.0080**.")
    bm = pd.DataFrame([
        {"Liga": "Bundesliga", "Distancia al cierre": 0.0060},
        {"Liga": "LaLiga", "Distancia al cierre": 0.0064},
        {"Liga": "Ligue 1", "Distancia al cierre": 0.0075},
        {"Liga": "Serie A", "Distancia al cierre": 0.0092},
        {"Liga": "Premier League", "Distancia al cierre": 0.0099},
    ])
    figb = go.Figure(go.Bar(x=bm["Distancia al cierre"], y=bm["Liga"], orientation="h",
                            marker_color="#3b82f6", text=[f"{v:.4f}" for v in bm["Distancia al cierre"]],
                            textposition="outside"))
    figb.add_vline(x=0.012, line_dash="dot", line_color="#9ca3af",
                   annotation_text="rango élite académico", annotation_position="top")
    figb.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=0),
                       xaxis=dict(range=[0, 0.014], title="RPS gap (menos = mejor)"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figb, use_container_width=True)

    # ── calibración (computed live from the walk-forward cache) ────────────────
    @st.cache_data(show_spinner=False)
    def _calibration_bins():
        # deployed-chain walk-forward predictions (generate_deployed_walkforward.py)
        p = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
        if not p.exists():
            return None
        w = pd.read_csv(p)
        o = np.where(w.hg > w.ag, "home", np.where(w.hg < w.ag, "away", "draw"))
        y = np.concatenate([(o == "home").astype(float), (o == "draw").astype(float),
                            (o == "away").astype(float)])
        pr = np.concatenate([w["ph"].to_numpy(float), w["pd"].to_numpy(float),
                             w["pa"].to_numpy(float)])
        edges = np.linspace(0, 0.9, 10)
        rows = []
        for lo in edges:
            msk = (pr >= lo) & (pr < lo + 0.1)
            if msk.sum() > 200:
                rows.append({"pred": pr[msk].mean(), "real": y[msk].mean(), "n": int(msk.sum())})
        return pd.DataFrame(rows)

    cal = _calibration_bins()
    if cal is not None and not cal.empty:
        st.markdown("### Calibración: cuando decimos X%, ocurre X%")
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=[0, 0.9], y=[0, 0.9], mode="lines", name="perfecto",
                                  line=dict(dash="dot", color="#9ca3af")))
        figc.add_trace(go.Scatter(x=cal["pred"], y=cal["real"], mode="markers+lines",
                                  name="modelo", marker=dict(size=9, color="#3b82f6")))
        figc.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="Probabilidad anunciada", yaxis_title="Frecuencia real",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(figc, use_container_width=True)
        ece_v = float((cal.n / cal.n.sum() * (cal.pred - cal.real).abs()).sum())
        st.caption(f"Miles de selecciones 1X2 fuera de muestra. Error de calibración medio: "
                   f"{ece_v:.3f} (por debajo de 0.02 se considera excelente).")

    # ── props validation (marketing level) ────────────────────────────────────
    st.markdown("### Mercados de props: validación 2021–2026")
    st.markdown("Cada mercado se validó temporada a temporada contra referencias "
                "estadísticas exigentes antes de publicarse. Solo se activa lo que gana "
                "de forma consistente.")
    pv = pd.DataFrame([
        {"Mercado": "Amarillas (total y por equipo)", "Temporadas ganadas": "5/5", "Calibración": "Excelente"},
        {"Mercado": "Faltas", "Temporadas ganadas": "5/5", "Calibración": "Excelente"},
        {"Mercado": "Disparos (total y por equipo)", "Temporadas ganadas": "5/5", "Calibración": "Excelente"},
        {"Mercado": "A puerta", "Temporadas ganadas": "5/5", "Calibración": "Muy buena"},
        {"Mercado": "Córners (total y por equipo)", "Temporadas ganadas": "4-5/5", "Calibración": "Muy buena"},
        {"Mercado": "Booking points", "Temporadas ganadas": "5/5", "Calibración": "Muy buena"},
        {"Mercado": "Props de jugador (6 mercados)", "Temporadas ganadas": "5/5", "Calibración": "Excelente"},
    ])
    st.dataframe(pv, hide_index=True, use_container_width=True)

    # ── live track record ──────────────────────────────────────────────────────
    st.markdown("### 📌 Track record en vivo")
    st.caption("Cada predicción se registra **antes** de que el partido se juegue, con fecha "
               "y hora. Nada se añade a posteriori: por eso esto es un track record y no una "
               "lista de aciertos elegida después.")

    pend = pending_predictions()
    if not pend.empty:
        pc1, pc2, pc3 = st.columns(3)
        pc1.markdown(metric_card("Partidos comprometidos", f"{pend['partido'].nunique()}"),
                     unsafe_allow_html=True)
        pc2.markdown(metric_card("Predicciones en juego", f"{len(pend):,}"),
                     unsafe_allow_html=True)
        pc3.markdown(metric_card("Se resuelven", f"{pend['fecha'].min()} → {pend['fecha'].max()}"),
                     unsafe_allow_html=True)

    ev = evaluate_prediction_log(len(df_clubs))
    if ev.empty:
        st.info("Predicciones ya registradas y esperando a que se jueguen los partidos. "
                "En cuanto haya resultados aparecerá aquí el acierto real frente al anunciado.")
    else:
        n = len(ev)
        acc = ev["acierto"].mean()
        exp = ev["confianza"].mean()
        tc1, tc2, tc3 = st.columns(3)
        tc1.markdown(metric_card("Predicciones evaluadas", f"{n}"), unsafe_allow_html=True)
        tc2.markdown(metric_card("Acierto real", f"{acc:.1%}",
                                 "#16a34a" if acc >= exp - 0.02 else "#dc2626"), unsafe_allow_html=True)
        tc3.markdown(metric_card("Acierto esperado", f"{exp:.1%}"), unsafe_allow_html=True)
        st.caption("Un modelo honesto acierta ≈ lo que anuncia. Real muy por encima = suerte; "
                   "muy por debajo = problema.")
        by_mk = (ev.groupby("mercado").agg(N=("acierto", "size"), Acierto=("acierto", "mean"),
                                           Esperado=("confianza", "mean")).reset_index())
        by_mk[["Acierto", "Esperado"]] = (by_mk[["Acierto", "Esperado"]] * 100).round(1)
        st.dataframe(by_mk, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in ["Acierto", "Esperado"]})
        ev["banda"] = pd.cut(ev["confianza"], [0.5, 0.55, 0.6, 0.7, 1.0],
                             labels=["50-55%", "55-60%", "60-70%", "70%+"])
        by_b = (ev.groupby("banda", observed=True)
                .agg(N=("acierto", "size"), Acierto=("acierto", "mean"),
                     Esperado=("confianza", "mean")).reset_index())
        by_b[["Acierto", "Esperado"]] = (by_b[["Acierto", "Esperado"]] * 100).round(1)
        st.markdown("**Por banda de confianza**")
        st.dataframe(by_b, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                    for c in ["Acierto", "Esperado"]})
        ou = ev[ev["lado"].isin(["OVER", "UNDER"])]
        if len(ou) > 20:
            by_s = (ou.groupby("lado").agg(N=("acierto", "size"), Acierto=("acierto", "mean"),
                                           Esperado=("confianza", "mean")).reset_index()
                    .rename(columns={"lado": "Lado"}))
            by_s[["Acierto", "Esperado"]] = (by_s[["Acierto", "Esperado"]] * 100).round(1)
            st.markdown("**Overs vs Unders** — el valor vive en ambos lados")
            st.dataframe(by_s, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                        for c in ["Acierto", "Esperado"]})

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

elif page == "🎯  Props":
    st.title("🎯  Props")
    st.caption("Córners, tarjetas, faltas, tiros, booking points y props de jugadores.")

    tp_m, pp_m = load_props_models()
    if tp_m is None:
        st.warning("Modelos de props no disponibles (faltan datos).")
        st.stop()

    colp1, colp2, colp3 = st.columns([2, 2, 2])
    with colp1:
        comp_p = st.selectbox("Competición", LEAGUE_COMPETITIONS, key="p_comp")
    with colp2:
        comp_id_p = COMP_CONFIG[comp_p]["comp_id"]
        seasons_p = sorted(df_clubs.loc[df_clubs["competition"] == comp_id_p, "season"].unique(), reverse=True)
        season_p = st.selectbox("Temporada", seasons_p, key="p_season")
    df_round_p, total_rounds_p = get_round_fixtures(comp_p, season_p, 1, df_clubs)
    if total_rounds_p == 0:
        st.warning("Sin datos para esta competición/temporada.")
        st.stop()
    with colp3:
        jornada_p = st.slider("Jornada", 1, total_rounds_p, total_rounds_p, key="p_round")
    df_round_p, _ = get_round_fixtures(comp_p, season_p, jornada_p, df_clubs)
    if df_round_p.empty:
        st.info("Sin partidos en esta jornada.")
        st.stop()

    fixture_labels = [f"{r.home_team.title()} vs {r.away_team.title()}" for r in df_round_p.itertuples()]
    pick = st.selectbox("Partido", fixture_labels, key="p_fixture")
    row_p = df_round_p.iloc[fixture_labels.index(pick)]
    home_p, away_p = row_p["home_team"], row_p["away_team"]

    referee_p = None
    if tp_m.known_referees and tp_m.is_epl_fixture(home_p, away_p):
        referee_p = st.selectbox("Árbitro (opcional — mejora tarjetas y faltas)",
                                 ["—"] + tp_m.known_referees, key="p_ref")
        referee_p = None if referee_p == "—" else referee_p

    pred_p = predict_safe(engine_clubs, home_p, away_p, comp_p, False)
    lamh = pred_p.lambda_home if pred_p else None
    lama = pred_p.lambda_away if pred_p else None
    fx_p = tp_m.predict_fixture(home_p, away_p, referee=referee_p, lam_home=lamh, lam_away=lama)
    if not fx_p:
        st.warning("Sin histórico suficiente para estos equipos.")
        st.stop()

    prof_h, prof_a = tp_m.team_profile(home_p), tp_m.team_profile(away_p)
    if lamh:
        st.markdown(f"**xGoals del motor:** {home_p.title()} {lamh:.2f} – {lama:.2f} {away_p.title()}"
                    + (f" · Árbitro: {referee_p}" if referee_p else ""))

    def ladder_html(over: dict) -> str:
        """Each line shows its DOMINANT side — a 26% over is a 74% UNDER and
        must read as such (value lives on both sides)."""
        html = ""
        for ln, p in over.items():
            side, sp = ("O", p) if p >= 0.5 else ("U", 1 - p)
            base = "#3b82f6" if side == "O" else "#f59e0b"
            color = "#16a34a" if sp >= 0.62 else base
            html += (f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                     f'<span style="width:52px;font-weight:700;font-size:.85rem">{side} {ln}</span>'
                     f'<div style="flex:1;background:#e5e7eb33;border-radius:4px;height:14px">'
                     f'<div style="width:{sp*100:.0f}%;background:{color};border-radius:4px;height:14px"></div></div>'
                     f'<span style="width:44px;text-align:right;font-size:.85rem;font-weight:600">{sp:.0%}</span>'
                     f'</div>')
        return html

    st.markdown("### Mercados del partido (totales)")
    m_cols = st.columns(2)
    for i, (mk, d) in enumerate([(k, v) for k, v in fx_p.items() if k != "booking_pts"]):
        with m_cols[i % 2]:
            ref_tag = " 🧑‍⚖️" if d.get("referee_used") else ""
            st.markdown(f"**{MARKET_ES.get(mk, mk)}{ref_tag}** — λ {d['lambda_home']} + "
                        f"{d['lambda_away']} = **{d['lambda_total']}**")
            ph = prof_h.get(mk)
            pa = prof_a.get(mk)
            if ph and pa and ph.get("league_avg_side"):
                st.caption(f"Forma (últ. 10): {home_p.title()} {ph['for_r10']} a favor / "
                           f"{ph['against_r10']} en contra · {away_p.title()} {pa['for_r10']} / "
                           f"{pa['against_r10']} · media liga {ph['league_avg_side']}/equipo")
            st.markdown(ladder_html(d["over"]), unsafe_allow_html=True)
            st.markdown("")

    if "booking_pts" in fx_p:
        bp = fx_p["booking_pts"]
        st.markdown(f"**{MARKET_ES['booking_pts']}** — λ amarillas {bp['lambda_yellows']} · "
                    f"λ rojas {bp['lambda_reds']} (media liga)")
        st.markdown(ladder_html(bp["over"]), unsafe_allow_html=True)

    side_any = any("over_home" in d for d in fx_p.values())
    if side_any:
        st.markdown("### Líneas por equipo")
        sc1, sc2 = st.columns(2)
        for col, side_key, tname in [(sc1, "over_home", home_p.title()), (sc2, "over_away", away_p.title())]:
            with col:
                st.markdown(f"**{tname}**")
                for mk, d in fx_p.items():
                    if side_key in d:
                        st.markdown(f"*{MARKET_ES.get(mk, mk)}*")
                        st.markdown(ladder_html(d[side_key]), unsafe_allow_html=True)

    with st.expander("📈 Distribución de un mercado a fondo"):
        mk_deep = st.selectbox("Mercado", [k for k in fx_p if k != "booking_pts"],
                               format_func=lambda k: MARKET_ES.get(k, k), key="p_deep")
        d = fx_p[mk_deep]
        from scipy.stats import nbinom as _nb
        lam_t, disp_t = d["lambda_total"], max(d["dispersion"], 1.05)
        r_nb = lam_t / (disp_t - 1.0)
        ks = list(range(0, int(lam_t * 2.4) + 2))
        pmf = [float(_nb.pmf(k, r_nb, 1.0 / disp_t)) for k in ks]
        figd = go.Figure(go.Bar(x=ks, y=pmf, marker_color="#3b82f6"))
        for ln in d["over"]:
            figd.add_vline(x=ln, line_dash="dot", line_color="#9ca3af")
        figd.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title=f"{MARKET_ES.get(mk_deep, mk_deep)} totales",
                           yaxis_title="P", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(figd, use_container_width=True)
        st.caption(f"λ {lam_t} · líneas punteadas = líneas de apuesta.")

    if pp_m is not None:
        st.markdown("### Props de jugadores")
        try:
            players_p = pp_m.predict_fixture(home_p, away_p, lam_home=lamh, lam_away=lama)
        except Exception:
            players_p = pd.DataFrame()
        if not players_p.empty:
            render_player_props_table(players_p, home_p, away_p, height=380)

    st.markdown("---")
    st.markdown("### 🔍 Escáner de la jornada")
    col_scan, col_log = st.columns([1, 1])
    scan_key = f"scan_{comp_p}_{season_p}_{jornada_p}"
    with col_log:
        if st.button("📌 Registrar predicciones de la jornada", key="p_log_btn"):
            n_new = log_round_predictions(df_round_p, comp_p, season_p, jornada_p, tp_m, pp_m)
            evaluate_prediction_log.clear()
            st.success(f"{n_new} predicciones nuevas registradas (las repetidas conservan "
                       "el primer registro). Track record en 📈 Resultados.")
    with col_scan:
        scan_clicked = st.button("Escanear jornada", key="p_scan_btn")
    if scan_clicked:
        rows_s = []
        prog = st.progress(0.0)
        for k, r in enumerate(df_round_p.itertuples()):
            pr = predict_safe(engine_clubs, r.home_team, r.away_team, comp_p, False)
            fx_s = tp_m.predict_fixture(r.home_team, r.away_team,
                                        lam_home=pr.lambda_home if pr else None,
                                        lam_away=pr.lambda_away if pr else None)
            label = f"{r.home_team.title()} vs {r.away_team.title()}"
            for mk, d in fx_s.items():
                for ln, p in d.get("over", {}).items():
                    rows_s.append({"Partido": label, "Mercado": MARKET_ES.get(mk, mk),
                                   "Ámbito": "Total", "Línea": ln, "P(Over)": p})
                for skey, amb in [("over_home", r.home_team.title()), ("over_away", r.away_team.title())]:
                    for ln, p in d.get(skey, {}).items():
                        rows_s.append({"Partido": label, "Mercado": MARKET_ES.get(mk, mk),
                                       "Ámbito": amb, "Línea": ln, "P(Over)": p})
            prog.progress((k + 1) / len(df_round_p))
        prog.empty()
        st.session_state[scan_key] = pd.DataFrame(rows_s)
    if scan_key in st.session_state:
        sc = st.session_state[scan_key].copy()
        mks = st.multiselect("Mercados", sorted(sc["Mercado"].unique()),
                             default=sorted(sc["Mercado"].unique()), key="p_scan_mks")
        sc = sc[sc["Mercado"].isin(mks)]
        sc["Señal"] = (sc["P(Over)"] - 0.5).abs()
        sc["Lado"] = np.where(sc["P(Over)"] >= 0.5, "OVER", "UNDER")
        sc["Confianza"] = np.where(sc["Lado"] == "OVER", sc["P(Over)"], 1 - sc["P(Over)"])
        top = sc.sort_values("Señal", ascending=False).head(30)
        top = top[["Partido", "Mercado", "Ámbito", "Línea", "Lado", "Confianza"]]
        top["Confianza"] = (top["Confianza"] * 100).round(1)
        st.dataframe(top, hide_index=True, use_container_width=True, height=500,
                     column_config={"Confianza": st.column_config.NumberColumn(format="%.1f%%")})

elif page == "🧪  SquadLab":
    spec = importlib.util.spec_from_file_location(
        "squadlab_page", Path(__file__).parent / "squadlab_page.py")
    sl_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sl_page)
    sl_page.render(engine=engine_clubs, df_clubs=df_clubs)
