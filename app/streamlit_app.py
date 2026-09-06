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

import mv_ui
from mv_ui import bar_row, brand_header, chart, metric_card, split_stat
from mundialytics.statistical_core.prediction_engine import PredictionEngine
from mundialytics.statistical_core.engine_utils import (
    load_clubs_data, load_international_data,
    get_h2h, get_form, h2h_summary, bracket_html,
)
from mundialytics.ratings.elo import EloRater, EloConfig
from mundialytics.identity.normalization import canonical_team_name

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mundialytics", page_icon="⚽", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**Mundialytics** — motor probabilístico de fútbol: 1X2, goles, "
                 "córners, tarjetas y props de jugador sobre las 5 grandes ligas "
                 "y las competiciones UEFA. Cada predicción se registra antes del "
                 "partido y se liquida contra el resultado real.",
        "Report a bug": "https://github.com/Beansent48/mundialytics/issues",
    })

mv_ui.inject()

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
# `type` decides which engine and which screen a competition gets:
#   liga   — domestic league, club engine, real published calendar
#   euro   — UEFA club competition, ClubElo on a single European scale
#   torneo — national-team tournament with fixed groups, international engine
COMP_CONFIG = {
    "LaLiga":            {"type": "liga",   "engine": "clubs", "comp_id": "LaLiga"},
    "Premier League":    {"type": "liga",   "engine": "clubs", "comp_id": "Premier League"},
    "Serie A":           {"type": "liga",   "engine": "clubs", "comp_id": "Serie A"},
    "Bundesliga":        {"type": "liga",   "engine": "clubs", "comp_id": "Bundesliga"},
    "Ligue 1":           {"type": "liga",   "engine": "clubs", "comp_id": "Ligue 1"},
    "Champions League":  {"type": "euro",   "engine": "elo",   "comp_id": "champions"},
    "Europa League":     {"type": "euro",   "engine": "elo",   "comp_id": "europa"},
    "Conference League": {"type": "euro",   "engine": "elo",   "comp_id": "conference"},
    "World Cup":         {"type": "torneo", "engine": "intl",  "groups": WC_2022},
    "UEFA Euro":         {"type": "torneo", "engine": "intl",  "groups": EURO_2024},
    "Copa América":      {"type": "torneo", "engine": "intl",  "groups": COPA_2024},
}
COMPETITIONS = list(COMP_CONFIG.keys())
LEAGUE_COMPETITIONS = [c for c, cfg in COMP_CONFIG.items() if cfg["type"] == "liga"]
EURO_COMPETITIONS = [c for c, cfg in COMP_CONFIG.items() if cfg["type"] == "euro"]
# every competition with a matchday to browse — what the main screen offers
MATCHDAY_COMPETITIONS = LEAGUE_COMPETITIONS + EURO_COMPETITIONS


# ── Favourites ─────────────────────────────────────────────────────────────────
# Kept on disk rather than in session_state: a preference that disappears when
# the tab reloads is not a preference. One small JSON, written on every toggle.
FAVORITES_PATH = ROOT / "data/processed/app_favorites.json"


@st.cache_data(show_spinner=False)
def load_favorites() -> dict:
    import json
    try:
        data = json.loads(FAVORITES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"competitions": [], "teams": []}
    return {"competitions": list(data.get("competitions", [])),
            "teams": list(data.get("teams", []))}


def toggle_favorite(kind: str, name: str) -> None:
    import json
    favs = load_favorites()
    items = favs.setdefault(kind, [])
    if name in items:
        items.remove(name)
    else:
        items.append(name)
    try:
        FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAVORITES_PATH.write_text(json.dumps(favs, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    except OSError:
        pass
    load_favorites.clear()


def is_favorite(kind: str, name: str) -> bool:
    return name in load_favorites().get(kind, [])


def favorites_first(names: list[str], kind: str = "competitions") -> list[str]:
    """Starred entries float to the top of a selector, order otherwise intact."""
    favs = load_favorites().get(kind, [])
    return sorted(names, key=lambda n: (n not in favs, names.index(n)))


def star_button(kind: str, name: str, key: str, label: str | None = None) -> None:
    """The star that toggles `name`. Rendered by the caller inside its own column.

    `label` puts the thing being starred on the button itself — two bare stars
    side by side do not say which team each one follows.
    """
    on = is_favorite(kind, name)
    # a bare star is fine beside a selector and meaningless once Streamlit stacks
    # the columns on a phone, so it always carries words
    if label is None:
        label = "En favoritos" if on else "A favoritos"
    text = ("★ " if on else "☆ ") + label
    if st.button(text.strip(), key=key, use_container_width=True,
                 help="Quitar de favoritos" if on else "Añadir a favoritos"):
        toggle_favorite(kind, name)
        st.rerun()


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
    # the roster half of the fitted model lives here: change how a squad member
    # is matched to his history and the cached fit is wrong, not merely old
    "src/mundialytics/identity/current_squads.py",
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


# Nothing below this point can run without the engines, and a missing data file
# used to end the visit on a Python traceback. Say what broke, in the app's own
# language, and stop there.
try:
    engine_clubs, CLUB_TEAMS, df_clubs = load_club_engine()
    engine_intl,  INTL_TEAMS,  df_intl  = load_intl_engine()
except Exception as exc:
    st.markdown(mv_ui.brand_header(), unsafe_allow_html=True)
    st.markdown(mv_ui.notice(
        "🛠️", "No se han podido cargar los modelos",
        "La app necesita el dataset de partidos y los modelos ajustados sobre él. "
        f"El fallo fue: <code>{type(exc).__name__}: {str(exc)[:200]}</code><br><br>"
        "Suele arreglarse regenerando los datos con "
        "<code>python scripts/update_season.py</code>."), unsafe_allow_html=True)
    st.stop()

ALL_TEAMS_COMBINED = sorted(set(CLUB_TEAMS) | set(INTL_TEAMS))


@st.cache_resource(show_spinner="🎯  Cargando modelos de props...")
def load_props_models():
    """Team-event + player-prop models (clubs only). Fails soft: (None, None).
    Fitted objects are disk-cached (fit costs ~45s: 2 AD-MLE fits + Platt
    walk-forward); the key auto-invalidates on data OR props-code change."""
    import joblib
    from mundialytics.identity.current_squads import load_current_squads, squads_fingerprint
    cache_dir = ROOT / "data/processed/cache"
    squads = load_current_squads()
    key = (f"{len(df_clubs)}_{str(df_clubs['date'].max())[:10]}_{_PROPS_FP}"
           f"_{squads_fingerprint(squads)}")
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
        # `current_squads` decides only WHO is in each roster. Without it the
        # squad is "whoever featured in this club's last ten Understat games",
        # which for a promoted club reaches back to its last top-flight season.
        pp = PlayerPropsModel().fit(
            pm, shots_path=ROOT / "data/external/advanced/understat/understat_shots.csv",
            current_squads=squads)
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


@st.cache_data(show_spinner=False, max_entries=12)
def simulate_league_cached(teams: tuple[str, ...], n_sims: int, competition: str,
                           home_away: bool, fp: str = _ENGINE_FP) -> pd.DataFrame:
    """A full-season Monte Carlo, keyed on its inputs.

    It used to run on every rerun of the page: ticking "ida y vuelta" or even
    re-picking the same competition re-simulated 10,000 seasons. Same numbers,
    same code path — only computed once per set of inputs.
    """
    res = engine_clubs.simulate_league(list(teams), n_sims=n_sims,
                                       competition=competition, home_away=home_away)
    return res.team_stats


@st.cache_data(show_spinner=False, max_entries=12)
def simulate_tournament_cached(groups: tuple[tuple[str, tuple[str, ...]], ...], n_sims: int,
                               competition: str, neutral: bool, bracket_format: str,
                               engine_kind: str, fp: str = _ENGINE_FP) -> pd.DataFrame:
    """Group-stage-plus-bracket Monte Carlo, cached like the league one above.

    `engine_kind` ("clubs"/"intl") is a plain string on purpose: it selects the
    engine inside the cached body instead of making an unhashable engine object
    part of the key.
    """
    eng = engine_intl if engine_kind == "intl" else engine_clubs
    res = eng.simulate_tournament(
        {g: list(t) for g, t in groups}, knockout_slots=2, n_sims=n_sims,
        competition=competition, neutral=neutral, bracket_format=bracket_format)
    return res.team_stats


@st.cache_data(show_spinner=False, max_entries=64)
def round_probabilities(pairs: tuple[tuple[str, str], ...], competition: str,
                        neutral: bool, fp: str = _ENGINE_FP) -> dict:
    """1X2 for a whole matchday at once, keyed on the round rather than the widget.

    The matchday list re-renders on every slider drag and checkbox click; without
    this each of those reruns re-predicted all ten fixtures from scratch.
    """
    out = {}
    for h, a in pairs:
        pr = predict_safe(engine_clubs, h, a, competition, neutral)
        if pr is not None:
            out[(h, a)] = (pr.p_home_win, pr.p_draw, pr.p_away_win)
    return out


def prob_color(p: float, hi: float = 0.55, lo: float = 0.45) -> str:
    """One rule for 'is this probability worth a second look', app-wide: green
    above `hi`, red below `lo`, neutral blue in the middle."""
    return mv_ui.C["green"] if p >= hi else (mv_ui.C["red"] if p <= lo else mv_ui.C["blue"])


def form_badges(form_list: list[dict]) -> str:
    html = ""
    for f in form_list:
        r = f["result"]
        html += f'<span class="form-badge form-{r}" title="{f["score"]} vs {f["opponent"].title()}">{r}</span>'
    return html


def prob_bar_chart(p_home: float, p_draw: float, p_away: float,
                   home: str, away: str) -> go.Figure:
    fig = go.Figure()
    for name, val, color in [
        (home.title(), p_home, mv_ui.HOME),
        ("Empate", p_draw, mv_ui.DRAW),
        (away.title(), p_away, mv_ui.AWAY),
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
        texttemplate="%{text}", showscale=False,
        colorscale=mv_ui.HEAT,
        textfont=dict(color=mv_ui.C["text"], size=11),
        xgap=2, ygap=2,
        hovertemplate=f"{home.title()} %{{y}}–%{{x}} {away.title()}: %{{z:.2f}}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title=f"⚽ {away.title()}", yaxis_title=f"⚽ {home.title()}",
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=False), height=340,
        margin=dict(l=55, r=10, t=10, b=55),
    )
    return fig


def tourn_bar(df: pd.DataFrame, col: str, title: str, color=None) -> go.Figure:
    top = df[df[col].notna()].head(20).sort_values(col)
    fig = go.Figure(go.Bar(
        y=top["team"].str.title(), x=top[col], orientation="h",
        marker_color=color or mv_ui.C["blue"], text=[f"{v:.1%}" for v in top[col]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(280, len(top) * 26),
        margin=dict(l=120, r=80, t=40, b=10),
        xaxis=dict(tickformat=".0%", showgrid=True),
    )
    return fig


def render_stat_grid(events: list[tuple[str, float, float]],
                     preds: dict[str, tuple[float, float]] | None = None) -> None:
    """Shared 'home vs away' stat row, used for both real and predicted stats.

    Each stat is a split card: the two numbers plus a bar showing how the total
    divides between the sides, so the comparison is read rather than computed.
    Pass `preds` to stack the model's numbers inside the same cards.
    """
    ev_cols = st.columns(len(events))
    for col, (label, hv, av) in zip(ev_cols, events):
        col.markdown(split_stat(label, hv, av,
                                pred=(preds or {}).get(label)), unsafe_allow_html=True)


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
    fig.patch.set_facecolor(mv_ui.C["bg"])
    ax.set_facecolor(mv_ui.C["bg"])
    ax.axis("off")
    ax.text(0.5, 0.93, "MUNDIALYTICS", ha="center", fontsize=15, color="#64748b",
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.80, f"{home.title()}  vs  {away.title()}", ha="center", fontsize=30,
            color="white", fontweight="bold", transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.72, sub, ha="center", fontsize=13, color="#94a3b8", transform=ax.transAxes)
    labels = [home.title()[:14], "Empate", away.title()[:14]]
    vals = [ph, pdr, pa]
    colors = [mv_ui.HOME, mv_ui.DRAW, mv_ui.AWAY]
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
    fig.patch.set_facecolor(mv_ui.C["bg"])
    ax.set_facecolor(mv_ui.C["bg"])
    ax.barh(top["team"], top["p_champion"] * 100, color=mv_ui.C["blue"])
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
    keep the FIRST serve. Returns rows newly logged.

    Only fixtures that have NOT kicked off are written. A round now comes from the
    published calendar, so it can hold played and unplayed matches side by side —
    and a prediction logged after the result is known is not a track record, which
    is the exact failure this log was rebuilt to stop.
    """
    rows = []
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    for r in df_round.itertuples():
        if pd.notna(getattr(r, "home_goals", None)) and pd.notna(getattr(r, "away_goals", None)):
            continue
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


from mundialytics.serving import track_record as tr  # noqa: E402
from mundialytics.serving.track_record import (  # noqa: E402
    EVENT_COLS, PLAYER_MARKETS_ES, evaluate_log,
)

# Settlement itself lives in mundialytics.serving.track_record so the HTTP API
# grades the log exactly the same way; Streamlit only adds its caching on top.
_european_results = st.cache_data(show_spinner=False)(tr._european_results)
_player_match_actuals = st.cache_data(show_spinner=False)(tr._player_match_actuals)

@st.cache_data(ttl=1800, show_spinner=False)
def _uefa_fixtures(competition: str, year: int):
    """UEFA fixtures for the European page, cached for half an hour.

    fetch_season_fixtures now asks ESPN before falling back to disk, so without
    this every Streamlit rerun -- a dropdown, a slider -- would make a network
    call. The TTL is what keeps results arriving during a matchday.
    """
    from mundialytics.statistical_core.competition.european import fetch_season_fixtures
    return fetch_season_fixtures(ROOT, competition, year)


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


@st.cache_data(show_spinner=False)
def evaluate_prediction_log(_df_clubs_len: int) -> pd.DataFrame:
    """Settled predictions, labelled in Spanish for this app."""
    return evaluate_log(df_clubs, labels={**MARKET_ES, **PLAYER_MARKETS_ES})


def ladder_html(over: dict) -> str:
    """An over/under ladder, each line showing its DOMINANT side.

    A 26% over is a 74% UNDER and has to read as such — the value lives on both
    sides, and a column of small percentages hides half of it.
    """
    html = ""
    for ln, p in over.items():
        side, sp = ("O", p) if p >= 0.5 else ("U", 1 - p)
        base = mv_ui.C["blue"] if side == "O" else mv_ui.C["amber"]
        color = mv_ui.C["green"] if sp >= 0.62 else base
        html += bar_row(f"{side} {ln}", sp, f"{sp:.0%}", color, label_w=52)
    return html


def render_market_ladders(fx: dict, home: str, away: str, profiles=None) -> None:
    """Every team-event market as ladders — totals, then the per-team lines.

    This is the whole of the old Props page. It belongs in the match analysis:
    corners and cards are facts about a fixture, not a separate part of the app.
    """
    totals = [(k, v) for k, v in fx.items() if k != "booking_pts" and "over" in v]
    m_cols = st.columns(2)
    for i, (mk, d) in enumerate(totals):
        with m_cols[i % 2]:
            ref_tag = " 🧑‍⚖️" if d.get("referee_used") else ""
            lam_txt = ""
            if d.get("lambda_total") is not None:
                lam_txt = (f" — λ {d.get('lambda_home', '?')} + {d.get('lambda_away', '?')}"
                           f" = **{d['lambda_total']}**")
            st.markdown(f"**{MARKET_ES.get(mk, mk)}{ref_tag}**{lam_txt}")
            if profiles:
                ph, pa = profiles[0].get(mk), profiles[1].get(mk)
                if ph and pa and ph.get("league_avg_side"):
                    st.markdown(
                        f'<div class="mv-bar-sub">Forma (últ. 10): {home.title()} '
                        f'{ph["for_r10"]} a favor / {ph["against_r10"]} en contra · '
                        f'{away.title()} {pa["for_r10"]} / {pa["against_r10"]} · '
                        f'media liga {ph["league_avg_side"]}/equipo</div>',
                        unsafe_allow_html=True)
            st.markdown(ladder_html(d["over"]), unsafe_allow_html=True)
            st.markdown("")

    if "booking_pts" in fx:
        bp = fx["booking_pts"]
        st.markdown(f"**{MARKET_ES['booking_pts']}** — λ amarillas {bp['lambda_yellows']} · "
                    f"λ rojas {bp['lambda_reds']}")
        st.markdown(ladder_html(bp["over"]), unsafe_allow_html=True)

    if any("over_home" in d for d in fx.values()):
        st.markdown("#### Líneas por equipo")
        sc1, sc2 = st.columns(2)
        for col, side_key, tname in [(sc1, "over_home", home.title()),
                                     (sc2, "over_away", away.title())]:
            with col:
                st.markdown(f"**{tname}**")
                for mk, d in fx.items():
                    if side_key in d:
                        st.markdown(f'<div class="mv-bar-sub">{MARKET_ES.get(mk, mk)}</div>',
                                    unsafe_allow_html=True)
                        st.markdown(ladder_html(d[side_key]), unsafe_allow_html=True)


def render_market_distribution(fx: dict, key: str) -> None:
    """The full count distribution behind one market, with its betting lines marked."""
    options = [k for k in fx if k != "booking_pts" and "dispersion" in fx[k]]
    if not options:
        return
    with st.expander("📈 Distribución de un mercado a fondo"):
        mk = st.selectbox("Mercado", options, format_func=lambda k: MARKET_ES.get(k, k),
                          key=f"deep_{key}")
        d = fx[mk]
        from scipy.stats import nbinom as _nb
        lam_t, disp_t = d["lambda_total"], max(d["dispersion"], 1.05)
        r_nb = lam_t / (disp_t - 1.0)
        ks = list(range(0, int(lam_t * 2.4) + 2))
        pmf = [float(_nb.pmf(k, r_nb, 1.0 / disp_t)) for k in ks]
        figd = go.Figure(go.Bar(x=ks, y=pmf, marker_color=mv_ui.C["blue"]))
        for ln in d["over"]:
            figd.add_vline(x=ln, line_dash="dot", line_color=mv_ui.C["dim"])
        figd.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title=f"{MARKET_ES.get(mk, mk)} totales", yaxis_title="P",
                           showlegend=False)
        chart(figd)
        st.caption(f"λ {lam_t} · líneas punteadas = líneas de apuesta.")


def render_props_section(home: str, away: str, pred) -> None:
    """Team-event and player props for one fixture — the old Props page, in place."""
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
    if fx:
        render_market_ladders(fx, home, away,
                              profiles=(tp.team_profile(home), tp.team_profile(away)))
        st.caption("O = probabilidad de superar la línea, U = de quedarse corto; se muestra "
                   "el lado dominante · booking pts = 10·amarilla + 25·roja.")
        render_market_distribution(fx, key=f"{home}_{away}")

    if not players.empty:
        render_top_scorers(players, home, away)
        with st.expander("Todos los jugadores (props completos)"):
            render_player_props_table(players, home, away, height=420)
            st.caption("Probabilidades condicionadas a que el jugador juegue. "
                       "Min esp. = minutos esperados si juega.")


def render_player_track_record(ev_players: pd.DataFrame) -> None:
    """Player markets judged on RANKING, which is the only fair way to read them.

    Accuracy here is meaningless: 8.6% of starters score and the model never puts
    anyone above 50%, so "no" is right ~92% of the time and says nothing about
    skill. What the model is actually good at is ordering — measured over 6,327
    settled player-matches its top pick scored 28.8% of the time, 3.4x a random
    starter. So this panel reports how our shortlist fared, plus calibration.
    """
    if ev_players is None or ev_players.empty:
        return
    st.markdown("#### ⚽ Mercados de jugador")
    st.caption("El acierto bruto no sirve aquí: solo el 8,6% de los titulares marca, así que "
               "responder «no» acierta el 92% de las veces sin mérito alguno. Lo que se mide "
               "es si acertamos **a quién**.")

    sc = ev_players[ev_players["mercado_id"] == "jug_goleador"].copy()
    if not sc.empty:
        sc["rank"] = sc.groupby("partido")["prob"].rank(ascending=False, method="first")
        # `acierto` records whether the SI/NO call was right; the event itself
        # (did he score) is that call read back through the side it took
        sc["marco"] = np.where(sc["lado"] == "SI", sc["acierto"], 1 - sc["acierto"])
        rate = sc["marco"].mean()
        c1, c2, c3 = st.columns(3)
        t1 = sc[sc["rank"] <= 1]["marco"].mean() if (sc["rank"] <= 1).any() else float("nan")
        t3 = sc[sc["rank"] <= 3]["marco"].mean() if (sc["rank"] <= 3).any() else float("nan")
        c1.markdown(metric_card("Nuestro favorito marcó", f"{t1:.0%}" if t1 == t1 else "—"),
                    unsafe_allow_html=True)
        c2.markdown(metric_card("Del top-3, marcaron", f"{t3:.0%}" if t3 == t3 else "—"),
                    unsafe_allow_html=True)
        c3.markdown(metric_card("Titular cualquiera", f"{rate:.0%}"), unsafe_allow_html=True)
        if rate > 0 and t1 == t1:
            st.caption(f"Nuestro favorito marca **{t1/rate:.1f} veces más** que un titular "
                       f"tomado al azar de la misma alineación.")

    st.markdown("**Calibración por banda** — cuando decimos X%, ¿ocurre X%?")
    d = ev_players.copy()
    d["ocurre"] = np.where(d["lado"] == "SI", d["acierto"], 1 - d["acierto"])
    d["banda"] = pd.cut(d["prob"], [0, .05, .10, .20, .35, 1.0],
                        labels=["0-5%", "5-10%", "10-20%", "20-35%", "35%+"])
    by_b = (d.groupby("banda", observed=True)
            .agg(N=("ocurre", "size"), Anunciado=("prob", "mean"),
                 Ocurrio=("ocurre", "mean")).reset_index())
    by_b[["Anunciado", "Ocurrio"]] = (by_b[["Anunciado", "Ocurrio"]] * 100).round(1)
    st.dataframe(by_b, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                for c in ["Anunciado", "Ocurrio"]})


def render_top_scorers(players: pd.DataFrame, home: str, away: str, n: int = 3) -> None:
    """The n likeliest scorers per side — the headline of the player model.

    Ranked rather than yes/no on purpose. Over 6,327 settled player-matches the
    model never once put a player above 50% to score, so a yes/no read just
    returns the base rate (8.6% of starters score). Ranking is where the skill
    shows: the top pick scores 28.8% of the time, 3.4x a random starter, and in
    79.5% of team-matches with a goal a real scorer sat in this top three.
    """
    if players.empty or "p_anytime_scorer" not in players.columns:
        return
    st.markdown("#### ⚽ Quién puede marcar")
    cols = st.columns(2)
    for col, team in zip(cols, (home, away)):
        # `side` is the reliable key: the model labels teams the Understat way
        # ("Real Betis") while the app passes football-data's ("betis"), so a
        # name match silently returns nothing for half the fixtures.
        want = "home" if team == home else "away"
        if "side" in players.columns:
            side = players[players["side"] == want]
        else:
            side = players[players["team"].astype(str).str.lower() == str(team).lower()]
        # restrict to the likely XI first: exp_min is minutes-when-featuring,
        # not minutes spread across the squad, so most of a 26-man list clears it
        top = side.nlargest(11, "exp_min").nlargest(n, "p_anytime_scorer")
        with col:
            st.markdown(f"**{str(team).title()}**")
            if top.empty:
                st.caption("Sin datos de jugador para este equipo.")
                continue
            for r in top.itertuples(index=False):
                pct = float(r.p_anytime_scorer)
                # the shortlist is ranked, so the bars are scaled to the leader:
                # absolute widths would all be stubs (nobody clears ~35%) and the
                # ordering — the thing the model is actually good at — would not read
                st.markdown(bar_row(str(r.player)[:22], pct / max(float(top["p_anytime_scorer"].max()), 1e-9),
                                    f"{pct:.0%}", mv_ui.C["blue_soft"], label_w=132),
                            unsafe_allow_html=True)
    st.caption("Probabilidad de marcar al menos un gol, condicionada a que juegue. "
               "Escala con los goles esperados de su equipo en este partido.")


def real_match_events(row: dict) -> list[tuple[str, float, float]]:
    """Every measured stat of a played match, in display order. Goals lead: they
    are the one the model is actually judged on."""
    raw = [
        ("Goles",     row.get("home_goals"),  row.get("away_goals")),
        ("Disparos",  row.get("home_shots"),  row.get("away_shots")),
        ("A puerta",  row.get("home_sot"),    row.get("away_sot")),
        ("Córners",   row.get("home_corners"), row.get("away_corners")),
        ("Faltas",    row.get("home_fouls"),  row.get("away_fouls")),
        ("Amarillas", row.get("home_yellow_cards"), row.get("away_yellow_cards")),
    ]
    return [(label, float(hv) if pd.notna(hv) else 0.0, float(av) if pd.notna(av) else 0.0)
            for label, hv, av in raw]


def predicted_match_events(pred) -> dict[str, tuple[float, float]]:
    """The model's number for each of those stats, keyed by the same label."""
    return {
        "Goles":     (pred.lambda_home, pred.lambda_away),
        "Disparos":  (pred.expected_shots_home, pred.expected_shots_away),
        "A puerta":  (pred.expected_sot_home, pred.expected_sot_away),
        "Córners":   (pred.expected_corners_home, pred.expected_corners_away),
        "Faltas":    (pred.expected_fouls_home, pred.expected_fouls_away),
        "Amarillas": (pred.expected_yellows_home, pred.expected_yellows_away),
    }


def render_result_comparison(pred, row: dict, home: str, away: str) -> None:
    """How each market we priced actually resolved.

    The stat cards above answer "how close were the numbers"; this answers the
    other half — of the calls we made, which ones came in. Every market shown is
    one the engine prices, so nothing is graded that was never claimed.
    """
    hg, ag = int(row["home_goals"]), int(row["away_goals"])
    outcome = "1" if hg > ag else ("X" if hg == ag else "2")
    trio = {"1": pred.p_home_win, "X": pred.p_draw, "2": pred.p_away_win}
    pick = max(trio, key=trio.get)
    total = hg + ag
    calls = [
        ("1X2", f"{pick} ({trio[pick]:.0%})", outcome, pick == outcome),
        ("Over 2.5", f"{'OVER' if pred.p_over_25 >= .5 else 'UNDER'} "
                     f"({max(pred.p_over_25, 1 - pred.p_over_25):.0%})",
         f"{total} goles", (total > 2.5) == (pred.p_over_25 >= 0.5)),
        ("Over 1.5", f"{'OVER' if pred.p_over_15 >= .5 else 'UNDER'} "
                     f"({max(pred.p_over_15, 1 - pred.p_over_15):.0%})",
         f"{total} goles", (total > 1.5) == (pred.p_over_15 >= 0.5)),
        ("BTTS", f"{'SÍ' if pred.p_btts >= .5 else 'NO'} "
                 f"({max(pred.p_btts, 1 - pred.p_btts):.0%})",
         "sí" if (hg and ag) else "no", (hg > 0 and ag > 0) == (pred.p_btts >= 0.5)),
    ]
    st.markdown("### Nuestras apuestas, una a una")
    cols = st.columns(len(calls))
    for col, (market, said, happened, hit) in zip(cols, calls):
        col.markdown(
            metric_card(market, said,
                        mv_ui.C["green"] if hit else mv_ui.C["red"],
                        sub=f"{'✔' if hit else '✘'}  real: {happened}"),
            unsafe_allow_html=True)

    exact = next((i["probability"] for i in pred.top_scorelines
                  if i["score"] == f"{hg}-{ag}"), None)
    if exact is not None:
        rank = next((n for n, i in enumerate(pred.top_scorelines, 1)
                     if i["score"] == f"{hg}-{ag}"), None)
        st.caption(f"Al resultado exacto {hg}–{ag} le dimos **{exact:.1%}**"
                   + (f", el {rank}º más probable de los {len(pred.top_scorelines)} "
                      "marcadores que valoramos." if rank else "."))


def render_real_match_stats(row: dict, home: str, away: str, pred=None) -> None:
    """A played match's real stats, optionally with the model's numbers under them.

    Same cards either way — the comparison is a second row inside each card, not
    a different screen, so nothing has to be held in memory to read it.
    """
    events = real_match_events(row)
    if pred is None:
        render_stat_grid(events)
        return
    preds = predicted_match_events(pred)
    render_stat_grid(events, preds=preds)
    total_err = sum(abs(preds[k][0] - hv) + abs(preds[k][1] - av)
                    for k, hv, av in events if k in preds)
    st.caption(f"Arriba lo que pasó, debajo lo que dijimos antes de jugarse. "
               f"Error total del partido: **{total_err:.1f}** eventos sobre "
               f"{len(events)} estadísticas.")


def render_partido_detail(home: str, away: str, competition: str, neutral: bool,
                          actual_result: str | None = None, row: dict | None = None):
    """Full match breakdown, reused by the Jornada flow.

    A played match opens on what happened. From there two more views, both in the
    same card layout: the model's numbers stacked under the real ones, and the
    full pre-match breakdown. They are views of one screen rather than separate
    screens, because the question — did we get it right — is a comparison.
    """
    eng, eng_teams, df_hist = pick_engine(competition)
    pred = predict_safe(eng, home, away, competition, neutral)

    if row is not None:
        st.markdown(f"## ⚽ {home.title()} {int(row['home_goals'])} – "
                    f"{int(row['away_goals'])} {away.title()}")
        c_fav1, c_fav2, c_rest = st.columns([2, 2, 5], vertical_alignment="center")
        with c_fav1:
            star_button("teams", home, key=f"fav_h_{home}_{away}", label=home.title())
        with c_fav2:
            star_button("teams", away, key=f"fav_a_{home}_{away}", label=away.title())
        c_rest.caption("Sigue un equipo para verlo destacado en cada jornada · "
                       + mv_ui.fmt_date_long(row.get("date", "")))

        views = ["📊  Datos reales", "🎯  Real vs predicción", "🔮  Predicción completa"]
        view = st.radio("Vista", views, horizontal=True, label_visibility="collapsed",
                        key=f"view_{home}_{away}")
        st.markdown("### Estadísticas del partido")
        if view == views[0]:
            render_real_match_stats(row, home, away)
            return
        if view == views[1]:
            if pred is None:
                st.error("Error al generar predicción para este partido.")
                return
            render_real_match_stats(row, home, away, pred=pred)
            render_result_comparison(pred, row, home, away)
            return
        render_real_match_stats(row, home, away)
        st.markdown("---")
        st.caption("Predicción del modelo para este partido (no es lo que ocurrió realmente).")

    if pred is None:
        st.error("Error al generar predicción para este partido.")
        return

    if row is None:
        st.markdown(f"## ⚽ {home.title()} vs {away.title()}")
        c_fav1, c_fav2, c_rest = st.columns([2, 2, 5], vertical_alignment="center")
        with c_fav1:
            star_button("teams", home, key=f"fav_h_{home}_{away}", label=home.title())
        with c_fav2:
            star_button("teams", away, key=f"fav_a_{home}_{away}", label=away.title())
        c_rest.caption("Sigue un equipo para verlo destacado en cada jornada")
        if actual_result:
            st.markdown(f"**Resultado real:** {actual_result}")

    if home not in eng_teams or away not in eng_teams:
        missing = [t.title() for t in [home, away] if t not in eng_teams]
        st.info(f"ℹ️ {', '.join(missing)} no está en el dataset de entrenamiento. Se usa prior global.")

    st.markdown("### Resultado")
    chart(prob_bar_chart(pred.p_home_win, pred.p_draw, pred.p_away_win, home, away))

    cols = st.columns(6)
    data = [
        (home.title(),  f"{pred.p_home_win:.1%}", mv_ui.HOME),
        ("Empate",      f"{pred.p_draw:.1%}",     mv_ui.DRAW),
        (away.title(),  f"{pred.p_away_win:.1%}",  mv_ui.AWAY),
        ("xGoals",      f"{pred.lambda_home:.2f} – {pred.lambda_away:.2f}", ""),
        ("BTTS",        f"{pred.p_btts:.1%}", prob_color(pred.p_btts, lo=0.0)),
        ("Over 2.5",    f"{pred.p_over_25:.1%}", prob_color(pred.p_over_25, lo=0.0)),
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
        chart(score_heatmap(pred.score_matrix, home, away))

    with col_scores:
        st.markdown("### Resultados más probables")
        top_sl = pred.top_scorelines[:8]
        lead_sl = max((i["probability"] for i in top_sl), default=1.0) or 1.0
        for item in top_sl:
            pct = item["probability"]
            st.markdown(bar_row(item["score"], pct / lead_sl, f"{pct:.1%}",
                                label_w=38, value_w=44),
                        unsafe_allow_html=True)

    st.markdown("### Mercados de goles")
    ocols = st.columns(5)
    ou = [("Over 1.5", pred.p_over_15), ("Under 2.5", pred.p_under_25),
          ("Over 2.5", pred.p_over_25), ("Over 3.5", pred.p_over_35),
          ("BTTS",     pred.p_btts)]
    for col, (label, val) in zip(ocols, ou):
        col.markdown(metric_card(label, f"{val:.1%}", prob_color(val)), unsafe_allow_html=True)

    st.markdown("### Estadísticas esperadas")
    render_stat_grid([(k, v[0], v[1]) for k, v in predicted_match_events(pred).items()])

    if eng is engine_clubs:
        render_props_section(home, away, pred)

    st.markdown("---")
    col_h2h, col_form = st.columns(2)

    with col_h2h:
        st.markdown("### Historial H2H")
        summary = h2h_summary(home, away, df_hist)
        if summary["total"] > 0:
            w, d, l, tot = summary["w"], summary["d"], summary["l"], summary["total"]
            # the record as a bar first: 8V 3E 5D says less at a glance than the
            # split it describes
            st.markdown(
                f'<div style="display:flex;gap:14px;align-items:baseline;margin-bottom:8px">'
                f'<span style="color:{mv_ui.HOME};font-weight:800;font-size:1.3rem">{w}V</span>'
                f'<span style="color:{mv_ui.DRAW};font-weight:800;font-size:1.3rem">{d}E</span>'
                f'<span style="color:{mv_ui.AWAY};font-weight:800;font-size:1.3rem">{l}D</span>'
                f'<span style="color:{mv_ui.C["dim"]};font-size:.8rem">({tot} partidos)</span></div>'
                f'<div class="mv-split-bar" style="margin-bottom:14px">'
                f'<div style="width:{w / tot * 100:.1f}%;background:{mv_ui.HOME}"></div>'
                f'<div style="width:{d / tot * 100:.1f}%;background:{mv_ui.DRAW}"></div>'
                f'<div style="flex:1;background:{mv_ui.AWAY}"></div></div>',
                unsafe_allow_html=True)
            h2h_df = get_h2h(home, away, df_hist, n=6)
            for _, row in h2h_df.iterrows():
                res = row["result_t1"]
                c = (mv_ui.C["green"] if res == "W"
                     else (mv_ui.C["muted"] if res == "D" else mv_ui.C["red"]))
                st.markdown(mv_ui.hist_row(
                    mv_ui.fmt_date_long(row["date"]),
                    f'{row["home_team"].title()} {row["score"]} {row["away_team"].title()}',
                    res, c), unsafe_allow_html=True)
        else:
            st.caption("Sin historial disponible entre estos equipos.")

    with col_form:
        st.markdown("### Forma reciente")
        for team, color in [(home, mv_ui.HOME), (away, mv_ui.AWAY)]:
            form = get_form(team, df_hist, n=5)
            st.markdown(
                f'<div style="margin-bottom:6px">'
                f'<span style="font-weight:700;color:{color}">{team.title()}</span> '
                f'<span style="font-size:.72rem;color:{mv_ui.C["dim"]}">'
                f'(últimos {len(form)})</span><br>'
                f'<div style="margin-top:5px">{form_badges(form)}</div></div>',
                unsafe_allow_html=True)
            for f in form[:3]:
                st.markdown(
                    f'<div class="mv-hist-note">{f["date"]} · {f["result"]} {f["score"]} '
                    f'vs {f["opponent"].title()} ({f["home_away"]})</div>',
                    unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=1800, max_entries=16)
def season_calendar(comp_id: str, season: str) -> pd.DataFrame:
    """The published calendar for a league-season: every fixture, played or not.

    The foundation holds only played matches, so matchdays derived from it were
    approximate AND could never contain a fixture before kick-off — on a
    prediction app, the round you actually want to see. ESPN publishes the whole
    season in one request (the same source the pre-kickoff logger already uses).
    Returns empty when unavailable; the caller then falls back to slicing.
    """
    try:
        from mundialytics.providers.espn_fixtures import fetch_season_fixtures
        fx = fetch_season_fixtures(comp_id, season, root=ROOT)
    except Exception:
        return pd.DataFrame()
    if fx is None or fx.empty or "matchday" not in fx.columns:
        return pd.DataFrame()
    cal = fx[["date", "matchday", "home_team", "away_team", "completed"]].copy()
    # the provider's own score, kept apart from the foundation's: the foundation
    # is refreshed on a schedule, so a match played yesterday is not in it yet and
    # would otherwise be offered as a prediction after it had been decided
    cal["cal_hg"] = pd.to_numeric(fx.get("home_goals"), errors="coerce")
    cal["cal_ag"] = pd.to_numeric(fx.get("away_goals"), errors="coerce")
    return cal


def upcoming_round(competition: str, season: str, total_rounds: int) -> int:
    """The first matchday with a fixture still to play; the last one otherwise."""
    cfg = COMP_CONFIG.get(competition, {})
    cal = season_calendar(cfg.get("comp_id", competition), season)
    if cal.empty:
        return total_rounds
    today = pd.Timestamp.today().normalize()
    pending = cal.loc[pd.to_datetime(cal["date"]) >= today, "matchday"]
    return int(pending.min()) if len(pending) else total_rounds


def calendar_season(comp_id: str, season: str, df_season: pd.DataFrame):
    """The season as calendar-plus-results, or None when the calendar can't be trusted.

    The trust test is coverage: every result already in the foundation has to be
    found in the calendar. ESPN's older seasons fail it — about 10% of fixtures
    carry a club name that does not map, and an unmapped row would come back
    goalless, i.e. a match settled years ago offered as one still to play, with a
    prediction attached. When the test fails the caller slices the foundation as
    it always did.
    """
    cal = season_calendar(comp_id, season)
    if cal.empty:
        return None
    merged = cal.merge(df_season.drop(columns=["date"], errors="ignore"),
                       on=["home_team", "away_team"], how="left")
    if "home_goals" not in merged.columns:
        merged["home_goals"] = np.nan
        merged["away_goals"] = np.nan
    if int(merged["home_goals"].notna().sum()) < len(df_season):
        return None
    merged["competition"] = comp_id
    merged["season"] = season
    # `has_stats` is the foundation row, not the score: only that row carries
    # shots/corners/cards, so a match the calendar has settled but the foundation
    # has not ingested yet shows its result without inventing zeros for the rest
    merged["has_stats"] = merged["home_goals"].notna()
    merged["home_goals"] = merged["home_goals"].fillna(merged["cal_hg"])
    merged["away_goals"] = merged["away_goals"].fillna(merged["cal_ag"])
    return merged.drop(columns=["cal_hg", "cal_ag", "completed"], errors="ignore")


def get_round_fixtures(competition: str, season: str, round_num: int, df: pd.DataFrame):
    """One matchday's fixtures, from the real calendar when it is published.

    Played matches keep their foundation row, which carries the shots/corners/cards
    the match view needs; a fixture still to come simply has no goals, and the
    caller treats it as pending.
    """
    cfg = COMP_CONFIG.get(competition, {})
    comp_id = cfg.get("comp_id", competition)
    mask = (df["competition"] == comp_id) & (df["season"] == season)
    df_season = df[mask].copy().sort_values("date").reset_index(drop=True)

    full = calendar_season(comp_id, season, df_season)
    if full is not None:
        total_rounds = int(full["matchday"].max())
        rnd = full[full["matchday"] == round_num].sort_values("date")
        return rnd.reset_index(drop=True), total_rounds

    if df_season.empty:
        return pd.DataFrame(), 0
    n_teams = len(set(df_season["home_team"]) | set(df_season["away_team"]))
    n_per_round = max(1, n_teams // 2)
    total_rounds = max(1, -(-len(df_season) // n_per_round))
    start = (round_num - 1) * n_per_round
    return df_season.iloc[start:start + n_per_round], total_rounds


# ── European competitions ──────────────────────────────────────────────────────
# Champions/Europa/Conference used to be a page of their own. They are
# competitions, not a section: the main screen picks them like any league. The
# work below is the old page's, split into the two things it actually did —
# render a matchday, and simulate the rest of the tournament.
@st.cache_data(show_spinner="🌍  Cargando Elo europeo...", ttl=1800)
def euro_base():
    """Elo→goals calibration and today's ClubElo table, shared by every euro view."""
    from mundialytics.statistical_core.competition.european import (
        fetch_current_elo, load_calibration)
    return load_calibration(ROOT), fetch_current_elo(ROOT)


def euro_seasons(comp_id: str) -> list[int]:
    """Seasons on offer: whatever is cached on disk, plus the one starting now."""
    import re as _re
    from mundialytics.statistical_core.competition.european import FD_SLUG
    today = pd.Timestamp.today()
    auto_yr = today.year if today.month >= 7 else today.year - 1
    cached = {int(m.group(1)) for p in (ROOT / "data/external/uefa").glob(
        f"raw_{FD_SLUG[comp_id]}_*.csv") if (m := _re.search(r"_(\d{4})\.csv$", p.name))}
    return sorted(cached | {auto_yr}, reverse=True), auto_yr, cached


def euro_probs(elo_home: float, elo_away: float, calib: dict) -> dict:
    """1X2 and goal markets from the Elo gap, on the calibrated European scale."""
    from mundialytics.statistical_core.distributions import outcome_probabilities
    d400 = (elo_home - elo_away) / 400.0
    lh = float(np.exp(calib["c"] + calib["hfa"] + calib["b"] * d400))
    la = float(np.exp(calib["c"] - calib["b"] * d400))
    p = outcome_probabilities(lh, la, dixon_coles_rho=-0.07)
    return {**p, "lambda_home": lh, "lambda_away": la}


def euro_state(comp_label: str, season_yr: int):
    """Everything one European competition-season needs, or None when unavailable."""
    from mundialytics.statistical_core.competition.european import (
        make_resolver, normalize_club, parse_fixturedownload)
    cfg = COMP_CONFIG[comp_label]
    comp_id = cfg["comp_id"]
    calib, elo_all = euro_base()
    resolver = make_resolver(list(elo_all))
    elo_by_norm = {normalize_club(k): v for k, v in elo_all.items()}
    raw = _uefa_fixtures(comp_id, season_yr)
    league, ko, teams = None, None, {}
    if raw is not None:
        league, ko = parse_fixturedownload(raw, resolver)
        real = sorted(set(league.home) | set(league.away))
        teams = {t: elo_by_norm.get(normalize_club(t)) for t in real}
        teams = {t: e for t, e in teams.items() if e is not None}
    return dict(comp_id=comp_id, calib=calib, elo_all=elo_all, resolver=resolver,
                elo_by_norm=elo_by_norm, raw=raw, league=league, ko=ko, teams=teams)


def euro_elo_pair(state: dict, home_label: str, away_label: str):
    """(elo_home, elo_away, canonical_home, canonical_away) — None where unmapped."""
    from mundialytics.statistical_core.competition.european import normalize_club
    h_ce = state["resolver"](home_label)
    a_ce = state["resolver"](away_label)
    eh = state["elo_by_norm"].get(normalize_club(h_ce)) if h_ce else None
    ea = state["elo_by_norm"].get(normalize_club(a_ce)) if a_ce else None
    return eh, ea, h_ce, a_ce


def render_euro_match_detail(state: dict, comp_label: str, home_lbl: str, away_lbl: str,
                             result: str | None = None) -> None:
    """A European fixture in the same shape as a domestic one: 1X2, goal markets,
    team props and — where both clubs are big-5 — player props."""
    from mundialytics.statistical_core.competition.club_aliases import CLUBELO_TO_FD
    from mundialytics.statistical_core.competition.european import (
        load_event_calibration, predict_euro_events)
    from mundialytics.props.team_props import SIDE_LINES as _SIDE_EU

    eh, ea, h_ce, a_ce = euro_elo_pair(state, home_lbl, away_lbl)
    st.markdown(f"## ⚽ {home_lbl} vs {away_lbl}")
    if result:
        st.markdown(f"**Resultado real:** {result}")
    if not (eh and ea):
        st.warning("Sin Elo para alguno de los equipos: no se puede valorar este partido.")
        return

    p = euro_probs(eh, ea, state["calib"])
    lh, la = p["lambda_home"], p["lambda_away"]

    st.markdown("### Resultado")
    chart(prob_bar_chart(p["p_home_win"], p["p_draw"], p["p_away_win"], home_lbl, away_lbl))
    cols = st.columns(6)
    for col, (lab, val, color) in zip(cols, [
            (home_lbl, f"{p['p_home_win']:.1%}", mv_ui.HOME),
            ("Empate", f"{p['p_draw']:.1%}", mv_ui.DRAW),
            (away_lbl, f"{p['p_away_win']:.1%}", mv_ui.AWAY),
            ("xGoals", f"{lh:.2f} – {la:.2f}", ""),
            ("BTTS", f"{p['p_btts']:.1%}", prob_color(p["p_btts"], lo=0.0)),
            ("Over 2.5", f"{p['p_over_25']:.1%}", prob_color(p["p_over_25"], lo=0.0))]):
        col.markdown(metric_card(lab, val, color), unsafe_allow_html=True)

    st.download_button(
        "📸 Tarjeta para compartir",
        make_match_card(home_lbl, away_lbl, p["p_home_win"], p["p_draw"], p["p_away_win"],
                        p["p_over_25"], lh, la, sub=comp_label),
        file_name=f"{home_lbl}_{away_lbl}.png".replace(" ", "_"), mime="image/png",
        key=f"eu_card_{home_lbl}_{away_lbl}")

    st.markdown("### Mercados de goles")
    ocols = st.columns(4)
    for col, (lab, val) in zip(ocols, [("Over 1.5", p["p_over_15"]), ("Over 2.5", p["p_over_25"]),
                                       ("Over 3.5", p["p_over_35"]), ("BTTS", p["p_btts"])]):
        col.markdown(metric_card(lab, f"{val:.1%}", prob_color(val)), unsafe_allow_html=True)

    # team props: the full domestic model when both clubs are big-5, the
    # Elo→events mapping otherwise
    fd_h, fd_a = CLUBELO_TO_FD.get(h_ce), CLUBELO_TO_FD.get(a_ce)
    tp_eu, pp_eu = load_props_models()
    fx_props, src_props = None, ""
    if fd_h and fd_a and tp_eu is not None:
        fx_props = tp_eu.predict_fixture(fd_h, fd_a, lam_home=lh, lam_away=la)
        src_props = "modelo completo (ambos clubes big-5)"
    if not fx_props:
        ev_cal = load_event_calibration(ROOT)
        if ev_cal:
            fx_props = predict_euro_events(eh, ea, ev_cal, _SIDE_EU)
            src_props = "mapping Elo→eventos (calibrado en 17k partidos)"
    if fx_props:
        st.markdown(f"### 🎯 Props del partido "
                    f"<span style='color:{mv_ui.C['dim']};font-size:.75rem;font-weight:600'>"
                    f"{src_props}</span>", unsafe_allow_html=True)
        render_market_ladders(fx_props, home_lbl, away_lbl)

    if pp_eu is not None:
        shown = False
        for fd_t, lam_t, lbl_t in [(fd_h, lh, home_lbl), (fd_a, la, away_lbl)]:
            if not fd_t:
                continue
            try:
                pl = pp_eu.team_players_for_lambda(fd_t, lam_t)
            except Exception:
                continue
            if pl is None or pl.empty:
                continue
            if not shown:
                st.markdown("### Props de jugadores")
                shown = True
            st.markdown(f"**{lbl_t}**")
            render_player_props_table(pl.assign(side="home"), lbl_t.lower(), lbl_t.lower(),
                                      height=300)
        if shown:
            st.caption("Solo clubes de las 5 grandes ligas (cobertura de datos de jugadores); "
                       "ratios domésticos + contexto del partido europeo.")


def render_euro_matchday(comp_label: str) -> None:
    """League-phase matchday for a European competition, list → detail like a league."""
    state_key = f"eu_sel_{comp_label}"
    try:
        seasons, auto_yr, cached = euro_seasons(COMP_CONFIG[comp_label]["comp_id"])
    except Exception as exc:
        st.warning(f"Capa europea no disponible: {exc}")
        return

    season_yr = st.selectbox(
        "Temporada", seasons, key=f"eu_season_{comp_label}",
        format_func=lambda y: f"{y}/{str(y + 1)[2:]}"
        + ("  (próxima)" if y == auto_yr and y not in cached else ""),
        on_change=lambda: st.session_state.update({state_key: None}))
    state = euro_state(comp_label, season_yr)
    raw = state["raw"]
    if raw is None or state["league"] is None or not len(state["league"]):
        st.info(f"El calendario de {comp_label} {season_yr}/{str(season_yr + 1)[2:]} "
                "aún no está publicado. La simulación del torneo sigue disponible en "
                "🏆 Competición.")
        return

    sel = st.session_state.get(state_key)
    if sel:
        back, title = st.columns([1, 6], vertical_alignment="center")
        if back.button("← Jornada", key=f"eu_back_{comp_label}"):
            st.session_state[state_key] = None
            st.rerun()
        title.markdown(
            f'<div style="font-size:.78rem;color:{mv_ui.C["dim"]};font-weight:600">'
            f'{comp_label} · Jornada {sel["round"]}</div>', unsafe_allow_html=True)
        render_euro_match_detail(state, comp_label, sel["home"], sel["away"], sel["result"])
        return

    rr = raw.copy()
    rr["rnum"] = pd.to_numeric(rr["Round Number"], errors="coerce")
    lg = rr[rr["rnum"].notna()].copy()
    lg["played"] = lg["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+")
    rounds = sorted(lg["rnum"].astype(int).unique())
    pend = [r for r in rounds if not lg[lg.rnum == r]["played"].all()]
    default_r = pend[0] if pend else rounds[-1]
    rnd_sel = st.slider("Jornada", min(rounds), max(rounds), default_r,
                        key=f"eu_round_{comp_label}",
                        on_change=lambda: st.session_state.update({state_key: None}))
    sel_rows = lg[lg.rnum == rnd_sel].copy()
    sel_rows["fecha"] = pd.to_datetime(sel_rows["Date"], dayfirst=True, errors="coerce",
                                       format="mixed")
    sel_rows = sel_rows.sort_values("fecha")

    st.markdown(f"### Jornada {rnd_sel} de {max(rounds)} — fase liga")
    favs = load_favorites().get("teams", [])
    pend_count, last_day = 0, None
    for i, (_, r) in enumerate(sel_rows.iterrows()):
        h_lbl, a_lbl = str(r["Home Team"]), str(r["Away Team"])
        fecha = r["fecha"]
        day = None if pd.isna(fecha) else fecha.normalize()
        if day is not None and day != last_day:
            st.markdown(mv_ui.day_separator(day), unsafe_allow_html=True)
            last_day = day
        probs, score, note = None, None, ""
        if r["played"]:
            score = str(r["Result"])
        else:
            eh, ea, _, _ = euro_elo_pair(state, h_lbl, a_lbl)
            if eh and ea:
                p = euro_probs(eh, ea, state["calib"])
                probs = (p["p_home_win"], p["p_draw"], p["p_away_win"])
                note = f"O2.5 {p['p_over_25']:.0%}"
                pend_count += 1
            else:
                score = "sin Elo"
        star = "★ " if (h_lbl in favs or a_lbl in favs) else ""
        c_row, c_btn = st.columns([9, 2], vertical_alignment="center")
        c_row.markdown(mv_ui.fixture_row(star + h_lbl, a_lbl, score=score,
                                         probs=probs, note=note),
                       unsafe_allow_html=True)
        if c_btn.button("Analizar  ›", key=f"eu_btn_{comp_label}_{i}", use_container_width=True):
            st.session_state[state_key] = {"home": h_lbl, "away": a_lbl, "round": rnd_sel,
                                           "result": score if r["played"] else None}
            st.rerun()

    if pend_count:
        st.caption(f"{pend_count} partidos pendientes con predicción (1 · X · 2, Elo del día).")
        if st.button("📌 Registrar predicciones de esta jornada", key=f"eu_log_{comp_label}"):
            now_iso = pd.Timestamp.now().isoformat(timespec="seconds")
            rows_log = []
            for _, r in sel_rows.iterrows():
                if r["played"]:
                    continue
                eh, ea, h_ce, a_ce = euro_elo_pair(state, str(r["Home Team"]),
                                                   str(r["Away Team"]))
                if not (eh and ea):
                    continue
                p = euro_probs(eh, ea, state["calib"])
                base = dict(logged_at=now_iso, season=f"EU {season_yr}-{season_yr + 1}",
                            jornada=f"{comp_label} J{rnd_sel}",
                            partido=f"{h_ce} vs {a_ce}", fecha=str(r["fecha"])[:10],
                            home=h_ce, away=a_ce)
                trio = {"1": p["p_home_win"], "X": p["p_draw"], "2": p["p_away_win"]}
                pick = max(trio, key=trio.get)
                rows_log.append({**base, "mercado": "1X2", "ambito": "Total", "linea": "",
                                 "prob": round(trio[pick], 4), "seleccion": pick})
                rows_log.append({**base, "mercado": "Goles", "ambito": "Total", "linea": 2.5,
                                 "prob": round(p["p_over_25"], 4),
                                 "seleccion": "OVER" if p["p_over_25"] >= 0.5 else "UNDER"})
            n_new = append_predictions(rows_log)
            evaluate_prediction_log.clear()
            st.success(f"{n_new} predicciones europeas registradas — "
                       "track record en 📈 Resultados.")


def render_euro_simulation(comp_label: str) -> None:
    """Whole-tournament probabilities: Swiss league phase, playoff, knockouts."""
    from mundialytics.statistical_core.competition.european import (
        KO_ROUNDS, EuropeanTournament)
    try:
        seasons, auto_yr, cached = euro_seasons(COMP_CONFIG[comp_label]["comp_id"])
    except Exception as exc:
        st.warning(f"Capa europea no disponible: {exc}")
        return
    season_yr = st.selectbox("Temporada", seasons, key=f"eusim_season_{comp_label}",
                             format_func=lambda y: f"{y}/{str(y + 1)[2:]}")
    state = euro_state(comp_label, season_yr)
    comp_id, teams_eu = state["comp_id"], state["teams"]
    league_eu, ko_eu = state["league"], state["ko"]

    if state["raw"] is not None and league_eu is not None:
        played_lg = int(league_eu["home_goals"].notna().sum())
        ko_played = int(ko_eu["hg"].notna().sum()) if ko_eu is not None and len(ko_eu) else 0
        if ko_played:
            cur = next((r for r in reversed(KO_ROUNDS)
                        if len(ko_eu[(ko_eu["round"] == r) & ko_eu["hg"].notna()])), "playoff")
            estado = {"playoff": "Playoff", "r16": "Octavos", "qf": "Cuartos",
                      "sf": "Semifinales", "final": "Final"}[cur]
            st.success(f"Temporada {season_yr}/{str(season_yr + 1)[2:]} REAL cargada — "
                       f"eliminatorias en curso ({estado}). Las probabilidades parten del "
                       "estado actual del torneo.")
        else:
            st.success(f"Temporada {season_yr}/{str(season_yr + 1)[2:]} REAL cargada — "
                       f"fase liga {played_lg}/{len(league_eu)} partidos jugados.")
    else:
        tier = {"champions": (0, 36), "europa": (36, 72), "conference": (72, 108)}[comp_id]
        elo_sorted = sorted(state["elo_all"].items(), key=lambda kv: -kv[1])
        teams_eu = dict(elo_sorted[tier[0]:tier[1]])
        st.info(f"⚠️ La temporada {season_yr}/{str(season_yr + 1)[2:]} aún no está publicada — "
                "participantes ESTIMADOS por ranking Elo y modo PRE-SORTEO (cada simulación "
                "sortea una fase liga válida por bombos). En cuanto exista el calendario "
                "oficial, se cargará automáticamente.")

    res_key = f"eu_res_{comp_id}_{season_yr}"
    if st.button("🎲 Simular torneo (2.000 iteraciones)", key=f"eu_sim_{comp_id}",
                 type="primary"):
        with st.spinner("Simulando desde el estado actual..."):
            tour = EuropeanTournament(comp_id, teams_eu, state["calib"], league_eu, ko_eu)
            st.session_state[res_key] = tour.simulate(2000)

    res_eu = st.session_state.get(res_key)
    if res_eu is None:
        st.info(f"Listo para simular **{comp_label}** con {len(teams_eu)} equipos.")
        return

    top12 = res_eu.head(12)
    figeu = go.Figure(go.Bar(
        x=top12["p_champion"] * 100, y=top12["team"], orientation="h",
        marker_color=mv_ui.C["blue"], text=[f"{v:.1%}" for v in top12["p_champion"]],
        textposition="outside"))
    figeu.update_layout(height=360, margin=dict(l=0, r=40, t=10, b=0),
                        yaxis=dict(autorange="reversed"), xaxis_title="P(campeón) %")
    st.markdown(f"### ¿Quién gana la {comp_label}?")
    chart(figeu)
    st.download_button("📸 Tarjeta para compartir",
                       make_tournament_card(res_eu, comp_label),
                       file_name=f"{comp_id}_campeon.png", mime="image/png",
                       key=f"eu_card_dl_{comp_id}")

    tbl_eu = res_eu.rename(columns={
        "team": "Equipo", "elo": "Elo", "p_top24": "Pasar fase (Top 24)",
        "p_top8": "Top 8 directo", "p_playoff": "Playoff", "p_r16": "Octavos",
        "p_qf": "Cuartos", "p_sf": "Semis", "p_final": "Final", "p_champion": "Campeón"})
    pct_eu = ["Pasar fase (Top 24)", "Top 8 directo", "Playoff", "Octavos",
              "Cuartos", "Semis", "Final", "Campeón"]
    for c in pct_eu:
        tbl_eu[c] = (tbl_eu[c] * 100).round(1)
    st.dataframe(tbl_eu, hide_index=True, use_container_width=True, height=520,
                 column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                for c in pct_eu})
    st.caption("Fuerzas: ClubElo (escala única europea) con mapping Elo→goles calibrado "
               "sobre 20.000 partidos propios. Eliminatorias a doble partido con prórroga "
               "y penaltis; final única en campo neutral.")


# ── Top-scorer race ────────────────────────────────────────────────────────────
# The simulation itself lives in mundialytics.serving.scorer_race, shared with
# the HTTP API. Two front ends running two copies of a Monte Carlo would quote
# two different favourites for the Golden Boot, and a product that disagrees
# with itself about who is winning has no business publishing either number.
def _scorer_race_path(comp_id: str, season: str) -> Path:
    slug = f"{comp_id}_{season}".lower().replace(" ", "-").replace("/", "-")
    return ROOT / "data/processed/competition_cache" / f"scorers_{slug}.json"


def top_scorer_race(comp: str, season: str, n_sims: int = 20_000) -> pd.DataFrame:
    """P(finish top scorer) per player, with goals so far.

    Saved to disk and keyed on how many matches the season has played, so the
    page opens on a finished answer and only recomputes when a matchday lands.
    """
    import json

    from mundialytics.serving import scorer_race as sr

    comp_id = COMP_CONFIG[comp]["comp_id"]
    rows = df_clubs[(df_clubs["competition"] == comp_id) & (df_clubs["season"] == season)]
    n_played = int(len(rows))
    path = _scorer_race_path(comp_id, season)
    if path.exists():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            if blob.get("_nPlayed") == n_played and blob.get("players"):
                return _race_frame(blob["players"])
        except Exception:
            pass

    cal = season_calendar(comp_id, season)
    _, pp = load_props_models()
    if cal.empty or rows.empty or pp is None:
        return pd.DataFrame()
    remaining = cal[~cal["completed"].astype(bool)]

    prog = st.progress(0.0, text="Recorriendo el calendario que queda…")

    def on_progress(done, total, home, away):
        prog.progress(done / total,
                      text=f"Partido {done} de {total} · {home.title()} vs {away.title()}")

    goals = sr.goals_so_far(
        _player_match_actuals(),
        set(rows["home_team"]) | set(rows["away_team"]),
        since=str(rows["date"].min())[:10],
        until=str(pd.Timestamp.today())[:10],
    )
    rates = sr.remaining_rates(
        remaining,
        lambda h, a: predict_safe(engine_clubs, h, a, comp, False),
        pp.team_players_for_lambda,
        progress=on_progress,
    )
    prog.empty()
    race = sr.simulate(goals, rates, n_sims=n_sims)
    if race.empty:
        return pd.DataFrame()

    players = [
        {"player": str(r.player), "team": str(r.team).title(), "goals": int(r.goals),
         "expected": float(r.expected), "p": float(r.pTopScorer)}
        for r in race.itertuples()
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"_nPlayed": n_played, "players": players},
                                   default=str), encoding="utf-8")
    except OSError:
        pass
    return _race_frame(players)


def _race_frame(players: list[dict]) -> pd.DataFrame:
    """The shared payload, in the Spanish column names this page renders."""
    return pd.DataFrame([
        {"Jugador": p["player"], "Equipo": p["team"], "Goles": int(p["goals"]),
         "Goles esperados": round(float(p["expected"]), 1),
         "P(Bota de Oro)": float(p["p"])}
        for p in players
    ])



# ── Sidebar ────────────────────────────────────────────────────────────────────
# Each page carries a URL slug so a view can be linked and survives a refresh —
# the label alone could not, and the labels themselves (emoji + double space)
# were being compared as literals in eight places.
# Props and the European competitions used to be pages of their own. They are
# not sections of an app, they are things you look at about a match or a
# competition — props now live inside the match analysis, and Champions/Europa/
# Conference are competitions you pick on the main screen like any other.
PAGES = [
    ("jornada",    "🗓️  Jornada"),
    ("liga",       "📊  Pronóstico de liga"),
    ("simulador",  "🏆  Competición"),
    ("resultados", "📈  Resultados"),
    ("premios",    "🥇  Premios Individuales"),
    ("squadlab",   "🧪  SquadLab"),
]
PAGE_LABELS = [lbl for _, lbl in PAGES]
SLUG_OF = {lbl: slug for slug, lbl in PAGES}
LABEL_OF = dict(PAGES)

st.sidebar.markdown(brand_header(), unsafe_allow_html=True)
st.sidebar.divider()

# A `?p=` nobody recognises is a wrong address, not a reason to silently show
# something else: an old bookmark would have looked like the app had simply
# forgotten the page.
_slug = st.query_params.get("p")
_unknown_slug = _slug if (_slug and _slug not in LABEL_OF) else None
_default = PAGE_LABELS.index(LABEL_OF[_slug]) if _slug in LABEL_OF else 0
page = st.sidebar.radio("Sección", PAGE_LABELS, index=_default, key="nav_page",
                        label_visibility="collapsed")
if not _unknown_slug and st.query_params.get("p") != SLUG_OF[page]:
    st.query_params["p"] = SLUG_OF[page]

page_slug = SLUG_OF[page]

st.sidebar.divider()
# Read off the data rather than typed in: the old footer still claimed "Big5
# 2021-26" three seasons after that stopped being true, and a stale coverage
# claim on a prediction app is worse than none.
_clubs_from = str(df_clubs["season"].min())[:4]
_clubs_to = str(df_clubs["season"].max())[-4:]
_intl_from = str(df_intl["date"].min())[:4]
_last_match = df_clubs["date"].max()
st.sidebar.markdown(
    f"<div style='font-size:.68rem;color:{mv_ui.C['dim']};line-height:1.6'>"
    f"Big5 {_clubs_from}–{_clubs_to} · Selecciones desde {_intl_from} · "
    "Europa (ClubElo)<br>"
    f"Datos hasta el <b>{mv_ui.fmt_date_long(_last_match)}</b> · "
    f"{mv_ui.num(len(df_clubs))} partidos<br>"
    f"<span style='color:{mv_ui.C['green']}'>●</span> 1X2 a 4% del cierre Bet365 · "
    "props 5/5 folds validados</div>", unsafe_allow_html=True)


if _unknown_slug:
    st.markdown(mv_ui.notice(
        "🧭", "Esa sección no existe",
        f"La dirección pedía <code>?p={_unknown_slug}</code>, y Mundialytics no tiene "
        "ninguna sección con ese nombre. Puede que el enlace sea antiguo: "
        "<b>Props</b> ahora vive dentro del análisis de cada partido, y "
        "<b>Europa</b> es una competición más del selector de Jornada.",
        links=[(lbl, f"?p={slug}") for slug, lbl in PAGES]), unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  JORNADA  (competición → jornada → partido, todo encadenado)
# ══════════════════════════════════════════════════════════════════════════════
if page_slug == "jornada":
    st.title("🗓️  Jornada")
    st.caption("Elige una competición y una jornada para ver todos sus partidos y análisis.")

    if "j_selected" not in st.session_state:
        st.session_state.j_selected = None

    # Every competition with a matchday lives in this one selector — the European
    # ones included. Starred competitions come first.
    comp_options = favorites_first(MATCHDAY_COMPETITIONS)
    col_c, col_star, col_rest = st.columns([3, 0.5, 4], vertical_alignment="bottom")
    with col_c:
        comp_j = st.selectbox("Competición", comp_options, key="j_comp",
                              on_change=lambda: st.session_state.update(j_selected=None))
    with col_star:
        star_button("competitions", comp_j, key="j_comp_star")

    if COMP_CONFIG[comp_j]["type"] == "euro":
        render_euro_matchday(comp_j)
        st.stop()

    with col_rest:
        cs1, cs2 = st.columns([3, 2], vertical_alignment="bottom")
        with cs1:
            comp_id_j = COMP_CONFIG[comp_j]["comp_id"]
            seasons_j = sorted(df_clubs.loc[df_clubs["competition"] == comp_id_j, "season"].unique(),
                               reverse=True)
            season_j = st.selectbox("Temporada", seasons_j, key="j_season",
                                    on_change=lambda: st.session_state.update(j_selected=None))
        with cs2:
            neutral_j = st.checkbox("Campo neutro", value=False, key="j_neutral")

    df_round, total_rounds = get_round_fixtures(comp_j, season_j, 1, df_clubs)
    if total_rounds == 0:
        st.warning("No hay datos disponibles para esta competición/temporada.")
        st.stop()

    # land on the round about to be played, not the last one on record: the
    # useful default for a prediction app is the fixture list nobody knows yet
    jornada_num = st.slider("Jornada", 1, total_rounds,
                            upcoming_round(comp_j, season_j, total_rounds), key="j_num",
                            on_change=lambda: st.session_state.update(j_selected=None))
    df_round, _ = get_round_fixtures(comp_j, season_j, jornada_num, df_clubs)

    if df_round.empty:
        st.info("Sin partidos para esta jornada.")
        st.stop()

    sel = st.session_state.j_selected

    # Master → detail, not master-then-detail-below: the list is ten rows tall, so
    # picking a match used to mean scrolling past every other one to reach the
    # analysis. Selecting swaps the view; a back button returns to the round.
    if sel:
        back, title = st.columns([1, 6], vertical_alignment="center")
        if back.button("← Jornada", key="j_back"):
            st.session_state.j_selected = None
            st.rerun()
        title.markdown(
            f'<div style="font-size:.78rem;color:{mv_ui.C["dim"]};font-weight:600">'
            f'{comp_j} · Jornada {jornada_num}</div>', unsafe_allow_html=True)
        render_partido_detail(sel["home"], sel["away"], comp_j, neutral_j,
                              actual_result=sel["result"], row=sel.get("row"))
        st.stop()

    date_min, date_max = df_round["date"].min(), df_round["date"].max()
    st.markdown(f"### Jornada {jornada_num} de {total_rounds} "
                f"<span style='color:{mv_ui.C['dim']};font-size:.85rem;font-weight:600'>"
                f"({mv_ui.fmt_date(date_min)} – {mv_ui.fmt_date(date_max)} "
                f"{date_max.year})</span>", unsafe_allow_html=True)

    # a pending fixture shows its 1X2 in the row itself: the list is the matchday
    # overview, and an empty "vs" made every unplayed row look identical. Cached
    # per round so dragging the slider does not re-predict what it already knows.
    pending_pairs = tuple(
        (r.home_team, r.away_team) for r in df_round.itertuples()
        if not (pd.notna(r.home_goals) and pd.notna(r.away_goals)))
    round_probs = round_probabilities(pending_pairs, comp_j, neutral_j) if pending_pairs else {}

    fav_teams = load_favorites().get("teams", [])
    last_day = None
    for i, (_, row) in enumerate(df_round.iterrows()):
        home, away = row["home_team"], row["away_team"]
        played = pd.notna(row["home_goals"]) and pd.notna(row["away_goals"])
        result_str = f"{int(row['home_goals'])} – {int(row['away_goals'])}" if played else None
        has_stats = bool(row.get("has_stats", played))
        probs = round_probs.get((home, away))

        # one heading per matchday date: ten rows in a block read as a wall, the
        # same ten split by day read as a weekend
        day = pd.to_datetime(row["date"]).normalize() if pd.notna(row["date"]) else None
        if day is not None and day != last_day:
            st.markdown(mv_ui.day_separator(day), unsafe_allow_html=True)
            last_day = day

        star = "★ " if (home in fav_teams or away in fav_teams) else ""
        c_row, c_btn = st.columns([9, 2], vertical_alignment="center")
        # date omitted on purpose: the day separator above already carries it
        c_row.markdown(mv_ui.fixture_row(star + home.title(), away.title(), score=result_str,
                                         probs=probs),
                       unsafe_allow_html=True)
        if c_btn.button("Analizar  ›", key=f"j_btn_{i}", use_container_width=True,
                        help="Resultado real y estadísticas" if has_stats else "Predicción completa"):
            st.session_state.j_selected = {
                "home": home, "away": away,
                "result": result_str,
                "row": row.to_dict() if has_stats else None,
            }
            st.rerun()

    # Matchday-level tools, where the matchday is: the round scanner and the
    # pre-kickoff log used to sit on a Props page of their own, one click away
    # from the fixtures they are about.
    if pending_pairs:
        st.markdown("---")
        st.markdown("### 🔍 Herramientas de la jornada")
        tp_j, pp_j = load_props_models()
        col_scan, col_log = st.columns(2)
        with col_log:
            if tp_j is not None and st.button("📌 Registrar predicciones de la jornada",
                                              key="j_log_btn", use_container_width=True):
                n_new = log_round_predictions(df_round, comp_j, season_j, jornada_num, tp_j, pp_j)
                evaluate_prediction_log.clear()
                st.success(f"{n_new} predicciones nuevas registradas (solo partidos aún no "
                           "jugados; las repetidas conservan el primer registro). "
                           "Track record en 📈 Resultados.")
        with col_scan:
            scan_clicked = tp_j is not None and st.button(
                "Escanear jornada", key="j_scan_btn", use_container_width=True)
        scan_key = f"scan_{comp_j}_{season_j}_{jornada_num}"
        if scan_clicked:
            rows_s = []
            pend_df = df_round[df_round["home_goals"].isna()]
            prog = st.progress(0.0)
            for k, r in enumerate(pend_df.itertuples()):
                pr = predict_safe(engine_clubs, r.home_team, r.away_team, comp_j, False)
                fx_s = tp_j.predict_fixture(r.home_team, r.away_team,
                                            lam_home=pr.lambda_home if pr else None,
                                            lam_away=pr.lambda_away if pr else None)
                label = f"{r.home_team.title()} vs {r.away_team.title()}"
                for mk, d in fx_s.items():
                    for ln, p in d.get("over", {}).items():
                        rows_s.append({"Partido": label, "Mercado": MARKET_ES.get(mk, mk),
                                       "Ámbito": "Total", "Línea": ln, "P(Over)": p})
                    for skey, amb in [("over_home", r.home_team.title()),
                                      ("over_away", r.away_team.title())]:
                        for ln, p in d.get(skey, {}).items():
                            rows_s.append({"Partido": label, "Mercado": MARKET_ES.get(mk, mk),
                                           "Ámbito": amb, "Línea": ln, "P(Over)": p})
                prog.progress((k + 1) / max(len(pend_df), 1))
            prog.empty()
            st.session_state[scan_key] = pd.DataFrame(rows_s)
        if scan_key in st.session_state and len(st.session_state[scan_key]):
            sc = st.session_state[scan_key].copy()
            mks = st.multiselect("Mercados", sorted(sc["Mercado"].unique()),
                                 default=sorted(sc["Mercado"].unique()), key="j_scan_mks")
            sc = sc[sc["Mercado"].isin(mks)]
            sc["Señal"] = (sc["P(Over)"] - 0.5).abs()
            sc["Lado"] = np.where(sc["P(Over)"] >= 0.5, "OVER", "UNDER")
            sc["Confianza"] = np.where(sc["Lado"] == "OVER", sc["P(Over)"], 1 - sc["P(Over)"])
            top = sc.sort_values("Señal", ascending=False).head(30)
            top = top[["Partido", "Mercado", "Ámbito", "Línea", "Lado", "Confianza"]]
            top["Confianza"] = (top["Confianza"] * 100).round(1)
            st.dataframe(top, hide_index=True, use_container_width=True, height=500,
                         column_config={"Confianza": st.column_config.NumberColumn(format="%.1f%%")})



# ══════════════════════════════════════════════════════════════════════════════
#  COMPETICIÓN  (formato auto-detectado, sin elegir liga/torneo a mano)
# ══════════════════════════════════════════════════════════════════════════════
elif page_slug == "simulador":
    st.title("🏆  Simulación de competición")

    col_cs, col_cstar, _sp = st.columns([3, 0.5, 4], vertical_alignment="bottom")
    with col_cs:
        competition_c = st.selectbox("Competición", favorites_first(COMPETITIONS),
                                     key="comp_select")
    with col_cstar:
        star_button("competitions", competition_c, key="c_comp_star")
    cfg = COMP_CONFIG[competition_c]

    # ── UEFA (formato suizo real sobre ClubElo) ─────────────────────────────
    if cfg["type"] == "euro":
        render_euro_simulation(competition_c)
        st.stop()

    eng_c, teams_c, _ = pick_engine(competition_c)

    # ── LIGA (formato detectado automáticamente) ────────────────────────────
    if cfg["type"] == "liga":
        comp_id_c = cfg["comp_id"]
        comp_rows = df_clubs[df_clubs["competition"] == comp_id_c]
        seasons_c = sorted(comp_rows["season"].unique(), reverse=True)

        col_l1, col_l2 = st.columns([2, 3])
        with col_l1:
            # The field is one season's, not the league's whole history: without
            # this the simulation ran on every club that ever played in LaLiga —
            # 42 teams instead of 20, a competition that never existed, and four
            # times the work to compute it.
            season_c = st.selectbox("Temporada", seasons_c, key="liga_season")
            n_sims_l = st.select_slider("Simulaciones", [1_000, 10_000, 50_000, 100_000], value=10_000, key="liga_n")
            home_away_l = st.checkbox("Ida y vuelta", value=True, key="liga_ha")

        season_rows = comp_rows[comp_rows["season"] == season_c]
        league_teams = sorted(set(season_rows["home_team"]) | set(season_rows["away_team"]))
        league_teams = [t for t in league_teams if t in teams_c]
        with col_l1:
            st.caption(f"{len(league_teams)} equipos de {competition_c} {season_c}, "
                       "cargados del calendario real.")

        # A 20-team double round-robin at 10,000 draws is ~40s of work. Opening
        # the page should not spend it unasked: the run is explicit, and re-running
        # the same setup is free because `simulate_league_cached` already has it.
        key_l = ("liga", competition_c, season_c, n_sims_l, home_away_l, tuple(league_teams))
        with col_l1:
            if st.button("▶  Simular temporada", key="liga_go", type="primary",
                         use_container_width=True):
                st.session_state.sim_key = key_l

        with col_l2:
            if len(league_teams) < 2:
                st.warning("No se encontraron suficientes equipos para esta liga en el dataset.")
            elif st.session_state.get("sim_key") != key_l:
                st.info(f"Listo para simular **{competition_c} {season_c}** con "
                        f"{len(league_teams)} equipos, {mv_ui.num(n_sims_l)} veces.")
            else:
                with st.spinner(f"Simulando {mv_ui.num(n_sims_l)} temporadas..."):
                    stats_l = simulate_league_cached(tuple(league_teams), n_sims_l,
                                                     competition_c, home_away_l)
                df_l = stats_l.copy()
                df_l["team"] = df_l["team"].str.title()

                st.markdown(f"### Resultados — {mv_ui.num(n_sims_l)} simulaciones")
                show_l = [c for c in ["team","p_win","p_top2","p_top4","avg_pts","avg_goals"] if c in df_l.columns]
                fmt_l = df_l[show_l].copy()
                for c in ["p_win","p_top2","p_top4"]:
                    if c in fmt_l: fmt_l[c] = fmt_l[c].apply(lambda x: f"{x:.1%}")
                if "avg_pts" in fmt_l: fmt_l["avg_pts"] = fmt_l["avg_pts"].apply(lambda x: f"{x:.1f}")
                if "avg_goals" in fmt_l: fmt_l["avg_goals"] = fmt_l["avg_goals"].apply(lambda x: f"{x:.1f}")
                fmt_l.columns = [c.replace("_"," ").title() for c in fmt_l.columns]
                st.dataframe(fmt_l, hide_index=True, use_container_width=True)

                if "p_win" in stats_l.columns:
                    chart(tourn_bar(stats_l, "p_win", f"% Campeón de {competition_c}", mv_ui.C["amber"]))

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

        groups_key = tuple((g, tuple(t)) for g, t in groups.items())
        key_t = ("torneo", competition_c, n_sims_t, neutral_t, bracket_fmt, groups_key)
        with col_t1:
            if st.button("▶  Simular torneo", key="t_go", type="primary",
                         use_container_width=True):
                st.session_state.sim_key = key_t

        with col_t2:
            if not groups:
                st.warning("Ningún equipo de esta competición está en el dataset seleccionado.")
            else:
                for g, tms in groups.items():
                    st.markdown(f"**Grupo {g}**: " + "  ·  ".join(t.title() for t in tms))

            if groups and st.session_state.get("sim_key") != key_t:
                st.info(f"Listo para simular **{competition_c}** {mv_ui.num(n_sims_t)} veces.")
            elif groups:
                with st.spinner(f"Simulando {competition_c} ({mv_ui.num(n_sims_t)} veces)..."):
                    df_t = simulate_tournament_cached(
                        tuple((g, tuple(t)) for g, t in groups.items()), n_sims_t,
                        competition_c, neutral_t, bracket_fmt,
                        cfg["engine"])

                st.markdown(f"### Resultados — {mv_ui.num(n_sims_t)} simulaciones")
                n_groups = len(groups)
                st.markdown(bracket_html(df_t, n_groups), unsafe_allow_html=True)

                st.markdown("---")
                tab1, tab2, tab3, tab4 = st.tabs(["🏆 Campeón", "🎯 Final", "🏅 Semifinal", "📈 Grupos"])
                with tab1:
                    if "p_win" in df_t: chart(tourn_bar(df_t, "p_win", "% Campeón", mv_ui.C["amber"]))
                with tab2:
                    if "p_final" in df_t: chart(tourn_bar(df_t, "p_final", "% Llegar a la Final", mv_ui.C["violet"]))
                with tab3:
                    if "p_semis" in df_t: chart(tourn_bar(df_t, "p_semis", "% Llegar a Semifinales", mv_ui.C["blue"]))
                with tab4:
                    if "p_advance_groups" in df_t: chart(tourn_bar(df_t, "p_advance_groups", "% Pasar Fase de Grupos", mv_ui.C["green"]))


# ══════════════════════════════════════════════════════════════════════════════
#  PREMIOS INDIVIDUALES
# ══════════════════════════════════════════════════════════════════════════════
elif page_slug == "resultados":
    st.title("📈  Resultados y fiabilidad")
    st.caption("Validación fuera de muestra, comparación con el mercado y track record en vivo.")

    # ── hero ───────────────────────────────────────────────────────────────────
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.markdown(metric_card("Partidos de validación", "10.400+"), unsafe_allow_html=True)
    hc2.markdown(metric_card("vs cierre de Bet365 (1X2)", "4.1%", mv_ui.C["green"]), unsafe_allow_html=True)
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
                            marker_color=mv_ui.C["blue"], text=[f"{v:.4f}" for v in bm["Distancia al cierre"]],
                            textposition="outside"))
    figb.add_vline(x=0.012, line_dash="dot", line_color=mv_ui.C["dim"],
                   annotation_text="rango élite académico", annotation_position="top")
    figb.update_layout(height=240, margin=dict(l=0, r=0, t=20, b=0),
                       xaxis=dict(range=[0, 0.014], title="RPS gap (menos = mejor)"),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    chart(figb)

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
                                  line=dict(dash="dot", color=mv_ui.C["dim"])))
        figc.add_trace(go.Scatter(x=cal["pred"], y=cal["real"], mode="markers+lines",
                                  name="modelo", marker=dict(size=9, color=mv_ui.C["blue"])))
        figc.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="Probabilidad anunciada", yaxis_title="Frecuencia real",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=1.1))
        chart(figc)
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
        pc2.markdown(metric_card("Predicciones en juego", mv_ui.num(len(pend))),
                     unsafe_allow_html=True)
        pc3.markdown(metric_card("Se resuelven", f"{pend['fecha'].min()} → {pend['fecha'].max()}"),
                     unsafe_allow_html=True)

    ev_all = evaluate_prediction_log(len(df_clubs))
    # Player markets are split out on purpose. Their accuracy is dominated by the
    # base rate: only 8.6% of starters score, the model never puts anyone above
    # 50%, so answering "no" is right ~92% of the time. Leaving them in the
    # headline would lift it from ~65% to ~80% with no better model. They are
    # judged on ranking instead, in their own panel below.
    is_pl = ev_all["es_jugador"] if "es_jugador" in ev_all.columns else pd.Series(dtype=bool)
    ev_players = ev_all[is_pl] if len(is_pl) else ev_all.iloc[0:0]
    ev = ev_all[~is_pl] if len(is_pl) else ev_all

    if ev_all.empty:
        st.info("Predicciones ya registradas y esperando a que se jueguen los partidos. "
                "En cuanto haya resultados aparecerá aquí el acierto real frente al anunciado.")
    if not ev.empty:
        n = len(ev)
        acc = ev["acierto"].mean()
        exp = ev["confianza"].mean()
        tc1, tc2, tc3 = st.columns(3)
        tc1.markdown(metric_card("Predicciones evaluadas", f"{n}"), unsafe_allow_html=True)
        tc2.markdown(metric_card("Acierto real", f"{acc:.1%}",
                                 mv_ui.C["green"] if acc >= exp - 0.02 else mv_ui.C["red"]), unsafe_allow_html=True)
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

    render_player_track_record(ev_players)

elif page_slug == "premios":
    st.title("🥇  Premios Individuales")

    tab_gb, tab_profile, tab_teams = st.tabs(
        ["🥾 Bota de Oro", "👤 Perfil de jugador", "📊 Ranking de equipos"])

    # ── BOTA DE ORO — la carrera entera, ya calculada ───────────────────────
    # It used to ask you to name the players to follow, which made it a
    # calculator rather than an award race: you only ever saw the answer for the
    # four names you already suspected. The race is over every player who has
    # scored or is expected to, computed once per matchday and saved.
    with tab_gb:
        cg1, cg2 = st.columns([2, 2])
        with cg1:
            comp_gb = st.selectbox("Competición", favorites_first(LEAGUE_COMPETITIONS),
                                   key="gb_comp")
        with cg2:
            comp_id_gb = COMP_CONFIG[comp_gb]["comp_id"]
            seasons_gb = sorted(df_clubs.loc[df_clubs["competition"] == comp_id_gb,
                                             "season"].unique(), reverse=True)
            season_gb = st.selectbox("Temporada", seasons_gb, key="gb_season")

        with st.spinner("Simulando lo que queda de temporada jugador a jugador… "
                        "(se guarda para las próximas visitas)"):
            race = top_scorer_race(comp_gb, season_gb)

        if race is None or race.empty:
            st.info("Todavía no hay datos de goleadores para esta temporada.")
        else:
            leader = race.iloc[0]
            top_now = race.sort_values("Goles", ascending=False).iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.markdown(metric_card("Favorito", str(leader["Jugador"]),
                                    mv_ui.C["amber"],
                                    sub=f"{leader['P(Bota de Oro)']:.0%} · "
                                        f"{int(leader['Goles'])} goles ahora"),
                        unsafe_allow_html=True)
            m2.markdown(metric_card("Líder actual", str(top_now["Jugador"]), "",
                                    sub=f"{int(top_now['Goles'])} goles"),
                        unsafe_allow_html=True)
            m3.markdown(metric_card("Jugadores en carrera",
                                    f"{int((race['P(Bota de Oro)'] > 0.005).sum())}",
                                    sub="con al menos un 0,5% de opciones"),
                        unsafe_allow_html=True)

            top15 = race.head(15).copy()
            st.markdown("### Probabilidad de acabar como máximo goleador")
            lead_p = max(float(top15["P(Bota de Oro)"].max()), 1e-9)
            for _, r in top15.iterrows():
                p_win = float(r["P(Bota de Oro)"])
                c_name, c_bar = st.columns([3, 5], vertical_alignment="center")
                c_name.markdown(
                    f'<div style="font-weight:650;font-size:.92rem;color:{mv_ui.C["text"]}">'
                    f'{r["Jugador"]}</div>'
                    f'<div class="mv-bar-sub">{str(r["Equipo"]).title()} · '
                    f'<b style="color:{mv_ui.C["text"]}">{int(r["Goles"])} '
                    f'{"gol" if int(r["Goles"]) == 1 else "goles"}</b> ahora · '
                    f'{float(r["Goles esperados"]):.1f} esperados al final</div>',
                    unsafe_allow_html=True)
                c_bar.markdown(
                    bar_row("", p_win / lead_p, f"{p_win:.1%}", mv_ui.C["amber"], value_w=52),
                    unsafe_allow_html=True)

            st.caption("Barras a escala del favorito. La probabilidad sale de simular "
                       "20.000 veces lo que queda de liga: los goles de cada jugador en "
                       "cada partido pendiente salen del modelo de props, sobre los "
                       "goles esperados de su equipo en ese partido concreto.")
            with st.expander("Tabla completa"):
                st.dataframe(race, hide_index=True, use_container_width=True, height=520,
                             column_config={"P(Bota de Oro)":
                                            st.column_config.NumberColumn(format="%.1f%%")})

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
                        marker_color=mv_ui.C["blue"],
                        text=[f"{v[1]:.2f}" for v in vals_sorted], textposition="outside",
                    ))
                    fig_p.update_layout(
                        height=260, margin=dict(l=10,r=10,t=10,b=50),
                        yaxis_title="Por partido",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    chart(fig_p)

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
                                   name="Ataque", orientation="h", marker_color=mv_ui.C["blue"]))
        fig_rank.add_trace(go.Bar(y=top_r["team"][::-1], x=top_r["defense"][::-1],
                                   name="Defensa", orientation="h", marker_color=mv_ui.C["green"]))
        fig_rank.update_layout(
            barmode="stack", title="Ataque + Defensa por equipo (parámetros MLE, escala logarítmica)",
            height=max(350, len(top_r)*22),
            margin=dict(l=120,r=40,t=50,b=20),
            xaxis_title="Parámetro MLE (relativo a la media de su liga)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        chart(fig_rank)


# ══════════════════════════════════════════════════════════════════════════════
#  SQUADLAB
# ══════════════════════════════════════════════════════════════════════════════
elif page_slug == "liga":
    spec = importlib.util.spec_from_file_location(
        "competition_forecast_page", Path(__file__).parent / "competition_forecast_page.py")
    cf_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf_page)
    cf_page.render()

elif page_slug == "squadlab":
    spec = importlib.util.spec_from_file_location(
        "squadlab_page", Path(__file__).parent / "squadlab_page.py")
    sl_page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sl_page)
    sl_page.render(engine=engine_clubs, df_clubs=df_clubs)
