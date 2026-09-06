"""
Mundialytics — shared UI layer (design tokens, CSS, components, chart theme).

Everything visual lives here so the pages can stop hand-writing inline styles.
Three rules the app follows since this module exists:

1. No page hard-codes a hex colour. Ask this module (`C`) or use a CSS token,
   otherwise pages drift apart — the league-forecast page used to paint its bars
   #2a78d6 while every other page used #3b82f6, and light-theme greys (#f3f4f6
   gridlines, #e5e7eb borders) leaked into a dark app.
2. One component per visual idea. There were three separate implementations of
   "label + progress bar + percentage"; now there is `bar_row`.
3. Plotly figures go through `chart()`, which paints them from the same tokens
   and disables Streamlit's own theme so what you set is what you get.
"""
from __future__ import annotations

import streamlit as st

# ── Design tokens ──────────────────────────────────────────────────────────────
# Mirrors the CSS custom properties below; used where Python builds the markup
# (plotly figures, f-string HTML) and a var() is not available.
C = {
    "bg":        "#0b1220",
    "surface":   "#131c2e",
    "surface2":  "#182338",
    "line":      "#22314c",
    "line_soft": "#1a2740",
    "blue":      "#3b82f6",
    "blue_soft": "#60a5fa",
    "green":     "#10b981",
    "amber":     "#f59e0b",
    "red":       "#ef4444",
    "violet":    "#8b5cf6",
    "text":      "#e6edf7",
    "muted":     "#93a3bd",
    "dim":       "#6f7f9c",
}

# home / draw / away keep one meaning app-wide: blue is the home side, red the
# away side, grey the draw. Charts and HTML both read them from here.
HOME, DRAW, AWAY = C["blue"], "#64748b", C["red"]

# Categorical series (league-forecast evolution lines, multi-team charts).
SERIES = [C["blue"], C["violet"], C["red"], C["green"], C["amber"], C["blue_soft"]]

# Sequential scale for heatmaps. Plotly's "Blues" bottoms out at near-white, so
# a mostly-low matrix rendered as a white slab on a dark page; this one starts
# at the panel colour and stays inside the app.
HEAT = [[0.0, C["surface"]], [0.25, "#1c3358"], [0.6, "#2a5fae"], [1.0, C["blue_soft"]]]

_MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun",
              "jul", "ago", "sep", "oct", "nov", "dic"]


CSS = """
<style>
:root{
  --mv-bg:#0b1220; --mv-surface:#131c2e; --mv-surface-2:#182338;
  --mv-line:#22314c; --mv-line-soft:#1a2740;
  --mv-blue:#3b82f6; --mv-blue-soft:#60a5fa; --mv-blue-ghost:rgba(59,130,246,.13);
  --mv-green:#10b981; --mv-amber:#f59e0b; --mv-red:#ef4444;
  --mv-text:#e6edf7; --mv-muted:#93a3bd; --mv-dim:#6f7f9c;
  --mv-r:12px;
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
html,body,[class*="css"]{font-feature-settings:"tnum" 1;}
.block-container{padding-top:2.4rem;padding-bottom:4rem;max-width:1440px}
[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"]{display:none}
div[data-testid="stSidebarNav"]{display:none}
a{color:var(--mv-blue-soft);text-decoration:none}
a:hover{text-decoration:underline}

/* ── Typography: one scale, one rhythm ────────────────────────────────────── */
h1{font-size:1.95rem!important;font-weight:800!important;letter-spacing:-.028em;
   margin-bottom:.1rem!important}
h2{font-weight:700!important;letter-spacing:-.018em}
h3{font-size:1.06rem!important;font-weight:700!important;letter-spacing:-.008em;
   margin:1.9rem 0 .7rem!important;padding-bottom:.4rem;
   border-bottom:1px solid var(--mv-line-soft)}
h4{font-size:.94rem!important;font-weight:700!important;margin:1.1rem 0 .4rem!important;
   color:var(--mv-text)}
[data-testid="stCaptionContainer"] p{color:var(--mv-muted)!important;font-size:.8rem!important}
hr{border-color:var(--mv-line-soft)!important;margin:1.6rem 0!important}

/* ── Sidebar: radio group rendered as a nav rail ──────────────────────────── */
section[data-testid="stSidebar"]{background:#0a111f;border-right:1px solid var(--mv-line)}
section[data-testid="stSidebar"] [role="radiogroup"]{gap:3px!important}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"]{
  padding:7px 11px;margin:0;border-radius:9px;border:1px solid transparent;
  cursor:pointer;transition:background .13s,border-color .13s}
section[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"]:hover{
  background:var(--mv-surface)}
/* the bare radio dot — the pill itself is the affordance */
section[data-testid="stSidebar"] [role="radiogroup"] label[data-baseweb="radio"]>div:first-child{
  display:none!important}
section[data-testid="stSidebar"] [role="radiogroup"] label p{
  font-size:.885rem!important;font-weight:600;color:var(--mv-muted);margin:0!important}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:var(--mv-blue-ghost);border-color:rgba(59,130,246,.38)}
/* fallback for browsers without :has() — the active label still marks itself */
section[data-testid="stSidebar"] [role="radiogroup"] label input:checked ~ div p{
  color:var(--mv-text)!important}

/* ── Brand wordmark ───────────────────────────────────────────────────────── */
.mv-brand{display:flex;align-items:center;gap:11px;padding:2px 2px 4px}
.mv-logo{width:38px;height:38px;border-radius:11px;flex:0 0 38px;
  background:linear-gradient(140deg,var(--mv-blue),#1e40af);
  display:flex;align-items:center;justify-content:center;
  font-weight:900;font-size:20px;color:#fff;box-shadow:0 3px 14px rgba(59,130,246,.35)}
.mv-name{font-size:1.3rem;font-weight:800;letter-spacing:-.035em;line-height:1;color:var(--mv-text)}
.mv-name .mv-accent{color:var(--mv-blue-soft)}
.mv-tag{font-size:.61rem;color:var(--mv-dim);letter-spacing:.17em;text-transform:uppercase;
  margin-top:4px;font-weight:700}

/* ── Metric card ──────────────────────────────────────────────────────────── */
.metric-card{background:linear-gradient(158deg,var(--mv-surface),var(--mv-surface-2));
  border:1px solid var(--mv-line);border-radius:var(--mv-r);padding:12px 14px;
  min-height:86px;display:flex;flex-direction:column;justify-content:center;gap:5px;
  transition:border-color .15s,transform .15s}
.metric-card:hover{border-color:var(--mv-blue);transform:translateY(-1px)}
.metric-label{font-size:.665rem;color:var(--mv-muted);text-transform:uppercase;
  letter-spacing:.07em;font-weight:700;line-height:1.3}
.metric-value{font-size:1.55rem;font-weight:800;line-height:1.08;color:var(--mv-text);
  word-break:break-word;font-variant-numeric:tabular-nums}
.metric-value.is-long{font-size:1rem;line-height:1.25}
.metric-sub{font-size:.68rem;color:var(--mv-dim);line-height:1.3}

/* ── Fixture row ──────────────────────────────────────────────────────────── */
.mv-fx{display:grid;grid-template-columns:66px 1fr 96px 1fr;align-items:center;gap:14px;
  padding:15px 16px;border-radius:12px;border:1px solid var(--mv-line-soft);
  background:var(--mv-surface);transition:border-color .13s,background .13s}
.mv-fx:hover{border-color:var(--mv-line);background:var(--mv-surface-2)}
.mv-fx.is-sel{border-color:var(--mv-blue);background:var(--mv-blue-ghost)}
.mv-fx.no-date{grid-template-columns:1fr 96px 1fr}
.mv-fx-date{font-size:.68rem;color:var(--mv-dim);text-transform:uppercase;letter-spacing:.05em;
  font-weight:700;line-height:1.2}
.mv-fx-home{text-align:right;font-weight:600;font-size:.97rem;color:var(--mv-text)}
.mv-fx-away{text-align:left;font-weight:600;font-size:.97rem;color:var(--mv-text)}
/* breathing room between rows: Streamlit's column blocks sit flush, so the gap
   has to come from the row itself */
[data-testid="stHorizontalBlock"]:has(.mv-fx){margin-bottom:10px}
/* the matchday's date separator */
.mv-daysep{display:flex;align-items:center;gap:10px;margin:22px 0 10px;
  font-size:.68rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;
  color:var(--mv-muted)}
.mv-daysep::after{content:"";flex:1;height:1px;background:var(--mv-line-soft)}
.mv-fx-score{text-align:center;font-weight:800;font-size:.93rem;padding:4px 0;border-radius:7px;
  background:var(--mv-bg);border:1px solid var(--mv-line);font-variant-numeric:tabular-nums}
.mv-fx-score.is-vs{color:var(--mv-dim);font-weight:700;font-size:.7rem;letter-spacing:.1em;
  background:transparent;border-color:transparent}
.mv-fx-odds{text-align:center;font-size:.76rem;color:var(--mv-muted);line-height:1.35;
  font-variant-numeric:tabular-nums}
.mv-fx-odds b{color:var(--mv-text)}
.mv-fx-note{text-align:center;font-size:.64rem;color:var(--mv-dim);letter-spacing:.03em;margin-top:2px;font-weight:600}

/* ── Bar row (the one progress-bar component) ─────────────────────────────── */
.mv-bar{display:flex;align-items:center;gap:10px;margin:5px 0}
.mv-bar-k{font-weight:600;font-size:.86rem;color:var(--mv-text)}
.mv-bar-track{flex:1;height:9px;border-radius:5px;background:#1b2740;overflow:hidden}
.mv-bar-fill{height:100%;border-radius:5px;transition:width .25s}
.mv-bar-v{font-size:.85rem;font-weight:700;text-align:right;color:var(--mv-text);
  font-variant-numeric:tabular-nums}
.mv-bar-sub{font-size:.74rem;color:var(--mv-dim);margin:-2px 0 4px}

/* ── Stat card (home vs away, optionally vs the prediction) ───────────────── */
.mv-split{background:var(--mv-surface);border:1px solid var(--mv-line);border-radius:10px;
  padding:10px 12px;border-left:3px solid var(--mv-accent,var(--mv-blue));height:100%}
.mv-split-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:7px;
  gap:6px}
.mv-split-l{font-weight:800;font-size:1.05rem;color:var(--mv-blue-soft);
  font-variant-numeric:tabular-nums}
.mv-split-r{font-weight:800;font-size:1.05rem;color:#f87171;font-variant-numeric:tabular-nums}
.mv-split-k{font-size:.64rem;color:var(--mv-accent,var(--mv-muted));text-transform:uppercase;
  letter-spacing:.07em;font-weight:800}
.mv-split-bar{display:flex;height:6px;border-radius:3px;overflow:hidden;background:#1b2740}
/* the predicted pair, shown under the real one when comparing */
.mv-split-pred{display:flex;justify-content:space-between;align-items:baseline;gap:6px;
  margin-top:9px;padding-top:7px;border-top:1px dashed var(--mv-line)}
.mv-split-pv{font-weight:700;font-size:.82rem;color:var(--mv-dim);
  font-variant-numeric:tabular-nums}
.mv-split-pk{font-size:.58rem;color:var(--mv-dim);text-transform:uppercase;letter-spacing:.07em;
  font-weight:700}
.mv-split-bar.is-pred{height:4px;margin-top:5px;opacity:.5}
.mv-split-err{font-size:.6rem;font-weight:700;letter-spacing:.04em;text-align:center;
  margin-top:6px}

/* ── Form badges ──────────────────────────────────────────────────────────── */
.form-badge{display:inline-block;width:26px;height:26px;border-radius:50%;font-weight:700;
  font-size:.78rem;text-align:center;line-height:26px;margin:1px}
.form-W{background:var(--mv-green);color:#04120c}
.form-D{background:#64748b;color:#fff}
.form-L{background:var(--mv-red);color:#fff}

/* ── History list (H2H / recent form) ─────────────────────────────────────── */
.mv-hist{display:flex;justify-content:space-between;gap:10px;padding:5px 2px;
  border-bottom:1px solid var(--mv-line-soft);font-size:.83rem;color:var(--mv-text)}
.mv-hist span:first-child{color:var(--mv-dim);font-size:.78rem;min-width:78px}
.mv-hist-note{font-size:.78rem;color:var(--mv-dim);padding:1px 0}

/* ── Empty / error state ──────────────────────────────────────────────────── */
.mv-notice{background:var(--mv-surface);border:1px solid var(--mv-line);border-radius:16px;
  padding:44px 32px;text-align:center;margin:26px 0}
.mv-notice-icon{font-size:2.6rem;line-height:1;margin-bottom:14px;opacity:.9}
.mv-notice-title{font-size:1.15rem;font-weight:800;color:var(--mv-text);
  letter-spacing:-.01em;margin-bottom:8px}
.mv-notice-body{font-size:.88rem;color:var(--mv-muted);max-width:52ch;margin:0 auto;
  line-height:1.55}
.mv-notice-links{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:20px}
.mv-notice-link{display:inline-block;padding:7px 15px;border-radius:9px;font-size:.83rem;
  font-weight:600;border:1px solid var(--mv-line);background:var(--mv-surface-2);
  color:var(--mv-muted)!important;text-decoration:none!important}
.mv-notice-link:hover{border-color:var(--mv-blue);color:var(--mv-text)!important}

/* ── Narrow screens ───────────────────────────────────────────────────────── */
/* Streamlit stacks its columns on a phone, but the grids inside them do not:
   a fixture row kept four fixed tracks and squeezed the team names to nothing. */
@media (max-width:640px){
  .block-container{padding-left:.85rem;padding-right:.85rem;padding-top:1.4rem}
  h1{font-size:1.45rem!important}
  h3{font-size:.98rem!important;margin:1.4rem 0 .6rem!important}
  .mv-fx,.mv-fx.no-date{grid-template-columns:1fr 74px 1fr;gap:8px;padding:12px 10px}
  .mv-fx-date{display:none}
  .mv-fx-home,.mv-fx-away{font-size:.83rem}
  .mv-fx-score{font-size:.82rem;padding:3px 0}
  .mv-fx-odds{font-size:.66rem;line-height:1.3}
  .mv-fx-note{font-size:.58rem}
  .metric-card{min-height:70px;padding:10px 11px}
  .metric-value{font-size:1.2rem}
  .metric-value.is-long{font-size:.85rem}
  .mv-split-l,.mv-split-r{font-size:.95rem}
  .mv-notice{padding:32px 18px}
}

/* ── Controls ─────────────────────────────────────────────────────────────── */
.stButton>button{border-radius:9px;border:1px solid var(--mv-line);font-weight:600;
  font-size:.83rem;padding:.34rem .8rem;background:var(--mv-surface);color:var(--mv-muted);
  transition:all .14s;line-height:1.25}
.stButton>button:hover{border-color:var(--mv-blue);color:var(--mv-text);
  background:var(--mv-surface-2)}
.stButton>button[data-testid$="-primary"]{background:var(--mv-blue);border-color:var(--mv-blue);
  color:#fff}
.stButton>button[data-testid$="-primary"]:hover{background:var(--mv-blue-soft);color:#04121f}
.stDownloadButton>button{border-radius:9px;font-weight:600;font-size:.83rem}
[data-testid="stDataFrame"]{border:1px solid var(--mv-line);border-radius:10px;overflow:hidden}
.stTabs [data-baseweb="tab-list"]{gap:3px;border-bottom:1px solid var(--mv-line)}
.stTabs [data-baseweb="tab"]{padding:7px 15px;font-weight:600;font-size:.86rem}
[data-testid="stExpander"] details{border:1px solid var(--mv-line);border-radius:10px;
  background:var(--mv-surface)}
[data-testid="stMetric"]{background:var(--mv-surface);border:1px solid var(--mv-line);
  border-radius:var(--mv-r);padding:11px 14px}
/* dropdowns render over the page on a translucent sheet, so the fixture list
   showed through the options; give the popover a floor of its own */
div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] ul{background:var(--mv-surface-2)!important;
  border:1px solid var(--mv-line);border-radius:10px;
  box-shadow:0 10px 30px rgba(0,0,0,.45)}
div[data-baseweb="popover"] li[role="option"]{font-size:.87rem}
div[data-baseweb="popover"] li[aria-selected="true"]{background:var(--mv-blue-ghost)!important}
/* the view switcher inside a match: a row of segmented options, not loose radios */
[data-testid="stMain"] [role="radiogroup"]{gap:6px!important}
[data-testid="stMain"] [role="radiogroup"] label[data-baseweb="radio"]{
  padding:6px 13px;border-radius:9px;border:1px solid var(--mv-line);
  background:var(--mv-surface);cursor:pointer;transition:all .13s}
[data-testid="stMain"] [role="radiogroup"] label[data-baseweb="radio"]>div:first-child{
  display:none!important}
[data-testid="stMain"] [role="radiogroup"] label p{font-size:.83rem!important;
  font-weight:600;color:var(--mv-muted);margin:0!important}
[data-testid="stMain"] [role="radiogroup"] label:has(input:checked){
  background:var(--mv-blue-ghost);border-color:var(--mv-blue)}
[data-testid="stMain"] [role="radiogroup"] label input:checked ~ div p{
  color:var(--mv-text)!important}
</style>
"""


def inject() -> None:
    """Install the stylesheet. Cheap enough to call on every rerun."""
    st.markdown(CSS, unsafe_allow_html=True)


# ── Components ─────────────────────────────────────────────────────────────────
def brand_header() -> str:
    return ('<div class="mv-brand">'
            '<div class="mv-logo">M</div>'
            '<div><div class="mv-name">Mundia<span class="mv-accent">lytics</span></div>'
            '<div class="mv-tag">Precision Football Models</div></div></div>')


def metric_card(label: str, value: str, color: str = "", sub: str = "") -> str:
    """Label + big number. Long values (date ranges, scorelines) shrink instead
    of overflowing, which the old fixed 92px height could not do."""
    style = f"color:{color}" if color else ""
    long_cls = " is-long" if len(str(value)) > 12 else ""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value{long_cls}" style="{style}">{value}</div>'
            f'{sub_html}</div>')


def bar_row(label: str, pct: float, value_txt: str | None = None,
            color: str | None = None, label_w: int = 0, value_w: int = 46) -> str:
    """The single progress-bar primitive: label, filled track, right-aligned value.

    `pct` is 0-1 and drives the fill width; `value_txt` defaults to the
    percentage. Used for scorelines, over/under ladders and scorer shortlists —
    all three used to carry their own hand-rolled markup and drift apart.
    """
    color = color or C["blue"]
    txt = value_txt if value_txt is not None else f"{pct:.0%}"
    w = max(0.0, min(1.0, float(pct))) * 100
    kw = f"width:{label_w}px;" if label_w else "flex:0 0 auto;"
    return (f'<div class="mv-bar">'
            f'<span class="mv-bar-k" style="{kw}">{label}</span>'
            f'<div class="mv-bar-track"><div class="mv-bar-fill" '
            f'style="width:{w:.1f}%;background:{color}"></div></div>'
            f'<span class="mv-bar-v" style="width:{value_w}px">{txt}</span></div>')


# Each measured stat carries its own accent, so a row of cards reads as a row of
# different things rather than one blue block. Home stays blue and away red
# inside every card — the accent labels the stat, it never labels a side.
STAT_ACCENT = {
    "Goles":     C["green"],
    "Disparos":  C["blue"],
    "A puerta":  C["blue_soft"],
    "Córners":   C["violet"],
    "Faltas":    C["amber"],
    "Amarillas": "#eab308",
    "Rojas":     C["red"],
}


def split_stat(label: str, home_val: float, away_val: float, dec: int = 1,
               pred: tuple[float, float] | None = None, accent: str | None = None) -> str:
    """Home-vs-away number pair with a proportional split bar underneath.

    Two bare numbers with a 'vs' between them made the reader do the comparison;
    the bar does it for them. Pass `pred` to stack the model's numbers under the
    real ones in the same card — same layout, dimmer, with the total error — so
    "what happened" and "what we said" can be read without switching views.
    """
    accent = accent or STAT_ACCENT.get(label, C["blue"])
    tot = float(home_val) + float(away_val)
    share = 0.5 if tot <= 0 else float(home_val) / tot
    html = (f'<div class="mv-split" style="--mv-accent:{accent}">'
            f'<div class="mv-split-top">'
            f'<span class="mv-split-l">{home_val:.{dec}f}</span>'
            f'<span class="mv-split-k">{label}</span>'
            f'<span class="mv-split-r">{away_val:.{dec}f}</span></div>'
            f'<div class="mv-split-bar">'
            f'<div style="width:{share*100:.1f}%;background:{HOME}"></div>'
            f'<div style="flex:1;background:{AWAY}"></div></div>')
    if pred is not None:
        ph, pa = float(pred[0]), float(pred[1])
        ptot = ph + pa
        pshare = 0.5 if ptot <= 0 else ph / ptot
        err = abs(ph - float(home_val)) + abs(pa - float(away_val))
        # graded on the total miss across both sides: under one event is a hit,
        # over three is a miss, and the middle is honestly amber
        tone = C["green"] if err <= 1.0 else (C["amber"] if err <= 3.0 else C["red"])
        html += (f'<div class="mv-split-pred">'
                 f'<span class="mv-split-pv">{ph:.{dec}f}</span>'
                 f'<span class="mv-split-pk">previsto</span>'
                 f'<span class="mv-split-pv">{pa:.{dec}f}</span></div>'
                 f'<div class="mv-split-bar is-pred">'
                 f'<div style="width:{pshare*100:.1f}%;background:{HOME}"></div>'
                 f'<div style="flex:1;background:{AWAY}"></div></div>'
                 f'<div class="mv-split-err" style="color:{tone}">error {err:.1f}</div>')
    return html + "</div>"


def num(value, dec: int = 0) -> str:
    """`15.371`, `10.000`, `1.234,5` — Spanish digit grouping.

    Python's `:,` writes 15,371, which a Spanish reader parses as fifteen point
    three. The app was mixing both formats on the same screen.
    """
    try:
        whole, _, frac = f"{float(value):,.{dec}f}".partition(".")
    except (TypeError, ValueError):
        return str(value)
    whole = whole.replace(",", ".")
    return f"{whole},{frac}" if frac else whole


def fmt_date(value) -> str:
    """`28 AGO` — short, Spanish, locale-independent (Windows locales are not
    reliable, so the month table is inlined)."""
    try:
        import pandas as pd
        ts = pd.to_datetime(value)
        return f"{ts.day} {_MONTHS_ES[ts.month - 1]}".upper()
    except Exception:
        return ""


def fixture_row(home: str, away: str, score: str | None = None, date=None,
                selected: bool = False, note: str = "",
                probs: tuple[float, float, float] | None = None) -> str:
    """One match as a grid row: date · home · score-or-probabilities · away.

    `probs` (1, X, 2) takes the score column for a match not yet played, so a
    pending fixture says something instead of showing an empty 'vs'. `note` adds
    a second line under it (an over/under, the expected goals) — inside the cell,
    so it stays under the number it belongs to.
    """
    cls = "mv-fx is-sel" if selected else "mv-fx"
    tail = f'<div class="mv-fx-note">{note}</div>' if note else ""
    if score:
        mid = f'<div><div class="mv-fx-score">{score}</div>{tail}</div>'
    elif probs:
        ph, pdr, pa = probs
        mid = (f'<div><div class="mv-fx-odds"><b>{ph:.0%}</b> · {pdr:.0%} · '
               f'<b>{pa:.0%}</b></div>{tail}</div>')
    else:
        mid = f'<div><div class="mv-fx-score is-vs">VS</div>{tail}</div>'
    # with a day separator above the block the per-row date is noise, so an
    # omitted date collapses its column instead of leaving a gutter
    if date is None:
        return (f'<div class="{cls} no-date">'
                f'<div class="mv-fx-home">{home}</div>{mid}'
                f'<div class="mv-fx-away">{away}</div></div>')
    return (f'<div class="{cls}">'
            f'<div class="mv-fx-date">{fmt_date(date)}</div>'
            f'<div class="mv-fx-home">{home}</div>{mid}'
            f'<div class="mv-fx-away">{away}</div></div>')


def hist_row(left: str, middle: str, right: str, right_color: str = "") -> str:
    style = f' style="color:{right_color};font-weight:700"' if right_color else ""
    return (f'<div class="mv-hist"><span>{left}</span><span>{middle}</span>'
            f'<span{style}>{right}</span></div>')


# ── Plotly theme ───────────────────────────────────────────────────────────────
def style_fig(fig, height: int | None = None):
    """Paint a figure from the tokens: transparent canvas, readable tick labels,
    hairline grid, themed hover box. Merges into whatever the caller already set."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["muted"], size=12),
        title_font=dict(color=C["text"], size=14),
        hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["line"],
                        font=dict(color=C["text"], size=12)),
        legend=dict(font=dict(color=C["muted"], size=11)),
        # automargin: several figures were written with margin l=0/b=0 and relied
        # on Streamlit's own plotly theme to find room. Left to plotly they clipped
        # their own tick labels (the per-league RPS bars lost every league name).
        xaxis=dict(gridcolor=C["line_soft"], zerolinecolor=C["line"], automargin=True,
                   linecolor=C["line"], tickfont=dict(color=C["dim"], size=11),
                   title_font=dict(color=C["muted"], size=11)),
        yaxis=dict(gridcolor=C["line_soft"], zerolinecolor=C["line"], automargin=True,
                   linecolor=C["line"], tickfont=dict(color=C["muted"], size=11),
                   title_font=dict(color=C["muted"], size=11)),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


_DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def fmt_date_long(value) -> str:
    """`5 sep 2026` — for captions, where a raw 2026-09-05 reads like a database."""
    try:
        import pandas as pd
        ts = pd.to_datetime(value)
        return f"{ts.day} {_MONTHS_ES[ts.month - 1]} {ts.year}"
    except Exception:
        return str(value)


def day_separator(value) -> str:
    """`SÁBADO 5 SEP` — heads the block of fixtures played that day."""
    try:
        import pandas as pd
        ts = pd.to_datetime(value)
        label = f"{_DAYS_ES[ts.weekday()]} {ts.day} {_MONTHS_ES[ts.month - 1]}"
    except Exception:
        label = str(value)
    return f'<div class="mv-daysep">{label}</div>'


def notice(icon: str, title: str, body: str = "", links: list[tuple[str, str]] | None = None) -> str:
    """A full-width empty/error state: icon, headline, explanation, ways out.

    Used where the app has nothing to show — an unknown section, a failed load —
    so a dead end still tells you where you are and what to do next.
    """
    link_html = ""
    if links:
        items = "".join(f'<a class="mv-notice-link" href="{href}">{label}</a>'
                        for label, href in links)
        link_html = f'<div class="mv-notice-links">{items}</div>'
    body_html = f'<div class="mv-notice-body">{body}</div>' if body else ""
    return (f'<div class="mv-notice"><div class="mv-notice-icon">{icon}</div>'
            f'<div class="mv-notice-title">{title}</div>{body_html}{link_html}</div>')


def chart(fig, height: int | None = None, **kwargs) -> None:
    """style_fig + render. `theme=None` so the tokens above win over Streamlit's
    own plotly theme instead of fighting it."""
    st.plotly_chart(style_fig(fig, height), use_container_width=True, theme=None, **kwargs)
