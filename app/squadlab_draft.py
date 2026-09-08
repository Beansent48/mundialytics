"""SquadLab Champions Draft — the game mode.

You draft eleven players out of packs, one slot at a time, and your club takes
the place of the weakest side in the real Champions League field. Nothing is
chosen by league any more: the pool is every player of every club we simulate,
plus their prime seasons, plus the icons.

The three card faces are deliberately different objects rather than three
colours of the same one. A prime carries a year because that is the whole claim
it makes — this is Suárez in 2015/16, not Suárez. An icon carries no year
because the claim is the opposite one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.squadlab import cards as C  # noqa: E402
from mundialytics.statistical_core.squadlab.champions import (  # noqa: E402
    SQUAD_TEAM_NAME, ChampionsRun, ROUND_LABELS, load_field, squad_elo_scale,
)

CANDIDATES_PER_SLOT = 5
REROLLS = 3

# Pitch coordinates (% of width/height) per formation, attacking toward y=0.
FORMATION_COORDS = {
    "4-3-3": {"Goalkeeper": [(50, 92)],
              "Defender": [(14, 72), (38, 77), (62, 77), (86, 72)],
              "Midfielder": [(25, 50), (50, 45), (75, 50)],
              "Forward": [(20, 17), (50, 11), (80, 17)]},
    "4-4-2": {"Goalkeeper": [(50, 92)],
              "Defender": [(14, 72), (38, 77), (62, 77), (86, 72)],
              "Midfielder": [(14, 48), (38, 44), (62, 44), (86, 48)],
              "Forward": [(35, 15), (65, 15)]},
    "3-4-3": {"Goalkeeper": [(50, 92)],
              "Defender": [(26, 76), (50, 80), (74, 76)],
              "Midfielder": [(13, 50), (38, 46), (62, 46), (87, 50)],
              "Forward": [(20, 16), (50, 11), (80, 16)]},
    "3-5-2": {"Goalkeeper": [(50, 92)],
              "Defender": [(26, 76), (50, 80), (74, 76)],
              "Midfielder": [(9, 52), (30, 44), (50, 39), (70, 44), (91, 52)],
              "Forward": [(35, 15), (65, 15)]},
    "5-2-3": {"Goalkeeper": [(50, 92)],
              "Defender": [(8, 68), (28, 78), (50, 82), (72, 78), (92, 68)],
              "Midfielder": [(36, 48), (64, 48)],
              "Forward": [(20, 17), (50, 11), (80, 17)]},
    "5-3-2": {"Goalkeeper": [(50, 92)],
              "Defender": [(8, 68), (28, 78), (50, 82), (72, 78), (92, 68)],
              "Midfielder": [(25, 50), (50, 45), (75, 50)],
              "Forward": [(35, 15), (65, 15)]},
}
FORMATIONS = list(FORMATION_COORDS)

POS_ABBR = {"Goalkeeper": "POR", "Defender": "DEF", "Midfielder": "MED", "Forward": "DEL"}
POS_ES = {"Goalkeeper": "Portero", "Defender": "Defensa",
          "Midfielder": "Medio", "Forward": "Delantero"}

# Card skins. Tier colours for ordinary cards; primes and icons get their own,
# so rarity reads at a glance from across the screen.
#
# The icon is the one card that is LIGHT. Its first version was gold-on-dark and
# sat two shades from the ordinary gold tier — the rarest card in the pool
# reading as the third-rarest. Every other card in a dark app is dark, so
# inverting it is the only treatment nothing else can accidentally approach; it
# is also what a pearl/platinum icon looks like, which is the association wanted.
SKINS = {
    "bronce": {"bg": ("#3a2a1c", "#6b4a2c"), "line": "#a97142", "ink": "#f7e6d2", "sub": "#d9b895"},
    "plata": {"bg": ("#2b3138", "#5b6672"), "line": "#9fb0c0", "ink": "#f4f8fb", "sub": "#cfd9e2"},
    "oro": {"bg": ("#4a3a10", "#a8842a"), "line": "#e0bd5a", "ink": "#fff8e2", "sub": "#f0dda6"},
    "oro_raro": {"bg": ("#2a1f05", "#c9a227"), "line": "#ffd75e", "ink": "#fffaf0", "sub": "#ffe9a8"},
    "elite": {"bg": ("#141425", "#3c2f6b"), "line": "#b39dff", "ink": "#ffffff", "sub": "#dcd2ff"},
    "prime": {"bg": ("#07231f", "#0f6b57"), "line": "#3ee0b0", "ink": "#eafff8", "sub": "#9ff0da"},
    "icono": {"bg": ("#c9cddb", "#f7f8fc"), "line": "#5b6478", "ink": "#12151f",
              "sub": "#4a5265"},
}


def skin_for(card: C.Card) -> dict:
    if card.kind == "icono":
        return SKINS["icono"]
    if card.kind == "prime":
        return SKINS["prime"]
    return SKINS.get(card.tier, SKINS["bronce"])


# ── card art ───────────────────────────────────────────────────────────────────
def card_svg(card: C.Card, width: int = 158, reveal: bool = False) -> str:
    """One card face. `reveal` plays the entrance animation for a special pull."""
    t = skin_for(card)
    uid = abs(hash((card.card_id, reveal))) % 10**7
    w, h = 150, 214
    is_gk = card.position == "Goalkeeper"
    dfn = card.gk if is_gk else card.defense
    special = card.kind != "actual"

    anim = ""
    if reveal and special:
        # Two different arrivals on purpose: a prime slides up and settles, an
        # icon lands. The distinction is the reward.
        name = f"pull{uid}"
        if card.kind == "icono":
            keys = ("@keyframes {n}{{0%{{opacity:0;transform:scale(.4) rotate(-14deg)}}"
                    "55%{{opacity:1;transform:scale(1.16) rotate(4deg)}}"
                    "75%{{transform:scale(.97) rotate(-2deg)}}"
                    "100%{{transform:scale(1) rotate(0)}}}}").format(n=name)
            dur = "1.05s"
        else:
            keys = ("@keyframes {n}{{0%{{opacity:0;transform:translateY(26px) scale(.86)}}"
                    "65%{{opacity:1;transform:translateY(-6px) scale(1.05)}}"
                    "100%{{transform:translateY(0) scale(1)}}}}").format(n=name)
            dur = "0.7s"
        glow = "#e9edff" if card.kind == "icono" else "#3ee0b0"
        anim = (f"<style>{keys}"
                f"@keyframes halo{uid}{{0%,100%{{filter:drop-shadow(0 0 2px {glow}80)}}"
                f"50%{{filter:drop-shadow(0 0 14px {glow})}}}}"
                f".pull{uid}{{animation:{name} {dur} cubic-bezier(.2,.9,.25,1) both,"
                f"halo{uid} 2.4s ease-in-out 0.5s infinite}}</style>")

    cls = f' class="pull{uid}"' if anim else ""
    s = [anim, f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"{cls} '
                f'style="width:100%;max-width:{width}px;display:block;margin:0 auto">',
         '<defs>',
         f'<linearGradient id="g{uid}" x1="0" y1="0" x2=".35" y2="1">'
         f'<stop offset="0%" stop-color="{t["bg"][1]}"/>'
         f'<stop offset="100%" stop-color="{t["bg"][0]}"/></linearGradient>']
    if special:
        # a wash of the accent over the body. On the light icon skin the accent
        # is dark, so it is dialled right down or the card turns grey.
        op = ".14" if card.kind == "icono" else ".38"
        s.append(f'<radialGradient id="r{uid}" cx="50%" cy="34%" r="62%">'
                 f'<stop offset="0%" stop-color="{t["line"]}" stop-opacity="{op}"/>'
                 f'<stop offset="100%" stop-color="{t["line"]}" stop-opacity="0"/>'
                 f'</radialGradient>')
    s.append('</defs>')
    s.append(f'<path d="M8 14 Q8 4 20 4 L130 4 Q142 4 142 14 L142 176 '
             f'Q142 190 128 196 L79 210 Q75 211 71 210 L22 196 Q8 190 8 176 Z" '
             f'fill="url(#g{uid})" stroke="{t["line"]}" stroke-width="{2.6 if special else 1.5}"/>')
    if special:
        s.append(f'<path d="M8 14 Q8 4 20 4 L130 4 Q142 4 142 14 L142 176 '
                 f'Q142 190 128 196 L79 210 Q75 211 71 210 L22 196 Q8 190 8 176 Z" '
                 f'fill="url(#r{uid})"/>')
    # rating + position + role
    s.append(f'<text x="30" y="46" text-anchor="middle" font-size="30" font-weight="800" '
             f'fill="{t["ink"]}">{card.overall:.0f}</text>')
    s.append(f'<text x="30" y="62" text-anchor="middle" font-size="11" font-weight="700" '
             f'fill="{t["sub"]}">{POS_ABBR.get(card.position, "—")}</text>')
    role = (card.role or "").upper()[:9]
    if role:
        s.append(f'<text x="30" y="77" text-anchor="middle" font-size="7.5" font-weight="600" '
                 f'fill="{t["sub"]}" opacity=".85">{role}</text>')
    s.append(f'<line x1="52" y1="26" x2="52" y2="78" stroke="{t["line"]}" stroke-width="1" opacity=".5"/>')
    # emblem: the year for a prime, a star for an icon, a monogram otherwise
    disc = "#ffffff66" if card.kind == "icono" else "#00000038"
    s.append(f'<circle cx="99" cy="52" r="26" fill="{disc}" stroke="{t["line"]}" '
             f'stroke-width="1" opacity=".85"/>')
    if card.kind == "prime" and card.season_label:
        s.append(f'<text x="99" y="57" text-anchor="middle" font-size="15" font-weight="800" '
                 f'fill="{t["ink"]}">{card.season_label}</text>')
    elif card.kind == "icono":
        s.append(f'<text x="99" y="62" text-anchor="middle" font-size="26" '
                 f'fill="{t["ink"]}">★</text>')
    else:
        s.append(f'<text x="99" y="62" text-anchor="middle" font-size="24" font-weight="800" '
                 f'fill="{t["ink"]}" opacity=".92">{card.display[:1].upper()}</text>')
    # long names shrink rather than truncate: "CRISTIANO RONAL" is a worse card
    # than a slightly smaller "CRISTIANO RONALDO"
    name = _esc(card.display).upper()[:20]
    fs = 13 if len(name) <= 13 else (11 if len(name) <= 16 else 9.5)
    s.append(f'<text x="75" y="112" text-anchor="middle" font-size="{fs}" font-weight="800" '
             f'fill="{t["ink"]}" letter-spacing=".4">{name}</text>')
    s.append(f'<line x1="28" y1="122" x2="122" y2="122" stroke="{t["line"]}" stroke-width="1" opacity=".55"/>')
    if is_gk:
        # One number, centred. A keeper's attacking and creative axes do not
        # exist — printing a placeholder next to a real rating claimed they did.
        s.append(f'<text x="75" y="146" text-anchor="middle" font-size="19" font-weight="800" '
                 f'fill="{t["ink"]}">{dfn:.0f}</text>')
        s.append(f'<text x="75" y="159" text-anchor="middle" font-size="8" font-weight="700" '
                 f'fill="{t["sub"]}">PARADAS</text>')
    else:
        s.append(f'<text x="52" y="146" text-anchor="middle" font-size="18" font-weight="800" '
                 f'fill="{t["ink"]}">{card.attack:.0f}</text>')
        s.append(f'<text x="52" y="159" text-anchor="middle" font-size="8" font-weight="700" '
                 f'fill="{t["sub"]}">ATAQUE</text>')
        s.append(f'<text x="98" y="146" text-anchor="middle" font-size="18" font-weight="800" '
                 f'fill="{t["ink"]}">{dfn:.0f}</text>')
        s.append(f'<text x="98" y="159" text-anchor="middle" font-size="8" font-weight="700" '
                 f'fill="{t["sub"]}">DEFENSA</text>')
    s.append(f'<text x="75" y="180" text-anchor="middle" font-size="8" font-weight="600" '
             f'fill="{t["sub"]}" opacity=".92">{_esc(card.subtitle)[:22]}</text>')
    if special:
        band = "ICONO" if card.kind == "icono" else "PRIME"
        s.append(f'<text x="75" y="193" text-anchor="middle" font-size="8" font-weight="800" '
                 f'fill="{t["line"]}" letter-spacing="2.4">{band}</text>')
    s.append('</svg>')
    return "".join(s)


def _esc(x: object) -> str:
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ── pitch ──────────────────────────────────────────────────────────────────────
def pitch_svg(coords: dict, picks: dict, active: str | None) -> str:
    w, h = 330, 440
    s = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
         f'style="width:100%;max-width:340px">',
         f'<rect width="{w}" height="{h}" rx="10" fill="#123f22"/>',
         f'<rect x="6" y="6" width="{w-12}" height="{h-12}" rx="6" fill="none" '
         f'stroke="#ffffff55" stroke-width="1.5"/>',
         f'<line x1="6" y1="{h//2}" x2="{w-6}" y2="{h//2}" stroke="#ffffff44" stroke-width="1.2"/>',
         f'<circle cx="{w//2}" cy="{h//2}" r="42" fill="none" stroke="#ffffff44" stroke-width="1.2"/>',
         f'<rect x="{w//2-60}" y="6" width="120" height="52" fill="none" stroke="#ffffff44" stroke-width="1.2"/>',
         f'<rect x="{w//2-60}" y="{h-58}" width="120" height="52" fill="none" stroke="#ffffff44" stroke-width="1.2"/>']
    for slot, (px, py) in coords.items():
        x, y = px / 100 * w, py / 100 * h
        card = picks.get(slot)
        if card is not None:
            t = skin_for(card)
            s.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="15" fill="{t["bg"][1]}" '
                     f'stroke="{t["line"]}" stroke-width="2"/>')
            s.append(f'<text x="{x:.0f}" y="{y+4:.0f}" text-anchor="middle" font-size="11" '
                     f'font-weight="800" fill="{t["ink"]}">{card.overall:.0f}</text>')
            s.append(f'<text x="{x:.0f}" y="{y+28:.0f}" text-anchor="middle" font-size="8.5" '
                     f'font-weight="600" fill="#eaf6ee">{_esc(card.display)[:12]}</text>')
        else:
            on = slot == active
            dash = "" if on else ' stroke-dasharray="4,3"'
            s.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="14" '
                     f'fill="{"#ffffff26" if on else "#00000033"}" '
                     f'stroke="{"#7dd3fc" if on else "#ffffff66"}" '
                     f'stroke-width="{2.5 if on else 1.4}"{dash}/>')
            s.append(f'<text x="{x:.0f}" y="{y+4:.0f}" text-anchor="middle" font-size="13" '
                     f'fill="#eaf6ee" opacity=".8">+</text>')
    s.append('</svg>')
    return "".join(s)


def slot_list(formation: str) -> list[tuple[str, str]]:
    """[(slot key, position)] in the order the draft fills them."""
    out = []
    for pos, pts in FORMATION_COORDS[formation].items():
        for i in range(len(pts)):
            out.append((f"{pos}#{i}", pos))
    return out


def slot_coords(formation: str) -> dict:
    return {f"{pos}#{i}": pt
            for pos, pts in FORMATION_COORDS[formation].items()
            for i, pt in enumerate(pts)}


# ── resources ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cards() -> pd.DataFrame:
    return C.load_cards()


@st.cache_resource(show_spinner="Preparando la Champions...")
def _field_and_scale():
    field = load_field()
    elo = pd.read_csv(ROOT / "data/processed/clubelo_local.csv").dropna(subset=["club", "elo"])
    scale = squad_elo_scale(None, C.load_cards(), dict(zip(elo["club"], elo["elo"])))
    return field, scale


# ── state ──────────────────────────────────────────────────────────────────────
def _reset(formation: str, seed: int) -> None:
    st.session_state.cd_formation = formation
    st.session_state.cd_seed = int(seed)
    st.session_state.cd_pool = C.DraftPool(_cards(), np.random.default_rng(int(seed)))
    st.session_state.cd_picks = {}
    st.session_state.cd_cands = {}
    st.session_state.cd_rerolls = REROLLS
    st.session_state.cd_last_pull = None
    for k in ("cd_result", "cd_probs"):
        st.session_state.pop(k, None)


def _candidates(slot: str, pos: str) -> list[C.Card]:
    cands = st.session_state.cd_cands.get(slot)
    if cands is None:
        cands = st.session_state.cd_pool.deal(pos, CANDIDATES_PER_SLOT)
        st.session_state.cd_cands[slot] = cands
    return cands


# ── the page ───────────────────────────────────────────────────────────────────
def render() -> None:
    cards = _cards()
    if cards.empty:
        st.warning("Falta el catálogo de cartas. Genéralo con "
                   "`python scripts/build_squadlab_cards.py`.")
        return
    field, scale = _field_and_scale()

    st.markdown("#### 🎮 Champions Draft")
    st.caption(
        f"Ficha once jugadores de sobres y juega la Champions 2026/27 real: tu club "
        f"ocupa la plaza del **{min(field['elo'], key=field['elo'].get)}**, el más débil "
        f"del cuadro, y hereda sus ocho rivales de la fase liga. Cualquier liga, "
        f"cualquier época — cartas ACTUAL, PRIME (una gran temporada, con su año) e "
        f"ICONO (retirados generacionales).")

    c1, c2, c3 = st.columns([1.4, 1, 1])
    formation = c1.selectbox("Formación", FORMATIONS,
                             index=FORMATIONS.index(st.session_state.get("cd_formation", "4-3-3")),
                             key="cd_formation_sel")
    seed = c2.number_input("Semilla del sobre", 1, 999_999,
                           int(st.session_state.get("cd_seed", 7)), key="cd_seed_in")
    c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c3.button("🎲 Nuevo draft", use_container_width=True, key="cd_new"):
        _reset(formation, seed)
        st.rerun()

    if "cd_pool" not in st.session_state or st.session_state.get("cd_formation") != formation:
        _reset(formation, seed)

    _draw_probabilities()

    slots = slot_list(formation)
    coords = slot_coords(formation)
    picks: dict[str, C.Card] = st.session_state.cd_picks
    active = next((s for s, _ in slots if s not in picks), None)

    left, right = st.columns([1, 1.35])
    with left:
        st.markdown(pitch_svg(coords, picks, active), unsafe_allow_html=True)
        n = len(picks)
        st.markdown(f"<div style='text-align:center;font-weight:700;margin-top:6px'>"
                    f"{n}/11 fichados</div>", unsafe_allow_html=True)
        if picks:
            chosen = list(picks.values())
            _squad_meters(chosen, scale)

    with right:
        if active is None:
            st.success("Once completo. Abajo tienes el botón para jugar la Champions.")
        else:
            pos = dict(slots)[active]
            idx = [s for s, _ in slots].index(active) + 1
            st.markdown(f"**Elige {POS_ES[pos].lower()}** · slot {idx} de 11")
            cands = _candidates(active, pos)
            just = st.session_state.get("cd_last_pull")
            cols = st.columns(len(cands))
            for col, card in zip(cols, cands):
                with col:
                    reveal = card.is_special and just != active
                    st.markdown(card_svg(card, reveal=reveal), unsafe_allow_html=True)
                    if st.button("Fichar", key=f"cd_pick_{active}_{card.card_id}",
                                 use_container_width=True):
                        st.session_state.cd_pool.take(card)
                        picks[active] = card
                        st.session_state.cd_last_pull = active
                        st.rerun()
            left_rr = st.session_state.cd_rerolls
            b1, b2 = st.columns([1, 2])
            if b1.button(f"🔄 Otro sobre ({left_rr})", key=f"cd_rr_{active}",
                         disabled=left_rr <= 0, use_container_width=True):
                st.session_state.cd_rerolls -= 1
                st.session_state.cd_cands.pop(active, None)
                st.session_state.cd_last_pull = None
                st.rerun()
            b2.caption("Los re-rolls son limitados: rechazar un sobre cuesta.")

    if active is None:
        st.markdown("---")
        _play_section(field, scale)


def _draw_probabilities() -> None:
    with st.expander("🎰 Probabilidades del sobre", expanded=False):
        rows = [{"Tipo": "ACTUAL — jugador de esta temporada", "Prob.": C.KIND_P["actual"]},
                {"Tipo": "PRIME — gran temporada, con su año", "Prob.": C.KIND_P["prime"]},
                {"Tipo": "ICONO — retirado generacional", "Prob.": C.KIND_P["icono"]}]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Prob.": st.column_config.NumberColumn(format="%.1f%%")})
        t = C.TIER_P["actual"]
        st.caption("Dentro de una carta ACTUAL: bronce {:.0%} · plata {:.0%} · oro {:.0%} · "
                   "oro raro {:.0%} · élite {:.0%}. Con 5 candidatos por slot y 11 slots son "
                   "55 tiradas: lo normal es ver un icono por draft, y a veces ninguno."
                   .format(t["bronce"], t["plata"], t["oro"], t["oro_raro"], t["elite"]))


def _squad_meters(chosen: list[C.Card], scale) -> None:
    rating = C.squad_rating(chosen)
    chem = C.squad_chemistry(chosen)
    luck = C.pack_luck(chosen)
    elo = scale.elo_for([c.overall for c in chosen])
    tiles = (_meter("Media", f"{rating:.1f}")
             + _meter("Elo estimado", f"{elo:.0f}")
             + _meter("Química", f"{chem['pct']:.0f}%")
             + _meter("Suerte", f"{luck['ratio']:.2f}×"))
    st.markdown(
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px'>"
        + tiles + "</div>", unsafe_allow_html=True)
    if luck["primes"] or luck["iconos"]:
        bits = []
        if luck["iconos"]:
            bits.append(f"{luck['iconos']} icono" + ("s" if luck["iconos"] > 1 else ""))
        if luck["primes"]:
            bits.append(f"{luck['primes']} prime" + ("s" if luck["primes"] > 1 else ""))
        st.caption("En el once: " + " · ".join(bits))


def _meter(label: str, value: str) -> str:
    return (f"<div style='background:var(--secondary-background-color);border-radius:8px;"
            f"padding:6px 10px'><div style='font-size:10px;color:#9ca3af'>{label}</div>"
            f"<div style='font-size:17px;font-weight:800'>{value}</div></div>")


# ── playing it ─────────────────────────────────────────────────────────────────
def _play_section(field: dict, scale) -> None:
    chosen = list(st.session_state.cd_picks.values())
    elo = scale.elo_for([c.overall for c in chosen])
    rank = sum(1 for v in field["elo"].values() if v > elo) + 1
    st.markdown(f"**Tu equipo** · media {C.squad_rating(chosen):.1f} · Elo estimado "
                f"**{elo:.0f}** — sería el **{rank}º** cuadro más fuerte de los 36.")

    b1, b2 = st.columns([1, 2])
    if b1.button("🏆 Jugar la Champions", type="primary", use_container_width=True,
                 key="cd_play"):
        squad = [c.to_profile() for c in chosen]
        rng = np.random.default_rng(int(st.session_state.cd_seed) * 31 + 5)
        run = ChampionsRun(SQUAD_TEAM_NAME, squad, elo, field, rng=rng)
        with st.spinner("Jugando la fase liga y las eliminatorias..."):
            st.session_state.cd_result = run.play()
        st.rerun()
    b2.caption("Se juegan tus 8 partidos de fase liga con goleadores y asistencias, "
               "y después el playoff, octavos, cuartos, semifinales y final.")

    _squad_strip(chosen)

    res = st.session_state.get("cd_result")
    if res is None:
        return
    _render_result(res)
    st.markdown("---")
    r1, r2 = st.columns([1, 2])
    if r1.button("🔁 Otra campaña con este once", key="cd_replay", use_container_width=True):
        # Same squad, new dice. The point is to see the variance: a 1811-Elo
        # side reaching the final once and going out in the playoff the next
        # time is the tournament being a tournament, not the model wobbling.
        st.session_state.cd_replay = st.session_state.get("cd_replay", 0) + 1
        squad = [c.to_profile() for c in chosen]
        rng = np.random.default_rng(int(st.session_state.cd_seed) * 31
                                    + 977 * st.session_state.cd_replay)
        with st.spinner("Otra vez desde la fase liga..."):
            st.session_state.cd_result = ChampionsRun(
                SQUAD_TEAM_NAME, squad, elo, field, rng=rng).play()
        st.session_state.pop("cd_live", None)
        st.rerun()
    r2.caption("Mismo equipo, otro sorteo de resultados — para ver cuánto pesa la suerte.")


def _squad_strip(chosen: list[C.Card]) -> None:
    """The finished eleven, laid out as cards. Worth its own row: after eleven
    picks the pitch dots no longer say what you actually assembled."""
    with st.expander("🃏 Tu once, carta a carta", expanded=False):
        order = {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
        squad = sorted(chosen, key=lambda c: (order.get(c.position, 9), -c.overall))
        for row in range(0, len(squad), 6):
            cols = st.columns(6)
            for col, card in zip(cols, squad[row:row + 6]):
                with col:
                    st.markdown(card_svg(card, width=120), unsafe_allow_html=True)
        best = max(chosen, key=lambda c: c.overall)
        st.caption(f"Mejor carta: **{best.display}** ({best.overall:.0f}"
                   f"{', ' + best.kind.upper() if best.is_special else ''}).")


def _render_result(res) -> None:
    won = res.champion == SQUAD_TEAM_NAME
    st.markdown(
        f'<div style="text-align:center;padding:16px;margin:10px 0;border-radius:14px;'
        f'background:linear-gradient(140deg,{"#3b2a05,#8a6a12" if won else "#101a2e,#1e2c4a"});'
        f'border:2px solid {"#ffd75e" if won else "#31405f"}">'
        f'<div style="font-size:.75rem;letter-spacing:.2em;color:#cfd8ea">TU CAMPAÑA</div>'
        f'<div style="font-size:1.7rem;font-weight:900;color:{"#ffd75e" if won else "#e2e8f0"}">'
        f'{res.squad_stage}</div>'
        f'<div style="font-size:.8rem;color:#cfd8ea">{res.squad_position}º en la fase liga · '
        f'campeón: {res.champion}</div></div>', unsafe_allow_html=True)

    own = [m for m in res.matches if m.stage == "liga" and SQUAD_TEAM_NAME in (m.home, m.away)]
    st.markdown("##### 📅 Tu fase liga")
    for k, m in enumerate(own):
        _match_row(m)
        if st.button("▶️ Ver en directo", key=f"cd_live_{k}", use_container_width=False):
            st.session_state.cd_live = k
            st.rerun()
        if st.session_state.get("cd_live") == k:
            _live(m)

    ties = [(k, t) for k in ("playoff", "r16", "qf", "sf", "final")
            for t in res.rounds[k] if SQUAD_TEAM_NAME in (t["team_a"], t["team_b"])]
    if ties:
        st.markdown("##### 🗝️ Tus eliminatorias")
        for key, t in ties:
            won_tie = t["winner"] == SQUAD_TEAM_NAME
            st.markdown(
                f'<div style="display:flex;gap:10px;align-items:center;padding:6px 10px;'
                f'background:#151f30;border:1px solid {"#2f6b4a" if won_tie else "#5b2b2b"};'
                f'border-radius:8px;margin:3px 0">'
                f'<div style="width:96px;color:#9aa9bf;font-size:.72rem">{ROUND_LABELS[key]}</div>'
                f'<div style="flex:1;font-size:.86rem">{t["team_a"]} <b>{t["agg"]}</b> {t["team_b"]}'
                f' <span style="color:#f59e0b;font-size:.7rem">{t["note"]}</span></div>'
                f'<div style="font-size:.75rem;color:{"#4ade80" if won_tie else "#f87171"}">'
                f'{"pasas" if won_tie else "fuera"}</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### ⚽ Tus goleadores")
        if len(res.scorers):
            st.dataframe(res.scorers.head(12), hide_index=True, use_container_width=True)
        else:
            st.caption("Ningún gol. Pasa.")
    with c2:
        st.markdown("##### 📊 Fase liga")
        tbl = res.table.copy()
        tbl = tbl[["pos", "team", "played", "pts", "gf", "ga", "gd"]].rename(columns={
            "pos": "#", "team": "Equipo", "played": "PJ", "pts": "Pts",
            "gf": "GF", "ga": "GC", "gd": "DG"})
        st.dataframe(tbl, hide_index=True, use_container_width=True, height=330)


def _live(m) -> None:
    """Replay one of your matches minute by minute.

    Reuses the playback the domestic season already had — nothing about the
    result changes, it is revealed. A Champions match carries its own shots and
    corners now (see champions.ChampionsRun._draw_events), so the stat bars grow
    toward real numbers rather than placeholders.
    """
    from mundialytics.statistical_core.squadlab.season_simulator import MatchResult

    import squadlab_page as page
    is_home = m.home == SQUAD_TEAM_NAME
    mr = MatchResult(
        matchday=m.matchday, home=m.home, away=m.away,
        home_goals=m.home_goals, away_goals=m.away_goals,
        home_events=m.ratings if is_home else None,
        away_events=None if is_home else m.ratings,
        home_goal_events=m.goal_events if is_home else None,
        away_goal_events=None if is_home else m.goal_events,
        home_card_players=m.card_players if is_home else None,
        away_card_players=None if is_home else m.card_players,
        home_shots=m.shots[0], away_shots=m.shots[1],
        home_sot=m.sot[0], away_sot=m.sot[1],
        home_corners=m.corners[0], away_corners=m.corners[1],
        home_yellow_cards=m.yellows[0], away_yellow_cards=m.yellows[1],
        home_xg=m.xg[0], away_xg=m.xg[1],
    )
    formation = st.session_state.get("cd_formation", "4-3-3")
    picks = {k: v.to_profile() for k, v in st.session_state.cd_picks.items()}
    page.play_live_match(mr, squad_team_name=SQUAD_TEAM_NAME,
                         picks=picks, coords=slot_coords(formation))


def _match_row(m) -> None:
    is_home = m.home == SQUAD_TEAM_NAME
    gf, ga = (m.home_goals, m.away_goals) if is_home else (m.away_goals, m.home_goals)
    colour = "#4ade80" if gf > ga else ("#f87171" if ga > gf else "#facc15")
    goals = " · ".join(f"⚽ {s}" + (f" ({a})" if a else "") for s, a in (m.goal_events or []))
    st.markdown(
        f'<div style="padding:7px 10px;background:#151f30;border-left:3px solid {colour};'
        f'border-radius:6px;margin:3px 0">'
        f'<div style="display:flex;gap:8px;align-items:center">'
        f'<div style="flex:1;text-align:right;font-size:.85rem">{m.home}</div>'
        f'<div style="min-width:48px;text-align:center;font-weight:800;color:{colour}">'
        f'{m.home_goals}-{m.away_goals}</div>'
        f'<div style="flex:1;font-size:.85rem">{m.away}</div></div>'
        + (f'<div style="font-size:.72rem;color:#9aa9bf;margin-top:3px">{goals}</div>'
           if goals else "")
        + '</div>', unsafe_allow_html=True)
