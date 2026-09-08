#!/usr/bin/env python3
"""Check the player data files for the failures that hide in plain sight.

WHY THIS EXISTS. Every serious defect found in the player pipeline so far was
silent: a name key that missed 31% of squads, a hyphen that sank 29% of
goalkeepers, 891 duplicate names in an FBref table handing Manchester City's
Rodri the numbers of Villarreal B's, a display curve pinning 53.8% of ratings
to one value, ESPN filing a goalkeeper under "D". None of them raised an error.
All of them would have been caught by looking.

So this looks — at the keys, the duplicates, the placeholders, the ranges and
the joins BETWEEN files, which is where the pipeline actually breaks. It reports
and exits non-zero on anything fatal, so it can gate a rebuild.

Run: .venv/Scripts/python.exe scripts/audit_player_data.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

P = ROOT / "data/processed"
FB = ROOT / "data/external/advanced/fbref_kaggle"

_ODD = re.compile(r"[^\w\s'’‘ʼ´`\-\.]")

FATAL: list[str] = []
WARN: list[str] = []


def fatal(msg: str) -> None:
    FATAL.append(msg)
    print(f"  FALLO   {msg}")


def warn(msg: str) -> None:
    WARN.append(msg)
    print(f"  aviso   {msg}")


def ok(msg: str) -> None:
    print(f"  ok      {msg}")


def head(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def read(path: Path, **kw) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"no existe: {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path, low_memory=False, **kw)


# ── 1. keys and duplicates ────────────────────────────────────────────────────
def check_keys() -> dict:
    head("1. CLAVES Y DUPLICADOS")
    frames = {}
    specs = [
        ("cards", P / "squadlab_cards.csv", ["card_id"], True),
        ("cards_person", P / "squadlab_cards.csv", ["kind", "player", "club", "season_label"], True),
        ("squads", P / "current_squads.csv", ["team", "player"], True),
        ("roles", P / "player_ratings_roles.csv", ["pn"], True),
        ("profiles", P / "player_profiles_with_positions.csv", ["player"], False),
        ("defense", P / "player_defense_fbref.csv", ["player"], True),
        ("attack", P / "player_attack_fbref.csv", ["player"], True),
    ]
    for name, path, keys, must_be_unique in specs:
        df = read(path)
        if df is None:
            continue
        frames[name] = df
        missing = [k for k in keys if k not in df.columns]
        if missing:
            fatal(f"{name}: faltan columnas clave {missing}")
            continue
        dup = int(df.duplicated(subset=keys).sum())
        label = f"{name} [{'+'.join(keys)}]"
        if dup and must_be_unique:
            ej = df[df.duplicated(subset=keys, keep=False)].head(3)[keys].to_dict("records")
            fatal(f"{label}: {dup:,} duplicados — p.ej. {ej}")
        elif dup:
            warn(f"{label}: {dup:,} duplicados (esperado en este fichero)")
        else:
            ok(f"{label}: {len(df):,} filas, sin duplicados")
    return frames


# ── 2. names: the key every join in this project runs on ──────────────────────
def check_names(frames: dict) -> None:
    head("2. NOMBRES (la clave de la que depende todo)")
    for name, col in (("cards", "player"), ("squads", "player"),
                      ("roles", "player"), ("defense", "player"), ("attack", "player")):
        df = frames.get(name)
        if df is None or col not in df.columns:
            continue
        s = df[col].astype(str)
        bad = []
        if (s != s.str.strip()).any():
            bad.append(f"{int((s != s.str.strip()).sum())} con espacios al borde")
        if s.str.contains(r"\s{2,}", regex=True).any():
            bad.append(f"{int(s.str.contains(r'  ').sum())} con espacios dobles")
        empty = int((s.str.strip() == "").sum() + s.isin(["nan", "None"]).sum())
        if empty:
            bad.append(f"{empty} vacíos o 'nan'")
        # Characters `_fold` cannot reconcile. Accents are fine (it strips them)
        # and so are the apostrophe variants (it normalises them since the
        # U+2019 bug); anything else is a name two files will never agree on.
        # `re.search` explicitly, not `Series.str.contains`: on this data the two
        # disagree — pandas flags "Facundo Garcés" against a pattern that
        # `re.search` finds nothing in. An audit that cries wolf is not an audit.
        odd = s[s.map(lambda x: bool(_ODD.search(x)))]
        if len(odd):
            bad.append(f"{len(odd)} con caracteres que el folding no reconcilia "
                       f"(p.ej. {odd.iloc[0]!r})")
        if bad:
            warn(f"{name}.{col}: " + "; ".join(bad))
        else:
            ok(f"{name}.{col}: limpio")

    # the same person spelled two ways INSIDE one file breaks its own dedupe
    for name in ("roles", "attack", "defense"):
        df = frames.get(name)
        if df is None:
            continue
        fold = df["player"].astype(str).map(
            lambda x: "".join(c for c in unicodedata.normalize("NFKD", x.lower())
                              if not unicodedata.combining(c)).strip())
        dup = int(fold.duplicated().sum())
        if dup:
            ej = df.loc[fold.duplicated(keep=False), "player"].head(4).tolist()
            warn(f"{name}: {dup:,} nombres que solo difieren en tildes/mayúsculas — {ej}")
        else:
            ok(f"{name}: sin colisiones por tildes")


# ── 3. the joins that actually matter ─────────────────────────────────────────
def check_joins(frames: dict) -> None:
    head("3. ¿RESUELVEN LOS CRUCES ENTRE FICHEROS?")
    from mundialytics.identity.current_squads import NameIndex

    squads, roles = frames.get("squads"), frames.get("roles")
    if squads is None or roles is None:
        return
    idx = NameIndex()
    for r in roles.itertuples(index=False):
        idx.add(r.player, float(r.base_ovr), rank=float(r.matches or 0))
    hits = [idx.lookup(w) is not None for w in squads["player"]]
    rate = float(np.mean(hits))
    msg = f"plantillas -> ratings: {int(np.sum(hits)):,}/{len(hits):,} ({rate:.0%})"
    (ok if rate >= 0.70 else warn)(msg)

    for other, label in (("attack", "ataque FBref"), ("defense", "defensa FBref")):
        df = frames.get(other)
        if df is None:
            continue
        i2 = NameIndex()
        for r in df.itertuples(index=False):
            i2.add(r.player, 1, rank=float(getattr(r, "n90", 0) or 0))
        h = float(np.mean([i2.lookup(w) is not None for w in squads["player"]]))
        (ok if h >= 0.40 else warn)(f"plantillas -> {label}: {h:.0%}")


# ── 4. placeholders and dead columns ──────────────────────────────────────────
# A repeated value is only a defect when the column is supposed to describe the
# player. A squad member with zero goals in September is not a defect; a career
# profile whose duel-win rate is exactly 0.500 for 73% of players is the whole
# reason this file exists. So the check knows which is which instead of shouting
# at both.
EXPECTED_SPARSE = {
    # counting stats a few weeks into a season, legitimately zero for most
    ("squads", "goals"), ("squads", "shots"), ("squads", "sot"), ("squads", "assists"),
    ("squads", "yellow_cards"), ("squads", "red_cards"), ("squads", "squad_share"),
    ("squads", "starts"), ("squads", "apps"), ("squads", "squad_matches"),
    # design placeholders, documented where they are set
    ("cards", "gk"), ("cards", "raw_gk"), ("cards", "n_def"), ("cards", "n_att"),
    ("cards", "matches"),
    # one row per player, and 86% of them come from the fuller 23/24 tables
    ("attack", "season"), ("defense", "season"),
    # a coverage flag: 1.0 means every signal present, which is the good case
    ("attack", "att_signals"), ("defense", "def_signals"),
    # rare events
    ("profiles", "big_chances_missed_per_match"), ("profiles", "big_chance_miss_rate"),
    ("profiles", "cut_backs_per_match"), ("profiles", "through_balls_per_match"),
}
# The neutral values the profile builder writes when a stat was never parsed.
# These are not "sparse data", they are a measurement standing in for itself.
NEUTRAL = {0.5, 0.75}


def check_placeholders(frames: dict) -> None:
    head("4. RELLENOS Y COLUMNAS MUERTAS")
    blind = []
    for name, df in frames.items():
        if name == "cards_person":
            continue
        for c in df.columns:
            if (name, c) in EXPECTED_SPARSE:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() < len(df) * 0.5:
                continue
            if s.nunique() <= 1:
                fatal(f"{name}.{c}: columna constante ({s.dropna().iloc[0] if s.notna().any() else '?'})")
                continue
            top = s.value_counts(normalize=True)
            if not len(top) or top.iloc[0] < 0.30:
                continue
            v, share = float(top.index[0]), float(top.iloc[0])
            if name == "profiles" and (v in NEUTRAL or v == 0.0):
                blind.append((c, share, v))
            elif share >= 0.60:
                warn(f"{name}.{c}: {share:.0%} de las filas valen exactamente {v}")
    if blind:
        worst = max(s for _, s, _ in blind)
        print(f"  aviso   profiles: {len(blind)} columnas de EVENTOS con su valor neutro "
              f"en hasta el {worst:.0%} de las filas — es el perfil ciego "
              f"(defense_creation_matches = 0), conocido y corregido aguas abajo")
        WARN.append("profiles: columnas de eventos en su valor neutro (perfil ciego)")


# ── 5. ranges and internal consistency ────────────────────────────────────────
def check_ranges(frames: dict) -> None:
    head("5. RANGOS Y COHERENCIA INTERNA")
    cards = frames.get("cards")
    if cards is not None:
        for c, lo, hi in (("overall", 40, 99), ("attack", 20, 99), ("defense", 20, 99),
                          ("creation", 20, 99), ("gk", 20, 99)):
            if c not in cards.columns:
                continue
            v = pd.to_numeric(cards[c], errors="coerce").dropna()
            out = int(((v < lo) | (v > hi)).sum())
            (ok if not out else fatal)(f"cards.{c}: {len(v):,} valores, {out} fuera de [{lo},{hi}]")

        gk = cards[cards["position"] == "Goalkeeper"]
        bad = int(gk["attack"].notna().sum())
        (ok if not bad else fatal)(f"porteros sin eje de ataque: {len(gk) - bad}/{len(gk)}")
        nogk = cards[cards["position"] != "Goalkeeper"]
        bad = int(nogk["defense"].isna().sum())
        (ok if not bad else fatal)(f"jugadores de campo con defensa: {len(nogk) - bad}/{len(nogk)}")

        empty_role = int((cards["role"].isna() | (cards["role"].astype(str).str.strip() == "")
                          | (cards["role"].astype(str) == "nan")).sum())
        (ok if not empty_role else warn)(f"cartas sin rol: {empty_role:,} de {len(cards):,}")

        from mundialytics.statistical_core.squadlab import cards as C
        role_pos = {
            "Portero": "Goalkeeper", "Central stopper": "Defender",
            "Central de salida": "Defender", "Lateral ofensivo": "Defender",
            "Lateral defensivo": "Defender", "Destructor": "Midfielder",
            "Pivote organizador": "Midfielder", "Creador": "Midfielder",
            "Box-to-box": "Midfielder", "Mediapunta": "Midfielder",
            "Extremo": "Forward", "Extremo interior": "Forward", "Killer": "Forward",
            "Target man": "Forward", "Delantero completo": "Forward", "Falso 9": "Forward",
        }
        want = cards["role"].map(role_pos)
        clash = cards[want.notna() & (want != cards["position"])]
        if len(clash):
            byp = clash.groupby(["position", "role"]).size().sort_values(ascending=False)
            warn(f"rol y posición se contradicen en {len(clash):,} cartas "
                 f"({len(clash) / len(cards):.0%}) — top: {byp.head(4).to_dict()}")
        else:
            ok("rol y posición coherentes en todas las cartas")
        assert C  # imported for the season-label helper below

        seasons = cards.loc[cards["kind"] == "prime", "season_label"].astype(str)
        bad = seasons[~seasons.str.match(r"^\d{2}/\d{2}$")]
        (ok if bad.empty else fatal)(
            f"etiquetas de temporada de los primes: {len(seasons) - len(bad)}/{len(seasons)} "
            f"con formato AA/AA" + (f" — malas: {bad.unique()[:4].tolist()}" if len(bad) else ""))

    for name in ("attack", "defense"):
        df = frames.get(name)
        if df is None:
            continue
        for c in ("duel_pct", "aerial_pct", "challenge_pct", "att_signals", "def_signals"):
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            hi = 1.0 if "signals" in c else 100.0
            out = int(((v < 0) | (v > hi)).sum())
            (ok if not out else fatal)(f"{name}.{c}: {out} fuera de [0,{hi}]")


# ── 6. the FBref source tables, where the duplicate bug lived ─────────────────
def check_source() -> None:
    head("6. TABLAS FUENTE DE FBREF")
    for f in sorted(FB.glob("2324_*.csv")):
        df = pd.read_csv(f)
        if "player" not in df.columns:
            continue
        dup_name = int(df["player"].duplicated().sum())
        keyed = ["player", "club"] if "club" in df.columns else ["player"]
        dup_key = int(df.duplicated(subset=keyed).sum())
        if dup_key:
            # two different men with the same name at the same club. Not fixable
            # from a name key, and handled where it matters: fbref_quality keeps
            # the fuller season before merging, so the tables never go
            # many-to-many (that bug inflated 10,578 rows to 10,596 and blended
            # the two "João Pedro" at Grêmio).
            warn(f"{f.name}: {dup_key:,} homónimos en el mismo club — "
                 "se desduplica al cargar, nunca cruzar sin ello")
        elif dup_name:
            ok(f"{f.name}: {dup_name:,} nombres repetidos, únicos por (jugador, club) — "
               "cruzar SIEMPRE por ambos")
        else:
            ok(f"{f.name}: sin duplicados")


def main() -> int:
    print("AUDITORÍA DE DATOS DE JUGADOR")
    frames = check_keys()
    check_names(frames)
    check_joins(frames)
    check_placeholders(frames)
    check_ranges(frames)
    check_source()
    head("RESUMEN")
    print(f"  fallos: {len(FATAL)}   avisos: {len(WARN)}")
    for m in FATAL:
        print(f"    FALLO  {m}")
    return 1 if FATAL else 0


if __name__ == "__main__":
    raise SystemExit(main())
