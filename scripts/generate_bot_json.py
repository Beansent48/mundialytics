"""Generate predictions_latest.json for the Telegram bot.

Extracts a ±3 day window from the prediction log with only bot-relevant
markets (1X2, Goles, corners, yellows) and embeds actual results from
the foundation CSVs and UEFA fixture files so the bot can settle markets
without calling ESPN (which blocks Cloudflare Worker IPs).

Output: data/processed/logs/predictions_latest.json
Format: {"predictions": [...], "results": {"home|away": {hg, ag, hc, ac, hy, ay}}}
"""
import csv
import json
import pathlib
import re
import unicodedata
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "processed" / "logs" / "predictions_log.csv"
OUT = ROOT / "data" / "processed" / "logs" / "predictions_latest.json"

KEEP = {
    "fecha", "home", "away", "mercado", "linea", "prob", "seleccion",
    "lambda_home", "lambda_away", "p_home", "p_draw", "p_away", "ambito",
}
MKTS = {"1X2", "Goles", "corners", "yellows"}

_SCORE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")

_TRANSLIT = {"æ": "ae", "ø": "oe", "å": "aa", "ß": "ss", "ð": "d", "þ": "th",
             "ł": "l", "đ": "d", "ı": "i", "œ": "oe",
             "Æ": "ae", "Ø": "oe", "Å": "aa", "Ð": "d", "Þ": "th",
             "Ł": "l", "Đ": "d", "Œ": "oe"}


def _norm(name: str) -> str:
    """Normalize a team name for matching: transliterate, strip accents, lowercase."""
    for k, v in _TRANSLIT.items():
        name = name.replace(k, v)
    name = "".join(ch for ch in unicodedata.normalize("NFKD", name) if not unicodedata.combining(ch))
    return name.lower().strip()


def main():
    if not LOG.exists():
        print("no prediction log yet")
        return

    today = date.today()
    window = {str(today + timedelta(d)) for d in range(-3, 4)}
    window_dd_mm = set()
    for d in range(-3, 4):
        dt = today + timedelta(d)
        window_dd_mm.add(f"{dt.day:02d}/{dt.month:02d}/{dt.year}")

    rows = []
    with open(LOG, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("fecha") in window and r.get("mercado") in MKTS:
                rows.append({k: r[k] for k in KEEP if k in r})

    results = {}
    for fn in ("foundation_big5_multi_season.csv", "foundation_second_divisions.csv"):
        fp = ROOT / "data" / "processed" / fn
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = r.get("date", "")
                if d in window and r.get("home_goals", ""):
                    key = r["home_team"].lower() + "|" + r["away_team"].lower()
                    results[key] = {
                        "hg": int(float(r["home_goals"])),
                        "ag": int(float(r["away_goals"])),
                        "hc": r.get("home_corners", ""),
                        "ac": r.get("away_corners", ""),
                        "hy": r.get("home_yellow_cards", ""),
                        "ay": r.get("away_yellow_cards", ""),
                    }

    # ESPN UEFA: match by (date, score) → event_id, then team stats by event_id
    espn_uefa_stats: dict[str, dict] = {}
    espn_by_score: dict[tuple, str] = {}  # (date, hg, ag) → event_id (+ teams)
    espn_match_f = ROOT / "data" / "external" / "advanced" / "espn" / "espn_matches_uefa.csv"
    espn_team_f = ROOT / "data" / "external" / "advanced" / "espn" / "espn_team_match_uefa.csv"
    if espn_match_f.exists() and espn_team_f.exists():
        with open(espn_team_f, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                espn_uefa_stats[(r["event_id"], r["team"].lower())] = r
        with open(espn_match_f, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                eid = r["event_id"]
                espn_by_score[(r["date"], r["home_goals"], r["away_goals"],
                               r["home_team"].lower(), r["away_team"].lower())] = eid

    def _name_overlap(a: str, b: str) -> bool:
        na, nb = _norm(a), _norm(b)
        if na == nb:
            return True
        wa = set(na.replace("/", " ").split())
        wb = set(nb.replace("/", " ").split())
        if wa & wb:
            return True
        return na in nb or nb in na

    def _espn_enrich(date_iso: str, hg: int, ag: int,
                     home_uefa: str, away_uefa: str) -> tuple:
        """Find ESPN event matching a UEFA result and return (hc, ac, hy, ay)."""
        for (d, ehg, eag, eh, ea), eid in espn_by_score.items():
            if d != date_iso or int(ehg) != hg or int(eag) != ag:
                continue
            if _name_overlap(home_uefa, eh):
                sh = espn_uefa_stats.get((eid, eh), {})
                sa = espn_uefa_stats.get((eid, ea), {})
                return (sh.get("corners", ""), sa.get("corners", ""),
                        sh.get("yellows", ""), sa.get("yellows", ""))
        return ("", "", "", "")

    uefa_dir = ROOT / "data" / "external" / "uefa"
    if uefa_dir.is_dir():
        for fp in uefa_dir.glob("raw_*_2026.csv"):
            with open(fp, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    raw_date = r.get("Date", "")
                    dd_mm = raw_date.split(" ")[0] if raw_date else ""
                    if dd_mm not in window_dd_mm:
                        continue
                    result_str = r.get("Result", "").strip()
                    m = _SCORE_RE.match(result_str)
                    if not m:
                        continue
                    hg, ag = int(m.group(1)), int(m.group(2))
                    # convert DD/MM/YYYY to YYYY-MM-DD for ESPN lookup
                    parts = dd_mm.split("/")
                    date_iso = f"{parts[2]}-{parts[1]}-{parts[0]}" if len(parts) == 3 else ""
                    hc, ac, hy, ay = _espn_enrich(
                        date_iso, hg, ag, r["Home Team"], r["Away Team"])
                    key = r["Home Team"].lower() + "|" + r["Away Team"].lower()
                    if key not in results:
                        results[key] = {
                            "hg": hg, "ag": ag,
                            "hc": hc, "ac": ac, "hy": hy, "ay": ay,
                        }

    # ESPN UEFA matches as a direct result source (UEFA CSVs often lack the
    # Result field for recent matchdays while ESPN already has scores + stats).
    # Match ESPN names to prediction-log names by date + word overlap.
    pred_teams_by_date: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        d = r.get("fecha", "")
        h, a = r.get("home", ""), r.get("away", "")
        if d and h and a:
            pred_teams_by_date.setdefault(d, [])
            if (h, a) not in pred_teams_by_date[d]:
                pred_teams_by_date[d].append((h, a))

    for (d, ehg, eag, eh, ea), eid in espn_by_score.items():
        if d not in window or not ehg or not eag:
            continue
        matched_key = None
        for ph, pa in pred_teams_by_date.get(d, []):
            if _name_overlap(eh, ph) and _name_overlap(ea, pa):
                matched_key = ph.lower() + "|" + pa.lower()
                break
        if not matched_key:
            matched_key = eh + "|" + ea
        if matched_key not in results:
            sh = espn_uefa_stats.get((eid, eh), {})
            sa = espn_uefa_stats.get((eid, ea), {})
            results[matched_key] = {
                "hg": int(ehg), "ag": int(eag),
                "hc": sh.get("corners", ""), "ac": sa.get("corners", ""),
                "hy": sh.get("yellows", ""), "ay": sa.get("yellows", ""),
            }

    # ── Real odds from The Odds API log ──
    odds: dict[str, dict] = {}
    odds_csv = ROOT / "data" / "external" / "odds_api" / "odds_log.csv"
    if odds_csv.exists():
        mkt_map = {
            "totals": "Goles",
            "alternate_totals_corners": "corners",
            "alternate_totals_cards": "yellows",
        }
        with open(odds_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ct = r.get("commence_time", "")
                if not ct:
                    continue
                match_date = ct[:10]
                if match_date not in window:
                    continue
                raw_mkt = r.get("market", "")
                h = _norm(r.get("home_team", ""))
                a = _norm(r.get("away_team", ""))
                if not (h and a):
                    continue
                # Match API names to prediction-log names
                matched_key = None
                for ph, pa in pred_teams_by_date.get(match_date, []):
                    if _name_overlap(r["home_team"], ph) and _name_overlap(r["away_team"], pa):
                        matched_key = ph.lower() + "|" + pa.lower()
                        break
                if not matched_key:
                    matched_key = r["home_team"].lower() + "|" + r["away_team"].lower()
                entry = odds.setdefault(matched_key, {})

                # h2h: best odds per outcome (home/away/draw)
                if raw_mkt == "h2h":
                    over_p = r.get("over_price", "")
                    under_p = r.get("under_price", "")
                    draw_p = r.get("draw_price", "")
                    if not (over_p and under_p and draw_p):
                        continue
                    h2h = entry.setdefault("h2h", [0.0, 0.0, 0.0])
                    ho, ao, do_ = float(over_p), float(under_p), float(draw_p)
                    if ho > h2h[0]: h2h[0] = round(ho, 3)
                    if do_ > h2h[1]: h2h[1] = round(do_, 3)
                    if ao > h2h[2]: h2h[2] = round(ao, 3)
                    continue

                mkt = mkt_map.get(raw_mkt)
                if not mkt:
                    continue
                line = r.get("line", "")
                over_p = r.get("over_price", "")
                under_p = r.get("under_price", "")
                if not (line and over_p and under_p):
                    continue
                mkt_entry = entry.setdefault(mkt, {})
                line_entry = mkt_entry.setdefault(line, [0.0, 0.0])
                o, u = float(over_p), float(under_p)
                if o > line_entry[0]:
                    line_entry[0] = round(o, 3)
                if u > line_entry[1]:
                    line_entry[1] = round(u, 3)
        odds_matches = sum(1 for v in odds.values() if v)
        odds_lines = sum(sum(len(m) for m in v.values()) for v in odds.values())
        print(f"{odds_matches} matches with real odds, {odds_lines} lines")
    else:
        print("no odds_log.csv — skipping real odds")

    OUT.write_text(
        json.dumps({"predictions": rows, "results": results, "odds": odds},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{len(rows)} preds, {len(results)} results, {len(odds)} odds -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
