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

    def _espn_enrich(date_iso: str, hg: int, ag: int,
                     home_uefa: str, away_uefa: str) -> tuple:
        """Find ESPN event matching a UEFA result and return (hc, ac, hy, ay)."""
        hu, au = home_uefa.lower(), away_uefa.lower()
        for (d, ehg, eag, eh, ea), eid in espn_by_score.items():
            if d != date_iso or int(ehg) != hg or int(eag) != ag:
                continue
            # score + date match; check if team names overlap (fuzzy)
            hw = set(hu.replace("/", " ").split())
            ew = set(eh.replace("/", " ").split())
            if hw & ew or hu in eh or eh in hu:
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

    OUT.write_text(
        json.dumps({"predictions": rows, "results": results}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{len(rows)} preds, {len(results)} results -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
