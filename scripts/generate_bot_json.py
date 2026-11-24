"""Generate predictions_latest.json for the Telegram bot.

Extracts a ±3 day window from the prediction log with only bot-relevant
markets (1X2, Goles, corners, yellows) and embeds actual results from
the foundation CSVs so the bot can settle markets without calling ESPN.

Output: data/processed/logs/predictions_latest.json
Format: {"predictions": [...], "results": {"home|away": {hg, ag, hc, ac, hy, ay}}}
"""
import csv
import json
import pathlib
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "data" / "processed" / "logs" / "predictions_log.csv"
OUT = ROOT / "data" / "processed" / "logs" / "predictions_latest.json"

KEEP = {
    "fecha", "home", "away", "mercado", "linea", "prob", "seleccion",
    "lambda_home", "lambda_away", "p_home", "p_draw", "p_away", "ambito",
}
MKTS = {"1X2", "Goles", "corners", "yellows"}


def main():
    if not LOG.exists():
        print("no prediction log yet")
        return

    today = date.today()
    window = {str(today + timedelta(d)) for d in range(-3, 4)}

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

    OUT.write_text(
        json.dumps({"predictions": rows, "results": results}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{len(rows)} preds, {len(results)} results -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
