"""Human-readable catalogue of useful free/reliable football data sources.

This module does not automatically scrape everything. It documents where each
loader should point and keeps URLs in one place.
"""

DATA_SOURCES = {
    "national_results_martj42": {
        "scope": "National teams: men's full internationals results from 1872 onward.",
        "url": "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017",
        "best_for": ["ELO", "Poisson goals", "tournament backtesting"],
        "limitations": "Kaggle download needs manual/API setup; no rich event data.",
    },
    "statsbomb_open_data": {
        "scope": "Event, lineup and match data for selected competitions/seasons.",
        "url": "https://github.com/statsbomb/open-data",
        "best_for": ["shots", "fouls", "passes", "lineups", "player props prototypes"],
        "limitations": "Coverage is open-data subset, not all matches. Respect licence attribution.",
    },
    "football_data_uk": {
        "scope": "Club match results, odds and match stats in CSV.",
        "url": "https://www.football-data.co.uk/",
        "best_for": ["club results", "current fixtures where available", "odds backtesting", "1X2 / totals"],
        "limitations": "Mostly team/match-level; not player props.",
    },
    "clubelo": {
        "scope": "Historical/current ELO ratings for European club football.",
        "url": "https://clubelo.com/",
        "best_for": ["club power ratings", "club model features"],
        "limitations": "European club focus; use responsibly and cache.",
    },
    "world_football_elo": {
        "scope": "International team ELO ratings.",
        "url": "https://eloratings.net/",
        "best_for": ["national team benchmark", "sanity-check own ELO"],
        "limitations": "Not packaged as clean API; prefer own ELO for reproducibility.",
    },
    "openfootball": {
        "scope": "Public domain fixtures/results in JSON/TXT for clubs and national tournaments.",
        "url": "https://github.com/openfootball",
        "best_for": ["future fixtures", "club schedules", "world cup schedules", "lightweight demos"],
        "limitations": "Not rich stats/events; schema varies by repository.",
    },
    "soccerdata_py": {
        "scope": "Python scrapers for Club Elo, FBref, Football-Data, SoFIFA, Understat, etc.",
        "url": "https://soccerdata.readthedocs.io/",
        "best_for": ["Python-first data ingestion", "club/player stats"],
        "limitations": "Scraping can break; respect websites' terms and rate limits.",
    },
}
