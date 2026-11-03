# The model server, and -- as a second stage -- the daily refresh that feeds it.
#
# Two things about this image are deliberate and easy to undo by accident:
#
#   1. **No data.** `data/` is mounted, never copied. The daily refresh rewrites
#      the foundation, so an image with a CSV inside would have to be rebuilt
#      once a day; and the tree is 26.5 GB, of which the serving path opens a
#      fraction (measured by scripts/measure_serving_bundle.py).
#   2. **PYTHONPATH, not `pip install .`** The engine finds its own data with
#      `Path(__file__).resolve().parents[3]`, so `mundialytics` has to be
#      imported from the source tree that sits next to `data/`. A non-editable
#      install would put it in site-packages and the engine would look for
#      `data/` inside site-packages.
#
# Build:  docker compose build api
# Run:    docker compose up -d api        (see docker-compose.yml)

# 3.14 to match requirements-lock.txt, which is the exact set the full suite
# last passed with on this project's own machine. CI additionally runs the
# ranges in requirements.txt on 3.10 and 3.12; the image does not have to.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim AS api

# tzdata because kickoff rendering and the "today" window of /fixtures/day are
# date-sensitive and glibc needs the zoneinfo files to honour TZ. The Python
# `tzdata` wheel in requirements covers zoneinfo, not the C library.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    TZ=Europe/Madrid

WORKDIR /app

# Dependencies first, so editing the engine does not reinstall scipy.
#
# The lock is the whole project's known-good set, and three of its entries have
# no business in a serving image: `soccerdata` (which pulls Selenium) belongs to
# the daily refresh, `matplotlib` draws the figures in docs/, and `pytest` runs
# the suite. Filtered rather than split into a second file, so there is still
# one list of versions — and checked: nothing the serving path reads imports
# any of the three (measured, with scripts/measure_serving_bundle.py's read set).
COPY requirements-lock.txt ./
RUN grep -vE '^(soccerdata|matplotlib|pytest)' requirements-lock.txt > /tmp/serving.txt \
 && pip install --no-cache-dir -r /tmp/serving.txt \
 && rm /tmp/serving.txt

# The serving path, and the scripts the refresh stage below runs. `data/` is
# absent on purpose; `config/` carries the YAML the engine reads at import.
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml README.md ./

# Writes under /app/data go to the mounted volume; nothing else is written.
RUN useradd --create-home --uid 10001 mundialytics \
 && mkdir -p /app/data \
 && chown -R mundialytics:mundialytics /app
USER mundialytics

EXPOSE 8000

# `ready` and not just a 200: /health answers 200 while the models load, which
# it has to -- a probe must not be what triggers the loading -- so a container
# that is up but still warming is not yet a container that can serve a page.
# The start_period that makes this survivable is in docker-compose.yml, where
# the arithmetic behind it is written down: during a startup fit uvicorn has
# not bound the port at all and this probe gets a refused connection.
HEALTHCHECK --interval=15s --timeout=10s --start-period=420s --retries=4 \
  CMD ["python", "-c", "import json,urllib.request,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)).get('ready') else 1)"]

# One worker on purpose. Each would hold its own copy of the fit (29 MB of
# joblib plus the foundation in pandas) and pay its own cold start; the engine
# is CPU-bound per request and the box has two cores.
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "65"]


# ── The daily refresh ─────────────────────────────────────────────────────────
# A separate stage so the serving image does not carry the scraping stack
# (soccerdata pulls Selenium). Same code, same data mount, different entrypoint:
# scripts/refresh_loop.py waits for the next REFRESH_AT and runs
# scripts/update_season.py, exactly as the Windows scheduled task does today.
FROM api AS refresh

USER root
# The three the serving stage filtered out, at the lock's own versions, plus the
# two the refresh needs that the lock does not carry (they are pyproject's
# `data` extra: the transfermarkt provider parses HTML).
RUN grep -E '^(soccerdata|matplotlib)' requirements-lock.txt > /tmp/refresh.txt \
 && pip install --no-cache-dir -r /tmp/refresh.txt \
      "beautifulsoup4>=4.12" "lxml>=5.0" \
 && rm /tmp/refresh.txt
USER mundialytics

# HOME and SOCCERDATA_DIR both default to the user's home directory, and this
# container may run as a uid that has none: compose sets `user:` to whoever owns
# the data mount, because the refresh has to be able to write it. Pointing them
# at the volume gives soccerdata and anything else home-seeking a writable path,
# and makes the scrape cache survive a restart instead of being re-downloaded.
# scripts/refresh_loop.py creates both on startup.
ENV REFRESH_AT=10:00 \
    REVALIDATE_URL=http://web:3000/api/revalidate \
    HOME=/app/data/processed/container_home \
    SOCCERDATA_DIR=/app/data/processed/soccerdata_cache

# No healthcheck: this process is asleep almost all day, and a probe that calls
# it healthy only while it is refreshing would restart it every night.
HEALTHCHECK NONE

CMD ["python", "scripts/refresh_loop.py"]
