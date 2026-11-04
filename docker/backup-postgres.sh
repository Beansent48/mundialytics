#!/bin/sh
# Nightly dump of the accounts database.
#
# Postgres is the only thing in this stack that needs a backup, and that is a
# measured statement rather than a cautious one: `data/` is reproducible from
# the daily refresh, `outputs/` is not served, the images rebuild from the
# repository — and the accounts are the only bytes in the whole deployment that
# exist nowhere else. Losing them signs every user out of an account they cannot
# get back.
#
# Plain `pg_dump` into a gzip, kept for KEEP days. No object storage and no
# second provider: at this size, a file on the same disk protects against the
# failure that actually happens (a bad migration, a wrong DELETE), and the
# off-site copy is whatever pulls this directory down -- see the note at the
# bottom.
#
# Install on the server as root's crontab, which is the whole of "scheduled":
#
#   30 4 * * *  /srv/mundialytics/docker/backup-postgres.sh >> /var/log/mundialytics-backup.log 2>&1
#
# 04:30 is deliberate: after the 10:00 refresh has long finished and well before
# it starts again, so a dump never competes with a rebuild for the disk.
set -eu

# The compose project directory, so this works from cron with no cwd.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups}"
KEEP="${KEEP:-14}"

cd "$PROJECT_DIR"

# Read the credentials the stack itself uses, so there is one place to change
# them. `set -a` exports every assignment in .env.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
USER_NAME="${POSTGRES_USER:-mundialytics}"
DB_NAME="${POSTGRES_DB:-mundialytics}"

mkdir -p "$BACKUP_DIR"
STAMP="$(date +%Y-%m-%d_%H%M)"
OUT="$BACKUP_DIR/mundialytics_${STAMP}.sql.gz"

# -T: no TTY, because cron has none. Written to a .part first and renamed, so a
# dump interrupted halfway is never mistaken for a backup.
docker compose exec -T postgres pg_dump -U "$USER_NAME" -d "$DB_NAME" \
  | gzip -9 > "$OUT.part"
mv "$OUT.part" "$OUT"

SIZE="$(du -h "$OUT" | cut -f1)"
echo "$(date -Is) backed up $DB_NAME -> $OUT ($SIZE)"

# Retention. `-mtime +N` counts whole days, so KEEP=14 keeps a fortnight.
find "$BACKUP_DIR" -name 'mundialytics_*.sql.gz' -mtime "+$KEEP" -delete
find "$BACKUP_DIR" -name 'mundialytics_*.sql.gz.part' -mtime +1 -delete

# Restoring, for when it matters and nobody remembers:
#
#   gunzip -c backups/mundialytics_2026-11-03_0430.sql.gz \
#     | docker compose exec -T postgres psql -U mundialytics -d mundialytics
#
# Into an EMPTY database. A dump replayed over a live one leaves both halves.
#
# Off-site: nothing here copies the file off the machine. The simplest honest
# answer for this project is to pull it from the PC that already holds
# everything else, e.g. from the Windows box's own scheduled task:
#
#   scp root@server:/srv/mundialytics/backups/*.sql.gz D:\mundialytics-backups\
