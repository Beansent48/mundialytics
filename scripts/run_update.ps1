<#
.SYNOPSIS
    Unattended weekly/daily refresh for the Mundialytics engine.

.DESCRIPTION
    Wraps scripts/update_season.py for Windows Task Scheduler so predictions and
    results never go stale on their own. It:
      * runs the light refresh every day (download results, rebuild foundation,
        settle player markets, log the upcoming round pre-kickoff);
      * runs the heavy --full refresh once a week (regenerates the deployed
        walk-forward cache behind the Resultados page, ~15 min) on -FullDay;
      * tees everything to data/processed/logs/runs/update_<date>.log and keeps
        the last 30;
      * writes data/processed/logs/last_run.json (timestamp, exit code, log tail)
        so a missed or failed run is visible instead of silently leaving a gap.

    The Python step is already crash-safe: a bad download is rolled back by the
    foundation integrity check before any model or cache is touched, so a failed
    run leaves the deployed data exactly as it was.

.PARAMETER FullDay
    Day-of-week whose run also passes --full. Default Monday.

.PARAMETER Force
    Pass --full regardless of the weekday (for a manual catch-up run).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_update.ps1
#>
[CmdletBinding()]
param(
    [ValidateSet('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday')]
    [string]$FullDay = 'Monday',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root   = Split-Path -Parent $PSScriptRoot           # repo root (scripts/..)
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'scripts\update_season.py'

if (-not (Test-Path $python)) { throw "venv python not found at $python" }
if (-not (Test-Path $script)) { throw "update_season.py not found at $script" }

$logDir = Join-Path $root 'data\processed\logs\runs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp  = Get-Date -Format 'yyyy-MM-dd_HHmm'
$logIso = (Get-Date).ToString('o')
$log    = Join-Path $logDir "update_$stamp.log"

# --full only on the weekly day (or when forced): the heavy walk-forward cache
# regeneration does not need to run every single day.
$isFull = $Force -or ((Get-Date).DayOfWeek.ToString() -eq $FullDay)
$argv   = @($script)
if ($isFull) { $argv += '--full' }

"=== run_update $logIso  (full=$isFull) ===" | Tee-Object -FilePath $log
# Native stdout+stderr straight to the log; the Python side prints its own
# step-by-step progress and never prompts.
& $python @argv *>&1 | Tee-Object -FilePath $log -Append
$code = $LASTEXITCODE

# A heartbeat a monitor (or you) can read to spot a gap without opening logs.
$tail = (Get-Content $log -Tail 25 -ErrorAction SilentlyContinue) -join "`n"
$state = [ordered]@{
    finished_at = (Get-Date).ToString('o')
    started_at  = $logIso
    full        = [bool]$isFull
    exit_code   = $code
    ok          = ($code -eq 0)
    log         = $log
    tail        = $tail
}
$state | ConvertTo-Json -Depth 4 |
    Out-File -FilePath (Join-Path $root 'data\processed\logs\last_run.json') -Encoding utf8

# Retain the 30 most recent run logs.
Get-ChildItem $logDir -Filter 'update_*.log' |
    Sort-Object LastWriteTime -Descending |
    Select-Object -Skip 30 |
    Remove-Item -Force -ErrorAction SilentlyContinue

if ($code -ne 0) { Write-Error "update_season.py exited $code (see $log)" }
exit $code
