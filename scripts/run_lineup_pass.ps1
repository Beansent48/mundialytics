<#
.SYNOPSIS
    Pre-kickoff lineup pass, for Windows Task Scheduler (every ~10 minutes).

.DESCRIPTION
    Runs scripts/log_lineup_pass.py: any Big Five match kicking off within the next
    75 minutes whose two starting XIs are out on ESPN is re-priced with the
    regulars-missing tilt and logged once to data/processed/logs/lineup_pass_log.csv.
    Most runs find nothing to do and exit in a couple of seconds (one scoreboard
    call per league). Output is appended to data/processed/logs/runs/lineup_pass_<date>.log;
    the last 14 days are kept.

    Same process plumbing as run_update.ps1 (Start-Process with OS-level
    redirection, WaitForExit on the process itself) for the same reasons.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_lineup_pass.ps1
#>
$ErrorActionPreference = 'Stop'
$root   = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'scripts\log_lineup_pass.py'
if (-not (Test-Path $python)) { throw "venv python not found at $python" }

$logDir = Join-Path $root 'data\processed\logs\runs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$day  = Get-Date -Format 'yyyy-MM-dd'
$log  = Join-Path $logDir "lineup_pass_$day.log"
$tmp  = Join-Path $logDir "lineup_pass_last.out"
$err  = Join-Path $logDir "lineup_pass_last.err"

$proc = Start-Process -FilePath $python -ArgumentList @($script) -NoNewWindow -PassThru `
    -RedirectStandardOutput $tmp -RedirectStandardError $err
$null = $proc.Handle
$proc.WaitForExit()
$code = $proc.ExitCode

Add-Content -Path $log -Value ("--- {0} exit {1}" -f (Get-Date).ToString('s'), $code)
Get-Content $tmp -ErrorAction SilentlyContinue | Add-Content -Path $log
if ($code -ne 0) { Get-Content $err -Tail 20 -ErrorAction SilentlyContinue | Add-Content -Path $log }

Get-ChildItem $logDir -Filter 'lineup_pass_2*.log' |
    Sort-Object LastWriteTime -Descending | Select-Object -Skip 14 |
    Remove-Item -Force -ErrorAction SilentlyContinue
exit $code
