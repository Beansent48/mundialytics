$ErrorActionPreference = "Stop"

if (!(Test-Path "scripts") -or !(Test-Path "src") -or !(Test-Path "requirements.txt")) {
    Write-Error "Ejecuta esto desde la raíz del proyecto, donde existen scripts/, src/ y requirements.txt"
    exit 1
}

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "full_advanced_pipeline_local_$RunId.log"
Start-Transcript -Path $LogFile -Append | Out-Null

$FailedSteps = New-Object System.Collections.Generic.List[string]

function Log-Step($Message) {
    Write-Host ""
    Write-Host "================================================================================"
    Write-Host $Message
    Write-Host "================================================================================"
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    $script:PythonExe = "python"
    $script:PythonArgs = @()
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $script:PythonExe = "py"
    $script:PythonArgs = @("-3")
} else {
    Write-Error "No encuentro python ni py en PATH."
    exit 1
}

function Invoke-Python {
    $baseArgs = @()
    if ($script:PythonArgs) { $baseArgs = $script:PythonArgs }
    & $script:PythonExe @baseArgs @args
}

function Invoke-Required($Name, [scriptblock]$Command) {
    Log-Step "REQUIRED: $Name"
    $global:LASTEXITCODE = 0
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Required step failed: $Name with exit code $LASTEXITCODE"
    }
}

function Invoke-Optional($Name, [scriptblock]$Command) {
    Log-Step "OPTIONAL: $Name"
    try {
        $global:LASTEXITCODE = 0
        & $Command
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARN: $Name failed with exit code $LASTEXITCODE. Continuing."
            $FailedSteps.Add($Name) | Out-Null
        } else {
            Write-Host "OK: $Name"
        }
    } catch {
        Write-Host "WARN: $Name failed. Continuing."
        Write-Host $_
        $FailedSteps.Add($Name) | Out-Null
    }
}

if (!(Test-Path ".venv")) {
    Invoke-Required "create virtualenv" { Invoke-Python -m venv .venv }
}

$script:PythonExe = Resolve-Path ".venv\Scripts\python.exe"
$script:PythonArgs = @()

Invoke-Required "upgrade packaging tools" { Invoke-Python -m pip install --upgrade pip wheel setuptools }
Invoke-Required "install requirements" { Invoke-Python -m pip install -r requirements.txt }
Invoke-Optional "install optional provider libs: soccerdata + kaggle" { Invoke-Python -m pip install "soccerdata>=1.8" kaggle }

Invoke-Required "compile source and scripts" { Invoke-Python -m compileall -q src scripts }
Invoke-Optional "run v0.50 smoke tests" { Invoke-Python -m pytest -q tests/test_v0500_advanced_football_data_layer.py }

$FdSeasons = @("2526","2425","2324","2223","2122","2021","1920","1819","1718")
$FdLeagues = @("E0","SP1","I1","D1","F1")

$FdExisting = @()
if (Test-Path "data\raw\football_data") {
    $FdExisting = @(Get-ChildItem "data\raw\football_data" -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue)
}

if ($FdExisting.Count -eq 0) {
    $args = @("scripts\download_football_data_stats.py","--seasons") + $FdSeasons + @("--mode","csv","--leagues") + $FdLeagues + @("--out-dir","data\raw\football_data","--sleep","0.60")
    Invoke-Optional "download Football-Data Big 5 CSVs" { Invoke-Python @args }
} else {
    Write-Host "OK: detected local Football-Data CSVs: $($FdExisting.Count)"
}

$FoundationMatches = New-Object System.Collections.Generic.List[string]

function Build-Foundation($Label, $Code, $Dataset) {
    $files = @()
    if (Test-Path "data\raw\football_data") {
        $files = @(Get-ChildItem "data\raw\football_data" -Recurse -Filter "*_$Code.csv" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    }

    if ($files.Count -eq 0) {
        Write-Host "WARN: no Football-Data files found for $Label ($Code)."
        return
    }

    $cmdArgs = @(
        "scripts\build_match_dataset.py",
        "--source","football-data-uk",
        "--inputs"
    ) + $files + @(
        "--out-dir","data\processed\$Dataset",
        "--dataset-name",$Dataset,
        "--drop-incomplete-goals",
        "--skip-bad-files"
    )

    Invoke-Optional "build foundation dataset: $Label" { Invoke-Python @cmdArgs }

    $out = "data\processed\$Dataset\canonical_matches.csv"
    if (Test-Path $out) {
        $FoundationMatches.Add($out) | Out-Null
    }
}

Build-Foundation "EPL" "E0" "foundation_epl_multi_season"
Build-Foundation "LaLiga" "SP1" "foundation_laliga_multi_season"
Build-Foundation "Serie A" "I1" "foundation_seriea_multi_season"
Build-Foundation "Bundesliga" "D1" "foundation_bundesliga_multi_season"
Build-Foundation "Ligue 1" "F1" "foundation_ligue1_multi_season"

if ($FoundationMatches.Count -gt 0) {
    Log-Step "Build combined Big 5 canonical matches"
    New-Item -ItemType Directory -Force -Path "data\processed\foundation_big5_multi_season" | Out-Null

    $CombinePy = @"
from pathlib import Path
import sys
import pandas as pd

paths = [Path(p) for p in sys.argv[1:]]
frames = []
for p in paths:
    if p.exists() and p.stat().st_size > 0:
        df = pd.read_csv(p)
        df["foundation_source_file"] = str(p)
        frames.append(df)

if not frames:
    raise SystemExit("No canonical matches to combine.")

out_dir = Path("data/processed/foundation_big5_multi_season")
out_dir.mkdir(parents=True, exist_ok=True)
combined = pd.concat(frames, ignore_index=True, sort=False)

dedupe_cols = [c for c in ["match_id", "date", "home_team", "away_team", "competition", "season"] if c in combined.columns]
if dedupe_cols:
    combined = combined.drop_duplicates(subset=dedupe_cols)

combined.to_csv(out_dir / "canonical_matches.csv", index=False)
print(f"combined_rows={len(combined)}")
print(f"output={out_dir / 'canonical_matches.csv'}")
"@

    $TmpCombine = "outputs\logs\_combine_big5_$RunId.py"
    $CombinePy | Set-Content -Path $TmpCombine -Encoding UTF8
    $combineArgs = @($TmpCombine) + $FoundationMatches.ToArray()
    Invoke-Python @combineArgs

    $registryArgs = @("scripts\build_team_registry.py","--matches") + $FoundationMatches.ToArray() + @("--out-dir","data\processed\entities","--dataset-name","team_registry")
    Invoke-Optional "build Big 5 team registry" { Invoke-Python @registryArgs }

    Invoke-Optional "download or refresh ClubElo team histories" {
        Invoke-Python scripts\download_clubelo.py `
            --matches data\processed\foundation_big5_multi_season\canonical_matches.csv `
            --registry data\processed\entities\team_registry.csv `
            --out-dir data\external\clubelo `
            --mode team-history
    }

    foreach ($matches in $FoundationMatches) {
        $dataset = Split-Path (Split-Path $matches -Parent) -Leaf
        Invoke-Optional "enrich with ClubElo: $dataset" {
            Invoke-Python scripts\enrich_matches_with_clubelo.py `
                --matches $matches `
                --registry data\processed\entities\team_registry.csv `
                --clubelo-dir data\external\clubelo `
                --out-dir "data\processed\enriched\${dataset}_clubelo" `
                --dataset-name "${dataset}_clubelo" `
                --source-mode team-history
        }
    }

    Invoke-Optional "enrich combined Big 5 with ClubElo" {
        Invoke-Python scripts\enrich_matches_with_clubelo.py `
            --matches data\processed\foundation_big5_multi_season\canonical_matches.csv `
            --registry data\processed\entities\team_registry.csv `
            --clubelo-dir data\external\clubelo `
            --out-dir data\processed\enriched\foundation_big5_multi_season_clubelo `
            --dataset-name foundation_big5_multi_season_clubelo `
            --source-mode team-history
    }
} else {
    Write-Host "WARN: no foundation matches were built."
}

if (Test-Path "data\raw\statsbomb\open-data\data\competitions.json") {
    Write-Host "OK: detected local StatsBomb Open Data."
} else {
    Invoke-Optional "setup StatsBomb Open Data" {
        Invoke-Python scripts\setup_statsbomb_open_data.py `
            --out data\raw\statsbomb\open-data\data `
            --keep-zip data\raw\statsbomb\statsbomb_open_data_master.zip
    }
}

$sbArgs = @("scripts\import_statsbomb_open_advanced.py","--data-dir","data\raw\statsbomb\open-data\data","--out-dir","data\external\advanced\statsbomb")
if ($env:STATSBOMB_MAX_MATCHES) {
    $sbArgs += @("--max-matches",$env:STATSBOMB_MAX_MATCHES)
}
Invoke-Optional "import StatsBomb Open Data advanced stats" { Invoke-Python @sbArgs }

$FbrefExisting = @()
if (Test-Path "data\external\advanced\fbref") {
    $FbrefExisting = @(Get-ChildItem "data\external\advanced\fbref" -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue)
}

if ($FbrefExisting.Count -eq 0) {
    $fbrefLeagues = @("ENG-Premier League","ESP-La Liga","ITA-Serie A","GER-Bundesliga","FRA-Ligue 1")
    $fbrefSeasons = @("2021","2022","2023","2024","2025")
    $fbrefStatTypes = @("schedule","shooting","passing","passing_types","gca","defense","possession","misc","keeper","keeper_adv")
    $fbrefArgs = @("scripts\download_fbref_advanced.py","--league") + $fbrefLeagues + @("--season") + $fbrefSeasons + @("--stat-type") + $fbrefStatTypes + @("--out-dir","data\external\advanced\fbref\raw")
    Invoke-Optional "download FBref/soccerdata advanced Big 5" { Invoke-Python @fbrefArgs }
} else {
    Write-Host "OK: detected local FBref CSVs: $($FbrefExisting.Count)"
}

$fbrefRaw = @()
if (Test-Path "data\external\advanced\fbref") {
    $fbrefRaw = @(Get-ChildItem "data\external\advanced\fbref" -Recurse -Filter "fbref_team_match_stats_*.csv" -ErrorAction SilentlyContinue)
}

foreach ($csv in $fbrefRaw) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($csv.Name)
    $statType = $base.Replace("fbref_team_match_stats_","")
    Invoke-Optional "normalize FBref provider CSV: $statType" {
        Invoke-Python scripts\import_advanced_csv.py `
            --input $csv.FullName `
            --provider "fbref_$statType" `
            --out-dir "data\external\advanced\fbref\normalized\$statType"
    }
}

New-Item -ItemType Directory -Force -Path "data\external\advanced\kaggle_understat\raw" | Out-Null

$FindKagglePy = @"
from pathlib import Path
import os
import pandas as pd

explicit = os.environ.get("KAGGLE_UNDERSTAT_CSV", "").strip()
if explicit and Path(explicit).exists():
    print(explicit)
    raise SystemExit(0)

roots = [
    Path("data/external/advanced/kaggle_understat"),
    Path("data/raw/kaggle_understat"),
    Path("data/raw/understat"),
    Path("data/external/understat"),
]

candidates = []
for root in roots:
    if root.exists():
        candidates.extend(root.rglob("*.csv"))

best = None
best_score = -1

for p in candidates:
    try:
        df = pd.read_csv(p, nrows=20)
    except Exception:
        continue

    cols = {str(c).lower().strip() for c in df.columns}
    norm = {"".join(ch for ch in c if ch.isalnum()) for c in cols}

    score = 0
    if {"date", "match_date", "datetime"} & cols:
        score += 2
    if {"home_team", "home", "h_team", "hometeam"} & cols or {"hometeam", "hteam"} & norm:
        score += 2
    if {"away_team", "away", "a_team", "awayteam"} & cols or {"awayteam", "ateam"} & norm:
        score += 2
    if {"home_xg", "hxg", "h_xg", "xg_home"} & cols or {"homexg", "hxg", "xghome"} & norm:
        score += 3
    if {"away_xg", "axg", "a_xg", "xg_away"} & cols or {"awayxg", "axg", "xgaway"} & norm:
        score += 3
    if {"player", "player_name"} & cols:
        score -= 3

    if score > best_score:
        best_score = score
        best = p

if best is not None and best_score >= 5:
    print(best)
"@

$TmpFindKaggle = "outputs\logs\_find_kaggle_understat_$RunId.py"
$FindKagglePy | Set-Content -Path $TmpFindKaggle -Encoding UTF8
$KaggleCsv = (& $script:PythonExe $TmpFindKaggle | Select-Object -First 1)

if ($KaggleCsv -and (Test-Path $KaggleCsv)) {
    Invoke-Optional "import Kaggle Understat CSV: $KaggleCsv" {
        Invoke-Python scripts\import_kaggle_understat.py `
            --input $KaggleCsv `
            --out-dir data\external\advanced\kaggle_understat
    }
} else {
    Write-Host "WARN: no usable local Kaggle Understat match CSV detected."
    Write-Host "TIP: rerun with `$env:KAGGLE_UNDERSTAT_CSV='C:\path\to\games.csv'"
}

$SourceArgs = New-Object System.Collections.Generic.List[string]
$PriorityArgs = New-Object System.Collections.Generic.List[string]

function Add-Source($Provider, $Path) {
    if (Test-Path $Path) {
        $item = Get-Item $Path
        if ($item.Length -gt 0) {
            $SourceArgs.Add("${Provider}=$Path") | Out-Null
            $PriorityArgs.Add($Provider) | Out-Null
            Write-Host "Source ready: $Provider -> $Path"
            return
        }
    }
    Write-Host "Source missing/empty: $Provider -> $Path"
}

Add-Source "kaggle_understat" "data\external\advanced\kaggle_understat\kaggle_understat_advanced_match_stats.csv"

if (Test-Path "data\external\advanced\fbref\normalized") {
    Get-ChildItem "data\external\advanced\fbref\normalized" -Recurse -Filter "fbref_*_advanced_match_stats.csv" -ErrorAction SilentlyContinue | ForEach-Object {
        $provider = "fbref_" + (Split-Path $_.DirectoryName -Leaf)
        Add-Source $provider $_.FullName
    }
}

Add-Source "statsbomb_open_data" "data\external\advanced\statsbomb\statsbomb_advanced_match_stats.csv"

if ($SourceArgs.Count -gt 0) {
    $mergeArgs = @("scripts\merge_advanced_sources.py","--source") + $SourceArgs.ToArray() + @("--provider-priority") + $PriorityArgs.ToArray() + @("provider_csv","--out-dir","data\external\advanced\canonical")
    Invoke-Optional "merge advanced sources by provider priority" { Invoke-Python @mergeArgs }
} else {
    Write-Host "WARN: no advanced source files available; skipping merge/enrichment."
}

$AdvancedCanonical = "data\external\advanced\canonical\canonical_advanced_match_stats.csv"

if ((Test-Path $AdvancedCanonical) -and $FoundationMatches.Count -gt 0) {
    foreach ($matches in $FoundationMatches) {
        $dataset = Split-Path (Split-Path $matches -Parent) -Leaf
        $clubeloMatches = "data\processed\enriched\${dataset}_clubelo\canonical_matches_with_clubelo.csv"
        $inputMatches = $matches
        if (Test-Path $clubeloMatches) {
            $inputMatches = $clubeloMatches
        }

        Invoke-Optional "enrich canonical matches with advanced stats: $dataset" {
            Invoke-Python scripts\enrich_matches_with_advanced_stats.py `
                --matches $inputMatches `
                --advanced $AdvancedCanonical `
                --registry data\processed\entities\team_registry.csv `
                --provider-alias-column football_data_name `
                --out-dir "data\processed\enriched\${dataset}_advanced" `
                --dataset-name "${dataset}_advanced"
        }

        $enriched = "data\processed\enriched\${dataset}_advanced\canonical_matches_with_advanced_stats.csv"
        if (Test-Path $enriched) {
            Invoke-Optional "audit advanced coverage: $dataset" {
                Invoke-Python scripts\audit_advanced_data_coverage.py `
                    --matches $enriched `
                    --out-dir "data\processed\enriched\${dataset}_advanced" `
                    --dataset-name "${dataset}_advanced"
            }

            Invoke-Optional "build model-ready snapshots: $dataset" {
                Invoke-Python scripts\build_model_ready_dataset.py `
                    --matches $enriched `
                    --out-dir "data\processed\model_ready\${dataset}_advanced" `
                    --dataset-name "${dataset}_advanced"
            }
        }
    }

    $combinedInput = "data\processed\enriched\foundation_big5_multi_season_clubelo\canonical_matches_with_clubelo.csv"
    if (!(Test-Path $combinedInput)) {
        $combinedInput = "data\processed\foundation_big5_multi_season\canonical_matches.csv"
    }

    if (Test-Path $combinedInput) {
        Invoke-Optional "enrich combined Big 5 with advanced stats" {
            Invoke-Python scripts\enrich_matches_with_advanced_stats.py `
                --matches $combinedInput `
                --advanced $AdvancedCanonical `
                --registry data\processed\entities\team_registry.csv `
                --provider-alias-column football_data_name `
                --out-dir data\processed\enriched\foundation_big5_multi_season_advanced `
                --dataset-name foundation_big5_multi_season_advanced
        }

        $combinedEnriched = "data\processed\enriched\foundation_big5_multi_season_advanced\canonical_matches_with_advanced_stats.csv"

        if (Test-Path $combinedEnriched) {
            Invoke-Optional "audit advanced coverage: combined Big 5" {
                Invoke-Python scripts\audit_advanced_data_coverage.py `
                    --matches $combinedEnriched `
                    --out-dir data\processed\enriched\foundation_big5_multi_season_advanced `
                    --dataset-name foundation_big5_multi_season_advanced
            }

            Invoke-Optional "build model-ready snapshots: combined Big 5" {
                Invoke-Python scripts\build_model_ready_dataset.py `
                    --matches $combinedEnriched `
                    --out-dir data\processed\model_ready\foundation_big5_multi_season_advanced `
                    --dataset-name foundation_big5_multi_season_advanced
            }
        }
    }
} else {
    Write-Host "WARN: skipping advanced enrichment/model-ready snapshots because advanced canonical data or foundation matches are missing."
}

$Snapshots = "data\processed\model_ready\foundation_big5_multi_season_advanced\model_ready_match_snapshots.csv"

if (Test-Path $Snapshots) {
    Invoke-Optional "run rolling model lab on advanced model-ready snapshots" {
        Invoke-Python scripts\run_rolling_model_lab.py `
            --historical-events $Snapshots `
            --out-dir outputs\rolling_model_lab_foundation_big5_advanced `
            --clean-out-dir `
            --n-trials 14 `
            --min-train-matches 900 `
            --calibration-matches 500 `
            --test-matches 250 `
            --step-matches 250 `
            --max-folds 6
    }
} else {
    Write-Host "WARN: model-ready snapshots not found; skipping rolling model lab."
}

Log-Step "Pipeline summary"
Write-Host "Log file: $LogFile"

Write-Host ""
Write-Host "Main outputs:"
$roots = @("data\external\advanced","data\processed\enriched","data\processed\model_ready","outputs")
foreach ($root in $roots) {
    if (Test-Path $root) {
        Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in @(".json",".csv",".html") } |
            Sort-Object FullName |
            Select-Object -Last 160 |
            ForEach-Object { Write-Host $_.FullName }
    }
}

if ($FailedSteps.Count -gt 0) {
    Write-Host ""
    Write-Host "Optional steps that failed or were skipped:"
    foreach ($step in $FailedSteps) {
        Write-Host " - $step"
    }
}

Write-Host ""
Write-Host "DONE. Revisa los coverage reports antes de confiar en los modelos."
Stop-Transcript | Out-Null
