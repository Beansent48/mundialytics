import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# There is no quarantine list any more. The 21 legacy test files that could not
# be imported (they needed symbols stubbed out of the initial commit) were
# removed together with the modules they tested on 2026-09-17, so every file in
# tests/ is collected and an import break anywhere fails the run.
