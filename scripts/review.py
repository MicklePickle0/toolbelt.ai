"""Open the rally review UI (Milestone 6) in the browser.

Reviews the event DB that find_rallies.py writes (paths.db_path): watch each rally, fix its
touches, set who won the point. Re-running the pipeline keeps reviewed rallies.

Usage:
  python scripts/review.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "src" / "setter_eyes" / "ui" / "review_app.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP),
                                      "--server.headless", "false"]))
