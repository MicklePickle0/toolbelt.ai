"""Pipeline jerseys and pass scores vs the ones the owner confirmed in review.

Truth: our touches in reviewed rallies of the event DB whose player / pass score has
confidence 1 (set or ticked in review). Pipeline: the cached contacts.json of the last
find_rallies.py run, the touch within 0.15 s. So a set reviewed in the web app measures
identity and pass scoring without an events.csv.

Usage:
  python scripts/eval_review_labels.py --video pakmen-black_set2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from setter_eyes import config, db


def main() -> int:
    paths = config.load("pipeline")["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    args = ap.parse_args()
    conn = db.connect(config.repo_path(paths["db_path"]))
    truth = conn.execute(
        "SELECT c.t, c.action, p.jersey_number, c.player_confidence, c.pass_score, "
        "c.pass_confidence FROM contacts c JOIN rallies r ON c.rally_id = r.id "
        "JOIN sets s ON s.id = r.set_id JOIN games g ON g.id = s.game_id "
        "LEFT JOIN players p ON p.id = c.player_id "
        "WHERE g.video_key = ? AND c.team = 'us' AND r.reviewed = 1", (args.video,)).fetchall()
    conn.close()
    cache = config.repo_path(paths["cache_dir"]) / args.video
    pipe = [x for x in json.loads((cache / "contacts.json").read_text(encoding="utf-8"))
            ["contacts"] if x["kind"] == "touch"]
    times = np.array([x["t"] for x in pipe])
    jerseys, named, passes = Counter(), [], []
    for t, _action, jersey, pconf, pscore, sconf in truth:
        k = int(np.argmin(np.abs(times - t))) if len(times) else -1
        m = pipe[k] if k >= 0 and abs(times[k] - t) <= 0.15 else None
        if jersey is not None and pconf is not None and pconf >= 1:
            jerseys["confirmed"] += 1
            if m is None:
                jerseys["no pipeline touch there"] += 1
            elif m.get("jersey") is None:
                jerseys["left unnamed"] += 1
            else:
                named.append((m["player_confidence"], jersey, m["jersey"]))
        if pscore is not None and sconf is not None and sconf >= 1:
            passes.append((pscore, m.get("pass_score") if m else None))
    print(f"{args.video}: {dict(jerseys)}")
    if named:
        right = [a == b for _, a, b in named]
        print(f"  jerseys named {len(named)}/{jerseys['confirmed']} "
              f"({len(named) / jerseys['confirmed']:.0%}), right {sum(right)} "
              f"({np.mean(right):.0%})")
        print("  wrong (true -> given):",
              dict(Counter((a, b) for _, a, b in named if a != b)))
        for thr in (0.6, 0.7, 0.8):
            hi = [a == b for c, a, b in named if c >= thr]
            if hi:
                print(f"  confidence >= {thr}: {len(hi)} named, right {np.mean(hi):.0%}")
    scored = [(a, b) for a, b in passes if b is not None]
    print(f"  passes: {len(passes)} confirmed, {len(scored)} scored by the pipeline")
    if scored:
        a, b = np.array(scored).T
        pairs = sorted(Counter(zip(a.tolist(), b.tolist(), strict=True)).items())
        print(f"  exact {np.mean(a == b):.0%}, within 1 {np.mean(abs(a - b) <= 1):.0%}; "
              f"mean owner {a.mean():.2f}, pipeline {b.mean():.2f}; owner->pipeline {pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
