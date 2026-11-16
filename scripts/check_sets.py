"""Score and stat sanity checks over every set in an event DB (roadmap E2; rally/sanity.py).

Prints, per set: the match format and set number (match.py, data/match_formats.yaml), the
score from the point winners, whether it is a legal set end, and how many rallies each
check flags (reviewed rallies separately: a flag there means a review disagrees with
another). CPU only, from the DB; nothing is changed.

Usage:
  python scripts/check_sets.py                       # every set in the pipeline DB
  python scripts/check_sets.py --video pakmen-black_set2 --rallies
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter

from setter_eyes import config, match
from setter_eyes import folders as fold
from setter_eyes.rally.sanity import check_set, load_set


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=None, help="event DB (default paths.db_path)")
    ap.add_argument("--video", action="append", help="only these videos")
    ap.add_argument("--rallies", action="store_true", help="list each flagged rally")
    args = ap.parse_args()
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    path = config.repo_path(args.db or paths["db_path"])
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)   # read only
    conn.row_factory = sqlite3.Row
    formats = match.load(config.repo_path(paths["match_formats_path"]))
    folders = fold.load(config.repo_path(paths["folders_path"]))
    sets = conn.execute("SELECT s.id, g.video_key FROM sets s JOIN games g ON s.game_id = g.id "
                        "ORDER BY g.video_key").fetchall()
    total = Counter()
    for set_id, key in sets:
        if args.video and key not in args.video:
            continue
        game = fold.folder_of(key, folders)["game"]
        fmt = match.format_for(game, formats, cfg)
        n = match.set_number(key, formats)
        rallies = load_set(conn, set_id)
        res = check_set(rallies, fmt, n, cfg)
        by_code, rev_code = Counter(), Counter()
        reviewed = {r["id"] for r in rallies if r["reviewed"]}
        for rid, c in res["rallies"].items():
            for f in c["flags"]:
                (rev_code if rid in reviewed else by_code)[f["code"]] += 1
        total.update(by_code)
        total["sets"] += 1
        total[f"status {res['status']}"] += 1
        print(f"\n{key}  (set {n}, {fmt.label()})  {len(rallies)} rallies, {len(reviewed)} "
              f"reviewed")
        print(f"  {res['status']}: {res['message']}")
        print(f"  flagged unreviewed: {len([r for r in res['flagged'] if r not in reviewed])}"
              f"  {dict(by_code.most_common())}")
        if rev_code:
            print(f"  flagged reviewed: {len(res['reviewed_flagged'])}  {dict(rev_code)}")
        if args.rallies:
            num = {r["id"]: r for r in rallies}
            for rid in res["flagged"]:
                r = num[rid]
                c = res["rallies"][rid]
                print(f"    #{r['rally_number']:>3} {r['start_t']:7.1f}s "
                      f"{'reviewed ' if r['reviewed'] else ''}priority {c['priority']:.2f}: "
                      + " | ".join(f["text"] for f in c["flags"]))
    print("\nall sets:", dict(total.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
