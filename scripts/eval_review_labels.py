"""Pipeline jerseys / pass scores (cache contacts.json) vs the owner's confirmed ones in
the event DB (confidence 1). Args: video key."""
import json
import sqlite3
import sys
from collections import Counter

import numpy as np

key = sys.argv[1]
c = sqlite3.connect("data/setter_eyes.db")
truth = c.execute(
    "SELECT c.t, c.action, p.jersey_number, c.player_confidence, c.pass_score, "
    "c.pass_confidence FROM contacts c JOIN rallies r ON c.rally_id = r.id JOIN sets s ON "
    "s.id = r.set_id JOIN games g ON g.id = s.game_id LEFT JOIN players p ON p.id = c.player_id "
    "WHERE g.video_key = ? AND c.team = 'us' AND r.reviewed = 1", (key,)).fetchall()
pipe = [x for x in json.load(open(f"data/cache/{key}/contacts.json"))["contacts"]
        if x["kind"] == "touch"]
times = np.array([x["t"] for x in pipe])
J = Counter()
conf_rows = []
P = []
for t, action, jersey, pconf, pscore, sconf in truth:
    k = int(np.argmin(np.abs(times - t))) if len(times) else None
    m = pipe[k] if k is not None and abs(times[k] - t) <= 0.15 else None
    if jersey is not None and pconf is not None and pconf >= 1:
        J["labelled"] += 1
        if m is None:
            J["no pipeline touch there"] += 1
        elif m.get("jersey") is None:
            J["pipeline left it unnamed"] += 1
        else:
            J["named"] += 1
            J["named right"] += m["jersey"] == jersey
            conf_rows.append((m["player_confidence"], m["jersey"] == jersey, jersey, m["jersey"]))
    if pscore is not None and sconf is not None and sconf >= 1:
        P.append((pscore, m.get("pass_score") if m else None, action))
print(f"{key} jerseys: {dict(J)}")
if J["named"]:
    print(f"  named {J['named']}/{J['labelled']} ({J['named'] / J['labelled']:.0%}), "
          f"right {J['named right']}/{J['named']} ({J['named right'] / J['named']:.0%})")
    wrong = Counter((a, b) for _, ok, a, b in conf_rows if not ok)
    print("  wrong (true -> given):", dict(wrong))
    for thr in (0.6, 0.7, 0.8):
        hi = [r for r in conf_rows if r[0] >= thr]
        if hi:
            print(f"  confidence >= {thr}: {len(hi)} named, right {np.mean([r[1] for r in hi]):.0%}")
scored = [(a, b) for a, b, _ in P if b is not None]
print(f"passes: {len(P)} confirmed, pipeline scored {len(scored)}")
if scored:
    a, b = np.array(scored).T
    print(f"  exact {np.mean(a == b):.0%}, within 1 {np.mean(abs(a - b) <= 1):.0%}; "
          f"mean owner {a.mean():.2f} vs pipeline {b.mean():.2f}; pairs {sorted(Counter(zip(a.tolist(), b.tolist())).items())}")
print("  confirmed pass actions:", dict(Counter(x[2] for x in P)))
