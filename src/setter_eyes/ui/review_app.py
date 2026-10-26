"""Review UI (Milestone 6), Streamlit. Run: python scripts/review.py

Pick a video, go through its rallies (the review queue puts the least confident first),
watch each rally's clip, fix its touches, set who won the point, save. Saving writes to
the event DB (paths.db_path) and logs every change as a correction (ui/review.py).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from setter_eyes import config, db
from setter_eyes.labels import load_manifest
from setter_eyes.labels.event_editor import clock
from setter_eyes.labels.events import parse_time
from setter_eyes.ui import review as rv

PAD_BEFORE, PAD_AFTER = 2.0, 2.0          # clip margin around the rally, seconds
WINNERS = {"Us": "us", "Them": "them", "Replay (no point)": rv.REPLAY,
           "Not a rally": rv.NOT_A_RALLY}

cfg = config.load("pipeline")
paths = cfg["paths"]
ACTIONS = config.load("labels")["actions"]
OUTCOMES = config.load("labels")["outcomes"]

st.set_page_config(page_title="Rally review", layout="wide")



@st.cache_resource
def manifest():
    return load_manifest(config.repo_path(paths["manifest_path"]))


# A fresh connection per run: Streamlit runs each interaction in a new thread, and a
# SQLite connection can't be shared between threads.
conn = db.connect(config.repo_path(paths["db_path"]))
games = rv.games(conn)
if not games:
    st.error("The event DB has no videos yet: run scripts/find_rallies.py first.")
    st.stop()

# ---- sidebar: video, queue, rally ---------------------------------------------------------
with st.sidebar:
    st.title("Rally review")
    g = st.selectbox("Video", games, format_func=lambda x: f"{x['video_key']} "
                     f"({x['reviewed']}/{x['rallies']} reviewed)")
    rows = rv.rallies(conn, g["set_id"])
    us, them = rv.score(rows)
    st.metric("Score so far (reviewed rallies)", f"{us} - {them}")
    st.progress(sum(r["reviewed"] for r in rows) / max(1, len(rows)))
    mode = st.radio("Show", ["Review queue (least confident first)", "All rallies in order"])
    shown = rv.review_queue(rows) if mode.startswith("Review") else rows
    if not shown:
        st.success("Every rally of this video is reviewed.")
        st.stop()
    key = f"pos_{g['set_id']}_{mode}"
    st.session_state.setdefault(key, 0)
    st.session_state[key] = min(st.session_state[key], len(shown) - 1)

    def label(r):
        done = "✓ " if r["reviewed"] else ""
        return (f"{done}#{r['rally_number']}  {clock(r['start_t'])}  "
                f"conf {r['segment_confidence'] or 0:.2f}")

    pick = st.selectbox("Rally", range(len(shown)), index=st.session_state[key],
                        format_func=lambda i: label(shown[i]))
    st.session_state[key] = pick
    rally = shown[pick]

# ---- main: clip and touches ----------------------------------------------------------------
entry = manifest().video(g["video_key"])
fps = g["fps"]
t0 = max(0.0, rally["start_t"] - PAD_BEFORE)
t1 = rally["end_t"] + PAD_AFTER
proxy = config.repo_path(paths["cache_dir"]) / entry.key / "proxy.mp4"
src = proxy if proxy.exists() else entry.path
clip_file = config.repo_path(paths["cache_dir"]) / entry.key / "clips" / \
    f"rally_{rally['id']}_{t0:.2f}_{t1:.2f}.mp4"

st.subheader(f"Rally {rally['rally_number']}: {clock(rally['start_t'])} - "
             f"{clock(rally['end_t'])}  ({rally['end_t'] - rally['start_t']:.1f} s)")
left, right = st.columns([3, 2])
with left:
    try:
        with st.spinner("cutting the clip..."):
            st.video(str(rv.rally_clip(src, clip_file, t0, t1)))
    except Exception as e:                       # noqa: BLE001 - show, don't crash the UI
        st.error(f"could not cut the clip: {e}")
    st.caption(f"The clip starts at {clock(t0)} (2 s before the rally). Times in the table "
               "are seconds into the clip, as the player shows them.")

with right:
    st.markdown("**Touches** (edit team or action, delete rows, add rows for missed touches)")
    tlist = rv.touches(conn, rally["id"])
    table = pd.DataFrame([{"id": x["id"], "clip_s": round(x["t"] - t0, 2), "team": x["team"],
                           "action": x["action"],
                           "sure": round(x["action_confidence"] or 0.0, 2)} for x in tlist],
                         columns=["id", "clip_s", "team", "action", "sure"])
    edited = st.data_editor(
        table, num_rows="dynamic", hide_index=True, key=f"touches_{rally['id']}",
        column_config={
            "id": None,
            "clip_s": st.column_config.NumberColumn("time in clip (s)", step=0.01,
                                                    format="%.2f", required=True),
            "team": st.column_config.SelectboxColumn("team", options=["us", "them"],
                                                     required=True),
            "action": st.column_config.SelectboxColumn("action", options=ACTIONS),
            "sure": st.column_config.NumberColumn("pipeline sure", disabled=True),
        })
    kept_ids = {int(i) for i in edited["id"].dropna()}
    edits = [rv.TouchEdit(None if pd.isna(r.id) else int(r.id), float(r.clip_s) + t0,
                          r.team, None if pd.isna(r.action) else r.action)
             for r in edited.itertuples() if not pd.isna(r.clip_s)]
    edits += [rv.TouchEdit(x["id"], x["t"], x["team"], x["action"], delete=True)
              for x in tlist if x["id"] not in kept_ids]

    end_text = st.text_input("Rally ended at (m:ss.ss; touches after it are removed, e.g. "
                             "after a net touch or a caught ball)", clock(rally["end_t"]),
                             key=f"end_{rally['id']}")
    try:                                         # untouched: keep the exact time
        end_t: float | None = rally["end_t"] if end_text == clock(rally["end_t"]) else \
            parse_time(end_text)
    except ValueError:
        end_t = None
        st.warning(f"Rally end {end_text!r} isn't a time like 1:02.25.")

    st.markdown("**Point**")
    current = {v: k for k, v in WINNERS.items()}.get(
        rally["point_winner"] or (rally["outcome"] if rally["outcome"] in
                                  (rv.REPLAY, rv.NOT_A_RALLY) else None))
    choice = st.radio("Who won the point?", list(WINNERS), horizontal=True,
                      index=list(WINNERS).index(current) if current else None,
                      key=f"winner_{rally['id']}")
    winner = WINNERS.get(choice) if choice else None
    suggested = rv.suggest_outcome(edits, winner, end_t)
    outcome = None
    if winner in ("us", "them"):
        default = rally["outcome"] if rally["reviewed"] and rally["outcome"] else suggested
        outcome = st.selectbox(f"How (suggested from the last touch: {suggested})",
                               [o for o in OUTCOMES if o != rv.REPLAY],
                               index=[o for o in OUTCOMES if o != rv.REPLAY].index(default)
                               if default in OUTCOMES else 0, key=f"outcome_{rally['id']}")
    serving = st.radio("Serving team", ["us", "them"], horizontal=True,
                       index=["us", "them"].index(rally["serving_team"])
                       if rally["serving_team"] in ("us", "them") else None,
                       key=f"serve_{rally['id']}")
    checked = st.checkbox("All touches above are right (use them as training data)",
                          value=True, key=f"checked_{rally['id']}",
                          disabled=winner == rv.NOT_A_RALLY)

    def save() -> bool:
        if winner is None:
            st.warning("Set who won the point first.")
            return False
        if end_t is None:
            return False
        rv.save_review(conn, rally["id"], fps, winner, outcome, serving, end_t, edits,
                       checked)
        return True

    a, b = st.columns(2)
    if a.button("Save", use_container_width=True) and save():
        st.success("Saved.")
        st.rerun()
    if b.button("Save & next", type="primary", use_container_width=True) and save():
        if not mode.startswith("Review"):
            st.session_state[key] = min(pick + 1, len(shown) - 1)
        st.rerun()
