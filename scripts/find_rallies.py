"""Find ball contacts, serves and rallies in a video (Milestone 4).

Needs the ball path and arcs (scripts/track_ball.py) and the people tracks and roles
(scripts/track_people.py). Writes
  data/cache/<id>/contacts.json   touches and floor contacts: time, team, toucher's track,
                                  ball position, evidence, confidence, serve flag
  data/cache/<id>/rallies.json    rallies: start (serve) and end (floor / ball lost),
                                  serving team, contacts, net crossings, confidence
Point winners are inferred from who serves next (the review UI overrides them); our serve
receives get a 0-4 pass score with a confidence (stats/pass_score.py, config/pass_scale.yaml).
Also writes rallies and touches (with actions) to the event DB (paths.db_path), unreviewed,
replacing what was there for this video; --no-db skips it.
Touches are chosen by the learned scorer when its model exists (contacts.scorer.path,
scripts/train_touch_scorer.py), else by the hand-tuned fusion in contacts/events.py.
Settings: `contacts:`, `serve:` and `rally:` in config/pipeline.yaml.
Check by eye: render_overlay.py --rallies.

--oof builds rallies from data/cache/<id>/contacts_oof.json instead (touches chosen
without training on them, from train_touch_scorer.py) and writes rallies_oof.json, for
an honest test of the rally logic with eval_contacts.py --oof.

Usage:
  python scripts/find_rallies.py --video titan-rise_set1 [--oof]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from setter_eyes import config, db
from setter_eyes import rosters as rost
from setter_eyes.ball.arcs import Arc
from setter_eyes.ball.tracker import BallPath
from setter_eyes.calibration import load as load_calibration
from setter_eyes.classify.rules import classify_rally, fix_overpasses, launch_up
from setter_eyes.contacts.blocks import add_stuff_blocks, blocks_from_landing
from setter_eyes.contacts.events import Contact, add_hidden_touches, find_contacts
from setter_eyes.contacts.scorer import TouchScorer, scored_contacts
from setter_eyes.db.pipeline import write_video
from setter_eyes.identity.numbers import counted_reads, load_numbers
from setter_eyes.identity.server import name_servers
from setter_eyes.identity.toucher import LiberoColour, Poses, TouchPicker, assign_players
from setter_eyes.labels import load_manifest
from setter_eyes.rally.live import PlayerMotion
from setter_eyes.rally.segment import (
    SERVE_FROM_FLIGHT,
    flag_gaps,
    merge_close,
    segment_from_touches,
    segment_rallies,
)
from setter_eyes.rally.serve_entry import find_our_serves
from setter_eyes.rally.winner import infer_winners
from setter_eyes.stats.pass_score import score_passes
from setter_eyes.track.people import PeopleTracks
from setter_eyes.track.roles import load_roles


def mark_our_serves(contacts: list, path, fps: float, key: str, cfg: dict,
                    paths: dict) -> None:
    """Our serves seen entering the frame (rally/serve_entry.py): the pipeline touch at
    that moment becomes the serve, or a serve touch is added when there was none."""
    cal = load_calibration(key, config.repo_path(paths["calibration_dir"]))
    if cal is None:
        return
    s = cfg["rally"]["our_serve"]
    touch_t = np.array([c.t for c in contacts if c.kind == "touch"])
    added = marked = 0
    for t in find_our_serves(path, fps, touch_t, cal.camera.height, s):
        near = [c for c in contacts if c.kind == "touch" and abs(c.t - t) <= s["match_touch_s"]]
        if near:
            c = min(near, key=lambda c: abs(c.t - t))
            c.entry_serve, c.team = True, "us"
            marked += 1
            continue
        k = int(np.clip(np.searchsorted(path.frames / fps, t), 0, len(path.frames) - 1))
        contacts.append(Contact(k, int(path.frames[k]), round(t, 3), "touch", "us", -1, None,
                                [round(float(v), 1) for v in path.uv[k]], 0.0,
                                float("inf"), s["confidence"], entry_serve=True))
        added += 1
    contacts.sort(key=lambda c: c.t)
    print(f"our serves from the ball entering the frame: {marked + added} "
          f"({marked} on a touch already found, {added} added)")


def identify(rallies: list, contacts: list, cache: Path, fps: float, cfg: dict,
             key: str, people: PeopleTracks) -> Counter:
    """Jerseys on our touches from scripts/pose_people.py's output, when it's there."""
    if not (cache / "pose.npz").exists():
        return Counter({"not posed yet (scripts/pose_people.py)": 1})
    id_cfg = cfg["identity"]
    poses = Poses.load(cache / "pose.npz")
    picker = TouchPicker(poses, fps, id_cfg["toucher"])
    numbers = libero = None
    liberos: list[int] = []
    reads = cache / "number_reads.npz"
    if reads.exists():
        numbers = load_numbers(reads, fps, id_cfg["numbers"], people)
        rosters_dir = config.repo_path(cfg["paths"]["rosters_dir"])
        roster = rost.load_all(rosters_dir).get(rost.load_assignments(rosters_dir).get(key))
        liberos = [p.number for p in roster.players if p.libero] if roster else []
        libero = LiberoColour(poses, *counted_reads(reads, id_cfg["numbers"]), liberos,
                              id_cfg["libero"])
        print("libero by jersey colour: " + (f"#{libero.number}" if libero.ok else
                                             "not used (no libero, few reads or same colour)"))
    how = assign_players(rallies, contacts, picker, numbers, id_cfg, libero, tuple(liberos))
    how.update(name_servers(rallies, contacts, people, numbers, fps, id_cfg, tuple(liberos)))
    return how


def main() -> int:
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--no-db", action="store_true", help="don't write the event DB")
    ap.add_argument("--oof", action="store_true",
                    help="rallies from contacts_oof.json (honest test of the rally logic)")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    entry = manifest.video(args.video)
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    path = BallPath.load(cache / "ball_path.npz")
    arcs_doc = json.loads((cache / "ball_arcs.json").read_text(encoding="utf-8"))
    arcs = [Arc(**a) for a in arcs_doc["arcs"]]
    fps = float(arcs_doc["fps"])
    # Pixel settings for this video (ball_arcs.json from before 1080p support: 4K).
    cfg = config.for_video(cfg, int(arcs_doc.get("width", 3840)))
    g = np.array([0.0, 0.0, -float(arcs_doc["gravity"])])
    people = PeopleTracks.load(cache / "people_tracks.npz")
    roles = load_roles(cache / "people_roles.json")

    model_path = config.repo_path(cfg["contacts"]["scorer"]["path"])
    suffix = "_oof" if args.oof else ""
    if args.oof:
        doc = json.loads((cache / "contacts_oof.json").read_text(encoding="utf-8"))
        contacts = [Contact.from_dict(c) for c in doc["contacts"]]
        print("touches: out-of-fold (contacts_oof.json)")
    elif model_path.exists():
        scorer = TouchScorer.load(model_path)
        print(f"touches: learned scorer (trained on {', '.join(scorer.trained_on)})")
        contacts = scored_contacts(path, arcs, g, fps, people, roles, cfg, scorer)
    else:
        print("touches: hand-tuned fusion (no learned scorer yet)")
        contacts = find_contacts(path, arcs, g, fps, people, roles, cfg)
    learned = args.oof or model_path.exists()
    if learned:
        mark_our_serves(contacts, path, fps, entry.key, cfg, paths)
        rallies = segment_from_touches(path, arcs, contacts, g, fps, cfg,
                                       PlayerMotion(people, fps))
        n = len(rallies)
        rallies = merge_close(rallies, contacts, cfg)    # one point cut in two
        print(f"merged {n - len(rallies)} rallies closer than {cfg['rally']['merge_gap_s']} s; "
              f"{flag_gaps(rallies, cfg)} after a gap over {cfg['rally']['missed_point_gap_s']} s "
              "(a point may be missing there)")
    else:
        rallies = segment_rallies(path, arcs, contacts, g, fps, cfg)
    hidden = []                                        # touches the timing says were hidden
    for r in rallies:                                  # actions, once the rally is known
        idx = [i for i in r.contacts if contacts[i].kind == "touch"]
        touches = [contacts[i] for i in idx]
        serve = contacts[r.serve_contact] if r.serve_contact is not None and \
            r.serve_contact >= 0 else None
        launches = [launch_up(c, arcs, cfg["classify"]) for c in touches]
        # A serve traced back from the ball's flight (not seen as a touch) starts the rally.
        serve_t = r.start_t if serve is None and r.serve_contact == SERVE_FROM_FLIGHT else None
        missing = []
        for c, (action, conf) in zip(touches, classify_rally(touches, serve, r.serving_team,
                                                             cfg["classify"], launches,
                                                             serve_t=serve_t,
                                                             missing_out=missing),
                                     strict=True):
            c.action, c.action_confidence = action, conf
        hidden += [(idx[k], action, team, t) for k, action, team, t in missing]
    n_hidden = add_hidden_touches(rallies, contacts, hidden, fps,
                                  cfg["classify"]["inferred_touch_confidence"])
    print(f"hidden touches added (a receive before a late first touch, a set in a long gap): "
          f"{n_hidden}; overpasses our side played again (now a receive or dig): "
          f"{fix_overpasses(rallies, contacts)}")
    infer_winners(rallies, contacts, cfg)              # who serves next won the point
    print(f"blocks from the attack landing on the attacker's side: "
          f"{blocks_from_landing(rallies, contacts, cfg)}")
    checks_path = cache / f"block_checks{suffix}.json"  # scripts/check_blocks.py (GPU)
    if checks_path.exists():
        checks = {c["frame"]: c for c in
                  json.loads(checks_path.read_text(encoding="utf-8"))["checks"]}
        print(f"stuff blocks added: {add_stuff_blocks(rallies, contacts, checks, cfg)} "
              f"({len(checks)} point-ending attacks checked)")
    passes = score_passes(rallies, contacts, path, people, arcs, g, fps,
                          config.load("pass_scale"))
    print(f"serve-receive pass scores: {dict(passes)}")
    print(f"who touched it (our touches): "
          f"{dict(identify(rallies, contacts, cache, fps, cfg, entry.key, people))}")
    (cache / f"contacts{suffix}.json").write_text(json.dumps(
        {"video": entry.key, "fps": fps, "contacts": [c.to_dict() for c in contacts]},
        indent=1), encoding="utf-8")
    (cache / f"rallies{suffix}.json").write_text(json.dumps(
        {"video": entry.key, "fps": fps, "rallies": [r.to_dict() for r in rallies]},
        indent=1), encoding="utf-8")

    kinds = Counter(c.kind for c in contacts)
    in_rally = sum(len(r.contacts) for r in rallies)
    serve_team = Counter(r.serving_team for r in rallies)
    lengths = np.array([r.end_t - r.start_t for r in rallies]) if rallies else np.zeros(1)
    review = cfg["review"]["default_min_confidence"]
    print(f"contacts: {dict(kinds)} ({sum(c.serve for c in contacts)} look like serves), "
          f"{in_rally} inside rallies")
    traced = sum(r.serve_contact == -2 for r in rallies)
    inferred = sum(r.serve_contact == -3 for r in rallies)
    print(f"rallies: {len(rallies)}; serves found for "
          f"{sum(r.serve_contact is not None for r in rallies)} ({traced} traced back from "
          f"flight, {inferred} inferred from the rally)"
          f" (us {serve_team.get('us', 0)}, them {serve_team.get('them', 0)}); "
          f"ended on the floor: {sum(r.end_reason == 'floor' for r in rallies)}")
    print(f"ended by the live-ball check (touches after the whistle dropped): "
          f"{sum(r.end_reason == 'dead_ball' for r in rallies)}")
    print(f"rally length s: median {np.median(lengths):.1f}, max {lengths.max():.1f}; "
          f"below review confidence {review}: {sum(r.confidence < review for r in rallies)}")
    print(f"-> {cache / f'contacts{suffix}.json'}, {cache / f'rallies{suffix}.json'}")
    if not args.no_db and not args.oof:
        conn = db.connect(config.repo_path(paths["db_path"]))
        w = write_video(conn, manifest, entry, fps, [r.to_dict() for r in rallies],
                        [c.to_dict() for c in contacts],
                        rosters_dir=config.repo_path(paths["rosters_dir"]))
        conn.close()
        print(f"event DB: {w.rallies} rallies, {w.contacts} touches written, {w.kept_reviewed} "
              f"reviewed rallies kept ({w.guesses} pass scores / players guessed on their "
              f"touches) -> {paths['db_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
