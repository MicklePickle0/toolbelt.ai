"""Score the pipeline's touches and rallies against hand labels (events.csv).

Needs data/cache/<id>/contacts.json and rallies.json (scripts/find_rallies.py) and
data/labels/<id>/events.csv (docs/labelling.md). Only the labelled rallies are judged.
Writes, for going through the misses and false alarms by eye:
  data/outputs/eval/<id>_touches.csv   every labelled and predicted touch, matched or not
  data/outputs/eval/<id>_rallies.csv   each labelled rally against its predicted rally
Settings: `evaluate:` in config/pipeline.yaml.

--oof scores contacts_oof.json / rallies_oof.json (find_rallies.py --oof): touches chosen
without training on them, the honest figure for a video the scorer was trained on.

Usage:
  python scripts/eval_contacts.py --video titan-rise_set1 [--oof]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from setter_eyes import config
from setter_eyes.contacts.evaluate import evaluate
from setter_eyes.labels import load_manifest
from setter_eyes.labels.loader import load_video_labels


def _clock(t: float | None) -> str:
    if t is None:
        return ""
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:05.2f}"


def main() -> int:
    cfg = config.load("pipeline")
    paths = cfg["paths"]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="manifest key")
    ap.add_argument("--manifest", type=Path, default=config.repo_path(paths["manifest_path"]))
    ap.add_argument("--oof", action="store_true", help="score the out-of-fold outputs")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    entry = manifest.video(args.video)
    if entry.events is None:
        print(f"no events.csv for {entry.key} (expected data/labels/{entry.key}/events.csv)")
        return 1
    vl = load_video_labels(manifest, entry)
    for issue in vl.issues:
        print(issue)
    if not vl.ok:
        print("fix the label errors above first (scripts/ingest_labels.py --check)")
        return 1

    model_path = config.repo_path(cfg["contacts"]["scorer"]["path"])
    if model_path.exists() and not args.oof:
        from setter_eyes.contacts.scorer import TouchScorer

        if entry.key in TouchScorer.load(model_path).trained_on:
            print(f"NOTE: the touch scorer was trained on {entry.key}, so these numbers are "
                  "optimistic; the honest figure is train_touch_scorer.py's validation, or a "
                  "labelled video it wasn't trained on")
    cache = config.repo_path(paths["cache_dir"]) / entry.key
    suffix = "_oof" if args.oof else ""
    c = json.loads((cache / f"contacts{suffix}.json").read_text(encoding="utf-8"))
    r = json.loads((cache / f"rallies{suffix}.json").read_text(encoding="utf-8"))
    ev = evaluate(vl.rallies, c["contacts"], r["rallies"], float(c["fps"]), cfg)

    out = config.repo_path(paths["output_dir"]) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{entry.key}_touches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["result", "rally", "label_time", "pred_time", "label_team", "pred_team",
                    "action", "pred_action", "pred_confidence", "pred_source"])
        for m in ev.touches:
            result = "miss" if m.pred_t is None else "false_alarm" if m.label_t is None else \
                "hit" if m.label_team == m.pred_team else "hit_wrong_team"
            w.writerow([result, m.rally, _clock(m.label_t), _clock(m.pred_t), m.label_team,
                        m.pred_team, m.action, m.pred_action, m.confidence, m.source])
    with open(out / f"{entry.key}_rallies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(ev.rallies[0])) if ev.rallies else [])
        w.writeheader()
        for m in ev.rallies:
            w.writerow(asdict(m))

    s = ev.summary()
    pct = lambda x: "-" if x is None else f"{x:.0%}"         # noqa: E731
    sec = lambda x: "-" if x is None else f"{x:.2f} s"       # noqa: E731
    print(f"touches: {s['touches_labelled']} labelled, {s['touches_predicted']} predicted; "
          f"recall {pct(s['touch_recall'])}, precision {pct(s['touch_precision'])}, "
          f"team right {pct(s['team_accuracy'])}, timing {sec(s['median_time_error_s'])}")
    print(f"recall by action: {s['recall_by_action']}")
    if s["action_accuracy"] is not None:
        print(f"actions right on found touches: {s['action_accuracy']:.0%}  "
              f"(right/labelled: {s['action_accuracy_by_label']})")
    print(f"rallies: {s['rallies_found']}/{s['rallies_labelled']} found, "
          f"{s['false_rallies']} false (no rally there), serve found in "
          f"{s['serves_found']} (serving team right in {s['serving_team_correct']}); "
          f"start error {sec(s['median_start_error_s'])}, end error "
          f"{sec(s['median_end_error_s'])}")
    for ra in ev.false_rallies:
        print(f"  false rally {ra['index']}: {_clock(ra['start_t'])}-{_clock(ra['end_t'])}")
    print(f"-> {out / (entry.key + '_touches.csv')}, {out / (entry.key + '_rallies.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
