"""The web app (FastAPI): videos and their processing queue, rally review and the stat sheet,
in a browser on the laptop or phone. Run: python scripts/serve_app.py

Everything reads and writes the event DB (paths.db_path), the same as the Streamlit review
(ui/review.py does the saving, so corrections are logged the same way). The page itself is
static/index.html + app.js; this module is the JSON API under /api.

Calibration is part of processing: the owner taps the court points on a frame in the
browser (calibration/solve.py, the same solve as scripts/calibrate.py's click UI). An
automatic fit only counts when `app.accept_auto_calibration` is on: on high-school gym
floors it is often wrong while looking sure, so by default every video is tapped.
"""

from __future__ import annotations

import base64
import shutil
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from setter_eyes import config, db
from setter_eyes import rosters as rost
from setter_eyes.calibration import court_model, draw_court, line_mask
from setter_eyes.calibration import store as cal_store
from setter_eyes.calibration.lines import LineScorer
from setter_eyes.calibration.manual import CLICKABLE_LINES, LINE_HINTS
from setter_eyes.calibration.solve import error_limit_px, manual_calibration, solve_manual
from setter_eyes.db.read import load_rallies
from setter_eyes.ingest import frame_filename, grab_frame, probe, write_image
from setter_eyes.labels import load_manifest
from setter_eyes.stats.sheet import COLUMNS, SHORT, stat_rows
from setter_eyes.ui import review as rv

from .jobs import JobQueue, last_line

STATIC = Path(__file__).with_name("static")
WINNERS = ("us", "them", rv.REPLAY, rv.NOT_A_RALLY)
MAX_CLIP_S = 180.0


@dataclass
class Settings:
    db_path: Path
    labels_db_path: Path
    manifest_path: Path
    calibration_dir: Path
    cache_dir: Path
    log_dir: Path
    steps: list[str]
    needs_calibration: list[str]
    clip_width: int = 1280
    clip_pad_s: float = 2.0
    actions: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    end_details: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    accept_auto_calibration: bool = False
    frame_width: int = 1920               # px, frames sent to the calibration page
    step_weights: dict = field(default_factory=dict)
    rosters_dir: Path | None = None
    missed_point_gap_s: float = 20.0      # a gap this long before a rally: a point may be missing

    @classmethod
    def from_config(cls) -> Settings:
        cfg = config.load("pipeline")
        p, a, labels = cfg["paths"], cfg["app"], config.load("labels")
        return cls(db_path=config.repo_path(p["db_path"]),
                   labels_db_path=config.repo_path(p["labels_db_path"]),
                   manifest_path=config.repo_path(p["manifest_path"]),
                   calibration_dir=config.repo_path(p["calibration_dir"]),
                   cache_dir=config.repo_path(p["cache_dir"]),
                   log_dir=config.repo_path(a["log_dir"]),
                   steps=a["steps"], needs_calibration=a["needs_calibration"],
                   clip_width=a["clip_width"], clip_pad_s=a["clip_pad_s"],
                   actions=labels["actions"], outcomes=labels["outcomes"],
                   end_details=labels.get("end_details", []),
                   output_dir=config.repo_path(p["output_dir"]),
                   accept_auto_calibration=a.get("accept_auto_calibration", False),
                   frame_width=a.get("calibration_frame_width", 1920),
                   step_weights=a.get("step_weights", {}),
                   rosters_dir=config.repo_path(p["rosters_dir"]),
                   missed_point_gap_s=cfg["rally"].get("missed_point_gap_s", 20.0))


class Touch(BaseModel):
    id: int | None = None
    t: float
    team: str | None = None
    action: str | None = None
    delete: bool = False


class Review(BaseModel):
    winner: str
    outcome: str | None = None
    serving_team: str | None = None
    start_t: float
    end_t: float
    touches: list[Touch]
    touches_checked: bool = True
    end_detail: str | None = None         # why it ended, when it wasn't the ball


class Suggest(BaseModel):
    winner: str | None = None
    end_t: float | None = None
    touches: list[Touch]


class Merge(BaseModel):
    other_id: int


class Split(BaseModel):
    t: float                              # seconds into the video: the second point's serve
    start_t: float | None = None          # the rally's start / end as edited on the page
    end_t: float | None = None            # (applied first, so a widened rally can be split)


class NewRally(BaseModel):
    start_t: float
    end_t: float
    serving_team: str | None = None


class Clicks(BaseModel):
    frame: int
    clicks: dict[str, list[float]]
    line_clicks: dict[str, list[list[float]]] = {}


class PlayerIn(BaseModel):
    number: int
    name: str
    height: str | float | None = None      # cm, or feet and inches (6'2")
    position: str | None = None
    libero: bool = False


class RosterIn(BaseModel):
    name: str
    season: str | None = None
    notes: str | None = None
    players: list[PlayerIn] = []


class RosterChoice(BaseModel):
    roster: str | None = None


class NewJob(BaseModel):
    video: str
    from_step: str | None = None


def _edits(touches: list[Touch]) -> list[rv.TouchEdit]:
    return [rv.TouchEdit(t.id, t.t, t.team, t.action, t.delete) for t in touches]


def create_app(settings: Settings | None = None, start_worker: bool = True) -> FastAPI:
    s = settings or Settings.from_config()
    app = FastAPI(title="Toolbelt")

    def usable(cal) -> bool:
        return cal is not None and (cal.method == "manual" or s.accept_auto_calibration)

    def calibration(video: str) -> dict | None:
        cal = cal_store.load(video, s.calibration_dir)
        if cal is None:
            return None
        extra = len(cal_store.load_segments(video, s.calibration_dir))
        return {"method": cal.method, "confidence": cal.confidence,
                "needs_review": cal.needs_review, "extra_segments": extra,
                "usable": usable(cal), "frame": cal.frame,
                "rms_px": cal.reprojection_rms_px}

    jobs = JobQueue(s.log_dir, s.steps, s.needs_calibration,
                    lambda v: usable(cal_store.load(v, s.calibration_dir)),
                    config.REPO_ROOT, step_weights=s.step_weights)
    app.state.jobs = jobs
    if start_worker:
        jobs.start()

    @contextmanager
    def conn(path: Path | None = None) -> Iterator[sqlite3.Connection]:
        c = db.connect(path or s.db_path)
        try:
            yield c
        finally:
            c.close()

    def sync_rosters() -> None:
        if s.rosters_dir is not None:
            with conn() as c:
                rost.sync_db(c, s.rosters_dir)

    try:
        sync_rosters()                          # the DB copy follows the files (e.g. after a pull)
    except rost.RosterError as e:
        print(f"rosters not synced: {e}")

    def manifest():
        return load_manifest(s.manifest_path)

    # ---- meta and videos -------------------------------------------------------------------
    @app.get("/api/meta")
    def meta() -> dict:
        m = manifest()
        return {"team": m.team, "actions": s.actions, "outcomes": s.outcomes,
                "steps": s.steps, "winners": list(WINNERS), "clip_pad_s": s.clip_pad_s,
                "missed_point_gap_s": s.missed_point_gap_s, "end_details": s.end_details,
                "roster": [{"number": r.number, "name": r.name} for r in m.roster]}

    @app.get("/api/videos")
    def videos() -> list[dict]:
        with conn() as c:
            in_db = {g["video_key"]: g for g in rv.games(c)}
        assigned = rost.load_assignments(s.rosters_dir) if s.rosters_dir else {}
        roster_names = ({k: r.name for k, r in rost.load_all(s.rosters_dir).items()}
                        if s.rosters_dir else {})
        latest = {}
        for j in jobs.snapshot():                       # newest first
            latest.setdefault(j["video"], j)
        out = []
        for v in manifest().videos:
            cache = s.cache_dir / v.key
            g = in_db.get(v.key)
            job = latest.get(v.key)
            out.append({
                "key": v.key, "venue": v.venue, "opponent": v.opponent,
                "played_on": v.played_on, "split": v.split, "has_video": v.path.exists(),
                "calibration": calibration(v.key),
                "cache": {name: (cache / f).exists() for name, f in (
                    ("play", "play_spans.json"), ("ball", "ball_detections.npz"),
                    ("people", "person_detections.npz"), ("rallies", "rallies.json"))},
                "roster": {"slug": assigned[v.key], "name": roster_names.get(assigned[v.key])}
                if v.key in assigned else None,
                "set_id": g["set_id"] if g else None,
                "rallies": g["rallies"] if g else 0,
                "reviewed": g["reviewed"] if g else 0,
                "job": {k: job[k] for k in ("id", "status", "step_name", "progress")}
                if job else None})
        return out

    # ---- stats -----------------------------------------------------------------------------
    @app.get("/api/stats")
    def stats(videos: str, source: str = "pipeline", pass_scale: int = 4) -> dict:
        if pass_scale not in (3, 4):
            raise HTTPException(400, "pass_scale is 3 or 4")
        path = {"pipeline": s.db_path, "labels": s.labels_db_path}.get(source)
        if path is None:
            raise HTTPException(400, "source is pipeline or labels")
        if not path.exists():
            raise HTTPException(404, f"no {source} DB ({path.name})")
        keys = [k for k in videos.split(",") if k]
        names = {r.number: (r.name, None) for r in manifest().roster}
        with conn(path) as c:
            if keys:
                for row in c.execute(
                        "SELECT p.jersey_number, p.name, p.height_cm FROM players p JOIN games g "
                        f"ON p.team_id = g.team_id WHERE g.video_key IN "
                        f"({','.join('?' * len(keys))})", keys):
                    names[row[0]] = (row[1], row[2])
            rallies = [r for k in keys for r in load_rallies(c, k)]
            counts = c.execute(
                f"SELECT COUNT(r.id), COALESCE(SUM(r.reviewed), 0) FROM rallies r "
                f"JOIN sets st ON r.set_id = st.id JOIN games g ON st.game_id = g.id "
                f"WHERE g.video_key IN ({','.join('?' * len(keys))})", keys).fetchone() \
                if keys else (0, 0)
        if not rallies:
            raise HTTPException(404, "no rallies with a point winner for these videos")
        rows = stat_rows(rallies, config.load("pass_scale"), pass_scale)
        team = rows.pop("team")
        unknown = rows.pop(None, None)
        players = [{"player": str(k), "name": names.get(int(k), (None, None))[0],
                    "height_cm": names.get(int(k), (None, None))[1], **v}
                   for k, v in sorted(rows.items(), key=lambda kv: int(kv[0]))]
        if unknown:
            players.append({"player": "?", "name": "player not known", **unknown})
        return {"columns": COLUMNS, "short": SHORT, "team": team, "players": players,
                "rallies": counts[0], "reviewed": counts[1], "counted": len(rallies),
                "pass_scale": pass_scale}

    # ---- review ----------------------------------------------------------------------------
    @app.get("/api/sets/{set_id}/rallies")
    def set_rallies(set_id: int) -> dict:
        with conn() as c:
            rows = rv.rallies(c, set_id)
        us, them = rv.score(rows)
        return {"rallies": rows, "queue": [r["id"] for r in rv.review_queue(rows)],
                "score_reviewed": [us, them]}

    def _rally(c: sqlite3.Connection, rally_id: int) -> dict:
        row = c.execute(
            "SELECT r.*, g.video_key, g.fps FROM rallies r JOIN sets st ON r.set_id = st.id "
            "JOIN games g ON st.game_id = g.id WHERE r.id = ?", (rally_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"no rally {rally_id}")
        return dict(row)

    @app.get("/api/rallies/{rally_id}")
    def rally(rally_id: int) -> dict:
        with conn() as c:
            r = _rally(c, rally_id)
            rows = rv.rallies(c, r["set_id"])
            touches = rv.touches(c, rally_id)
        prev, nxt = rv.neighbours(rows, rally_id)
        try:
            has_video = _source(r["video_key"]).exists()
        except KeyError:
            has_video = False
        return {"rally": r, "touches": touches, "has_video": has_video,
                "prev": prev and {"id": prev["id"], "rally_number": prev["rally_number"],
                                  "gap_s": round(r["start_t"] - prev["end_t"], 2)},
                "next": nxt and {"id": nxt["id"], "rally_number": nxt["rally_number"],
                                 "gap_s": round(nxt["start_t"] - r["end_t"], 2)}}

    def _source(video_key: str) -> Path:
        proxy = s.cache_dir / video_key / "proxy.mp4"
        return proxy if proxy.exists() else manifest().video(video_key).path

    @app.get("/api/rallies/{rally_id}/clip")
    def clip(rally_id: int, t0: float, t1: float) -> FileResponse:
        if not 0 < t1 - t0 <= MAX_CLIP_S:
            raise HTTPException(400, f"clip must be 0-{MAX_CLIP_S:.0f} s long")
        with conn() as c:
            r = _rally(c, rally_id)
        src = _source(r["video_key"])
        if not src.exists():
            raise HTTPException(404, "the video isn't on this machine")
        t0 = max(0.0, t0)
        out = s.cache_dir / r["video_key"] / "clips" / f"rally_{rally_id}_{t0:.2f}_{t1:.2f}.mp4"
        try:
            rv.rally_clip(src, out, t0, t1, s.clip_width)
        except Exception as e:                    # noqa: BLE001 - report, don't crash
            raise HTTPException(500, f"could not cut the clip: {e}") from e
        return FileResponse(out, media_type="video/mp4")

    @app.post("/api/rallies/{rally_id}/suggest")
    def suggest(rally_id: int, body: Suggest) -> dict:
        return {"outcome": rv.suggest_outcome(_edits(body.touches), body.winner, body.end_t)}

    @app.post("/api/rallies/{rally_id}/review")
    def review(rally_id: int, body: Review) -> dict:
        if body.winner not in WINNERS:
            raise HTTPException(400, f"winner is one of {WINNERS}")
        if body.end_t <= body.start_t:
            raise HTTPException(400, "the rally has to end after it starts")
        if body.outcome is not None and body.outcome not in s.outcomes:
            raise HTTPException(400, f"unknown outcome {body.outcome!r}")
        if body.end_detail is not None and body.end_detail not in s.end_details:
            raise HTTPException(400, f"unknown end detail {body.end_detail!r}")
        for t in body.touches:
            if t.team not in ("us", "them", None) or (t.action and t.action not in s.actions):
                raise HTTPException(400, f"bad touch {t}")
        with conn() as c:
            r = _rally(c, rally_id)
            rv.save_review(c, rally_id, r["fps"], body.winner, body.outcome, body.serving_team,
                           body.end_t, _edits(body.touches),
                           body.touches_checked and body.winner != rv.NOT_A_RALLY,
                           start_t=body.start_t, end_detail=body.end_detail)
        return {"ok": True}

    @app.post("/api/rallies/{rally_id}/merge")
    def merge(rally_id: int, body: Merge) -> dict:
        with conn() as c:
            a, b = _rally(c, rally_id), _rally(c, body.other_id)
            if a["set_id"] != b["set_id"]:
                raise HTTPException(400, "rallies are from different sets")
            rv.merge_rallies(c, rally_id, body.other_id)
        return {"ok": True}

    @app.post("/api/rallies/{rally_id}/split")
    def split(rally_id: int, body: Split) -> dict:
        with conn() as c:
            r = _rally(c, rally_id)
            try:
                start = body.start_t if body.start_t is not None else r["start_t"]
                end = body.end_t if body.end_t is not None else r["end_t"]
                if (start, end) != (r["start_t"], r["end_t"]):
                    rv.set_span(c, rally_id, start, end)
                new_id = rv.split_rally(c, rally_id, body.t)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        return {"ok": True, "new_id": new_id}

    @app.post("/api/sets/{set_id}/rallies")
    def add_rally(set_id: int, body: NewRally) -> dict:
        with conn() as c:
            try:
                return {"id": rv.add_rally(c, set_id, body.start_t, body.end_t, body.serving_team)}
            except ValueError as e:
                raise HTTPException(400, str(e)) from e

    # ---- calibration -----------------------------------------------------------------------
    def _entry(key: str):
        try:
            return manifest().video(key)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    def _full_frame(key: str, index: int):
        """The exact frame at full resolution, cached like the click UI's frames."""
        path = s.cache_dir / key / "frames" / frame_filename(key, index)
        if path.exists():
            return cv2.imread(str(path))
        entry = _entry(key)
        if not entry.path.exists():
            raise HTTPException(404, "the video isn't on this machine")
        try:
            image = grab_frame(entry.path, index, probe(entry.path).fps)
        except IndexError as e:
            raise HTTPException(400, str(e)) from e
        path.parent.mkdir(parents=True, exist_ok=True)
        write_image(path, image)
        return image

    def _cached_frames(key: str) -> list[int]:
        out = []
        for f in (s.cache_dir / key / "frames").glob(f"{key}_f*.jpg"):
            tail = f.stem.rsplit("_f", 1)[-1]
            if tail.isdigit():
                out.append(int(tail))
        return sorted(out)

    @app.get("/api/videos/{key}/calibration")
    def calibration_info(key: str) -> dict:
        entry = _entry(key)
        if not entry.path.exists():
            raise HTTPException(404, "the video isn't on this machine")
        info = probe(entry.path)
        cal = cal_store.load(key, s.calibration_dir)
        model = court_model(entry.net_height_m)
        current = None if cal is None else {
            "frame": cal.frame, "method": cal.method, "confidence": cal.confidence,
            "needs_review": cal.needs_review, "usable": usable(cal),
            "clicks": cal.clicks, "line_clicks": cal.line_clicks,
            "rms_px": cal.reprojection_rms_px}
        return {"video": key, "width": info.width, "height": info.height, "fps": info.fps,
                "duration": info.duration, "net_height_m": entry.net_height_m,
                "keypoints": list(model.keypoints), "lines": list(CLICKABLE_LINES),
                "line_hints": LINE_HINTS, "cached_frames": _cached_frames(key),
                "current": current,
                "camera_tracked": (s.calibration_dir / f"{key}_track.npz").exists()}

    @app.get("/api/videos/{key}/frame/{index}")
    def calibration_frame(key: str, index: int) -> FileResponse:
        small = s.cache_dir / key / "calib_web" / f"f{index:06d}_{s.frame_width}.jpg"
        if not small.exists():
            image = _full_frame(key, index)
            f = min(1.0, s.frame_width / image.shape[1])
            small.parent.mkdir(parents=True, exist_ok=True)
            write_image(small, cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA),
                        jpeg_quality=92)
        return FileResponse(small, media_type="image/jpeg")

    def _solve(key: str, body: Clicks):
        entry = _entry(key)
        image = _full_frame(key, body.frame)
        h, w = image.shape[:2]
        cfg = config.load("court")["calibration"]
        model = court_model(entry.net_height_m)
        mask, scale = line_mask(image, cfg)
        scorer = LineScorer(mask, scale, model, cfg)
        try:
            solved = solve_manual(w, h, model, cfg, body.clicks, body.line_clicks, scorer)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # noqa: BLE001 - a bad set of clicks can fail the solver
            raise HTTPException(400, f"could not solve these points: {e}") from e
        return image, model, cfg, scorer, solved

    def _overlay(image, cam, model, label: str) -> bytes:
        out = draw_court(image, cam, model, label)
        f = min(1.0, s.frame_width / out.shape[1])
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    @app.post("/api/videos/{key}/calibration/solve")
    def calibration_solve(key: str, body: Clicks) -> dict:
        image, model, cfg, _, solved = _solve(key, body)
        limit = error_limit_px(cfg, image.shape[1])
        jpg = _overlay(image, solved.camera, model,
                       f"{key}  points {solved.rms:.1f}px  lines {solved.line_rms:.1f}px")
        return {"rms": round(solved.rms, 2), "line_rms": round(solved.line_rms, 2),
                "limit_px": round(limit, 2), "ok": solved.worst_rms <= limit,
                "n_points": solved.n_points, "n_line": solved.n_line,
                "overlay": "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")}

    @app.post("/api/videos/{key}/calibration")
    def calibration_save(key: str, body: Clicks) -> dict:
        image, model, cfg, scorer, solved = _solve(key, body)
        cal = manual_calibration(key, body.frame, image.shape[1], model, cfg, solved,
                                 body.clicks, body.line_clicks,
                                 scorer.evaluate(solved.camera).to_dict())
        old = s.calibration_dir / f"{key}.json"
        if old.exists():                      # clicks are human work: keep the previous one
            history = s.calibration_dir / "history"
            history.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, history / f"{key}_{time.strftime('%Y%m%d-%H%M%S')}.json")
        cal_store.save(cal, s.calibration_dir)
        if s.output_dir is not None:
            out = s.output_dir / "calibration" / f"{key}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(_overlay(image, cal.camera, model,
                                     f"{key}  f{cal.frame}  manual  {cal.confidence:.2f}"))
        stale = (s.calibration_dir / f"{key}_track.npz").exists()
        return {"saved": True, "needs_review": cal.needs_review,
                "rms_px": cal.reprojection_rms_px,
                # the camera track, and everything after it, was built on the old calibration
                "rerun_from": "track_camera" if stale and "track_camera" in s.steps else None}

    # ---- rosters ---------------------------------------------------------------------------
    def _rosters_dir() -> Path:
        if s.rosters_dir is None:
            raise HTTPException(404, "rosters aren't set up (paths.rosters_dir)")
        return s.rosters_dir

    @app.get("/api/rosters")
    def roster_list() -> list[dict]:
        d = _rosters_dir()
        assigned = rost.load_assignments(d)
        return [{"slug": r.slug, "name": r.name, "season": r.season, "notes": r.notes,
                 "players": [vars(p) for p in sorted(r.players, key=lambda p: p.number)],
                 "videos": sorted(k for k, v in assigned.items() if v == r.slug)}
                for r in rost.load_all(d).values()]

    @app.post("/api/rosters")
    def roster_new(body: RosterIn) -> dict:
        d = _rosters_dir()
        try:
            slug = rost.slugify(body.name)
        except rost.RosterError as e:
            raise HTTPException(400, str(e)) from e
        if slug in rost.load_all(d):
            raise HTTPException(400, f"there is already a roster called {body.name!r}")
        return roster_save(slug, body)

    @app.put("/api/rosters/{slug}")
    def roster_save(slug: str, body: RosterIn) -> dict:
        d = _rosters_dir()
        try:
            players = [rost.RosterPlayer(p.number, p.name.strip(), rost.parse_height(p.height),
                                         (p.position or "").strip() or None, p.libero)
                       for p in body.players]
            rost.save(d, rost.Roster(slug, body.name.strip() or slug,
                                     (body.season or "").strip() or None,
                                     (body.notes or "").strip() or None, players))
            sync_rosters()
        except rost.RosterError as e:
            raise HTTPException(400, str(e)) from e
        return {"slug": slug}

    @app.delete("/api/rosters/{slug}")
    def roster_delete(slug: str) -> dict:
        try:
            rost.delete(_rosters_dir(), slug)
        except rost.RosterError as e:
            raise HTTPException(400, str(e)) from e
        return {"deleted": slug}

    @app.put("/api/videos/{key}/roster")
    def video_roster(key: str, body: RosterChoice) -> dict:
        d = _rosters_dir()
        _entry(key)
        assigned = rost.load_assignments(d)
        if body.roster is None:
            assigned.pop(key, None)
        elif body.roster not in rost.load_all(d):
            raise HTTPException(404, f"no roster {body.roster!r}")
        else:
            assigned[key] = body.roster
        rost.save_assignments(d, assigned)
        sync_rosters()
        return {"video": key, "roster": body.roster}

    # ---- processing queue ------------------------------------------------------------------
    @app.get("/api/jobs")
    def job_list() -> list[dict]:
        return jobs.snapshot()

    @app.post("/api/jobs")
    def job_new(body: NewJob) -> dict:
        try:
            entry = manifest().video(body.video)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        if not entry.path.exists():
            raise HTTPException(400, f"{entry.path} isn't on this machine")
        try:
            return {"id": jobs.submit(body.video, from_step=body.from_step).id}
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: int) -> dict:
        try:
            return {"status": jobs.cancel(job_id).status}
        except KeyError as e:
            raise HTTPException(404, f"no job {job_id}") from e

    @app.post("/api/jobs/{job_id}/resume")
    def job_resume(job_id: int) -> dict:
        try:
            return {"id": jobs.resume(job_id).id}
        except KeyError as e:
            raise HTTPException(404, f"no job {job_id}") from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: int, tail: int = 20000) -> str:
        try:
            path = jobs.log_path(jobs.get(job_id))
        except KeyError as e:
            raise HTTPException(404, f"no job {job_id}") from e
        head = jobs.progress_text(job_id) + "\n" + "-" * 60 + "\n"
        if not path.exists():
            return head
        data = path.read_bytes()[-tail:].decode("utf-8", "replace")
        return head + (data.replace("\r", "\n") or last_line(path))

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
