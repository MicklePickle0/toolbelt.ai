"""The web app (FastAPI): videos and their processing queue, rally review and the stat sheet,
in a browser on the laptop or phone. Run: python scripts/serve_app.py

Everything reads and writes the event DB (paths.db_path), the same as the Streamlit review
(ui/review.py does the saving, so corrections are logged the same way). The page itself is
static/index.html + app.js; this module is the JSON API under /api.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from setter_eyes import config, db
from setter_eyes.calibration import store as cal_store
from setter_eyes.db.read import load_rallies
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

    @classmethod
    def from_config(cls) -> Settings:
        cfg = config.load("pipeline")
        p, a, labels = cfg["paths"], cfg["app"], config.load("labels")
        return cls(config.repo_path(p["db_path"]), config.repo_path(p["labels_db_path"]),
                   config.repo_path(p["manifest_path"]), config.repo_path(p["calibration_dir"]),
                   config.repo_path(p["cache_dir"]), config.repo_path(a["log_dir"]),
                   a["steps"], a["needs_calibration"], a["clip_width"], a["clip_pad_s"],
                   labels["actions"], labels["outcomes"])


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


class Suggest(BaseModel):
    winner: str | None = None
    end_t: float | None = None
    touches: list[Touch]


class Merge(BaseModel):
    other_id: int


class NewJob(BaseModel):
    video: str
    from_step: str | None = None


def _edits(touches: list[Touch]) -> list[rv.TouchEdit]:
    return [rv.TouchEdit(t.id, t.t, t.team, t.action, t.delete) for t in touches]


def create_app(settings: Settings | None = None, start_worker: bool = True) -> FastAPI:
    s = settings or Settings.from_config()
    app = FastAPI(title="Setter Eyes")

    def calibration(video: str) -> dict | None:
        cal = cal_store.load(video, s.calibration_dir)
        if cal is None:
            return None
        extra = len(cal_store.load_segments(video, s.calibration_dir))
        return {"method": cal.method, "confidence": cal.confidence,
                "needs_review": cal.needs_review, "extra_segments": extra}

    jobs = JobQueue(s.log_dir, s.steps, s.needs_calibration,
                    lambda v: cal_store.load(v, s.calibration_dir) is not None,
                    config.REPO_ROOT)
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

    def manifest():
        return load_manifest(s.manifest_path)

    # ---- meta and videos -------------------------------------------------------------------
    @app.get("/api/meta")
    def meta() -> dict:
        m = manifest()
        return {"team": m.team, "actions": s.actions, "outcomes": s.outcomes,
                "steps": s.steps, "winners": list(WINNERS), "clip_pad_s": s.clip_pad_s,
                "roster": [{"number": r.number, "name": r.name} for r in m.roster]}

    @app.get("/api/videos")
    def videos() -> list[dict]:
        with conn() as c:
            in_db = {g["video_key"]: g for g in rv.games(c)}
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
                "set_id": g["set_id"] if g else None,
                "rallies": g["rallies"] if g else 0,
                "reviewed": g["reviewed"] if g else 0,
                "job": {k: job[k] for k in ("id", "status", "step_name", "progress")}
                if job else None})
        return out

    # ---- stats -----------------------------------------------------------------------------
    @app.get("/api/stats")
    def stats(videos: str, source: str = "pipeline") -> dict:
        path = {"pipeline": s.db_path, "labels": s.labels_db_path}.get(source)
        if path is None:
            raise HTTPException(400, "source is pipeline or labels")
        if not path.exists():
            raise HTTPException(404, f"no {source} DB ({path.name})")
        keys = [k for k in videos.split(",") if k]
        names = {r.number: r.name for r in manifest().roster}
        with conn(path) as c:
            rallies = [r for k in keys for r in load_rallies(c, k)]
            counts = c.execute(
                f"SELECT COUNT(r.id), COALESCE(SUM(r.reviewed), 0) FROM rallies r "
                f"JOIN sets st ON r.set_id = st.id JOIN games g ON st.game_id = g.id "
                f"WHERE g.video_key IN ({','.join('?' * len(keys))})", keys).fetchone() \
                if keys else (0, 0)
        if not rallies:
            raise HTTPException(404, "no rallies with a point winner for these videos")
        rows = stat_rows(rallies, config.load("pass_scale"))
        team = rows.pop("team")
        unknown = rows.pop(None, None)
        players = [{"player": str(k), "name": names.get(int(k)), **v}
                   for k, v in sorted(rows.items(), key=lambda kv: int(kv[0]))]
        if unknown:
            players.append({"player": "?", "name": "player not known", **unknown})
        return {"columns": COLUMNS, "short": SHORT, "team": team, "players": players,
                "rallies": counts[0], "reviewed": counts[1], "counted": len(rallies)}

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
        for t in body.touches:
            if t.team not in ("us", "them", None) or (t.action and t.action not in s.actions):
                raise HTTPException(400, f"bad touch {t}")
        with conn() as c:
            r = _rally(c, rally_id)
            rv.save_review(c, rally_id, r["fps"], body.winner, body.outcome, body.serving_team,
                           body.end_t, _edits(body.touches),
                           body.touches_checked and body.winner != rv.NOT_A_RALLY,
                           start_t=body.start_t)
        return {"ok": True}

    @app.post("/api/rallies/{rally_id}/merge")
    def merge(rally_id: int, body: Merge) -> dict:
        with conn() as c:
            a, b = _rally(c, rally_id), _rally(c, body.other_id)
            if a["set_id"] != b["set_id"]:
                raise HTTPException(400, "rallies are from different sets")
            rv.merge_rallies(c, rally_id, body.other_id)
        return {"ok": True}

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
        if not path.exists():
            return ""
        data = path.read_bytes()[-tail:].decode("utf-8", "replace")
        return data.replace("\r", "\n") or last_line(path)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app
