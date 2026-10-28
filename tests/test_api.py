import sys
import textwrap
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from setter_eyes import db  # noqa: E402
from setter_eyes.api import jobs as jb  # noqa: E402
from setter_eyes.api.app import Settings, create_app  # noqa: E402
from setter_eyes.db.pipeline import write_video  # noqa: E402
from setter_eyes.labels import load_manifest  # noqa: E402


def _touch(t, team, action):
    return {"kind": "touch", "t": t, "frame": int(t * 60), "confidence": 0.6, "team": team,
            "ball_xyz": None, "action": action, "action_confidence": 0.5}


@pytest.fixture
def client(tmp_path):
    (tmp_path / "v1.mp4").write_bytes(b"")
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(textwrap.dedent("""\
        team: Venom
        roster:
          - {number: 7, name: Seven}
        videos:
          - id: v1
            path: v1.mp4
            venue: Humber
            net_height: boys_17_18u
            ball: mikasa_v200w_yellow
        """), encoding="utf-8")
    m = load_manifest(mpath)
    conn = db.connect(tmp_path / "events.db")
    contacts = [_touch(10.0, "them", "serve"), _touch(11.0, "us", "receive"),
                _touch(12.0, "us", "set"), _touch(13.0, "us", "attack")]
    rallies = [{"index": 1, "start_frame": 600, "end_frame": 840, "start_t": 10.0,
                "end_t": 14.0, "confidence": 0.4, "serving_team": "them",
                "contacts": [0, 1, 2, 3], "point_winner": "us", "winner_confidence": 0.9,
                "outcome": "kill", "outcome_confidence": 0.5, "score_us": 1, "score_them": 0}]
    write_video(conn, m, m.video("v1"), 60.0, rallies, contacts)
    conn.close()
    s = Settings(tmp_path / "events.db", tmp_path / "labels.db", mpath, tmp_path / "cal",
                 tmp_path / "cache", tmp_path / "logs", ["find_play"], [], 640, 2.0,
                 ["serve", "receive", "set", "attack", "dig", "block"],
                 ["kill", "attack_error", "ace", "service_error", "other", "replay"])
    return TestClient(create_app(s, start_worker=False))


def test_videos_meta_and_stats(client):
    meta = client.get("/api/meta").json()
    assert meta["team"] == "Venom" and meta["roster"] == [{"number": 7, "name": "Seven"}]
    v = client.get("/api/videos").json()
    assert [x["key"] for x in v] == ["v1"]
    assert v[0]["rallies"] == 1 and v[0]["calibration"] is None and v[0]["has_video"]
    st = client.get("/api/stats", params={"videos": "v1"}).json()
    assert st["team"]["kills"] == 1 and st["team"]["points_us"] == 1
    assert st["rallies"] == 1 and st["reviewed"] == 0
    assert client.get("/api/stats", params={"videos": "v1", "source": "labels"}).status_code \
        == 404                                            # no labels DB here
    assert client.get("/").status_code == 200             # the page itself


def test_review_round_trip(client):
    set_id = client.get("/api/videos").json()[0]["set_id"]
    lst = client.get(f"/api/sets/{set_id}/rallies").json()
    rid = lst["queue"][0]
    d = client.get(f"/api/rallies/{rid}").json()
    assert d["rally"]["point_winner_source"] == "next_serve" and len(d["touches"]) == 4
    touches = d["touches"]
    touches[3]["action"] = "dig"                          # not an attack after all
    body = {"winner": "us", "touches": touches, "end_t": 14.0}
    assert client.post(f"/api/rallies/{rid}/suggest", json=body).json()["outcome"] != "kill"
    body = {**body, "outcome": "other", "serving_team": "them", "start_t": 10.0}
    assert client.post(f"/api/rallies/{rid}/review", json=body).status_code == 200
    lst = client.get(f"/api/sets/{set_id}/rallies").json()
    assert lst["queue"] == [] and lst["score_reviewed"] == [1, 0]
    d = client.get(f"/api/rallies/{rid}").json()
    assert d["rally"]["reviewed"] == 1 and d["touches"][3]["action"] == "dig"
    bad = {**body, "winner": "maybe"}
    assert client.post(f"/api/rallies/{rid}/review", json=bad).status_code == 400


def test_job_queue_runs_steps_and_stops_without_calibration(tmp_path):
    (tmp_path / "scripts").mkdir()
    for step in ("a", "b"):
        (tmp_path / "scripts" / f"{step}.py").write_text("print('ok')", encoding="utf-8")
    calibrated = set()
    q = jb.JobQueue(tmp_path / "logs", ["a", "b"], ["b"], lambda v: v in calibrated, tmp_path,
                    command=lambda step, video: [sys.executable, "-c",
                                                 f"print('{step} {video} 3/10')"])
    job = q.submit("v1")
    with pytest.raises(ValueError):
        q.submit("v1")                                    # already queued
    q.run_job(job)
    assert job.status == jb.NEEDS_CALIBRATION and job.steps[job.step] == "b"
    calibrated.add("v1")
    again = q.resume(job.id)
    assert again.steps == ["b"]
    q.run_job(again)
    assert again.status == jb.DONE and again.progress == "b v1 3/10"
    log = q.log_path(again).read_text(encoding="utf-8")
    assert "=== " in log and "b v1" in log
    # the list survives a restart; a job running at the time shows as interrupted
    again.status = jb.RUNNING
    q._save()
    q2 = jb.JobQueue(tmp_path / "logs", ["a", "b"], [], lambda v: True, tmp_path)
    assert [j.status for j in q2.jobs] == [jb.NEEDS_CALIBRATION, jb.INTERRUPTED]


def test_job_cancel_stops_the_running_step(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "slow.py").write_text("", encoding="utf-8")
    q = jb.JobQueue(tmp_path / "logs", ["slow"], [], lambda v: True, tmp_path,
                    command=lambda step, video: [sys.executable, "-c",
                                                 "import time; time.sleep(30)"])
    job = q.submit("v1")
    q.start()
    deadline = time.time() + 10
    while q._proc is None and time.time() < deadline:
        time.sleep(0.05)
    q.cancel(job.id)
    while job.status == jb.RUNNING and time.time() < deadline:
        time.sleep(0.05)
    assert job.status == jb.CANCELLED
