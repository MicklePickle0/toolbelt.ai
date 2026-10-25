from setter_eyes import db
from setter_eyes.db.pipeline import write_video
from setter_eyes.labels.manifest import Manifest, VideoEntry


def test_pipeline_output_lands_in_the_event_db(tmp_path):
    conn = db.connect(":memory:")
    manifest = Manifest(tmp_path / "manifest.yaml", "Venom", [], [])
    entry = VideoEntry("v1", tmp_path / "v1.mp4", venue="Humber", ball="mikasa")
    touch = {"kind": "touch", "t": 1.0, "frame": 60, "confidence": 0.8, "team": "us",
             "ball_xyz": [4.0, 5.0, 1.0], "action": "receive", "action_confidence": 0.7}
    floor = {**touch, "kind": "floor", "action": None}
    rally = {"index": 1, "start_frame": 30, "end_frame": 120, "start_t": 0.5, "end_t": 2.0,
             "confidence": 0.9, "serving_team": "them", "contacts": [0, 1]}
    for _ in range(2):                                   # re-writing replaces, not duplicates
        w = write_video(conn, manifest, entry, 60.0, [rally], [touch, floor])
    assert (w.rallies, w.contacts) == (1, 1)
    row = conn.execute("SELECT c.action, c.reviewed, r.point_winner FROM contacts c "
                       "JOIN rallies r ON c.rally_id = r.id").fetchall()
    assert [tuple(x) for x in row] == [("receive", 0, None)]
