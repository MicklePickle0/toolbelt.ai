import json

import pytest

from conftest import FPS, N_FRAMES, SIZE, coco
from setter_eyes import db
from setter_eyes.labels import (
    LabelError,
    import_video,
    load_manifest,
    load_video_labels,
    parse_coco,
    parse_events,
    parse_time,
)

HEADER = "set,frame,t,type,team,jersey,action,pass_score,point_winner,outcome\n"


def events(tmp_path, body: str, **kw):
    path = tmp_path / "events.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return parse_events(path, FPS, kw.pop("frame_count", N_FRAMES), **kw)


def messages(issues, severity="error"):
    return " | ".join(i.message for i in issues.items if i.severity == severity)


def test_parse_time():
    assert parse_time("12.5") == 12.5
    assert parse_time("1:02.25") == 62.25
    assert parse_time("1:01:02") == 3662
    with pytest.raises(ValueError):
        parse_time("abc")


def test_happy_path(workspace):
    m = load_manifest(workspace)
    rallies, issues = parse_events(m.videos[0].events, FPS, N_FRAMES, m.roster_numbers)
    assert issues.items == []
    assert [(r.set_number, r.rally_number) for r in rallies] == [(1, 1), (1, 2)]
    r1, r2 = rallies
    assert (r1.start_frame, r1.end_frame, r1.point_winner, r1.outcome) == (2, 28, "us", "kill")
    assert r1.serving_team == "them"
    assert [c.action for c in r1.contacts] == ["serve", "receive", "set", "attack"]
    assert r1.contacts[1].pass_score == 3 and r1.contacts[1].jersey == 7
    # t column converted with fps.
    assert (r2.start_frame, r2.contacts[0].frame, r2.end_frame) == (30, 33, 45)
    assert r2.serving_team == "us"


def test_rows_out_of_order_are_sorted(tmp_path):
    rallies, issues = events(tmp_path, "1,9,,rally_end,,,,,us,\n"
                                       "1,1,,rally_start,,,,,,\n"
                                       "1,3,,contact,us,,serve,,,\n")
    assert not issues.errors
    assert rallies[0].start_frame == 1 and rallies[0].end_frame == 9


@pytest.mark.parametrize("body, expected", [
    ("1,5,,contact,us,,serve,,,\n", "outside a rally"),
    ("1,1,,rally_start,,,,,,\n", "missing rally_end"),
    ("1,1,,rally_start,,,,,,\n1,2,,rally_start,,,,,,\n1,3,,rally_end,,,,,us,\n",
     "still open"),
    ("1,1,,rally_start,,,,,,\n1,2,,contact,us,,spike,,,\n1,3,,rally_end,,,,,us,\n",
     "unknown action"),
    ("1,1,,rally_start,,,,,,\n1,2,,contact,us,,set,3,,\n1,3,,rally_end,,,,,us,\n",
     "pass_score only applies"),
    ("1,1,,rally_start,,,,,,\n1,2,,contact,us,,receive,5,,\n1,3,,rally_end,,,,,us,\n",
     "0-4"),
    ("1,1,,rally_start,,,,,,\n1,3,,rally_end,,,,,,\n", "point_winner"),
    ("1,1,,rally_start,,,,,,\n1,3,,rally_end,,,,,us,bad\n", "unknown outcome"),
    ("1,1,,rally_start,,,,,,\n1,999,,rally_end,,,,,us,\n", "outside video"),
    ("1,1,,rally_start,,,,,,\n1,2,,contact,,,serve,,,\n1,3,,rally_end,,,,,us,\n",
     "team us|them"),
    ("1,1,,serve,,,,,,\n", "type must be"),
    ("2,1,,rally_start,,,,,,\n1,3,,rally_end,,,,,us,\n", "comes after set"),
])
def test_errors(tmp_path, body, expected):
    _, issues = events(tmp_path, body)
    assert expected in messages(issues)


def test_contact_chain_warnings(tmp_path):
    body = ("1,1,,rally_start,,,,,,\n"
            "1,2,,contact,them,,serve,,,\n"
            "1,3,,contact,us,,dig,,,\n"
            "1,4,,contact,us,,set,,,\n"
            "1,5,,contact,us,,set,,,\n"
            "1,6,,contact,us,,attack,,,\n"
            "1,7,,contact,them,,block,,,\n"
            "1,8,,contact,us,,receive,,,\n"
            "1,9,,rally_end,,,,,us,\n")
    _, issues = events(tmp_path, body)
    assert not issues.errors
    warnings = messages(issues, "warning")
    assert "4 consecutive contacts by us" in warnings
    assert "receive should directly follow" in warnings


def test_jersey_checked_against_roster(tmp_path):
    body = "1,1,,rally_start,,,,,,\n1,2,,contact,us,99,serve,,,\n1,3,,rally_end,,,,,us,\n"
    _, issues = events(tmp_path, body, roster_numbers={1, 7})
    assert "not in the roster" in messages(issues, "warning")


def test_coco_parsing(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(coco([5, 10, 15], with_ball={5, 15})))
    labels, issues = parse_coco(path, SIZE, N_FRAMES)
    assert not issues.errors
    assert sorted(labels.frames) == [5, 10, 15]
    assert labels.frames[10] == []            # negative kept
    assert labels.count("ball") == 2          # "Volleyball" alias mapped
    assert "referee" in messages(issues, "warning")


def test_coco_errors(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(coco([5, 900], set(), file_name=lambda f: f"frame_{f}.png")))
    _, issues = parse_coco(path, (320, 240), N_FRAMES)
    errors = messages(issues)
    assert "outside the video" in errors
    assert "label the original resolution" in errors

    path.write_text(json.dumps(coco([5], set(), file_name=lambda f: "IMG_1.jpg")))
    _, issues = parse_coco(path, SIZE, N_FRAMES)
    assert "cannot get a frame index" in messages(issues)


def test_manifest_validation(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("team: Us\nroster: [{number: 1, name: A}]\nvideos:\n"
                    "  - {id: 'bad key', path: x.mp4}\n"
                    "  - {id: ok, path: x.mp4, split: training, lineups: {1: [1, 2, 3]}}\n"
                    "  - {id: tall, path: x.mp4, venue: Gym, net_height: 3.5}\n")
    with pytest.raises(LabelError) as e:
        load_manifest(path)
    text = str(e.value)
    assert "'id' must be" in text and "split must be" in text and "six distinct" in text
    assert "ok: missing 'venue'" in text and "tall: net_height must be" in text
    assert "tall: ball must be one of" in text


def test_manifest_net_height(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text("team: Us\nvideos:\n"
                    "  - {id: a, path: a.mp4, venue: Gym, net_height: boys_15_16u"
                    ", ball: mikasa_v200w_yellow}\n"
                    "  - {id: b, path: b.mp4, venue: Gym, net_height: 2.3"
                    ", ball: mikasa_vq200w_can_red}\n")
    a, b = load_manifest(path).videos
    assert (a.net_height_m, a.net_height_preset) == (2.35, "boys_15_16u")
    assert (b.net_height_m, b.net_height_preset) == (2.3, None)
    assert (a.ball, b.ball) == ("mikasa_v200w_yellow", "mikasa_vq200w_can_red")


def test_import_round_trip(workspace, tmp_path):
    m = load_manifest(workspace)
    vl = load_video_labels(m, m.videos[0])
    assert vl.ok, vl.issues
    conn = db.connect(tmp_path / "labels.db")
    summary = import_video(conn, m, vl.entry, vl.info, vl.rallies, vl.boxes)
    assert (summary.sets, summary.rallies, summary.contacts, summary.ball_samples) == (1, 2, 5, 4)

    # Re-import replaces instead of duplicating.
    import_video(conn, m, vl.entry, vl.info, vl.rallies, vl.boxes)
    assert conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0] == 5
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1

    row = conn.execute(
        "SELECT p.jersey_number, c.pass_score, c.team, c.reviewed FROM contacts c "
        "JOIN players p ON p.id = c.player_id WHERE c.action = 'receive'").fetchone()
    assert tuple(row) == (7, 3, "us", 1)
    lineup = conn.execute("SELECT starting_lineup FROM sets").fetchone()[0]
    assert json.loads(lineup) == [1, 7, 2, 3, 4, 5]
    in_rally = conn.execute(
        "SELECT COUNT(*) FROM ball_samples WHERE rally_id IS NOT NULL").fetchone()[0]
    assert in_rally == 4  # frames 5-20 all inside rally 1 (frames 2-28)
    conn.close()


def test_missing_label_file_is_an_error(workspace):
    (workspace.parent / "labels" / "clip" / "events.csv").unlink()
    m = load_manifest(workspace)
    vl = load_video_labels(m, m.videos[0])
    assert not vl.ok
    assert any("label file not found" in i.message for i in vl.issues)


def test_label_batch_helpers_and_default_label_paths(tmp_path):
    import importlib.util

    from setter_eyes.config import REPO_ROOT

    spec = importlib.util.spec_from_file_location("label_batch",
                                                  REPO_ROOT / "scripts" / "label_batch.py")
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)
    assert lb.bursts([5, 1, 2, 3, 9, 10]) == [[1, 2, 3], [5], [9, 10]]
    assert lb.spread(list(range(20)), 2) == [5, 15]
    assert lb._video_of("mac-iron_set1_f000123_jpg.rf.abc123.jpg", ["mac-iron_set1"]) == \
        "mac-iron_set1"

    (tmp_path / "labels" / "a").mkdir(parents=True)
    (tmp_path / "labels" / "a" / "boxes.coco.json").write_text("{}")
    path = tmp_path / "m.yaml"
    path.write_text("team: Us\nvideos:\n"
                    "  - {id: a, path: a.mp4, venue: Gym, net_height: women"
                    ", ball: mikasa_v200w_yellow}\n"
                    "  - {id: b, path: b.mp4, venue: Gym, net_height: women"
                    ", ball: mikasa_v200w_yellow}\n")
    a, b = load_manifest(path).videos
    assert a.boxes == tmp_path / "labels" / "a" / "boxes.coco.json" and a.events is None
    assert b.boxes is None
