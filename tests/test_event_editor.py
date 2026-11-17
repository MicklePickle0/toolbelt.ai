from setter_eyes.labels.event_editor import ENTER, ESC, EventEditor
from setter_eyes.labels.events import parse_events

FPS = 60.0


def _keys(ed, frame, keys):
    for ch in keys:
        ed.key(ch, frame)


def test_a_rally_by_keyboard_round_trips_through_the_csv(tmp_path):
    path = tmp_path / "events.csv"
    ed = EventEditor(path, FPS)
    _keys(ed, 600, "s")
    _keys(ed, 640, "t1")                          # their serve
    _keys(ed, 700, ["u", "2", "7", ENTER, "3"])   # our #7 receives, a 3 pass
    _keys(ed, 760, ["u", "3", ENTER])             # set, jersey unknown
    _keys(ed, 800, ["u", "4", "1", "3", ENTER])   # #13 attacks
    _keys(ed, 830, ["e", "u", "k"])               # our point, kill
    rallies, issues = parse_events(path, FPS)
    assert not issues.errors, [str(i) for i in issues.errors]
    (r,) = rallies
    assert (r.start_frame, r.end_frame, r.point_winner, r.outcome) == (600, 830, "us", "kill")
    assert [(c.team, c.action, c.jersey, c.pass_score) for c in r.contacts] == [
        ("them", "serve", None, None), ("us", "receive", 7, 3), ("us", "set", None, None),
        ("us", "attack", 13, None)]
    assert r.serving_team == "them"

    again = EventEditor(path, FPS)                # reopening continues the file
    assert len(again.events) == 6
    so_far = [e for e in again.rally_at(750) if e.frame <= 750]
    assert so_far[-1].fields["action"] == "receive"
    assert again.rally_at(900) is None


def test_undo_cancel_and_delete(tmp_path):
    ed = EventEditor(tmp_path / "events.csv", FPS)
    _keys(ed, 100, "s")
    _keys(ed, 150, ["u", ESC])                    # cancelled: nothing added
    assert len(ed.events) == 1 and ed.pending is None
    _keys(ed, 160, "t6")
    _keys(ed, 160, "z")                           # undo the dig
    assert [e.type for e in ed.events] == ["rally_start"]
    _keys(ed, 100, "x")                           # delete the start
    assert ed.events == []
    assert ed.neighbour(0, True) is None


def test_a_replayed_point_has_no_winner_and_passes_the_check(tmp_path):
    path = tmp_path / "events.csv"
    ed = EventEditor(path, FPS)
    _keys(ed, 600, "s")
    _keys(ed, 640, "t1")
    _keys(ed, 700, ["u", "2", "7", ENTER, "3"])
    _keys(ed, 760, ["e", "r"])                    # a ball rolled on: replayed
    rallies, issues = parse_events(path, FPS)
    assert not issues.errors, [str(i) for i in issues.errors]
    (r,) = rallies
    assert r.replayed and r.point_winner is None
