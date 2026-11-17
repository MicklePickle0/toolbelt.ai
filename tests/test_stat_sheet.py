from setter_eyes import config, db
from setter_eyes.db.read import load_rallies
from setter_eyes.ingest.video import VideoInfo
from setter_eyes.labels.events import ContactLabel, RallyLabel
from setter_eyes.labels.importer import import_video
from setter_eyes.labels.manifest import Manifest, VideoEntry
from setter_eyes.stats.sheet import stat_rows

SCALE = config.load("pass_scale")


def _r(n, contacts, winner, outcome):
    return RallyLabel(1, n, 10 * n, 10 * n + 9, winner, outcome,
                      [ContactLabel(10 * n + i, t, a, j, p)
                       for i, (t, a, j, p) in enumerate(contacts)])


RALLIES = [
    # their serve, our 3-pass, set, #13 kill past their block
    _r(1, [("them", "serve", None, None), ("us", "receive", 7, 3), ("us", "set", 1, None),
           ("us", "attack", 13, None), ("them", "block", None, None)], "us", "kill"),
    # our serve, their attack, our dig, #13 blocked
    _r(2, [("us", "serve", 7, None), ("them", "receive", None, None),
           ("them", "attack", None, None), ("us", "dig", 7, None), ("us", "set", 1, None),
           ("us", "attack", 13, None), ("them", "block", None, None)], "them", "block_point"),
    # our service error
    _r(3, [("us", "serve", 1, None)], "them", "service_error"),
    # replayed: counts for nothing
    _r(4, [("us", "serve", 7, None), ("them", "receive", None, None)], None, "replay"),
]


def test_team_and_player_stats():
    rows = stat_rows(RALLIES, SCALE)
    t = rows["team"]
    assert (t["points_us"], t["points_them"]) == (1, 2)
    assert (t["kills"], t["attack_errors"], t["attacks"]) == (1, 1, 2)
    assert t["hitting_eff"] == 0.0
    assert t["service_errors"] == 1 and t["serves"] == 2
    assert rows[13]["kills"] == 1 and rows[13]["attack_errors"] == 1
    assert rows[7]["pass_avg"] == 3.0 and rows[7]["digs"] == 1
    assert (t["side_out_won"], t["side_out_chances"]) == (1, 1)


def test_stats_from_the_db_match(tmp_path):
    conn = db.connect(":memory:")
    entry = VideoEntry("v1", tmp_path / "v.mp4", venue="Humber", ball="m")
    info = VideoInfo(str(tmp_path / "v.mp4"), 3840, 2160, 60.0, 36000, 600.0, "av1", 0, False)
    import_video(conn, Manifest(tmp_path / "m.yaml", "Venom", [], [entry]), entry, info,
                 RALLIES)
    back = load_rallies(conn, "v1")
    assert stat_rows(back, SCALE) == stat_rows(RALLIES, SCALE)
