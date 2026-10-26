import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clips import moments  # noqa: E402
from test_stat_sheet import RALLIES  # noqa: E402


def test_kills_of_one_player_and_every_receive():
    fps = 10.0
    kills = moments(RALLIES, fps, "kills", None, 13, 2.0, 2.0)
    assert kills == [(1.0 - 2.0, 1.9 + 2.0)]                 # rally 1 spans frames 10-19
    assert moments(RALLIES, fps, "kills", None, 7, 2.0, 2.0) == []
    receives = moments(RALLIES, fps, None, "receive", None, 3.0, 2.0)
    assert len(receives) == 1                                # ours only (rally 1)
    serves = moments(RALLIES, fps, None, "serve", None, 3.0, 2.0)
    assert len(serves) == 2                                  # rallies 2, 3; not the replay
