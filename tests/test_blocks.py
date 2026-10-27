import numpy as np

from setter_eyes import config
from setter_eyes.contacts.blocks import add_stuff_blocks, second_turn
from setter_eyes.contacts.events import Contact
from setter_eyes.rally.segment import Rally

CFG = config.load("pipeline")
S = CFG["contacts"]["stuff_block"]
FPS = 60.0


def _path(turns):
    """Image path at 60 Hz from t=0.85 to 1.35 s, changing velocity at each (t, (vx, vy))."""
    t = np.arange(51, 82) / FPS
    v, p, uv = np.array([0.0, 1500.0]), np.array([1000.0, 500.0]), []
    for x in t:
        for tt, nv in turns:
            if abs(x - tt) < 0.5 / FPS:
                v = np.array(nv, float)
        uv.append(p.copy())
        p = p + v / FPS
    return np.arange(51, 82), np.array(uv)


def test_a_block_is_a_second_sharp_turn_right_after_the_attack():
    attack = (1.0, (0.0, 2500.0))                     # spiked down
    frames, uv = _path([attack, (1.1, (0.0, -1500.0))])   # back up off the block
    turn, dt = second_turn(frames, uv, 1.0, FPS, S)
    assert turn >= S["min_turn_px_s"] and 0.05 <= dt <= 0.15
    frames, uv = _path([attack])                      # nothing after the attack
    assert second_turn(frames, uv, 1.0, FPS, S)[0] < S["min_turn_px_s"]


def _touch(t, team, action):
    c = Contact(int(t * 30), int(t * 60), t, "touch", team, -1, None, [0.0, 0.0], 3.0, 0.5, 0.9)
    c.action, c.action_confidence = action, 0.9
    return c


def test_a_stuff_block_is_added_and_wins_the_point():
    contacts = [_touch(10, "them", "serve"), _touch(11, "us", "receive"),
                _touch(12, "us", "set"), _touch(13, "us", "attack"), _touch(40, "us", "serve")]
    r1 = Rally(1, 600, 840, 10.0, 14.0, "them", 0, "floor", [0, 1, 2, 3], 1, 0.8)
    r2 = Rally(2, 2400, 2640, 40.0, 44.0, "us", 4, "floor", [4], 1, 0.8)
    r1.point_winner, r1.winner_confidence, r1.outcome = "them", 0.8, "attack_error"
    checks = {contacts[3].frame: {"second_turn_px_s": 2500.0, "turn_dt_s": 0.08, "fps": FPS}}
    assert add_stuff_blocks([r1, r2], contacts, checks, CFG) == 1
    block = contacts[4]
    assert (block.action, block.team, round(block.t, 2)) == ("block", "them", 13.08)
    assert r1.contacts == [0, 1, 2, 3, 4] and r1.outcome == "block_point"
    assert r2.contacts == [5] and r2.serve_contact == 5   # shifted, still the serve
