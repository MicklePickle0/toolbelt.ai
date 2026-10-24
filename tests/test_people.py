import numpy as np

from setter_eyes import config
from setter_eyes.track.people import boxes_to_reference, feet_on_floor, reference_view, track_people
from test_calibration import H, look_at

CFG = config.load("pipeline")["track"]
FPS = 59.94


def person_box(cam, xy, height=1.85, half_width=0.25):
    """Image box around a standing person at court (x, y)."""
    x, y = xy
    pts = np.array([[x - half_width, y, 0], [x + half_width, y, 0],
                    [x - half_width, y, height], [x + half_width, y, height]])
    uv, _ = cam.project(pts)
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def scene():
    """6 players moving in our half while the camera pans (same centre, turning)."""
    frames = np.arange(0, 240, 2)
    starts = np.array([[1.5, 1.5], [4.5, 1.5], [7.5, 1.5], [1.5, 6.5], [4.5, 6.5], [7.5, 6.5]])
    vel = np.array([[0.6, 0.2], [-0.4, 0.3], [0.2, -0.3], [0.5, -0.4], [-0.6, 0.1], [-0.2, 0.4]])
    cams = {int(f): look_at((4.5, 13.0, 1.6), (4.5 + 1.5 * np.sin(f / 60), 0.0, 0.6))
            for f in frames}
    return frames, starts, vel, cams


def detections(seed=0):
    rng = np.random.default_rng(seed)
    frames, starts, vel, cams = scene()
    det_f, det_b, det_s, truth = [], [], [], []
    for f in frames:
        for p, (s, v) in enumerate(zip(starts, vel, strict=True)):
            xy = s + v * f / FPS
            if rng.random() < 0.05:
                continue
            det_f.append(f)
            det_b.append(person_box(cams[int(f)], xy) + rng.normal(0, 1.0, 4))
            det_s.append(rng.uniform(0.45, 0.9))
            truth.append((p, *xy))
    return (frames, np.array(det_f), np.array(det_b), np.array(det_s), np.array(truth), cams)


def test_reference_view_cancels_the_pan():
    ref = look_at((4.5, 13.0, 1.6), (4.5, 0.0, 0.6))
    panned = look_at((4.5, 13.0, 1.6), (6.0, 0.0, 0.6))
    p = np.array([[3.0, 4.0, 1.0], [8.0, 1.0, 0.0]])
    assert np.allclose(reference_view(panned, ref, panned.project(p)[0]), ref.project(p)[0],
                       atol=0.05)
    box = person_box(panned, (3.0, 4.0))
    ref_box = boxes_to_reference(panned, ref, box[None])[0]
    assert np.abs(ref_box - person_box(ref, (3.0, 4.0))).max() < 3.0


def test_feet_on_floor_recovers_court_position():
    cam = look_at((4.5, 13.0, 1.6), (5.0, 0.0, 0.6))
    xy = np.array([[1.0, 2.0], [6.5, 7.5], [8.8, 0.5]])
    boxes = np.array([person_box(cam, p) for p in xy])
    # The box bottom is the outline's lowest corner, slightly ahead of the feet centre.
    assert np.abs(feet_on_floor(cam, boxes) - xy).max() < 0.05


def test_tracks_survive_the_pan_and_land_in_the_right_place():
    frames, det_f, det_b, det_s, truth, cams = detections()
    ref = cams[0]
    tracks = track_people(frames, det_f, det_b, det_s, FPS, lambda f: cams[f], ref, H, CFG)
    # Map each tracked row back to its true player by frame + box.
    true_of = {}
    for f, b, (p, x, y) in zip(det_f, det_b, truth, strict=True):
        true_of[(int(f), round(float(b[0]), 3))] = (int(p), x, y)
    ids_per_player: dict[int, list[int]] = {}
    errs = []
    for f, tid, b, xy, src in zip(tracks.frame, tracks.track_id, tracks.box, tracks.court_xy,
                                  tracks.source, strict=True):
        if src == 1:                                        # filled rows have no detection
            continue
        p, x, y = true_of[(int(f), round(float(b[0]), 3))]
        ids_per_player.setdefault(p, []).append(int(tid))
        errs.append(np.hypot(xy[0] - x, xy[1] - y))
    assert len(ids_per_player) == 6
    for ids in ids_per_player.values():
        main = np.bincount(ids).max()
        assert main / len(ids) > 0.9                       # one ID for nearly all samples
    assert np.median(errs) < 0.1                            # metres, from noisy boxes
    summary = tracks.summary(CFG)
    assert all(s["share_in_our_half"] > 0.9 for s in summary if s["samples"] > 30)


def test_hidden_player_gap_is_filled_and_a_jump_is_corrected():
    from setter_eyes.track.people import clean_tracks

    frames = np.arange(0, 200, 2)
    t = frames / FPS
    xy = np.stack([2.0 + 1.0 * t, 3.0 + 0.5 * t], axis=1)          # walking steadily
    keep = np.ones(len(frames), bool)
    keep[40:52] = False                                             # hidden for ~0.4 s
    glitch = xy.copy()
    glitch[70] = [8.5, -2.0]                                        # one frame on the wrong side
    box = np.stack([100 + 10 * t, 200 + 0 * t, 150 + 10 * t, 400 + 0 * t], axis=1)
    cat = {"frame": frames[keep], "track_id": np.zeros(keep.sum(), int), "box": box[keep],
           "score": np.full(keep.sum(), 0.8), "court_xy": glitch[keep],
           "feet_visible": np.ones(keep.sum(), bool)}
    tr = clean_tracks(cat, frames, FPS, CFG)
    assert np.array_equal(tr.frame, frames)                         # the gap is filled
    assert (tr.source[40:52] == 1).all() and tr.source[70] == 2
    assert np.abs(tr.court_xy - xy).max() < 0.02                    # filled + corrected
    assert np.abs(tr.court_xy_smooth - xy).max() < 0.05


def test_roles_from_time_on_each_side():
    from setter_eyes.track.people import PeopleTracks
    from setter_eyes.track.roles import classify_tracks

    n = 200
    frames = np.arange(0, 2 * n, 2)
    tracks = {0: (3.0, 4.0), 1: (5.0, -4.0), 2: (-4.0, 2.0), 3: (4.0, 5.0)}   # near, far, bench
    rows = []
    for tid, (x, y) in tracks.items():
        m = n if tid != 3 else 20                          # track 3: a 0.7 s fragment
        rows += [(f, tid, x, y) for f in frames[:m]]
    f, tid, x, y = (np.array(c) for c in zip(*rows, strict=True))
    xy = np.stack([x, y], axis=1).astype(float)
    t = PeopleTracks(f, tid, np.zeros((len(f), 4)), np.ones(len(f)), xy, xy,
                     np.ones(len(f), bool), np.zeros(len(f), np.int8))
    roles = {r["track_id"]: r["role"] for r in classify_tracks(t, FPS, CFG)}
    assert roles == {0: "near_player", 1: "far_player", 2: "other", 3: "other"}
