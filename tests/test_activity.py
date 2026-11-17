import numpy as np

from setter_eyes import config
from setter_eyes.rally.activity import in_spans, play_spans

FPS = 30.0
CFG = config.load("pipeline")["activity"]["play"]


def series(active_s: list[tuple[float, float]], length_s: float = 300.0, seed: int = 0):
    """Activity every 0.1 s: low noise, high inside the given (start, end) seconds."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.1, length_s, 0.1)
    a = rng.uniform(0.0, 0.01, len(t))
    for s, e in active_s:
        a[(t >= s) & (t < e)] = rng.uniform(0.08, 0.12, ((t >= s) & (t < e)).sum())
    return np.round(t * FPS).astype(int), a


def test_rallies_padded_and_downtime_skipped():
    idx, a = series([(30, 40), (80, 95)])
    spans = play_spans(idx, a, FPS, int(300 * FPS), CFG)
    assert len(spans) == 2
    (s1, e1), (s2, e2) = spans
    before, after = CFG["pad_before_s"] - 0.2, CFG["pad_after_s"] - 0.2
    assert s1 <= (30 - before) * FPS and e1 >= (40 + after) * FPS
    assert s2 <= (80 - before) * FPS and e2 >= (95 + after) * FPS
    kept = sum(e - s for s, e in spans) / (300 * FPS)
    assert kept < 0.25


def test_short_pause_inside_a_rally_is_kept_and_blips_dropped():
    idx, a = series([(30, 34), (36, 40), (200, 200.3)])
    spans = play_spans(idx, a, FPS, int(300 * FPS), CFG)
    assert len(spans) == 1                                     # the blip at 200 s is noise
    s, e = spans[0]
    assert s < 30 * FPS < 40 * FPS < e


def test_spans_clipped_to_the_video():
    idx, a = series([(0.5, 10), (295, 300)])
    spans = play_spans(idx, a, FPS, int(300 * FPS), CFG)
    assert spans[0][0] == 0 and spans[-1][1] == int(300 * FPS)


def test_in_spans():
    inside = in_spans([(10, 20), (40, 50)])
    assert [f for f in range(0, 60, 5) if inside(f)] == [10, 15, 40, 45]
