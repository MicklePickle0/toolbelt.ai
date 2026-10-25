from setter_eyes.calibration.alignment import Sample, _bisect


class _FakeProbe:
    """The camera moved at frame 5000: offsets jump from 5 px to 100 px."""

    def __init__(self):
        self.calls = 0

    def measure(self, frame):
        self.calls += 1
        return Sample(frame, 100.0 if frame >= 5000 else 5.0, 200)


def test_bisect_finds_the_move():
    probe = _FakeProbe()
    lo, hi = _bisect(probe, 3600, 7200, moved=40, resolution=120)
    assert lo < 5000 <= hi and hi - lo <= 120
    assert probe.calls <= 6
