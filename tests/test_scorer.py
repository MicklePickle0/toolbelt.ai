import numpy as np

from setter_eyes import config
from setter_eyes.contacts.features import NAMES
from setter_eyes.contacts.scorer import TouchScorer, decode, touch_targets

CFG = config.load("pipeline")["contacts"]["scorer"]


def test_each_label_gets_exactly_one_candidate():
    cand = np.array([1.00, 1.10, 2.00, 3.50])
    y, which = touch_targets(cand, np.array([1.08, 3.40]), 0.25)
    assert y.tolist() == [0, 1, 0, 1]             # the closer of the two breaks near 1.08
    assert which.tolist() == [-1, 0, -1, 1]


def test_decode_keeps_one_per_window_and_drops_weak():
    t = np.array([1.0, 1.1, 1.2, 2.0, 3.0])
    p = np.array([0.4, 0.9, 0.5, 0.2, 0.6])
    assert decode(t, p, threshold=0.3, nms_s=0.35) == [1, 4]


def test_fit_predict_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(NAMES)))
    y = (X[:, 0] > 0.5).astype(int)                # touches: high first feature
    ours = np.where(y == 1, (X[:, 1] > 0).astype(int), -1)
    m = TouchScorer.fit(X, y, ours, ["v"], CFG)
    pt, po = m.predict(X)
    assert ((pt >= 0.5) == y).mean() > 0.95
    real = y == 1
    assert ((po[real] >= 0.5) == ours[real]).mean() > 0.9
    again = TouchScorer.load(m.save(tmp_path / "s.pkl"))
    np.testing.assert_allclose(again.predict(X)[0], pt)
    assert again.trained_on == ["v"]
