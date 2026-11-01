"""Pose estimation.

RTMPose (rtmlib/ONNX) on player crops for wrist/forearm proximity and arm-swing cues.
rtmlib runs one box at a time; `pose_batch` runs many boxes (from any frames) through the
model at once, with rtmlib's own pre- and post-processing.
"""

from __future__ import annotations

import numpy as np


def pose_batch(model, items: list[tuple[np.ndarray, list[float]]]) -> tuple[np.ndarray,
                                                                          np.ndarray]:
    """(keypoints (N, K, 2), scores (N, K)) for [(image, box x0 y0 x1 y1), ...]; `model`
    an rtmlib RTMPose on the onnxruntime backend."""
    if not items:
        return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
    pre = [model.preprocess(image, box) for image, box in items]
    x = np.ascontiguousarray(np.stack([p[0].transpose(2, 0, 1) for p in pre]), np.float32)
    sess = model.session
    outs = sess.run([o.name for o in sess.get_outputs()], {sess.get_inputs()[0].name: x})
    kps, scores = [], []
    for i, (_, center, scale) in enumerate(pre):
        k, s = model.postprocess([o[i:i + 1] for o in outs], center, scale)
        kps.append(k[0])
        scores.append(s[0])
    return np.stack(kps), np.stack(scores)
