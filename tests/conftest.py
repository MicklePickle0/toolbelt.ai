import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytest

FPS = 30.0
N_FRAMES = 60
SIZE = (160, 120)  # w, h


BITS = 6       # N_FRAMES < 2**BITS
BLOCK = 20     # px; each bit is a black/white square along the top edge


def encode_index(img: np.ndarray, index: int) -> None:
    """Paint a frame's index as binary blocks (survives lossy encoding)."""
    for bit in range(BITS):
        value = 255 if index >> bit & 1 else 0
        img[:BLOCK, bit * BLOCK:(bit + 1) * BLOCK] = value


def decoded_index(image: np.ndarray) -> int:
    """Inverse of encode_index."""
    return sum(1 << bit for bit in range(BITS)
               if image[5:BLOCK - 5, bit * BLOCK + 5:(bit + 1) * BLOCK - 5].mean() > 127)


def ball_xy(index: int) -> tuple[int, int]:
    return 20 + 2 * index, 90 - index


def write_video(path: Path, n_frames: int = N_FRAMES, fps: float = FPS) -> Path:
    w, h = SIZE
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for i in range(n_frames):
        img = np.full((h, w, 3), 90, np.uint8)
        encode_index(img, i)
        cv2.circle(img, ball_xy(i), 4, (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture(scope="session")
def video(tmp_path_factory) -> Path:
    return write_video(tmp_path_factory.mktemp("video") / "clip.mp4")


EVENTS_CSV = """\
set,frame,t,type,team,jersey,action,pass_score,point_winner,outcome,note
1,2,,rally_start,,,,,,,
1,4,,contact,them,,serve,,,,
1,10,,contact,us,7,receive,3,,,
1,16,,contact,us,1,set,,,,
1,22,,contact,us,7,attack,,,,
1,28,,rally_end,,,,,us,kill,
1,,0:01.0,rally_start,,,,,,,
1,,1.1,contact,us,1,serve,,,,
1,,1.5,rally_end,,,,,them,service_error,
"""


def coco(frames: list[int], with_ball: set[int], file_name=lambda f: f"clip_f{f:06d}.jpg"):
    images, anns = [], []
    for i, f in enumerate(frames, start=1):
        images.append({"id": i, "file_name": file_name(f), "width": SIZE[0], "height": SIZE[1]})
        if f in with_ball:
            x, y = ball_xy(f)
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": 3,
                         "bbox": [x - 5, y - 5, 10, 10]})
    return {"images": images, "annotations": anns,
            "categories": [{"id": 3, "name": "Volleyball"}, {"id": 4, "name": "referee"}]}


@pytest.fixture
def workspace(tmp_path, video) -> Path:
    """A data/ dir with manifest, video and both label files. Returns the manifest path."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "clip.mp4").write_bytes(video.read_bytes())
    labels = tmp_path / "labels" / "clip"
    labels.mkdir(parents=True)
    (labels / "events.csv").write_text(EVENTS_CSV, encoding="utf-8")
    (labels / "boxes.coco.json").write_text(
        json.dumps(coco([5, 10, 15, 20, 25], with_ball={5, 10, 15, 20})), encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(textwrap.dedent("""\
        team: Us
        roster:
          - {number: 1, name: Setter}
          - {number: 7, name: Hitter}
          - {number: 2, name: B}
          - {number: 3, name: C}
          - {number: 4, name: D}
          - {number: 5, name: E}
        videos:
          - id: clip
            path: videos/clip.mp4
            venue: Test Gym
            net_height: boys_17_18u
            ball: mikasa_v200w_yellow
            split: train
            labels:
              events: labels/clip/events.csv
              boxes: labels/clip/boxes.coco.json
            lineups:
              1: [1, 7, 2, 3, 4, 5]
        """), encoding="utf-8")
    return manifest
