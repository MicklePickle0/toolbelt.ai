import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from conftest import decoded_index
from setter_eyes.config import REPO_ROOT
from setter_eyes.datasets import assign_splits, export_detection_dataset, split_for
from setter_eyes.labels import LabelError, load_manifest, load_video_labels
from setter_eyes.labels.manifest import Manifest, VideoEntry

RATIOS = {"train": 0.7, "valid": 0.15, "test": 0.15}


def test_split_is_stable_and_respects_pins():
    videos = [VideoEntry(key=f"v{i}", path=Path("x")) for i in range(200)]
    videos.append(VideoEntry(key="pinned", path=Path("x"), split="test"))
    m = Manifest(Path("m"), "Us", [], videos)
    a = assign_splits(m, RATIOS, split_by="video")
    assert a == assign_splits(m, RATIOS, split_by="video")
    assert a["pinned"] == "test"
    counts = {s: list(a.values()).count(s) for s in RATIOS}
    assert 110 < counts["train"] < 170 and counts["valid"] > 10 and counts["test"] > 10
    # Adding videos never moves an existing one.
    assert all(split_for(f"v{i}", RATIOS) == a[f"v{i}"] for i in range(200))


def test_detection_export(workspace, tmp_path):
    m = load_manifest(workspace)
    vl = load_video_labels(m, m.videos[0])
    out = tmp_path / "ds"
    summary = export_detection_dataset(out, [(vl.entry, vl.boxes)], {"clip": "train"},
                                       ["ball", "person"])
    assert summary.images == {"train": 5, "valid": 0, "test": 0}
    assert summary.boxes["train"] == {"ball": 4, "person": 0}

    ann = json.loads((out / "train" / "_annotations.coco.json").read_text())
    assert [c["name"] for c in ann["categories"]] == ["ball", "person"]
    for img in ann["images"]:
        image = cv2.imread(str(out / "train" / img["file_name"]))
        assert decoded_index(image) == img["frame"]
    for split in ("valid", "test"):
        assert (out / split / "_annotations.coco.json").exists()
    assert json.loads((out / "provenance.json").read_text())["videos"] == {"clip": "train"}

    with pytest.raises(FileExistsError):
        export_detection_dataset(out, [], {}, ["ball"])


def test_ingest_labels_script(workspace, tmp_path):
    db_path = tmp_path / "out.db"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "ingest_labels.py"),
           "--manifest", str(workspace), "--db", str(db_path)]
    check = subprocess.run([*cmd, "--check"], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    assert not db_path.exists()

    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "imported: game 1, 1 sets, 2 rallies, 5 contacts, 4 ball samples" in run.stdout
    assert db_path.exists()


def test_split_by_venue_keeps_venues_together():
    def v(key, venue, split=None):
        return VideoEntry(key=key, path=Path("x"), venue=venue, split=split)

    videos = [v(f"g{i}_{j}", f"Gym {i}") for i in range(60) for j in range(3)]
    videos += [v("arc1", "Queen's ARC"), v("arc2", "queens arc", split="test")]
    a = assign_splits(Manifest(Path("m"), "Us", [], videos), RATIOS, split_by="venue")
    for i in range(60):
        assert len({a[f"g{i}_{j}"] for j in range(3)}) == 1
    assert len(set(a.values())) == 3
    # Pinning one video pins its whole venue (names are normalised).
    assert a["arc1"] == a["arc2"] == "test"

    videos.append(v("arc3", "Queens ARC", split="train"))
    with pytest.raises(LabelError, match="pinned to several splits"):
        assign_splits(Manifest(Path("m"), "Us", [], videos), RATIOS, split_by="venue")
