import shutil

import pytest

from conftest import FPS, N_FRAMES, SIZE, decoded_index
from setter_eyes.ingest import frame_filename, iter_frames, normalise, probe, read_frames
from setter_eyes.labels import frame_from_name


def test_probe(video):
    info = probe(video)
    assert (info.width, info.height) == SIZE
    assert info.fps == pytest.approx(FPS)
    assert info.frame_count == N_FRAMES
    assert not info.needs_normalising


def test_probe_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")


def test_read_frames_is_frame_accurate(video):
    wanted = [0, 7, 7, 31, 59, 3]
    frames = list(read_frames(video, wanted))
    assert [f.index for f in frames] == [0, 3, 7, 31, 59]
    for f in frames:
        assert decoded_index(f.image) == f.index
        assert f.t == pytest.approx(f.index / FPS, abs=1e-3)


def test_read_frames_past_end(video):
    with pytest.raises(IndexError):
        list(read_frames(video, [N_FRAMES + 5]))


def test_iter_frames_range_and_step(video):
    frames = list(iter_frames(video, start=10, end=30, step=7))
    assert [f.index for f in frames] == [10, 17, 24]
    assert all(decoded_index(f.image) == f.index for f in frames)


def test_frame_filename_round_trip():
    assert frame_from_name(frame_filename("game_1", 1234)) == 1234
    assert frame_from_name("frame_000042.PNG") == 42
    # Roboflow re-export: the hex hash must not be mistaken for a frame number.
    assert frame_from_name("game_1_f000077_jpg.rf.f12345abcdef.jpg") == 77
    assert frame_from_name("IMG_1234.jpg") is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_normalise_keeps_frames(video, tmp_path):
    out = normalise(video, tmp_path / "norm.mp4")
    info = probe(out, count_frames=True)
    assert info.frame_count == N_FRAMES
    assert not info.variable_frame_rate
    f = next(read_frames(out, [40]))
    assert decoded_index(f.image) == 40
