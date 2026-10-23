"""Video probing, frame-accurate reading and CFR normalisation.

Frame index (decode order from frame 0) is the canonical time key everywhere:
labels, cache files and the event DB all refer to frames. Seeking with
CAP_PROP_POS_FRAMES is not frame-accurate for every codec, so reads that must
hit exact frames decode sequentially and skip with grab().
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

# r_frame_rate vs avg_frame_rate may differ by rounding on CFR files.
_VFR_TOLERANCE = 0.01


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float                 # average frame rate
    frame_count: int | None    # None if the container doesn't say and counting was skipped
    duration: float | None     # seconds
    codec: str | None
    rotation: int              # degrees from container metadata; 0 if none
    variable_frame_rate: bool

    @property
    def needs_normalising(self) -> bool:
        """VFR or rotated files should be normalised before labelling."""
        return self.variable_frame_rate or self.rotation != 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Frame:
    index: int
    t: float            # seconds, from the decoder timestamp
    image: np.ndarray   # BGR


def _fraction(text: str | None) -> Fraction | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def probe(path: str | Path, count_frames: bool = False) -> VideoInfo:
    """Read video metadata with ffprobe, falling back to OpenCV.

    count_frames decodes the whole file to get an exact count when the
    container doesn't store one (slow on long matches).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if shutil.which("ffprobe"):
        return _probe_ffprobe(path, count_frames)
    return _probe_opencv(path)


def _probe_ffprobe(path: Path, count_frames: bool) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,"
           "codec_name,duration:stream_tags=rotate:stream_side_data=rotation:"
           "format=duration",
           "-of", "json"]
    if count_frames:
        cmd.append("-count_frames")
    out = subprocess.run([*cmd, str(path)], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    if not data.get("streams"):
        raise ValueError(f"no video stream in {path}")
    stream = data["streams"][0]

    r_rate = _fraction(stream.get("r_frame_rate"))
    avg_rate = _fraction(stream.get("avg_frame_rate")) or r_rate
    if avg_rate is None:
        raise ValueError(f"could not determine frame rate of {path}")
    vfr = r_rate is not None and abs(float(r_rate) - float(avg_rate)) > _VFR_TOLERANCE

    frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    duration = stream.get("duration") or data.get("format", {}).get("duration")

    rotation = int(float(stream.get("tags", {}).get("rotate", 0)))
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    return VideoInfo(
        path=str(path),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=float(avg_rate),
        frame_count=int(frames) if frames not in (None, "N/A") else None,
        duration=float(duration) if duration not in (None, "N/A") else None,
        codec=stream.get("codec_name"),
        rotation=rotation % 360,
        variable_frame_rate=vfr,
    )


def _probe_opencv(path: Path) -> VideoInfo:
    cap = _open(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=count,
            duration=count / fps if count and fps else None,
            codec=None,
            rotation=0,
            variable_frame_rate=False,  # OpenCV can't tell
        )
    finally:
        cap.release()


def _open(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"could not open video {path}")
    return cap


def _timestamp(cap: cv2.VideoCapture, index: int, fps: float) -> float:
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    # Some backends report 0 for every frame; fall back to the nominal rate.
    return t if t > 0 or index == 0 else index / fps


def iter_frames(path: str | Path, start: int = 0, end: int | None = None,
                step: int = 1) -> Iterator[Frame]:
    """Yield frames start, start+step, ... up to (not including) end, decoding sequentially."""
    if step < 1:
        raise ValueError("step must be >= 1")
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        index = 0
        while end is None or index < end:
            if index < start or (index - start) % step:
                if not cap.grab():
                    return
            else:
                ok, image = cap.read()
                if not ok:
                    return
                yield Frame(index, _timestamp(cap, index, fps), image)
            index += 1
    finally:
        cap.release()


def read_frames(path: str | Path, indices: Iterable[int]) -> Iterator[Frame]:
    """Yield exactly the requested frames (sorted, deduplicated), decoding sequentially.

    Raises IndexError if the video ends before a requested frame.
    """
    wanted = sorted(set(indices))
    if not wanted:
        return
    if wanted[0] < 0:
        raise IndexError(f"negative frame index {wanted[0]}")
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        index = 0
        for target in wanted:
            while index < target:
                if not cap.grab():
                    raise IndexError(f"video ends at frame {index}, wanted {target}")
                index += 1
            ok, image = cap.read()
            if not ok:
                raise IndexError(f"video ends at frame {index}, wanted {target}")
            yield Frame(index, _timestamp(cap, index, fps), image)
            index += 1
    finally:
        cap.release()


def frame_filename(video_key: str, index: int, ext: str = ".jpg") -> str:
    """Canonical cached-frame name; config/labels.yaml frame_name_patterns parses it back."""
    return f"{video_key}_f{index:06d}{ext}"


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.suffix.lower() in (".jpg", ".jpeg") \
        else []
    # imencode + write_bytes instead of imwrite: handles non-ASCII paths on Windows.
    ok, buf = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise OSError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())


def normalise(src: str | Path, dst: str | Path, fps: float | None = None, crf: int = 16) -> Path:
    """Re-encode to constant frame rate H.264 with rotation baked in.

    Label the normalised file, not the original: frame indices of VFR files
    differ between decoders. fps defaults to the source's average rate.
    """
    src, dst = Path(src), Path(dst)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
    rate = fps or probe(src).fps
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-map", "0:v:0", "-an",                      # no audio features, by design
         "-fps_mode", "cfr", "-r", f"{rate:.6f}",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
         str(dst)],
        check=True,
    )
    return dst
