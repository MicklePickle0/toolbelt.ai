"""Video ingest. Milestone 1.

Read video (OpenCV/ffmpeg), keep original fps + timestamps, cache frames/crops.
"""

from setter_eyes.ingest.video import (
    Frame,
    VideoInfo,
    frame_filename,
    iter_frames,
    normalise,
    probe,
    read_frames,
    write_image,
)

__all__ = ["Frame", "VideoInfo", "frame_filename", "iter_frames", "normalise", "probe",
           "read_frames", "write_image"]
