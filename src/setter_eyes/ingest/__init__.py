"""Video ingest. Milestone 1.

Read video (OpenCV/ffmpeg), keep original fps + timestamps, cache frames/crops.
"""

from setter_eyes.ingest.video import (
    Frame,
    VideoInfo,
    burst_indices,
    frame_filename,
    grab_frame,
    iter_frames,
    iter_frames_scaled,
    normalise,
    probe,
    read_frames,
    write_image,
)

__all__ = ["Frame", "VideoInfo", "burst_indices", "frame_filename", "grab_frame", "iter_frames",
           "iter_frames_scaled", "normalise", "probe", "read_frames", "write_image"]
