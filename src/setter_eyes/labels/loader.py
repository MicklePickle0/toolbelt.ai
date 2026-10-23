"""Load and validate everything labelled for one manifest video."""

from __future__ import annotations

from dataclasses import dataclass, field

from setter_eyes.ingest.video import VideoInfo, probe
from setter_eyes.labels.boxes import BoxLabels, parse_coco
from setter_eyes.labels.events import RallyLabel, parse_events
from setter_eyes.labels.issues import Issue
from setter_eyes.labels.manifest import Manifest, VideoEntry


@dataclass
class VideoLabels:
    entry: VideoEntry
    info: VideoInfo | None
    rallies: list[RallyLabel] = field(default_factory=list)
    boxes: BoxLabels | None = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.info is not None and not any(i.severity == "error" for i in self.issues)


def load_video_labels(manifest: Manifest, entry: VideoEntry,
                      count_frames: bool = False) -> VideoLabels:
    """Probe the video and parse its label files, collecting every issue."""
    try:
        info = probe(entry.path, count_frames=count_frames)
    except (FileNotFoundError, OSError, ValueError) as e:
        return VideoLabels(entry, None, issues=[Issue("error", str(entry.path), str(e))])

    result = VideoLabels(entry, info)
    if info.needs_normalising:
        what = " and ".join(x for x in (
            "variable frame rate" if info.variable_frame_rate else "",
            f"rotation {info.rotation}" if info.rotation else "") if x)
        result.issues.append(Issue(
            "warning", str(entry.path),
            f"{what}: frame indices may differ between tools; normalise with "
            "scripts/normalize_video.py and label the normalised file"))
    if info.frame_count is None:
        result.issues.append(Issue("warning", str(entry.path),
                                   "container has no frame count; frame range not checked"))

    for path in (entry.events, entry.boxes):
        if path is not None and not path.is_file():
            result.issues.append(Issue("error", str(path), "label file not found"))
    if result.issues and not result.ok:
        return result

    if entry.events is not None:
        result.rallies, issues = parse_events(
            entry.events, info.fps, info.frame_count, manifest.roster_numbers or None)
        result.issues += issues.items
    if entry.boxes is not None:
        result.boxes, issues = parse_coco(entry.boxes, (info.width, info.height),
                                          info.frame_count)
        result.issues += issues.items
    return result
