"""Hand-labelled footage: manifest, event labels (CSV), box labels (COCO), DB import.

Format reference: docs/labelling.md.
"""

from setter_eyes.labels.boxes import Box, BoxLabels, frame_from_name, parse_coco
from setter_eyes.labels.events import ContactLabel, RallyLabel, parse_events, parse_time
from setter_eyes.labels.importer import ImportSummary, import_video
from setter_eyes.labels.issues import Issue, Issues, LabelError
from setter_eyes.labels.loader import VideoLabels, load_video_labels
from setter_eyes.labels.manifest import Manifest, RosterEntry, VideoEntry, load_manifest

__all__ = ["Box", "BoxLabels", "ContactLabel", "ImportSummary", "Issue", "Issues", "LabelError",
           "Manifest", "RallyLabel", "RosterEntry", "VideoEntry", "VideoLabels",
           "frame_from_name", "import_video", "load_manifest", "load_video_labels",
           "parse_coco", "parse_events", "parse_time"]
