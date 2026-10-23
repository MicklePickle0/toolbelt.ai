"""Training datasets built from labelled footage. Splits are always by video."""

from setter_eyes.datasets.detection import ExportSummary, export_detection_dataset
from setter_eyes.datasets.splits import assign_splits, split_for

__all__ = ["ExportSummary", "assign_splits", "export_detection_dataset", "split_for"]
