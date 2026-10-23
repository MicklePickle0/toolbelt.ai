"""Validation issues collected while reading labels (all reported at once, not fail-fast)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    severity: Severity
    source: str          # file the issue is in
    message: str
    line: int | None = None

    def __str__(self) -> str:
        where = f"{self.source}:{self.line}" if self.line is not None else self.source
        return f"{self.severity.upper():7} {where}: {self.message}"


@dataclass
class Issues:
    source: str
    items: list[Issue] = field(default_factory=list)

    def error(self, message: str, line: int | None = None) -> None:
        self.items.append(Issue("error", self.source, message, line))

    def warning(self, message: str, line: int | None = None) -> None:
        self.items.append(Issue("warning", self.source, message, line))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.items if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.items if i.severity == "warning"]


class LabelError(ValueError):
    """Raised when a label file cannot be used at all (e.g. a broken manifest)."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("\n".join(str(i) for i in issues))
