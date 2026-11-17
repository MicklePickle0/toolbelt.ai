"""Processing queue for the web app: pipeline steps run one video at a time on this machine.

A job is one video and a list of steps (`app.steps` in config/pipeline.yaml); each step is
`python scripts/<step>.py --video KEY`, its output appended to the job's log. One worker
thread runs the jobs in the order they were queued (one GPU). Detection resumes where it
stopped, so a cancelled or failed job can simply be queued again.

Steps that need the court calibration (`app.needs_calibration`) are not started without
one: the job stops at that step as "needs calibration", the worker goes on with the next
job, and the job is resumed from that step once the video is calibrated.

A processor-only step (`app.background_steps`, e.g. track_camera) is started alongside
the GPU steps (`app.background_during`: the detectors) once the video is calibrated, with
its own log (<job log>.bg.log, copied into the job's log when it ends); when the job
reaches that step it just waits for it. If it failed, the step runs again normally.

Jobs are kept in <log_dir>/jobs.json, so the list survives a restart; a job that was
running when the app stopped is shown as interrupted.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"
CANCELLED, NEEDS_CALIBRATION, INTERRUPTED = "cancelled", "needs_calibration", "interrupted"
FINISHED = (DONE, FAILED, CANCELLED, NEEDS_CALIBRATION, INTERRUPTED)


@dataclass
class Job:
    id: int
    video: str
    steps: list[str]
    status: str = QUEUED
    step: int = 0                       # index into steps of the running (or stopping) step
    progress: str = ""                  # the step's last output line
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    step_started: float | None = None
    error: str | None = None

    @property
    def log_name(self) -> str:
        return f"job_{self.id:04d}_{self.video}.log"


def last_line(path: Path, tail: int = 4096) -> str:
    """The last non-empty line of a log (progress lines may end in \\r)."""
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - tail))
        text = f.read().decode("utf-8", "replace")
    lines = [s.strip() for s in text.replace("\r", "\n").split("\n") if s.strip()]
    return lines[-1][:300] if lines else ""


_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")
_ETA = re.compile(r"~\s*(\d+(?:\.\d+)?)\s*min left")


def parse_progress(line: str) -> tuple[float | None, float | None]:
    """(share done 0-1, minutes left) from a step's progress line, e.g. "frame 7998:
    4000/34436 at 8.8/s, ~58 min left" -> (0.116, 58). None where the line doesn't say."""
    fracs = [(int(a), int(b)) for a, b in _FRACTION.findall(line) if int(b) > 0]
    frac = next((min(1.0, a / b) for a, b in reversed(fracs) if a <= b), None)
    eta = _ETA.search(line)
    return frac, float(eta.group(1)) if eta else None


def progress_bar(share: float, width: int = 30) -> str:
    """[#########---------------------]  30%"""
    n = round(max(0.0, min(1.0, share)) * width)
    return f"[{'#' * n}{'-' * (width - n)}] {share:4.0%}"


class JobQueue:
    """`is_calibrated(video)` says whether a video has its court calibration. `command(step,
    video)` builds the step's command line (default: this Python on scripts/<step>.py)."""

    def __init__(self, log_dir: Path, steps: list[str], needs_calibration: list[str],
                 is_calibrated: Callable[[str], bool], repo_root: Path,
                 command: Callable[[str, str], list[str]] | None = None,
                 step_weights: dict[str, float] | None = None,
                 background_steps: list[str] | None = None,
                 background_during: list[str] | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.steps = list(steps)
        self.needs_calibration = set(needs_calibration)
        self.is_calibrated = is_calibrated
        self.repo_root = repo_root
        self.step_weights = step_weights or {}      # rough share of a job's time per step
        self.background_steps = set(background_steps or [])
        self.background_during = set(background_during or [])
        self._bg: subprocess.Popen | None = None
        self.command = command or (lambda step, video: [
            sys.executable, "-u", str(repo_root / "scripts" / f"{step}.py"), "--video", video])
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._cancel: set[int] = set()
        self.jobs: list[Job] = self._load()
        self._thread: threading.Thread | None = None

    # ---- persistence ---------------------------------------------------------------------
    @property
    def _store(self) -> Path:
        return self.log_dir / "jobs.json"

    def _load(self) -> list[Job]:
        if not self._store.exists():
            return []
        jobs = [Job(**d) for d in json.loads(self._store.read_text(encoding="utf-8"))]
        for j in jobs:
            if j.status == RUNNING:
                j.status, j.finished = INTERRUPTED, j.finished or time.time()
        return jobs

    def _save(self) -> None:
        tmp = self._store.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(j) for j in self.jobs], indent=1), encoding="utf-8")
        os.replace(tmp, self._store)

    # ---- API -----------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="jobs", daemon=True)
            self._thread.start()

    def submit(self, video: str, steps: list[str] | None = None,
               from_step: str | None = None) -> Job:
        steps = list(steps or self.steps)
        unknown = [s for s in steps if not (self.repo_root / "scripts" / f"{s}.py").exists()]
        if unknown:
            raise ValueError(f"unknown steps: {unknown}")
        if from_step is not None:
            if from_step not in steps:
                raise ValueError(f"{from_step!r} is not one of the steps")
            steps = steps[steps.index(from_step):]
        with self._lock:
            if any(j.video == video and j.status in (QUEUED, RUNNING) for j in self.jobs):
                raise ValueError(f"{video} is already queued or running")
            job = Job(max((j.id for j in self.jobs), default=0) + 1, video, steps)
            self.jobs.append(job)
            self._save()
        self._wake.set()
        return job

    def resume(self, job_id: int) -> Job:
        """Queue a stopped job again from the step it stopped at."""
        old = self.get(job_id)
        if old.status not in FINISHED or old.status == DONE:
            raise ValueError(f"job {job_id} is {old.status}")
        return self.submit(old.video, old.steps[old.step:])

    def cancel(self, job_id: int) -> Job:
        with self._lock:
            job = self.get(job_id)
            if job.status == QUEUED:
                job.status, job.finished = CANCELLED, time.time()
                self._save()
            elif job.status == RUNNING:
                self._cancel.add(job_id)
                if self._proc is not None:
                    self._proc.terminate()
                if self._bg is not None:
                    self._bg.terminate()
        return job

    def get(self, job_id: int) -> Job:
        for j in self.jobs:
            if j.id == job_id:
                return j
        raise KeyError(job_id)

    def log_path(self, job: Job) -> Path:
        return self.log_dir / job.log_name

    def snapshot(self) -> list[dict]:
        """Jobs, newest first, with the running step's latest output line and how far along
        the step and the whole job are (step_share, share: 0-1, None if unknown)."""
        out = []
        for j in reversed(self.jobs):
            d = asdict(j)
            if j.status == RUNNING:
                d["progress"] = last_line(self.log_path(j))
            d["step_name"] = j.steps[j.step] if j.step < len(j.steps) else None
            step_share, eta = (parse_progress(d["progress"]) if j.status == RUNNING
                               else (None, None))
            d["step_share"], d["step_eta_min"] = step_share, eta
            d["share"] = self._share(j, step_share or 0.0)
            out.append(d)
        return out

    def _share(self, job: Job, step_share: float) -> float:
        """The job's share done, weighting steps by step_weights (unlisted steps: 1)."""
        if job.status == DONE:
            return 1.0
        w = [float(self.step_weights.get(s, 1.0)) for s in job.steps]
        done = sum(w[:job.step]) + (w[job.step] * step_share if job.step < len(w) else 0.0)
        return round(done / max(1e-9, sum(w)), 4)

    def progress_text(self, job_id: int) -> str:
        """A text progress bar for the top of the log view."""
        d = next(x for x in self.snapshot() if x["id"] == job_id)
        n = len(d["steps"])
        lines = [f"job {d['id']} {d['video']}: {d['status']}",
                 f"all steps  {progress_bar(d['share'])}"]
        if d["step_name"] and d["status"] in (RUNNING, QUEUED):
            step = f"step {min(d['step'] + 1, n)}/{n} {d['step_name']}"
            if d["step_share"] is not None:
                step += f"  {progress_bar(d['step_share'])}"
            if d["step_eta_min"] is not None:
                step += f"  ~{d['step_eta_min']:.0f} min left in this step"
            lines.append(step)
        return "\n".join(lines)

    # ---- worker --------------------------------------------------------------------------
    def _next(self) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs if j.status == QUEUED), None)

    def _run(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self._wake.wait(timeout=5)
                self._wake.clear()
                continue
            self.run_job(job)

    def _start_background(self, job: Job, i: int) -> tuple[str, subprocess.Popen] | None:
        """Start the first later background step (calibrated video only), if any."""
        for step in job.steps[i + 1:]:
            if step not in self.background_steps:
                continue
            if step in self.needs_calibration and not self.is_calibrated(job.video):
                return None
            f = self.log_path(job).with_suffix(".bg.log").open("wb")
            proc = subprocess.Popen(self.command(step, job.video), cwd=self.repo_root,
                                    stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            f.close()                               # the child keeps its own handle
            with self._lock:
                self._bg = proc
            return step, proc
        return None

    def _join_background(self, job: Job, bg: tuple[str, subprocess.Popen]) -> int:
        """Wait for the background step; copy its log into the job's. -> exit code."""
        step, proc = bg
        code = proc.wait()
        with self._lock:
            self._bg = None
        side = self.log_path(job).with_suffix(".bg.log")
        with self.log_path(job).open("ab") as f:
            f.write(f"\n=== {step} (ran alongside detection, exit {code})\n".encode())
            if side.exists():
                f.write(side.read_bytes())
                side.unlink()
        return code

    def run_job(self, job: Job) -> None:
        with self._lock:
            job.status, job.started = RUNNING, time.time()
            self._save()
        log = self.log_path(job)
        bg = None                                   # (step, process) running alongside
        for i in range(job.step, len(job.steps)):
            step = job.steps[i]
            with self._lock:
                job.step, job.step_started = i, time.time()
                self._save()
            if bg is not None and step == bg[0]:
                if self._join_background(job, bg) == 0:
                    bg = None
                    continue                        # done alongside the detectors
                bg = None                           # failed: run it again, normally
            if step in self.needs_calibration and not self.is_calibrated(job.video):
                self._stop_background(job, bg)
                self._finish(job, NEEDS_CALIBRATION,
                             f"{step} needs the court calibration: calibrate, then resume")
                return
            if bg is None and step in self.background_during:
                bg = self._start_background(job, i)
            with log.open("ab") as f:
                f.write(f"\n=== {time.strftime('%H:%M:%S')} {step}\n".encode())
                f.flush()
                proc = subprocess.Popen(self.command(step, job.video), cwd=self.repo_root,
                                        stdout=f, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"})
                with self._lock:
                    self._proc = proc
                code = proc.wait()
                with self._lock:
                    self._proc = None
            if job.id in self._cancel:
                self._cancel.discard(job.id)
                self._stop_background(job, bg)
                self._finish(job, CANCELLED, f"cancelled during {step}")
                return
            if code != 0:
                self._stop_background(job, bg)
                self._finish(job, FAILED, f"{step} exited with {code}: {last_line(log)}")
                return
        if bg is not None:
            self._join_background(job, bg)
        self._finish(job, DONE, None, step=len(job.steps))

    def _stop_background(self, job: Job, bg: tuple[str, subprocess.Popen] | None) -> None:
        if bg is not None:
            bg[1].terminate()
            self._join_background(job, bg)

    def _finish(self, job: Job, status: str, error: str | None, step: int | None = None) -> None:
        with self._lock:
            job.status, job.error, job.finished = status, error, time.time()
            if step is not None:
                job.step = step
            job.progress = last_line(self.log_path(job))
            self._save()
