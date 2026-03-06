#!/usr/bin/env python3
"""
Batch 2D Pose Estimation Runner
================================
Discovers all subfolders under a root directory that contain a video
subfolder (default: 'synchronized_videos/'), then runs MediaPipe 2D
pose estimation on each folder and applies Butterworth low-pass
smoothing to the resulting keypoints.

What this script does
---------------------
  1. Scans a root directory for subfolders matching the expected layout
  2. For each valid folder:
     a) Runs MediaPipe Pose on every video in synchronized_videos/
        (cameras are processed in parallel via multiprocessing)
     b) Saves labeled overlay videos + raw 2D keypoint .npy files
     c) Applies a 4th-order Butterworth low-pass filter (default 4 Hz)
        to smooth the X/Y trajectories (confidence channel untouched)
     d) Overwrites the keypoint .npy files with smoothed data
  3. Saves a summary CSV (batch_pose_summary.csv) to the root directory
     with per-folder, per-camera detection rates, confidence stats, and
     any anomalies/warnings encountered

  NOTE: camkit3d.pose2d's own logging and tqdm progress bars are
  suppressed in batch mode so the rich live table is clean. Pass
  --verbose to re-enable them (useful for debugging).

Expected subfolder structure
----------------------------
    <root_dir>/
        block_01_left/
            synchronized_videos/
                camera_0_synchronized.mp4
                camera_1_synchronized.mp4
                ...
        block_02_right/
            synchronized_videos/
                ...

Output structure (created per folder)
-------------------------------------
    <subfolder>/
        mediapipe_output/
            labeled_videos/
                camera_0_synchronized_labeled.mp4
                ...
            data_2d/
                camera_0_synchronized_keypoints.npy    ← smoothed
                camera_0_synchronized_keypoints.npz
                camera_0_synchronized_metadata.json
                ...
            pose_estimation_summary.txt

Options
-------
    --model-complexity   0 (lite/fast), 1 (full/balanced), 2 (heavy/accurate) [default: 2]
    --det-conf           Min detection confidence 0.0-1.0 [default: 0.5]
    --track-conf         Min tracking confidence 0.0-1.0  [default: 0.5]
    --cutoff-freq        Butterworth cutoff in Hz          [default: 4.0]
    --filter-order       Butterworth filter order           [default: 4]
    --sampling-freq      Video frame rate in Hz             [default: 30.0]
    --no-smooth          Skip Butterworth smoothing
    --no-labeled-video   Skip generating labeled overlay videos (faster)
    --workers            Max folders to process in parallel [default: 1]
                         NOTE: each folder already parallelises across cameras
                         internally via multiprocessing, so keep this low (1-2)
                         unless you have many CPU cores and few cameras.
    --cam-workers        Max parallel cameras WITHIN a folder [default: auto]
    --video-subdir       Name of video subfolder [default: synchronized_videos]
    --output-subdir      Name of output subfolder [default: mediapipe_output]
    --pattern            Glob pattern for subfolder names [default: *]
    --verbose            Show camkit3d.pose2d's own logging + tqdm (off by default)

Example calls
-------------
    # Process all subfolders with defaults (model_complexity=2, 4Hz smoothing)
    python run_pose_batch.py /path/to/recordings

    # Fast mode: lower model complexity, no labeled videos
    python run_pose_batch.py /path/to/recordings --model-complexity 0 --no-labeled-video

    # Custom smoothing: 6 Hz cutoff at 60 FPS
    python run_pose_batch.py /path/to/recordings --cutoff-freq 6.0 --sampling-freq 60.0

    # Process only block_* folders, 2 folders at a time
    python run_pose_batch.py /path/to/recordings --pattern "block_*" --workers 2

    # No smoothing at all
    python run_pose_batch.py /path/to/recordings --no-smooth

    # Limit to 1 camera worker per folder (sequential within each folder)
    python run_pose_batch.py /path/to/recordings --cam-workers 1

    # See all internal camkit3d.pose2d output (debug)
    python run_pose_batch.py /path/to/recordings --verbose

Author: Batch runner for camkit3d.pose2d
"""

import argparse
import csv
import logging
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import numpy as np

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[warn] 'rich' not installed — pip install rich for the nice display.\n")

try:
    from camkit3d.pose2d import PoseProcessor
except ImportError as e:
    sys.exit(
        f"[error] Could not import camkit3d.pose2d: {e}\n"
        "Make sure camkit3d is installed (pip install -e .)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Silence camkit3d.pose2d internals in batch mode
# ═══════════════════════════════════════════════════════════════════════════════

def _suppress_inner_output():
    """
    Suppress camkit3d.pose2d's logger and tqdm bars so the rich table
    is the only thing on screen. Called unless --verbose is passed.

    camkit3d.pose2d calls logging.basicConfig(level=INFO) which
    configures the ROOT logger with a StreamHandler. So even if we
    mute the 'camkit3d.pose2d' named logger, messages still leak
    through the root handler. We need to silence both.

    - Raises the root logger to WARNING (catches everything)
    - Raises the 'camkit3d.pose2d' logger to WARNING (belt and braces)
    - Sets TQDM_DISABLE=1 env var so tqdm bars in subprocesses are no-ops
    """
    logging.getLogger().setLevel(logging.WARNING)              # root logger
    logging.getLogger("camkit3d.pose2d").setLevel(logging.WARNING)
    os.environ["TQDM_DISABLE"] = "1"


def _restore_inner_output():
    """Undo suppression."""
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("camkit3d.pose2d").setLevel(logging.INFO)
    os.environ.pop("TQDM_DISABLE", None)


# ═══════════════════════════════════════════════════════════════════════════════
#  Folder discovery & validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_folder(folder: Path, video_subdir: str = "synchronized_videos") -> tuple[bool, str]:
    """
    Check whether a subfolder has videos to process.

    Returns (True, "") if valid, or (False, reason) if not.
    """
    if not folder.is_dir():
        return False, "not a directory"

    video_dir = folder / video_subdir
    if not video_dir.is_dir():
        return False, f"missing '{video_subdir}/' subfolder"

    video_files = (
        list(video_dir.glob("*.mp4"))
        + list(video_dir.glob("*.avi"))
        + list(video_dir.glob("*.mov"))
    )
    if not video_files:
        return False, f"no video files in {video_subdir}/"

    return True, ""


def discover_folders(
    root: Path,
    pattern: str = "*",
    video_subdir: str = "synchronized_videos",
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """
    Scan subfolders of *root* matching *pattern* for valid recording layouts.
    """
    valid = []
    skipped = []

    candidates = sorted(p for p in root.iterdir() if p.is_dir() and p.match(pattern))
    for folder in candidates:
        ok, reason = validate_folder(folder, video_subdir)
        if ok:
            valid.append(folder)
        else:
            skipped.append((folder, reason))

    return valid, skipped


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-folder status tracker (thread-safe)
# ═══════════════════════════════════════════════════════════════════════════════

class FolderStatus:
    """Thread-safe status for one folder — written by worker, read by display."""

    def __init__(self, name: str, n_cameras: int):
        self.name = name
        self.state = "waiting"           # waiting | pose | smoothing | done | error
        self.phase = ""
        self.n_cameras = n_cameras       # known at discovery time
        self.cameras_done = 0
        self.detection_rates: list[float] = []
        self.start_time: float | None = None
        self.end_time: float | None = None
        self._lock = Lock()

    # ── worker writes ──────────────────────────────────────────────────────
    def mark_pose(self):
        with self._lock:
            self.state = "pose"
            self.cameras_done = 0
            self.phase = "estimating pose"
            self.start_time = self.start_time or time.monotonic()

    def camera_finished(self, detection_rate: float):
        with self._lock:
            self.cameras_done += 1
            self.detection_rates.append(detection_rate)

    def mark_smoothing(self):
        with self._lock:
            self.state = "smoothing"
            self.phase = "butterworth filter"

    def mark_done(self):
        with self._lock:
            self.state = "done"
            avg_det = (
                sum(self.detection_rates) / len(self.detection_rates)
                if self.detection_rates else 0.0
            )
            self.phase = f"{avg_det:.0%} avg det"
            self.end_time = time.monotonic()

    def mark_error(self, msg: str):
        with self._lock:
            self.state = "error"
            self.phase = msg[:80]
            self.end_time = time.monotonic()

    # ── display reads ──────────────────────────────────────────────────────
    def snapshot(self):
        with self._lock:
            if self.start_time is None:
                elapsed = 0.0
            elif self.end_time is not None:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.monotonic() - self.start_time
            return dict(
                name=self.name,
                state=self.state,
                phase=self.phase,
                cameras_done=self.cameras_done,
                n_cameras=self.n_cameras,
                elapsed=elapsed,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FolderResult:
    """Everything we want to know about a processed folder."""
    folder: Path
    ok: bool = False
    error: str = ""
    n_cameras: int = 0
    elapsed_s: float = 0.0
    smoothing_applied: bool = False
    camera_metrics: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker: process one folder
# ═══════════════════════════════════════════════════════════════════════════════

def process_folder(
    folder: Path,
    status: FolderStatus,
    video_subdir: str,
    output_subdir: str,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    smooth_landmarks: bool,
    save_labeled_videos: bool,
    cam_workers: Optional[int],
    do_smooth: bool,
    cutoff_freq: float,
    sampling_freq: float,
    filter_order: int,
) -> FolderResult:
    """
    End-to-end processing for one folder:
      1. MediaPipe pose estimation (parallel across cameras)
      2. Butterworth smoothing of all keypoints
    """
    result = FolderResult(folder=folder, n_cameras=status.n_cameras)
    t0 = time.perf_counter()

    try:
        input_video_dir = folder / video_subdir
        output_dir = folder / output_subdir

        status.mark_pose()

        # ── Step 1: Pose estimation ────────────────────────────────────────
        processor = PoseProcessor(
            input_dir=str(input_video_dir),
            output_dir=str(output_dir),
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            max_workers=cam_workers,
        )

        metrics_all = processor.process_all_videos(save_labeled_videos=save_labeled_videos)

        # Collect per-camera metrics and flag anomalies
        for cam_name, m in metrics_all.items():
            result.camera_metrics[cam_name] = {
                "total_frames": m.total_frames,
                "frames_with_detection": m.frames_with_detection,
                "detection_rate": m.detection_rate,
                "avg_confidence": m.avg_confidence,
                "confidence_std": m.confidence_std,
                "min_confidence": m.min_confidence,
                "max_confidence": m.max_confidence,
            }

            # Report camera completion to status (updates the live table)
            status.camera_finished(m.detection_rate)

            # Anomaly checks
            if m.detection_rate < 0.5:
                result.warnings.append(f"{cam_name}: LOW detection rate {m.detection_rate:.1%}")
            if m.detection_rate < 0.1:
                result.warnings.append(f"{cam_name}: CRITICAL — almost no detections ({m.detection_rate:.1%})")
            if m.avg_confidence < 0.3:
                result.warnings.append(f"{cam_name}: LOW avg confidence {m.avg_confidence:.3f}")
            if m.total_frames == 0:
                result.warnings.append(f"{cam_name}: 0 frames read — corrupt video?")

        # ── Step 2: Butterworth smoothing ──────────────────────────────────
        if do_smooth:
            status.mark_smoothing()
            data_2d_dir = output_dir / "data_2d"
            keypoint_files = sorted(data_2d_dir.glob("*_keypoints.npy"))

            if not keypoint_files:
                result.warnings.append("No keypoint .npy files found for smoothing")
            else:
                smoother = PoseProcessor(input_dir=".", output_dir=".")
                for kp_file in keypoint_files:
                    keypoints = np.load(kp_file)
                    try:
                        smoothed = smoother.smooth_keypoints_butterworth(
                            keypoints,
                            cutoff_freq=cutoff_freq,
                            sampling_freq=sampling_freq,
                            order=filter_order,
                        )
                        np.save(kp_file, smoothed)
                    except Exception as e:
                        cam = kp_file.stem.replace("_keypoints", "")
                        result.warnings.append(f"{cam}: smoothing failed — {e}")

                result.smoothing_applied = True

        # ── Done ───────────────────────────────────────────────────────────
        result.ok = True
        result.elapsed_s = time.perf_counter() - t0
        status.mark_done()

    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = time.perf_counter() - t0
        status.mark_error(str(exc)[:80])
        traceback.print_exc()

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Rich live table
# ═══════════════════════════════════════════════════════════════════════════════

# Muted colour palette — works on dark and light terminals
C = {
    "waiting":   "dim",
    "pose":      "#5ea8f5",       # soft blue
    "smoothing": "#c084fc",       # lavender
    "done":      "#4ade80",       # mint green
    "error":     "#f87171",       # coral red
    "accent":    "#fbbf24",       # warm amber
    "muted":     "#94a3b8",       # slate grey
}

ICON = {
    "waiting":   " · ",
    "pose":      " ◉ ",
    "smoothing": " ≋ ",
    "done":      " ✓ ",
    "error":     " ✗ ",
}


def _bar(done: int, total: int, width: int = 14) -> str:
    """Compact block progress bar."""
    if total <= 0:
        return "·" * width
    frac = min(done / total, 1.0)
    filled = int(frac * width)
    return "━" * filled + "╌" * (width - filled)


def build_table(
    statuses: list[FolderStatus],
    n_workers: int,
    config_line: str,
) -> Table:
    """Build the live-updating rich table."""

    # Count how many individual videos are being processed RIGHT NOW
    total_vids_active = 0
    for s in statuses:
        snap = s.snapshot()
        if snap["state"] == "pose":
            total_vids_active += snap["n_cameras"] - snap["cameras_done"]

    tbl = Table(
        box=box.ROUNDED,
        expand=True,
        title=(
            f"[bold]  BATCH 2D POSE ESTIMATION  [/bold]\n"
            f"[{C['muted']}]{config_line}[/{C['muted']}]"
        ),
        title_style="",
        border_style=C["muted"],
        header_style=f"bold {C['accent']}",
        show_lines=False,
        pad_edge=True,
        padding=(0, 1),
    )
    tbl.add_column("",          width=3,  no_wrap=True)
    tbl.add_column("Folder",    ratio=4,  no_wrap=True)
    tbl.add_column("Cameras",   ratio=4,  no_wrap=True)
    tbl.add_column("Status",    ratio=3,  no_wrap=True)
    tbl.add_column("Time",      width=8,  justify="right", no_wrap=True)

    for s in statuses:
        snap = s.snapshot()
        state = snap["state"]
        style = C[state]
        icon = ICON[state]

        nd, nt = snap["cameras_done"], snap["n_cameras"]

        # ── Cameras column ─────────────────────────────────────────────
        if state == "pose":
            bar = _bar(nd, nt)
            cam_str = f"[{style}]{bar}[/{style}]  [{C['muted']}]{nd}/{nt} done[/{C['muted']}]"
        elif state == "smoothing":
            cam_str = f"[{style}]{'≋ ' * 7}[/{style}]  [{C['muted']}]{nt} cams[/{C['muted']}]"
        elif state == "done":
            cam_str = f"[{style}]{nt} cams[/{style}]"
        elif state == "error":
            cam_str = f"[{style}]—[/{style}]"
        else:
            cam_str = f"[{C['muted']}]{nt} cams[/{C['muted']}]"

        # ── Status column ──────────────────────────────────────────────
        if state == "waiting":
            status_str = f"[{C['muted']}]queued[/{C['muted']}]"
        elif state == "pose":
            active_now = nt - nd
            status_str = f"[{style}]{active_now} video{'s' if active_now != 1 else ''} processing[/{style}]"
        elif state == "smoothing":
            status_str = f"[{style}]filtering keypoints[/{style}]"
        elif state == "done":
            status_str = f"[{style}]{snap['phase']}[/{style}]"
        elif state == "error":
            status_str = f"[{style}]{snap['phase'][:40]}[/{style}]"
        else:
            status_str = ""

        # ── Time column ────────────────────────────────────────────────
        elapsed = snap["elapsed"]
        if elapsed > 0:
            mins, secs = divmod(elapsed, 60)
            elapsed_str = f"[{C['muted']}]{int(mins):02d}:{secs:04.1f}[/{C['muted']}]"
        else:
            elapsed_str = f"[{C['muted']}]  ·  [/{C['muted']}]"

        tbl.add_row(
            f"[{style}]{icon}[/{style}]",
            f"[{style}]{snap['name']}[/{style}]",
            cam_str,
            status_str,
            elapsed_str,
        )

    # ── Footer ─────────────────────────────────────────────────────────
    n_done = sum(1 for s in statuses if s.snapshot()["state"] == "done")
    n_err = sum(1 for s in statuses if s.snapshot()["state"] == "error")
    n_active = sum(1 for s in statuses if s.snapshot()["state"] in ("pose", "smoothing"))

    parts = [
        f"[{C['muted']}]folders[/{C['muted']}] [{C['done']}]{n_done}[/{C['done']}][{C['muted']}]/{len(statuses)}[/{C['muted']}]",
        f"[{C['muted']}]active[/{C['muted']}] [{C['pose']}]{n_active}[/{C['pose']}]",
    ]
    if total_vids_active > 0:
        parts.append(
            f"[{C['accent']}]{total_vids_active} video{'s' if total_vids_active != 1 else ''} processing now[/{C['accent']}]"
        )
    if n_err:
        parts.append(f"[{C['error']}]{n_err} error{'s' if n_err != 1 else ''}[/{C['error']}]")

    tbl.caption = Text.from_markup("  ·  ".join(parts))
    return tbl


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary CSV writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary_csv(results: list[FolderResult], output_path: Path):
    """
    Write a summary CSV with one row per camera per folder.

    Columns: folder, camera, status, total_frames, frames_with_detection,
    detection_rate, avg_confidence, confidence_std, min_confidence,
    max_confidence, elapsed_s, smoothing_applied, warnings
    """
    rows = []
    for r in results:
        folder_name = r.folder.name

        if not r.ok:
            rows.append({
                "folder": folder_name, "camera": "ALL", "status": "ERROR",
                "total_frames": 0, "frames_with_detection": 0,
                "detection_rate": 0.0, "avg_confidence": 0.0,
                "confidence_std": 0.0, "min_confidence": 0.0,
                "max_confidence": 0.0, "elapsed_s": round(r.elapsed_s, 1),
                "smoothing_applied": False, "warnings": r.error,
            })
            continue

        for cam_name, m in r.camera_metrics.items():
            cam_warns = [w for w in r.warnings if cam_name in w]
            rows.append({
                "folder": folder_name, "camera": cam_name, "status": "OK",
                "total_frames": m["total_frames"],
                "frames_with_detection": m["frames_with_detection"],
                "detection_rate": round(m["detection_rate"], 4),
                "avg_confidence": round(m["avg_confidence"], 4),
                "confidence_std": round(m["confidence_std"], 4),
                "min_confidence": round(m["min_confidence"], 4),
                "max_confidence": round(m["max_confidence"], 4),
                "elapsed_s": round(r.elapsed_s, 1),
                "smoothing_applied": r.smoothing_applied,
                "warnings": "; ".join(cam_warns) if cam_warns else "",
            })

        # Folder-level warnings not tied to a specific camera
        general_warns = [
            w for w in r.warnings
            if not any(cam in w for cam in r.camera_metrics)
        ]
        if general_warns:
            rows.append({
                "folder": folder_name, "camera": "FOLDER_NOTE", "status": "OK",
                "total_frames": 0, "frames_with_detection": 0,
                "detection_rate": 0.0, "avg_confidence": 0.0,
                "confidence_std": 0.0, "min_confidence": 0.0,
                "max_confidence": 0.0, "elapsed_s": round(r.elapsed_s, 1),
                "smoothing_applied": r.smoothing_applied,
                "warnings": "; ".join(general_warns),
            })

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch(
    root_dir: Path,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    smooth_landmarks: bool = True,
    save_labeled_videos: bool = True,
    cam_workers: Optional[int] = None,
    do_smooth: bool = True,
    cutoff_freq: float = 4.0,
    sampling_freq: float = 30.0,
    filter_order: int = 4,
    n_workers: int = 1,
    video_subdir: str = "synchronized_videos",
    output_subdir: str = "mediapipe_output",
    pattern: str = "*",
    verbose: bool = False,
):
    """Discover folders, process them, write summary CSV."""

    # ── Suppress inner noise unless verbose ────────────────────────────────
    if not verbose:
        _suppress_inner_output()

    # ── Config string (shown in table header and banner) ───────────────────
    model_label = {0: "lite", 1: "full", 2: "heavy"}[model_complexity]
    config_line = f"model {model_label}  ·  det {min_detection_confidence}  ·  track {min_tracking_confidence}"
    if do_smooth:
        config_line += f"  ·  butter {cutoff_freq} Hz order-{filter_order} @ {sampling_freq} fps"
    else:
        config_line += "  ·  no smoothing"

    # ── Banner ─────────────────────────────────────────────────────────────
    if HAS_RICH:
        console = Console()
        console.print()
        console.print(Panel.fit(
            f"[bold]BATCH 2D POSE ESTIMATION[/bold]\n"
            f"[{C['muted']}]MediaPipe Pose  ·  Butterworth Smooth[/{C['muted']}]",
            border_style=C["accent"],
            padding=(1, 4),
        ))
        console.print()
        console.print(f"  [{C['muted']}]root[/{C['muted']}]          {root_dir}")
        console.print(f"  [{C['muted']}]pattern[/{C['muted']}]       {pattern}")
        console.print(f"  [{C['muted']}]model[/{C['muted']}]         {model_label} (complexity={model_complexity})")
        console.print(f"  [{C['muted']}]detection[/{C['muted']}]     {min_detection_confidence}")
        console.print(f"  [{C['muted']}]tracking[/{C['muted']}]      {min_tracking_confidence}")
        console.print(f"  [{C['muted']}]labeled vid[/{C['muted']}]   {'yes' if save_labeled_videos else 'no'}")
        if do_smooth:
            console.print(f"  [{C['muted']}]smoothing[/{C['muted']}]     butterworth {cutoff_freq} Hz, order {filter_order}, fs={sampling_freq} Hz")
        else:
            console.print(f"  [{C['muted']}]smoothing[/{C['muted']}]     disabled")
        console.print(f"  [{C['muted']}]workers[/{C['muted']}]       {n_workers} folder(s)  ·  {cam_workers or 'auto'} cam(s)/folder")
        console.print()
    else:
        print(f"\n  BATCH 2D POSE ESTIMATION")
        print(f"  {config_line}\n")

    # ── Discover folders ───────────────────────────────────────────────────
    folders, skipped = discover_folders(root_dir, pattern, video_subdir)

    if skipped:
        msg = f"  {len(skipped)} subfolder(s) skipped:"
        if HAS_RICH:
            console.print(f"[{C['accent']}]{msg}[/{C['accent']}]")
            for folder, reason in skipped:
                console.print(f"    [{C['muted']}]✗  {folder.name}/  —  {reason}[/{C['muted']}]")
            console.print()
        else:
            print(msg)
            for folder, reason in skipped:
                print(f"    ✗  {folder.name}/  —  {reason}")
            print()

    if not folders:
        sys.exit(
            f"[error] No valid folders found under {root_dir}\n"
            f"        Expected: <subfolder>/{video_subdir}/*.mp4|avi|mov"
        )

    # Count cameras per folder (needed for status objects and display)
    folder_cam_counts = {}
    for f in folders:
        vdir = f / video_subdir
        n = len(list(vdir.glob("*.mp4")) + list(vdir.glob("*.avi")) + list(vdir.glob("*.mov")))
        folder_cam_counts[f] = n

    total_videos = sum(folder_cam_counts.values())

    if HAS_RICH:
        console.print(
            f"  [{C['done']}]✓[/{C['done']}]  "
            f"[bold]{len(folders)}[/bold] folder(s)  ·  "
            f"[bold]{total_videos}[/bold] total videos\n"
        )
        for f in folders:
            n = folder_cam_counts[f]
            console.print(f"    [{C['muted']}]╸[/{C['muted']}] {f.name}  [{C['muted']}]({n} video{'s' if n != 1 else ''})[/{C['muted']}]")
        console.print()
    else:
        print(f"  ✓  Found {len(folders)} folder(s), {total_videos} total videos\n")
        for f in folders:
            print(f"    · {f.name}  ({folder_cam_counts[f]} videos)")
        print()

    # ── Prepare status objects ─────────────────────────────────────────────
    statuses = [FolderStatus(f.name, folder_cam_counts[f]) for f in folders]
    status_map = {f: s for f, s in zip(folders, statuses)}

    all_results: list[FolderResult] = []

    # ── Launch workers ─────────────────────────────────────────────────────
    # ThreadPoolExecutor at the folder level is fine: the heavy lifting
    # (MediaPipe inference) happens in multiprocessing subprocesses inside
    # camkit3d.pose2d, so the GIL is not a bottleneck here.
    executor = ThreadPoolExecutor(max_workers=n_workers)
    futures = {
        executor.submit(
            process_folder,
            folder, status_map[folder],
            video_subdir, output_subdir,
            model_complexity, min_detection_confidence, min_tracking_confidence,
            smooth_landmarks, save_labeled_videos, cam_workers,
            do_smooth, cutoff_freq, sampling_freq, filter_order,
        ): folder
        for folder in folders
    }

    # ── Live display or plain fallback ─────────────────────────────────────
    if HAS_RICH:
        results_lock = Lock()

        def collector():
            for fut in as_completed(futures):
                with results_lock:
                    all_results.append(fut.result())

        t = Thread(target=collector, daemon=True)
        t.start()

        with Live(
            build_table(statuses, n_workers, config_line),
            console=console,
            refresh_per_second=8,
            transient=False,
        ) as live:
            while t.is_alive() or len(all_results) < len(folders):
                live.update(build_table(statuses, n_workers, config_line))
                time.sleep(0.12)
            live.update(build_table(statuses, n_workers, config_line))

        t.join()
        # One final clean render
        console.print()
        console.print(build_table(statuses, n_workers, config_line))
    else:
        for fut in as_completed(futures):
            r = fut.result()
            all_results.append(r)
            s = status_map[futures[fut]]
            snap = s.snapshot()
            icon = "✓" if snap["state"] == "done" else "✗"
            print(f"  [{icon}] {snap['name']:30s}  {snap['phase']:30s}  {snap['elapsed']:.1f}s")

    executor.shutdown(wait=True)

    # ── Summary CSV ────────────────────────────────────────────────────────
    csv_path = root_dir / "batch_pose_summary.csv"
    write_summary_csv(all_results, csv_path)

    # ── Final report ───────────────────────────────────────────────────────
    ok = sum(1 for r in all_results if r.ok)
    err = len(all_results) - ok
    total_warnings = sum(len(r.warnings) for r in all_results)

    if HAS_RICH:
        console.print()
        border = C["done"] if err == 0 else C["error"]
        console.print(Panel.fit(
            f"[bold]COMPLETE[/bold]    "
            f"[{C['done']}]{ok} succeeded[/{C['done']}]    "
            f"[{C['error']}]{err} failed[/{C['error']}]    "
            f"[{C['muted']}]{total_warnings} warnings[/{C['muted']}]\n"
            f"[{C['muted']}]summary csv  →  {csv_path}[/{C['muted']}]",
            border_style=border,
            padding=(1, 3),
        ))

        if total_warnings:
            console.print(f"\n  [{C['accent']}]Warnings[/{C['accent']}]")
            for r in all_results:
                if r.warnings:
                    console.print(f"\n    [{C['muted']}]{r.folder.name}/[/{C['muted']}]")
                    for w in r.warnings:
                        console.print(f"      [{C['accent']}]⚠[/{C['accent']}]  {w}")

        if err:
            console.print(f"\n  [{C['error']}]Errors[/{C['error']}]")
            for r in all_results:
                if not r.ok:
                    console.print(f"    [{C['error']}]✗[/{C['error']}]  {r.folder.name}: {r.error}")

        console.print()
    else:
        print(f"\n  Complete: {ok} succeeded, {err} failed, {total_warnings} warnings")
        print(f"  Summary CSV: {csv_path}\n")

    # Restore logging if we suppressed it
    if not verbose:
        _restore_inner_output()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Batch 2D pose estimation with MediaPipe + Butterworth smoothing.\n\n"
            "Scans subfolders of root_dir for a video subfolder (default:\n"
            "'synchronized_videos/') and runs pose estimation on all videos,\n"
            "then applies a low-pass Butterworth filter to smooth keypoints.\n\n"
            "A summary CSV is saved to root_dir/batch_pose_summary.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pose_batch.py /path/to/recordings\n"
            "  python run_pose_batch.py /path/to/recordings --model-complexity 0 --no-labeled-video\n"
            "  python run_pose_batch.py /path/to/recordings --cutoff-freq 6.0 --sampling-freq 60.0\n"
            "  python run_pose_batch.py /path/to/recordings --pattern 'block_*' --workers 2\n"
            "  python run_pose_batch.py /path/to/recordings --no-smooth\n"
            "  python run_pose_batch.py /path/to/recordings --verbose\n"
        ),
    )

    p.add_argument("root_dir", type=Path, help="Root directory containing recording subfolders")

    pose = p.add_argument_group("Pose estimation")
    pose.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2],
                      help="0=lite/fast, 1=full/balanced, 2=heavy/accurate (default: 2)")
    pose.add_argument("--det-conf", type=float, default=0.5,
                      help="Min detection confidence 0.0-1.0 (default: 0.5)")
    pose.add_argument("--track-conf", type=float, default=0.5,
                      help="Min tracking confidence 0.0-1.0 (default: 0.5)")
    pose.add_argument("--no-labeled-video", action="store_true",
                      help="Skip generating labeled overlay videos (faster)")

    smooth = p.add_argument_group("Butterworth smoothing")
    smooth.add_argument("--no-smooth", action="store_true",
                        help="Skip Butterworth smoothing entirely")
    smooth.add_argument("--cutoff-freq", type=float, default=4.0,
                        help="Low-pass cutoff in Hz (default: 4.0)")
    smooth.add_argument("--filter-order", type=int, default=4,
                        help="Filter order (default: 4)")
    smooth.add_argument("--sampling-freq", type=float, default=30.0,
                        help="Video frame rate in Hz (default: 30.0)")

    par = p.add_argument_group("Parallelism")
    par.add_argument("--workers", type=int, default=1,
                     help="Max folders simultaneously (default: 1). Each folder "
                          "already parallelises cameras internally.")
    par.add_argument("--cam-workers", type=int, default=None,
                     help="Max parallel cameras per folder (default: auto)")

    layout = p.add_argument_group("Directory layout")
    layout.add_argument("--video-subdir", type=str, default="synchronized_videos",
                        help="Video subfolder name (default: synchronized_videos)")
    layout.add_argument("--output-subdir", type=str, default="mediapipe_output",
                        help="Output subfolder name (default: mediapipe_output)")
    layout.add_argument("--pattern", type=str, default="*",
                        help="Glob pattern for subfolder names (default: *)")

    p.add_argument("--verbose", action="store_true",
                   help="Show camkit3d.pose2d's own logging and tqdm bars (off by default)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_batch(
        root_dir=args.root_dir,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
        smooth_landmarks=True,
        save_labeled_videos=not args.no_labeled_video,
        cam_workers=args.cam_workers,
        do_smooth=not args.no_smooth,
        cutoff_freq=args.cutoff_freq,
        sampling_freq=args.sampling_freq,
        filter_order=args.filter_order,
        n_workers=args.workers,
        video_subdir=args.video_subdir,
        output_subdir=args.output_subdir,
        pattern=args.pattern,
        verbose=args.verbose,
    )
