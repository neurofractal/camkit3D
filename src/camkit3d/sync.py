"""Timestamp-based multi-camera video synchronization and plotting for CamKit3D.

When several cameras record at once, each has its own internal clock and its
own frame drops and timing jitter, so frame 100 on one camera is not the same
instant as frame 100 on another. Lining up frame numbers therefore does NOT
give temporally aligned video.

This module resynchronises the recordings against a common reference clock and
provides diagnostics to confirm the result.

Key features:

- Ideal-clock resampling. Rather than aligning to a chosen master camera,
  it builds a perfect timing grid at the target FPS over the window common
  to all cameras, then picks each camera's nearest real frame by timestamp.
  Every camera is treated equally and the output tracks the desired FPS as
  closely as the data allow.
- Drop and jitter handling. Because matching is by capture time, a dropped
  frame is covered by reusing the nearest neighbour and a fast-running camera
  simply skips one, so all outputs end up with identical frame counts.
- Built-in diagnostics. Timing-error statistics plus per-camera plots of
  raw vs synchronised timing, dropped-frame counts, and error distributions
  let you judge sync quality at a glance.
- Tiled review video. Composites all synchronised feeds into one labelled
  grid for quick visual confirmation.

Quick-start:

    from camkit3d.sync import vid_sync

    trial = "./recordings/2026-02-04_12-00-00"
    results, figs = vid_sync(trial, target_fps=30.0)

Dependencies: numpy, opencv-python, matplotlib

Author: Dr. Robert Seymour, OHBA, University of Oxford
License: GNU General Public License v3, 2026
"""

from pathlib import Path
import math
import re
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _find_nearest_frame_indices(
    target_times: np.ndarray,
    source_times: np.ndarray,
) -> np.ndarray:
    """Return the index in *source_times* closest to each value in *target_times*.

    This is the core of the timestamp-based sync approach.  For every tick of
    the ideal clock (target_times) we need to pick the single frame from a
    real camera (source_times) whose capture time is nearest.

    Algorithm (brute-force, O(N·M)):
        For each target time tₖ compute |source_times − tₖ| and take argmin.

    This is simple and perfectly adequate for typical video lengths (a few
    thousand frames).  A binary-search approach (np.searchsorted) could be
    used for very long recordings if needed.

    Parameters
    ----------
    target_times : array-like, shape (N,)
        The ideal clock ticks we want to map *to* (seconds).
    source_times : array-like, shape (M,)
        The real camera timestamps we are mapping *from* (seconds).

    Returns
    -------
    indices : np.ndarray, shape (N,), dtype int64
        ``indices[i]`` is the position in *source_times* whose value is
        closest to ``target_times[i]``.
    """
    target_times = np.asarray(target_times).ravel()
    source_times = np.asarray(source_times).ravel()

    indices = np.zeros(len(target_times), dtype=np.int64)
    for i, t in enumerate(target_times):
        indices[i] = np.argmin(np.abs(source_times - t))

    return indices


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNCHRONISATION
# ═══════════════════════════════════════════════════════════════════════════════

def synchronize_videos_to_ideal_fps(
    trial_folder: str,
    target_fps: float = 30.0,
    raw_videos_subdir: str = "raw_videos",
    out_subdir: str = "synchronized_videos",
    max_time_diff_ms: float = 100.0,
    progress_callback=None,
) -> Dict:
    """Synchronise every camera to an ideal clock at *target_fps*.

    How it works - step by step
    ---------------------------
    1. **Load timestamps** - Each camera must have a corresponding
       ``camera_<id>_timestamps.npy`` file in *trial_folder* containing a 1-D
       array of POSIX-style seconds (one entry per captured frame).

    2. **Find the common time window** - The overlap region is defined as
       ``[max(start times), min(end times)]`` across all cameras.  Only this
       window is included in the output so every camera contributes data for
       every output frame.

    3. **Build an ideal timing grid** - A perfect array of timestamps at
       exactly ``1 / target_fps`` spacing is created over the common window.
       This is the "reference clock" that all cameras will be mapped onto.

    4. **Map real frames → ideal ticks** - For each ideal tick and each camera
       the nearest real frame (by timestamp) is selected.  A single source
       frame may be reused (if the camera dropped a frame) or skipped (if the
       camera ran slightly fast).

    5. **Write synchronised videos** - New MP4 files are written where frame
       *n* of every camera corresponds to the same ideal time point.

    6. **Verify & report** - Output videos are re-opened and their frame
       counts are checked.  Timing-error statistics are returned so you can
       judge sync quality.

    Why align to an *ideal* clock rather than to cam 0?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    No single camera is perfect - all have jitter and drops.  Using an ideal
    grid treats every camera equally and produces output whose timing is as
    close to the desired FPS as possible.

    Parameters
    ----------
    trial_folder : str
        Path to the trial directory.  Must contain ``camera_*_timestamps.npy``
        files and a subdirectory of raw video files.
    target_fps : float
        Desired output frame rate (default 30.0).
    raw_videos_subdir : str
        Name of the subdirectory that holds the raw camera videos.
    out_subdir : str
        Name of the subdirectory where synchronised videos will be written.
    max_time_diff_ms : float
        If a matched frame is further than this many milliseconds from the
        ideal tick a warning is printed (default 100 ms).
    progress_callback : callable or None
        Optional ``callback(frames_done, total_frames)`` for progress bars.

    Returns
    -------
    results : dict
        Keys include ``'frame_count'``, ``'ideal_times'``, ``'sync_metrics'``,
        ``'frame_mappings'``, ``'timestamps_original'``, ``'output_files'``,
        and more - everything needed by the plotting functions.
    """
    trial_folder = Path(trial_folder)
    raw_dir = trial_folder / raw_videos_subdir
    out_dir = trial_folder / out_subdir

    print(f"\n{'=' * 70}")
    print(f"Timestamp-based Synchronization to Ideal {target_fps} FPS Clock")
    print(f"{'=' * 70}")
    print(f"Trial folder: {trial_folder}")
    print(f"Raw videos:   {raw_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Target FPS:   {target_fps}")

    # -- 1. Load timestamps ---------------------------------------------------
    print(f"\n[1] Loading timestamps...")

    ts_files = list(trial_folder.glob("camera_*_timestamps.npy"))
    if not ts_files:
        raise FileNotFoundError(f"No timestamp files found in {trial_folder}")

    cam_re = re.compile(r"camera_(\d+)_timestamps\.npy$")
    timestamps: Dict[int, np.ndarray] = {}

    for ts_file in ts_files:
        match = cam_re.search(ts_file.name)
        if match:
            cam_id = int(match.group(1))
            ts = np.load(ts_file)
            timestamps[cam_id] = ts
            print(f"  Camera {cam_id}: {len(ts)} timestamps, "
                  f"range: [{ts[0]:.3f}, {ts[-1]:.3f}]s")

    camera_ids = sorted(timestamps.keys())

    # -- 2. Global time range (common overlap) ---------------------------------
    print(f"\n[2] Finding global time range...")

    global_start = max(ts[0] for ts in timestamps.values())   # latest start
    global_end   = min(ts[-1] for ts in timestamps.values())   # earliest end

    print(f"  Global start time: {global_start:.3f}s")
    print(f"  Global end time:   {global_end:.3f}s")
    print(f"  Duration:          {global_end - global_start:.3f}s")

    # -- 3. Ideal timing grid --------------------------------------------------
    print(f"\n[3] Creating ideal {target_fps} FPS timing grid...")

    frame_duration = 1.0 / target_fps
    num_ideal_frames = int(np.floor((global_end - global_start) / frame_duration)) + 1
    ideal_times = global_start + np.arange(num_ideal_frames) * frame_duration
    ideal_times = ideal_times[ideal_times <= global_end]
    num_ideal_frames = len(ideal_times)

    print(f"  Ideal frame count: {num_ideal_frames}")
    print(f"  Ideal duration:    {ideal_times[-1] - ideal_times[0]:.3f}s")

    # -- 4. Locate raw video files ---------------------------------------------
    print(f"\n[4] Finding video files...")

    video_files: Dict[int, Path] = {}
    for cam_id in camera_ids:
        for ext in (".avi", ".mp4", ".mov"):
            video_path = raw_dir / f"camera_{cam_id}{ext}"
            if video_path.exists():
                video_files[cam_id] = video_path
                print(f"  Camera {cam_id}: {video_path.name}")
                break
        if cam_id not in video_files:
            raise FileNotFoundError(f"Video file not found for camera {cam_id}")

    # -- 5. Video properties (resolution from first camera) --------------------
    print(f"\n[5] Getting video properties...")

    ref_cam_id = camera_ids[0]
    cap_ref = cv2.VideoCapture(str(video_files[ref_cam_id]))
    if not cap_ref.isOpened():
        raise RuntimeError(f"Cannot open camera {ref_cam_id} video")

    width_out  = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_out = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_ref.release()

    print(f"  Output FPS:        {target_fps:.2f}")
    print(f"  Output resolution: {width_out}x{height_out} (from camera {ref_cam_id})")
    print(f"  Output frames:     {num_ideal_frames}")

    # -- 6. Build frame mapping (nearest-neighbour in time) --------------------
    print(f"\n[6] Building frame mappings to ideal clock...")

    frame_maps: Dict[int, np.ndarray] = {}
    sync_metrics: Dict[int, Dict] = {}

    for cam_id in camera_ids:
        cam_times = timestamps[cam_id]
        nearest_indices = _find_nearest_frame_indices(ideal_times, cam_times)

        matched_times = cam_times[nearest_indices]
        time_diffs = np.abs(matched_times - ideal_times) * 1000  # → ms

        frame_maps[cam_id] = nearest_indices
        sync_metrics[cam_id] = {
            "time_diffs_ms": time_diffs,
            "mean_diff_ms":  float(np.mean(time_diffs)),
            "max_diff_ms":   float(np.max(time_diffs)),
            "rms_diff_ms":   float(np.sqrt(np.mean(time_diffs ** 2))),
            "p95_diff_ms":   float(np.percentile(time_diffs, 95)),
            "p99_diff_ms":   float(np.percentile(time_diffs, 99)),
        }

        print(f"  Camera {cam_id}:")
        print(f"    Mean time diff: {sync_metrics[cam_id]['mean_diff_ms']:.2f} ms")
        print(f"    Max time diff:  {sync_metrics[cam_id]['max_diff_ms']:.2f} ms")
        print(f"    RMS time diff:  {sync_metrics[cam_id]['rms_diff_ms']:.2f} ms")
        print(f"    P95 time diff:  {sync_metrics[cam_id]['p95_diff_ms']:.2f} ms")

        bad_matches = np.sum(time_diffs > max_time_diff_ms)
        if bad_matches > 0:
            print(f"    ⚠ WARNING: {bad_matches}/{len(time_diffs)} frames "
                  f"exceed {max_time_diff_ms}ms threshold")

    # -- 7. Write synchronised videos ------------------------------------------
    print(f"\n[7] Writing synchronized videos...")
    out_dir.mkdir(parents=True, exist_ok=True)

    captures: Dict[int, cv2.VideoCapture] = {}
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(str(video_files[cam_id]))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for camera {cam_id}")
        captures[cam_id] = cap

    writers: Dict[int, cv2.VideoWriter] = {}
    output_paths: Dict[int, Path] = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for cam_id in camera_ids:
        output_path = out_dir / f"camera_{cam_id}_synchronized.mp4"
        writer = cv2.VideoWriter(
            str(output_path), fourcc, target_fps, (width_out, height_out),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for camera {cam_id}")
        writers[cam_id] = writer
        output_paths[cam_id] = output_path
        print(f"  Camera {cam_id}: {output_path.name}")

    # -- 8. Process frames -----------------------------------------------------
    print(f"\n[8] Processing {num_ideal_frames} frames...")

    try:
        for frame_idx in range(num_ideal_frames):
            if progress_callback is not None:
                progress_callback(frame_idx, num_ideal_frames)
            elif frame_idx % 100 == 0:
                pct = (frame_idx / num_ideal_frames) * 100
                print(f"  Progress: {pct:.1f}% "
                      f"({frame_idx}/{num_ideal_frames})", end="\r")

            for cam_id in camera_ids:
                src_idx = int(frame_maps[cam_id][frame_idx])
                cap = captures[cam_id]
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError(
                        f"Failed to read frame {src_idx} from camera {cam_id} "
                        f"(frame {frame_idx}/{num_ideal_frames} in sync sequence)"
                    )

                if frame.shape[1] != width_out or frame.shape[0] != height_out:
                    frame = cv2.resize(frame, (width_out, height_out))

                writers[cam_id].write(frame)

        print(f"\n  Progress: 100.0% ({num_ideal_frames}/{num_ideal_frames})")
        print(f"  ✓ All frames written")

    finally:
        for cap in captures.values():
            cap.release()
        for writer in writers.values():
            writer.release()

    # -- 9. Save frame-mapping indices -----------------------------------------
    print(f"\n[9] Saving frame mappings...")

    npz_path = trial_folder / "frame_mappings_to_ideal_fps.npz"
    np.savez(
        npz_path,
        target_fps=target_fps,
        frame_count=num_ideal_frames,
        ideal_times=ideal_times,
        global_start=global_start,
        global_end=global_end,
        **{f"cam_{cid}_indices": frame_maps[cid] for cid in camera_ids},
    )
    print(f"  Saved: {npz_path}")

    # -- 10. Verify output videos ----------------------------------------------
    print(f"\n[10] Verifying output videos...")

    verification: Dict[int, Dict] = {}
    all_match = True

    for cam_id in camera_ids:
        cap = cv2.VideoCapture(str(output_paths[cam_id]))
        out_fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        out_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        ok = out_fc == num_ideal_frames
        verification[cam_id] = {
            "frame_count": out_fc,
            "fps": out_fps,
            "resolution": (out_w, out_h),
            "matches_ideal": ok,
        }

        sym = "✓" if ok else "✗"
        print(f"  Camera {cam_id}: {sym} {out_fc} frames "
              f"(expected {num_ideal_frames}), "
              f"{out_fps:.2f} fps, {out_w}x{out_h}")
        if not ok:
            all_match = False

    if all_match:
        print(f"\n  ✓ All videos have matching frame counts!")
    else:
        print(f"\n  ✗ WARNING: Some videos don't match expected frame count")

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"Synchronization Complete")
    print(f"{'=' * 70}")
    print(f"Synchronized to: Ideal {target_fps} FPS clock")
    print(f"Output frames: {num_ideal_frames}")
    print(f"Output FPS: {target_fps:.2f}")
    print(f"Output resolution: {width_out}x{height_out}")
    print(f"Time range: {global_start:.3f}s to {global_end:.3f}s")
    print(f"Duration: {global_end - global_start:.3f}s")
    print(f"Synchronized videos saved to: {out_dir}")
    print(f"{'=' * 70}\n")

    return {
        "trial_folder":        str(trial_folder),
        "sync_method":         "ideal_fps",
        "target_fps":          target_fps,
        "frame_count":         num_ideal_frames,
        "resolution":          (width_out, height_out),
        "camera_ids":          camera_ids,
        "ideal_times":         ideal_times,
        "global_start":        global_start,
        "global_end":          global_end,
        "duration":            global_end - global_start,
        "output_dir":          str(out_dir),
        "output_files":        {cid: str(output_paths[cid]) for cid in camera_ids},
        "frame_mappings":      frame_maps,
        "sync_metrics":        sync_metrics,
        "verification":        verification,
        "mapping_file":        str(npz_path),
        "timestamps_original": timestamps,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def _cam_colour_map(camera_ids: List[int]) -> Dict[int, np.ndarray]:
    """Return a consistent colour for each camera id using matplotlib Set1."""
    colours = plt.cm.Set1(np.linspace(0, 1, max(len(camera_ids), 9)))
    return {cid: colours[i] for i, cid in enumerate(camera_ids)}


def plot_sync_results(
    results: Dict,
    trial_folder: str,
    figsize: Tuple[int, int] = (16, 12),
    save_plots: bool = True,
    show_plots: bool = True,
) -> Dict:
    """Create a comprehensive 6-panel diagnostic figure.

    Panels
    ------
    1. **Original frame times vs ideal** - shows how each camera's raw
       timestamps drift from the perfect clock.
    2. **Synchronised frame times vs ideal** - same view *after* sync; all
       cameras should now hug the ideal line.
    3. **Original timing error** - per-frame signed error (ms) before sync.
    4. **Synchronised timing error** - per-frame signed error after sync.
    5. **Frame duration scatter** - inter-frame intervals over time; useful
       for spotting jitter and dropped frames.
    6. **Cumulative dropped frames** - running count of frames each camera
       has dropped relative to the expected cadence.

    Parameters
    ----------
    results : dict
        Output of :func:`synchronize_videos_to_ideal_fps`.
    trial_folder : str
        Path to the trial directory (used for titles and save paths).
    figsize : tuple
        Figure size in inches ``(width, height)``.
    save_plots : bool
        If True, save PNG and PDF to ``<trial_folder>/synchronization_plots/``.
    show_plots : bool
        If True, call ``plt.show()``.

    Returns
    -------
    figs : dict
        ``{'main': <Figure>}``
    """
    trial_folder = Path(trial_folder)

    timestamps_original = results["timestamps_original"]
    target_fps  = results["target_fps"]
    camera_ids  = results["camera_ids"]
    ideal_times = results["ideal_times"]

    # Build synchronised timestamp arrays from the frame mappings
    timestamps_synced = {
        cid: timestamps_original[cid][results["frame_mappings"][cid]]
        for cid in camera_ids
    }

    t0 = results["global_start"]
    cam_colors = _cam_colour_map(camera_ids)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    ideal_rel = ideal_times - t0

    # -- Panel 1: Original frame times vs ideal --------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    for cid in camera_ids:
        ax1.plot(timestamps_original[cid] - t0,
                 linewidth=2, alpha=0.8, color=cam_colors[cid],
                 label=f"Cam {cid} (original)")
    ax1.plot(ideal_rel, "k--", linewidth=1.5, alpha=0.5,
             label=f"Ideal {target_fps} fps")
    ax1.set_xlabel("Frame Number", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Time Since Start (s)", fontsize=11, fontweight="bold")
    ax1.set_title("Original Frame Times vs Ideal", fontsize=12, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # -- Panel 2: Synchronised frame times vs ideal ----------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for cid in camera_ids:
        ax2.plot(timestamps_synced[cid] - t0,
                 linewidth=2, alpha=0.8, color=cam_colors[cid],
                 label=f"Cam {cid} (synced)")
    ax2.plot(ideal_rel, "k--", linewidth=1.5, alpha=0.5,
             label=f"Ideal {target_fps} fps")
    ax2.set_xlabel("Frame Number", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Time Since Start (s)", fontsize=11, fontweight="bold")
    ax2.set_title("Synchronized Frame Times vs Ideal", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # -- Panel 3: Original timing error ----------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    for cid in camera_ids:
        t_orig = timestamps_original[cid]
        n = min(len(t_orig), len(ideal_times))
        err = (t_orig[:n] - ideal_times[:n]) * 1000
        ax3.plot(err, linewidth=2, alpha=0.8, color=cam_colors[cid],
                 label=f"Cam {cid} (mean: {np.mean(err):.2f}ms)")
    ax3.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Frame Number", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Timing Error (ms)", fontsize=11, fontweight="bold")
    ax3.set_title(f"Original: Error from Ideal {target_fps} FPS",
                  fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # -- Panel 4: Synchronised timing error ------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    for cid in camera_ids:
        err = (timestamps_synced[cid] - ideal_times) * 1000
        ax4.plot(err, linewidth=2, alpha=0.8, color=cam_colors[cid],
                 label=f"Cam {cid} (mean: {np.mean(err):.2f}ms)")
    ax4.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax4.set_xlabel("Frame Number", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Timing Error (ms)", fontsize=11, fontweight="bold")
    ax4.set_title(f"Synchronized: Error from Ideal {target_fps} FPS",
                  fontsize=12, fontweight="bold")
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    # -- Panel 5: Frame duration scatter ---------------------------------------
    ax5 = fig.add_subplot(gs[2, :])
    for cid in camera_ids:
        t_orig = timestamps_original[cid]
        if len(t_orig) > 1:
            dur_ms = np.diff(t_orig) * 1000
            ax5.scatter(np.arange(len(dur_ms)), dur_ms, s=3, alpha=0.6,
                        color=cam_colors[cid], label=f"Cam {cid}")
    ideal_dur = 1000.0 / target_fps
    ax5.axhline(ideal_dur, color="k", linestyle="--", linewidth=2, alpha=0.7,
                label=f"Ideal ({ideal_dur:.2f}ms @ {target_fps}fps)")
    ax5.set_xlabel("Frame Number", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Frame Duration (ms)", fontsize=11, fontweight="bold")
    ax5.set_title("Frame Duration Over Time (Original)", fontsize=12, fontweight="bold")
    ax5.legend(loc="best", fontsize=9, markerscale=3)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)

    # -- Panel 6: Cumulative dropped frames ------------------------------------
    ax6 = fig.add_subplot(gs[3, :])
    expected_dur = 1.0 / target_fps
    for cid in camera_ids:
        t_orig = timestamps_original[cid]
        if len(t_orig) > 1:
            durations = np.diff(t_orig)
            dropped = np.maximum(
                0, np.round(durations / expected_dur).astype(int) - 1
            )
            cum_dropped = np.cumsum(dropped)
            total = int(cum_dropped[-1]) if len(cum_dropped) else 0
            ax6.plot(np.arange(len(cum_dropped)), cum_dropped,
                     linewidth=2.5, color=cam_colors[cid], alpha=0.8,
                     label=f"Cam {cid} (total: {total})")
    ax6.set_xlabel("Original Frame Number", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Cumulative Dropped Frames", fontsize=11, fontweight="bold")
    ax6.set_title("Dropped Frames Over Time (All Cameras)",
                  fontsize=12, fontweight="bold")
    ax6.legend(loc="best", fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(bottom=0)

    fig.suptitle(f"Synchronization Analysis – {trial_folder.name}",
                 fontsize=14, fontweight="bold", y=0.995)

    # -- Save ------------------------------------------------------------------
    figs = {"main": fig}

    if save_plots:
        plot_dir = trial_folder / "synchronization_plots"
        plot_dir.mkdir(exist_ok=True)

        png_path = plot_dir / "sync_analysis.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Plots saved to: {png_path}")

        pdf_path = plot_dir / "sync_analysis.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"📊 PDF saved to: {pdf_path}")

    if show_plots:
        plt.show()

    return figs


def plot_sync_summary_stats(
    results: Dict,
    trial_folder: str,
    save_plots: bool = True,
) -> plt.Figure:
    """Create a 2-panel summary figure with bar chart and box plot of timing errors.

    Panel 1 - **Bar chart**: mean / RMS / max timing error for each camera.
    Panel 2 - **Box plot**: full distribution of timing errors per camera.

    Parameters
    ----------
    results : dict
        Output of :func:`synchronize_videos_to_ideal_fps`.
    trial_folder : str
        Path to the trial directory.
    save_plots : bool
        If True, save PNG to ``<trial_folder>/synchronization_plots/``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    trial_folder = Path(trial_folder)

    camera_ids   = results["camera_ids"]
    sync_metrics = results["sync_metrics"]
    target_fps   = results["target_fps"]
    cam_colors   = _cam_colour_map(camera_ids)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Bar chart -------------------------------------------------------------
    labels, means, rmss, maxes, colors = [], [], [], [], []
    for cid in camera_ids:
        m = sync_metrics[cid]
        labels.append(f"Cam {cid}")
        means.append(m["mean_diff_ms"])
        rmss.append(m["rms_diff_ms"])
        maxes.append(m["max_diff_ms"])
        colors.append(cam_colors[cid])

    x = np.arange(len(labels))
    w = 0.25
    ax1.bar(x - w, means, w, label="Mean", color=colors, alpha=0.8)
    ax1.bar(x,     rmss,  w, label="RMS",  color=colors, alpha=0.6)
    ax1.bar(x + w, maxes, w, label="Max",  color=colors, alpha=0.4)
    ax1.set_xlabel("Camera", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Timing Error (ms)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Timing Errors vs Ideal {target_fps} FPS",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # -- Box plot --------------------------------------------------------------
    all_diffs = [sync_metrics[cid]["time_diffs_ms"] for cid in camera_ids]
    bp = ax2.boxplot(all_diffs, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax2.set_xlabel("Camera", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Timing Error (ms)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Timing Error Distribution vs Ideal {target_fps} FPS",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(0, color="r", linestyle="--", linewidth=1, alpha=0.5)

    fig.suptitle(f"Synchronization Summary – {trial_folder.name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_plots:
        plot_dir = trial_folder / "synchronization_plots"
        plot_dir.mkdir(exist_ok=True)
        path = plot_dir / "sync_summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"📊 Summary plots saved to: {path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def vid_sync(
    trial_folder: str,
    target_fps: float = 30.0,
    raw_videos_subdir: str = "raw_videos",
    out_subdir: str = "synchronized_videos",
    max_time_diff_ms: float = 100.0,
    progress_callback=None,
    figsize: Tuple[int, int] = (16, 12),
    save_plots: bool = True,
    show_plots: bool = True,
) -> Tuple[Dict, Dict]:
    """One-call entry point: synchronise videos **and** generate all plots.

    This simply calls :func:`synchronize_videos_to_ideal_fps` followed by
    :func:`plot_sync_results` and :func:`plot_sync_summary_stats`, returning
    everything in a neat bundle.

    Parameters
    ----------
    trial_folder : str
        Path to the trial directory.
    target_fps : float
        Desired output FPS (default 30).
    raw_videos_subdir, out_subdir, max_time_diff_ms, progress_callback
        Forwarded to :func:`synchronize_videos_to_ideal_fps`.
    figsize, save_plots, show_plots
        Forwarded to :func:`plot_sync_results`.

    Returns
    -------
    results : dict
        Synchronisation results (metrics, paths, mappings, etc.).
    figs : dict
        Matplotlib figures - ``'main'`` and ``'summary'``.

    Example
    -------
    >>> from camkit3d.sync import vid_sync
    >>> results, figs = vid_sync("./recordings/2026-02-04_12-00-00",
    ...                           target_fps=30.0)
    """
    results = synchronize_videos_to_ideal_fps(
        trial_folder=trial_folder,
        target_fps=target_fps,
        raw_videos_subdir=raw_videos_subdir,
        out_subdir=out_subdir,
        max_time_diff_ms=max_time_diff_ms,
        progress_callback=progress_callback,
    )

    figs = plot_sync_results(
        results=results,
        trial_folder=trial_folder,
        figsize=figsize,
        save_plots=save_plots,
        show_plots=show_plots,
    )

    figs["summary"] = plot_sync_summary_stats(
        results=results,
        trial_folder=trial_folder,
        save_plots=save_plots,
    )

    return results, figs


# ═══════════════════════════════════════════════════════════════════════════════
#  TILED REVIEW VIDEO
# ═══════════════════════════════════════════════════════════════════════════════

def save_synced_videos_tiled(
    trial_folder: str,
    synced_subdir: str = "synchronized_videos",
    out_name: str = "synchronized_tiled.mp4",
    max_cols: int = 3,
    max_width: int = 1920,
    pad_px: int = 6,
    pad_value: int = 0,
    font_scale: float = 0.9,
    thickness: int = 2,
) -> Path:
    """Tile all synchronised camera feeds into a single side-by-side review video.

    After synchronisation every camera has the same number of frames at the
    same FPS.  This function composites them into a grid layout (up to
    *max_cols* per row, wrapping to new rows as needed) so you can visually
    confirm that all views are in sync.

    Each cell is letterboxed to the largest camera resolution and labelled
    with the camera ID and frame number.  The final grid is optionally
    downscaled so its width does not exceed *max_width*.

    Codec selection tries H.264 (``avc1`` → ``H264``) and falls back to
    ``mp4v`` if neither is available.

    Parameters
    ----------
    trial_folder : str
        Path to the trial directory.
    synced_subdir : str
        Subdirectory containing the synchronised MP4 files.
    out_name : str
        Filename for the tiled output video.
    max_cols : int
        Maximum number of cameras per row in the grid.
    max_width : int
        If the grid is wider than this (in pixels) it will be proportionally
        downscaled.
    pad_px : int
        Padding in pixels between grid cells.
    pad_value : int
        Pixel intensity for padding / letterbox bars (0 = black).
    font_scale, thickness : float, int
        OpenCV ``putText`` parameters for the overlay label.

    Returns
    -------
    out_path : pathlib.Path
        Path to the written tiled video.
    """
    trial_folder = Path(trial_folder)
    synced_dir = trial_folder / synced_subdir
    out_path = trial_folder / out_name

    videos = sorted(synced_dir.glob("camera_*_synchronized.mp4"))
    if not videos:
        raise FileNotFoundError(f"No synchronized videos found in {synced_dir}")

    caps = []
    cam_ids = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {v}")
        caps.append(cap)
        cam_ids.append(int(v.stem.split("_")[1]))

    # -- Sanity checks ---------------------------------------------------------
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    if len(set(frame_counts)) != 1:
        raise ValueError(
            f"Frame count mismatch: {dict(zip(cam_ids, frame_counts))}"
        )

    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = frame_counts[0]

    # -- Grid geometry ---------------------------------------------------------
    cell_w = max(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps)
    cell_h = max(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps)

    n = len(caps)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    grid_w = cols * cell_w + (cols - 1) * pad_px
    grid_h = rows * cell_h + (rows - 1) * pad_px

    scale = 1.0
    if max_width and grid_w > max_width:
        scale = max_width / grid_w

    out_w = int(round(grid_w * scale))
    out_h = int(round(grid_h * scale))

    # -- Codec selection -------------------------------------------------------
    def _make_writer(codec: str) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    writer = _make_writer("avc1")
    if not writer.isOpened():
        writer = _make_writer("H264")
    if not writer.isOpened():
        writer = _make_writer("mp4v")
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    print("Writing tiled video:")
    print(f"  cameras: {cam_ids}")
    print(f"  videos:  {n} -> grid {rows}x{cols}")
    print(f"  frames:  {n_frames}")
    print(f"  fps:     {fps:.2f}")
    print(f"  cell:    {cell_w}x{cell_h} px")
    print(f"  output:  {out_path}")

    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -- Write frames ----------------------------------------------------------
    for i in range(n_frames):
        canvas = np.full((grid_h, grid_w, 3), pad_value, dtype=np.uint8)

        for idx, (cam_id, cap) in enumerate(zip(cam_ids, caps)):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed reading frame {i} from camera {cam_id}"
                )

            # Letterbox into cell
            fh, fw = frame.shape[:2]
            s = min(cell_w / fw, cell_h / fh)
            new_w, new_h = int(round(fw * s)), int(round(fh * s))
            resized = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

            cell = np.full((cell_h, cell_w, 3), pad_value, dtype=np.uint8)
            y0 = (cell_h - new_h) // 2
            x0 = (cell_w - new_w) // 2
            cell[y0 : y0 + new_h, x0 : x0 + new_w] = resized

            cv2.putText(
                cell, f"Cam {cam_id} | frame {i}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                thickness, cv2.LINE_AA,
            )

            r, c = idx // cols, idx % cols
            top  = r * (cell_h + pad_px)
            left = c * (cell_w + pad_px)
            canvas[top : top + cell_h, left : left + cell_w] = cell

        if scale != 1.0:
            canvas = cv2.resize(
                canvas, (out_w, out_h), interpolation=cv2.INTER_AREA
            )

        writer.write(canvas)

    # -- Cleanup ---------------------------------------------------------------
    writer.release()
    for cap in caps:
        cap.release()

    print("Done ✅")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Change this to your trial folder
    trial_dir = "./recordings/2026-02-04_12-00-00"

    results, figs = vid_sync(
        trial_folder=trial_dir,
        target_fps=30.0,
        max_time_diff_ms=50.0,
    )

    # Print per-camera metrics
    print("\nSynchronization metrics:")
    for cam_id, metrics in results["sync_metrics"].items():
        print(f"\nCamera {cam_id} vs Ideal {results['target_fps']} FPS:")
        print(f"  Mean time difference: {metrics['mean_diff_ms']:.2f} ms")
        print(f"  RMS time difference:  {metrics['rms_diff_ms']:.2f} ms")
        print(f"  Max time difference:  {metrics['max_diff_ms']:.2f} ms")

    # Create tiled review video
    tiled_path = save_synced_videos_tiled(trial_dir)
    print(f"\nTiled video: {tiled_path}")