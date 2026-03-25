"""
Batch Video Synchronization Runner
===================================
Discovers all subfolders under a root directory that contain the expected
recording structure, then processes each one in parallel (ThreadPoolExecutor).
A rich Live table refreshes every ~100ms showing per-folder progress: frame
count, elapsed time, and current phase.

Expected subfolder structure
----------------------------
    <subfolder>/
        camera_0_timestamps.npy
        camera_1_timestamps.npy
        ...
        raw_videos/
            camera_0.avi   (or .mp4)
            camera_1.avi   (or .mp4)
            ...

Any subfolder that does not match this layout is reported as skipped.

Usage
-----
    python run_sync_batch.py /path/to/recordings
    python run_sync_batch.py /path/to/recordings --fps 60 --workers 4 --max-diff 50
    python run_sync_batch.py /path/to/recordings --no-plots

Arguments
---------
    root_dir        Root directory containing recording subfolders
    --fps           Target FPS for synchronization (default: 30.0)
    --max-diff      Max allowed time difference in ms (default: 50.0)
    --workers       Number of parallel folders to process (default: cpu_count // 2)
    --no-plots      Skip plot generation
    --show-plots    Show plots interactively (default: False when batching)
"""

import argparse
import sys
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, Thread

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[WARNING] 'rich' not installed – pip install rich for nice progress bars.\n")

try:
    from camkit3d.sync import synchronize_videos_to_ideal_fps, plot_sync_results, plot_sync_summary_stats
except ImportError as e:
    sys.exit(
        f"[ERROR] Could not import camkit3d.sync: {e}\n"
        "Make sure camkit3d is installed (pip install -e .)."
    )

import matplotlib
matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════
# Folder discovery & validation
# ═══════════════════════════════════════════════════════════════════════════

def _is_macos_tempfile(path: Path) -> bool:
    """Return True for macOS resource-fork / temp files (e.g. '._camera_0')."""
    return path.name.startswith("._")


def validate_folder(folder: Path) -> tuple[bool, str]:
    """
    Check whether a subfolder has the expected recording layout.

    Returns (True, "") if valid, or (False, reason) if not.
    """
    if not folder.is_dir():
        return False, "not a directory"

    raw_videos = folder / "raw_videos"
    if not raw_videos.is_dir():
        return False, "missing 'raw_videos/' subfolder"

    # Look for video files (.avi or .mp4), skipping macOS temp files
    video_files = sorted(
        v for v in (list(raw_videos.glob("*.avi")) + list(raw_videos.glob("*.mp4")))
        if not _is_macos_tempfile(v)
    )
    if not video_files:
        return False, "no .avi or .mp4 files in raw_videos/"

    # Look for matching camera_X_timestamps.npy files, skipping macOS temp files
    timestamp_files = sorted(
        t for t in folder.glob("camera_*_timestamps.npy")
        if not _is_macos_tempfile(t)
    )
    if not timestamp_files:
        return False, "no camera_*_timestamps.npy files found"

    # Check that each video has a corresponding timestamp file (and vice versa)
    video_stems = set()
    for v in video_files:
        # e.g. camera_0.avi -> camera_0
        video_stems.add(v.stem)

    ts_stems = set()
    for t in timestamp_files:
        # e.g. camera_0_timestamps.npy -> camera_0
        stem = t.stem  # camera_0_timestamps
        stem = stem.replace("_timestamps", "")
        ts_stems.add(stem)

    videos_without_ts = video_stems - ts_stems
    ts_without_videos = ts_stems - video_stems

    warnings = []
    if videos_without_ts:
        warnings.append(f"videos without timestamps: {sorted(videos_without_ts)}")
    if ts_without_videos:
        warnings.append(f"timestamps without videos: {sorted(ts_without_videos)}")

    if warnings:
        return False, "; ".join(warnings)

    return True, ""


def discover_folders(root: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """
    Scan all immediate subfolders of *root*.

    Returns
    -------
    valid : list[Path]
        Folders that match the expected structure.
    skipped : list[tuple[Path, str]]
        Folders that were skipped, with a reason string.
    """
    valid = []
    skipped = []

    candidates = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not _is_macos_tempfile(p)
    )
    for folder in candidates:
        ok, reason = validate_folder(folder)
        if ok:
            valid.append(folder)
        else:
            skipped.append((folder, reason))

    return valid, skipped


# ═══════════════════════════════════════════════════════════════════════════
# Per-folder status (written by worker threads, read by display thread)
# ═══════════════════════════════════════════════════════════════════════════

class FolderStatus:
    def __init__(self, name: str):
        self.name = name
        self.state = "waiting"      # waiting | running | plotting | done | error
        self.phase = "queued"       # human-readable current step
        self.frames_done = 0
        self.frames_total = 0
        self.start_time: float | None = None
        self.end_time: float | None = None
        self._lock = Lock()

    # ── called from worker thread ──────────────────────────────────────────
    def mark_running(self):
        with self._lock:
            self.state = "running"
            self.phase = "loading timestamps…"
            self.start_time = time.monotonic()

    def set_phase(self, phase: str):
        with self._lock:
            self.phase = phase

    def set_total_frames(self, n: int):
        with self._lock:
            self.frames_total = n
            self.state = "running"
            self.phase = "writing frames"

    def update_frames(self, done: int):
        with self._lock:
            self.frames_done = done

    def mark_plotting(self):
        with self._lock:
            self.state = "plotting"
            self.phase = "generating plots…"

    def mark_done(self, msg: str = ""):
        with self._lock:
            self.state = "done"
            self.phase = msg or "done"
            self.end_time = time.monotonic()

    def mark_error(self, msg: str):
        with self._lock:
            self.state = "error"
            self.phase = msg[:100]
            self.end_time = time.monotonic()

    # ── called from display thread (read-only snapshot) ───────────────────
    def snapshot(self):
        with self._lock:
            if self.start_time is None:
                elapsed = 0.0
            elif self.end_time is not None:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.monotonic() - self.start_time
            return (
                self.name, self.state, self.phase,
                self.frames_done, self.frames_total, elapsed,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════

def process_folder(
    folder: Path,
    status: FolderStatus,
    target_fps: float,
    max_time_diff_ms: float,
    make_plots: bool,
    show_plots: bool,
) -> dict:

    status.mark_running()

    try:
        # Callback fired every frame by the patched sync function
        def on_frame(done: int, total: int):
            status.set_total_frames(total)   # idempotent after first call
            status.update_frames(done)

        results = synchronize_videos_to_ideal_fps(
            trial_folder=str(folder),
            target_fps=target_fps,
            max_time_diff_ms=max_time_diff_ms,
            progress_callback=on_frame,
        )

        if make_plots:
            status.mark_plotting()
            plot_sync_results(
                results=results,
                trial_folder=str(folder),
                save_plots=True,
                show_plots=show_plots,
            )
            plot_sync_summary_stats(
                results=results,
                trial_folder=str(folder),
                save_plots=True,
            )

        n = results.get("frame_count", "?")
        dur = results.get("duration", 0)
        status.mark_done(f"{n} frames  {dur:.1f}s")
        return {"folder": folder, "ok": True, "results": results}

    except Exception as exc:
        status.mark_error(str(exc))
        return {"folder": folder, "ok": False,
                "error": str(exc), "traceback": traceback.format_exc()}


# ═══════════════════════════════════════════════════════════════════════════
# Rich table builder  (called ~10× per second from Live thread)
# ═══════════════════════════════════════════════════════════════════════════

STATE_ICON = {
    "waiting":  "[dim]⏳[/dim]",
    "running":  "[bold cyan]⚙[/bold cyan]",
    "plotting": "[bold yellow]📊[/bold yellow]",
    "done":     "[bold green]✓[/bold green]",
    "error":    "[bold red]✗[/bold red]",
}
STATE_STYLE = {
    "waiting":  "dim",
    "running":  "cyan",
    "plotting": "yellow",
    "done":     "green",
    "error":    "red",
}


def _bar(done: int, total: int, width: int = 20) -> str:
    """Simple ASCII progress bar string."""
    if total <= 0:
        return " " * width
    frac = min(done / total, 1.0)
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def build_table(statuses: list[FolderStatus], n_workers: int) -> Table:
    """Build a Rich table that fits in the terminal.

    When there are many folders (>20) we collapse finished rows to keep the
    table within a reasonable height.  Active, waiting, and error rows are
    always shown.  Completed rows are trimmed to fill remaining space, with
    a summary line for any hidden ones.
    """
    import shutil
    term_lines = shutil.get_terminal_size((80, 40)).lines
    # Reserve lines for: title(2) + header(2) + caption(2) + border(2) + summary line(1) + some breathing room(3)
    max_rows = max(10, term_lines - 12)

    # Partition rows by priority
    active  = []  # running / plotting — always shown
    waiting = []
    errors  = []
    done    = []

    for s in statuses:
        snap = s.snapshot()
        state = snap[1]
        if state in ("running", "plotting"):
            active.append(s)
        elif state == "waiting":
            waiting.append(s)
        elif state == "error":
            errors.append(s)
        else:
            done.append(s)

    # Decide which rows to display
    must_show = active + errors          # always visible
    budget    = max_rows - len(must_show)

    # Show waiting next, then done (most recent first)
    show_waiting = waiting[:budget]
    budget -= len(show_waiting)

    show_done = done[-budget:] if budget > 0 else []
    hidden_done = len(done) - len(show_done)
    hidden_waiting = len(waiting) - len(show_waiting)

    display_order = active + show_waiting + show_done + errors

    tbl = Table(
        box=box.SIMPLE_HEAVY,
        expand=True,
        title=f"[bold]Batch Sync  —  {len(statuses)} folders  ({n_workers} workers)[/bold]",
        show_lines=False,
    )
    tbl.add_column("",        width=2,  no_wrap=True)          # icon
    tbl.add_column("Folder",  ratio=3,  no_wrap=True)
    tbl.add_column("Frames",  ratio=3,  no_wrap=True)          # bar + N/X
    tbl.add_column("Phase",   ratio=3,  no_wrap=True)
    tbl.add_column("Elapsed", width=8,  justify="right", no_wrap=True)

    total_done = total_frames = 0

    # Insert a summary row for hidden completed folders
    if hidden_done > 0 or hidden_waiting > 0:
        parts = []
        if hidden_done > 0:
            parts.append(f"[green]{hidden_done} done[/green]")
        if hidden_waiting > 0:
            parts.append(f"[dim]{hidden_waiting} queued[/dim]")
        tbl.add_row(
            "[dim]…[/dim]",
            f"[dim]({', '.join(parts)} — not shown)[/dim]",
            "", "", "",
        )

    # Accumulate totals from ALL statuses (including hidden)
    for s in statuses:
        _, state, _, fdone, ftotal, _ = s.snapshot()
        if state in ("running",) and ftotal > 0:
            total_done   += fdone
            total_frames += ftotal
        elif state in ("done", "plotting"):
            total_done   += ftotal
            total_frames += ftotal

    for s in display_order:
        name, state, phase, fdone, ftotal, elapsed = s.snapshot()
        style = STATE_STYLE[state]
        icon  = STATE_ICON[state]

        # Frame progress column
        if state in ("running",) and ftotal > 0:
            bar = _bar(fdone, ftotal)
            frame_str = f"{bar} [dim]{fdone}/{ftotal}[/dim]"
        elif state == "done":
            frame_str = f"[green]{phase}[/green]"
        elif state == "plotting":
            frame_str = f"[yellow]{'█' * 20}[/yellow]"
        elif state == "error":
            frame_str = "[red]—[/red]"
        else:
            frame_str = "[dim]—[/dim]"

        # Phase column (skip when already shown in frame col)
        if state in ("done", "error"):
            phase_str = ""
        else:
            phase_str = f"[{style}]{phase}[/{style}]"

        elapsed_str = f"[dim]{elapsed:6.1f}s[/dim]" if elapsed else "[dim]      [/dim]"

        tbl.add_row(
            icon,
            f"[{style}]{name}[/{style}]",
            frame_str,
            phase_str,
            elapsed_str,
        )

    running = sum(1 for s in statuses if s.state == "running")
    done_n  = sum(1 for s in statuses if s.state == "done")
    errors_n = sum(1 for s in statuses if s.state == "error")
    pct     = f"  ({100*total_done//total_frames}%)" if total_frames else ""

    tbl.caption = Text.from_markup(
        f"[dim]Active:[/dim] [cyan]{running}[/cyan]  "
        f"[dim]Done:[/dim] [green]{done_n}/{len(statuses)}[/green]  "
        f"[dim]Errors:[/dim] [red]{errors_n}[/red]  "
        f"[dim]Total frames:[/dim] {total_done}/{total_frames}{pct}"
    )
    return tbl


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(
    root_dir: Path,
    target_fps: float = 30.0,
    max_time_diff_ms: float = 50.0,
    n_workers: int = 2,
    make_plots: bool = True,
    show_plots: bool = False,
):
    folders, skipped = discover_folders(root_dir)

    # ── Report skipped folders ──────────────────────────────────────────────
    if skipped:
        print(f"\n⚠  {len(skipped)} subfolder(s) skipped (unexpected layout):\n")
        for folder, reason in skipped:
            print(f"  ✗  {folder.name}/")
            print(f"     └─ {reason}")
        print()

    if not folders:
        sys.exit(
            f"[ERROR] No valid recording folders found under {root_dir}\n"
            f"        Expected: <subfolder>/camera_*_timestamps.npy + raw_videos/*.avi|mp4"
        )

    print(f"✓  Found {len(folders)} valid folder(s) under: {root_dir}\n")
    for f in folders:
        n_ts = len([t for t in f.glob("camera_*_timestamps.npy") if not _is_macos_tempfile(t)])
        raw = f / "raw_videos"
        n_vid = len([v for v in (list(raw.glob("*.avi")) + list(raw.glob("*.mp4"))) if not _is_macos_tempfile(v)])
        print(f"     {f.name}/  ({n_ts} cameras, {n_vid} videos)")
    print()

    statuses = [FolderStatus(f.name) for f in folders]
    status_map = {f: s for f, s in zip(folders, statuses)}

    all_results: list[dict] = []
    executor = ThreadPoolExecutor(max_workers=n_workers)

    futures = {
        executor.submit(
            process_folder,
            folder, status_map[folder],
            target_fps, max_time_diff_ms,
            make_plots, show_plots,
        ): folder
        for folder in folders
    }

    if HAS_RICH:
        console = Console()
        # Use a background thread that collects futures; Live runs on main thread
        results_lock = Lock()

        def collector():
            for fut in as_completed(futures):
                with results_lock:
                    all_results.append(fut.result())

        t = Thread(target=collector, daemon=True)
        t.start()

        with Live(
            build_table(statuses, n_workers),
            console=console,
            refresh_per_second=10,   # 10 Hz → elapsed ticks visibly every 100 ms
            transient=False,
        ) as live:
            while t.is_alive() or len(all_results) < len(folders):
                live.update(build_table(statuses, n_workers))
                time.sleep(0.1)
            # One final render after everything finishes
            live.update(build_table(statuses, n_workers))

        t.join()
        console.print(build_table(statuses, n_workers))

    else:
        # Plain fallback
        for fut in as_completed(futures):
            r = fut.result()
            all_results.append(r)
            s = status_map[futures[fut]]
            _, state, phase, fdone, ftotal, elapsed = s.snapshot()
            print(f"[{state.upper():8s}] {s.name}  {fdone}/{ftotal} frames  {elapsed:.1f}s  {phase}")

    executor.shutdown(wait=True)

    # ── summary ─────────────────────────────────────────────────────────────
    ok  = sum(1 for r in all_results if r["ok"])
    err = len(all_results) - ok

    print("\n" + "═" * 60)
    print(f"  Batch complete:  {ok} succeeded,  {err} failed")
    if skipped:
        print(f"  Skipped:         {len(skipped)} folder(s) with bad layout")
    print("═" * 60)

    if err:
        print("\nFailed folders:")
        for r in all_results:
            if not r["ok"]:
                print(f"  • {r['folder'].name}:  {r['error']}")
        print()

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    import multiprocessing
    default_workers = max(1, multiprocessing.cpu_count() // 2)
    p = argparse.ArgumentParser(
        description="Batch-synchronise recording folders in parallel.\n\n"
                    "Each subfolder of root_dir should contain:\n"
                    "  - camera_X_timestamps.npy files\n"
                    "  - a raw_videos/ subfolder with camera_X.avi or .mp4 files\n\n"
                    "Subfolders that don't match this layout are reported and skipped.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root_dir",     type=Path,  help="Root directory containing recording subfolders")
    p.add_argument("--fps",        type=float, default=30.0,           help="Target FPS (default: 30.0)")
    p.add_argument("--max-diff",   type=float, default=50.0,           help="Max time diff in ms (default: 50.0)")
    p.add_argument("--workers",    type=int,   default=default_workers, help=f"Parallel workers (default: {default_workers})")
    p.add_argument("--no-plots",   action="store_true",                help="Skip plot generation")
    p.add_argument("--show-plots", action="store_true",                help="Show plots interactively")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.show_plots:
        matplotlib.use("TkAgg")
    run_batch(
        root_dir=args.root_dir,
        target_fps=args.fps,
        max_time_diff_ms=args.max_diff,
        n_workers=args.workers,
        make_plots=not args.no_plots,
        show_plots=args.show_plots,
    )
