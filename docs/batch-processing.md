# Batch processing

When processing many trials, CamKit3D has two batch scripts in
[`scripts/`](https://github.com/neurofractal/camkit3D/tree/main/scripts) that handle the
synchronisation and 2D pose stages across a whole directory of recordings. Both
discover valid subfolders, process them in parallel, and render a
live progress table (via `rich`).

## 1. Batch synchronisation — `run_sync_batch.py`

Synchronises every recording folder under a root directory to a target FPS.

```bash
python scripts/run_sync_batch.py /path/to/recordings
python scripts/run_sync_batch.py /path/to/recordings --fps 30 --workers 4
```

It scans each subfolder for `camera_*_timestamps.npy` files plus a
`raw_videos/` folder of `.avi`/`.mp4` videos, skips any that don't match (with a
reason), then writes synchronised videos for each valid trial. The live table
shows per-folder frame progress, current phase, and elapsed time; a final
summary reports how many folders succeeded, failed, or were skipped.

| Option | Default | Description |
|--------|---------|-------------|
| `--fps` | `30.0` | Target frame rate |
| `--max-diff` | `50.0` | Max allowed inter-camera time difference (ms) |
| `--workers` | `cpu_count // 2` | Folders processed in parallel |
| `--no-plots` | off | Skip QC plot generation |

## 2. Batch 2D pose — `run_pose_batch.py`

Runs MediaPipe pose estimation + Butterworth smoothing on every folder
containing a `synchronized_videos/` subfolder.

```bash
python scripts/run_pose_batch.py /path/to/recordings
python scripts/run_pose_batch.py /path/to/recordings --model-complexity 0 --no-labeled-video
```

For each trial it runs pose on all cameras (parallelised internally), saves
labelled overlay videos and 2D keypoint files to `mediapipe_output/`, and
applies a low-pass Butterworth filter to smooth the trajectories. A consolidated
`batch_pose_summary.csv` is written to the root directory with per-camera
detection rates, confidence stats, and any warnings.

| Option | Default | Description |
|--------|---------|-------------|
| `--model-complexity` | `2` | `0` fast / `1` balanced / `2` accurate |
| `--no-labeled-video` | off | Skip overlay videos (faster) |
| `--cutoff-freq` | `4.0` | Butterworth low-pass cutoff (Hz) |
| `--sampling-freq` | `30.0` | Video frame rate (Hz) |
| `--no-smooth` | off | Skip smoothing |
| `--pattern` | `*` | Only process folders matching this glob |
| `--workers` | `1` | Folders in parallel (cameras already parallelise within each) |

!!! tip "Why batching is faster"
    Both scripts parallelise work — `run_sync_batch.py` across folders, 
    `run_pose_batch.py` across cameras within
    each folder — and suppress per-trial logging in favour of one live table, so
    you can launch a full session's recordings and leave them to run.

## 3. Batch 3D pose

!!! warning "Not yet implemented"
    A batch runner for the triangulation stage (stage 4) does not exist yet.