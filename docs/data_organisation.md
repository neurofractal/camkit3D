# Data organisation

## The trial folder

Everything for one recording lives under a single trial directory:

```
recordings/                          <- base_output_dir (default "./recordings")
└── 2026-02-06_14-54-43/             <- one trial  (YYYY-MM-DD_HH-MM-SS by default)
```

The trial name is auto-generated from the start time, or passed explicitly to
`start_recording(trial_name=...)`. Subsequent stages write their outputs
back inside this folder.

One input is **not** produced by the pipeline: the camera calibration file
(`camera_calibration.toml`), which comes from your calibration step and is
supplied to the 3D triangulation stage. I would advise putting this at the
top of your recordings directory.

---

## Stage 1 — Recording (`recorder.py`)

`MultiCamRecorder` writes one raw MJPEG video and one timestamp array per
camera, plus a human-readable metadata file.

```
recordings/2026-02-06_14-54-43/
├── raw_videos/
│   ├── camera_0.avi                 <- MJPEG, native capture (per camera)
│   ├── camera_1.avi
│   └── camera_2.avi
├── camera_0_timestamps.npy          <- POSIX capture time per frame, shape (n_frames,)
├── camera_1_timestamps.npy
├── camera_2_timestamps.npy
└── metadata.txt                     <- resolution, fps, codec, per-camera frame counts
```

---

## Stage 2 — Synchronisation (`sync.py`)

`vid_sync()` resamples every camera onto one ideal timing grid at a target FPS.
It reads `raw_videos/` and the `camera_*_timestamps.npy` arrays, and writes:

```
recordings/2026-02-06_14-54-43/
├── synchronized_videos/
│   ├── camera_0_synchronized.mp4    <- all cameras now share a frame clock
│   ├── camera_1_synchronized.mp4
│   └── camera_2_synchronized.mp4
├── frame_mappings_to_ideal_fps.npz  <- ideal_times + per-camera source-frame indices
├── synchronization_plots/
│   ├── sync_analysis.png            <- 6-panel timing diagnostics
│   ├── sync_analysis.pdf
│   └── sync_summary.png             <- per-camera error bar/box plots
└── synchronized_tiled.mp4           <- optional tiled grid for visual QC
```

After this stage every synchronised video has an **identical frame count**, so
frame *n* means the same time point in all cameras. `frame_mappings_to_ideal_fps.npz`
records exactly which raw frame was chosen for each output frame, so the mapping
is fully reproducible.

---

## Stage 3 — 2D pose (`pose2d.py`)

`PoseProcessor` runs MediaPipe Pose on each synchronised video. Its input is the
`synchronized_videos/` folder; its output directory is customisable (default is
`mediapipe_output/`). Camera names carry the video stem, e.g.
`camera_0_synchronized`.

```
recordings/2026-02-06_14-54-43/
└── mediapipe_output/                        <- output_dir (your choice)
    ├── data_2d/
    │   ├── camera_0_synchronized_keypoints.npy     <- (n_frames, 33, 3) = x, y, confidence (pixels)
    │   ├── camera_0_synchronized_keypoints.npz     <- same data + landmark names, compressed
    │   ├── camera_0_synchronized_metadata.json     <- shape, landmark names, coord system
    │   └── ...                                      (one set per camera)
    ├── labeled_videos/
    │   └── camera_0_synchronized_labeled.mp4       <- skeleton overlay for QC
    └── pose_estimation_summary.txt                 <- detection rate + confidence per camera
```

The `.npy` keypoint arrays in `data_2d/` are the files the 3D stage consumes.

**Optional cleaning** (smoothing, lower-body removal, face-edge cleaning) runs
over `data_2d/`. By default it **overwrites the keypoint files in place**; pass
an `output_dir` to write a cleaned copy instead:

```
    ├── data_2d/                 <- raw 2D keypoints
    ├── data_2d_upper/           <- e.g. lower body removed (optional copy)
    └── data_2d_cleaned/         <- e.g. face-edge cleaned (optional copy)
        └── *_keypoints_face_edge_mask.npy   <- mask saved alongside, if requested
```

---

## Stage 4 — 3D triangulation (`pose3d.py`)

`Pose3DProjector` triangulates the 2D keypoints into 3D using the camera
calibration. It reads **two** inputs — the `data_2d/` keypoints and your
`camera_calibration.toml` — and writes the 3D points plus a metrics report.

```
recordings/2026-02-06_14-54-43/
└── pose_3d/                        <- output_dir (your choice)
    ├── pose_3d.npy                 <- (n_frames, 33, 3) = x, y, z in millimetres
    ├── pose_3d.json                <- shape, camera names, skeleton id, units
    └── projection_metrics.txt      <- reprojection error summary
```

`camera_calibration.toml` holds per-camera intrinsics and extrinsics in `cam_*`
sections with units in mm.

**Camera naming is important.** Each `cam_*` section carries a `name`
field, and that name must exactly match the camera name used for the 2D
keypoints — i.e. the synchronised video stem, which is `camera_<id>_synchronized`
by default (e.g. `camera_1_synchronized`).

---

## Stage 5 — Analysis & visualisation (`analysis.py`, `viewer.py`)

This stage reads `pose_3d.npy` and produces cleaned/aligned arrays, static
plots, animations, and interactive viewers. Gap-filling (`interpolate_nans`) and
anatomical alignment (`align_pose_to_standard_frame`) happen **in memory** — you
save the result yourself if you want to keep it.

```
recordings/2026-02-06_14-54-43/
├── pose_3d_filled.npy              <- gap-filled / aligned array (whatever you save)
├── animations/
│   └── pose_front.mp4              <- rendered skeleton video (FFmpeg)
├── pose_viewer.html                <- standalone interactive 3D viewer
└── reprojection_errors.png         <- quality plots
```

Filenames here are conventions, not fixed by the code — you choose the
`output_path` for each plot, animation, and viewer.

---

## Naming conventions

| Placeholder | Meaning | Example |
|---|---|---|
| `<trial>` | one recording, `YYYY-MM-DD_HH-MM-SS` by default | `2026-02-06_14-54-43` |
| `<id>` | integer camera id used at recording | `0`, `1`, `2` |
| `<cam>` | camera name downstream = synchronised video stem | `camera_0_synchronized` |

Camera identity threads through the whole pipeline: the recording `<id>` becomes
the `<cam>` stem after synchronisation, and that stem must match a `cam_*`
`name` in `camera_calibration.toml` for triangulation to pair the views
correctly.

## File formats at a glance

| Format | Holds | Where |
|---|---|---|
| `.avi` (MJPEG) | raw per-camera video | `raw_videos/` |
| `.mp4` | synchronised, labelled, tiled, animation video | stages 2, 3, 5 |
| `.npy` | timestamps (1D), 2D keypoints (n×33×3), 3D points (n×33×3) | stages 1, 3, 4 |
| `.npz` | frame mappings; compressed 2D keypoints + names | stages 2, 3 |
| `.json` | per-camera / 3D metadata (shapes, names, units) | stages 3, 4 |
| `.toml` | camera intrinsics + extrinsics (external input) | stage 4 input |
| `.txt` | human-readable metadata / summaries / metrics | stages 1, 3, 4 |
| `.png` / `.pdf` | diagnostic and analysis plots | stages 2, 5 |
| `.html` | standalone interactive viewers | stage 5 |

## Full trial folder

```
recordings/
└── 2026-02-06_14-54-43/
    ├── raw_videos/                         [1] camera_<id>.avi
    ├── camera_<id>_timestamps.npy          [1]
    ├── metadata.txt                        [1]
    ├── synchronized_videos/                [2] camera_<id>_synchronized.mp4
    ├── frame_mappings_to_ideal_fps.npz     [2]
    ├── synchronization_plots/              [2]
    ├── synchronized_tiled.mp4              [2]
    ├── mediapipe_output/                   [3]
    │   ├── data_2d/                             *_keypoints.npy / .npz / .json
    │   ├── labeled_videos/                      *_labeled.mp4
    │   └── pose_estimation_summary.txt
    ├── pose_3d/                            [4]
    │   ├── pose_3d.npy / .json
    │   └── projection_metrics.txt
    └── (analysis outputs)                  [5] animations/, *.html, plots

    external input: camera_calibration.toml -> [4]
```