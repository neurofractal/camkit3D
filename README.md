# CamKit3D

<p align="center">
  <a href="https://github.com/neurofractal/camkit3D">
    <img src="images/logo.png" width="300">
  </a>
</p>

Multi-camera 3D pose estimation pipeline for naturalistic behaviour research.

CamKit3D turns a set of cheap USB webcams into a markerless motion-capture system. It handles the full journey from raw video to 3D skeleton data: recording, temporal synchronisation, 2D pose estimation, triangulation, and visualisation.

## Installation

```bash
pip install camkit3d
```

Or install from source in editable mode (recommended for development):

```bash
git clone https://github.com/your-repo/camkit3d
cd camkit3d
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, USB 3.0 ports for multi-camera recording, a TOML camera calibration file (e.g. from [Anipose](https://anipose.readthedocs.io/) or [FreeMoCap](https://freemocap.org/)).

---

## Pipeline overview

CamKit3D is organised as five stages that run sequentially. Each stage reads the output of the previous one, so you can re-run any step independently.

### Stage 1 — Multi-camera recording (`camkit3d.recorder`)

Captures video from multiple USB webcams simultaneously. Each camera runs on its own thread; frames are written to disk via a dedicated writer thread so that OS I/O flushes never stall the capture loop. Hardware timestamps (Python `time.time()`) are saved alongside each frame for use in the synchronisation step.

Optional integration with VPixx DataPixx hardware sends digital triggers on recording start/stop, useful for synchronising with EEG, eye-tracking, or other physiology equipment.

```python
from camkit3d import createRecorder
import time

with createRecorder(camera_ids=[0, 1, 2], fps=30) as rec:
    rec.connect_cameras()
    rec.start_recording("trial_001")
    time.sleep(60)
    rec.stop_recording()
```

**Outputs:** `raw_videos/camera_*.avi` + per-camera `.npy` timestamp arrays.

### Stage 2 — Offline synchronisation (`camkit3d.sync`)

Webcams do not share a common clock, so frame *N* from camera 0 and frame *N* from camera 1 will not correspond to the same moment in time. This stage fixes that.

It builds an ideal timing grid at the target FPS spanning the time window common to all cameras. For each tick of that grid it selects the nearest real frame from each camera (using the saved timestamps), then writes new video files that are frame-aligned across cameras. Diagnostic plots show per-camera timing jitter and any dropped frames.

```python
from camkit3d import vidSync

results, figures = vidSync("recordings/trial_001", target_fps=30.0)
```

**Outputs:** `synchronized_videos/camera_*_synchronized.avi` + sync report and plots.

### Stage 3 — 2D pose estimation (`camkit3d.pose2d`)

Runs MediaPipe Pose on each synchronised video to extract 33 body keypoints per frame. Processing is parallelised across cameras using Python multiprocessing. An optional Butterworth low-pass filter can smooth the 2D trajectories before triangulation, which tends to improve 3D reconstruction quality (the idea is to remove jitter *before* it gets amplified by triangulation geometry).

```python
from camkit3d import PoseProcessor

processor = PoseProcessor(
    input_dir="recordings/trial_001/synchronized_videos",
    output_dir="recordings/trial_001/mediapipe_output",
    model_complexity=2,
)
metrics = processor.process_all_videos(save_labeled_videos=True)

# Optional: Butterworth smoothing on 2D keypoints before triangulation
smoothed = processor.smooth_keypoints_butterworth(
    keypoints, cutoff_freq=2.0, sampling_freq=30.0, order=4
)
```

**Outputs:** `data_2d/camera_*_keypoints.npy` (shape: frames × 33 × 3 for x, y, confidence) + labelled overlay videos.

### Stage 4 — 3D triangulation (`camkit3d.pose3d`)

Projects the 2D keypoints from all cameras into 3D world coordinates using the Direct Linear Transform (DLT). Requires a camera calibration file in TOML format containing each camera's intrinsic matrix, distortion coefficients, and extrinsic rotation/translation. Reprojection error is computed for every keypoint on every frame as a quality metric.

```python
from camkit3d import Pose3DProjector

projector = Pose3DProjector(
    calibration_path="camera_calibration.toml",
    keypoints_dir="mediapipe_output/data_2d",
    min_cameras_for_triangulation=2,
    confidence_threshold=0.3,
)
points_3d, metrics = projector.triangulate_all_frames()
projector.save_3d_data(points_3d, "data_3d/pose_3d")
```

**Outputs:** `pose_3d.npy` (shape: frames × 33 × 3 in mm) + projection metrics report.

### Stage 5 — Analysis and animation (`camkit3d.analysis`)

Tools for inspecting and presenting the 3D pose data. Includes automatic detection of body orientation (which way the person is facing), alignment into a standard anatomical reference frame, static skeleton plots, reprojection-error time series, and video export of animated skeletons from single or multiple viewpoints.

```python
from camkit3d import (
    detect_person_orientation,
    align_pose_to_standard_frame,
    animate_3d_pose_auto_orient,
    plot_reprojection_errors,
)

import numpy as np
points_3d = np.load("data_3d/pose_3d.npy")

# Inspect reprojection quality
plot_reprojection_errors(metrics, output_path="reproj_errors.png")

# Auto-orient and animate
orientation = detect_person_orientation(points_3d)
aligned = align_pose_to_standard_frame(points_3d, orientation)
animate_3d_pose_auto_orient(aligned, output_path="skeleton.mp4", fps=30)
```

---

## Smoothing strategy

The recommended approach is **smooth 2D → triangulate → (optional) light 3D smoothing**. Smoothing in 2D first keeps each camera's trajectory internally consistent, gives the DLT cleaner input rays, and makes NaN handling simpler because missing detections can be interpolated per-camera independently.

| Filter | When | Purpose | Typical parameters |
|---|---|---|---|
| OneEuro | During MediaPipe | Real-time jitter reduction | Built-in (`smooth_landmarks=True`) |
| Butterworth | After 2D estimation | Remove remaining high-frequency noise | `cutoff_freq=2.0` Hz, `order=4` |
| Butterworth (3D) | After triangulation | Final polish (optional) | `cutoff_freq=6.0` Hz |

Cutoff frequency depends on the speed of the movement you are studying. Slow movements like conversation gestures or postural sway can use a low cutoff (1–2 Hz in 2D). Faster actions like sports or dance need higher values (5–7 Hz).

---

## File structure

```
trial_001/
├── raw_videos/                     # Stage 1 output
│   ├── camera_0.avi
│   └── camera_0_timestamps.npy
├── synchronized_videos/            # Stage 2 output
│   └── camera_0_synchronized.avi
├── mediapipe_output/               # Stage 3 output
│   ├── data_2d/
│   │   └── camera_0_keypoints.npy
│   └── labeled_videos/
│       └── camera_0_labeled.mp4
└── data_3d/                        # Stage 4 output
    ├── pose_3d.npy
    └── projection_metrics.txt
```

---

## Camera calibration

CamKit3D expects a TOML calibration file with intrinsic and extrinsic parameters for each camera. This format is compatible with Anipose and FreeMoCap calibration workflows. See the [Anipose calibration guide](https://anipose.readthedocs.io/en/latest/calibration.html) for instructions on creating one using a ChArUco board.

---

## Acknowledgements

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) (Google) for 2D keypoint detection
- [Anipose](https://anipose.readthedocs.io/) for triangulation methodology
- [FreeMoCap](https://freemocap.org/) for calibration format and inspiration

---

## License

MIT — see [LICENSE](LICENSE) for details.
