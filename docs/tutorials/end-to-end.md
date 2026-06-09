# End-to-end walkthrough

This tutorial runs a single trial through all five stages. It stitches together the per-stage snippets from the [Pipeline](../pipeline/index.md) section into one continuous workflow.

!!! note "Prerequisites"
    You'll need a [calibration file](calibration.md) before Stage 4.

## 1. Record

```python
from camkit3d import createRecorder
import time

with createRecorder(camera_ids=[0, 1, 2], fps=30) as rec:
    rec.connect_cameras()
    rec.start_recording("trial_001")
    time.sleep(60)
    rec.stop_recording()
```

## 2. Synchronise

```python
from camkit3d import vidSync

results, figures = vidSync("recordings/trial_001", target_fps=30.0)
```

## 3. 2D pose estimation

```python
from camkit3d import PoseProcessor

processor = PoseProcessor(
    input_dir="recordings/trial_001/synchronized_videos",
    output_dir="recordings/trial_001/mediapipe_output",
    model_complexity=2,
)
metrics = processor.process_all_videos(save_labeled_videos=True)
```

## 4. Triangulate to 3D

```python
from camkit3d import Pose3DProjector

projector = Pose3DProjector(
    calibration_path="camera_calibration.toml",
    keypoints_dir="recordings/trial_001/mediapipe_output/data_2d",
    min_cameras_for_triangulation=2,
    confidence_threshold=0.3,
)
points_3d, metrics = projector.triangulate_all_frames()
projector.save_3d_data(points_3d, "recordings/trial_001/data_3d/pose_3d")
```

## 5. Analyse and animate

```python
import numpy as np
from camkit3d import (
    detect_person_orientation,
    align_pose_to_standard_frame,
    animate_3d_pose_auto_orient,
)

points_3d = np.load("recordings/trial_001/data_3d/pose_3d.npy")
orientation = detect_person_orientation(points_3d)
aligned = align_pose_to_standard_frame(points_3d, orientation)
animate_3d_pose_auto_orient(aligned, output_path="skeleton.mp4", fps=30)
```

That's a full trial from raw webcam video to an animated 3D skeleton.
