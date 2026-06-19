# Stage 3 — 2D pose estimation

`camkit3d.pose2d`

Runs MediaPipe Pose on each synchronised video to extract 33 body keypoints per frame (x, y, confidence). Processing is parallelised across cameras. After estimation, a set of optional cleaning steps prepare the 2D trajectories for triangulation.

## Estimate keypoints

```python
from camkit3d.pose2d import PoseProcessor

processor = PoseProcessor(
    input_dir="recordings/trial_001/synchronized_videos",
    output_dir="recordings/trial_001/mediapipe_output",
    model_complexity=2,          # 0 fastest … 2 most accurate
)
metrics = processor.process_all_videos(save_labeled_videos=True)
```

**Outputs:** `data_2d/camera_*_keypoints.npy` (frames × 33 × 3) + labelled overlay videos. `metrics` reports per-camera detection rate and confidence.

## Clean the keypoints

Three directory-level functions prepare the data for triangulation. Each takes an `input_dir` and, by default, overwrites the files in place (pass `output_dir` to keep originals).

```python
from camkit3d.pose2d import (
    smooth_keypoints_directory,
    remove_lower_body_directory,
    clean_face_points_directory,
)

data_2d = "recordings/trial_001/mediapipe_output/data_2d"

# 1. Butterworth low-pass on x, y — removes jitter before triangulation amplifies it
smooth_keypoints_directory(data_2d, cutoff_freq=4.0, sampling_freq=30.0)

# 2. Drop hips and below (NaN) — keeps face, shoulders, arms, hands
remove_lower_body_directory(data_2d)

# 3. Down-weight or remove unreliable face landmarks near frame edges
clean_face_points_directory(data_2d, frame_width=1920, frame_height=1080)
```

**Smoothing** filters only the x/y channels (confidence untouched); set `sampling_freq` to your true frame rate. **Lower-body removal** affects only hips/legs/feet. **Face cleaning** flags face landmarks that are occluded or near a frame border and, by default, reduces their confidence so weighted triangulation discounts them — see [Face Cleaning](clean_face_points.md) for the strategy options.

!!! tip "Smooth before you triangulate"
    See the [Smoothing Strategy](smoothing.md) page for why 2D smoothing
    before triangulation is the recommended default.