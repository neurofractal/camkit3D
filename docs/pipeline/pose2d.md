# Stage 3 — 2D pose estimation

`camkit3d.pose2d`

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

!!! tip "Smooth before you triangulate"
    See the [Smoothing Strategy](smoothing.md) page for why 2D smoothing
    before triangulation is the recommended default.
