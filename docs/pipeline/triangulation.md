# Stage 4 — 3D triangulation

`camkit3d.pose3d`

Projects the 2D keypoints from all cameras into 3D world coordinates using the Direct Linear Transform (DLT). Requires a camera calibration file in TOML format containing each camera's intrinsic matrix, distortion coefficients, and extrinsic rotation/translation. Reprojection error is computed for every keypoint on every frame and can be used as a quality metric.

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

!!! note "You need a calibration file"
    The TOML calibration file is required here. See
    [Camera Calibration](../tutorials/calibration.md) to create one.
