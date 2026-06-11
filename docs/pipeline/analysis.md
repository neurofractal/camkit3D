# Stage 5 — Analysis and animation

`camkit3d.analysis`

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
```

### Reprojection Quality
plot_reprojection_errors(metrics, output_path="reproj_errors.png")

### Auto-orient
orientation = detect_person_orientation(points_3d)
aligned = align_pose_to_standard_frame(points_3d, orientation)

### Interpolate Missing Data


### Plotting
animate_3d_pose_auto_orient(aligned, output_path="skeleton.mp4", fps=30)

