# Batch processing

!!! warning "Placeholder"
    This page is a scaffold. The README does not yet document a batch API,
    so the example below shows a simple loop pattern over trials. Replace it
    with the real batch interface once it exists.

When processing many trials, the simplest approach is to loop the [end-to-end
workflow](tutorials/end-to-end.md) over a list of trial directories.

```python
from pathlib import Path
from camkit3d import vidSync, PoseProcessor, Pose3DProjector

trials = sorted(Path("recordings").glob("trial_*"))

for trial in trials:
    # Stage 2 — synchronise
    vidSync(str(trial), target_fps=30.0)

    # Stage 3 — 2D pose
    processor = PoseProcessor(
        input_dir=trial / "synchronized_videos",
        output_dir=trial / "mediapipe_output",
        model_complexity=2,
    )
    processor.process_all_videos(save_labeled_videos=False)

    # Stage 4 — triangulate
    projector = Pose3DProjector(
        calibration_path="camera_calibration.toml",
        keypoints_dir=trial / "mediapipe_output" / "data_2d",
        min_cameras_for_triangulation=2,
        confidence_threshold=0.3,
    )
    points_3d, metrics = projector.triangulate_all_frames()
    projector.save_3d_data(points_3d, trial / "data_3d" / "pose_3d")
```

<!--
TODO: Document the real batch interface, e.g.:
- a CLI entry point (camkit3d batch ...) if one exists
- parallelisation across trials
- resume / skip-completed behaviour
- consolidated QC report across the batch
-->
