# Stage 2 — Offline synchronisation

`camkit3d.sync`

Webcams do not share a common clock, so frame *N* from camera 0 and frame *N* from camera 1 will not correspond to the same moment in time. This stage fixes that.

It builds an ideal timing grid at the target FPS spanning the time window common to all cameras. For each tick of that grid it selects the nearest real frame from each camera (using the saved timestamps), then writes new video files that are frame-aligned across cameras. Diagnostic plots show per-camera timing jitter and any dropped frames.

```python
from camkit3d import vidSync

results, figures = vidSync("recordings/trial_001", target_fps=30.0)
```

**Outputs:** `synchronized_videos/camera_*_synchronized.avi` + sync report and plots.
