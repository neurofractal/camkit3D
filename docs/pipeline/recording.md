# Stage 1 — Multi-camera recording

`camkit3d.recorder`

Captures video from multiple USB webcams simultaneously. Each camera runs on its own thread; frames are written to disk via a dedicated writer thread so that OS I/O flushes never stall the capture loop. Hardware timestamps (Python `time.time()`) are saved alongside each frame for use in post-hoc synchronisation.

Optional integration with VPixx DataPixx hardware sends digital triggers on recording start/stop, useful for synchronising with OPM/MEG/EEG, eye-tracking, or other physiology equipment.

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

!!! note "Timestamps"
    The per-frame timestamps written here alongside the video feeds into
    [Stage 2](synchronisation.md). Don't discard the `.npy` files.
