# Smoothing strategy

The recommended approach is **smooth 2D → triangulate → (optional) light 3D smoothing**. Smoothing in 2D first keeps each camera's trajectory internally consistent, gives the DLT cleaner input rays, and makes NaN handling simpler because missing detections can be interpolated per-camera independently.

| Filter | When | Purpose | Typical parameters |
|---|---|---|---|
| OneEuro | During MediaPipe | Real-time jitter reduction | Built-in (`smooth_landmarks=True`) |
| Butterworth | After 2D estimation | Remove remaining high-frequency noise | `cutoff_freq=2.0` Hz, `order=4` |
| Butterworth (3D) | After triangulation | Final polish (optional) | `cutoff_freq=6.0` Hz |

Cutoff frequency depends on the speed of the movement you are studying. Slow movements like conversation gestures or postural sway can use a low cutoff (1–2 Hz in 2D). Faster actions like sports or dance need higher values (5–7 Hz).
