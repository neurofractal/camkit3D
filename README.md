# CamKit3D: Multi-Camera 3D Pose Estimation Pipeline

A complete pipeline for synchronized multi-camera recording, 2D pose estimation with MediaPipe, 3D triangulation, and animation generation for biomechanical research.

## Overview

**CamKit3D** provides a streamlined workflow for markerless motion capture:

1. **Multi-camera recording** with hardware-timed synchronization
2. **2D pose estimation** using MediaPipe Pose (33 keypoints)
3. **3D triangulation** via Direct Linear Transform (DLT)
4. **Orientation-aware animation** with automatic viewpoint detection

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/camkit3d
cd camkit3d

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- USB 3.0 ports for multi-camera recording
- Camera calibration file (TOML format)

---

## Quick Start

### 1. Multi-Camera Recording & Synchronization

```python
from multicam_recorder import MultiCameraRecorder
from video_synchronizer import VideoSynchronizer

# Connect and record
recorder = MultiCameraRecorder(camera_ids=[0, 1, 2], fps=30)
recorder.connect_cameras()
recorder.start_recording("trial_001")
time.sleep(10)
recorder.stop_recording()
recorder.disconnect_cameras()

# Synchronize using hardware timestamps
synchronizer = VideoSynchronizer(trial_folder="recordings/trial_001")
results = synchronizer.synchronize_to_ideal_fps(target_fps=30.0)
```

---

### 2. 2D Pose Estimation

```python
from pose_processor import MediaPipePoseProcessor

processor = MediaPipePoseProcessor(
    input_dir="recordings/trial_001/synchronized_videos",
    output_dir="recordings/trial_001/mediapipe_output",
    model_complexity=2,
    smooth_landmarks=True
)
metrics = processor.process_all_videos(save_labeled_videos=True)
```

**Optional Butterworth Smoothing (recommended before 3D triangulation):**

```python
keypoints = np.load("mediapipe_output/data_2d/camera_0_keypoints.npy")
smoothed = processor.smooth_keypoints_butterworth(
    keypoints, cutoff_freq=2.0, sampling_freq=30.0, order=4
)
np.save("mediapipe_output/data_2d/camera_0_keypoints_smoothed.npy", smoothed)
```

---

### 3. 3D Triangulation

```python
from pose_3d_projector import Pose3DProjector

projector = Pose3DProjector(
    calibration_path="camera_calibration.toml",
    keypoints_dir="mediapipe_output/data_2d_smoothed",
    min_cameras_for_triangulation=2,
    confidence_threshold=0.3
)
points_3d, metrics = projector.triangulate_all_frames()
projector.save_3d_data(points_3d, "data_3d/pose_3d")
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RECORDING                                                │
│    - Multi-camera synchronized recording                   │
│    - Hardware timestamp collection                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SYNCHRONIZATION                                          │
│    - Frame-perfect alignment using timestamps               │
│    - Ideal timing grid generation                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 2D POSE ESTIMATION                                       │
│    - MediaPipe Pose (33 keypoints)                          │
│    - Optional: Butterworth smoothing                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 3D TRIANGULATION                                         │
│    - DLT with camera calibration                            │
│    - World coordinates in mm                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Smoothing Strategy

**Recommended approach: Smooth 2D → Triangulate → (Optional) Light 3D smoothing**

### Why Smooth 2D Before 3D?

1. **Geometric consistency**: Each camera's trajectory becomes internally consistent
2. **Better triangulation**: DLT works better with stable, smooth input rays
3. **Outlier prevention**: Removes noise before it propagates to 3D
4. **Easier NaN handling**: Missing detections handled independently per camera

### Smoothing Options

| Filter | When Applied | Purpose | Parameters |
|--------|-------------|---------|------------|
| **OneEuro** | During MediaPipe | Real-time jitter reduction | `smooth_landmarks=True` (automatic) |
| **Butterworth** | Post 2D estimation | Remove remaining noise | `cutoff_freq=2.0` Hz, `order=4` |
| **Butterworth (3D)** | After triangulation | Final polish (optional) | `cutoff_freq=6.0` Hz (lighter) |

### Cutoff Frequency Guidelines

| Motion Type | 2D Cutoff | 3D Cutoff |
|-------------|-----------|-----------|
| Slow (yoga, walking) | 1-2 Hz | 5-6 Hz |
| Normal (gestures, sports) | 2-3 Hz | 6-8 Hz |
| Fast (running, dance) | 5-7 Hz | 10+ Hz |

---

## Camera Calibration

Calibration file format (TOML):

```toml
[cam_0]
name = "camera_0_synchronized"
size = [1280, 720]
matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
distortions = [k1, 0.0, 0.0, 0.0, 0.0]
rotation = [rx, ry, rz]  # Rodrigues vector
translation = [tx, ty, tz]
world_orientation = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
world_position = [x, y, z]

[cam_1]
# Similar structure for each camera

[metadata]
charuco_square_size = 51.0
date_time_calibrated = "2026-02-06T15:02:56"
```

**Note:** Compatible with FreeMoCap/Anipose calibration workflows.

---

## File Structure

```
trial_001/
├── raw_videos/
│   └── camera_*.avi
├── synchronized_videos/
│   └── camera_*_synchronized.avi
├── mediapipe_output/
│   ├── data_2d/
│   │   └── camera_*_keypoints.npy
│   └── labeled_videos/
│       └── camera_*_labeled.mp4
└── data_3d/
    ├── pose_3d.npy
    └── projection_metrics.txt
```

---

## Performance

Tested on M1 MacBook Pro:

| Stage | Speed | Notes |
|-------|-------|-------|
| Recording | Real-time (30 fps) | Multiple cameras simultaneously |
| Synchronization | ~500 fps | Offline processing |
| 2D Pose | ~30-60 fps | GPU-accelerated |
| 3D Triangulation | ~1000 fps | CPU-based |
| Animation | ~10-30 fps | Depends on quality settings |

**Memory Usage:**
- ~500 MB for typical recording (1000 frames, 3 cameras)
- Scales linearly with frame count and camera number

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Cameras won't connect | Try different IDs [0-4], close other apps |
| Dropped frames | Reduce resolution/FPS, use USB 3.0 |
| Low 2D detection rate | Improve lighting, lower confidence threshold |
| High reprojection error | Check calibration, verify synchronization |

---

## Citation

If you use CamKit3D in your research, please cite:

```bibtex
@software{camkit3d,
  title={CamKit3D: Multi-Camera 3D Pose Estimation Pipeline},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo/camkit3d},
  note={Based on MediaPipe Pose and Anipose methodology}
}
```

**Acknowledgments:**
- MediaPipe Pose (Google)
- Anipose triangulation methodology
- FreeMoCap calibration format

---

## License

MIT License - See LICENSE file for details

---

## Support

- **Documentation:** See notebooks in `examples/`
- **Issues:** GitHub issue tracker
- **Calibration help:** Use FreeMoCap calibration tools

---

**CamKit3D** - Professional markerless motion capture for biomechanical research.
