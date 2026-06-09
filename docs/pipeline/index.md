# Pipeline overview

CamKit3D is organised as five stages that run sequentially. Each stage reads the output of the previous one, so you can re-run any step independently.

| Stage | Module | Page |
|---|---|---|
| 1 | `camkit3d.recorder` | [Multi-camera Recording](recording.md) |
| 2 | `camkit3d.sync` | [Offline Synchronisation](synchronisation.md) |
| 3 | `camkit3d.pose2d` | [2D Pose Estimation](pose2d.md) |
| 4 | `camkit3d.pose3d` | [3D Triangulation](triangulation.md) |
| 5 | `camkit3d.analysis` | [Analysis & Animation](analysis.md) |

See also the [Smoothing Strategy](smoothing.md) for guidance on where and how to filter the data.

## File structure

A complete trial directory after all five stages:

```
trial_001/
├── raw_videos/                     # Stage 1 output
│   ├── camera_0.avi
│   └── camera_0_timestamps.npy
├── synchronized_videos/            # Stage 2 output
│   └── camera_0_synchronized.avi
├── mediapipe_output/               # Stage 3 output
│   ├── data_2d/
│   │   └── camera_0_keypoints.npy
│   └── labeled_videos/
│       └── camera_0_labeled.mp4
└── data_3d/                        # Stage 4 output
    ├── pose_3d.npy
    └── projection_metrics.txt
```
