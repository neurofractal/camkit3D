"""
CamKit3D — Multi-camera 3D pose estimation pipeline.

A complete workflow for markerless motion capture: synchronised multi-camera
recording, 2D pose estimation (MediaPipe), 3D triangulation (DLT), and
analysis / animation of the resulting skeleton data.
"""

__version__ = "0.1.0"

# Stage 1 – Recording
from camkit3d.recorder import MultiCamRecorder, CameraThread, create_recorder

# Stage 2 – Synchronisation
from camkit3d.sync import (
    synchronize_videos_to_ideal_fps,
    plot_sync_results,
    plot_sync_summary_stats,
    vid_sync,
)

# Stage 2b – Sync QA
from camkit3d.sync_qa import VideoAnalyzer

# Stage 3 – 2D Pose Estimation
from camkit3d.pose2d import PoseProcessor, PoseEstimationMetrics

# Stage 4 – 3D Triangulation
from camkit3d.pose3d import Pose3DProjector, CameraCalibration, Projection3DMetrics

# Stage 5 – Analysis & Animation
from camkit3d.analysis import (
    plot_reprojection_errors,
    detect_person_orientation,
    get_optimal_camera_angles,
    visualize_orientation,
    align_pose_to_standard_frame,
    plot_aligned_skeleton,
    animate_3d_pose,
    animate_3d_pose_auto_orient,
    animate_3d_pose_multiangle,
    SKELETON_CONNECTIONS,
    BODY_PART_COLORS,
)

__all__ = [
    # Recording
    "MultiCamRecorder",
    "CameraThread",
    "create_recorder",
    # Synchronisation
    "synchronize_videos_to_ideal_fps",
    "plot_sync_results",
    "plot_sync_summary_stats",
    "vid_sync",
    # Sync QA
    "VideoAnalyzer",
    # 2D Pose
    "PoseProcessor",
    "PoseEstimationMetrics",
    # 3D Projection
    "Pose3DProjector",
    "CameraCalibration",
    "Projection3DMetrics",
    # Analysis & Animation
    "plot_reprojection_errors",
    "detect_person_orientation",
    "get_optimal_camera_angles",
    "visualize_orientation",
    "align_pose_to_standard_frame",
    "plot_aligned_skeleton",
    "animate_3d_pose",
    "animate_3d_pose_auto_orient",
    "animate_3d_pose_multiangle",
    "SKELETON_CONNECTIONS",
    "BODY_PART_COLORS",
]
