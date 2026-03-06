"""Smoke tests — verify that the package imports cleanly."""

import pytest


def test_import_package():
    import camkit3d
    assert hasattr(camkit3d, "__version__")


def test_import_recorder():
    from camkit3d.recorder import MultiCamRecorder, create_recorder


def test_import_sync():
    from camkit3d.sync import synchronize_videos_to_ideal_fps, vid_sync


def test_import_sync_qa():
    from camkit3d.sync_qa import VideoAnalyzer


def test_import_pose2d():
    from camkit3d.pose2d import PoseProcessor, PoseEstimationMetrics


def test_import_pose3d():
    from camkit3d.pose3d import Pose3DProjector, CameraCalibration, Projection3DMetrics


def test_import_analysis():
    from camkit3d.analysis import (
        animate_3d_pose,
        animate_3d_pose_auto_orient,
        animate_3d_pose_multiangle,
        detect_person_orientation,
        align_pose_to_standard_frame,
        plot_reprojection_errors,
        SKELETON_CONNECTIONS,
        BODY_PART_COLORS,
    )
