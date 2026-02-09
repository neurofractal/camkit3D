"""
Complete Pipeline: 2D Pose Estimation → 3D Projection

This script demonstrates the complete workflow:
1. Process synchronized videos with MediaPipe Pose
2. Extract 2D keypoints with confidence scores
3. Generate labeled videos with skeleton overlay
4. Triangulate 2D keypoints to 3D using camera calibration
5. Compute quality metrics for both stages
6. Save all outputs in FreeMoCap-compatible format

Author: Generated for FreeMoCap-style workflow
Date: 2026-02-06
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from pose_processor import MediaPipePoseProcessor
from pose_3d_projector import Pose3DProjector

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def visualize_reprojection_errors(metrics, output_path: str):
    """
    Create visualization of reprojection errors over time.
    
    Args:
        metrics: Projection3DMetrics object
        output_path: Path to save figure
    """
    errors = metrics.reprojection_errors  # (n_frames, n_keypoints, n_cameras)
    
    # Compute mean error per frame
    mean_errors_per_frame = np.nanmean(errors, axis=(1, 2))
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Mean error over time
    axes[0].plot(mean_errors_per_frame, linewidth=1, alpha=0.7)
    axes[0].axhline(
        metrics.mean_reprojection_error,
        color='r',
        linestyle='--',
        label=f'Overall Mean: {metrics.mean_reprojection_error:.2f}px'
    )
    axes[0].axhline(
        metrics.good_reconstruction_threshold,
        color='g',
        linestyle='--',
        label=f'Good Threshold: {metrics.good_reconstruction_threshold:.2f}px'
    )
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mean Reprojection Error (pixels)')
    axes[0].set_title('Reprojection Error Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Error histogram
    valid_errors = errors[~np.isnan(errors)].flatten()
    axes[1].hist(valid_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(
        metrics.mean_reprojection_error,
        color='r',
        linestyle='--',
        label=f'Mean: {metrics.mean_reprojection_error:.2f}px'
    )
    axes[1].axvline(
        metrics.median_reprojection_error,
        color='b',
        linestyle='--',
        label=f'Median: {metrics.median_reprojection_error:.2f}px'
    )
    axes[1].set_xlabel('Reprojection Error (pixels)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Reprojection Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved reprojection error visualization to {output_path}")


def visualize_3d_trajectory(points_3d: np.ndarray, output_path: str, keypoint_names: list):
    """
    Create 3D visualization of keypoint trajectories.
    
    Args:
        points_3d: Array of shape (n_frames, n_keypoints, 3)
        output_path: Path to save figure
        keypoint_names: List of keypoint names
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Select a few key points to visualize
    key_indices = [0, 11, 12, 23, 24, 15, 16]  # nose, shoulders, hips, wrists
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    
    for idx in key_indices:
        trajectory = points_3d[:, idx, :]
        valid = ~np.isnan(trajectory).any(axis=1)
        
        if np.sum(valid) > 0:
            ax1.plot(
                trajectory[valid, 0],
                trajectory[valid, 1],
                trajectory[valid, 2],
                alpha=0.6,
                label=keypoint_names[idx] if idx < len(keypoint_names) else f'Keypoint {idx}'
            )
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Keypoint Trajectories')
    ax1.legend(fontsize=8)
    
    # Plot 2: Top view (XY)
    ax2 = fig.add_subplot(132)
    
    for idx in key_indices:
        trajectory = points_3d[:, idx, :]
        valid = ~np.isnan(trajectory).any(axis=1)
        
        if np.sum(valid) > 0:
            ax2.plot(
                trajectory[valid, 0],
                trajectory[valid, 1],
                alpha=0.6,
                label=keypoint_names[idx] if idx < len(keypoint_names) else f'Keypoint {idx}'
            )
    
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Top View (XY Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.axis('equal')
    
    # Plot 3: Side view (XZ)
    ax3 = fig.add_subplot(133)
    
    for idx in key_indices:
        trajectory = points_3d[:, idx, :]
        valid = ~np.isnan(trajectory).any(axis=1)
        
        if np.sum(valid) > 0:
            ax3.plot(
                trajectory[valid, 0],
                trajectory[valid, 2],
                alpha=0.6,
                label=keypoint_names[idx] if idx < len(keypoint_names) else f'Keypoint {idx}'
            )
    
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('Side View (XZ Plane)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved 3D trajectory visualization to {output_path}")


def run_complete_pipeline(
    recording_dir: str,
    calibration_file: str,
    output_base_dir: str = None
):
    """
    Run the complete pipeline from synchronized videos to 3D pose.
    
    Args:
        recording_dir: Directory containing 'synchronized_videos' folder
        calibration_file: Path to camera calibration TOML file
        output_base_dir: Base directory for outputs (default: recording_dir)
    """
    recording_dir = Path(recording_dir)
    calibration_file = Path(calibration_file)
    
    if output_base_dir is None:
        output_base_dir = recording_dir
    else:
        output_base_dir = Path(output_base_dir)
    
    # Setup directories
    synchronized_videos_dir = recording_dir / "synchronized_videos"
    mediapipe_output_dir = output_base_dir / "mediapipe_output"
    data_3d_dir = output_base_dir / "data_3d"
    visualization_dir = output_base_dir / "visualizations"
    
    visualization_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("STARTING COMPLETE PIPELINE")
    logger.info("="*70)
    logger.info(f"Recording directory: {recording_dir}")
    logger.info(f"Calibration file: {calibration_file}")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info("")
    
    # =========================================================================
    # STAGE 1: MediaPipe Pose Estimation
    # =========================================================================
    logger.info("="*70)
    logger.info("STAGE 1: MediaPipe 2D Pose Estimation")
    logger.info("="*70)
    
    processor = MediaPipePoseProcessor(
        input_dir=str(synchronized_videos_dir),
        output_dir=str(mediapipe_output_dir),
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    
    pose_2d_metrics = processor.process_all_videos(save_labeled_videos=True)
    
    logger.info("\nStage 1 Complete!")
    logger.info(f"Processed {len(pose_2d_metrics)} videos")
    
    for camera_name, metrics in pose_2d_metrics.items():
        logger.info(f"\n{camera_name}:")
        logger.info(f"  Detection Rate: {metrics.detection_rate:.2%}")
        logger.info(f"  Avg Confidence: {metrics.avg_confidence:.4f}")
    
    # =========================================================================
    # STAGE 2: 3D Triangulation
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: 3D Triangulation")
    logger.info("="*70)
    
    projector = Pose3DProjector(
        calibration_path=str(calibration_file),
        keypoints_dir=str(mediapipe_output_dir / "data_2d"),
        min_cameras_for_triangulation=2,
        confidence_threshold=0.3
    )
    
    points_3d, projection_metrics = projector.triangulate_all_frames()
    
    # Save 3D data
    projector.save_3d_data(
        points_3d,
        output_path=str(data_3d_dir / "pose_3d"),
        metadata={
            'processing_date': datetime.now().isoformat(),
            'mediapipe_model_complexity': 2,
            'min_cameras_for_triangulation': 2,
            'confidence_threshold': 0.3
        }
    )
    
    projector.save_metrics(
        projection_metrics,
        output_path=str(data_3d_dir / "projection_metrics.txt")
    )
    
    logger.info("\nStage 2 Complete!")
    logger.info(f"Mean Reprojection Error: {projection_metrics.mean_reprojection_error:.4f} pixels")
    logger.info(f"Good Reconstruction Rate: {projection_metrics.frames_with_good_reconstruction}/{projection_metrics.n_frames}")
    
    # =========================================================================
    # STAGE 3: Visualization
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STAGE 3: Creating Visualizations")
    logger.info("="*70)
    
    try:
        # Reprojection error visualization
        visualize_reprojection_errors(
            projection_metrics,
            str(visualization_dir / "reprojection_errors.png")
        )
        
        # 3D trajectory visualization
        visualize_3d_trajectory(
            points_3d,
            str(visualization_dir / "3d_trajectories.png"),
            processor.MEDIAPIPE_LANDMARK_NAMES
        )
        
        logger.info("Visualizations created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    
    logger.info("\nOUTPUT SUMMARY:")
    logger.info(f"  Labeled Videos:     {mediapipe_output_dir / 'labeled_videos'}")
    logger.info(f"  2D Keypoint Data:   {mediapipe_output_dir / 'data_2d'}")
    logger.info(f"  3D Pose Data:       {data_3d_dir}")
    logger.info(f"  Visualizations:     {visualization_dir}")
    
    logger.info("\n2D POSE ESTIMATION SUMMARY:")
    avg_detection = np.mean([m.detection_rate for m in pose_2d_metrics.values()])
    avg_confidence = np.mean([m.avg_confidence for m in pose_2d_metrics.values()])
    logger.info(f"  Average Detection Rate:  {avg_detection:.2%}")
    logger.info(f"  Average Confidence:      {avg_confidence:.4f}")
    
    logger.info("\n3D PROJECTION SUMMARY:")
    logger.info(f"  Frames Processed:           {projection_metrics.n_frames}")
    logger.info(f"  Keypoints per Frame:        {projection_metrics.n_keypoints}")
    logger.info(f"  Cameras Used:               {projection_metrics.n_cameras}")
    logger.info(f"  Mean Reprojection Error:    {projection_metrics.mean_reprojection_error:.4f} px")
    logger.info(f"  Median Reprojection Error:  {projection_metrics.median_reprojection_error:.4f} px")
    logger.info(f"  Good Reconstruction Rate:   {projection_metrics.frames_with_good_reconstruction}/{projection_metrics.n_frames}")
    
    logger.info("\n" + "="*70)
    
    return {
        'pose_2d_metrics': pose_2d_metrics,
        'projection_metrics': projection_metrics,
        'points_3d': points_3d
    }


if __name__ == "__main__":
    # Example usage
    RECORDING_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43"
    CALIBRATION_FILE = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/camera_calibration.toml"
    
    # Run pipeline
    results = run_complete_pipeline(
        recording_dir=RECORDING_DIR,
        calibration_file=CALIBRATION_FILE
    )
    
    print("\n✓ Pipeline completed successfully!")
    print("\nYou can now:")
    print("  1. Review labeled videos with pose overlay")
    print("  2. Analyze 2D keypoint data")
    print("  3. Examine 3D reconstructed poses")
    print("  4. Check quality metrics and visualizations")
