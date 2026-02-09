"""
MediaPipe Pose Processing Module

This module processes synchronized multi-camera videos using MediaPipe Pose to:
1. Extract 2D pose keypoints from each frame
2. Generate labeled videos with skeleton overlay
3. Save keypoint data in FreeMoCap-compatible format
4. Provide quality metrics for pose estimation

Author: Generated for FreeMoCap-style workflow
Date: 2026-02-06
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from tqdm import tqdm
import logging
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PoseEstimationMetrics:
    """Container for pose estimation quality metrics"""
    camera_name: str
    total_frames: int
    frames_with_detection: int
    detection_rate: float
    avg_confidence: float
    confidence_std: float
    min_confidence: float
    max_confidence: float
    keypoints_per_frame: int
    
    def __str__(self):
        return (
            f"\n{'='*60}\n"
            f"Pose Estimation Metrics - {self.camera_name}\n"
            f"{'='*60}\n"
            f"Total Frames:           {self.total_frames}\n"
            f"Frames with Detection:  {self.frames_with_detection}\n"
            f"Detection Rate:         {self.detection_rate:.2%}\n"
            f"Average Confidence:     {self.avg_confidence:.4f}\n"
            f"Confidence Std Dev:     {self.confidence_std:.4f}\n"
            f"Min Confidence:         {self.min_confidence:.4f}\n"
            f"Max Confidence:         {self.max_confidence:.4f}\n"
            f"Keypoints per Frame:    {self.keypoints_per_frame}\n"
            f"{'='*60}\n"
        )


class MediaPipePoseProcessor:
    """
    Process synchronized videos with MediaPipe Pose estimation.
    
    This class handles:
    - Loading synchronized videos from multiple cameras
    - Running MediaPipe Pose on each frame
    - Extracting 2D keypoint coordinates and confidence scores
    - Generating labeled videos with skeleton overlay
    - Saving data in FreeMoCap-compatible format
    - Computing quality metrics
    
    Attributes:
        input_dir (Path): Directory containing synchronized videos
        output_dir (Path): Directory for output files
        mp_pose: MediaPipe Pose detector instance
        mediapipe_keypoint_names (List[str]): Names of MediaPipe landmarks
        
    Example:
        >>> processor = MediaPipePoseProcessor(
        ...     input_dir="path/to/synchronized_videos",
        ...     output_dir="path/to/output"
        ... )
        >>> metrics = processor.process_all_videos()
        >>> for metric in metrics.values():
        ...     print(metric)
    """
    
    # MediaPipe Pose landmark names (33 landmarks)
    MEDIAPIPE_LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    # Skeleton connections for drawing
    SKELETON_CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye to ear
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye to ear
        (9, 10),  # Mouth
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        # Right arm
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32),
    ]
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 2,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True
    ):
        """
        Initialize the MediaPipe Pose Processor.
        
        Args:
            input_dir: Path to directory containing synchronized videos
            output_dir: Path to directory for output files
            min_detection_confidence: Minimum confidence for detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0)
            model_complexity: Model complexity (0, 1, or 2, higher = more accurate)
            enable_segmentation: Whether to generate segmentation mask
            smooth_landmarks: Whether to apply temporal smoothing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        self.labeled_videos_dir = self.output_dir / "labeled_videos"
        self.labeled_videos_dir.mkdir(exist_ok=True)
        
        self.data_2d_dir = self.output_dir / "data_2d"
        self.data_2d_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"MediaPipe Pose Processor initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model complexity: {model_complexity}")
        
    def get_video_files(self) -> List[Path]:
        """
        Get all video files from the input directory.
        
        Returns:
            List of Path objects for video files
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(f'*{ext}'))
        
        video_files = sorted(video_files)
        logger.info(f"Found {len(video_files)} video files")
        
        return video_files
    
    def process_video(
        self,
        video_path: Path,
        save_labeled_video: bool = True,
        draw_skeleton: bool = True
    ) -> Tuple[np.ndarray, PoseEstimationMetrics]:
        """
        Process a single video file with MediaPipe Pose.
        
        Args:
            video_path: Path to input video
            save_labeled_video: Whether to save video with skeleton overlay
            draw_skeleton: Whether to draw skeleton on frames
            
        Returns:
            Tuple of (keypoints_array, metrics)
            - keypoints_array: shape (n_frames, n_keypoints, 3) [x, y, confidence]
            - metrics: PoseEstimationMetrics object
        """
        camera_name = video_path.stem
        logger.info(f"Processing video: {camera_name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Prepare video writer if needed
        video_writer = None
        if save_labeled_video:
            output_video_path = self.labeled_videos_dir / f"{camera_name}_labeled.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_video_path), fourcc, fps, (width, height)
            )
        
        # Storage for keypoints
        n_landmarks = 33  # MediaPipe Pose has 33 landmarks
        keypoints_data = []
        confidences = []
        frames_with_detection = 0
        
        # Process each frame
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {camera_name}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(image_rgb)
            
            # Extract keypoints
            frame_keypoints = np.zeros((n_landmarks, 3))  # x, y, confidence
            
            if results.pose_landmarks:
                frames_with_detection += 1
                
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    frame_keypoints[idx, 0] = landmark.x * width
                    frame_keypoints[idx, 1] = landmark.y * height
                    frame_keypoints[idx, 2] = landmark.visibility
                    confidences.append(landmark.visibility)
                
                # Draw skeleton if requested
                if draw_skeleton and video_writer:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Add frame number and detection status
                    cv2.putText(
                        frame,
                        f"Frame: {frame_idx} | Detection: YES",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            else:
                # No detection - add text
                if video_writer:
                    cv2.putText(
                        frame,
                        f"Frame: {frame_idx} | Detection: NO",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            
            keypoints_data.append(frame_keypoints)
            
            # Write frame if saving video
            if video_writer:
                video_writer.write(frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if video_writer:
            video_writer.release()
        
        # Convert to numpy array
        keypoints_array = np.array(keypoints_data)  # Shape: (n_frames, n_landmarks, 3)
        
        # Calculate metrics
        confidences = np.array(confidences)
        metrics = PoseEstimationMetrics(
            camera_name=camera_name,
            total_frames=total_frames,
            frames_with_detection=frames_with_detection,
            detection_rate=frames_with_detection / total_frames if total_frames > 0 else 0,
            avg_confidence=np.mean(confidences) if len(confidences) > 0 else 0,
            confidence_std=np.std(confidences) if len(confidences) > 0 else 0,
            min_confidence=np.min(confidences) if len(confidences) > 0 else 0,
            max_confidence=np.max(confidences) if len(confidences) > 0 else 0,
            keypoints_per_frame=n_landmarks
        )
        
        # Save keypoints data
        self._save_keypoints_data(camera_name, keypoints_array)
        
        logger.info(f"Completed processing {camera_name}")
        logger.info(f"Detection rate: {metrics.detection_rate:.2%}")
        
        return keypoints_array, metrics
    
    def _save_keypoints_data(self, camera_name: str, keypoints: np.ndarray):
        """
        Save keypoints data in multiple formats.
        
        Args:
            camera_name: Name of the camera
            keypoints: Keypoint array (n_frames, n_landmarks, 3)
        """
        # Save as .npy (binary, fast)
        npy_path = self.data_2d_dir / f"{camera_name}_keypoints.npy"
        np.save(npy_path, keypoints)
        
        # Save as .npz (compressed)
        npz_path = self.data_2d_dir / f"{camera_name}_keypoints.npz"
        np.savez_compressed(
            npz_path,
            keypoints=keypoints,
            landmark_names=self.MEDIAPIPE_LANDMARK_NAMES
        )
        
        # Save metadata as JSON
        metadata = {
            'camera_name': camera_name,
            'n_frames': int(keypoints.shape[0]),
            'n_landmarks': int(keypoints.shape[1]),
            'landmark_names': self.MEDIAPIPE_LANDMARK_NAMES,
            'data_format': 'n_frames x n_landmarks x 3 (x, y, confidence)',
            'coordinate_system': 'pixel coordinates (top-left origin)'
        }
        
        json_path = self.data_2d_dir / f"{camera_name}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved keypoints data to {self.data_2d_dir}")
    
    def process_all_videos(
        self,
        save_labeled_videos: bool = True
    ) -> Dict[str, PoseEstimationMetrics]:
        """
        Process all videos in the input directory.
        
        Args:
            save_labeled_videos: Whether to save videos with skeleton overlay
            
        Returns:
            Dictionary mapping camera names to metrics
        """
        video_files = self.get_video_files()
        
        if not video_files:
            logger.warning("No video files found!")
            return {}
        
        all_metrics = {}
        all_keypoints = {}
        
        for video_path in video_files:
            keypoints, metrics = self.process_video(
                video_path,
                save_labeled_video=save_labeled_videos
            )
            
            camera_name = video_path.stem
            all_metrics[camera_name] = metrics
            all_keypoints[camera_name] = keypoints
        
        # Save combined summary
        self._save_summary(all_metrics)
        
        return all_metrics
    
    def _save_summary(self, metrics_dict: Dict[str, PoseEstimationMetrics]):
        """
        Save a summary of all processing metrics.
        
        Args:
            metrics_dict: Dictionary of camera names to metrics
        """
        summary_path = self.output_dir / "pose_estimation_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("MEDIAPIPE POSE ESTIMATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for camera_name, metrics in metrics_dict.items():
                f.write(str(metrics))
                f.write("\n")
            
            # Overall statistics
            f.write("\nOVERALL STATISTICS\n")
            f.write("=" * 70 + "\n")
            
            avg_detection_rate = np.mean([m.detection_rate for m in metrics_dict.values()])
            avg_confidence = np.mean([m.avg_confidence for m in metrics_dict.values()])
            
            f.write(f"Number of cameras: {len(metrics_dict)}\n")
            f.write(f"Average detection rate: {avg_detection_rate:.2%}\n")
            f.write(f"Average confidence: {avg_confidence:.4f}\n")
        
        logger.info(f"Summary saved to {summary_path}")
    
    def smooth_keypoints_butterworth(
        self,
        keypoints: np.ndarray,
        cutoff_freq: float = 2.0,
        sampling_freq: float = 30.0,
        order: int = 4,
        interpolate_nans: bool = True,
        restore_nans: bool = True
    ) -> np.ndarray:
        """
        Apply Butterworth low-pass filter to keypoint trajectories.
        
        This method smooths keypoint data while properly handling NaN values.
        The filter is applied using zero-phase filtering (filtfilt) to avoid
        temporal phase shifts.
        
        Args:
            keypoints: Array of shape (n_frames, n_keypoints, 3) containing
                      [x, y, confidence/visibility] for each keypoint
            cutoff_freq: Cutoff frequency in Hz (default: 2.0 Hz)
                        Lower values = more smoothing
                        Typical range: 1-10 Hz
            sampling_freq: Sampling frequency (frame rate) in Hz (default: 30.0)
            order: Filter order (default: 4)
                  Higher order = sharper cutoff but potential instability
                  Typical range: 2-6
            interpolate_nans: Whether to interpolate NaN values before filtering
                            (default: True)
            restore_nans: Whether to restore original NaN positions after filtering
                         (default: True)
        
        Returns:
            Smoothed keypoints array of same shape as input
            
        Example:
            >>> # After processing videos
            >>> keypoints = np.load('data_2d/camera_0_keypoints.npy')
            >>> smoothed = processor.smooth_keypoints_butterworth(
            ...     keypoints,
            ...     cutoff_freq=2.0,
            ...     sampling_freq=30.0,
            ...     order=4
            ... )
            >>> np.save('data_2d/camera_0_keypoints_smoothed.npy', smoothed)
        
        Notes:
            - NaN handling strategy:
              1. Detect NaN positions
              2. Interpolate NaNs (if interpolate_nans=True)
              3. Apply Butterworth filter
              4. Restore original NaNs (if restore_nans=True)
            
            - Confidence/visibility channel (index 2) is NOT filtered
            
            - Edge effects: The filtfilt function uses padding to minimize
              edge effects, but first/last few frames may still show artifacts
            
            - Nyquist frequency: cutoff_freq must be < sampling_freq/2
        """
        logger.info(f"Applying Butterworth filter (cutoff={cutoff_freq}Hz, order={order})")
        
        # Validate inputs
        if keypoints.ndim != 3:
            raise ValueError(f"Expected 3D array (n_frames, n_keypoints, 3), got shape {keypoints.shape}")
        
        if cutoff_freq >= sampling_freq / 2:
            raise ValueError(
                f"Cutoff frequency ({cutoff_freq} Hz) must be less than "
                f"Nyquist frequency ({sampling_freq/2} Hz)"
            )
        
        n_frames, n_keypoints, n_dims = keypoints.shape
        
        # Design Butterworth filter
        nyquist = sampling_freq / 2.0
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # Initialize output array
        smoothed = keypoints.copy()
        
        # Only filter x, y coordinates (not confidence channel)
        dims_to_filter = 2  # x and y only
        
        # Track statistics
        n_interpolated = 0
        n_skipped = 0
        
        # Process each keypoint independently
        for kp_idx in range(n_keypoints):
            for dim in range(dims_to_filter):
                # Extract time series for this keypoint dimension
                signal = keypoints[:, kp_idx, dim].copy()
                
                # Find NaN positions
                nan_mask = np.isnan(signal)
                n_nans = nan_mask.sum()
                
                # Skip if all values are NaN
                if n_nans == n_frames:
                    n_skipped += 1
                    continue
                
                # Skip if too few valid points for filtering
                # Need at least 2*order + 1 points for filtfilt
                min_points = max(2 * order + 1, 10)
                if n_frames - n_nans < min_points:
                    logger.warning(
                        f"Keypoint {kp_idx}, dim {dim}: Too few valid points "
                        f"({n_frames - n_nans}/{min_points} required). Skipping."
                    )
                    n_skipped += 1
                    continue
                
                # Handle NaNs
                if n_nans > 0:
                    if interpolate_nans:
                        # Interpolate NaNs for filtering
                        valid_indices = np.where(~nan_mask)[0]
                        valid_values = signal[~nan_mask]
                        
                        # Create interpolation function
                        # Use 'linear' for interior, 'nearest' for extrapolation
                        interp_func = interp1d(
                            valid_indices,
                            valid_values,
                            kind='linear',
                            bounds_error=False,
                            fill_value=(valid_values[0], valid_values[-1])
                        )
                        
                        # Interpolate all points
                        signal_interp = interp_func(np.arange(n_frames))
                        n_interpolated += n_nans
                    else:
                        # Skip this signal if NaNs present and interpolation disabled
                        n_skipped += 1
                        continue
                else:
                    signal_interp = signal
                
                # Apply Butterworth filter (zero-phase)
                try:
                    signal_filtered = filtfilt(b, a, signal_interp)
                except Exception as e:
                    logger.error(
                        f"Filtering failed for keypoint {kp_idx}, dim {dim}: {e}"
                    )
                    n_skipped += 1
                    continue
                
                # Restore NaNs if requested
                if restore_nans and n_nans > 0:
                    signal_filtered[nan_mask] = np.nan
                
                # Store filtered signal
                smoothed[:, kp_idx, dim] = signal_filtered
        
        # Log statistics
        logger.info(f"Filtering complete:")
        logger.info(f"  - Total signals processed: {n_keypoints * dims_to_filter}")
        logger.info(f"  - Signals skipped: {n_skipped}")
        if interpolate_nans:
            logger.info(f"  - NaN values interpolated: {n_interpolated}")
        
        return smoothed
    
    @staticmethod
    def smooth_keypoints_batch(
        keypoints_dict: Dict[str, np.ndarray],
        cutoff_freq: float = 2.0,
        sampling_freq: float = 30.0,
        order: int = 4,
        save_dir: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply Butterworth smoothing to multiple cameras' keypoint data.
        
        This is a convenience method for batch processing all cameras at once.
        
        Args:
            keypoints_dict: Dictionary mapping camera names to keypoint arrays
            cutoff_freq: Cutoff frequency in Hz (default: 2.0)
            sampling_freq: Sampling frequency in Hz (default: 30.0)
            order: Filter order (default: 4)
            save_dir: Optional directory to save smoothed data (default: None)
        
        Returns:
            Dictionary mapping camera names to smoothed keypoint arrays
            
        Example:
            >>> # Load all keypoint files
            >>> keypoints = {}
            >>> for npy_file in Path('data_2d').glob('*_keypoints.npy'):
            ...     cam_name = npy_file.stem.replace('_keypoints', '')
            ...     keypoints[cam_name] = np.load(npy_file)
            >>> 
            >>> # Smooth all cameras
            >>> smoothed = MediaPipePoseProcessor.smooth_keypoints_batch(
            ...     keypoints,
            ...     cutoff_freq=2.0,
            ...     save_dir=Path('data_2d_smoothed')
            ... )
        """
        logger.info(f"Batch smoothing {len(keypoints_dict)} camera views")
        
        smoothed_dict = {}
        
        for camera_name, keypoints in keypoints_dict.items():
            logger.info(f"Smoothing {camera_name}...")
            
            # Create temporary processor instance just for the smoothing method
            # (This is a static method, so we need to create instance)
            temp_processor = MediaPipePoseProcessor.__new__(MediaPipePoseProcessor)
            temp_processor.input_dir = Path(".")
            temp_processor.output_dir = Path(".")
            
            smoothed = temp_processor.smooth_keypoints_butterworth(
                keypoints,
                cutoff_freq=cutoff_freq,
                sampling_freq=sampling_freq,
                order=order
            )
            
            smoothed_dict[camera_name] = smoothed
            
            # Save if directory provided
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = save_dir / f"{camera_name}_keypoints_smoothed.npy"
                np.save(output_path, smoothed)
                logger.info(f"Saved smoothed data to {output_path}")
        
        logger.info("Batch smoothing complete!")
        return smoothed_dict
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


# Example usage
if __name__ == "__main__":
    # Example configuration
    INPUT_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/synchronized_videos"
    OUTPUT_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/mediapipe_output"
    
    # Create processor
    processor = MediaPipePoseProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2
    )
    
    # Process all videos
    print("\nProcessing all synchronized videos with MediaPipe Pose...\n")
    metrics = processor.process_all_videos(save_labeled_videos=True)
    
    # Print results
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70 + "\n")
    
    for camera_name, metric in metrics.items():
        print(metric)
    
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"  - Labeled videos: {OUTPUT_DIR}/labeled_videos/")
    print(f"  - 2D keypoint data: {OUTPUT_DIR}/data_2d/")
    print(f"  - Summary report: {OUTPUT_DIR}/pose_estimation_summary.txt")