"""
MediaPipe Pose Processing Module

This module processes synchronized multi-camera videos using MediaPipe Pose to:
1. Extract 2D pose keypoints from each frame
2. Generate labeled videos with skeleton overlay
3. Save keypoint data in FreeMoCap-compatible format
4. Provide quality metrics for pose estimation

Author: Generated for FreeMoCap-style workflow
Date: 2026-02-06

Performance optimizations (2026-02-11):
- Parallel video processing across cameras using multiprocessing
- Pre-allocated numpy arrays instead of list appending
- Writeable flag on RGB conversion to avoid copy
- Configurable model_complexity for speed/accuracy trade-off
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
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

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


# ── Worker function for multiprocessing ──────────────────────────────────────
# Lives at module level so it can be pickled by multiprocessing.

def _process_single_video_worker(
    video_path: str,
    labeled_videos_dir: str,
    data_2d_dir: str,
    save_labeled_video: bool,
    draw_skeleton: bool,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    model_complexity: int,
    enable_segmentation: bool,
    smooth_landmarks: bool,
) -> Tuple[str, np.ndarray, PoseEstimationMetrics]:
    """
    Standalone worker that processes one video in its own process.
    
    Each worker creates its own MediaPipe Pose instance (they are not
    picklable / thread-safe) so this is safe for multiprocessing.
    """
    video_path = Path(video_path)
    labeled_videos_dir = Path(labeled_videos_dir)
    data_2d_dir = Path(data_2d_dir)
    camera_name = video_path.stem

    # ── Create a fresh MediaPipe Pose per worker ──
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=enable_segmentation,
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # ── Open video ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Prepare video writer ──
    video_writer = None
    if save_labeled_video:
        output_video_path = labeled_videos_dir / f"{camera_name}_labeled.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, (width, height)
        )

    # ── Pre-allocate arrays (avoids repeated list.append + final np.array copy) ──
    n_landmarks = 33
    keypoints_array = np.zeros((total_frames, n_landmarks, 3), dtype=np.float32)
    confidences_list: List[float] = []
    frames_with_detection = 0

    pbar = tqdm(total=total_frames, desc=f"Processing {camera_name}", position=0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR→RGB; mark as non-writeable so MediaPipe can skip a copy
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks:
            frames_with_detection += 1
            landmarks = results.pose_landmarks.landmark

            # Vectorised landmark extraction (avoids per-landmark Python loop)
            xs = np.array([lm.x for lm in landmarks], dtype=np.float32) * width
            ys = np.array([lm.y for lm in landmarks], dtype=np.float32) * height
            vs = np.array([lm.visibility for lm in landmarks], dtype=np.float32)

            keypoints_array[frame_idx, :, 0] = xs
            keypoints_array[frame_idx, :, 1] = ys
            keypoints_array[frame_idx, :, 2] = vs
            confidences_list.extend(vs.tolist())

            if draw_skeleton and video_writer:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                cv2.putText(
                    frame,
                    f"Frame: {frame_idx} | Detection: YES",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        else:
            if video_writer:
                cv2.putText(
                    frame,
                    f"Frame: {frame_idx} | Detection: NO",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        if video_writer:
            video_writer.write(frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if video_writer:
        video_writer.release()
    pose.close()

    # Trim if video had fewer frames than reported
    keypoints_array = keypoints_array[:frame_idx]

    # ── Metrics ──
    confidences = np.array(confidences_list, dtype=np.float32)
    metrics = PoseEstimationMetrics(
        camera_name=camera_name,
        total_frames=frame_idx,
        frames_with_detection=frames_with_detection,
        detection_rate=frames_with_detection / frame_idx if frame_idx > 0 else 0,
        avg_confidence=float(np.mean(confidences)) if len(confidences) > 0 else 0,
        confidence_std=float(np.std(confidences)) if len(confidences) > 0 else 0,
        min_confidence=float(np.min(confidences)) if len(confidences) > 0 else 0,
        max_confidence=float(np.max(confidences)) if len(confidences) > 0 else 0,
        keypoints_per_frame=n_landmarks,
    )

    # ── Save keypoints (npy + npz + json) ──
    _save_keypoints(camera_name, keypoints_array, data_2d_dir)

    return camera_name, keypoints_array, metrics


def _save_keypoints(camera_name: str, keypoints: np.ndarray, data_2d_dir: Path):
    """Save keypoints data in multiple formats (module-level for pickling)."""
    data_2d_dir = Path(data_2d_dir)

    npy_path = data_2d_dir / f"{camera_name}_keypoints.npy"
    np.save(npy_path, keypoints)

    npz_path = data_2d_dir / f"{camera_name}_keypoints.npz"
    np.savez_compressed(
        npz_path,
        keypoints=keypoints,
        landmark_names=MediaPipePoseProcessor.MEDIAPIPE_LANDMARK_NAMES,
    )

    metadata = {
        'camera_name': camera_name,
        'n_frames': int(keypoints.shape[0]),
        'n_landmarks': int(keypoints.shape[1]),
        'landmark_names': MediaPipePoseProcessor.MEDIAPIPE_LANDMARK_NAMES,
        'data_format': 'n_frames x n_landmarks x 3 (x, y, confidence)',
        'coordinate_system': 'pixel coordinates (top-left origin)',
    }
    json_path = data_2d_dir / f"{camera_name}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


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
    
    Performance notes:
    - process_all_videos() runs cameras in parallel using multiprocessing
    - Each worker gets its own MediaPipe Pose instance
    - On Apple Silicon (M3), expect ~2-3x speedup with 3+ cameras
    - For single-camera use, the overhead is minimal
    
    Attributes:
        input_dir (Path): Directory containing synchronized videos
        output_dir (Path): Directory for output files
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
    
    # Landmark groups for filtering
    FACE_LANDMARKS = list(range(0, 11))  # 0-10: face landmarks
    SHOULDER_LANDMARKS = [11, 12]  # left_shoulder, right_shoulder
    ARM_LANDMARKS = [13, 14, 15, 16]  # elbows and wrists
    HAND_LANDMARKS = [17, 18, 19, 20, 21, 22]  # hand keypoints
    HIP_LANDMARKS = [23, 24]  # left_hip, right_hip
    LEG_LANDMARKS = [25, 26, 27, 28]  # knees and ankles
    FOOT_LANDMARKS = [29, 30, 31, 32]  # heels and foot indices
    
    # Landmarks to keep (upper body)
    KEEP_LANDMARKS = (
        FACE_LANDMARKS + 
        SHOULDER_LANDMARKS + 
        ARM_LANDMARKS + 
        HAND_LANDMARKS
    )
    
    # Landmarks to remove (hips and below)
    REMOVE_LANDMARKS = (
        HIP_LANDMARKS + 
        LEG_LANDMARKS + 
        FOOT_LANDMARKS
    )
    
    # Skeleton connections for drawing
    SKELETON_CONNECTIONS = [
        # Face
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
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
        smooth_landmarks: bool = True,
        max_workers: Optional[int] = None,
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
            max_workers: Max parallel camera workers. Defaults to min(n_cameras, cpu_count).
                         Set to 1 to disable parallelism.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output subdirectories
        self.labeled_videos_dir = self.output_dir / "labeled_videos"
        self.labeled_videos_dir.mkdir(exist_ok=True)
        
        self.data_2d_dir = self.output_dir / "data_2d"
        self.data_2d_dir.mkdir(exist_ok=True)
        
        # Store config for spawning workers (don't create Pose here —
        # it can't be pickled and each process needs its own instance)
        self._pose_config = dict(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            smooth_landmarks=smooth_landmarks,
        )
        
        # For the single-video convenience path we still keep a lazy instance
        self._pose: Optional[mp.solutions.pose.Pose] = None
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Parallelism settings
        self.max_workers = max_workers
        
        logger.info(f"MediaPipe Pose Processor initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model complexity: {model_complexity}")

    @property
    def pose(self):
        """Lazy-init a Pose instance for single-video / non-parallel use."""
        if self._pose is None:
            self._pose = self.mp_pose.Pose(
                static_image_mode=False,
                **self._pose_config,
            )
        return self._pose
        
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
        Process a single video file with MediaPipe Pose (in-process).
        
        For parallel multi-camera processing, prefer process_all_videos().
        
        Args:
            video_path: Path to input video
            save_labeled_video: Whether to save video with skeleton overlay
            draw_skeleton: Whether to draw skeleton on frames
            
        Returns:
            Tuple of (keypoints_array, metrics)
            - keypoints_array: shape (n_frames, n_keypoints, 3) [x, y, confidence]
            - metrics: PoseEstimationMetrics object
        """
        camera_name, keypoints, metrics = _process_single_video_worker(
            video_path=str(video_path),
            labeled_videos_dir=str(self.labeled_videos_dir),
            data_2d_dir=str(self.data_2d_dir),
            save_labeled_video=save_labeled_video,
            draw_skeleton=draw_skeleton,
            **self._pose_config,
        )
        return keypoints, metrics
    
    def process_all_videos(
        self,
        save_labeled_videos: bool = True
    ) -> Dict[str, PoseEstimationMetrics]:
        """
        Process all videos in the input directory — in parallel.
        
        Each camera is processed in its own subprocess with its own
        MediaPipe Pose instance.  On an M3 MacBook Air with 3 cameras
        this typically gives a ~2-3x wall-clock speedup.
        
        Args:
            save_labeled_videos: Whether to save videos with skeleton overlay
            
        Returns:
            Dictionary mapping camera names to metrics
        """
        video_files = self.get_video_files()
        
        if not video_files:
            logger.warning("No video files found!")
            return {}

        n_videos = len(video_files)
        # Default: one worker per camera, capped at CPU count
        n_workers = self.max_workers or min(n_videos, os.cpu_count() or 1)
        # Don't spawn more workers than videos
        n_workers = min(n_workers, n_videos)

        all_metrics: Dict[str, PoseEstimationMetrics] = {}
        all_keypoints: Dict[str, np.ndarray] = {}

        t0 = time.perf_counter()

        if n_workers <= 1:
            # ── Sequential fallback (single video or explicit max_workers=1) ──
            logger.info(f"Processing {n_videos} video(s) sequentially")
            for video_path in video_files:
                camera_name, keypoints, metrics = _process_single_video_worker(
                    video_path=str(video_path),
                    labeled_videos_dir=str(self.labeled_videos_dir),
                    data_2d_dir=str(self.data_2d_dir),
                    save_labeled_video=save_labeled_videos,
                    draw_skeleton=True,
                    **self._pose_config,
                )
                all_metrics[camera_name] = metrics
                all_keypoints[camera_name] = keypoints
        else:
            # ── Parallel processing ──────────────────────────────────────
            logger.info(
                f"Processing {n_videos} videos in parallel "
                f"({n_workers} workers)"
            )

            # Use 'spawn' context — safest on macOS and avoids fork issues
            # with MediaPipe / OpenCV.
            ctx = _mp.get_context("spawn")

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
            ) as executor:
                future_to_cam = {}
                for video_path in video_files:
                    fut = executor.submit(
                        _process_single_video_worker,
                        video_path=str(video_path),
                        labeled_videos_dir=str(self.labeled_videos_dir),
                        data_2d_dir=str(self.data_2d_dir),
                        save_labeled_video=save_labeled_videos,
                        draw_skeleton=True,
                        **self._pose_config,
                    )
                    future_to_cam[fut] = video_path.stem

                for future in as_completed(future_to_cam):
                    cam = future_to_cam[future]
                    try:
                        camera_name, keypoints, metrics = future.result()
                        all_metrics[camera_name] = metrics
                        all_keypoints[camera_name] = keypoints
                        logger.info(
                            f"✓ {camera_name} done — "
                            f"detection rate {metrics.detection_rate:.1%}"
                        )
                    except Exception as exc:
                        logger.error(f"✗ {cam} failed: {exc}")

        elapsed = time.perf_counter() - t0
        logger.info(f"All videos processed in {elapsed:.1f}s")

        # Save combined summary
        self._save_summary(all_metrics)
        
        return all_metrics
    
    def _save_keypoints_data(self, camera_name: str, keypoints: np.ndarray):
        """
        Save keypoints data in multiple formats.
        
        Args:
            camera_name: Name of the camera
            keypoints: Keypoint array (n_frames, n_landmarks, 3)
        """
        _save_keypoints(camera_name, keypoints, self.data_2d_dir)
    
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
                signal = keypoints[:, kp_idx, dim].copy()
                
                nan_mask = np.isnan(signal)
                n_nans = nan_mask.sum()
                
                if n_nans == n_frames:
                    n_skipped += 1
                    continue
                
                min_points = max(2 * order + 1, 10)
                if n_frames - n_nans < min_points:
                    logger.warning(
                        f"Keypoint {kp_idx}, dim {dim}: Too few valid points "
                        f"({n_frames - n_nans}/{min_points} required). Skipping."
                    )
                    n_skipped += 1
                    continue
                
                if n_nans > 0:
                    if interpolate_nans:
                        valid_indices = np.where(~nan_mask)[0]
                        valid_values = signal[~nan_mask]
                        
                        interp_func = interp1d(
                            valid_indices,
                            valid_values,
                            kind='linear',
                            bounds_error=False,
                            fill_value=(valid_values[0], valid_values[-1])
                        )
                        
                        signal_interp = interp_func(np.arange(n_frames))
                        n_interpolated += n_nans
                    else:
                        n_skipped += 1
                        continue
                else:
                    signal_interp = signal
                
                try:
                    signal_filtered = filtfilt(b, a, signal_interp)
                except Exception as e:
                    logger.error(
                        f"Filtering failed for keypoint {kp_idx}, dim {dim}: {e}"
                    )
                    n_skipped += 1
                    continue
                
                if restore_nans and n_nans > 0:
                    signal_filtered[nan_mask] = np.nan
                
                smoothed[:, kp_idx, dim] = signal_filtered
        
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
        
        Args:
            keypoints_dict: Dictionary mapping camera names to keypoint arrays
            cutoff_freq: Cutoff frequency in Hz (default: 2.0)
            sampling_freq: Sampling frequency in Hz (default: 30.0)
            order: Filter order (default: 4)
            save_dir: Optional directory to save smoothed data (default: None)
        
        Returns:
            Dictionary mapping camera names to smoothed keypoint arrays
        """
        logger.info(f"Batch smoothing {len(keypoints_dict)} camera views")
        
        smoothed_dict = {}
        
        for camera_name, keypoints in keypoints_dict.items():
            logger.info(f"Smoothing {camera_name}...")
            
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
            
            if save_dir is not None:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = save_dir / f"{camera_name}_keypoints_smoothed.npy"
                np.save(output_path, smoothed)
                logger.info(f"Saved smoothed data to {output_path}")
        
        logger.info("Batch smoothing complete!")
        return smoothed_dict
    
    @staticmethod
    def detect_face_at_edge(
        keypoints_2d: np.ndarray,
        frame_width: int,
        frame_height: int,
        edge_margin_px: int = 50,
        min_face_landmarks_visible: int = 5,
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Detect when face landmarks are near frame edges (likely problematic).
        
        Strategy: If face is within edge_margin_px of any frame edge AND fewer than
        min_face_landmarks_visible have good confidence, mark all face landmarks
        for that frame as suspicious.
        
        Args:
            keypoints_2d: Shape (n_frames, n_landmarks, 3) where last dim is [x, y, confidence]
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
            edge_margin_px: Distance from edge to consider "at edge" (default: 50px)
            min_face_landmarks_visible: Minimum face points needed for valid detection (default: 5)
            confidence_threshold: Confidence threshold for "visible" (default: 0.5)
        
        Returns:
            Boolean mask of shape (n_frames, n_landmarks) indicating problematic face landmarks
            True = suspicious face landmark, should be filtered/handled carefully
        """
        n_frames, n_landmarks, _ = keypoints_2d.shape
        
        suspicious_mask = np.zeros((n_frames, n_landmarks), dtype=bool)
        
        for frame_idx in range(n_frames):
            face_data = keypoints_2d[frame_idx, MediaPipePoseProcessor.FACE_LANDMARKS, :]
            
            # Extract x, y, confidence
            x_coords = face_data[:, 0]
            y_coords = face_data[:, 1]
            confidences = face_data[:, 2]
            
            # Count visible face landmarks
            n_visible = np.sum(confidences > confidence_threshold)
            
            if n_visible < min_face_landmarks_visible:
                # Too few face landmarks visible - likely at edge or occluded
                suspicious_mask[frame_idx, MediaPipePoseProcessor.FACE_LANDMARKS] = True
                continue
            
            # Check if any face landmark is near edge
            valid_points = confidences > confidence_threshold
            
            if not np.any(valid_points):
                continue
            
            x_valid = x_coords[valid_points]
            y_valid = y_coords[valid_points]
            
            # Check proximity to edges
            near_left = np.any(x_valid < edge_margin_px)
            near_right = np.any(x_valid > frame_width - edge_margin_px)
            near_top = np.any(y_valid < edge_margin_px)
            near_bottom = np.any(y_valid > frame_height - edge_margin_px)
            
            if near_left or near_right or near_top or near_bottom:
                suspicious_mask[frame_idx, MediaPipePoseProcessor.FACE_LANDMARKS] = True
        
        return suspicious_mask
    
    @staticmethod
    def filter_and_clean_keypoints_2d(
        keypoints_2d: np.ndarray,
        frame_width: int,
        frame_height: int,
        remove_lower_body: bool = True,
        detect_face_edge: bool = True,
        face_edge_strategy: str = 'reduce_confidence',
        edge_margin_px: int = 50,
        confidence_reduction_factor: float = 0.3
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Filter and clean 2D keypoints by removing problematic landmarks.
        
        Args:
            keypoints_2d: Shape (n_frames, n_landmarks, 3) where last dim is [x, y, confidence]
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
            remove_lower_body: Remove hips and all landmarks below (default: True)
            detect_face_edge: Detect face landmarks at frame edges (default: True)
            face_edge_strategy: How to handle suspicious face landmarks:
                - 'reduce_confidence': Multiply confidence by reduction factor (default)
                - 'remove': Set suspicious face landmarks to NaN
                - 'keep': Keep for manual handling (just return mask)
            edge_margin_px: Pixels from edge for face detection (default: 50)
            confidence_reduction_factor: Factor to reduce confidence by (default: 0.3)
        
        Returns:
            Tuple of:
            - Cleaned keypoints array
            - Face edge mask (if detect_face_edge=True), else None
        """
        cleaned = keypoints_2d.copy()
        
        # Step 1: Remove hips and below
        if remove_lower_body:
            cleaned[:, MediaPipePoseProcessor.REMOVE_LANDMARKS, :] = np.nan
            logger.info(f"  ✓ Removed {len(MediaPipePoseProcessor.REMOVE_LANDMARKS)} landmarks (hips and below)")
            logger.info(f"  ✓ Kept {len(MediaPipePoseProcessor.KEEP_LANDMARKS)} landmarks (face, shoulders, arms, hands)")
        
        # Step 2: Detect and handle face at edge
        face_edge_mask = None
        if detect_face_edge:
            face_edge_mask = MediaPipePoseProcessor.detect_face_at_edge(
                keypoints_2d,  # Use original, not cleaned
                frame_width,
                frame_height,
                edge_margin_px=edge_margin_px
            )
            
            n_suspicious = np.sum(face_edge_mask[:, MediaPipePoseProcessor.FACE_LANDMARKS])
            n_face_total = face_edge_mask.shape[0] * len(MediaPipePoseProcessor.FACE_LANDMARKS)
            pct_suspicious = 100 * n_suspicious / n_face_total
            
            # Apply strategy
            if face_edge_strategy == 'reduce_confidence':
                # Only reduce confidence for face landmarks
                face_mask_only = face_edge_mask.copy()
                for i in range(33):
                    if i not in MediaPipePoseProcessor.FACE_LANDMARKS:
                        face_mask_only[:, i] = False
                
                cleaned[face_mask_only, 2] *= confidence_reduction_factor
                logger.info(f"  ✓ Reduced confidence for {n_suspicious} suspicious face landmarks "
                          f"({pct_suspicious:.1f}%) to {confidence_reduction_factor*100:.0f}%")
            
            elif face_edge_strategy == 'remove':
                # Only remove face landmarks
                face_mask_only = face_edge_mask.copy()
                for i in range(33):
                    if i not in MediaPipePoseProcessor.FACE_LANDMARKS:
                        face_mask_only[:, i] = False
                
                cleaned[face_mask_only] = np.nan
                logger.info(f"  ✓ Removed {n_suspicious} suspicious face landmarks ({pct_suspicious:.1f}%)")
            
            elif face_edge_strategy == 'keep':
                logger.info(f"  ✓ Detected {n_suspicious}/{n_face_total} ({pct_suspicious:.1f}%) "
                          f"suspicious face landmarks (mask returned for manual handling)")
            
            else:
                raise ValueError(f"Unknown face_edge_strategy: {face_edge_strategy}")
        
        return cleaned, face_edge_mask
    
    @staticmethod
    def clean_keypoints_directory(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        frame_width: int,
        frame_height: int,
        remove_lower_body: bool = True,
        detect_face_edge: bool = True,
        face_edge_strategy: str = 'reduce_confidence',
        edge_margin_px: int = 50,
        confidence_reduction_factor: float = 0.3,
        pattern: str = "*_keypoints.npy",
        save_face_masks: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Clean all keypoint files in a directory.
        
        This is the main convenience method for batch processing 2D keypoints.
        
        Args:
            input_dir: Directory containing keypoint files
            output_dir: Directory to save cleaned files
            frame_width: Width of video frames
            frame_height: Height of video frames
            remove_lower_body: Remove hips and below (default: True)
            detect_face_edge: Detect face at edge (default: True)
            face_edge_strategy: 'reduce_confidence', 'remove', or 'keep' (default: 'reduce_confidence')
            edge_margin_px: Distance from frame edge (in pixels) to consider "at edge".
                           Lower = more strict (e.g., 30px), Higher = more lenient (e.g., 100px).
                           Default: 50px. Recommended range: 30-100px.
            confidence_reduction_factor: Factor to multiply confidence by for suspicious face landmarks.
                                        Lower = more aggressive filtering (e.g., 0.1 = 10% confidence),
                                        Higher = less aggressive (e.g., 0.5 = 50% confidence).
                                        Default: 0.3 (30% confidence). Recommended range: 0.1-0.5.
            pattern: Glob pattern for keypoint files (default: "*_keypoints.npy")
            save_face_masks: Save face edge masks (default: True)
        
        Returns:
            Dictionary mapping camera names to cleaned keypoint arrays
        
        Example:
            >>> from pose_processor import MediaPipePoseProcessor
            >>> 
            >>> # Standard cleaning
            >>> cleaned = MediaPipePoseProcessor.clean_keypoints_directory(
            ...     input_dir="/path/to/data_2d",
            ...     output_dir="/path/to/data_2d_cleaned",
            ...     frame_width=1920,
            ...     frame_height=1080
            ... )
            >>> 
            >>> # More aggressive face filtering
            >>> cleaned = MediaPipePoseProcessor.clean_keypoints_directory(
            ...     input_dir="/path/to/data_2d",
            ...     output_dir="/path/to/data_2d_cleaned",
            ...     frame_width=1920,
            ...     frame_height=1080,
            ...     edge_margin_px=75,  # Larger margin catches more
            ...     confidence_reduction_factor=0.1  # Reduce to 10%
            ... )
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all keypoint files
        files = list(input_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return {}
        
        logger.info(f"Found {len(files)} keypoint files to clean")
        logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
        logger.info(f"Remove lower body: {remove_lower_body}")
        logger.info(f"Detect face edge: {detect_face_edge}")
        if detect_face_edge:
            logger.info(f"Face edge strategy: {face_edge_strategy}")
            logger.info(f"Edge margin: {edge_margin_px}px (distance from frame edge)")
            if face_edge_strategy == 'reduce_confidence':
                logger.info(f"Confidence reduction: {confidence_reduction_factor} "
                          f"(suspicious faces → {confidence_reduction_factor*100:.0f}% confidence)")
        logger.info("="*70)
        
        cleaned_dict = {}
        
        for file_path in files:
            # Keep the original filename (e.g., camera_0_synchronized_keypoints.npy)
            original_filename = file_path.name
            camera_name = file_path.stem  # For dict key (e.g., camera_0_synchronized_keypoints)
            
            logger.info(f"\nProcessing {original_filename}...")
            
            # Load
            keypoints_2d = np.load(file_path)
            logger.info(f"  Loaded: {keypoints_2d.shape}")
            
            # Clean
            cleaned, face_mask = MediaPipePoseProcessor.filter_and_clean_keypoints_2d(
                keypoints_2d,
                frame_width,
                frame_height,
                remove_lower_body=remove_lower_body,
                detect_face_edge=detect_face_edge,
                face_edge_strategy=face_edge_strategy,
                edge_margin_px=edge_margin_px,
                confidence_reduction_factor=confidence_reduction_factor
            )
            
            cleaned_dict[camera_name] = cleaned
            
            # Save with ORIGINAL filename (not _cleaned suffix)
            output_path = output_dir / original_filename
            np.save(output_path, cleaned)
            logger.info(f"  ✓ Saved: {output_path}")
            
            # Save face edge mask (with _face_edge_mask suffix)
            if save_face_masks and face_mask is not None:
                mask_filename = original_filename.replace('.npy', '_face_edge_mask.npy')
                mask_path = output_dir / mask_filename
                np.save(mask_path, face_mask)
                logger.info(f"  ✓ Saved face mask: {mask_path}")
        
        logger.info("\n" + "="*70)
        logger.info("Cleaning complete!")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*70)
        
        return cleaned_dict
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, '_pose') and self._pose is not None:
            self._pose.close()


# Example usage
if __name__ == "__main__":
    INPUT_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/synchronized_videos"
    OUTPUT_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/mediapipe_output"
    
    processor = MediaPipePoseProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=2,
        # max_workers=None → auto (one per camera, capped at cpu_count)
        # max_workers=1   → sequential (same behaviour as before)
    )
    
    print("\nProcessing all synchronized videos with MediaPipe Pose...\n")
    metrics = processor.process_all_videos(save_labeled_videos=True)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70 + "\n")
    
    for camera_name, metric in metrics.items():
        print(metric)
    
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"  - Labeled videos: {OUTPUT_DIR}/labeled_videos/")
    print(f"  - 2D keypoint data: {OUTPUT_DIR}/data_2d/")
    print(f"  - Summary report: {OUTPUT_DIR}/pose_estimation_summary.txt")
