"""
MediaPipe Pose Processing Module

This module processes synchronized multi-camera videos using MediaPipe Pose to:
1. Extract 2D pose keypoints from each frame
2. Generate labeled videos with skeleton overlay
3. Save keypoint data in FreeMoCap-compatible format
4. Provide quality metrics for pose estimation

Author: CamKit3D (FreeMoCap-compatible workflow)
Date: 2026-02-06

"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from tqdm.auto import tqdm
import logging
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import queue

from camkit3d import skeletons as _skeletons
import threading

# Module-level default skeleton. Hoisted here (not just on the class) so that
# comprehensions inside the class body can reference it — class-body scope is
# not visible to comprehension/generator scopes, but module scope is.
_SKELETON = _skeletons.load()  # mediapipe_pose

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
    progress_queue=None,  # Queue for progress updates
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
    n_landmarks = 33  # fixed by MediaPipe Pose (BlazePose) — matches mediapipe_pose descriptor
    keypoints_array = np.zeros((total_frames, n_landmarks, 3), dtype=np.float32)
    confidences_list: List[float] = []
    frames_with_detection = 0

    # Progress tracking removed from worker - handled in main process
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
        
        # Report progress every 10 frames
        if progress_queue and frame_idx % 10 == 0:
            progress_queue.put((camera_name, frame_idx, total_frames))

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
        landmark_names=PoseProcessor.MEDIAPIPE_LANDMARK_NAMES,
    )

    metadata = {
        'camera_name': camera_name,
        'n_frames': int(keypoints.shape[0]),
        'n_landmarks': int(keypoints.shape[1]),
        'landmark_names': PoseProcessor.MEDIAPIPE_LANDMARK_NAMES,
        'data_format': 'n_frames x n_landmarks x 3 (x, y, confidence)',
        'coordinate_system': 'pixel coordinates (top-left origin)',
    }
    json_path = data_2d_dir / f"{camera_name}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


class PoseProcessor:
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
        >>> processor = PoseProcessor(
        ...     input_dir="path/to/synchronized_videos",
        ...     output_dir="path/to/output"
        ... )
        >>> metrics = processor.process_all_videos()
        >>> for metric in metrics.values():
        ...     print(metric)
    """
    
    # ── Skeleton topology (from the skeleton descriptor) ──
    # pose2d runs MediaPipe inference, so its default skeleton is MediaPipe
    # Pose. These class attributes are derived from the descriptor so the
    # topology lives in one place. To process with a different skeleton,
    # subclass and set SKELETON, or call set_skeleton() before instantiating.
    SKELETON = _skeletons.load()  # mediapipe_pose

    MEDIAPIPE_LANDMARK_NAMES = SKELETON.names

    # Landmark groups for filtering (resolved by group / anchor)
    FACE_LANDMARKS = list(SKELETON.group_indices("face"))
    HAND_LANDMARKS = list(SKELETON.group_indices("hand"))
    SHOULDER_LANDMARKS = [SKELETON.anchor("left_shoulder"),
                          SKELETON.anchor("right_shoulder")]
    HIP_LANDMARKS = [SKELETON.anchor("left_hip"), SKELETON.anchor("right_hip")]
    # Arm = arm-group members that aren't shoulders or hands
    _arm_all = set(SKELETON.group_indices("left_arm")) | set(SKELETON.group_indices("right_arm"))
    ARM_LANDMARKS = sorted(_arm_all - set(SHOULDER_LANDMARKS) - set(HAND_LANDMARKS))
    # Leg = leg-group members that aren't hips; foot = heels + foot indices
    _leg_all = sorted((set(SKELETON.group_indices("left_leg"))
                       | set(SKELETON.group_indices("right_leg"))) - set(HIP_LANDMARKS))
    _foot_names = ("left_heel", "right_heel", "left_foot_index", "right_foot_index")
    FOOT_LANDMARKS = sorted(_SKELETON.index_of(_n) for _n in _foot_names
                            if _n in _SKELETON.names)
    LEG_LANDMARKS = sorted(set(_leg_all) - set(FOOT_LANDMARKS))

    # Landmarks to keep (upper body) / remove (hips and below)
    KEEP_LANDMARKS = sorted(set(FACE_LANDMARKS) | set(SHOULDER_LANDMARKS)
                            | set(ARM_LANDMARKS) | set(HAND_LANDMARKS))
    REMOVE_LANDMARKS = sorted(set(HIP_LANDMARKS) | set(LEG_LANDMARKS) | set(FOOT_LANDMARKS))

    # Skeleton connections for drawing (kept for reference / non-MediaPipe
    # overlays; the labelled-video overlay below uses MediaPipe's own
    # POSE_CONNECTIONS styling).
    SKELETON_CONNECTIONS = SKELETON.edges
    
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
        
        logger.info(f"PoseProcessor initialized")
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
            pbar = tqdm(total=n_videos, desc="Processing videos", unit="video")
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
                pbar.set_postfix_str(f"{camera_name} - {metrics.detection_rate:.1%} detection")
                pbar.update(1)
            pbar.close()
        else:
            # ── Parallel processing ──────────────────────────────────────
            logger.info(
                f"Processing {n_videos} videos in parallel "
                f"({n_workers} workers)"
            )

            # Use 'spawn' context — safest on macOS and avoids fork issues
            # with MediaPipe / OpenCV.
            ctx = _mp.get_context("spawn")
            
            # Create progress queue for real-time updates
            manager = ctx.Manager()
            progress_queue = manager.Queue()
            
            # Get frame counts and create progress bars
            video_frame_counts = {}
            progress_bars = {}
            
            for idx, video_path in enumerate(video_files):
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cam_name = video_path.stem
                video_frame_counts[cam_name] = frame_count
                cap.release()
                
                # Create a progress bar for each video with unique position
                progress_bars[cam_name] = tqdm(
                    total=frame_count,
                    desc=f"{cam_name}",
                    position=idx,
                    leave=True,
                    unit="frame"
                )
            
            # Thread to monitor progress queue and update correct progress bar
            stop_monitoring = threading.Event()
            
            def monitor_progress():
                frames_processed = {cam: 0 for cam in video_frame_counts.keys()}
                while not stop_monitoring.is_set():
                    try:
                        camera_name, frame_idx, total = progress_queue.get(timeout=0.1)
                        if camera_name in progress_bars:
                            frames_done = frame_idx - frames_processed[camera_name]
                            frames_processed[camera_name] = frame_idx
                            progress_bars[camera_name].update(frames_done)
                    except:
                        pass
            
            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()
            
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
                        progress_queue=progress_queue,
                        **self._pose_config,
                    )
                    future_to_cam[fut] = video_path.stem

                for future in as_completed(future_to_cam):
                    cam = future_to_cam[future]
                    try:
                        camera_name, keypoints, metrics = future.result()
                        all_metrics[camera_name] = metrics
                        all_keypoints[camera_name] = keypoints
                        
                        # Close the progress bar for this camera
                        if camera_name in progress_bars:
                            progress_bars[camera_name].set_postfix_str(
                                f"✓ {metrics.detection_rate:.1%} detection"
                            )
                            progress_bars[camera_name].close()
                        
                        logger.info(
                            f"✓ {camera_name} done — "
                            f"detection rate {metrics.detection_rate:.1%}"
                        )
                    except Exception as exc:
                        logger.error(f"✗ {cam} failed: {exc}")
                        if cam in progress_bars:
                            progress_bars[cam].close()
            
            # Stop monitoring and clean up
            stop_monitoring.set()
            monitor_thread.join(timeout=1.0)
            
            # Ensure all progress bars are closed
            for pbar in progress_bars.values():
                if not pbar.disable:
                    pbar.close()


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
            face_data = keypoints_2d[frame_idx, PoseProcessor.FACE_LANDMARKS, :]
            
            # Extract x, y, confidence
            x_coords = face_data[:, 0]
            y_coords = face_data[:, 1]
            confidences = face_data[:, 2]
            
            # Count visible face landmarks
            n_visible = np.sum(confidences > confidence_threshold)
            
            if n_visible < min_face_landmarks_visible:
                # Too few face landmarks visible - likely at edge or occluded
                suspicious_mask[frame_idx, PoseProcessor.FACE_LANDMARKS] = True
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
                suspicious_mask[frame_idx, PoseProcessor.FACE_LANDMARKS] = True
        
        return suspicious_mask
    
    @staticmethod
    def remove_lower_body_keypoints(
        keypoints_2d: np.ndarray
    ) -> np.ndarray:
        """
        Remove lower-body landmarks (hips and everything below) by setting them to NaN.

        For upper-body tasks (face, shoulders, arms, hands) the legs and feet carry
        no useful signal and only add noise to downstream triangulation, so they are
        dropped here. Face, shoulder, arm and hand landmarks are left untouched.

        Args:
            keypoints_2d: Shape (n_frames, n_landmarks, 3), last dim is [x, y, confidence]

        Returns:
            A copy of the array with REMOVE_LANDMARKS (hips, legs, feet) set to NaN.
        """
        cleaned = keypoints_2d.copy()
        cleaned[:, PoseProcessor.REMOVE_LANDMARKS, :] = np.nan
        logger.info(f"  ✓ Removed {len(PoseProcessor.REMOVE_LANDMARKS)} landmarks (hips and below)")
        logger.info(f"  ✓ Kept {len(PoseProcessor.KEEP_LANDMARKS)} landmarks (face, shoulders, arms, hands)")
        return cleaned

    @staticmethod
    def clean_face_points(
        keypoints_2d: np.ndarray,
        frame_width: int,
        frame_height: int,
        face_edge_strategy: str = 'reduce_confidence',
        edge_margin_px: int = 50,
        confidence_reduction_factor: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and handle spurious face landmarks near the frame edges.

        Face landmarks tend to be unreliable when the face is close to a frame
        boundary or only partially visible. This detects those frames and either
        down-weights or removes the affected face landmarks. Only face landmarks
        are ever modified — shoulders, arms and hands keep their original values.

        Args:
            keypoints_2d: Shape (n_frames, n_landmarks, 3), last dim is [x, y, confidence]
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
            face_edge_strategy: How to handle suspicious face landmarks:
                - 'reduce_confidence': Multiply confidence by reduction factor (default)
                - 'remove': Set suspicious face landmarks to NaN
                - 'keep': Keep for manual handling (just return mask)
            edge_margin_px: Pixels from edge for face detection (default: 50)
            confidence_reduction_factor: Factor to reduce confidence by (default: 0.3)

        Returns:
            Tuple of:
            - Cleaned keypoints array (a copy)
            - Face edge mask of shape (n_frames, n_landmarks)
        """
        cleaned = keypoints_2d.copy()

        face_edge_mask = PoseProcessor.detect_face_at_edge(
            keypoints_2d,  # Use the input as-is for detection
            frame_width,
            frame_height,
            edge_margin_px=edge_margin_px
        )

        n_suspicious = np.sum(face_edge_mask[:, PoseProcessor.FACE_LANDMARKS])
        n_face_total = face_edge_mask.shape[0] * len(PoseProcessor.FACE_LANDMARKS)
        pct_suspicious = 100 * n_suspicious / n_face_total

        # Restrict any action to face landmarks only
        face_mask_only = face_edge_mask.copy()
        for i in range(33):
            if i not in PoseProcessor.FACE_LANDMARKS:
                face_mask_only[:, i] = False

        if face_edge_strategy == 'reduce_confidence':
            cleaned[face_mask_only, 2] *= confidence_reduction_factor
            logger.info(f"  ✓ Reduced confidence for {n_suspicious} suspicious face landmarks "
                      f"({pct_suspicious:.1f}%) to {confidence_reduction_factor*100:.0f}%")

        elif face_edge_strategy == 'remove':
            cleaned[face_mask_only] = np.nan
            logger.info(f"  ✓ Removed {n_suspicious} suspicious face landmarks ({pct_suspicious:.1f}%)")

        elif face_edge_strategy == 'keep':
            logger.info(f"  ✓ Detected {n_suspicious}/{n_face_total} ({pct_suspicious:.1f}%) "
                      f"suspicious face landmarks (mask returned for manual handling)")

        else:
            raise ValueError(f"Unknown face_edge_strategy: {face_edge_strategy}")

        return cleaned, face_edge_mask

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, '_pose') and self._pose is not None:
            self._pose.close()


# ── Standalone batch functions ────────────────────────────────────────────────
# These operate on directories of keypoint files and don't need a PoseProcessor
# instance. They wrap the per-array methods on PoseProcessor.

def smooth_keypoints_directory(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    cutoff_freq: float = 4.0,
    sampling_freq: float = 30.0,
    order: int = 4,
    interpolate_nans: bool = True,
    restore_nans: bool = True,
    pattern: str = "*_keypoints.npy",
) -> Dict[str, np.ndarray]:
    """
    Batch temporal smoothing: Butterworth low-pass over every keypoint file in a directory.

    If output_dir is None, smoothed data overwrites the raw files in place.
    Only x, y are filtered; the confidence channel (index 2) is left untouched.

    Args:
        input_dir: Directory containing keypoint files (e.g. the data_2d folder)
        output_dir: Directory to save smoothed files. If None (default), files
                    are overwritten in place in input_dir.
        cutoff_freq: Low-pass cutoff in Hz (default: 4.0). Lower = smoother.
        sampling_freq: Frame rate in Hz (default: 30.0). Must exceed 2*cutoff_freq.
        order: Butterworth filter order (default: 4)
        interpolate_nans: Interpolate NaNs before filtering (default: True)
        restore_nans: Restore original NaN positions after filtering (default: True)
        pattern: Glob pattern for keypoint files (default: "*_keypoints.npy")

    Returns:
        Dictionary mapping camera names to smoothed keypoint arrays.

    Example:
        >>> from camkit3d.pose2d import smooth_keypoints_directory
        >>> # Overwrite raw keypoints in place
        >>> smooth_keypoints_directory(
        ...     input_dir="/path/to/data_2d",
        ...     cutoff_freq=4.0,
        ...     sampling_freq=29.97,
        ... )
    """
    input_dir = Path(input_dir)
    output_dir = input_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_place = output_dir.resolve() == input_dir.resolve()

    if cutoff_freq >= sampling_freq / 2:
        raise ValueError(
            f"cutoff_freq ({cutoff_freq} Hz) must be < Nyquist ({sampling_freq/2} Hz)"
        )

    nyquist = sampling_freq / 2.0
    b, a = butter(order, cutoff_freq / nyquist, btype="low", analog=False)

    files = [f for f in sorted(input_dir.glob(pattern))
             if not f.name.endswith("_face_edge_mask.npy")]
    if not files:
        logger.warning(f"No files found matching pattern: {pattern}")
        return {}

    logger.info(f"Found {len(files)} keypoint files to smooth")
    logger.info(f"Cutoff: {cutoff_freq}Hz | Sampling: {sampling_freq}Hz | Order: {order}")
    logger.info(f"Writing {'in place (overwriting raw files)' if in_place else f'to {output_dir}'}")

    min_points = max(2 * order + 1, 10)
    smoothed_dict = {}

    for file_path in files:
        camera_name = file_path.stem
        logger.info(f"\nProcessing {file_path.name}...")
        kp = np.load(file_path)
        logger.info(f"  Loaded: {kp.shape}")

        smoothed = kp.copy()
        n_frames, n_kp, _ = kp.shape

        for k in range(n_kp):
            for dim in range(2):  # x, y only
                sig = kp[:, k, dim].copy()
                nan_mask = np.isnan(sig)
                n_nan = int(nan_mask.sum())

                if n_frames - n_nan < min_points:
                    continue  # too few valid points; leave untouched

                if n_nan > 0 and interpolate_nans:
                    valid = np.where(~nan_mask)[0]
                    vals = sig[~nan_mask]
                    interp = interp1d(valid, vals, kind="linear",
                                      bounds_error=False,
                                      fill_value=(vals[0], vals[-1]))
                    sig = interp(np.arange(n_frames))
                elif n_nan > 0:
                    continue

                filt = filtfilt(b, a, sig)
                if restore_nans and n_nan > 0:
                    filt[nan_mask] = np.nan
                smoothed[:, k, dim] = filt

        smoothed_dict[camera_name] = smoothed
        np.save(output_dir / file_path.name, smoothed)
        logger.info(f"  ✓ Saved: {output_dir / file_path.name}")

    logger.info("\n" + "="*70)
    logger.info("Smoothing complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)

    return smoothed_dict


def remove_lower_body_directory(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*_keypoints.npy",
) -> Dict[str, np.ndarray]:
    """
    Batch lower-body removal: NaN out hips/legs/feet for every keypoint file in a directory.

    If output_dir is None, results overwrite the input files in place.

    Args:
        input_dir: Directory containing keypoint files
        output_dir: Directory to save results. If None (default), overwrite in place.
        pattern: Glob pattern for keypoint files (default: "*_keypoints.npy")

    Returns:
        Dictionary mapping camera names to processed keypoint arrays.

    Example:
        >>> from camkit3d.pose2d import remove_lower_body_directory
        >>> remove_lower_body_directory(
        ...     input_dir="/path/to/data_2d",
        ...     output_dir="/path/to/data_2d_upper",
        ... )
    """
    input_dir = Path(input_dir)
    output_dir = input_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_place = output_dir.resolve() == input_dir.resolve()

    files = [f for f in sorted(input_dir.glob(pattern))
             if not f.name.endswith("_face_edge_mask.npy")]
    if not files:
        logger.warning(f"No files found matching pattern: {pattern}")
        return {}

    logger.info(f"Found {len(files)} keypoint files for lower-body removal")
    logger.info(f"Writing {'in place (overwriting input files)' if in_place else f'to {output_dir}'}")
    logger.info("="*70)

    result_dict = {}
    for file_path in files:
        camera_name = file_path.stem
        logger.info(f"\nProcessing {file_path.name}...")
        kp = np.load(file_path)
        logger.info(f"  Loaded: {kp.shape}")

        kept = PoseProcessor.remove_lower_body_keypoints(kp)
        result_dict[camera_name] = kept

        np.save(output_dir / file_path.name, kept)
        logger.info(f"  ✓ Saved: {output_dir / file_path.name}")

    logger.info("\n" + "="*70)
    logger.info("Lower-body removal complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)

    return result_dict


def clean_face_points_directory(
    input_dir: Union[str, Path],
    frame_width: int,
    frame_height: int,
    output_dir: Optional[Union[str, Path]] = None,
    face_edge_strategy: str = 'reduce_confidence',
    edge_margin_px: int = 50,
    confidence_reduction_factor: float = 0.3,
    pattern: str = "*_keypoints.npy",
    save_face_masks: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Batch face cleaning: handle spurious edge face landmarks for every keypoint file.

    If output_dir is None, results overwrite the input files in place.

    Args:
        input_dir: Directory containing keypoint files
        frame_width: Width of video frames in pixels
        frame_height: Height of video frames in pixels
        output_dir: Directory to save results. If None (default), overwrite in place.
        face_edge_strategy: 'reduce_confidence', 'remove', or 'keep' (default: 'reduce_confidence')
        edge_margin_px: Distance from frame edge (px) to consider "at edge" (default: 50)
        confidence_reduction_factor: Confidence multiplier for suspicious faces (default: 0.3)
        pattern: Glob pattern for keypoint files (default: "*_keypoints.npy")
        save_face_masks: Save face edge masks alongside output (default: True)

    Returns:
        Dictionary mapping camera names to cleaned keypoint arrays.

    Example:
        >>> from camkit3d.pose2d import clean_face_points_directory
        >>> clean_face_points_directory(
        ...     input_dir="/path/to/data_2d_upper",
        ...     frame_width=1920,
        ...     frame_height=1080,
        ...     output_dir="/path/to/data_2d_cleaned",
        ... )
    """
    input_dir = Path(input_dir)
    output_dir = input_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_place = output_dir.resolve() == input_dir.resolve()

    files = [f for f in sorted(input_dir.glob(pattern))
             if not f.name.endswith("_face_edge_mask.npy")]
    if not files:
        logger.warning(f"No files found matching pattern: {pattern}")
        return {}

    logger.info(f"Found {len(files)} keypoint files for face cleaning")
    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
    logger.info(f"Face edge strategy: {face_edge_strategy}")
    logger.info(f"Edge margin: {edge_margin_px}px")
    if face_edge_strategy == 'reduce_confidence':
        logger.info(f"Confidence reduction: {confidence_reduction_factor} "
                  f"(suspicious faces → {confidence_reduction_factor*100:.0f}% confidence)")
    logger.info(f"Writing {'in place (overwriting input files)' if in_place else f'to {output_dir}'}")
    logger.info("="*70)

    cleaned_dict = {}
    for file_path in files:
        camera_name = file_path.stem
        logger.info(f"\nProcessing {file_path.name}...")
        kp = np.load(file_path)
        logger.info(f"  Loaded: {kp.shape}")

        cleaned, face_mask = PoseProcessor.clean_face_points(
            kp,
            frame_width,
            frame_height,
            face_edge_strategy=face_edge_strategy,
            edge_margin_px=edge_margin_px,
            confidence_reduction_factor=confidence_reduction_factor,
        )
        cleaned_dict[camera_name] = cleaned

        np.save(output_dir / file_path.name, cleaned)
        logger.info(f"  ✓ Saved: {output_dir / file_path.name}")

        if save_face_masks and face_mask is not None:
            mask_filename = file_path.name.replace('.npy', '_face_edge_mask.npy')
            np.save(output_dir / mask_filename, face_mask)
            logger.info(f"  ✓ Saved face mask: {output_dir / mask_filename}")

    logger.info("\n" + "="*70)
    logger.info("Face cleaning complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)

    return cleaned_dict


# Example usage
if __name__ == "__main__":
    INPUT_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/synchronized_videos"
    OUTPUT_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/mediapipe_output"
    
    processor = PoseProcessor(
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
    
    # for camera_name, metric in metrics.items():
    #     print(metric)
    
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"  - Labeled videos: {OUTPUT_DIR}/labeled_videos/")
    print(f"  - 2D keypoint data: {OUTPUT_DIR}/data_2d/")
    print(f"  - Summary report: {OUTPUT_DIR}/pose_estimation_summary.txt")
