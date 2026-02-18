"""
3D Pose Projection Module

This module projects 2D pose keypoints from multiple cameras into 3D space using:
1. Camera calibration parameters (intrinsics and extrinsics)
2. Triangulation via Direct Linear Transform (DLT)
3. Bundle adjustment for refinement (optional)
4. Quality metrics for 3D reconstruction

Based on Anipose methodology for multi-camera 3D pose estimation.

Author: Generated for FreeMoCap-style workflow
Date: 2026-02-06
"""

import numpy as np
import toml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Projection3DMetrics:
    """Container for 3D projection quality metrics"""
    n_frames: int
    n_keypoints: int
    n_cameras: int
    reprojection_errors: np.ndarray  # Shape: (n_frames, n_keypoints, n_cameras)
    mean_reprojection_error: float
    median_reprojection_error: float
    std_reprojection_error: float
    max_reprojection_error: float
    reconstruction_confidence: np.ndarray  # Shape: (n_frames, n_keypoints)
    frames_with_good_reconstruction: int
    good_reconstruction_threshold: float = 10.0  # pixels
    
    def __str__(self):
        good_rate = (self.frames_with_good_reconstruction / self.n_frames * 100 
                     if self.n_frames > 0 else 0)
        return (
            f"\n{'='*70}\n"
            f"3D Projection Metrics\n"
            f"{'='*70}\n"
            f"Frames Processed:              {self.n_frames}\n"
            f"Keypoints per Frame:           {self.n_keypoints}\n"
            f"Cameras Used:                  {self.n_cameras}\n"
            f"\nReprojection Errors (pixels):\n"
            f"  Mean:                        {self.mean_reprojection_error:.4f}\n"
            f"  Median:                      {self.median_reprojection_error:.4f}\n"
            f"  Std Dev:                     {self.std_reprojection_error:.4f}\n"
            f"  Max:                         {self.max_reprojection_error:.4f}\n"
            f"\nReconstruction Quality:\n"
            f"  Frames with good recon:      {self.frames_with_good_reconstruction}\n"
            f"  Good reconstruction rate:    {good_rate:.2f}%\n"
            f"  Threshold used:              {self.good_reconstruction_threshold:.2f} pixels\n"
            f"{'='*70}\n"
        )


class CameraCalibration:
    """
    Container for camera calibration parameters.
    
    Stores intrinsic and extrinsic parameters for a single camera.
    """
    
    def __init__(
        self,
        name: str,
        size: Tuple[int, int],
        matrix: np.ndarray,
        distortions: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        world_orientation: Optional[np.ndarray] = None,
        world_position: Optional[np.ndarray] = None
    ):
        """
        Initialize camera calibration.
        
        Args:
            name: Camera identifier
            size: Image size (width, height)
            matrix: 3x3 camera matrix (intrinsics)
            distortions: Distortion coefficients
            rotation: Rodrigues rotation vector
            translation: Translation vector
            world_orientation: 3x3 rotation matrix in world coordinates
            world_position: 3D position in world coordinates
        """
        self.name = name
        self.size = size
        self.matrix = np.array(matrix)
        self.distortions = np.array(distortions)
        self.rotation = np.array(rotation)
        self.translation = np.array(translation)
        
        # Convert rotation vector to matrix
        self.rotation_matrix, _ = cv2.Rodrigues(self.rotation)
        
        # World coordinates (optional)
        self.world_orientation = (np.array(world_orientation) 
                                 if world_orientation is not None else None)
        self.world_position = (np.array(world_position) 
                              if world_position is not None else None)
        
        # Compute projection matrix
        self.projection_matrix = self._compute_projection_matrix()
    
    def _compute_projection_matrix(self) -> np.ndarray:
        """
        Compute the 3x4 projection matrix.
        
        P = K @ [R | t]
        
        Returns:
            3x4 projection matrix
        """
        # Create extrinsic matrix [R | t]
        extrinsic = np.hstack([self.rotation_matrix, self.translation.reshape(3, 1)])
        
        # Multiply with intrinsic matrix
        P = self.matrix @ extrinsic
        
        return P
    
    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: Array of shape (N, 3) or (N, 4) for homogeneous coords
            
        Returns:
            Array of shape (N, 2) with 2D pixel coordinates
        """
        # Convert to homogeneous coordinates if needed
        if points_3d.shape[1] == 3:
            points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        else:
            points_3d_hom = points_3d
        
        # Project
        points_2d_hom = (self.projection_matrix @ points_3d_hom.T).T
        
        # Convert from homogeneous to Cartesian
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        
        return points_2d
    
    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Undistort 2D points using distortion coefficients.
        
        Args:
            points_2d: Array of shape (N, 2) with distorted pixel coordinates
            
        Returns:
            Array of shape (N, 2) with undistorted coordinates
        """
        # Reshape for cv2
        points_2d_reshaped = points_2d.reshape(-1, 1, 2).astype(np.float32)
        
        # Undistort
        points_undistorted = cv2.undistortPoints(
            points_2d_reshaped,
            self.matrix,
            self.distortions,
            P=self.matrix
        )
        
        return points_undistorted.reshape(-1, 2)


class Pose3DProjector:
    """
    Project 2D pose keypoints from multiple cameras into 3D space.
    
    This class implements:
    - Loading camera calibration from TOML files
    - Loading 2D keypoint data from multiple cameras
    - Triangulation using Direct Linear Transform (DLT)
    - Reprojection error computation
    - Quality metrics and visualization
    
    Attributes:
        calibration_path (Path): Path to camera calibration TOML file
        cameras (Dict[str, CameraCalibration]): Camera calibration objects
        keypoints_2d (Dict[str, np.ndarray]): 2D keypoints per camera
        
    Example:
        >>> projector = Pose3DProjector(
        ...     calibration_path="camera_calibration.toml",
        ...     keypoints_dir="path/to/data_2d"
        ... )
        >>> points_3d, metrics = projector.triangulate_all_frames()
        >>> projector.save_3d_data(points_3d, "output_3d.npy")
        >>> print(metrics)
    """
    
    # MediaPipe landmark groups — face landmarks are the ones that
    # hallucinate most when the face is occluded or at frame edges.
    FACE_LANDMARK_INDICES = list(range(0, 11))    # nose, eyes, ears, mouth
    HAND_LANDMARK_INDICES = [17, 18, 19, 20, 21, 22]  # pinky, index, thumb
    BODY_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    
    def __init__(
        self,
        calibration_path: str,
        keypoints_dir: str,
        min_cameras_for_triangulation: int = 2,
        confidence_threshold: float = 0.3,
        face_confidence_threshold: Optional[float] = None,
        hand_confidence_threshold: Optional[float] = None,
        reprojection_error_threshold: float = 15.0,
        use_iterative_rejection: bool = True,
    ):
        """
        Initialize the 3D projector.
        
        Args:
            calibration_path: Path to TOML calibration file
            keypoints_dir: Directory containing 2D keypoint .npy files
            min_cameras_for_triangulation: Minimum cameras needed to triangulate
            confidence_threshold: Minimum confidence to use a 2D point (body)
            face_confidence_threshold: Minimum confidence for face landmarks.
                Defaults to confidence_threshold + 0.2 (i.e. stricter).
                Face landmarks are especially prone to hallucination during occlusion.
            hand_confidence_threshold: Minimum confidence for hand landmarks.
                Defaults to confidence_threshold + 0.1.
            reprojection_error_threshold: Per-camera reprojection error (px) above
                which a camera view is considered an outlier and excluded in a
                second triangulation pass. Set to 0 or np.inf to disable.
            use_iterative_rejection: If True, after initial DLT triangulation,
                re-triangulate excluding any camera whose reprojection error
                exceeds reprojection_error_threshold. This is the key fix for
                bad face/hand points from a single camera blowing up the 3D result.
        """
        self.calibration_path = Path(calibration_path)
        self.keypoints_dir = Path(keypoints_dir)
        self.min_cameras = min_cameras_for_triangulation
        self.confidence_threshold = confidence_threshold
        self.face_confidence_threshold = (
            face_confidence_threshold if face_confidence_threshold is not None
            else confidence_threshold + 0.2
        )
        self.hand_confidence_threshold = (
            hand_confidence_threshold if hand_confidence_threshold is not None
            else confidence_threshold + 0.1
        )
        self.reprojection_error_threshold = reprojection_error_threshold
        self.use_iterative_rejection = use_iterative_rejection
        
        # Build a per-keypoint confidence threshold array (33 landmarks)
        self._per_kp_threshold = np.full(33, self.confidence_threshold)
        for idx in self.FACE_LANDMARK_INDICES:
            self._per_kp_threshold[idx] = self.face_confidence_threshold
        for idx in self.HAND_LANDMARK_INDICES:
            self._per_kp_threshold[idx] = self.hand_confidence_threshold
        
        # Load calibration
        self.cameras = self._load_calibration()
        logger.info(f"Loaded calibration for {len(self.cameras)} cameras")
        
        # Load 2D keypoints
        self.keypoints_2d = self._load_keypoints_2d()
        logger.info(f"Loaded 2D keypoints from {len(self.keypoints_2d)} cameras")
        
        # Validate
        self._validate_data()
    
    def _load_calibration(self) -> Dict[str, CameraCalibration]:
        """
        Load camera calibration from TOML file.
        
        Returns:
            Dictionary mapping camera names to CameraCalibration objects
        """
        with open(self.calibration_path, 'r') as f:
            calib_data = toml.load(f)
        
        cameras = {}
        
        for key, value in calib_data.items():
            if key.startswith('cam_'):
                cam_calib = CameraCalibration(
                    name=value['name'],
                    size=tuple(value['size']),
                    matrix=value['matrix'],
                    distortions=value['distortions'],
                    rotation=value['rotation'],
                    translation=value['translation'],
                    world_orientation=value.get('world_orientation'),
                    world_position=value.get('world_position')
                )
                cameras[value['name']] = cam_calib
        
        return cameras
    
    def _load_keypoints_2d(self) -> Dict[str, np.ndarray]:
        """
        Load 2D keypoints from .npy files.
        
        Returns:
            Dictionary mapping camera names to keypoint arrays
            Each array has shape (n_frames, n_keypoints, 3) [x, y, confidence]
        """
        keypoints = {}
        
        for npy_file in self.keypoints_dir.glob("*_keypoints.npy"):
            camera_name = npy_file.stem.replace("_keypoints", "")
            data = np.load(npy_file)
            keypoints[camera_name] = data
            logger.info(f"Loaded {camera_name}: shape {data.shape}")
        
        return keypoints
    
    def _validate_data(self):
        """Validate that cameras and keypoints match."""
        camera_names = set(self.cameras.keys())
        keypoint_names = set(self.keypoints_2d.keys())
        
        if not camera_names.intersection(keypoint_names):
            raise ValueError("No matching camera names between calibration and keypoints!")
        
        # Warn about mismatches
        only_calib = camera_names - keypoint_names
        only_keypoints = keypoint_names - camera_names
        
        if only_calib:
            logger.warning(f"Cameras in calibration but no keypoints: {only_calib}")
        if only_keypoints:
            logger.warning(f"Keypoints without calibration: {only_keypoints}")
        
        # Check frame counts match
        frame_counts = {name: data.shape[0] for name, data in self.keypoints_2d.items()}
        if len(set(frame_counts.values())) > 1:
            logger.warning(f"Frame counts differ: {frame_counts}")
    
    def triangulate_point_dlt(
        self,
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
        confidences: np.ndarray,
        kp_confidence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Triangulate a single 3D point using Direct Linear Transform (DLT).
        
        Args:
            points_2d: Array of shape (n_cameras, 2) with 2D pixel coordinates
            cameras: List of CameraCalibration objects
            confidences: Array of shape (n_cameras,) with confidence scores
            kp_confidence_threshold: Override confidence threshold for this
                specific keypoint (used for face/hand landmark groups).
                Falls back to self.confidence_threshold if not given.
            
        Returns:
            Tuple of (point_3d, mean_confidence)
            - point_3d: 3D point coordinates (3,)
            - mean_confidence: Average confidence across cameras used
        """
        threshold = (kp_confidence_threshold 
                     if kp_confidence_threshold is not None 
                     else self.confidence_threshold)
        
        # Filter by confidence
        mask = confidences >= threshold
        valid_points = points_2d[mask]
        valid_cameras = [cam for cam, m in zip(cameras, mask) if m]
        valid_confidences = confidences[mask]
        
        if len(valid_cameras) < self.min_cameras:
            return np.array([np.nan, np.nan, np.nan]), 0.0
        
        # ── Initial DLT solve ──
        point_3d = self._solve_dlt(valid_points, valid_cameras)
        
        # ── Iterative outlier rejection ──
        # If enabled, check reprojection error per camera and discard outliers,
        # then re-triangulate.  This is what fixes the case where one camera
        # has a hallucinated face landmark — its reprojection error will be
        # huge and it gets excluded.
        if self.use_iterative_rejection and len(valid_cameras) > self.min_cameras:
            for _iteration in range(2):  # at most 2 rejection passes
                errors = np.array([
                    np.linalg.norm(
                        pt - cam.project_points(point_3d.reshape(1, 3))[0]
                    )
                    for pt, cam in zip(valid_points, valid_cameras)
                ])
                
                inlier_mask = errors < self.reprojection_error_threshold
                n_inliers = inlier_mask.sum()
                
                if n_inliers < self.min_cameras:
                    break  # keep all — can't discard more
                if n_inliers == len(valid_cameras):
                    break  # all cameras are good
                
                # Re-triangulate with inliers only
                valid_points = valid_points[inlier_mask]
                valid_cameras = [c for c, m in zip(valid_cameras, inlier_mask) if m]
                valid_confidences = valid_confidences[inlier_mask]
                point_3d = self._solve_dlt(valid_points, valid_cameras)
        
        mean_confidence = np.mean(valid_confidences)
        return point_3d, mean_confidence
    
    @staticmethod
    def _solve_dlt(
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
    ) -> np.ndarray:
        """
        Core DLT solve: build the A matrix and solve via SVD.
        
        Args:
            points_2d: (n_cameras, 2) valid 2D observations
            cameras: corresponding CameraCalibration objects
            
        Returns:
            3D point in Cartesian coordinates (3,)
        """
        A = []
        for point, camera in zip(points_2d, cameras):
            x, y = point
            P = camera.projection_matrix
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        point_3d_hom = Vt[-1, :]
        point_3d = point_3d_hom[:3] / point_3d_hom[3]
        return point_3d
    
    def compute_reprojection_error(
        self,
        point_3d: np.ndarray,
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        Compute reprojection error for a 3D point.
        
        Args:
            point_3d: 3D point coordinates (3,)
            points_2d: Observed 2D points (n_cameras, 2)
            cameras: List of CameraCalibration objects
            confidences: Array of shape (n_cameras,)
            
        Returns:
            Array of reprojection errors per camera (n_cameras,)
        """
        if np.any(np.isnan(point_3d)):
            return np.full(len(cameras), np.nan)
        
        errors = []
        
        for i, camera in enumerate(cameras):
            if confidences[i] < self.confidence_threshold:
                errors.append(np.nan)
                continue
            
            # Project 3D point to 2D
            point_2d_proj = camera.project_points(point_3d.reshape(1, 3))
            
            # Compute Euclidean distance
            error = np.linalg.norm(points_2d[i] - point_2d_proj[0])
            errors.append(error)
        
        return np.array(errors)
    
    def triangulate_frame(
        self,
        frame_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Triangulate all keypoints for a single frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of (points_3d, confidences, reprojection_errors)
            - points_3d: Array of shape (n_keypoints, 3)
            - confidences: Array of shape (n_keypoints,)
            - reprojection_errors: Array of shape (n_keypoints, n_cameras)
        """
        # Get matching cameras and keypoints
        camera_names = list(self.cameras.keys() & self.keypoints_2d.keys())
        cameras = [self.cameras[name] for name in camera_names]
        
        # Get keypoints for this frame from all cameras
        frame_data = []
        for name in camera_names:
            keypoints = self.keypoints_2d[name]
            if frame_idx < keypoints.shape[0]:
                frame_data.append(keypoints[frame_idx])
            else:
                # Frame doesn't exist in this camera - use zeros
                n_keypoints = keypoints.shape[1]
                frame_data.append(np.zeros((n_keypoints, 3)))
        
        frame_data = np.array(frame_data)  # Shape: (n_cameras, n_keypoints, 3)
        n_keypoints = frame_data.shape[1]
        n_cameras = len(cameras)
        
        # Storage
        points_3d = np.zeros((n_keypoints, 3))
        confidences_3d = np.zeros(n_keypoints)
        reprojection_errors = np.zeros((n_keypoints, n_cameras))
        
        # Triangulate each keypoint
        for kp_idx in range(n_keypoints):
            # Extract 2D points and confidences for this keypoint across all cameras
            points_2d = frame_data[:, kp_idx, :2]  # (n_cameras, 2)
            confidences_2d = frame_data[:, kp_idx, 2]  # (n_cameras,)
            
            # Use the per-landmark-group threshold (stricter for face/hands)
            kp_threshold = (
                self._per_kp_threshold[kp_idx]
                if kp_idx < len(self._per_kp_threshold)
                else self.confidence_threshold
            )
            
            # Triangulate
            point_3d, mean_conf = self.triangulate_point_dlt(
                points_2d, cameras, confidences_2d,
                kp_confidence_threshold=kp_threshold,
            )
            
            # Compute reprojection error
            errors = self.compute_reprojection_error(
                point_3d, points_2d, cameras, confidences_2d
            )
            
            points_3d[kp_idx] = point_3d
            confidences_3d[kp_idx] = mean_conf
            reprojection_errors[kp_idx] = errors
        
        return points_3d, confidences_3d, reprojection_errors
    
    def triangulate_all_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> Tuple[np.ndarray, Projection3DMetrics]:
        """
        Triangulate all frames.
        
        Args:
            start_frame: First frame to process
            end_frame: Last frame to process (None = all frames)
            
        Returns:
            Tuple of (points_3d_all, metrics)
            - points_3d_all: Array of shape (n_frames, n_keypoints, 3)
            - metrics: Projection3DMetrics object
        """
        # Determine number of frames
        frame_counts = [data.shape[0] for data in self.keypoints_2d.values()]
        n_frames = min(frame_counts)
        
        if end_frame is None:
            end_frame = n_frames
        
        end_frame = min(end_frame, n_frames)
        n_frames_to_process = end_frame - start_frame
        
        logger.info(f"Triangulating frames {start_frame} to {end_frame}")
        
        # Get number of keypoints and cameras
        sample_data = list(self.keypoints_2d.values())[0]
        n_keypoints = sample_data.shape[1]
        n_cameras = len(self.cameras)
        
        # Storage
        all_points_3d = []
        all_confidences = []
        all_reprojection_errors = []
        
        # Process each frame
        from tqdm import tqdm
        for frame_idx in tqdm(range(start_frame, end_frame), desc="Triangulating"):
            points_3d, confidences, errors = self.triangulate_frame(frame_idx)
            
            all_points_3d.append(points_3d)
            all_confidences.append(confidences)
            all_reprojection_errors.append(errors)
        
        # Convert to arrays
        all_points_3d = np.array(all_points_3d)  # (n_frames, n_keypoints, 3)
        all_confidences = np.array(all_confidences)  # (n_frames, n_keypoints)
        all_reprojection_errors = np.array(all_reprojection_errors)  # (n_frames, n_keypoints, n_cameras)
        
        # Compute metrics
        valid_errors = all_reprojection_errors[~np.isnan(all_reprojection_errors)]
        
        # Count frames with good reconstruction
        # A frame has good reconstruction if mean reprojection error < threshold
        mean_errors_per_frame = np.nanmean(all_reprojection_errors, axis=(1, 2))
        good_threshold = 10.0  # pixels
        frames_with_good = np.sum(mean_errors_per_frame < good_threshold)
        
        metrics = Projection3DMetrics(
            n_frames=n_frames_to_process,
            n_keypoints=n_keypoints,
            n_cameras=n_cameras,
            reprojection_errors=all_reprojection_errors,
            mean_reprojection_error=np.mean(valid_errors),
            median_reprojection_error=np.median(valid_errors),
            std_reprojection_error=np.std(valid_errors),
            max_reprojection_error=np.max(valid_errors),
            reconstruction_confidence=all_confidences,
            frames_with_good_reconstruction=frames_with_good,
            good_reconstruction_threshold=good_threshold
        )
        
        logger.info("Triangulation complete!")
        
        return all_points_3d, metrics
    
    @staticmethod
    def nan_filter_by_reprojection_error(
        points_3d: np.ndarray,
        metrics: 'Projection3DMetrics',
        error_threshold: float = 10.0,
        confidence_threshold: float = 0.0,
        per_keypoint: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Post-processing: replace 3D points with NaN where quality is poor.
        
        This is intended to be called after triangulate_all_frames() to clean
        up the output.  Points that were triangulated but have high reprojection
        error (or low confidence) are replaced with NaN so that downstream
        smoothing / interpolation can fill them in properly rather than trying
        to work with garbage values.
        
        Args:
            points_3d: Array of shape (n_frames, n_keypoints, 3) from
                       triangulate_all_frames().
            metrics: The Projection3DMetrics returned alongside points_3d.
                     Must contain .reprojection_errors (n_frames, n_keypoints, n_cameras)
                     and .reconstruction_confidence (n_frames, n_keypoints).
            error_threshold: Mean reprojection error (px) above which a
                             keypoint in a frame is replaced with NaN.
                             10 px is a reasonable starting point; lower = stricter.
            confidence_threshold: Minimum mean 2D confidence below which a
                                  keypoint is NaN'd.  0.0 disables this check
                                  (useful if you already filtered during triangulation).
            per_keypoint: If True (default), evaluate each keypoint independently.
                          If False, NaN the entire frame if the *frame-level*
                          mean error exceeds the threshold.
            verbose: If True, log a summary of how many points were removed.
        
        Returns:
            Cleaned copy of points_3d with the same shape.  Bad points are
            set to [NaN, NaN, NaN].
            
        Example:
            >>> points_3d, metrics = projector.triangulate_all_frames()
            >>> # Remove any keypoint whose mean reproj error > 8 px
            >>> points_3d_clean = Pose3DProjector.nan_filter_by_reprojection_error(
            ...     points_3d, metrics, error_threshold=8.0
            ... )
            >>> # Then pass to Butterworth smoothing which will interpolate the gaps
        """
        cleaned = points_3d.copy()
        n_frames, n_keypoints, _ = cleaned.shape
        
        # Mean reprojection error per (frame, keypoint), ignoring NaN cameras
        # Shape: (n_frames, n_keypoints)
        mean_reproj = np.nanmean(metrics.reprojection_errors, axis=2)
        
        n_before = np.sum(~np.isnan(cleaned[:, :, 0]))
        
        if per_keypoint:
            # ── Per-keypoint filtering ──
            bad_error = mean_reproj > error_threshold
            
            bad_conf = np.zeros_like(bad_error)
            if confidence_threshold > 0.0:
                bad_conf = metrics.reconstruction_confidence < confidence_threshold
            
            bad_mask = bad_error | bad_conf  # (n_frames, n_keypoints)
            
            # Apply: set all 3 coords to NaN where bad
            cleaned[bad_mask] = np.nan
        else:
            # ── Per-frame filtering ──
            frame_mean_error = np.nanmean(mean_reproj, axis=1)  # (n_frames,)
            bad_frames = frame_mean_error > error_threshold
            
            if confidence_threshold > 0.0:
                frame_mean_conf = np.mean(
                    metrics.reconstruction_confidence, axis=1
                )
                bad_frames |= frame_mean_conf < confidence_threshold
            
            cleaned[bad_frames] = np.nan
        
        n_after = np.sum(~np.isnan(cleaned[:, :, 0]))
        n_removed = n_before - n_after
        
        if verbose:
            total = n_frames * n_keypoints
            logger.info(
                f"nan_filter_by_reprojection_error: "
                f"removed {n_removed}/{total} points "
                f"({n_removed / total * 100:.1f}%) "
                f"with error_threshold={error_threshold}px"
            )
            
            # Per-landmark-group breakdown
            if per_keypoint:
                for group_name, indices in [
                    ("face (0-10)", Pose3DProjector.FACE_LANDMARK_INDICES),
                    ("hands (17-22)", Pose3DProjector.HAND_LANDMARK_INDICES),
                    ("body", Pose3DProjector.BODY_LANDMARK_INDICES),
                ]:
                    idx = [i for i in indices if i < n_keypoints]
                    if not idx:
                        continue
                    orig = np.sum(~np.isnan(points_3d[:, idx, 0]))
                    kept = np.sum(~np.isnan(cleaned[:, idx, 0]))
                    removed_group = orig - kept
                    logger.info(
                        f"  {group_name}: removed {removed_group}/{orig} "
                        f"({removed_group / orig * 100:.1f}%)"
                    )
        
        return cleaned

    def save_3d_data(
        self,
        points_3d: np.ndarray,
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save 3D points and metadata.
        
        Args:
            points_3d: Array of shape (n_frames, n_keypoints, 3)
            output_path: Path for output file (without extension)
            metadata: Additional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as .npy
        npy_path = output_path.with_suffix('.npy')
        np.save(npy_path, points_3d)
        logger.info(f"Saved 3D points to {npy_path}")
        
        # Save metadata
        meta = {
            'shape': points_3d.shape,
            'n_frames': int(points_3d.shape[0]),
            'n_keypoints': int(points_3d.shape[1]),
            'n_cameras_used': len(self.cameras),
            'camera_names': list(self.cameras.keys()),
            'coordinate_system': '3D world coordinates (mm)',
            'data_format': 'n_frames x n_keypoints x 3 (x, y, z)'
        }
        
        if metadata:
            meta.update(metadata)
        
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved metadata to {json_path}")
    
    def save_metrics(self, metrics: Projection3DMetrics, output_path: str):
        """
        Save metrics to text file.
        
        Args:
            metrics: Projection3DMetrics object
            output_path: Path for output file
        """
        with open(output_path, 'w') as f:
            f.write(str(metrics))
        
        logger.info(f"Saved metrics to {output_path}")


# Example usage
if __name__ == "__main__":
    # Example configuration
    CALIBRATION_FILE = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/camera_calibration.toml"
    KEYPOINTS_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/mediapipe_output/data_2d"
    OUTPUT_DIR = "/Users/robertseymour/Documents/recordings/2026-02-06_14-54-43/data_3d"
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create projector
    projector = Pose3DProjector(
        calibration_path=CALIBRATION_FILE,
        keypoints_dir=KEYPOINTS_DIR,
        min_cameras_for_triangulation=2,
        confidence_threshold=0.3,          # body landmarks
        face_confidence_threshold=0.5,     # stricter for face (often hallucinated)
        hand_confidence_threshold=0.4,     # stricter for hands
        reprojection_error_threshold=15.0, # px — reject outlier camera views
        use_iterative_rejection=True,      # re-triangulate after dropping outliers
    )
    
    # Triangulate all frames
    print("\nTriangulating 2D keypoints to 3D...")
    points_3d, metrics = projector.triangulate_all_frames()
    
    # Post-process: NaN out points with high reprojection error.
    # This is where you clean up the face/hand hallucinations.
    # The Butterworth smoothing in pose_processor.py can then interpolate
    # short NaN gaps automatically.
    print("\nFiltering bad points by reprojection error...")
    points_3d_clean = Pose3DProjector.nan_filter_by_reprojection_error(
        points_3d,
        metrics,
        error_threshold=10.0,   # px — adjust to taste (lower = stricter)
        confidence_threshold=0.0,  # already filtered during triangulation
    )
    
    # Save results
    projector.save_3d_data(
        points_3d_clean,
        output_path=f"{OUTPUT_DIR}/pose_3d",
        metadata={'processing_date': '2026-02-06'}
    )
    
    projector.save_metrics(
        metrics,
        output_path=f"{OUTPUT_DIR}/projection_metrics.txt"
    )
    
    # Print results
    print("\n" + "="*70)
    print("3D PROJECTION COMPLETE!")
    print("="*70)
    print(metrics)
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"  - 3D points: {OUTPUT_DIR}/pose_3d.npy")
    print(f"  - Metadata: {OUTPUT_DIR}/pose_3d.json")
    print(f"  - Metrics: {OUTPUT_DIR}/projection_metrics.txt")
