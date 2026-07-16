"""Triangulation of multi-view 2D keypoints into 3D via the Direct Linear
Transform (DLT).

The projector loads per-camera intrinsics/extrinsics from a TOML calibration
file and per-camera 2D keypoint arrays, then reconstructs 3D coordinates frame
by frame.

Two combination schemes are available at the triangulation step:

``soft_weighting=True`` (default)
    Every camera contributes to every point; each view's DLT equations are
    scaled by a sigmoid of its 2D confidence. View influence therefore varies
    continuously with confidence and the contributing set does not change from
    frame to frame. This avoids the single-frame 3D discontinuities produced by
    a hard confidence gate, which arise when a view crosses the threshold and is
    added to or removed between consecutive frames.

``soft_weighting=False``
    Classical hard gate: views with confidence below the threshold are dropped
    and the remaining views are triangulated with equal weight.

In both schemes an optional reprojection-error step reduces the influence of
gross outliers (e.g. a single mis-detected view). Under soft weighting the
outlier is down-weighted (Gaussian in reprojection error) rather than removed,
so the contributing set remains fixed; under the hard gate it is discarded and
the point is re-triangulated.

Coordinates are in the units of the calibration extrinsics (mm).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from camkit3d import skeletons as _skeletons
from camkit3d.skeletons import PoseDefinition

logger = logging.getLogger(__name__)


@dataclass
class Projection3DMetrics:
    """Reprojection-error and reconstruction summary for a triangulated take.

    Attributes
    ----------
    n_frames, n_keypoints, n_cameras : int
        Dimensions of the reconstruction.
    reprojection_errors : ndarray, shape (n_frames, n_keypoints, n_cameras)
        Per-view reprojection error in pixels; NaN where a view did not
        contribute.
    mean_reprojection_error, median_reprojection_error, std_reprojection_error,
    max_reprojection_error : float
        Summary statistics over all contributing views.
    reconstruction_confidence : ndarray, shape (n_frames, n_keypoints)
        Mean 2D confidence of the views contributing to each point.
    frames_with_good_reconstruction : int
        Frames whose mean reprojection error is below
        ``good_reconstruction_threshold``.
    good_reconstruction_threshold : float
        Threshold (pixels) used for the count above.
    """

    n_frames: int
    n_keypoints: int
    n_cameras: int
    reprojection_errors: np.ndarray
    mean_reprojection_error: float
    median_reprojection_error: float
    std_reprojection_error: float
    max_reprojection_error: float
    reconstruction_confidence: np.ndarray
    frames_with_good_reconstruction: int
    good_reconstruction_threshold: float = 10.0

    def __str__(self) -> str:
        good_rate = (
            self.frames_with_good_reconstruction / self.n_frames * 100
            if self.n_frames > 0 else 0.0
        )
        return (
            f"3D projection metrics\n"
            f"  frames:                {self.n_frames}\n"
            f"  keypoints per frame:   {self.n_keypoints}\n"
            f"  cameras:               {self.n_cameras}\n"
            f"  reprojection error (px): "
            f"mean {self.mean_reprojection_error:.3f}, "
            f"median {self.median_reprojection_error:.3f}, "
            f"std {self.std_reprojection_error:.3f}, "
            f"max {self.max_reprojection_error:.3f}\n"
            f"  good frames (<{self.good_reconstruction_threshold:.0f} px): "
            f"{self.frames_with_good_reconstruction} ({good_rate:.1f}%)\n"
        )


class CameraCalibration:
    """Intrinsic and extrinsic parameters for a single camera.

    Parameters
    ----------
    name : str
        Camera identifier.
    size : (int, int)
        Image (width, height).
    matrix : array_like, shape (3, 3)
        Intrinsic matrix K.
    distortions : array_like
        OpenCV distortion coefficients.
    rotation : array_like, shape (3,)
        Rodrigues rotation vector (world -> camera).
    translation : array_like, shape (3,)
        Translation vector (world -> camera).
    world_orientation : array_like, shape (3, 3), optional
        Camera orientation in world coordinates (stored, not required for
        projection).
    world_position : array_like, shape (3,), optional
        Camera position in world coordinates (stored, not required for
        projection).
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
        world_position: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.size = size
        self.matrix = np.asarray(matrix, dtype=float)
        self.distortions = np.asarray(distortions, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.translation = np.asarray(translation, dtype=float)
        self.rotation_matrix, _ = cv2.Rodrigues(self.rotation)
        self.world_orientation = (
            np.asarray(world_orientation, dtype=float)
            if world_orientation is not None else None
        )
        self.world_position = (
            np.asarray(world_position, dtype=float)
            if world_position is not None else None
        )
        self.projection_matrix = self._compute_projection_matrix()

    def _compute_projection_matrix(self) -> np.ndarray:
        """Return the 3x4 projection matrix ``P = K [R | t]``."""
        extrinsic = np.hstack([self.rotation_matrix, self.translation.reshape(3, 1)])
        return self.matrix @ extrinsic

    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to pixel coordinates.

        Parameters
        ----------
        points_3d : ndarray, shape (N, 3) or (N, 4)
            Cartesian or homogeneous 3D points.

        Returns
        -------
        ndarray, shape (N, 2)
            Pixel coordinates.
        """
        if points_3d.shape[1] == 3:
            points_3d = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        projected = (self.projection_matrix @ points_3d.T).T
        return projected[:, :2] / projected[:, 2:3]

    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Undistort pixel coordinates using the stored distortion model.

        Parameters
        ----------
        points_2d : ndarray, shape (N, 2)
            Distorted pixel coordinates.

        Returns
        -------
        ndarray, shape (N, 2)
            Undistorted pixel coordinates, re-projected with K.
        """
        reshaped = points_2d.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(
            reshaped, self.matrix, self.distortions, P=self.matrix
        )
        return undistorted.reshape(-1, 2)


class Pose3DProjector:
    """Triangulate multi-view 2D keypoints to 3D.

    Parameters
    ----------
    calibration_path : str
        Path to the TOML calibration file (sections ``cam_*``).
    keypoints_dir : str
        Directory of ``<camera>_keypoints.npy`` arrays, each of shape
        (n_frames, n_keypoints, 3) holding (x, y, confidence).
    min_cameras_for_triangulation : int, default 2
        Minimum contributing views required to triangulate a point; otherwise
        the point is NaN.
    confidence_threshold : float, default 0.6
        Body-landmark confidence threshold. Under the hard gate, views below
        this are dropped. Under soft weighting it is not used as a hard cutoff
        (see ``soft_center``).
    face_confidence_threshold, hand_confidence_threshold : float, optional
        Per-group thresholds; default to ``confidence_threshold``.
    reprojection_error_threshold : float, default 15.0
        Scale (pixels) of the reprojection-error step. Under the hard gate,
        views with error above this are rejected; under soft weighting it sets
        the Gaussian down-weighting scale. Set to ``np.inf`` to disable.
    use_iterative_rejection : bool, default True
        Enable the reprojection-error step after the initial solve.
    soft_weighting : bool, default True
        Use continuous confidence weighting (see module docstring) instead of a
        hard confidence gate.
    soft_center : float, default 0.5
        Confidence at which the sigmoid weight equals 0.5.
    soft_sharpness : float, default 8.0
        Sigmoid steepness. Larger values approach a hard gate; smaller values
        give a gentler fade.
    soft_min_weight : float, default 0.02
        Views whose weight falls below this are treated as non-contributing
        (numerical floor). The minimum-camera constraint still applies.
    skeleton : str or PoseDefinition, optional
        Pose topology; defaults to MediaPipe Pose. Determines landmark count and
        face/hand/body groupings.

    Attributes
    ----------
    cameras : dict of str -> CameraCalibration
    keypoints_2d : dict of str -> ndarray
    skeleton : PoseDefinition
    """

    _DEFAULT_SKELETON = _skeletons.load()  # mediapipe_pose
    FACE_LANDMARK_INDICES = list(_DEFAULT_SKELETON.face_indices)
    HAND_LANDMARK_INDICES = list(_DEFAULT_SKELETON.hand_indices)
    BODY_LANDMARK_INDICES = list(_DEFAULT_SKELETON.body_indices)

    def __init__(
        self,
        calibration_path: str,
        keypoints_dir: str,
        min_cameras_for_triangulation: int = 2,
        confidence_threshold: float = 0.6,
        face_confidence_threshold: Optional[float] = None,
        hand_confidence_threshold: Optional[float] = None,
        reprojection_error_threshold: float = 15.0,
        use_iterative_rejection: bool = True,
        soft_weighting: bool = True,
        soft_center: float = 0.5,
        soft_sharpness: float = 8.0,
        soft_min_weight: float = 0.02,
        skeleton=None,
    ):
        self.calibration_path = Path(calibration_path)
        self.keypoints_dir = Path(keypoints_dir)
        self.min_cameras = min_cameras_for_triangulation

        # Confidence thresholds. Face and hand default to the body value; pass
        # explicit values to gate those groups more or less strictly.
        self.confidence_threshold = confidence_threshold
        self.face_confidence_threshold = (
            face_confidence_threshold if face_confidence_threshold is not None
            else confidence_threshold
        )
        self.hand_confidence_threshold = (
            hand_confidence_threshold if hand_confidence_threshold is not None
            else confidence_threshold
        )

        # Reprojection-outlier and view-combination settings (see class docstring).
        self.reprojection_error_threshold = reprojection_error_threshold
        self.use_iterative_rejection = use_iterative_rejection
        self.soft_weighting = soft_weighting
        self.soft_center = soft_center
        self.soft_sharpness = soft_sharpness
        self.soft_min_weight = soft_min_weight

        # Resolve the skeleton (landmark count and face/hand/body index groups).
        if skeleton is None:
            self.skeleton = self._DEFAULT_SKELETON
        elif isinstance(skeleton, PoseDefinition):
            self.skeleton = skeleton
        else:
            self.skeleton = _skeletons.load(skeleton)
        self.FACE_LANDMARK_INDICES = list(self.skeleton.face_indices)
        self.HAND_LANDMARK_INDICES = list(self.skeleton.hand_indices)
        self.BODY_LANDMARK_INDICES = list(self.skeleton.body_indices)

        # Per-keypoint threshold array: body value everywhere, overridden on the
        # face and hand indices.
        n_kp = self.skeleton.num_landmarks
        self._per_kp_threshold = np.full(n_kp, self.confidence_threshold)
        self._per_kp_threshold[self.FACE_LANDMARK_INDICES] = self.face_confidence_threshold
        self._per_kp_threshold[self.HAND_LANDMARK_INDICES] = self.hand_confidence_threshold

        self.cameras = self._load_calibration()
        logger.info("Loaded calibration for %d cameras", len(self.cameras))

        self.keypoints_2d = self._load_keypoints_2d()
        logger.info("Loaded 2D keypoints from %d cameras", len(self.keypoints_2d))

        self._validate_data()

        # Fixed camera ordering shared by all frames. A set intersection is
        # unordered, so sort it once here to keep view indices stable.
        self._camera_names = sorted(self.cameras.keys() & self.keypoints_2d.keys())

    # ------------------------------------------------------------------ loading

    def _load_calibration(self) -> Dict[str, CameraCalibration]:
        """Parse the TOML calibration into ``CameraCalibration`` objects."""
        with open(self.calibration_path, "r") as fh:
            calib = _toml_load(fh)

        cameras = {}
        for key, value in calib.items():
            if not key.startswith("cam_"):
                continue
            cameras[value["name"]] = CameraCalibration(
                name=value["name"],
                size=tuple(value["size"]),
                matrix=value["matrix"],
                distortions=value["distortions"],
                rotation=value["rotation"],
                translation=value["translation"],
                world_orientation=value.get("world_orientation"),
                world_position=value.get("world_position"),
            )
        return cameras

    def _load_keypoints_2d(self) -> Dict[str, np.ndarray]:
        """Load ``<camera>_keypoints.npy`` arrays keyed by camera name."""
        keypoints = {}
        for path in self.keypoints_dir.glob("*_keypoints.npy"):
            if path.name.startswith("._"):  # macOS resource fork
                continue
            name = path.stem.replace("_keypoints", "")
            keypoints[name] = np.load(path, allow_pickle=True)
            logger.info("Loaded %s: shape %s", name, keypoints[name].shape)
        return keypoints

    def _validate_data(self) -> None:
        """Check camera/keypoint correspondence and frame-count consistency."""
        cam_names = set(self.cameras)
        kp_names = set(self.keypoints_2d)
        if not cam_names & kp_names:
            raise ValueError(
                "No camera names shared between calibration and keypoints."
            )
        if cam_names - kp_names:
            logger.warning("Calibrated cameras without keypoints: %s", cam_names - kp_names)
        if kp_names - cam_names:
            logger.warning("Keypoints without calibration: %s", kp_names - cam_names)

        frame_counts = {n: d.shape[0] for n, d in self.keypoints_2d.items()}
        if len(set(frame_counts.values())) > 1:
            logger.warning("Frame counts differ across cameras: %s", frame_counts)

    # ------------------------------------------------------------- triangulation

    def triangulate_point_dlt(
        self,
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
        confidences: np.ndarray,
        kp_confidence_threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Triangulate one point across views.

        Parameters
        ----------
        points_2d : ndarray, shape (n_cameras, 2)
            Per-view pixel coordinates.
        cameras : list of CameraCalibration
            Views, aligned with ``points_2d`` and ``confidences``.
        confidences : ndarray, shape (n_cameras,)
            Per-view 2D confidence.
        kp_confidence_threshold : float, optional
            Hard-gate threshold for this keypoint; defaults to
            ``confidence_threshold``. Ignored under soft weighting.

        Returns
        -------
        point_3d : ndarray, shape (3,)
            Triangulated point, or NaN if fewer than ``min_cameras`` views
            contribute.
        mean_confidence : float
            Mean confidence of contributing views.
        """
        threshold = (
            kp_confidence_threshold if kp_confidence_threshold is not None
            else self.confidence_threshold
        )

        if self.soft_weighting:
            return self._triangulate_soft(points_2d, cameras, confidences)
        return self._triangulate_gated(points_2d, cameras, confidences, threshold)

    def _triangulate_soft(
        self,
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
        confidences: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Confidence-weighted DLT with continuous view weights.

        Views are weighted by ``sigmoid(sharpness * (confidence - center))`` and
        all finite views with weight above ``soft_min_weight`` contribute.
        The reprojection-error step, if enabled, multiplies the weights by a
        Gaussian in reprojection error rather than removing views, so the
        contributing set is unchanged.
        """
        # Map each view's confidence to a weight in (0, 1) via a sigmoid centred
        # at soft_center with steepness soft_sharpness. High-confidence views
        # approach weight 1, low-confidence views approach 0, with a smooth ramp
        # in between (no hard cutoff).
        weights = 1.0 / (1.0 + np.exp(
            -self.soft_sharpness * (confidences - self.soft_center)
        ))

        # A view contributes if its 2D point is finite and its weight is above
        # the numerical floor. Points with too few contributing views are NaN.
        finite = np.isfinite(points_2d).all(axis=1)
        usable = finite & (weights > self.soft_min_weight)
        if usable.sum() < self.min_cameras:
            return np.full(3, np.nan), 0.0

        pts = points_2d[usable]
        cams = [c for c, m in zip(cameras, usable) if m]
        w = weights[usable]
        conf = confidences[usable]

        # Weighted DLT solve using the confidence weights.
        point_3d = self._solve_dlt(pts, cams, w)

        # Optional reprojection-error step. Rather than removing an outlier view
        # (which would change the contributing set), multiply its weight by a
        # Gaussian in its reprojection error, so a grossly wrong view is
        # suppressed while the set stays fixed. Re-solve once with the adjusted
        # weights if enough views remain above the floor.
        if self.use_iterative_rejection and len(cams) > self.min_cameras:
            errors = np.array([
                np.linalg.norm(pt - cam.project_points(point_3d.reshape(1, 3))[0])
                for pt, cam in zip(pts, cams)
            ])
            scale = max(self.reprojection_error_threshold, 1e-6)
            w_adj = w * np.exp(-(errors / scale) ** 2)
            if (w_adj > self.soft_min_weight).sum() >= self.min_cameras:
                point_3d = self._solve_dlt(pts, cams, w_adj)

        return point_3d, float(np.mean(conf))

    def _triangulate_gated(
        self,
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
        confidences: np.ndarray,
        threshold: float,
    ) -> Tuple[np.ndarray, float]:
        """Hard-gate DLT: drop sub-threshold views, then reject reprojection
        outliers by re-triangulating on the inlier set (up to two passes)."""
        # Drop any view below the confidence threshold, then triangulate the
        # survivors with equal weight.
        mask = confidences >= threshold
        pts = points_2d[mask]
        cams = [c for c, m in zip(cameras, mask) if m]
        conf = confidences[mask]
        if len(cams) < self.min_cameras:
            return np.full(3, np.nan), 0.0

        point_3d = self._solve_dlt(pts, cams)

        # Reject reprojection outliers: recompute per-view error, keep the
        # inliers, and re-triangulate. Repeat up to twice, stopping if dropping
        # more views would fall below min_cameras or if all views are inliers.
        if self.use_iterative_rejection and len(cams) > self.min_cameras:
            for _ in range(2):
                errors = np.array([
                    np.linalg.norm(pt - cam.project_points(point_3d.reshape(1, 3))[0])
                    for pt, cam in zip(pts, cams)
                ])
                inliers = errors < self.reprojection_error_threshold
                if inliers.sum() < self.min_cameras or inliers.sum() == len(cams):
                    break
                pts = pts[inliers]
                cams = [c for c, m in zip(cams, inliers) if m]
                conf = conf[inliers]
                point_3d = self._solve_dlt(pts, cams)

        return point_3d, float(np.mean(conf))

    @staticmethod
    def _solve_dlt(
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve the (optionally weighted) DLT system by SVD.

        Each view contributes two rows to the design matrix ``A``; with weights,
        row pair ``i`` is scaled by ``weights[i]`` (weighted least squares). The
        solution is the right singular vector of ``A`` for the smallest singular
        value, de-homogenised.

        Parameters
        ----------
        points_2d : ndarray, shape (n_cameras, 2)
        cameras : list of CameraCalibration
        weights : ndarray, shape (n_cameras,), optional
            Per-view weights; equal weighting if None.

        Returns
        -------
        ndarray, shape (3,)
        """
        # Build the DLT design matrix: each view gives two linear constraints on
        # the homogeneous 3D point, from x*(P row 2) - (P row 0) and
        # y*(P row 2) - (P row 1). With weights, both rows for view i are scaled
        # by weights[i], turning the solve into weighted least squares.
        rows = []
        for i, (point, camera) in enumerate(zip(points_2d, cameras)):
            x, y = point
            P = camera.projection_matrix
            w = 1.0 if weights is None else float(weights[i])
            rows.append(w * (x * P[2, :] - P[0, :]))
            rows.append(w * (y * P[2, :] - P[1, :]))

        # The solution is the right singular vector for the smallest singular
        # value; de-homogenise to Cartesian coordinates.
        _, _, vt = np.linalg.svd(np.asarray(rows))
        homogeneous = vt[-1, :]
        return homogeneous[:3] / homogeneous[3]

    def compute_reprojection_error(
        self,
        point_3d: np.ndarray,
        points_2d: np.ndarray,
        cameras: List[CameraCalibration],
        confidences: np.ndarray,
    ) -> np.ndarray:
        """Per-view reprojection error (pixels) for a triangulated point.

        Views with confidence below ``confidence_threshold`` return NaN, as do
        all views if ``point_3d`` is NaN.
        """
        if np.any(np.isnan(point_3d)):
            return np.full(len(cameras), np.nan)

        errors = np.full(len(cameras), np.nan)
        for i, camera in enumerate(cameras):
            if confidences[i] < self.confidence_threshold:
                continue
            projected = camera.project_points(point_3d.reshape(1, 3))[0]
            errors[i] = np.linalg.norm(points_2d[i] - projected)
        return errors

    def triangulate_frame(
        self, frame_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triangulate all keypoints in one frame.

        Returns
        -------
        points_3d : ndarray, shape (n_keypoints, 3)
        confidences : ndarray, shape (n_keypoints,)
        reprojection_errors : ndarray, shape (n_keypoints, n_cameras)
        """
        # Fixed camera order so view indices are consistent across frames.
        camera_names = self._camera_names
        cameras = [self.cameras[name] for name in camera_names]

        # Gather this frame's 2D keypoints from every camera. If a camera is
        # short this frame, pad with zeros (confidence 0 excludes it downstream).
        frame_data = []
        for name in camera_names:
            arr = self.keypoints_2d[name]
            if frame_idx < arr.shape[0]:
                frame_data.append(arr[frame_idx])
            else:
                frame_data.append(np.zeros((arr.shape[1], 3)))
        frame_data = np.asarray(frame_data)  # (n_cameras, n_keypoints, 3)

        n_keypoints = frame_data.shape[1]
        n_cameras = len(cameras)

        points_3d = np.zeros((n_keypoints, 3))
        confidences_3d = np.zeros(n_keypoints)
        reprojection_errors = np.zeros((n_keypoints, n_cameras))

        # Triangulate each keypoint independently, using its group's confidence
        # threshold (body/face/hand) and recording the per-view reprojection
        # error for the quality metrics.
        for kp in range(n_keypoints):
            points_2d = frame_data[:, kp, :2]
            confidences_2d = frame_data[:, kp, 2]
            threshold = (
                self._per_kp_threshold[kp]
                if kp < len(self._per_kp_threshold)
                else self.confidence_threshold
            )
            point_3d, mean_conf = self.triangulate_point_dlt(
                points_2d, cameras, confidences_2d,
                kp_confidence_threshold=threshold,
            )
            reprojection_errors[kp] = self.compute_reprojection_error(
                point_3d, points_2d, cameras, confidences_2d
            )
            points_3d[kp] = point_3d
            confidences_3d[kp] = mean_conf

        return points_3d, confidences_3d, reprojection_errors

    def triangulate_all_frames(
        self, start_frame: int = 0, end_frame: Optional[int] = None
    ) -> Tuple[np.ndarray, Projection3DMetrics]:
        """Triangulate a range of frames.

        Parameters
        ----------
        start_frame : int, default 0
        end_frame : int, optional
            Exclusive end; defaults to the shortest per-camera frame count.

        Returns
        -------
        points_3d : ndarray, shape (n_frames, n_keypoints, 3)
        metrics : Projection3DMetrics
        """
        # Process the shortest common frame range across cameras.
        n_frames = min(d.shape[0] for d in self.keypoints_2d.values())
        end_frame = n_frames if end_frame is None else min(end_frame, n_frames)
        logger.info("Triangulating frames %d to %d", start_frame, end_frame)

        # Progress bar if tqdm is available, otherwise a plain range.
        try:
            from tqdm import tqdm
            frame_iter = tqdm(range(start_frame, end_frame), desc="Triangulating")
        except ImportError:
            frame_iter = range(start_frame, end_frame)

        points, confidences, errors = [], [], []
        for frame_idx in frame_iter:
            p, c, e = self.triangulate_frame(frame_idx)
            points.append(p)
            confidences.append(c)
            errors.append(e)

        points = np.asarray(points)
        confidences = np.asarray(confidences)
        errors = np.asarray(errors)

        # Summary statistics over contributing views, plus a count of frames
        # whose mean reprojection error is acceptable.
        valid = errors[~np.isnan(errors)]
        mean_error_per_frame = np.nanmean(errors, axis=(1, 2))
        good_threshold = 10.0

        metrics = Projection3DMetrics(
            n_frames=end_frame - start_frame,
            n_keypoints=points.shape[1],
            n_cameras=len(self.cameras),
            reprojection_errors=errors,
            mean_reprojection_error=float(np.mean(valid)),
            median_reprojection_error=float(np.median(valid)),
            std_reprojection_error=float(np.std(valid)),
            max_reprojection_error=float(np.max(valid)),
            reconstruction_confidence=confidences,
            frames_with_good_reconstruction=int(
                np.sum(mean_error_per_frame < good_threshold)
            ),
            good_reconstruction_threshold=good_threshold,
        )
        logger.info("Triangulation complete")
        return points, metrics

    # ------------------------------------------------------------ postprocessing

    @staticmethod
    def nan_filter_by_reprojection_error(
        points_3d: np.ndarray,
        metrics: "Projection3DMetrics",
        error_threshold: float = 10.0,
        confidence_threshold: float = 0.0,
        per_keypoint: bool = True,
        verbose: bool = True,
        skeleton=None,
    ) -> np.ndarray:
        """Set poorly reconstructed points to NaN for downstream interpolation.

        Parameters
        ----------
        points_3d : ndarray, shape (n_frames, n_keypoints, 3)
            Output of ``triangulate_all_frames``.
        metrics : Projection3DMetrics
            Provides ``reprojection_errors`` and ``reconstruction_confidence``.
        error_threshold : float, default 10.0
            Mean per-view reprojection error (pixels) above which a point is
            removed.
        confidence_threshold : float, default 0.0
            Minimum reconstruction confidence; 0 disables the check.
        per_keypoint : bool, default True
            If True, evaluate each keypoint independently. If False, remove a
            whole frame when its mean error exceeds the threshold; a single bad
            landmark then has little effect on the frame mean, so per-keypoint
            is preferred for localised errors.
        verbose : bool, default True
            Log removal counts, with a per-group breakdown when per_keypoint.
        skeleton : str or PoseDefinition, optional
            Skeleton for the group breakdown; defaults to the class skeleton.

        Returns
        -------
        ndarray
            Copy of ``points_3d`` with removed points set to NaN.
        """
        cleaned = points_3d.copy()
        n_frames, n_keypoints, _ = cleaned.shape
        # Mean reprojection error per (frame, keypoint) over contributing views.
        mean_reproj = np.nanmean(metrics.reprojection_errors, axis=2)
        n_before = np.sum(~np.isnan(cleaned[:, :, 0]))

        if per_keypoint:
            # Remove individual bad points. Preferred: a single bad landmark is
            # caught even though it barely moves the frame mean.
            bad = mean_reproj > error_threshold
            if confidence_threshold > 0.0:
                bad |= metrics.reconstruction_confidence < confidence_threshold
            cleaned[bad] = np.nan
        else:
            # Remove whole frames whose mean error is too high.
            frame_error = np.nanmean(mean_reproj, axis=1)
            bad_frames = frame_error > error_threshold
            if confidence_threshold > 0.0:
                bad_frames |= np.mean(
                    metrics.reconstruction_confidence, axis=1
                ) < confidence_threshold
            cleaned[bad_frames] = np.nan

        if verbose:
            n_after = np.sum(~np.isnan(cleaned[:, :, 0]))
            n_removed = n_before - n_after
            total = n_frames * n_keypoints
            logger.info(
                "nan_filter_by_reprojection_error: removed %d/%d points (%.1f%%) "
                "at error_threshold=%g px",
                n_removed, total, n_removed / total * 100, error_threshold,
            )
            if per_keypoint:
                if skeleton is None:
                    groups = [
                        ("face", Pose3DProjector.FACE_LANDMARK_INDICES),
                        ("hands", Pose3DProjector.HAND_LANDMARK_INDICES),
                        ("body", Pose3DProjector.BODY_LANDMARK_INDICES),
                    ]
                else:
                    skel = (skeleton if isinstance(skeleton, PoseDefinition)
                            else _skeletons.load(skeleton))
                    groups = [
                        ("face", list(skel.face_indices)),
                        ("hands", list(skel.hand_indices)),
                        ("body", list(skel.body_indices)),
                    ]
                for group_name, indices in groups:
                    idx = [i for i in indices if i < n_keypoints]
                    if not idx:
                        continue
                    orig = np.sum(~np.isnan(points_3d[:, idx, 0]))
                    kept = np.sum(~np.isnan(cleaned[:, idx, 0]))
                    if orig:
                        logger.info(
                            "  %s: removed %d/%d (%.1f%%)",
                            group_name, orig - kept, orig, (orig - kept) / orig * 100,
                        )

        return cleaned

    # -------------------------------------------------------------------- output

    def save_3d_data(
        self,
        points_3d: np.ndarray,
        output_path: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save 3D points (``.npy``) and a JSON sidecar of metadata.

        Parameters
        ----------
        points_3d : ndarray, shape (n_frames, n_keypoints, 3)
        output_path : str
            Path stem; ``.npy`` and ``.json`` are written alongside.
        metadata : dict, optional
            Extra fields merged into the sidecar.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path.with_suffix(".npy"), points_3d)
        logger.info("Saved 3D points to %s", output_path.with_suffix(".npy"))

        meta = {
            "shape": list(points_3d.shape),
            "n_frames": int(points_3d.shape[0]),
            "n_keypoints": int(points_3d.shape[1]),
            "n_cameras_used": len(self.cameras),
            "camera_names": list(self.cameras.keys()),
            "skeleton_id": self.skeleton.skeleton_id,
            "coordinate_system": "3D world coordinates (mm)",
            "data_format": "n_frames x n_keypoints x 3 (x, y, z)",
        }
        if metadata:
            meta.update(metadata)
        with open(output_path.with_suffix(".json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Saved metadata to %s", output_path.with_suffix(".json"))

    def save_metrics(self, metrics: Projection3DMetrics, output_path: str) -> None:
        """Write the string form of ``metrics`` to ``output_path``."""
        with open(output_path, "w") as fh:
            fh.write(str(metrics))
        logger.info("Saved metrics to %s", output_path)


def _toml_load(fh):
    """Load TOML from a file handle, using tomllib (3.11+) or toml."""
    try:
        import tomllib
        return tomllib.loads(fh.read())
    except ImportError:
        import toml
        return toml.load(fh)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Triangulate 2D keypoints to 3D.")
    parser.add_argument("calibration", help="Path to camera_calibration.toml")
    parser.add_argument("keypoints_dir", help="Directory of *_keypoints.npy")
    parser.add_argument("output_dir", help="Directory for 3D outputs")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--reprojection-error-threshold", type=float, default=15.0)
    parser.add_argument("--no-soft-weighting", action="store_true",
                        help="Use the hard confidence gate instead of soft weighting.")
    parser.add_argument("--soft-center", type=float, default=0.5)
    parser.add_argument("--soft-sharpness", type=float, default=8.0)
    parser.add_argument("--nan-filter-threshold", type=float, default=10.0,
                        help="Post hoc reprojection-error threshold (px).")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    projector = Pose3DProjector(
        calibration_path=args.calibration,
        keypoints_dir=args.keypoints_dir,
        confidence_threshold=args.confidence_threshold,
        reprojection_error_threshold=args.reprojection_error_threshold,
        soft_weighting=not args.no_soft_weighting,
        soft_center=args.soft_center,
        soft_sharpness=args.soft_sharpness,
    )

    points_3d, metrics = projector.triangulate_all_frames()
    points_3d = Pose3DProjector.nan_filter_by_reprojection_error(
        points_3d, metrics,
        error_threshold=args.nan_filter_threshold,
        per_keypoint=True,
    )
    projector.save_3d_data(points_3d, str(Path(args.output_dir) / "pose_3d"))
    projector.save_metrics(metrics, str(Path(args.output_dir) / "projection_metrics.txt"))
    print(metrics)
