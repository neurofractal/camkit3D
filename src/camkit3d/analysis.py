"""
CamKit3D Analysis Module

Combined toolkit for analysing, visualising, and animating 3D pose data.

Functions
---------
Analysis & static plots:
  - plot_reprojection_errors      : reprojection-error timeseries + histogram
  - detect_person_orientation     : auto-detect up / forward / right vectors
  - get_optimal_camera_angles     : recommended camera angles per view
  - visualize_orientation         : two-panel orientation check plot
  - align_pose_to_standard_frame  : rotate data into anatomical frame
  - plot_aligned_skeleton         : static 3D skeleton viewer (inline, no Qt)

Animation (video export):
  - animate_3d_pose               : single-view animation with many options

Author: CamKit3D (FreeMoCap-compatible workflow)
Date: 2026-02-13
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - safe for scripts & notebooks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Patch
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    # Lightweight fallback when tqdm is not installed
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is not None:
            return iterable
        class _Dummy:
            def update(self, n=1): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Dummy()
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# Skeleton topology (loaded from the skeleton descriptor)
# ============================================================================
#
# The default skeleton is MediaPipe Pose. To analyse data from a different
# skeleton, call set_skeleton("<id>") once at the top of your script/notebook,
# or pass skeleton=<PoseDefinition> to the public functions that accept it.

from camkit3d import skeletons
from camkit3d.skeletons import PoseDefinition

_SKELETON: PoseDefinition = skeletons.load()  # mediapipe_pose by default


def set_skeleton(skeleton):
    """Set the module-default skeleton used by drawing/orientation helpers.

    Parameters
    ----------
    skeleton : str or PoseDefinition
        Skeleton id (filename stem, e.g. "mediapipe_pose") or an already
        loaded PoseDefinition.
    """
    global _SKELETON
    _SKELETON = (skeleton if isinstance(skeleton, PoseDefinition)
                 else skeletons.load(skeleton))
    return _SKELETON


def _resolve(skeleton):
    """Return a PoseDefinition from a str/PoseDefinition/None argument."""
    if skeleton is None:
        return _SKELETON
    if isinstance(skeleton, PoseDefinition):
        return skeleton
    return skeletons.load(skeleton)


def _connection_color(start, end, skeleton=None):
    """Hex colour for a skeleton edge, from the skeleton's group colours."""
    pose = _resolve(skeleton)
    for c in pose.connections:
        if (c.start, c.end) == (start, end) or (c.end, c.start) == (start, end):
            return pose.groups[c.group].color if c.group in pose.groups else "#888888"
    return "#888888"


def _hide_axes_3d(ax):
    """Hide all axis furniture on a 3D axes (shared helper)."""
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["tick"]["color"] = (1, 1, 1, 0)
        a._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.tick_params(axis="both", which="both", colors="white", length=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)


def _draw_skeleton_on_ax(ax, pose, keypoint_size=50, line_width=2.5, skeleton=None):
    """Draw skeleton connections + keypoints onto a 3D axes.

    `pose` is the (n_keypoints, 3) array of points for one frame; `skeleton`
    is the topology (defaults to the module skeleton).
    """
    skel = _resolve(skeleton)
    for c in skel.connections:
        sp, ep = pose[c.start], pose[c.end]
        if not (np.any(np.isnan(sp)) or np.any(np.isnan(ep))):
            color = skel.groups[c.group].color if c.group in skel.groups else "#888888"
            ax.plot3D([sp[0], ep[0]], [sp[1], ep[1]], [sp[2], ep[2]],
                      color=color, linewidth=line_width, alpha=0.8)
    vmask = ~np.isnan(pose).any(axis=1)
    if vmask.any():
        vp = pose[vmask]
        ax.scatter(vp[:, 0], vp[:, 1], vp[:, 2], c="#2C3E50", s=keypoint_size,
                   alpha=0.9, edgecolors="white", linewidths=1, depthshade=True)


def _draw_floor(ax, x_range, y_range, z_range):
    """Draw a semi-transparent floor grid at the bottom of z_range."""
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 10),
                          np.linspace(y_range[0], y_range[1], 10))
    ax.plot_surface(xx, yy, np.ones_like(xx) * z_range[0],
                    alpha=0.1, color="gray", linewidth=0, antialiased=True)


def _compute_bounds(points_3d, padding=200):
    """Compute 1st/99th percentile bounds + padding from (possibly NaN) data."""
    vf = points_3d[~np.isnan(points_3d)]
    x_range = [np.percentile(vf[::3], 1) - padding, np.percentile(vf[::3], 99) + padding]
    y_range = [np.percentile(vf[1::3], 1) - padding, np.percentile(vf[1::3], 99) + padding]
    z_range = [np.percentile(vf[2::3], 1) - padding, np.percentile(vf[2::3], 99) + padding]
    return x_range, y_range, z_range


# ============================================================================
# 1. REPROJECTION ERROR PLOTS
# ============================================================================

def plot_reprojection_errors(
    reprojection_errors,
    mean_reprojection_error=None,
    median_reprojection_error=None,
    good_reconstruction_threshold=None,
    output_path=None,
    figsize=(14, 8),
):
    """
    Plot reprojection errors over time (top) and as a histogram (bottom).

    Parameters
    ----------
    reprojection_errors : np.ndarray
        Shape (n_frames, n_keypoints, n_cameras) or (n_frames, n_keypoints).
    mean_reprojection_error : float, optional
        Overall mean (auto-computed if None).
    median_reprojection_error : float, optional
        Overall median (auto-computed if None).
    good_reconstruction_threshold : float, optional
        Threshold line in pixels (default 5.0).
    output_path : str or Path, optional
        Save figure here.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    errors = np.asarray(reprojection_errors, dtype=float)
    axes_to_mean = tuple(range(1, errors.ndim))
    mean_errors_per_frame = np.nanmean(errors, axis=axes_to_mean)

    flat_valid = errors[~np.isnan(errors)].ravel()
    if mean_reprojection_error is None:
        mean_reprojection_error = float(np.mean(flat_valid))
    if median_reprojection_error is None:
        median_reprojection_error = float(np.median(flat_valid))
    if good_reconstruction_threshold is None:
        good_reconstruction_threshold = 5.0

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # --- timeseries ---
    axes[0].plot(mean_errors_per_frame, linewidth=1, alpha=0.7, color="steelblue")
    axes[0].axhline(mean_reprojection_error, color="red", linestyle="--", linewidth=2,
                     label=f"Overall Mean: {mean_reprojection_error:.2f}px")
    axes[0].axhline(good_reconstruction_threshold, color="green", linestyle="--", linewidth=2,
                     label=f"Good Threshold: {good_reconstruction_threshold:.2f}px")
    axes[0].set_ylim([0, min(100, np.nanmax(mean_errors_per_frame) * 1.2)])
    axes[0].set_xlabel("Frame Number", fontsize=12)
    axes[0].set_ylabel("Mean Reprojection Error (pixels)", fontsize=12)
    axes[0].set_title("Reprojection Error Over Time", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # --- histogram ---
    axes[1].hist(flat_valid, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    axes[1].axvline(mean_reprojection_error, color="red", linestyle="--", linewidth=2,
                     label=f"Mean: {mean_reprojection_error:.2f}px")
    axes[1].axvline(median_reprojection_error, color="blue", linestyle="--", linewidth=2,
                     label=f"Median: {median_reprojection_error:.2f}px")
    axes[1].set_xlabel("Reprojection Error (pixels)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Reprojection Error Distribution", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Reprojection error plot saved to: {output_path}")
    return fig


# ============================================================================
# 2. AUTO-ORIENTATION DETECTION
# ============================================================================

def detect_person_orientation(points_3d, frame_range=None, skeleton=None):
    """
    Detect which way is "up" and which direction the person is facing.

    Parameters
    ----------
    points_3d : np.ndarray - (n_frames, n_keypoints, 3)
    frame_range : tuple, optional - (start, end) slice
    skeleton : str or PoseDefinition, optional
        Skeleton providing the anatomy anchors (nose, shoulders, hips,
        ankles). Defaults to the module skeleton. Raises if the skeleton
        lacks the required anchors.

    Returns
    -------
    dict with keys: up_vector, forward_vector, right_vector,
         rotation_matrix, ground_plane_z, up_axis, forward_axis
    """
    skel = _resolve(skeleton)
    required = ("nose", "left_shoulder", "right_shoulder",
                "left_hip", "right_hip", "left_ankle", "right_ankle")
    if not skel.has_anchors(*required):
        missing = [r for r in required if r not in skel.anatomy]
        raise KeyError(
            f"detect_person_orientation requires anatomy anchors {missing} "
            f"which skeleton '{skel.skeleton_id}' does not define."
        )
    NOSE = skel.anchor("nose")
    L_SH, R_SH = skel.anchor("left_shoulder"), skel.anchor("right_shoulder")
    L_HIP, R_HIP = skel.anchor("left_hip"), skel.anchor("right_hip")
    L_ANK, R_ANK = skel.anchor("left_ankle"), skel.anchor("right_ankle")

    if frame_range is None:
        n = points_3d.shape[0]
        start, end = n // 4, 3 * n // 4
    else:
        start, end = frame_range
    sample = points_3d[start:end]

    print("\n" + "=" * 70)
    print("DETECTING PERSON ORIENTATION")
    print("=" * 70)

    # --- UP (hips -> shoulders) ---
    up_vecs = []
    for f in sample:
        pts = [f[L_HIP], f[R_HIP], f[L_SH], f[R_SH]]
        if any(np.any(np.isnan(p)) for p in pts):
            continue
        mid_hip = (f[L_HIP] + f[R_HIP]) / 2
        mid_sh = (f[L_SH] + f[R_SH]) / 2
        v = mid_sh - mid_hip
        n_ = np.linalg.norm(v)
        if n_ > 0:
            up_vecs.append(v / n_)

    up_vector = np.mean(up_vecs, axis=0) if up_vecs else np.array([0.0, 0, 1])
    up_vector /= np.linalg.norm(up_vector)
    print(f"\n✓ UP direction detected: [{up_vector[0]:.3f}, {up_vector[1]:.3f}, {up_vector[2]:.3f}]")
    print(f"  Dominant axis: {'X' if abs(up_vector[0]) > 0.7 else 'Y' if abs(up_vector[1]) > 0.7 else 'Z'}")

    # --- FORWARD (cross of shoulder line x up, toward nose) ---
    fwd_vecs = []
    for f in sample:
        pts = [f[L_SH], f[R_SH], f[NOSE]]
        if any(np.any(np.isnan(p)) for p in pts):
            continue
        sh_vec = f[L_SH] - f[R_SH]
        mid_sh = (f[L_SH] + f[R_SH]) / 2
        nose_dir = f[NOSE] - mid_sh
        fwd = np.cross(sh_vec, up_vector)
        if np.dot(fwd, nose_dir) < 0:
            fwd = -fwd
        n_ = np.linalg.norm(fwd)
        if n_ > 0:
            fwd_vecs.append(fwd / n_)

    forward_vector = np.mean(fwd_vecs, axis=0) if fwd_vecs else np.array([0.0, 1, 0])
    forward_vector /= np.linalg.norm(forward_vector)
    print(f"✓ FORWARD direction detected: [{forward_vector[0]:.3f}, {forward_vector[1]:.3f}, {forward_vector[2]:.3f}]")

    # --- RIGHT ---
    right_vector = np.cross(forward_vector, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    print(f"✓ RIGHT direction detected: [{right_vector[0]:.3f}, {right_vector[1]:.3f}, {right_vector[2]:.3f}]")

    # --- Ground plane ---
    ankle_h = []
    for f in sample:
        for idx in [L_ANK, R_ANK]:
            if not np.any(np.isnan(f[idx])):
                ankle_h.append(np.dot(f[idx], up_vector))
    ground_z = np.percentile(ankle_h, 5) if ankle_h else 0.0
    print(f"✓ Ground plane detected at: {ground_z:.2f} mm along up-axis")

    rotation_matrix = np.column_stack([right_vector, forward_vector, up_vector])
    print(f"\n✓ Orientation detection complete!")

    return {
        "up_vector": up_vector,
        "forward_vector": forward_vector,
        "right_vector": right_vector,
        "rotation_matrix": rotation_matrix,
        "ground_plane_z": ground_z,
        "up_axis": int(np.argmax(np.abs(up_vector))),
        "forward_axis": int(np.argmax(np.abs(forward_vector))),
    }


def get_optimal_camera_angles(orientation_info):
    """
    Get optimal camera angles based on detected orientation.

    Returns dict of views, each with elevation, azimuth, description.
    """
    forward_vec = orientation_info["forward_vector"]
    up_axis = np.argmax(np.abs(orientation_info["up_vector"]))

    azimuth_front = np.degrees(np.arctan2(forward_vec[1], forward_vec[0]))
    azimuth_side = azimuth_front + 90
    azimuth_back = azimuth_front + 180

    elevation_top = 90 if up_axis >= 1 else 0
    elevation_front = 0

    return {
        "front": {"elevation": elevation_front, "azimuth": azimuth_front,
                   "description": "Looking at person from front"},
        "back": {"elevation": elevation_front, "azimuth": azimuth_back,
                  "description": "Looking at person from back"},
        "left_side": {"elevation": elevation_front, "azimuth": azimuth_side - 90,
                       "description": "Looking at person from their left side"},
        "right_side": {"elevation": elevation_front, "azimuth": azimuth_side,
                        "description": "Looking at person from their right side"},
        "top": {"elevation": elevation_top, "azimuth": azimuth_front,
                 "description": "Looking down at person from above"},
        "diagonal": {"elevation": 30, "azimuth": azimuth_front + 45,
                      "description": "Diagonal view from front-right"},
    }


# ============================================================================
# 3. VISUALIZE ORIENTATION (two-panel diagnostic plot)
# ============================================================================

def visualize_orientation(points_3d, orientation, camera_angles, output_path=None,
                          frame_idx=None, skeleton=None):
    """
    Two-panel plot: LEFT = diagonal with orientation arrows, RIGHT = front view.

    Returns matplotlib.figure.Figure
    """
    skel = _resolve(skeleton)
    if frame_idx is None:
        frame_idx = points_3d.shape[0] // 2
    frame = points_3d[frame_idx]

    # Body centre (mean of hip-midpoint and shoulder-midpoint), with fallback
    body_center = None
    if skel.has_anchors("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        l_hip, r_hip = frame[skel.anchor("left_hip")], frame[skel.anchor("right_hip")]
        l_sh, r_sh = frame[skel.anchor("left_shoulder")], frame[skel.anchor("right_shoulder")]
        if not np.any(np.isnan([l_hip, r_hip, l_sh, r_sh])):
            body_center = ((l_hip + r_hip) / 2 + (l_sh + r_sh) / 2) / 2
    if body_center is None:
        valid_pts = frame[~np.isnan(frame).any(axis=1)]
        body_center = np.mean(valid_pts, axis=0)

    up_vec = orientation["up_vector"]
    forward_vec = orientation["forward_vector"]
    right_vec = orientation["right_vector"]
    scale = 500  # mm

    # Equal-axis limits
    valid_all = points_3d[~np.isnan(points_3d)]
    max_range = np.array([
        valid_all[::3].max() - valid_all[::3].min(),
        valid_all[1::3].max() - valid_all[1::3].min(),
        valid_all[2::3].max() - valid_all[2::3].min(),
    ]).max() / 2.0
    mid_x = (valid_all[::3].max() + valid_all[::3].min()) * 0.5
    mid_y = (valid_all[1::3].max() + valid_all[1::3].min()) * 0.5
    mid_z = (valid_all[2::3].max() + valid_all[2::3].min()) * 0.5

    # Simplified skeleton connections for this diagnostic plot: drop the
    # dense face mesh, keep torso + limbs.
    connections = [(c.start, c.end) for c in skel.connections if c.group != "face"]
    valid_mask = ~np.isnan(frame).any(axis=1)
    valid_pts = frame[valid_mask]

    fig = plt.figure(figsize=(18, 8))

    def _draw_skel(ax):
        if len(valid_pts) > 0:
            ax.scatter(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2],
                       c="lightgray", s=30, alpha=0.6)
        for si, ei in connections:
            s, e = frame[si], frame[ei]
            if not (np.any(np.isnan(s)) or np.any(np.isnan(e))):
                ax.plot3D([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                          "gray", linewidth=1, alpha=0.4)
        ax.scatter(*body_center, c="black", s=200, marker="o",
                   edgecolors="white", linewidths=2, zorder=10)

    def _draw_arrows(ax, show_opposites=True):
        for vec, label, col in [
            (up_vec, "UP", "green"), (forward_vec, "FORWARD", "red"), (right_vec, "RIGHT", "blue"),
        ]:
            ax.quiver(*body_center, *vec, length=scale, color=col,
                      arrow_length_ratio=0.15, linewidth=4, alpha=0.9)
            ax.text(*(body_center + vec * scale * 1.1), label,
                    fontsize=16, fontweight="bold", color=col)
            if show_opposites:
                opp_label = {"UP": "DOWN", "FORWARD": "BACK", "RIGHT": "LEFT"}[label]
                opp_col = {"UP": "lightgreen", "FORWARD": "pink", "RIGHT": "lightblue"}[label]
                ax.quiver(*body_center, *(-vec), length=scale * 0.7, color=opp_col,
                          arrow_length_ratio=0.15, linewidth=2, alpha=0.6)
                ax.text(*(body_center - vec * scale * 0.8), opp_label,
                        fontsize=12, color=opp_col, alpha=0.7)

    def _set_lims(ax):
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel("X (mm)", fontsize=10)
        ax.set_ylabel("Y (mm)", fontsize=10)
        ax.set_zlabel("Z (mm)", fontsize=10)
        ax.grid(True, alpha=0.3)

    # LEFT panel - diagonal
    ax1 = fig.add_subplot(121, projection="3d")
    _draw_skel(ax1); _draw_arrows(ax1, show_opposites=True); _set_lims(ax1)
    ax1.view_init(elev=20, azim=45)
    ax1.set_title("3D View: Detected Orientation\n(Diagonal Perspective)",
                  fontsize=14, fontweight="bold", pad=20)

    # RIGHT panel - front
    ax2 = fig.add_subplot(122, projection="3d")
    _draw_skel(ax2); _draw_arrows(ax2, show_opposites=False); _set_lims(ax2)
    fe = camera_angles["front"]["elevation"]
    fa = camera_angles["front"]["azimuth"]
    ax2.view_init(elev=fe, azim=fa)
    ax2.set_title(f"FRONT View\nElevation: {fe:.1f}°, Azimuth: {fa:.1f}°",
                  fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Orientation visualization saved to: {output_path}")
    return fig


# ============================================================================
# 4. ALIGN POSE TO STANDARD ANATOMICAL FRAME
# ============================================================================

def align_pose_to_standard_frame(points_3d, orientation=None):
    """
    Rotate 3D pose data into the standard anatomical frame:
        X = medial-lateral (right +),  Y = anterior-posterior (forward +),
        Z = superior-inferior (up +)

    Parameters
    ----------
    points_3d : np.ndarray - (n_frames, n_keypoints, 3) or (n_keypoints, 3)
    orientation : dict, optional - from detect_person_orientation(); auto-detected if None.

    Returns
    -------
    rotated : np.ndarray, rotation_matrix : (3,3), orientation : dict
    """
    if orientation is None:
        if points_3d.ndim == 3:
            orientation = detect_person_orientation(points_3d)
        else:
            raise ValueError("Cannot auto-detect orientation from a single frame. Supply orientation dict.")

    R = np.array([
        orientation["right_vector"],
        orientation["forward_vector"],
        orientation["up_vector"],
    ])

    if points_3d.ndim == 3:
        nan_mask = np.any(np.isnan(points_3d), axis=-1)
        rotated = np.einsum("ij,nkj->nki", R, points_3d)
        rotated[nan_mask] = np.nan
    else:
        nan_mask = np.any(np.isnan(points_3d), axis=-1)
        rotated = (R @ points_3d.T).T
        rotated[nan_mask] = np.nan

    up_new = R @ orientation["up_vector"]
    fwd_new = R @ orientation["forward_vector"]
    right_new = R @ orientation["right_vector"]
    print(f"Alignment check  UP->{np.round(up_new, 2)}  FWD->{np.round(fwd_new, 2)}  RIGHT->{np.round(right_new, 2)}")

    return rotated, R, orientation


# ============================================================================
# 5. PLOT ALIGNED SKELETON (inline-friendly, no Qt)
# ============================================================================

def plot_aligned_skeleton(
    points_3d_aligned,
    frame_idx=None,
    output_path=None,
    figsize=(16, 12),
    elev=0,
    azim=90,
    title_suffix="",
    skeleton=None,
):
    """
    Static 3D skeleton plot - works inline (Agg backend) or in Jupyter.

    Parameters
    ----------
    points_3d_aligned : np.ndarray - (n_frames, n_keypoints, 3), already aligned
    frame_idx : int, optional - default middle frame
    output_path : str or Path, optional
    elev, azim : float - viewing angles (default: front view)
    title_suffix : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    if frame_idx is None:
        frame_idx = points_3d_aligned.shape[0] // 2
    frame = points_3d_aligned[frame_idx]
    skel = _resolve(skeleton)

    valid_mask = ~np.isnan(frame).any(axis=1)
    valid_pts = frame[valid_mask]
    if len(valid_pts) == 0:
        raise ValueError(f"No valid keypoints in frame {frame_idx}")

    mins = valid_pts.min(axis=0)
    maxs = valid_pts.max(axis=0)
    ranges = maxs - mins
    pad = 0.2

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Skeleton
    for c in skel.connections:
        sp, ep = frame[c.start], frame[c.end]
        if not (np.any(np.isnan(sp)) or np.any(np.isnan(ep))):
            color = skel.groups[c.group].color if c.group in skel.groups else "#888888"
            ax.plot3D([sp[0], ep[0]], [sp[1], ep[1]], [sp[2], ep[2]],
                      color=color, linewidth=3, alpha=0.8)

    # Keypoints
    ax.scatter(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2],
               c="#2C3E50", s=100, alpha=0.9, edgecolors="white",
               linewidths=2, depthshade=True)

    # Body-centre axes
    center = None
    if skel.has_anchors("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        mid_hip = (frame[skel.anchor("left_hip")] + frame[skel.anchor("right_hip")]) / 2
        mid_sh = (frame[skel.anchor("left_shoulder")] + frame[skel.anchor("right_shoulder")]) / 2
        if not np.any(np.isnan([mid_hip, mid_sh])):
            center = (mid_hip + mid_sh) / 2
    if center is None:
        center = np.nanmean(valid_pts, axis=0)

    axis_len = max(ranges) * 0.5
    for vec, label, colour in [
        ([1, 0, 0], "RIGHT (+X)", "red"),
        ([0, 1, 0], "FORWARD (+Y)", "green"),
        ([0, 0, 1], "UP (+Z)", "blue"),
    ]:
        ax.quiver(center[0], center[1], center[2],
                  vec[0] * axis_len, vec[1] * axis_len, vec[2] * axis_len,
                  color=colour, linewidth=4, arrow_length_ratio=0.15, alpha=0.9)
        ax.text(center[0] + vec[0] * axis_len * 1.25,
                center[1] + vec[1] * axis_len * 1.25,
                center[2] + vec[2] * axis_len * 1.25,
                label, fontsize=12, color=colour, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_xlabel("X (mm) - Medial/Lateral", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y (mm) - Anterior/Posterior", fontsize=11, fontweight="bold")
    ax.set_zlabel("Z (mm) - Superior/Inferior", fontsize=11, fontweight="bold")
    ax.set_title(f"Aligned 3D Pose  (frame {frame_idx})  elev={elev} azim={azim}{title_suffix}",
                 fontsize=13, fontweight="bold", pad=18)
    ax.set_xlim([mins[0] - ranges[0] * pad, maxs[0] + ranges[0] * pad])
    ax.set_ylim([mins[1] - ranges[1] * pad, maxs[1] + ranges[1] * pad])
    ax.set_zlim([mins[2] - ranges[2] * pad, maxs[2] + ranges[2] * pad])
    ax.set_box_aspect(ranges if ranges.min() > 0 else [1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor=g.color, label=name.replace("_", " ").title())
        for name, g in skel.groups.items()
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Skeleton plot saved to: {output_path}")
    return fig


# ============================================================================
# 6. ANIMATE 3D POSE (single view)
# ============================================================================

def animate_3d_pose(
    points_3d,
    output_path,
    fps=30,
    frames_to_animate=None,
    view_mode="rotating",
    elevation=20,
    azimuth_start=45,
    rotation_speed=0.5,
    rotation_axis="z",
    show_floor=True,
    show_frame_number=True,
    show_timestamp=True,
    show_axes=True,
    keypoint_size=50,
    line_width=2.5,
    figure_size=(12, 10),
    dpi=100,
    quality="high",
    skeleton=None,
):
    """
    Create an animated 3D visualization of pose over time.

    Parameters
    ----------
    points_3d : np.ndarray - (n_frames, n_keypoints, 3)
    output_path : str - e.g. "pose.mp4"
    fps : int
    frames_to_animate : int or None (all)
    view_mode : str - 'rotating', 'front', 'side', 'top', 'diagonal', 'custom'
    elevation, azimuth_start : float - camera angles
    rotation_speed : float - degrees/frame for rotating mode
    rotation_axis : str - 'z', 'x', 'y', 'both'
    show_floor, show_frame_number, show_timestamp, show_axes : bool
    keypoint_size, line_width : numeric
    figure_size : tuple
    dpi : int
    quality : str - 'low', 'medium', 'high', 'ultra'

    Returns
    -------
    str - path to saved video
    """
    print("\n" + "=" * 70)
    print("CREATING 3D POSE ANIMATION")
    print("=" * 70)

    if points_3d.ndim != 3 or points_3d.shape[2] != 3:
        raise ValueError(f"Expected shape (n_frames, n_keypoints, 3), got {points_3d.shape}")

    n_frames = points_3d.shape[0]
    if frames_to_animate is None:
        frames_to_animate = n_frames
    else:
        frames_to_animate = min(frames_to_animate, n_frames)

    quality_presets = {
        "low": {"dpi": 75, "bitrate": 1500},
        "medium": {"dpi": 100, "bitrate": 3000},
        "high": {"dpi": 150, "bitrate": 5000},
        "ultra": {"dpi": 200, "bitrate": 8000},
    }
    if quality in quality_presets:
        dpi = quality_presets[quality]["dpi"]
        bitrate = quality_presets[quality]["bitrate"]
    else:
        bitrate = 3000

    print(f"\nSettings:")
    print(f"  Frames: {frames_to_animate}/{n_frames}  FPS: {fps}  Duration: {frames_to_animate / fps:.2f}s")
    print(f"  View: {view_mode}  Quality: {quality} (DPI: {dpi})")
    if view_mode == "rotating":
        print(f"  Rotation axis: {rotation_axis.upper()}  Speed: {rotation_speed} deg/frame")
    print(f"  Output: {output_path}")

    x_range, y_range, z_range = _compute_bounds(points_3d)

    ranges = np.array([x_range[1] - x_range[0],
                   y_range[1] - y_range[0],
                   z_range[1] - z_range[0]])
    
    box_aspect = ranges if ranges.min() > 0 else [1, 1, 1]

    view_presets = {
        "front": {"elev": 0, "azim": 0},
        "side": {"elev": 0, "azim": 90},
        "top": {"elev": 90, "azim": 0},
        "diagonal": {"elev": 30, "azim": 45},
        "rotating": {"elev": elevation, "azim": azimuth_start},
        "custom": {"elev": elevation, "azim": azimuth_start},
    }
    initial_view = view_presets.get(view_mode, view_presets["rotating"])

    fig = plt.figure(figsize=figure_size, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    def init():
        ax.clear()
        ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_zlim(z_range)
        ax.set_box_aspect(box_aspect)   # <-- add this line in both places
        if show_axes:
            ax.set_xlabel("X (mm)", fontsize=10, labelpad=10)
            ax.set_ylabel("Y (mm)", fontsize=10, labelpad=10)
            ax.set_zlabel("Z (mm)", fontsize=10, labelpad=10)
        else:
            _hide_axes_3d(ax)
        ax.view_init(elev=initial_view["elev"], azim=initial_view["azim"])
        if show_floor:
            _draw_floor(ax, x_range, y_range, z_range)
        return []

    def update(frame_num):
        ax.clear()
        ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_zlim(z_range)
        ax.set_box_aspect(box_aspect)   # <-- add this line in both places

        if show_axes:
            ax.set_xlabel("X (mm)", fontsize=10, labelpad=10)
            ax.set_ylabel("Y (mm)", fontsize=10, labelpad=10)
            ax.set_zlabel("Z (mm)", fontsize=10, labelpad=10)
        else:
            _hide_axes_3d(ax)

        # View angle
        if view_mode == "rotating":
            ra = rotation_axis.lower()
            if ra == "z":
                cur_azim = azimuth_start + frame_num * rotation_speed
                cur_elev = elevation
            elif ra == "x":
                cur_azim = azimuth_start
                cur_elev = elevation + frame_num * rotation_speed
            elif ra == "y":
                cur_azim = azimuth_start + frame_num * rotation_speed
                cur_elev = elevation + frame_num * rotation_speed * 0.5
            elif ra == "both":
                cur_azim = azimuth_start + frame_num * rotation_speed
                cur_elev = elevation + 30 * math.sin(frame_num * rotation_speed * math.pi / 180)
            else:
                cur_azim = azimuth_start + frame_num * rotation_speed
                cur_elev = elevation
            ax.view_init(elev=cur_elev, azim=cur_azim)
        else:
            ax.view_init(elev=initial_view["elev"], azim=initial_view["azim"])

        if show_floor:
            _draw_floor(ax, x_range, y_range, z_range)

        _draw_skeleton_on_ax(ax, points_3d[frame_num],
                             keypoint_size=keypoint_size, line_width=line_width,
                             skeleton=skeleton)

        if show_frame_number or show_timestamp:
            parts = []
            if show_frame_number:
                parts.append(f"Frame: {frame_num}/{frames_to_animate - 1}")
            if show_timestamp:
                parts.append(f"Time: {frame_num / fps:.2f}s")
            ax.text2D(0.05, 0.95, "\n".join(parts), transform=ax.transAxes, fontsize=12,
                      verticalalignment="top",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_title("3D Pose Reconstruction", fontsize=14, fontweight="bold", pad=20)
        return []

    print("\nGenerating animation frames...")
    anim = FuncAnimation(fig, update, init_func=init, frames=frames_to_animate,
                         interval=1000 / fps, blit=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving video to {output_path}...")
    writer = FFMpegWriter(fps=fps, bitrate=bitrate, codec="libx264")

    with tqdm(total=frames_to_animate, desc="Rendering") as pbar:
        def progress_callback(current_frame, total_frames):
            pbar.update(1)
        anim.save(str(output_path), writer=writer, dpi=dpi, progress_callback=progress_callback)

    plt.close(fig)
    file_size_mb = output_path.stat().st_size / 1_000_000
    print(f"\n✓ Animation saved! {output_path} ({file_size_mb:.2f} MB, {frames_to_animate / fps:.2f}s)")
    return str(output_path)

# ============================================================================
# ANIMATE 3D POSE - MULTI-ANGLE (2x2 grid)
# ============================================================================

def animate_3d_pose_multiangle(
    points_3d_aligned,
    output_path,
    fps=30,
    frames_to_animate=None,
    show_floor=False,
    show_axes=False,
    keypoint_size=70,
    line_width=4,
    figure_size=(16, 16),
    dpi=100,
    quality="high",
    skeleton=None,
):
    """
    2x2 multi-angle animation (Front, Right, Top, Left). Data must be pre-aligned.

    Parameters
    ----------
    points_3d_aligned : np.ndarray - (n_frames, n_keypoints, 3), already aligned
    output_path : str
    fps, frames_to_animate, show_floor, show_axes : as animate_3d_pose
    keypoint_size, line_width, figure_size, dpi, quality : as animate_3d_pose

    Returns
    -------
    str - path to saved video
    """
    print("\n" + "=" * 70)
    print("CREATING MULTI-ANGLE 3D POSE ANIMATION")
    print("=" * 70)

    if points_3d_aligned.ndim != 3 or points_3d_aligned.shape[2] != 3:
        raise ValueError(f"Expected shape (n_frames, n_keypoints, 3), got {points_3d_aligned.shape}")

    n_frames = points_3d_aligned.shape[0]
    if frames_to_animate is None:
        frames_to_animate = n_frames
    else:
        frames_to_animate = min(frames_to_animate, n_frames)

    quality_presets = {"low": (75, 1500), "medium": (100, 3000), "high": (150, 5000), "ultra": (200, 8000)}
    dpi, bitrate = quality_presets.get(quality, (100, 3000))

    print(f"\n  Frames: {frames_to_animate}/{n_frames}  FPS: {fps}  Duration: {frames_to_animate / fps:.2f}s")
    print(f"  Quality: {quality} (DPI: {dpi})  Output: {output_path}")

    x_range, y_range, z_range = _compute_bounds(points_3d_aligned)

    views = {
        "Front": {"elev": 0, "azim": 90, "pos": 1},
        "Right": {"elev": 0, "azim": 0, "pos": 2},
        "Top": {"elev": 90, "azim": 0, "pos": 3},
        "Left": {"elev": 0, "azim": 180, "pos": 4},
    }

    fig = plt.figure(figsize=figure_size, facecolor="white")
    ax_map = {}
    for vn, vi in views.items():
        ax_map[vn] = {
            "ax": fig.add_subplot(2, 2, vi["pos"], projection="3d", facecolor="white"),
            "elev": vi["elev"],
            "azim": vi["azim"],
        }

    def _setup_subplot(ax, elev, azim, title):
        ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_zlim(z_range)
        if not show_axes:
            ax.set_axis_off(); ax.grid(False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        if show_floor:
            _draw_floor(ax, x_range, y_range, z_range)

    def init():
        for vn, ai in ax_map.items():
            ai["ax"].clear()
            _setup_subplot(ai["ax"], ai["elev"], ai["azim"], vn)
        return []

    def update(frame_num):
        pose = points_3d_aligned[frame_num]
        for vn, ai in ax_map.items():
            ax = ai["ax"]; ax.clear()
            _setup_subplot(ax, ai["elev"], ai["azim"], vn)
            _draw_skeleton_on_ax(ax, pose, keypoint_size=keypoint_size,
                                 line_width=line_width, skeleton=skeleton)
        fig.suptitle(f"Frame: {frame_num}/{frames_to_animate - 1}  |  Time: {frame_num / fps:.2f}s",
                     fontsize=16, fontweight="bold", y=0.98)
        return []

    print("\nGenerating animation frames...")
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=tqdm(range(frames_to_animate), desc="Animating"),
                         interval=1000 / fps, blit=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving video to {output_path}...")
    writer = FFMpegWriter(fps=fps, metadata={"artist": "pose3DAnalysis"},
                          bitrate=bitrate, codec="h264")
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Multi-angle animation saved! {output_path} ({file_size_mb:.2f} MB)")
    return str(output_path)

# ============================================================================
# 9. Interactive Pose Viewer
# ============================================================================

def interactive_pose_viewer(
    points_3d,
    fps=30,
    keypoint_size=50,
    line_width=2.5,
    figure_size=(14, 10),
    initial_view="front",
    show_floor=True,
    window_title="3D Pose Viewer",
    skeleton=None,
):
    """
    Interactive 3D pose viewer with play/pause, timeline scrubbing, and
    free mouse-drag rotation.

    Requires a GUI backend (TkAgg or Qt5Agg). In a script, call this
    *without* setting matplotlib.use('Agg') beforehand — it will pick the
    right backend automatically.  In Jupyter, run  %matplotlib tk  or
    %matplotlib qt  before calling.

    Controls
    --------
    - Play / Pause button (or press SPACE)
    - Timeline slider   : drag to scrub to any frame
    - Mouse drag        : rotate the 3D view freely
    - Scroll wheel      : zoom in / out
    - Press 1-5         : snap to preset views
        1 = Front, 2 = Right, 3 = Back, 4 = Top, 5 = Diagonal

    Parameters
    ----------
    points_3d : np.ndarray
        (n_frames, 33, 3) — can be raw or pre-aligned. NaNs handled.
    fps : int
        Playback frame rate (default 30).
    keypoint_size : int
        Marker size for joints.
    line_width : float
        Skeleton bone thickness.
    figure_size : tuple
        Figure size in inches.
    initial_view : str
        Starting camera angle: 'front', 'right', 'back', 'top', 'diagonal'.
    show_floor : bool
        Draw a translucent ground plane.
    window_title : str
        Window title bar text.
    """
    import numpy as np
    import matplotlib
    # Try to get an interactive backend
    _backend = matplotlib.get_backend().lower()
    if "agg" == _backend or "agg" in _backend and "tk" not in _backend and "qt" not in _backend:
        # Pure Agg — try switching
        for candidate in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(candidate, force=True)
                break
            except Exception:
                continue
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # ------------------------------------------------------------------
    # Skeleton definition (from the skeleton descriptor)
    # ------------------------------------------------------------------
    skel = _resolve(skeleton)
    CONNECTIONS = [(c.start, c.end) for c in skel.connections]
    _edge_color = {
        (c.start, c.end): (skel.groups[c.group].color
                           if c.group in skel.groups else "#888888")
        for c in skel.connections
    }

    def _conn_color(s, e):
        return _edge_color.get((s, e), _edge_color.get((e, s), "#888888"))

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------
    data = np.asarray(points_3d, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected (n_frames, n_keypoints, 3), got {data.shape}")

    n_frames = data.shape[0]
    duration = n_frames / fps

    # Bounds (1st–99th percentile + padding)
    vf = data[~np.isnan(data)]
    pad = 200
    x_range = [np.percentile(vf[::3], 1) - pad, np.percentile(vf[::3], 99) + pad]
    y_range = [np.percentile(vf[1::3], 1) - pad, np.percentile(vf[1::3], 99) + pad]
    z_range = [np.percentile(vf[2::3], 1) - pad, np.percentile(vf[2::3], 99) + pad]

    # Preset views
    VIEW_PRESETS = {
        "front":    {"elev": 0,  "azim": 90},
        "right":    {"elev": 0,  "azim": 0},
        "back":     {"elev": 0,  "azim": -90},
        "top":      {"elev": 90, "azim": 0},
        "diagonal": {"elev": 25, "azim": 45},
    }
    init_view = VIEW_PRESETS.get(initial_view, VIEW_PRESETS["front"])

    # ------------------------------------------------------------------
    # Build figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figure_size, facecolor="white")
    fig.canvas.manager.set_window_title(window_title)

    # Main 3D axes — leave room at bottom for controls
    ax = fig.add_axes([0.05, 0.15, 0.90, 0.80], projection="3d", facecolor="white")

    # Slider axes
    ax_slider = fig.add_axes([0.15, 0.045, 0.55, 0.03], facecolor="#e8e8e8")
    slider = Slider(
        ax_slider, "Time", 0, n_frames - 1,
        valinit=0, valstep=1, valfmt="%d",
        color="#45B7D1",
    )

    # Play/Pause button
    ax_btn = fig.add_axes([0.76, 0.035, 0.08, 0.045])
    btn_play = Button(ax_btn, "▶  Play", color="#4ECDC4", hovercolor="#96CEB4")

    # Frame/time readout (text axes)
    ax_info = fig.add_axes([0.86, 0.035, 0.12, 0.045])
    ax_info.set_axis_off()
    info_text = ax_info.text(
        0.5, 0.5, "", transform=ax_info.transAxes,
        ha="center", va="center", fontsize=11,
        fontfamily="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    state = {"playing": False, "frame": 0, "timer": None}

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_frame(frame_idx):
        """Redraw the skeleton for a given frame, preserving the camera angle."""
        # Remember current view
        elev = ax.elev
        azim = ax.azim

        ax.clear()

        pose = data[frame_idx]

        # Skeleton bones
        for s_idx, e_idx in CONNECTIONS:
            sp, ep = pose[s_idx], pose[e_idx]
            if not (np.any(np.isnan(sp)) or np.any(np.isnan(ep))):
                ax.plot3D(
                    [sp[0], ep[0]], [sp[1], ep[1]], [sp[2], ep[2]],
                    color=_conn_color(s_idx, e_idx),
                    linewidth=line_width, alpha=0.85,
                )

        # Keypoints
        vmask = ~np.isnan(pose).any(axis=1)
        if vmask.any():
            vp = pose[vmask]
            ax.scatter(
                vp[:, 0], vp[:, 1], vp[:, 2],
                c="#2C3E50", s=keypoint_size, alpha=0.9,
                edgecolors="white", linewidths=1.2, depthshade=True,
            )

        # Floor
        if show_floor:
            xx, yy = np.meshgrid(
                np.linspace(x_range[0], x_range[1], 8),
                np.linspace(y_range[0], y_range[1], 8),
            )
            ax.plot_surface(
                xx, yy, np.ones_like(xx) * z_range[0],
                alpha=0.08, color="gray", linewidth=0, antialiased=True,
            )

        # Axis setup
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)

        # Make the 3D box match the real-world proportions
        ranges = np.array([x_range[1] - x_range[0],
                        y_range[1] - y_range[0],
                        z_range[1] - z_range[0]])
        ax.set_box_aspect(ranges if ranges.min() > 0 else [1, 1, 1])

        ax.set_xlabel("X (mm)", fontsize=9, labelpad=8)
        ax.set_ylabel("Y (mm)", fontsize=9, labelpad=8)
        ax.set_zlabel("Z (mm)", fontsize=9, labelpad=8)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.25)

        # Title
        t_sec = frame_idx / fps
        ax.set_title(
            f"Frame {frame_idx}/{n_frames - 1}   •   {t_sec:.2f} s / {duration:.2f} s",
            fontsize=13, fontweight="bold", pad=14,
        )

        # Info readout
        info_text.set_text(f"F {frame_idx:>5d}\n{t_sec:>6.2f}s")

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def on_slider_change(val):
        frame_idx = int(val)
        state["frame"] = frame_idx
        draw_frame(frame_idx)

    def advance_frame():
        """Timer callback — advance one frame."""
        if not state["playing"]:
            return
        nxt = state["frame"] + 1
        if nxt >= n_frames:
            nxt = 0  # loop
        state["frame"] = nxt
        slider.set_val(nxt)  # triggers on_slider_change → draw_frame

        # Re-schedule
        interval_ms = 1000.0 / fps
        state["timer"] = fig.canvas.new_timer(interval=interval_ms)
        state["timer"].add_callback(advance_frame)
        state["timer"].single_shot = True
        state["timer"].start()

    def on_play_pause(event):
        if state["playing"]:
            # Pause
            state["playing"] = False
            if state["timer"] is not None:
                state["timer"].stop()
                state["timer"] = None
            btn_play.label.set_text("▶  Play")
            btn_play.color = "#4ECDC4"
        else:
            # Play
            state["playing"] = True
            btn_play.label.set_text("❚❚ Pause")
            btn_play.color = "#FF6B6B"
            advance_frame()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            on_play_pause(event)
        elif event.key == "1":
            ax.view_init(**VIEW_PRESETS["front"]); fig.canvas.draw_idle()
        elif event.key == "2":
            ax.view_init(**VIEW_PRESETS["right"]); fig.canvas.draw_idle()
        elif event.key == "3":
            ax.view_init(**VIEW_PRESETS["back"]); fig.canvas.draw_idle()
        elif event.key == "4":
            ax.view_init(**VIEW_PRESETS["top"]); fig.canvas.draw_idle()
        elif event.key == "5":
            ax.view_init(**VIEW_PRESETS["diagonal"]); fig.canvas.draw_idle()
        elif event.key == "left":
            nxt = max(0, state["frame"] - 1)
            state["frame"] = nxt
            slider.set_val(nxt)
        elif event.key == "right":
            nxt = min(n_frames - 1, state["frame"] + 1)
            state["frame"] = nxt
            slider.set_val(nxt)

    # Connect callbacks
    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play_pause)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # ------------------------------------------------------------------
    # Initial draw & show
    # ------------------------------------------------------------------
    ax.view_init(**init_view)
    draw_frame(0)
    plt.show()

# ============================================================================
# Interpolate NaNs
# ============================================================================

def interpolate_nans(
    points_3d,
    method="pchip",
    max_gap_seconds=1.0,
    fps=30,
    savgol_window=7,
    savgol_polyorder=3,
    verbose=True,
):
    """
    Interpolate short NaN gaps in 3D pose data, leaving long gaps untouched.

    Each keypoint × coordinate is treated as an independent 1-D signal.
    Only *bounded* gaps (valid data on both sides) up to `max_gap_seconds`
    are filled.  Leading / trailing NaN runs and gaps longer than the
    threshold are left as NaN.

    Methods
    -------
    "pchip"  – Piecewise Cubic Hermite Interpolating Polynomial.
               Smooth C1 curve that passes through every data point and
               **never overshoots** between them.  Best default for mocap:
               preserves velocity continuity without the ringing artefacts
               of cubic spline.

    "linear" – Straight-line interpolation between gap boundaries.
               Fastest, zero-parameter, and completely safe from overshoot.
               Fine for 1–3 frame micro-gaps but produces visible kinks at
               boundary frames for anything longer.

    "savgol" – Linear interpolation followed by a Savitzky-Golay smoothing
               pass over the filled region ± a margin.  The SG filter fits
               a local polynomial (least-squares) so it softens the sharp
               transitions that linear interp creates, while keeping the
               overall trajectory close to the data.  Useful when the data
               around the gap is noisy.  Tune with `savgol_window` (must be
               odd, default 7) and `savgol_polyorder` (default 3).

    Parameters
    ----------
    points_3d : np.ndarray
        Shape (n_frames, n_keypoints, 3).  Modified in-place is *not* done;
        a copy is returned.
    method : str
        "pchip", "linear", or "savgol".
    max_gap_seconds : float
        Gaps longer than this (in seconds) are skipped. Default 1.0 s.
    fps : int
        Frame rate — used to convert max_gap_seconds to frames.
    savgol_window : int
        Window length for SG filter (odd integer, only used when method="savgol").
    savgol_polyorder : int
        Polynomial order for SG filter (only used when method="savgol").
    verbose : bool
        Print a summary report.

    Returns
    -------
    filled : np.ndarray
        Same shape as input, with eligible gaps interpolated.
    report : dict
        Keys: total_values, nan_before, nan_filled, nan_after,
              gaps_found, gaps_filled, gaps_skipped_long, gaps_skipped_boundary.
    """
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter

    method = method.lower().strip()
    if method not in ("pchip", "linear", "savgol"):
        raise ValueError(f"method must be 'pchip', 'linear', or 'savgol', got '{method}'")

    data = np.array(points_3d, dtype=np.float64)  # copy
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected (n_frames, n_keypoints, 3), got {data.shape}")

    n_frames, n_kp, _ = data.shape
    max_gap_frames = int(max_gap_seconds * fps)
    frames = np.arange(n_frames)

    # Counters
    total_values = data.size
    nan_before = int(np.isnan(data).sum())
    gaps_found = 0
    gaps_filled = 0
    gaps_skipped_long = 0
    gaps_skipped_boundary = 0
    values_filled = 0

    # Process each keypoint × coordinate independently
    for kp in range(n_kp):
        for coord in range(3):
            signal = data[:, kp, coord]
            is_nan = np.isnan(signal)

            if not is_nan.any():
                continue
            if is_nan.all():
                # Entirely NaN — nothing to interpolate
                gaps_found += 1
                gaps_skipped_boundary += 1
                continue

            # --- Find bounded gaps ---
            gap_starts = []
            gap_ends = []
            in_gap = False
            for i in range(n_frames):
                if is_nan[i] and not in_gap:
                    in_gap = True
                    gap_start = i
                elif not is_nan[i] and in_gap:
                    in_gap = False
                    gap_starts.append(gap_start)
                    gap_ends.append(i)  # exclusive
            # If still in gap at end → trailing (unbounded)
            if in_gap:
                gaps_found += 1
                gaps_skipped_boundary += 1

            # Classify each bounded gap
            fill_mask = np.zeros(n_frames, dtype=bool)
            for gs, ge in zip(gap_starts, gap_ends):
                gap_len = ge - gs
                gaps_found += 1

                if gs == 0:
                    # Leading gap — no left boundary
                    gaps_skipped_boundary += 1
                    continue

                if gap_len > max_gap_frames:
                    gaps_skipped_long += 1
                    continue

                # Mark for filling
                fill_mask[gs:ge] = True
                gaps_filled += 1
                values_filled += gap_len

            if not fill_mask.any():
                continue

            # --- Interpolate ---
            valid = ~is_nan  # original valid points (not yet filled)

            if method == "linear":
                interp_vals = np.interp(frames[fill_mask], frames[valid], signal[valid])
                signal[fill_mask] = interp_vals

            elif method == "pchip":
                pchip = PchipInterpolator(frames[valid], signal[valid])
                signal[fill_mask] = pchip(frames[fill_mask])

            elif method == "savgol":
                # Step 1: linear fill
                interp_vals = np.interp(frames[fill_mask], frames[valid], signal[valid])
                signal[fill_mask] = interp_vals

                # Step 2: SG smooth over filled regions + margin
                margin = savgol_window  # frames of context either side
                smooth_mask = np.zeros(n_frames, dtype=bool)
                for gs, ge in zip(gap_starts, gap_ends):
                    if not fill_mask[gs:ge].any():
                        continue
                    lo = max(0, gs - margin)
                    hi = min(n_frames, ge + margin)
                    smooth_mask[lo:hi] = True
                # Only smooth where we have continuous data (no remaining NaN)
                smooth_mask &= ~np.isnan(signal)
                smooth_indices = np.where(smooth_mask)[0]
                if len(smooth_indices) >= savgol_window:
                    # Find contiguous runs to filter independently
                    breaks = np.where(np.diff(smooth_indices) > 1)[0] + 1
                    runs = np.split(smooth_indices, breaks)
                    for run in runs:
                        if len(run) >= savgol_window:
                            win = min(savgol_window, len(run))
                            if win % 2 == 0:
                                win -= 1
                            if win >= savgol_polyorder + 2:
                                signal[run] = savgol_filter(signal[run], win, savgol_polyorder)

            data[:, kp, coord] = signal

    nan_after = int(np.isnan(data).sum())

    report = {
        "total_values": total_values,
        "nan_before": nan_before,
        "nan_filled": nan_before - nan_after,
        "nan_after": nan_after,
        "gaps_found": gaps_found,
        "gaps_filled": gaps_filled,
        "gaps_skipped_long": gaps_skipped_long,
        "gaps_skipped_boundary": gaps_skipped_boundary,
    }

    if verbose:
        pct = lambda n: n / total_values * 100
        print()
        print("=" * 62)
        print(f"  INTERPOLATION REPORT  —  method: {method.upper()}")
        print(f"  max gap: {max_gap_seconds}s ({max_gap_frames} frames @ {fps} fps)")
        print("=" * 62)
        print(f"  Data shape        : {data.shape}")
        print(f"  Total values      : {total_values:>10,d}")
        print(f"  NaN before        : {nan_before:>10,d}  ({pct(nan_before):5.1f}%)")
        print(f"  Interpolated      : {report['nan_filled']:>10,d}  ({pct(report['nan_filled']):5.1f}%)")
        print(f"  NaN after         : {nan_after:>10,d}  ({pct(nan_after):5.1f}%)")
        print("-" * 62)
        print(f"  Gaps found        : {gaps_found:>6d}")
        print(f"    Filled          : {gaps_filled:>6d}")
        print(f"    Skipped (long)  : {gaps_skipped_long:>6d}  (> {max_gap_seconds}s)")
        print(f"    Skipped (edge)  : {gaps_skipped_boundary:>6d}  (leading/trailing/all-NaN)")
        print("=" * 62)

    return data, report

# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "pose_3d.npy"
    points_3d = np.load(data_path)
    print(f"Loaded {data_path}: shape {points_3d.shape}")

    aligned, R, orient = align_pose_to_standard_frame(points_3d)
    plot_aligned_skeleton(aligned, output_path="aligned_skeleton_front.png", elev=0, azim=90)
    plot_aligned_skeleton(aligned, output_path="aligned_skeleton_diag.png", elev=25, azim=45)
    print("\nDone.")
