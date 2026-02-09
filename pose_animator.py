"""
3D Pose Animation Module

This module creates animated videos of 3D pose reconstructions with MediaPipe-style
skeleton visualization. Supports multiple view modes, customizable styling, and
high-quality video output.

Author: Generated for FreeMoCap-style workflow
Date: 2026-02-06
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# AUTO-ORIENTATION DETECTION
# ============================================================================

def detect_person_orientation(points_3d, frame_range=None):
    """
    Detect which way is "up" and which direction the person is facing.
    
    Args:
        points_3d (np.ndarray): Array of shape (n_frames, n_keypoints, 3)
        frame_range (tuple, optional): Tuple (start, end) for frames to analyze, or None for all
        
    Returns:
        dict: Orientation information containing:
            - 'up_vector': 3D vector pointing "up" 
            - 'forward_vector': 3D vector the person is facing
            - 'right_vector': 3D vector to the person's right
            - 'rotation_matrix': 3x3 matrix to align to standard orientation
            - 'ground_plane_z': Z-coordinate of the ground plane
            - 'up_axis': Index of dominant up axis (0=X, 1=Y, 2=Z)
            - 'forward_axis': Index of dominant forward axis
    """
    
    print("\n" + "="*70)
    print("DETECTING PERSON ORIENTATION")
    print("="*70)
    
    # Keypoint indices (MediaPipe pose)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Select frames to analyze (use middle portion for stability)
    if frame_range is None:
        n_frames = points_3d.shape[0]
        start = n_frames // 4
        end = 3 * n_frames // 4
    else:
        start, end = frame_range
    
    sample_frames = points_3d[start:end]
    
    # ========================================================================
    # 1. Detect "UP" direction (from hips to shoulders)
    # ========================================================================
    
    up_vectors = []
    
    for frame in sample_frames:
        # Get mid-hip point
        left_hip = frame[LEFT_HIP]
        right_hip = frame[RIGHT_HIP]
        
        # Get mid-shoulder point
        left_shoulder = frame[LEFT_SHOULDER]
        right_shoulder = frame[RIGHT_SHOULDER]
        
        # Check validity
        if (not np.any(np.isnan(left_hip)) and not np.any(np.isnan(right_hip)) and
            not np.any(np.isnan(left_shoulder)) and not np.any(np.isnan(right_shoulder))):
            
            mid_hip = (left_hip + right_hip) / 2
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            
            # Vector from hips to shoulders = "up"
            up_vec = mid_shoulder - mid_hip
            
            if np.linalg.norm(up_vec) > 0:
                up_vectors.append(up_vec / np.linalg.norm(up_vec))
    
    if len(up_vectors) == 0:
        print("⚠ Warning: Could not detect up direction, using Z-axis")
        up_vector = np.array([0, 0, 1])
    else:
        # Average and normalize
        up_vector = np.mean(up_vectors, axis=0)
        up_vector = up_vector / np.linalg.norm(up_vector)
    
    print(f"\n✓ UP direction detected: [{up_vector[0]:.3f}, {up_vector[1]:.3f}, {up_vector[2]:.3f}]")
    print(f"  Dominant axis: {'X' if abs(up_vector[0]) > 0.7 else 'Y' if abs(up_vector[1]) > 0.7 else 'Z'}")
    
    # ========================================================================
    # 2. Detect "FORWARD" direction (perpendicular to shoulders, from torso)
    # ========================================================================
    
    forward_vectors = []
    
    for frame in sample_frames:
        left_shoulder = frame[LEFT_SHOULDER]
        right_shoulder = frame[RIGHT_SHOULDER]
        nose = frame[NOSE]
        
        if (not np.any(np.isnan(left_shoulder)) and 
            not np.any(np.isnan(right_shoulder)) and
            not np.any(np.isnan(nose))):
            
            # Vector from right shoulder to left shoulder
            shoulder_vec = left_shoulder - right_shoulder
            
            # Mid-shoulder point
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            
            # Vector from mid-shoulder to nose
            nose_vec = nose - mid_shoulder
            
            # Forward = cross product of shoulder vector and up vector
            # This gives the direction perpendicular to the shoulder line
            forward_vec = np.cross(shoulder_vec, up_vector)
            
            # Make sure it points toward the nose (front of body)
            if np.dot(forward_vec, nose_vec) < 0:
                forward_vec = -forward_vec
            
            if np.linalg.norm(forward_vec) > 0:
                forward_vectors.append(forward_vec / np.linalg.norm(forward_vec))
    
    if len(forward_vectors) == 0:
        print("⚠ Warning: Could not detect forward direction, using Y-axis")
        forward_vector = np.array([0, 1, 0])
    else:
        forward_vector = np.mean(forward_vectors, axis=0)
        forward_vector = forward_vector / np.linalg.norm(forward_vector)
    
    print(f"✓ FORWARD direction detected: [{forward_vector[0]:.3f}, {forward_vector[1]:.3f}, {forward_vector[2]:.3f}]")
    
    # ========================================================================
    # 3. Compute "RIGHT" direction (perpendicular to both)
    # ========================================================================
    
    right_vector = np.cross(forward_vector, up_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)
    
    print(f"✓ RIGHT direction detected: [{right_vector[0]:.3f}, {right_vector[1]:.3f}, {right_vector[2]:.3f}]")
    
    # ========================================================================
    # 4. Detect ground plane
    # ========================================================================
    
    ankle_heights = []
    
    for frame in sample_frames:
        left_ankle = frame[LEFT_ANKLE]
        right_ankle = frame[RIGHT_ANKLE]
        
        if not np.any(np.isnan(left_ankle)):
            ankle_heights.append(np.dot(left_ankle, up_vector))
        if not np.any(np.isnan(right_ankle)):
            ankle_heights.append(np.dot(right_ankle, up_vector))
    
    if len(ankle_heights) > 0:
        ground_plane_z = np.percentile(ankle_heights, 5)  # 5th percentile (feet on ground)
    else:
        ground_plane_z = 0
    
    print(f"✓ Ground plane detected at: {ground_plane_z:.2f} mm along up-axis")
    
    # ========================================================================
    # 5. Create rotation matrix to standard orientation
    # ========================================================================
    # Standard: X=right, Y=forward, Z=up
    
    rotation_matrix = np.column_stack([right_vector, forward_vector, up_vector])
    
    print(f"\n✓ Orientation detection complete!")
    
    return {
        'up_vector': up_vector,
        'forward_vector': forward_vector,
        'right_vector': right_vector,
        'rotation_matrix': rotation_matrix,
        'ground_plane_z': ground_plane_z,
        'up_axis': np.argmax(np.abs(up_vector)),  # 0=X, 1=Y, 2=Z
        'forward_axis': np.argmax(np.abs(forward_vector))
    }


def get_optimal_camera_angles(orientation_info):
    """
    Get optimal camera angles based on detected orientation.
    
    Args:
        orientation_info (dict): Dict from detect_person_orientation()
        
    Returns:
        dict: Recommended camera angles for different views, each containing:
            - 'elevation': Camera elevation angle
            - 'azimuth': Camera azimuth angle  
            - 'description': Human-readable description
    """
    
    up_vec = orientation_info['up_vector']
    forward_vec = orientation_info['forward_vector']
    right_vec = orientation_info['right_vector']
    
    # Determine which axis is "up"
    up_axis = np.argmax(np.abs(up_vec))
    
    # Calculate angles
    # Azimuth: rotation around up axis
    azimuth_front = np.degrees(np.arctan2(forward_vec[1], forward_vec[0]))
    azimuth_side = azimuth_front + 90  # 90 degrees to the right
    azimuth_back = azimuth_front + 180
    
    # Adjust elevation based on actual up direction
    if up_axis == 0:  # X is up
        elevation_top = 0
        elevation_front = 0
    elif up_axis == 1:  # Y is up
        elevation_top = 90
        elevation_front = 0
    else:  # Z is up (standard)
        elevation_top = 90
        elevation_front = 0
    
    return {
        'front': {
            'elevation': elevation_front,
            'azimuth': azimuth_front,
            'description': 'Looking at person from front'
        },
        'back': {
            'elevation': elevation_front,
            'azimuth': azimuth_back,
            'description': 'Looking at person from back'
        },
        'left_side': {
            'elevation': elevation_front,
            'azimuth': azimuth_side - 90,
            'description': 'Looking at person from their left side'
        },
        'right_side': {
            'elevation': elevation_front,
            'azimuth': azimuth_side,
            'description': 'Looking at person from their right side'
        },
        'top': {
            'elevation': elevation_top,
            'azimuth': azimuth_front,
            'description': 'Looking down at person from above'
        },
        'diagonal': {
            'elevation': 30,
            'azimuth': azimuth_front + 45,
            'description': 'Diagonal view from front-right'
        }
    }


def visualize_orientation(points_3d, orientation, camera_angles, output_path=None, frame_idx=None):
    """
    Create a visualization showing the detected orientation vectors and camera angles.
    
    This creates a two-panel plot:
    - LEFT: Diagonal view showing all orientation vectors (up, forward, right)
    - RIGHT: Front view showing what the 'front' camera will see
    
    Args:
        points_3d (np.ndarray): Array of shape (n_frames, n_keypoints, 3)
        orientation (dict): Output from detect_person_orientation()
        camera_angles (dict): Output from get_optimal_camera_angles()
        output_path (str, optional): Path to save the visualization. If None, only displays.
        frame_idx (int, optional): Which frame to visualize (None = middle frame)
        
    Returns:
        matplotlib.figure.Figure: The created figure
        
    Example:
        >>> orientation = detect_person_orientation(points_3d)
        >>> camera_angles = get_optimal_camera_angles(orientation)
        >>> fig = visualize_orientation(
        ...     points_3d, 
        ...     orientation, 
        ...     camera_angles,
        ...     output_path="visualizations/orientation.png"
        ... )
    """
    
    # Select a frame to visualize
    if frame_idx is None:
        frame_idx = points_3d.shape[0] // 2  # Middle frame
    
    frame = points_3d[frame_idx]
    
    # Get key body points
    nose = frame[0]
    left_shoulder = frame[11]
    right_shoulder = frame[12]
    left_hip = frame[23]
    right_hip = frame[24]
    
    # Calculate body center (between hips and shoulders)
    if not np.any(np.isnan([left_hip, right_hip, left_shoulder, right_shoulder])):
        mid_hip = (left_hip + right_hip) / 2
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        body_center = (mid_hip + mid_shoulder) / 2
    else:
        # Fallback to center of all valid points
        valid_points = frame[~np.isnan(frame).any(axis=1)]
        body_center = np.mean(valid_points, axis=0)
    
    # Get orientation vectors
    up_vec = orientation['up_vector']
    forward_vec = orientation['forward_vector']
    right_vec = orientation['right_vector']
    
    # Scale vectors for visibility (make them long enough to see)
    scale = 500  # mm
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(18, 8))
    
    # ========================================================================
    # LEFT PLOT: Skeleton with orientation vectors
    # ========================================================================
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot the skeleton
    valid_mask = ~np.isnan(frame).any(axis=1)
    valid_points = frame[valid_mask]
    
    if len(valid_points) > 0:
        ax1.scatter(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            c='lightgray',
            s=30,
            alpha=0.6,
            label='Keypoints'
        )
    
    # Draw skeleton connections (simplified)
    connections = [
        (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]
    
    for start_idx, end_idx in connections:
        start = frame[start_idx]
        end = frame[end_idx]
        if not np.any(np.isnan(start)) and not np.any(np.isnan(end)):
            ax1.plot3D(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                'gray', linewidth=1, alpha=0.4
            )
    
    # Plot body center
    ax1.scatter(
        body_center[0], body_center[1], body_center[2],
        c='black', s=200, marker='o', 
        edgecolors='white', linewidths=2,
        label='Body Center', zorder=10
    )
    
    # Plot orientation vectors from body center
    # UP vector (green)
    ax1.quiver(
        body_center[0], body_center[1], body_center[2],
        up_vec[0], up_vec[1], up_vec[2],
        length=scale, color='green', arrow_length_ratio=0.15,
        linewidth=4, label='UP', alpha=0.9
    )
    ax1.text(
        body_center[0] + up_vec[0]*scale*1.1,
        body_center[1] + up_vec[1]*scale*1.1,
        body_center[2] + up_vec[2]*scale*1.1,
        'UP', fontsize=16, fontweight='bold', color='green'
    )
    
    # DOWN vector (light green, dashed)
    ax1.quiver(
        body_center[0], body_center[1], body_center[2],
        -up_vec[0], -up_vec[1], -up_vec[2],
        length=scale*0.7, color='lightgreen', arrow_length_ratio=0.15,
        linewidth=2, linestyle='--', alpha=0.6
    )
    ax1.text(
        body_center[0] - up_vec[0]*scale*0.8,
        body_center[1] - up_vec[1]*scale*0.8,
        body_center[2] - up_vec[2]*scale*0.8,
        'DOWN', fontsize=12, color='lightgreen', alpha=0.7
    )
    
    # FORWARD vector (red)
    ax1.quiver(
        body_center[0], body_center[1], body_center[2],
        forward_vec[0], forward_vec[1], forward_vec[2],
        length=scale, color='red', arrow_length_ratio=0.15,
        linewidth=4, label='FORWARD', alpha=0.9
    )
    ax1.text(
        body_center[0] + forward_vec[0]*scale*1.1,
        body_center[1] + forward_vec[1]*scale*1.1,
        body_center[2] + forward_vec[2]*scale*1.1,
        'FORWARD', fontsize=16, fontweight='bold', color='red'
    )
    
    # BACK vector (pink, dashed)
    ax1.quiver(
        body_center[0], body_center[1], body_center[2],
        -forward_vec[0], -forward_vec[1], -forward_vec[2],
        length=scale*0.7, color='pink', arrow_length_ratio=0.15,
        linewidth=2, linestyle='--', alpha=0.6
    )
    ax1.text(
        body_center[0] - forward_vec[0]*scale*0.8,
        body_center[1] - forward_vec[1]*scale*0.8,
        body_center[2] - forward_vec[2]*scale*0.8,
        'BACK', fontsize=12, color='pink', alpha=0.7
    )
    
    # RIGHT vector (blue)
    ax1.quiver(
        body_center[0], body_center[1], body_center[2],
        right_vec[0], right_vec[1], right_vec[2],
        length=scale, color='blue', arrow_length_ratio=0.15,
        linewidth=4, label='RIGHT', alpha=0.9
    )
    ax1.text(
        body_center[0] + right_vec[0]*scale*1.1,
        body_center[1] + right_vec[1]*scale*1.1,
        body_center[2] + right_vec[2]*scale*1.1,
        'RIGHT', fontsize=16, fontweight='bold', color='blue'
    )
    
    # LEFT vector (light blue, dashed)
    ax1.quiver(
        body_center[0], body_center[1], body_center[2],
        -right_vec[0], -right_vec[1], -right_vec[2],
        length=scale*0.7, color='lightblue', arrow_length_ratio=0.15,
        linewidth=2, linestyle='--', alpha=0.6
    )
    ax1.text(
        body_center[0] - right_vec[0]*scale*0.8,
        body_center[1] - right_vec[1]*scale*0.8,
        body_center[2] - right_vec[2]*scale*0.8,
        'LEFT', fontsize=12, color='lightblue', alpha=0.7
    )
    
    # Set labels and title
    ax1.set_xlabel('X (mm)', fontsize=10)
    ax1.set_ylabel('Y (mm)', fontsize=10)
    ax1.set_zlabel('Z (mm)', fontsize=10)
    ax1.set_title('3D View: Detected Orientation\n(Diagonal Perspective)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Set viewing angle (diagonal view to see all axes)
    ax1.view_init(elev=20, azim=45)
    
    # Make axes equal
    valid_points_all = points_3d[~np.isnan(points_3d)]
    max_range = np.array([
        valid_points_all[::3].max()-valid_points_all[::3].min(),
        valid_points_all[1::3].max()-valid_points_all[1::3].min(),
        valid_points_all[2::3].max()-valid_points_all[2::3].min()
    ]).max() / 2.0
    
    mid_x = (valid_points_all[::3].max()+valid_points_all[::3].min()) * 0.5
    mid_y = (valid_points_all[1::3].max()+valid_points_all[1::3].min()) * 0.5
    mid_z = (valid_points_all[2::3].max()+valid_points_all[2::3].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # ========================================================================
    # RIGHT PLOT: View from FRONT (what the front camera sees)
    # ========================================================================
    
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot the skeleton
    if len(valid_points) > 0:
        ax2.scatter(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            c='lightgray',
            s=30,
            alpha=0.6
        )
    
    # Draw skeleton connections
    for start_idx, end_idx in connections:
        start = frame[start_idx]
        end = frame[end_idx]
        if not np.any(np.isnan(start)) and not np.any(np.isnan(end)):
            ax2.plot3D(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                'gray', linewidth=1, alpha=0.4
            )
    
    # Plot body center
    ax2.scatter(
        body_center[0], body_center[1], body_center[2],
        c='black', s=200, marker='o',
        edgecolors='white', linewidths=2, zorder=10
    )
    
    # Plot orientation vectors (same as before)
    ax2.quiver(body_center[0], body_center[1], body_center[2],
               up_vec[0], up_vec[1], up_vec[2],
               length=scale, color='green', arrow_length_ratio=0.15,
               linewidth=4, alpha=0.9)
    ax2.text(body_center[0] + up_vec[0]*scale*1.1,
             body_center[1] + up_vec[1]*scale*1.1,
             body_center[2] + up_vec[2]*scale*1.1,
             'UP', fontsize=16, fontweight='bold', color='green')
    
    ax2.quiver(body_center[0], body_center[1], body_center[2],
               forward_vec[0], forward_vec[1], forward_vec[2],
               length=scale, color='red', arrow_length_ratio=0.15,
               linewidth=4, alpha=0.9)
    ax2.text(body_center[0] + forward_vec[0]*scale*1.1,
             body_center[1] + forward_vec[1]*scale*1.1,
             body_center[2] + forward_vec[2]*scale*1.1,
             'FORWARD\n(toward camera)', fontsize=16, fontweight='bold', color='red',
             ha='center')
    
    ax2.quiver(body_center[0], body_center[1], body_center[2],
               right_vec[0], right_vec[1], right_vec[2],
               length=scale, color='blue', arrow_length_ratio=0.15,
               linewidth=4, alpha=0.9)
    ax2.text(body_center[0] + right_vec[0]*scale*1.1,
             body_center[1] + right_vec[1]*scale*1.1,
             body_center[2] + right_vec[2]*scale*1.1,
             'RIGHT', fontsize=16, fontweight='bold', color='blue')
    
    ax2.quiver(body_center[0], body_center[1], body_center[2],
               -right_vec[0], -right_vec[1], -right_vec[2],
               length=scale*0.7, color='lightblue', arrow_length_ratio=0.15,
               linewidth=2, linestyle='--', alpha=0.6)
    ax2.text(body_center[0] - right_vec[0]*scale*0.8,
             body_center[1] - right_vec[1]*scale*0.8,
             body_center[2] - right_vec[2]*scale*0.8,
             'LEFT', fontsize=12, color='lightblue', alpha=0.7)
    
    # Set viewing angle to FRONT (camera looking at person's front)
    front_azimuth = camera_angles['front']['azimuth']
    front_elevation = camera_angles['front']['elevation']
    
    ax2.view_init(elev=front_elevation, azim=front_azimuth)
    
    ax2.set_xlabel('X (mm)', fontsize=10)
    ax2.set_ylabel('Y (mm)', fontsize=10)
    ax2.set_zlabel('Z (mm)', fontsize=10)
    ax2.set_title(f'FRONT View\n(Camera looking AT person)\nElevation: {front_elevation:.1f}°, Azimuth: {front_azimuth:.1f}°',
                  fontsize=14, fontweight='bold', pad=20)
    
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Orientation visualization saved to: {output_path}")
    
    return fig


def animate_3d_pose_auto_orient(
    points_3d,
    output_path,
    view_mode='front',
    auto_detect_orientation=True,
    **kwargs
):
    """
    Create 3D animation with automatic orientation detection.
    
    This wrapper automatically detects which way is up and which way the person
    is facing, then adjusts camera angles accordingly for natural viewing.
    
    Args:
        points_3d (np.ndarray): Array of shape (n_frames, n_keypoints, 3)
        output_path (str): Path to save video
        view_mode (str): Camera view to create. Options:
            - 'front': Looking at person from front (auto-detected)
            - 'back': Looking at person from back
            - 'left_side': Looking from person's left side
            - 'right_side': Looking from person's right side
            - 'top': Bird's eye view from above
            - 'diagonal': Diagonal view from front-right
            - 'rotating': Rotating view starting from front (auto-detected)
        auto_detect_orientation (bool): If True, auto-detect orientation. 
                                       If False, use manual angles from kwargs.
        **kwargs: Additional arguments passed to animate_3d_pose()
        
    Returns:
        str: Path to saved video
        
    Example:
        >>> # Load 3D data
        >>> points_3d = np.load("data_3d/pose_3d.npy")
        >>> 
        >>> # Create auto-oriented front view
        >>> animate_3d_pose_auto_orient(
        ...     points_3d,
        ...     output_path="animations/pose_front.mp4",
        ...     view_mode='front',
        ...     fps=30,
        ...     quality='high'
        ... )
        >>> 
        >>> # Create rotating view from correct angle
        >>> animate_3d_pose_auto_orient(
        ...     points_3d,
        ...     output_path="animations/pose_rotating.mp4",
        ...     view_mode='rotating',
        ...     fps=30
        ... )
    """
    
    if auto_detect_orientation:
        # Detect orientation
        orientation = detect_person_orientation(points_3d)
        camera_angles = get_optimal_camera_angles(orientation)
        
        # Print recommended views
        print("\n" + "="*70)
        print("RECOMMENDED CAMERA ANGLES")
        print("="*70)
        for view_name, angles in camera_angles.items():
            print(f"\n{view_name.upper()}:")
            print(f"  Elevation: {angles['elevation']:.1f}°")
            print(f"  Azimuth: {angles['azimuth']:.1f}°")
            print(f"  {angles['description']}")
        
        # Get angles for requested view
        if view_mode == 'rotating':
            # Use front view as starting point for rotation
            angles = camera_angles['front']
            kwargs['elevation'] = angles['elevation']
            kwargs['azimuth_start'] = angles['azimuth']
            kwargs['view_mode'] = 'rotating'
        elif view_mode in camera_angles:
            angles = camera_angles[view_mode]
            kwargs['elevation'] = angles['elevation']
            kwargs['azimuth_start'] = angles['azimuth']
            kwargs['view_mode'] = 'custom'
        else:
            print(f"⚠ Unknown view mode '{view_mode}', using default")
    
    # Create animation
    return animate_3d_pose(points_3d, output_path, **kwargs)


def animate_3d_pose(
    points_3d,
    output_path,
    fps=30,
    frames_to_animate=None,
    view_mode='rotating',
    elevation=20,
    azimuth_start=45,
    rotation_speed=0.5,
    show_floor=True,
    show_frame_number=True,
    show_timestamp=True,
    show_axes=True,
    keypoint_size=50,
    line_width=2.5,
    figure_size=(12, 10),
    dpi=100,
    quality='high'
):
    """
    Create an animated 3D visualization of pose over time with MediaPipe-style skeleton.
    
    Args:
        points_3d (np.ndarray): Array of shape (n_frames, n_keypoints, 3) with 3D coordinates in mm
        output_path (str): Path to save the output video (e.g., 'pose_animation.mp4')
        fps (int): Frames per second for the video (default: 30)
        frames_to_animate (int, optional): Number of frames to animate. None = all frames
        view_mode (str): Camera view mode. Options:
            - 'rotating': Camera rotates around the subject (default)
            - 'front': Fixed front view
            - 'side': Fixed side view  
            - 'top': Fixed top-down view
            - 'diagonal': Fixed diagonal view
            - 'custom': Custom view (use elevation and azimuth_start parameters)
        elevation (float): Camera elevation angle in degrees (0=level, 90=top-down) (default: 20)
        azimuth_start (float): Starting azimuth angle in degrees (default: 45)
        rotation_speed (float): Rotation speed in degrees per frame (for 'rotating' mode) (default: 0.5)
        show_floor (bool): Whether to show a floor grid (default: True)
        show_frame_number (bool): Whether to display frame counter (default: True)
        show_timestamp (bool): Whether to display time in seconds (default: True)
        show_axes (bool): Whether to show axis labels and ticks (default: True)
        keypoint_size (int): Size of joint markers (default: 50)
        line_width (float): Width of skeleton lines (default: 2.5)
        figure_size (tuple): Figure size as (width, height) in inches (default: (12, 10))
        dpi (int): Resolution of output video. Higher = better quality (default: 100)
        quality (str): Video quality preset. Options:
            - 'low': dpi=75, bitrate=1500
            - 'medium': dpi=100, bitrate=3000
            - 'high': dpi=150, bitrate=5000 (default)
            - 'ultra': dpi=200, bitrate=8000
    
    Returns:
        str: Path to the saved video file
        
    Example:
        >>> # Load your 3D data
        >>> points_3d = np.load("data_3d/pose_3d.npy")
        >>> 
        >>> # Create a rotating animation
        >>> animate_3d_pose(
        ...     points_3d,
        ...     output_path="animations/pose_rotating.mp4",
        ...     view_mode='rotating',
        ...     fps=30
        ... )
        >>> 
        >>> # Create a static front view
        >>> animate_3d_pose(
        ...     points_3d,
        ...     output_path="animations/pose_front.mp4",
        ...     view_mode='front',
        ...     fps=30
        ... )
    """
    
    # ========================================================================
    # MediaPipe skeleton connections and colors
    # ========================================================================
    
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
    
    BODY_PART_COLORS = {
        'face': '#FF6B6B',
        'torso': '#4ECDC4',
        'left_arm': '#45B7D1',
        'right_arm': '#96CEB4',
        'left_leg': '#FFEAA7',
        'right_leg': '#DDA15E',
    }
    
    def get_connection_color(connection):
        """Assign colors to skeleton connections."""
        start, end = connection
        if start <= 10 and end <= 10:
            return BODY_PART_COLORS['face']
        if connection in [(11, 12), (11, 23), (12, 24), (23, 24)]:
            return BODY_PART_COLORS['torso']
        if start in [11, 13, 15, 17, 19, 21] and end in [11, 13, 15, 17, 19, 21]:
            return BODY_PART_COLORS['left_arm']
        if start in [12, 14, 16, 18, 20, 22] and end in [12, 14, 16, 18, 20, 22]:
            return BODY_PART_COLORS['right_arm']
        if start in [23, 25, 27, 29, 31] and end in [23, 25, 27, 29, 31]:
            return BODY_PART_COLORS['left_leg']
        if start in [24, 26, 28, 30, 32] and end in [24, 26, 28, 30, 32]:
            return BODY_PART_COLORS['right_leg']
        return '#888888'
    
    # ========================================================================
    # Setup and validation
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING 3D POSE ANIMATION")
    print("="*70)
    
    # Validate input
    if points_3d.ndim != 3 or points_3d.shape[2] != 3:
        raise ValueError(f"Expected points_3d shape (n_frames, n_keypoints, 3), got {points_3d.shape}")
    
    n_frames = points_3d.shape[0]
    if frames_to_animate is None:
        frames_to_animate = n_frames
    else:
        frames_to_animate = min(frames_to_animate, n_frames)
    
    # Quality presets
    quality_presets = {
        'low': {'dpi': 75, 'bitrate': 1500},
        'medium': {'dpi': 100, 'bitrate': 3000},
        'high': {'dpi': 150, 'bitrate': 5000},
        'ultra': {'dpi': 200, 'bitrate': 8000}
    }
    
    if quality in quality_presets:
        dpi = quality_presets[quality]['dpi']
        bitrate = quality_presets[quality]['bitrate']
    else:
        bitrate = 3000
    
    print(f"\nSettings:")
    print(f"  Total frames available: {n_frames}")
    print(f"  Frames to animate: {frames_to_animate}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {frames_to_animate/fps:.2f} seconds")
    print(f"  View mode: {view_mode}")
    print(f"  Quality: {quality} (DPI: {dpi}, Bitrate: {bitrate})")
    print(f"  Output: {output_path}")
    
    # ========================================================================
    # Calculate bounds for consistent scaling
    # ========================================================================
    
    valid_points = points_3d[~np.isnan(points_3d)]
    x_min, x_max = np.percentile(valid_points[::3], [1, 99])
    y_min, y_max = np.percentile(valid_points[1::3], [1, 99])
    z_min, z_max = np.percentile(valid_points[2::3], [1, 99])
    
    padding = 200  # mm
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]
    z_range = [z_min - padding, z_max + padding]
    
    # ========================================================================
    # Setup view angles
    # ========================================================================
    
    view_presets = {
        'front': {'elev': 0, 'azim': 0},
        'side': {'elev': 0, 'azim': 90},
        'top': {'elev': 90, 'azim': 0},
        'diagonal': {'elev': 30, 'azim': 45},
        'rotating': {'elev': elevation, 'azim': azimuth_start},
        'custom': {'elev': elevation, 'azim': azimuth_start}
    }
    
    initial_view = view_presets.get(view_mode, view_presets['rotating'])
    
    # ========================================================================
    # Create figure and animation
    # ========================================================================
    
    fig = plt.figure(figsize=figure_size, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    def init():
        """Initialize the plot."""
        ax.clear()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # Conditionally show/hide axes
        if show_axes:
            ax.set_xlabel('X (mm)', fontsize=10, labelpad=10)
            ax.set_ylabel('Y (mm)', fontsize=10, labelpad=10)
            ax.set_zlabel('Z (mm)', fontsize=10, labelpad=10)
        else:
            # Hide axis labels and ticks
            ax.set_axis_off()
            ax._axis3don = False  # extra belt-and-braces
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')

            # Remove ticks completely
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            # Make axis lines invisible
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            # Force ticks and tick lines to white (effectively invisible)
            ax.tick_params(
                axis='both',
                which='both',
                colors='white',        # tick + label color
                length=0               # no tick marks
            )

            # 3D-specific: force tick line colors
            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a._axinfo['tick']['color'] = (1, 1, 1, 0)
                a._axinfo['axisline']['color'] = (1, 1, 1, 0)
            # Hide panes
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            # Hide grid
            ax.grid(False)
        
        ax.view_init(elev=initial_view['elev'], azim=initial_view['azim'])
        
        if show_floor:
            xx, yy = np.meshgrid(
                np.linspace(x_range[0], x_range[1], 10),
                np.linspace(y_range[0], y_range[1], 10)
            )
            zz = np.ones_like(xx) * z_range[0]
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
        return []
    
    def update(frame):
        """Update function for each frame."""
        ax.clear()
        
        current_pose = points_3d[frame]
        
        # Set consistent limits
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # Conditionally show/hide axes
        if show_axes:
            ax.set_xlabel('X (mm)', fontsize=10, labelpad=10)
            ax.set_ylabel('Y (mm)', fontsize=10, labelpad=10)
            ax.set_zlabel('Z (mm)', fontsize=10, labelpad=10)
        else:
            # Hide axis labels and ticks
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            # Make axis lines invisible
            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            # Hide panes
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')
            # Hide grid
            ax.grid(False)
        
        # Update view angle
        if view_mode == 'rotating':
            current_azim = azimuth_start + (frame * rotation_speed)
            ax.view_init(elev=elevation, azim=current_azim)
        else:
            ax.view_init(elev=initial_view['elev'], azim=initial_view['azim'])
        
        # Draw floor
        if show_floor:
            xx, yy = np.meshgrid(
                np.linspace(x_range[0], x_range[1], 10),
                np.linspace(y_range[0], y_range[1], 10)
            )
            zz = np.ones_like(xx) * z_range[0]
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray',
                          linewidth=0, antialiased=True)
        
        # Draw skeleton connections
        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = current_pose[start_idx]
            end_point = current_pose[end_idx]
            
            if not np.any(np.isnan(start_point)) and not np.any(np.isnan(end_point)):
                color = get_connection_color(connection)
                ax.plot3D(
                    [start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]],
                    color=color,
                    linewidth=line_width,
                    alpha=0.8
                )
        
        # Draw keypoints
        valid_mask = ~np.isnan(current_pose).any(axis=1)
        valid_points = current_pose[valid_mask]
        
        if len(valid_points) > 0:
            ax.scatter(
                valid_points[:, 0],
                valid_points[:, 1],
                valid_points[:, 2],
                c='#2C3E50',
                s=keypoint_size,
                alpha=0.9,
                edgecolors='white',
                linewidths=1,
                depthshade=True
            )
        
        # Add frame/time info
        if show_frame_number or show_timestamp:
            info_text = []
            if show_frame_number:
                info_text.append(f'Frame: {frame}/{frames_to_animate-1}')
            if show_timestamp:
                info_text.append(f'Time: {frame/fps:.2f}s')
            
            ax.text2D(
                0.05, 0.95,
                '\n'.join(info_text),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_title('3D Pose Reconstruction', fontsize=14, fontweight='bold', pad=20)
        
        return []
    
    # ========================================================================
    # Create and save animation
    # ========================================================================
    
    print("\nGenerating animation frames...")
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames_to_animate,
        interval=1000/fps,
        blit=False
    )
    
    print(f"Saving video to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = FFMpegWriter(fps=fps, bitrate=bitrate, codec='libx264')
    
    with tqdm(total=frames_to_animate, desc="Rendering") as pbar:
        def progress_callback(current_frame, total_frames):
            pbar.update(1)
        
        anim.save(
            str(output_path),
            writer=writer,
            dpi=dpi,
            progress_callback=progress_callback
        )
    
    plt.close(fig)
    
    file_size_mb = output_path.stat().st_size / 1_000_000
    
    print(f"\n✓ Animation saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Duration: {frames_to_animate/fps:.2f} seconds")
    
    return str(output_path)


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage when running this file directly.
    In practice, you'll import these functions into your Jupyter notebook.
    """
    
    # Example: Load 3D data and create animations
    POSE_3D_FILE = "data_3d/pose_3d.npy"
    OUTPUT_DIR = "animations"
    
    print("Loading 3D pose data...")
    points_3d = np.load(POSE_3D_FILE)
    print(f"✓ Loaded: {points_3d.shape}")
    
    # Create auto-oriented front view
    animate_3d_pose_auto_orient(
        points_3d,
        output_path=f"{OUTPUT_DIR}/pose_front_auto.mp4",
        view_mode='front',
        fps=30,
        quality='high'
    )
    
    # Create auto-oriented rotating view
    animate_3d_pose_auto_orient(
        points_3d,
        output_path=f"{OUTPUT_DIR}/pose_rotating_auto.mp4",
        view_mode='rotating',
        fps=30,
        rotation_speed=0.5,
        quality='high'
    )
    
    # Create auto-oriented side view
    animate_3d_pose_auto_orient(
        points_3d,
        output_path=f"{OUTPUT_DIR}/pose_side_auto.mp4",
        view_mode='right_side',
        fps=30,
        quality='high'
    )
    
    print("\n✓ All animations created!")