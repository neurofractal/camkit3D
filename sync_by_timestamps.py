"""
Timestamp-based Video Synchronization
Synchronizes videos to cam0 by finding nearest frame in time for each cam0 frame
Ensures all output videos have SAME frame count as cam0
"""

from pathlib import Path
import re
import numpy as np
import cv2
from typing import Dict, Tuple, List


def find_nearest_frame_indices(target_times: np.ndarray, source_times: np.ndarray) -> np.ndarray:
    """
    For each target time, find the index of the nearest source time.
    
    Args:
        target_times: Array of times to match (e.g., cam0 timestamps)
        source_times: Array of times to search in (e.g., cam1 timestamps)
    
    Returns:
        Array of indices into source_times, same length as target_times
    """
    target_times = np.asarray(target_times).ravel()
    source_times = np.asarray(source_times).ravel()
    
    # For each target time, find nearest source time
    indices = np.zeros(len(target_times), dtype=np.int64)
    
    for i, t_target in enumerate(target_times):
        # Find closest time in source
        time_diffs = np.abs(source_times - t_target)
        nearest_idx = np.argmin(time_diffs)
        indices[i] = nearest_idx
    
    return indices


def synchronize_videos_to_ideal_fps(
    trial_folder: str,
    target_fps: float = 30.0,
    raw_videos_subdir: str = "raw_videos",
    out_subdir: str = "synchronized_videos",
    max_time_diff_ms: float = 100.0,  # Maximum allowed time difference in milliseconds
    progress_callback=None,  # callable(frames_done: int, total_frames: int) or None
):
    """
    Synchronize all videos to an ideal clock at target_fps using saved timestamps.
    
    This creates a perfect timing grid from earliest to latest timestamp across all cameras,
    then finds the nearest actual frame from each camera for each ideal timepoint.
    
    This approach is BETTER than aligning to a specific camera because:
    - No camera is perfect (all have dropped frames, jitter, etc.)
    - Creates truly synchronized output regardless of individual camera issues
    - All cameras treated equally
    
    Args:
        trial_folder: Path to trial folder containing timestamps and raw_videos
        target_fps: Target frame rate for ideal clock (e.g., 30.0)
        raw_videos_subdir: Subdirectory containing raw videos
        out_subdir: Output subdirectory for synchronized videos
        max_time_diff_ms: Maximum allowed time difference for matching (ms)
    
    Returns:
        Dictionary with synchronization results and metrics
    """
    trial_folder = Path(trial_folder)
    raw_dir = trial_folder / raw_videos_subdir
    out_dir = trial_folder / out_subdir
    
    print(f"\n{'='*70}")
    print(f"Timestamp-based Synchronization to Ideal {target_fps} FPS Clock")
    print(f"{'='*70}")
    print(f"Trial folder: {trial_folder}")
    print(f"Raw videos:   {raw_dir}")
    print(f"Output dir:   {out_dir}")
    print(f"Target FPS:   {target_fps}")
    
    # =========================================================================
    # STEP 1: Load all timestamps
    # =========================================================================
    print(f"\n[1] Loading timestamps...")
    
    ts_files = list(trial_folder.glob("camera_*_timestamps.npy"))
    if not ts_files:
        raise FileNotFoundError(f"No timestamp files found in {trial_folder}")
    
    cam_re = re.compile(r"camera_(\d+)_timestamps\.npy$")
    timestamps = {}
    
    for ts_file in ts_files:
        match = cam_re.search(ts_file.name)
        if match:
            cam_id = int(match.group(1))
            ts = np.load(ts_file)
            timestamps[cam_id] = ts
            print(f"  Camera {cam_id}: {len(ts)} timestamps, "
                  f"range: [{ts[0]:.3f}, {ts[-1]:.3f}]s")
    
    camera_ids = sorted(timestamps.keys())
    
    # =========================================================================
    # STEP 2: Find global time range across all cameras
    # =========================================================================
    print(f"\n[2] Finding global time range...")
    
    all_start_times = [ts[0] for ts in timestamps.values()]
    all_end_times = [ts[-1] for ts in timestamps.values()]
    
    global_start = max(all_start_times)  # Latest start (common start)
    global_end = min(all_end_times)      # Earliest end (common end)
    
    print(f"  Global start time: {global_start:.3f}s")
    print(f"  Global end time:   {global_end:.3f}s")
    print(f"  Duration:          {global_end - global_start:.3f}s")
    
    # =========================================================================
    # STEP 3: Create ideal timing grid
    # =========================================================================
    print(f"\n[3] Creating ideal {target_fps} FPS timing grid...")
    
    frame_duration = 1.0 / target_fps
    num_ideal_frames = int(np.floor((global_end - global_start) / frame_duration)) + 1
    
    ideal_times = global_start + np.arange(num_ideal_frames) * frame_duration
    
    # Ensure we don't go past global_end
    ideal_times = ideal_times[ideal_times <= global_end]
    num_ideal_frames = len(ideal_times)
    
    print(f"  Ideal frame count: {num_ideal_frames}")
    print(f"  Ideal duration:    {ideal_times[-1] - ideal_times[0]:.3f}s")
    
    # =========================================================================
    # STEP 4: Find video files
    # =========================================================================
    print(f"\n[4] Finding video files...")
    
    video_files = {}
    for cam_id in camera_ids:
        # Try common extensions
        for ext in ['.avi', '.mp4', '.mov']:
            video_path = raw_dir / f"camera_{cam_id}{ext}"
            if video_path.exists():
                video_files[cam_id] = video_path
                print(f"  Camera {cam_id}: {video_path.name}")
                break
        
        if cam_id not in video_files:
            raise FileNotFoundError(f"Video file not found for camera {cam_id}")
    
    # =========================================================================
    # STEP 5: Get video properties (use first camera for output properties)
    # =========================================================================
    print(f"\n[5] Getting video properties...")
    
    ref_cam_id = camera_ids[0]  # Use first camera for resolution reference
    cap_ref = cv2.VideoCapture(str(video_files[ref_cam_id]))
    if not cap_ref.isOpened():
        raise RuntimeError(f"Cannot open camera {ref_cam_id} video")
    
    width_out = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_out = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_ref.release()
    
    print(f"  Output FPS:        {target_fps:.2f}")
    print(f"  Output resolution: {width_out}x{height_out} (from camera {ref_cam_id})")
    print(f"  Output frames:     {num_ideal_frames}")
    
    # =========================================================================
    # STEP 6: Build frame mapping for each camera
    # =========================================================================
    print(f"\n[6] Building frame mappings to ideal clock...")
    
    frame_maps = {}
    sync_metrics = {}
    
    for cam_id in camera_ids:
        # Find nearest frame in this camera for each ideal timepoint
        cam_times = timestamps[cam_id]
        nearest_indices = find_nearest_frame_indices(ideal_times, cam_times)
        
        # Calculate time differences
        matched_times = cam_times[nearest_indices]
        time_diffs = np.abs(matched_times - ideal_times) * 1000  # Convert to ms
        
        frame_maps[cam_id] = nearest_indices
        sync_metrics[cam_id] = {
            'time_diffs_ms': time_diffs,
            'mean_diff_ms': float(np.mean(time_diffs)),
            'max_diff_ms': float(np.max(time_diffs)),
            'rms_diff_ms': float(np.sqrt(np.mean(time_diffs**2))),
            'p95_diff_ms': float(np.percentile(time_diffs, 95)),
            'p99_diff_ms': float(np.percentile(time_diffs, 99))
        }
        
        print(f"  Camera {cam_id}:")
        print(f"    Mean time diff: {sync_metrics[cam_id]['mean_diff_ms']:.2f} ms")
        print(f"    Max time diff:  {sync_metrics[cam_id]['max_diff_ms']:.2f} ms")
        print(f"    RMS time diff:  {sync_metrics[cam_id]['rms_diff_ms']:.2f} ms")
        print(f"    P95 time diff:  {sync_metrics[cam_id]['p95_diff_ms']:.2f} ms")
        
        # Check if any matches exceed threshold
        bad_matches = np.sum(time_diffs > max_time_diff_ms)
        if bad_matches > 0:
            print(f"    ⚠ WARNING: {bad_matches}/{len(time_diffs)} frames exceed {max_time_diff_ms}ms threshold")
    
    # =========================================================================
    # STEP 7: Write synchronized videos
    # =========================================================================
    print(f"\n[7] Writing synchronized videos...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Open all video captures
    captures = {}
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(str(video_files[cam_id]))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for camera {cam_id}")
        captures[cam_id] = cap
    
    # Open all video writers
    writers = {}
    output_paths = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from MJPG to mp4v for MP4 output
    
    for cam_id in camera_ids:
        output_path = out_dir / f"camera_{cam_id}_synchronized.mp4"  # Changed from .avi to .mp4
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            target_fps,  # Use target FPS for all outputs
            (width_out, height_out)  # Use reference resolution for all outputs
        )
        
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer for camera {cam_id}")
        
        writers[cam_id] = writer
        output_paths[cam_id] = output_path
        print(f"  Camera {cam_id}: {output_path.name}")
    
    # Write frame by frame
    print(f"\n[8] Processing {num_ideal_frames} frames...")
    
    try:
        for frame_idx in range(num_ideal_frames):
            # Report progress via callback (used by batch runner for live display)
            if progress_callback is not None:
                progress_callback(frame_idx, num_ideal_frames)
            elif frame_idx % 100 == 0:
                progress = (frame_idx / num_ideal_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_idx}/{num_ideal_frames})", end='\r')
            
            for cam_id in camera_ids:
                # Get the source frame index for this camera
                src_frame_idx = int(frame_maps[cam_id][frame_idx])
                
                # Read frame at specific index
                cap = captures[cam_id]
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    raise RuntimeError(
                        f"Failed to read frame {src_frame_idx} from camera {cam_id} "
                        f"(frame {frame_idx}/{num_ideal_frames} in sync sequence)"
                    )
                
                # Resize to reference resolution if needed
                if frame.shape[1] != width_out or frame.shape[0] != height_out:
                    frame = cv2.resize(frame, (width_out, height_out))
                
                # Write frame
                writers[cam_id].write(frame)
        
        print(f"\n  Progress: 100.0% ({num_ideal_frames}/{num_ideal_frames})")
        print(f"  ✓ All frames written")
        
    finally:
        # Release all resources
        for cap in captures.values():
            cap.release()
        for writer in writers.values():
            writer.release()
    
    # =========================================================================
    # STEP 9: Save frame mapping indices and ideal times
    # =========================================================================
    print(f"\n[9] Saving frame mappings...")
    
    npz_path = trial_folder / "frame_mappings_to_ideal_fps.npz"
    np.savez(
        npz_path,
        target_fps=target_fps,
        frame_count=num_ideal_frames,
        ideal_times=ideal_times,
        global_start=global_start,
        global_end=global_end,
        **{f"cam_{cam_id}_indices": frame_maps[cam_id] for cam_id in camera_ids}
    )
    print(f"  Saved: {npz_path}")
    
    # =========================================================================
    # STEP 10: Verify output videos
    # =========================================================================
    print(f"\n[10] Verifying output videos...")
    
    verification = {}
    all_match = True
    
    for cam_id in camera_ids:
        cap = cv2.VideoCapture(str(output_paths[cam_id]))
        out_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = cap.get(cv2.CAP_PROP_FPS)
        out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        matches_expected = (out_frame_count == num_ideal_frames)
        verification[cam_id] = {
            'frame_count': out_frame_count,
            'fps': out_fps,
            'resolution': (out_width, out_height),
            'matches_ideal': matches_expected
        }
        
        status = "✓" if matches_expected else "✗"
        print(f"  Camera {cam_id}: {status} {out_frame_count} frames "
              f"(expected {num_ideal_frames}), {out_fps:.2f} fps, {out_width}x{out_height}")
        
        if not matches_expected:
            all_match = False
    
    if all_match:
        print(f"\n  ✓ All videos have matching frame counts!")
    else:
        print(f"\n  ✗ WARNING: Some videos don't match expected frame count")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Synchronization Complete")
    print(f"{'='*70}")
    print(f"Synchronized to: Ideal {target_fps} FPS clock")
    print(f"Output frames: {num_ideal_frames}")
    print(f"Output FPS: {target_fps:.2f}")
    print(f"Output resolution: {width_out}x{height_out}")
    print(f"Time range: {global_start:.3f}s to {global_end:.3f}s")
    print(f"Duration: {global_end - global_start:.3f}s")
    print(f"Synchronized videos saved to: {out_dir}")
    print(f"{'='*70}\n")
    
    # Return results
    results = {
        'trial_folder': str(trial_folder),
        'sync_method': 'ideal_fps',
        'target_fps': target_fps,
        'frame_count': num_ideal_frames,
        'resolution': (width_out, height_out),
        'camera_ids': camera_ids,
        'ideal_times': ideal_times,
        'global_start': global_start,
        'global_end': global_end,
        'duration': global_end - global_start,
        'output_dir': str(out_dir),
        'output_files': {cam_id: str(output_paths[cam_id]) for cam_id in camera_ids},
        'frame_mappings': frame_maps,
        'sync_metrics': sync_metrics,
        'verification': verification,
        'mapping_file': str(npz_path),
        'timestamps_original': timestamps  # Include original timestamps for plotting
    }
    
    return results


# ============================================================================
# USAGE EXAMPLE - Paste into notebook cell:
# ============================================================================
if __name__ == "__main__":
    # Example usage
    trial_dir = "./recordings/2026-02-04_12-00-00"  # Change this to your trial folder
    
    results = synchronize_videos_to_ideal_fps(
        trial_folder=trial_dir,
        target_fps=30.0,  # Your target FPS (e.g., 30.0, 60.0, etc.)
        max_time_diff_ms=50.0  # Maximum allowed time difference (ms)
    )
    
    # Check results
    print("\nSynchronization metrics:")
    for cam_id, metrics in results['sync_metrics'].items():
        print(f"\nCamera {cam_id} vs Ideal {results['target_fps']} FPS:")
        print(f"  Mean time difference: {metrics['mean_diff_ms']:.2f} ms")
        print(f"  RMS time difference:  {metrics['rms_diff_ms']:.2f} ms")
        print(f"  Max time difference:  {metrics['max_diff_ms']:.2f} ms")