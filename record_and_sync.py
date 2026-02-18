#!/usr/bin/env python3
"""
Multi-camera recording and synchronization script.
Records from multiple cameras, synchronizes them, outputs metrics, and creates a side-by-side video.
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from multicam_recorder import MultiCameraRecorder
from sync_by_timestamps import synchronize_videos_to_ideal_fps
from plot_sync_results import plot_sync_results, plot_sync_summary_stats


def countdown(seconds):
    """Display a countdown timer."""
    print(f"\nRecording will start in {seconds} seconds...")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end='\r')
        time.sleep(1)
    print("  Recording started!    ")


def save_synced_videos_side_by_side(
    trial_folder,
    synced_subdir="synchronized_videos",
    out_name="synchronized_side_by_side.mp4",
    max_width=1920,
):
    """
    Combine synchronized videos into a single side-by-side video for offline review.
    
    Args:
        trial_folder: Path to the recording trial folder
        synced_subdir: Subdirectory containing synchronized videos
        out_name: Output filename for the combined video
        max_width: Maximum width for the output video (will downscale if needed)
    
    Returns:
        Path to the output video
    """
    trial_folder = Path(trial_folder)
    synced_dir = trial_folder / synced_subdir
    out_path = trial_folder / out_name

    videos = sorted(synced_dir.glob("camera_*_synchronized.mp4"))
    if not videos:
        # Fallback to .avi if no .mp4 files found
        videos = sorted(synced_dir.glob("camera_*_synchronized.avi"))
    if not videos:
        raise FileNotFoundError(f"No synchronized videos found in {synced_dir}")

    caps = []
    cam_ids = []

    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {v}")
        caps.append(cap)
        cam_ids.append(int(v.stem.split("_")[1]))

    # Check frame counts
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    if len(set(frame_counts)) != 1:
        raise ValueError(f"Frame count mismatch: {dict(zip(cam_ids, frame_counts))}")

    fps = caps[0].get(cv2.CAP_PROP_FPS)
    n_frames = frame_counts[0]

    # Read one frame to determine sizes
    ok, first_frame = caps[0].read()
    if not ok:
        raise RuntimeError("Failed to read first frame")
    h, w = first_frame.shape[:2]

    # Reset all captures to frame 0
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tiled_w = w * len(caps)
    tiled_h = h

    # Optional downscale to fit screen width
    scale = 1.0
    if tiled_w > max_width:
        scale = max_width / tiled_w
        tiled_w = int(tiled_w * scale)
        tiled_h = int(tiled_h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or use 'avc1' for H.264
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        fps,
        (tiled_w, tiled_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    print(f"\n{'='*60}")
    print("Creating side-by-side synchronized video:")
    print(f"  Cameras:  {cam_ids}")
    print(f"  Frames:   {n_frames}")
    print(f"  FPS:      {fps:.2f}")
    print(f"  Output:   {out_path}")
    print(f"{'='*60}\n")

    # Write frames
    for i in range(n_frames):
        frames = []
        for cam_id, cap in zip(cam_ids, caps):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed reading frame {i} from camera {cam_id}")

            # Overlay text
            cv2.putText(
                frame,
                f"Cam {cam_id} | frame {i}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            frames.append(frame)

        combined = np.hstack(frames)

        if scale != 1.0:
            combined = cv2.resize(
                combined,
                (tiled_w, tiled_h),
                interpolation=cv2.INTER_AREA,
            )

        writer.write(combined)

        # Progress indicator
        if (i + 1) % 30 == 0 or i == n_frames - 1:
            progress = (i + 1) / n_frames * 100
            print(f"  Progress: {progress:.1f}% ({i+1}/{n_frames} frames)", end='\r')

    print()  # New line after progress

    # Cleanup
    writer.release()
    for cap in caps:
        cap.release()

    print("✅ Side-by-side video created successfully!\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Record from multiple cameras, synchronize, and create output video.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python record_and_sync.py 10           # Record for 10 seconds
  python record_and_sync.py 30 --fps 60  # Record for 30 seconds at 60 FPS
  python record_and_sync.py 5 --cameras 0 1  # Use only cameras 0 and 1
        """
    )
    
    parser.add_argument(
        'duration',
        type=float,
        help='Recording duration in seconds'
    )
    parser.add_argument(
        '--cameras',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='Camera IDs to use (default: 0 1 2)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Recording frame rate (default: 30)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Camera resolution width (default: 1280)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Camera resolution height (default: 720)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path.home() / 'Documents' / 'recordings'),
        help='Output directory for recordings (default: ~/Documents/recordings)'
    )
    parser.add_argument(
        '--countdown',
        type=int,
        default=3,
        help='Countdown before recording starts in seconds (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("MULTI-CAMERA RECORDING AND SYNCHRONIZATION")
    print("="*60)
    print(f"Duration:     {args.duration:.1f} seconds")
    print(f"Cameras:      {args.cameras}")
    print(f"Resolution:   {args.width}x{args.height}")
    print(f"FPS:          {args.fps}")
    print(f"Output dir:   {args.output_dir}")
    print("="*60 + "\n")
    
    try:
        # 1. Setup recorder
        print("Setting up multi-camera recorder...")
        recorder = MultiCameraRecorder(
            camera_ids=args.cameras,
            width=args.width,
            height=args.height,
            fps=args.fps,
            base_output_dir=args.output_dir
        )
        
        # 2. Connect to cameras
        print("Connecting to cameras...")
        connection_status = recorder.connect_cameras()
        
        connected = [cam_id for cam_id, status in connection_status.items() if status]
        if len(connected) < len(args.cameras):
            print(f"⚠️  Warning: Only {len(connected)}/{len(args.cameras)} cameras connected")
            if len(connected) == 0:
                print("❌ No cameras connected. Exiting.")
                return 1
        
        # 3. Countdown
        if args.countdown > 0:
            countdown(args.countdown)
        
        # 4. Start recording
        start_time = time.time()
        recorder.start_recording()
        
        # Get the trial directory that was created
        trial_dir = recorder.base_output_dir / recorder.current_trial_name
        
        # 5. Record for specified duration with progress
        print(f"\n🔴 Recording for {args.duration:.1f} seconds...")
        elapsed = 0
        while elapsed < args.duration:
            time.sleep(0.1)
            elapsed = time.time() - start_time
            remaining = args.duration - elapsed
            print(f"  Time remaining: {remaining:.1f}s", end='\r')
        
        print()  # New line after progress
        
        # 6. Stop recording
        print("\nStopping recording...")
        stats = recorder.stop_recording()
        
        print(f"\n✅ Recording complete!")
        print(f"  Saved to: {trial_dir}")
        print(f"  Frame counts: {stats}")
        
        # 7. Release cameras
        print("\nReleasing cameras...")
        recorder.disconnect_cameras()
        
        # 8. Synchronize videos
        print("\n" + "="*60)
        print("SYNCHRONIZING VIDEOS")
        print("="*60 + "\n")
        
        # Note: Individual synchronized videos are created by synchronize_videos_to_ideal_fps()
        # To change their format from .avi to .mp4, modify the sync_by_timestamps.py module
        results = synchronize_videos_to_ideal_fps(
            trial_folder=trial_dir,
            target_fps=args.fps
        )
        
        # 9. Generate plots and metrics
        print("\n" + "="*60)
        print("GENERATING SYNCHRONIZATION PLOTS AND METRICS")
        print("="*60 + "\n")
        
        figs = plot_sync_results(
            results=results,
            trial_folder=trial_dir,
            save_plots=True,
            show_plots=False
        )
        
        fig_summary = plot_sync_summary_stats(
            results=results,
            trial_folder=trial_dir,
            save_plots=True
        )
        
        print("✅ Plots and metrics saved")
        
        # 10. Create side-by-side video
        print("\n" + "="*60)
        print("CREATING COMBINED VIDEO")
        print("="*60)
        
        out_video = save_synced_videos_side_by_side(trial_dir)
        
        # Final summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Recording folder:     {trial_dir}")
        print(f"Synchronized videos:  {Path(trial_dir) / 'synchronized_videos'}")
        print(f"Combined video:       {out_video}")
        print(f"Plots:                {Path(trial_dir) / 'plots'}")
        print("="*60 + "\n")
        
        print("✅ All done!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Recording interrupted by user")
        if 'recorder' in locals():
            recorder.stop_recording()
            recorder.disconnect_cameras()
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if 'recorder' in locals():
            try:
                recorder.disconnect_cameras()
            except:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())