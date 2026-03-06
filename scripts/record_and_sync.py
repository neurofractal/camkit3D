#!/usr/bin/env python3
"""
Multi-camera recording and synchronization script.
Records from multiple cameras, synchronizes them, and outputs metrics/plots.
"""

import argparse
import time
import sys
from pathlib import Path

from camkit3d.recorder import MultiCamRecorder
from camkit3d.sync import synchronize_videos_to_ideal_fps, plot_sync_results, plot_sync_summary_stats


def countdown(seconds):
    """Display a countdown timer."""
    print(f"\nRecording will start in {seconds} seconds...")
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    print("  Recording started!    ")


def main():
    parser = argparse.ArgumentParser(
        description="Record from multiple cameras, synchronize, and generate metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python record_and_sync.py 10           # Record for 10 seconds
  python record_and_sync.py 30 --fps 60  # Record for 30 seconds at 60 FPS
  python record_and_sync.py 5 --cameras 0 1  # Use only cameras 0 and 1
        """,
    )

    parser.add_argument("duration", type=float, help="Recording duration in seconds")
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=[0, 1, 2],
        help="Camera IDs to use (default: 0 1 2)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Recording frame rate (default: 30)")
    parser.add_argument("--width", type=int, default=1280, help="Camera resolution width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera resolution height (default: 720)")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path.home() / "Documents" / "recordings"),
        help="Output directory for recordings (default: ~/Documents/recordings)",
    )
    parser.add_argument(
        "--countdown", type=int, default=3,
        help="Countdown before recording starts in seconds (default: 3)",
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 60)
    print("MULTI-CAMERA RECORDING AND SYNCHRONIZATION")
    print("=" * 60)
    print(f"Duration:     {args.duration:.1f} seconds")
    print(f"Cameras:      {args.cameras}")
    print(f"Resolution:   {args.width}x{args.height}")
    print(f"FPS:          {args.fps}")
    print(f"Output dir:   {args.output_dir}")
    print("=" * 60 + "\n")

    try:
        # 1. Setup recorder
        print("Setting up multi-camera recorder...")
        recorder = MultiCamRecorder(
            camera_ids=args.cameras,
            width=args.width,
            height=args.height,
            fps=args.fps,
            base_output_dir=args.output_dir,
        )

        # 2. Connect to cameras
        print("Connecting to cameras...")
        connection_status = recorder.connect_cameras()

        connected = [cid for cid, ok in connection_status.items() if ok]
        if len(connected) < len(args.cameras):
            print(f"⚠ Warning: Only {len(connected)}/{len(args.cameras)} cameras connected")
            if not connected:
                print("✗ No cameras connected. Exiting.")
                return 1

        # 3. Countdown
        if args.countdown > 0:
            countdown(args.countdown)

        # 4. Start recording
        start_time = time.time()
        recorder.start_recording()
        trial_dir = recorder.base_output_dir / recorder.current_trial_name

        # 5. Record for specified duration
        print(f"\n🔴 Recording for {args.duration:.1f} seconds...")
        elapsed = 0
        while elapsed < args.duration:
            time.sleep(0.1)
            elapsed = time.time() - start_time
            remaining = args.duration - elapsed
            print(f"  Time remaining: {remaining:.1f}s", end="\r")
        print()

        # 6. Stop recording
        print("\nStopping recording...")
        stats = recorder.stop_recording()

        print(f"\n✓ Recording complete!")
        print(f"  Saved to: {trial_dir}")
        print(f"  Frame counts: {stats}")

        # 7. Release cameras
        print("\nReleasing cameras...")
        recorder.disconnect_cameras()

        # 8. Synchronize videos
        print("\n" + "=" * 60)
        print("SYNCHRONIZING VIDEOS")
        print("=" * 60 + "\n")

        results = synchronize_videos_to_ideal_fps(
            trial_folder=trial_dir,
            target_fps=args.fps,
        )

        # 9. Generate plots and metrics
        print("\n" + "=" * 60)
        print("GENERATING SYNCHRONIZATION PLOTS AND METRICS")
        print("=" * 60 + "\n")

        plot_sync_results(
            results=results,
            trial_folder=trial_dir,
            save_plots=True,
            show_plots=False,
        )

        plot_sync_summary_stats(
            results=results,
            trial_folder=trial_dir,
            save_plots=True,
        )

        print("✓ Plots and metrics saved")

        # Final summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Recording folder:     {trial_dir}")
        print(f"Synchronized videos:  {Path(trial_dir) / 'synchronized_videos'}")
        print(f"Plots:                {Path(trial_dir) / 'synchronization_plots'}")
        print("=" * 60 + "\n")

        print("✓ All done!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Recording interrupted by user")
        if "recorder" in locals():
            recorder.stop_recording()
            recorder.disconnect_cameras()
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        if "recorder" in locals():
            try:
                recorder.disconnect_cameras()
            except Exception:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
