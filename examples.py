"""
Complete Example: All Features Demonstration
Shows every capability of the system
"""

from multicam_recorder import MultiCameraRecorder
from video_synchronizer import VideoSynchronizer
from post_processing import VideoAnalyzer
import time
from pathlib import Path


def example_1_basic_single_trial():
    """
    Example 1: Record and synchronize a single trial
    Most basic use case
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Single Trial")
    print("="*70)
    
    # Initialize
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1],
        width=1280,
        height=720,
        fps=30,
        base_output_dir="./examples/example1"
    )
    
    # Connect
    recorder.connect_cameras()
    
    # Record
    recorder.start_recording("my_first_trial")
    print("Recording for 5 seconds...")
    time.sleep(5)
    recorder.stop_recording()
    
    # Disconnect
    recorder.disconnect_cameras()
    
    # Synchronize
    sync = VideoSynchronizer()
    sync.synchronize_videos(
        video_dir="./examples/example1/my_first_trial/raw_videos",
        method='audio'
    )
    
    print("\n✓ Example 1 complete!")


def example_2_multiple_sequential_trials():
    """
    Example 2: Record multiple trials in sequence
    Typical neuroscience experiment workflow
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Sequential Trials")
    print("="*70)
    
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1],
        base_output_dir="./examples/example2"
    )
    
    # Connect once
    recorder.connect_cameras()
    
    # Record 3 trials
    trial_names = []
    for i in range(1, 4):
        trial_name = f"trial_{i:03d}"
        
        print(f"\nTrial {i}/3")
        input("Press ENTER to start...")
        
        recorder.start_recording(trial_name)
        time.sleep(3)  # 3 second recording
        recorder.stop_recording()
        
        trial_names.append(trial_name)
        
        # Inter-trial interval
        if i < 3:
            time.sleep(2)
    
    # Disconnect once
    recorder.disconnect_cameras()
    
    # Synchronize all trials
    sync = VideoSynchronizer()
    for trial_name in trial_names:
        video_dir = f"./examples/example2/{trial_name}/raw_videos"
        sync.synchronize_videos(video_dir, method='audio')
    
    print("\n✓ Example 2 complete!")


def example_3_with_preview():
    """
    Example 3: Using camera preview
    Good for setup and alignment
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Camera Preview")
    print("="*70)
    
    recorder = MultiCameraRecorder(
        camera_ids=[0],
        base_output_dir="./examples/example3"
    )
    
    recorder.connect_cameras()
    
    # Show preview before recording
    print("\nShowing preview for 3 seconds (press 'q' to skip)...")
    recorder.preview_cameras(duration=3.0)
    
    # Record
    recorder.start_recording("preview_test")
    time.sleep(5)
    recorder.stop_recording()
    
    recorder.disconnect_cameras()
    
    print("\n✓ Example 3 complete!")


def example_4_brightness_sync():
    """
    Example 4: Brightness-based synchronization
    Alternative to audio sync
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Brightness Synchronization")
    print("="*70)
    
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1],
        base_output_dir="./examples/example4"
    )
    
    recorder.connect_cameras()
    
    print("\n💡 IMPORTANT: Trigger a bright flash at the START of recording!")
    print("   (e.g., camera flash, turn on lights)")
    input("\nReady? Press ENTER to start recording...")
    
    recorder.start_recording("brightness_test")
    time.sleep(5)
    recorder.stop_recording()
    
    recorder.disconnect_cameras()
    
    # Synchronize using brightness method
    sync = VideoSynchronizer()
    sync.synchronize_videos(
        video_dir="./examples/example4/brightness_test/raw_videos",
        method='brightness'
    )
    
    print("\n✓ Example 4 complete!")


def example_5_custom_resolution_fps():
    """
    Example 5: Custom resolution and frame rate
    High-quality recording settings
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Resolution & FPS")
    print("="*70)
    
    # High-quality settings
    recorder = MultiCameraRecorder(
        camera_ids=[0],
        width=1920,   # Full HD
        height=1080,
        fps=60,       # High frame rate
        base_output_dir="./examples/example5"
    )
    
    recorder.connect_cameras()
    
    recorder.start_recording("high_quality_test")
    print("Recording high-quality video for 5 seconds...")
    time.sleep(5)
    stats = recorder.stop_recording()
    
    # Print recording info
    for cam_id, (frame_count, timestamps) in stats.items():
        print(f"\nCamera {cam_id}:")
        print(f"  Frames: {frame_count}")
        print(f"  Expected: ~{60 * 5} frames at 60fps")
    
    recorder.disconnect_cameras()
    
    print("\n✓ Example 5 complete!")


def example_6_quality_analysis():
    """
    Example 6: Post-recording quality analysis
    Verify synchronization quality
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Quality Analysis")
    print("="*70)
    
    # First, record and sync
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1],
        base_output_dir="./examples/example6"
    )
    
    recorder.connect_cameras()
    recorder.start_recording("analysis_test")
    time.sleep(5)
    recorder.stop_recording()
    recorder.disconnect_cameras()
    
    # Synchronize
    sync = VideoSynchronizer()
    sync.synchronize_videos(
        video_dir="./examples/example6/analysis_test/raw_videos",
        method='audio'
    )
    
    # Analyze
    analyzer = VideoAnalyzer()
    
    print("\n--- Quality Analysis ---")
    results = analyzer.analyze_synchronization(
        video_dir="./examples/example6/analysis_test/synchronized_videos"
    )
    
    # Create comparison video
    print("\nCreating side-by-side comparison...")
    video_files = sorted(
        Path("./examples/example6/analysis_test/synchronized_videos").glob("*.mp4")
    )
    
    if len(video_files) >= 2:
        analyzer.create_side_by_side_comparison(
            video_files,
            Path("./examples/example6/analysis_test/synchronized_videos/comparison.mp4"),
            duration=3.0
        )
    
    print("\n✓ Example 6 complete!")


def example_7_metadata_timestamps():
    """
    Example 7: Accessing metadata and timestamps
    Useful for detailed analysis
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Metadata & Timestamps")
    print("="*70)
    
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1],
        base_output_dir="./examples/example7"
    )
    
    recorder.connect_cameras()
    recorder.start_recording("metadata_test")
    time.sleep(5)
    stats = recorder.stop_recording()
    recorder.disconnect_cameras()
    
    # Access timestamp data
    import numpy as np
    
    print("\n--- Recording Statistics ---")
    for cam_id, (frame_count, timestamps) in stats.items():
        print(f"\nCamera {cam_id}:")
        print(f"  Total frames: {frame_count}")
        
        if timestamps:
            timestamps = np.array(timestamps)
            duration = timestamps[-1] - timestamps[0]
            actual_fps = frame_count / duration if duration > 0 else 0
            
            print(f"  Duration: {duration:.3f}s")
            print(f"  Actual FPS: {actual_fps:.2f}")
            print(f"  First frame: {timestamps[0]:.3f}")
            print(f"  Last frame: {timestamps[-1]:.3f}")
    
    # Check saved files
    trial_dir = Path("./examples/example7/metadata_test")
    print(f"\n--- Saved Files ---")
    print(f"Metadata file: {trial_dir / 'metadata.txt'}")
    
    for cam_id in stats.keys():
        timestamp_file = trial_dir / f"camera_{cam_id}_timestamps.npy"
        if timestamp_file.exists():
            print(f"Timestamps (camera {cam_id}): {timestamp_file}")
    
    print("\n✓ Example 7 complete!")


def example_8_error_handling():
    """
    Example 8: Proper error handling
    How to handle common issues
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Error Handling")
    print("="*70)
    
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1, 99],  # 99 probably doesn't exist
        base_output_dir="./examples/example8"
    )
    
    # Connect and handle failures
    print("\nAttempting to connect to cameras (some may fail)...")
    results = recorder.connect_cameras()
    
    for cam_id, success in results.items():
        if success:
            print(f"  ✓ Camera {cam_id} connected")
        else:
            print(f"  ✗ Camera {cam_id} failed to connect")
    
    # Check if any cameras connected
    if not any(results.values()):
        print("\n⚠ No cameras connected - cannot record")
        return
    
    # Try to record (will work with available cameras)
    try:
        recorder.start_recording("error_test")
        time.sleep(3)
        recorder.stop_recording()
        print("\n✓ Recording succeeded with available cameras")
    except Exception as e:
        print(f"\n✗ Recording error: {e}")
    finally:
        recorder.disconnect_cameras()
    
    print("\n✓ Example 8 complete!")


def main():
    """Run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Complete Feature Examples                          ║
║     Multi-Camera Synchronized Recording System              ║
╚══════════════════════════════════════════════════════════════╝

This script demonstrates all features of the system.
Each example is independent and shows a different use case.

Available Examples:
  [1] Basic single trial
  [2] Multiple sequential trials (neuroscience workflow)
  [3] Camera preview
  [4] Brightness-based synchronization
  [5] Custom resolution & FPS
  [6] Quality analysis
  [7] Metadata & timestamps
  [8] Error handling
  [9] Run all examples

Choose an example (or 'q' to quit): """)
    
    examples = {
        '1': example_1_basic_single_trial,
        '2': example_2_multiple_sequential_trials,
        '3': example_3_with_preview,
        '4': example_4_brightness_sync,
        '5': example_5_custom_resolution_fps,
        '6': example_6_quality_analysis,
        '7': example_7_metadata_timestamps,
        '8': example_8_error_handling,
    }
    
    while True:
        choice = input().strip()
        
        if choice == 'q':
            break
        elif choice == '9':
            # Run all
            for example_func in examples.values():
                try:
                    example_func()
                except KeyboardInterrupt:
                    print("\n\nSkipping to next example...")
                except Exception as e:
                    print(f"\n✗ Example error: {e}")
                    print("Continuing to next example...")
            break
        elif choice in examples:
            try:
                examples[choice]()
            except Exception as e:
                print(f"\n✗ Error: {e}")
            break
        else:
            print("Invalid choice. Try again (or 'q' to quit): ")


if __name__ == "__main__":
    main()
