"""
Complete Demo: Multi-Camera Recording and Synchronization
Demonstrates the full workflow for neuroscience experiments
"""

import time
import logging
from pathlib import Path
from multicam_recorder import MultiCameraRecorder
from video_synchronizer import VideoSynchronizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_basic_workflow():
    """
    Basic workflow demonstration:
    1. Connect to cameras
    2. Record a single trial
    3. Synchronize offline
    """
    print("\n" + "="*60)
    print("BASIC WORKFLOW DEMO")
    print("="*60)
    
    # Step 1: Initialize recorder
    print("\n[1] Initializing multi-camera recorder...")
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1],  # Adjust based on your cameras
        width=1280,
        height=720,
        fps=30,
        base_output_dir="./demo_recordings"
    )
    
    # Step 2: Connect to cameras
    print("\n[2] Connecting to cameras...")
    connection_results = recorder.connect_cameras()
    
    for cam_id, success in connection_results.items():
        status = "✓ Connected" if success else "✗ Failed"
        print(f"    Camera {cam_id}: {status}")
    
    if not any(connection_results.values()):
        print("\n⚠ No cameras connected! Exiting demo.")
        return
    
    # Step 3: Preview cameras
    print("\n[3] Showing camera preview for 3 seconds...")
    print("    (Press 'q' to skip preview)")
    recorder.preview_cameras(duration=3.0)
    
    # Step 4: Record a trial
    print("\n[4] Recording trial for 5 seconds...")
    print("    👏 Make some noise or movement for synchronization!")
    
    recorder.start_recording(trial_name="demo_trial_001")
    time.sleep(5.0)  # Record for 5 seconds
    stats = recorder.stop_recording()
    
    # Print recording stats
    print("\n    Recording statistics:")
    for cam_id, (frame_count, timestamps) in stats.items():
        print(f"      Camera {cam_id}: {frame_count} frames")
    
    # Step 5: Disconnect cameras
    print("\n[5] Disconnecting cameras...")
    recorder.disconnect_cameras()
    
    # Step 6: Synchronize offline
    print("\n[6] Synchronizing videos offline...")
    synchronizer = VideoSynchronizer()
    
    video_dir = Path("./demo_recordings/demo_trial_001/raw_videos")
    results = synchronizer.synchronize_videos(
        video_dir=video_dir,
        method='audio'
    )
    
    if results:
        print("\n    ✓ Synchronization complete!")
        print(f"    Synchronized videos saved to: {video_dir.parent / 'synchronized_videos'}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)


def demo_multiple_trials():
    """
    Advanced workflow demonstration:
    Record multiple trials in sequence, then synchronize all
    """
    print("\n" + "="*60)
    print("MULTIPLE TRIALS DEMO")
    print("="*60)
    
    # Initialize recorder
    print("\n[1] Initializing multi-camera recorder...")
    recorder = MultiCameraRecorder(
        camera_ids=[0, 1],
        width=1280,
        height=720,
        fps=30,
        base_output_dir="./experiment_recordings"
    )
    
    # Connect to cameras ONCE
    print("\n[2] Connecting to cameras...")
    connection_results = recorder.connect_cameras()
    
    connected_cameras = sum(connection_results.values())
    print(f"    Connected to {connected_cameras} cameras")
    
    if connected_cameras == 0:
        print("\n⚠ No cameras connected! Exiting demo.")
        return
    
    # Record multiple trials
    num_trials = 3
    trial_duration = 3.0  # seconds
    
    print(f"\n[3] Recording {num_trials} trials...")
    print(f"    Each trial will record for {trial_duration} seconds")
    print("    👏 Make distinct sounds/movements in each trial!\n")
    
    trial_dirs = []
    
    for trial_num in range(1, num_trials + 1):
        trial_name = f"trial_{trial_num:03d}"
        
        print(f"\n  Trial {trial_num}/{num_trials}: {trial_name}")
        print("    Press ENTER to start recording...")
        input()
        
        # Start recording
        print(f"    🔴 Recording...")
        recorder.start_recording(trial_name=trial_name)
        
        # Record for specified duration
        time.sleep(trial_duration)
        
        # Stop recording
        stats = recorder.stop_recording()
        print(f"    ⏹ Stopped")
        
        # Save trial directory
        trial_dirs.append(recorder.current_trial_dir)
        
        # Brief pause between trials
        if trial_num < num_trials:
            print("    Taking 2 second break...")
            time.sleep(2.0)
    
    # Disconnect cameras
    print("\n[4] Disconnecting cameras...")
    recorder.disconnect_cameras()
    
    # Synchronize all trials
    print(f"\n[5] Synchronizing all {num_trials} trials...")
    synchronizer = VideoSynchronizer()
    
    for i, trial_dir in enumerate(trial_dirs, 1):
        print(f"\n  Synchronizing trial {i}/{num_trials}...")
        results = synchronizer.synchronize_videos(
            video_dir=trial_dir,
            method='audio'
        )
        
        if results:
            print(f"    ✓ Trial {i} synchronized")
    
    print("\n" + "="*60)
    print("MULTIPLE TRIALS DEMO COMPLETE")
    print(f"All recordings saved to: ./experiment_recordings/")
    print("="*60)


def demo_custom_workflow():
    """
    Custom workflow showing full control over recording process
    """
    print("\n" + "="*60)
    print("CUSTOM WORKFLOW DEMO")
    print("="*60)
    
    # Initialize
    recorder = MultiCameraRecorder(
        camera_ids=[0],  # Single camera for demo
        width=640,
        height=480,
        fps=30,
        base_output_dir="./custom_recordings"
    )
    
    # Connect
    print("\nConnecting to cameras...")
    recorder.connect_cameras()
    
    try:
        # Interactive loop
        while True:
            print("\n" + "-"*60)
            print("Commands:")
            print("  [r] Start recording")
            print("  [s] Stop recording")
            print("  [p] Preview cameras")
            print("  [q] Quit and synchronize")
            print("-"*60)
            
            command = input("Enter command: ").strip().lower()
            
            if command == 'r':
                trial_name = input("Enter trial name (or press ENTER for auto): ").strip()
                trial_name = trial_name if trial_name else None
                
                if recorder.start_recording(trial_name):
                    print("🔴 Recording started")
                
            elif command == 's':
                if recorder.is_recording:
                    stats = recorder.stop_recording()
                    print("⏹ Recording stopped")
                    
                    for cam_id, (frame_count, _) in stats.items():
                        print(f"  Camera {cam_id}: {frame_count} frames")
                else:
                    print("Not currently recording")
            
            elif command == 'p':
                print("Showing preview (press 'q' to exit)...")
                recorder.preview_cameras(duration=10.0)
            
            elif command == 'q':
                break
            
            else:
                print("Invalid command")
    
    finally:
        # Cleanup
        print("\nDisconnecting cameras...")
        recorder.disconnect_cameras()
        
        # Ask if user wants to synchronize
        sync = input("\nSynchronize recorded videos? (y/n): ").strip().lower()
        
        if sync == 'y':
            print("\nSynchronizing all recordings...")
            synchronizer = VideoSynchronizer()
            
            # Find all trial directories
            base_dir = Path("./custom_recordings")
            for trial_dir in base_dir.iterdir():
                if trial_dir.is_dir():
                    raw_videos = trial_dir / "raw_videos"
                    if raw_videos.exists():
                        print(f"\nSynchronizing {trial_dir.name}...")
                        synchronizer.synchronize_videos(raw_videos, method='audio')
    
    print("\n" + "="*60)
    print("CUSTOM WORKFLOW COMPLETE")
    print("="*60)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Multi-Camera Recording & Synchronization Demo              ║
║  Based on FreeMoCap/SkellyCam Architecture                  ║
╚══════════════════════════════════════════════════════════════╝

This demo shows how to:
1. Connect to multiple webcams
2. Record synchronized videos in trials
3. Synchronize videos OFFLINE using audio cross-correlation

Choose a demo:
  [1] Basic workflow (single trial)
  [2] Multiple trials (neuroscience experiment simulation)
  [3] Custom interactive workflow
  [q] Quit
""")
    
    choice = input("Enter your choice: ").strip()
    
    if choice == '1':
        demo_basic_workflow()
    elif choice == '2':
        demo_multiple_trials()
    elif choice == '3':
        demo_custom_workflow()
    elif choice == 'q':
        print("\nExiting...")
    else:
        print("\nInvalid choice!")
