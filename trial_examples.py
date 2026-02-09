"""
Trial-Based Recording Examples
Record multiple trials without reconnecting to cameras between trials
"""

from multicam_recorder_ffmpeg import FFmpegMultiCamRecorder
import time


# ============================================================================
# EXAMPLE 1: Simple multiple trials
# ============================================================================
def example_simple_trials():
    """
    Record 5 trials of 10 seconds each with 2 second pauses between trials
    """
    print("Example 1: Simple Multiple Trials")
    print("-" * 70)
    
    with FFmpegMultiCamRecorder(
        camera_ids=[0, 1, 2, 3],
        output_dir="trial_recordings",
        framerate=30,
        resolution="1920x1080"
    ) as recorder:
        
        # Record 5 trials, 10 seconds each, 2 second pause between
        recorder.record_trials(
            num_trials=5,
            trial_duration=10,
            inter_trial_interval=2.0,
            session_name="experiment_001"
        )
    
    print("\nDone! Check the 'trial_recordings' folder")
    print("Files will be named: experiment_001_trial001_cam0.mp4, etc.")


# ============================================================================
# EXAMPLE 2: Manual trial control (for custom logic)
# ============================================================================
def example_manual_trial_control():
    """
    Manually control each trial - useful when you need custom logic between trials
    """
    print("\nExample 2: Manual Trial Control")
    print("-" * 70)
    
    with FFmpegMultiCamRecorder(
        camera_ids=[0, 1],
        output_dir="manual_trials",
        framerate=30,
        resolution="1280x720"
    ) as recorder:
        
        session_name = "manual_session"
        
        for trial_num in range(1, 4):  # 3 trials
            print(f"\n--- Starting Trial {trial_num} ---")
            
            # Your custom pre-trial logic here
            # (e.g., present stimulus, wait for subject ready, etc.)
            print("Preparing trial...")
            time.sleep(1)
            
            # Start recording
            recorder.start_recording(
                session_name=session_name,
                trial_number=trial_num
            )
            
            # Record for trial duration
            print(f"Recording trial {trial_num}...")
            time.sleep(5)  # 5 second trial
            
            # Stop recording
            output_files = recorder.stop_recording()
            
            # Your custom post-trial logic here
            # (e.g., save trial metadata, ask for ratings, etc.)
            print(f"Trial {trial_num} complete!")
            print(f"Files: {[f.name for f in output_files.values()]}")
            
            # Pause between trials
            if trial_num < 3:
                print("Inter-trial interval...")
                time.sleep(2)
    
    print("\nAll trials complete!")


# ============================================================================
# EXAMPLE 3: Event-triggered trials
# ============================================================================
def example_event_triggered_trials():
    """
    Start each trial based on events (e.g., button press, sensor trigger)
    """
    print("\nExample 3: Event-Triggered Trials")
    print("-" * 70)
    
    def wait_for_trigger():
        """Simulate waiting for a trigger event (e.g., button press)"""
        print("  Waiting for trigger... (simulating with 1s delay)")
        time.sleep(1)
        return True
    
    with FFmpegMultiCamRecorder(
        camera_ids=[0, 1],
        output_dir="event_trials",
        framerate=30,
        resolution="1920x1080"
    ) as recorder:
        
        session_name = "event_session"
        trial_num = 1
        max_trials = 3
        
        print(f"Ready to record up to {max_trials} trials")
        print("Waiting for triggers...\n")
        
        while trial_num <= max_trials:
            # Wait for external trigger
            if wait_for_trigger():
                print(f"\n>>> Trigger received! Starting trial {trial_num}")
                
                # Start recording
                recorder.start_recording(
                    session_name=session_name,
                    trial_number=trial_num
                )
                
                # Record for fixed duration or until next trigger
                time.sleep(5)
                
                # Stop recording
                output_files = recorder.stop_recording()
                print(f">>> Trial {trial_num} saved: {len(output_files)} cameras")
                
                trial_num += 1
    
    print("\nAll event-triggered trials complete!")


# ============================================================================
# EXAMPLE 4: Variable duration trials
# ============================================================================
def example_variable_duration_trials():
    """
    Record trials with different durations
    """
    print("\nExample 4: Variable Duration Trials")
    print("-" * 70)
    
    # Different duration for each trial
    trial_durations = [5, 10, 7, 12, 8]  # seconds
    
    with FFmpegMultiCamRecorder(
        camera_ids=[0, 1, 2],
        output_dir="variable_trials",
        framerate=30,
        resolution="1920x1080"
    ) as recorder:
        
        session_name = "variable_duration"
        
        for trial_num, duration in enumerate(trial_durations, start=1):
            print(f"\n--- Trial {trial_num}: {duration} seconds ---")
            
            # Start recording
            recorder.start_recording(
                session_name=session_name,
                trial_number=trial_num
            )
            
            # Record for this trial's specific duration
            print(f"Recording for {duration} seconds...")
            time.sleep(duration)
            
            # Stop recording
            recorder.stop_recording()
            
            # Brief pause between trials
            if trial_num < len(trial_durations):
                time.sleep(1)
    
    print("\nVariable duration trials complete!")


# ============================================================================
# EXAMPLE 5: Real experiment workflow
# ============================================================================
def example_real_experiment():
    """
    Realistic experimental workflow with multiple blocks and trials
    """
    print("\nExample 5: Real Experiment Workflow")
    print("-" * 70)
    
    # Experiment parameters
    num_blocks = 2
    trials_per_block = 3
    trial_duration = 5
    inter_trial_interval = 2
    inter_block_interval = 10
    
    with FFmpegMultiCamRecorder(
        camera_ids=[0, 1],
        output_dir="experiment_data",
        framerate=30,
        resolution="1920x1080"
    ) as recorder:
        
        overall_trial = 1
        
        for block in range(1, num_blocks + 1):
            print(f"\n{'='*70}")
            print(f"BLOCK {block}/{num_blocks}")
            print(f"{'='*70}")
            
            session_name = f"experiment_block{block}"
            
            for trial in range(1, trials_per_block + 1):
                print(f"\nBlock {block}, Trial {trial}/{trials_per_block} (Overall: {overall_trial})")
                
                # Pre-trial setup
                print("  Setting up trial...")
                time.sleep(0.5)
                
                # Record trial
                recorder.start_recording(
                    session_name=session_name,
                    trial_number=trial
                )
                
                print(f"  Recording ({trial_duration}s)...")
                time.sleep(trial_duration)
                
                output_files = recorder.stop_recording()
                print(f"  ✓ Trial saved: {list(output_files.values())[0].name}")
                
                # Inter-trial interval
                if trial < trials_per_block:
                    print(f"  Pause: {inter_trial_interval}s")
                    time.sleep(inter_trial_interval)
                
                overall_trial += 1
            
            # Inter-block interval
            if block < num_blocks:
                print(f"\n--- Inter-block break: {inter_block_interval}s ---")
                time.sleep(inter_block_interval)
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE!")
        print(f"{'='*70}")
        print(f"Total trials: {overall_trial - 1}")
        print(f"Total blocks: {num_blocks}")


# ============================================================================
# EXAMPLE 6: Quick test - just 3 short trials
# ============================================================================
def quick_test():
    """
    Quick test with 3 short trials
    """
    print("\nQuick Test: 3 Trials")
    print("-" * 70)
    
    with FFmpegMultiCamRecorder(
        camera_ids=[0, 1],
        output_dir="test_trials",
        framerate=30,
        resolution="1280x720"
    ) as recorder:
        
        recorder.record_trials(
            num_trials=3,
            trial_duration=3,
            inter_trial_interval=1,
            session_name="quick_test"
        )


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Trial-Based Recording Examples")
    print("=" * 70)
    
    # Choose which example to run:
    
    example_simple_trials()          # Easiest way - automatic
    # example_manual_trial_control()   # More control per trial
    # example_event_triggered_trials() # Triggered by events
    # example_variable_duration_trials() # Different durations
    # example_real_experiment()        # Complete experiment workflow
    # quick_test()                     # Quick 3-trial test
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
