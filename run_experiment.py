"""
Experiment Runner - Automated multi-trial recording and synchronization
Uses configuration from config.py
"""

import time
from pathlib import Path
from datetime import datetime
import logging

from multicam_recorder import MultiCameraRecorder
from video_synchronizer import VideoSynchronizer
from post_processing import VideoAnalyzer

# Import configuration
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Automated experiment runner for neuroscience experiments
    
    Features:
    - Automated multi-trial recording
    - Configuration-based setup
    - Automatic synchronization
    - Quality reports
    """
    
    def __init__(self, config_module=config):
        """Initialize from configuration module"""
        self.config = config_module
        
        # Initialize recorder
        self.recorder = MultiCameraRecorder(
            camera_ids=config.CAMERA_IDS,
            width=config.WIDTH,
            height=config.HEIGHT,
            fps=config.FPS,
            base_output_dir=config.OUTPUT_DIR
        )
        
        # Initialize synchronizer
        self.synchronizer = VideoSynchronizer(
            sample_rate=config.AUDIO_SAMPLE_RATE
        )
        
        # Initialize analyzer
        self.analyzer = VideoAnalyzer()
        
        self.trial_names = []
    
    def setup(self):
        """Setup experiment - connect to cameras"""
        logger.info("Setting up experiment...")
        
        print("\n" + "="*70)
        print("EXPERIMENT SETUP")
        print("="*70)
        print(f"Cameras: {self.config.CAMERA_IDS}")
        print(f"Resolution: {self.config.WIDTH}x{self.config.HEIGHT}")
        print(f"FPS: {self.config.FPS}")
        print(f"Trials: {self.config.NUM_TRIALS}")
        print(f"Trial duration: {self.config.TRIAL_DURATION}s")
        print("="*70)
        
        # Connect to cameras
        print("\nConnecting to cameras...")
        results = self.recorder.connect_cameras()
        
        connected = sum(results.values())
        if connected == 0:
            logger.error("No cameras connected!")
            return False
        
        print(f"✓ Connected to {connected}/{len(self.config.CAMERA_IDS)} cameras")
        
        # Preview if enabled
        if self.config.ENABLE_PREVIEW:
            print("\nShowing preview (press 'q' to continue)...")
            self.recorder.preview_cameras(duration=5.0)
        
        return True
    
    def run_trials(self):
        """Run all trials"""
        print("\n" + "="*70)
        print("STARTING TRIALS")
        print("="*70)
        
        for trial_num in range(1, self.config.NUM_TRIALS + 1):
            self._run_single_trial(trial_num)
            
            # Inter-trial interval
            if trial_num < self.config.NUM_TRIALS:
                print(f"\nInter-trial interval: {self.config.ITI}s")
                time.sleep(self.config.ITI)
        
        print("\n" + "="*70)
        print("ALL TRIALS COMPLETE")
        print("="*70)
    
    def _run_single_trial(self, trial_num: int):
        """Run a single trial"""
        # Generate trial name
        if self.config.TRIAL_NAMING == 'sequential':
            trial_name = f"{self.config.TRIAL_PREFIX}_{trial_num:03d}"
        elif self.config.TRIAL_NAMING == 'timestamp':
            trial_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            trial_name = f"trial_{trial_num}"
        
        print(f"\n{'='*70}")
        print(f"TRIAL {trial_num}/{self.config.NUM_TRIALS}: {trial_name}")
        print(f"{'='*70}")
        
        # Ready prompt
        print("\nReady to record...")
        if self.config.SYNC_METHOD == 'audio':
            print("👏 Remember to make a loud clap/sound at the start!")
        elif self.config.SYNC_METHOD == 'brightness':
            print("💡 Remember to trigger the flash at the start!")
        
        input("Press ENTER to start recording...")
        
        # Start recording
        print(f"\n🔴 RECORDING for {self.config.TRIAL_DURATION}s...")
        self.recorder.start_recording(trial_name)
        
        # Show countdown
        remaining = self.config.TRIAL_DURATION
        while remaining > 0:
            if remaining <= 3:
                print(f"  {remaining}...")
            elif remaining % 5 == 0:
                print(f"  {remaining}s remaining...")
            time.sleep(1)
            remaining -= 1
        
        # Stop recording
        stats = self.recorder.stop_recording()
        print("⏹ Recording stopped")
        
        # Print stats
        for cam_id, (frame_count, _) in stats.items():
            print(f"  Camera {cam_id}: {frame_count} frames")
        
        self.trial_names.append(trial_name)
    
    def cleanup(self):
        """Cleanup - disconnect cameras"""
        print("\nDisconnecting cameras...")
        self.recorder.disconnect_cameras()
        print("✓ Cameras disconnected")
    
    def synchronize_all(self):
        """Synchronize all recorded trials"""
        if not self.config.AUTO_SYNC:
            logger.info("Auto-sync disabled, skipping synchronization")
            return
        
        print("\n" + "="*70)
        print("SYNCHRONIZING TRIALS")
        print("="*70)
        
        for i, trial_name in enumerate(self.trial_names, 1):
            print(f"\nSynchronizing trial {i}/{len(self.trial_names)}: {trial_name}")
            
            video_dir = Path(self.config.OUTPUT_DIR) / trial_name / "raw_videos"
            
            if not video_dir.exists():
                logger.warning(f"Video directory not found: {video_dir}")
                continue
            
            try:
                results = self.synchronizer.synchronize_videos(
                    video_dir=video_dir,
                    method=self.config.SYNC_METHOD
                )
                
                if results:
                    print(f"  ✓ Trial {trial_name} synchronized")
                else:
                    print(f"  ✗ Synchronization failed for {trial_name}")
                    
            except Exception as e:
                logger.error(f"Error synchronizing {trial_name}: {e}")
        
        print("\n" + "="*70)
        print("SYNCHRONIZATION COMPLETE")
        print("="*70)
    
    def generate_reports(self):
        """Generate quality reports for all trials"""
        if not self.config.CREATE_REPORTS:
            logger.info("Report generation disabled")
            return
        
        print("\n" + "="*70)
        print("GENERATING QUALITY REPORTS")
        print("="*70)
        
        for trial_name in self.trial_names:
            sync_dir = Path(self.config.OUTPUT_DIR) / trial_name / "synchronized_videos"
            
            if not sync_dir.exists():
                continue
            
            print(f"\n{trial_name}:")
            self.analyzer.analyze_synchronization(sync_dir)
            
            # Create comparison video if enabled
            if self.config.CREATE_COMPARISON:
                video_files = sorted(sync_dir.glob("*.mp4"))
                if len(video_files) >= 2:
                    comparison_path = sync_dir / "comparison.mp4"
                    print(f"  Creating comparison video...")
                    self.analyzer.create_side_by_side_comparison(
                        video_files,
                        comparison_path,
                        duration=self.config.COMPARISON_DURATION
                    )
            
            # Visualize offsets
            report_path = sync_dir / "synchronization_report.txt"
            if report_path.exists():
                self.analyzer.visualize_sync_offsets(report_path)
    
    def run_complete_experiment(self):
        """Run complete experiment workflow"""
        try:
            # Setup
            if not self.setup():
                logger.error("Setup failed!")
                return
            
            # Run trials
            self.run_trials()
            
            # Cleanup
            self.cleanup()
            
            # Synchronize
            self.synchronize_all()
            
            # Generate reports
            self.generate_reports()
            
            print("\n" + "="*70)
            print("EXPERIMENT COMPLETE!")
            print("="*70)
            print(f"\nResults saved to: {self.config.OUTPUT_DIR}/")
            print(f"Total trials: {len(self.trial_names)}")
            
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user")
            self.cleanup()
        except Exception as e:
            logger.error(f"Experiment error: {e}")
            self.cleanup()
            raise


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Automated Multi-Camera Experiment Runner             ║
║           Neuroscience Video Recording System                ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Show configuration
    print("Current Configuration:")
    print(f"  Cameras: {config.CAMERA_IDS}")
    print(f"  Resolution: {config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps")
    print(f"  Trials: {config.NUM_TRIALS}")
    print(f"  Trial Duration: {config.TRIAL_DURATION}s")
    print(f"  Sync Method: {config.SYNC_METHOD}")
    print(f"  Output: {config.OUTPUT_DIR}")
    
    print("\nTo modify settings, edit config.py\n")
    
    # Confirm
    response = input("Start experiment? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Experiment cancelled")
        return
    
    # Run experiment
    runner = ExperimentRunner()
    runner.run_complete_experiment()


if __name__ == "__main__":
    main()
