"""
Multi-Camera FFmpeg Video Recorder
Uses ffmpeg with hardware acceleration for efficient multi-camera recording
"""

import subprocess
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FFmpegMultiCamRecorder:
    """Record multiple cameras simultaneously using FFmpeg subprocesses"""
    
    def __init__(
        self,
        camera_ids: List[int],
        output_dir: str = "recordings",
        framerate: int = 30,
        resolution: str = "1920x1080",
        codec: str = "h264_videotoolbox",  # Hardware acceleration on macOS
        input_format: str = "avfoundation",  # macOS camera input
        file_extension: str = "mp4"
    ):
        """
        Initialize the multi-camera recorder
        
        Args:
            camera_ids: List of camera device IDs (e.g., [0, 1, 2, 3])
            output_dir: Directory to save recordings
            framerate: Recording framerate (default: 30)
            resolution: Video resolution as "WIDTHxHEIGHT" (default: "1920x1080")
            codec: Video codec to use (default: "h264_videotoolbox" for macOS)
                   Use "libx264" for software encoding on Linux/Windows
            input_format: FFmpeg input format (default: "avfoundation" for macOS)
                         Use "v4l2" for Linux, "dshow" for Windows
            file_extension: Output file extension (default: "mp4")
        """
        self.camera_ids = camera_ids
        self.output_dir = Path(output_dir)
        self.framerate = framerate
        self.resolution = resolution
        self.codec = codec
        self.input_format = input_format
        self.file_extension = file_extension
        
        # FFmpeg process tracking
        self.processes: Dict[int, subprocess.Popen] = {}
        self.output_files: Dict[int, Path] = {}
        self.is_recording = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and termination signals gracefully"""
        logger.info("\nReceived interrupt signal, stopping recording...")
        self.stop_recording()
        sys.exit(0)
    
    def list_cameras(self) -> None:
        """List available cameras (macOS AVFoundation)"""
        try:
            logger.info("Listing available cameras...")
            cmd = ["ffmpeg", "-f", self.input_format, "-list_devices", "true", "-i", ""]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            # FFmpeg outputs device list to stderr
            print(result.stderr)
        except subprocess.TimeoutExpired:
            logger.warning("Camera listing timed out")
        except Exception as e:
            logger.error(f"Error listing cameras: {e}")
    
    def start_recording(self, session_name: Optional[str] = None, trial_number: Optional[int] = None) -> bool:
        """
        Start recording from all cameras
        
        Args:
            session_name: Optional name for this recording session.
                         If not provided, uses timestamp.
            trial_number: Optional trial number to append to filename
        
        Returns:
            True if all cameras started successfully, False otherwise
        """
        if self.is_recording:
            logger.warning("Recording already in progress")
            return False
        
        # Generate session name if not provided
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add trial number if provided
        if trial_number is not None:
            session_name = f"{session_name}_trial{trial_number:03d}"
        
        logger.info(f"Starting recording session: {session_name}")
        logger.info(f"Configuration: {self.resolution} @ {self.framerate}fps, codec: {self.codec}")
        
        success_count = 0
        
        for cam_id in self.camera_ids:
            output_file = self.output_dir / f"{session_name}_cam{cam_id}.{self.file_extension}"
            
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-f", self.input_format,
                "-framerate", str(self.framerate),
                "-video_size", self.resolution,
                "-i", str(cam_id),
                "-c:v", self.codec,
                str(output_file)
            ]
            
            try:
                # Start FFmpeg process
                logger.info(f"Starting camera {cam_id} -> {output_file.name}")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE
                )
                
                # Store process and output file
                self.processes[cam_id] = process
                self.output_files[cam_id] = output_file
                success_count += 1
                
                # Small delay between camera starts to avoid USB bandwidth issues
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Failed to start camera {cam_id}: {e}")
        
        if success_count > 0:
            self.is_recording = True
            logger.info(f"Successfully started {success_count}/{len(self.camera_ids)} cameras")
            return True
        else:
            logger.error("Failed to start any cameras")
            return False
    
    def stop_recording(self) -> Dict[int, Path]:
        """
        Stop all camera recordings gracefully
        
        Returns:
            Dictionary mapping camera IDs to their output file paths
        """
        if not self.is_recording:
            logger.warning("No recording in progress")
            return {}
        
        logger.info("Stopping all cameras...")
        
        for cam_id, process in self.processes.items():
            try:
                # Send 'q' to FFmpeg for graceful shutdown
                logger.info(f"Stopping camera {cam_id}...")
                process.stdin.write(b'q')
                process.stdin.flush()
                
                # Wait for process to finish (with timeout)
                try:
                    process.wait(timeout=5)
                    logger.info(f"Camera {cam_id} stopped successfully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Camera {cam_id} didn't stop gracefully, terminating...")
                    process.terminate()
                    process.wait(timeout=2)
                
            except Exception as e:
                logger.error(f"Error stopping camera {cam_id}: {e}")
                try:
                    process.kill()
                except:
                    pass
        
        self.is_recording = False
        output_files = self.output_files.copy()
        
        # Clear process tracking
        self.processes.clear()
        self.output_files.clear()
        
        logger.info(f"Recording stopped. Files saved to: {self.output_dir}")
        for cam_id, filepath in output_files.items():
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  Camera {cam_id}: {filepath.name} ({size_mb:.1f} MB)")
        
        return output_files
    
    def record_for_duration(self, duration_seconds: float, session_name: Optional[str] = None, trial_number: Optional[int] = None) -> Dict[int, Path]:
        """
        Record for a specific duration then stop
        
        Args:
            duration_seconds: How long to record in seconds
            session_name: Optional name for this recording session
            trial_number: Optional trial number to append to filename
        
        Returns:
            Dictionary mapping camera IDs to their output file paths
        """
        if not self.start_recording(session_name, trial_number):
            return {}
        
        try:
            logger.info(f"Recording for {duration_seconds} seconds...")
            logger.info("Press Ctrl+C to stop early")
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            logger.info("\nRecording interrupted by user")
        finally:
            return self.stop_recording()
    
    def record_trials(
        self, 
        num_trials: int, 
        trial_duration: float,
        inter_trial_interval: float = 1.0,
        session_name: Optional[str] = None
    ) -> Dict[int, List[Path]]:
        """
        Record multiple trials in sequence without disconnecting cameras
        
        Args:
            num_trials: Number of trials to record
            trial_duration: Duration of each trial in seconds
            inter_trial_interval: Pause between trials in seconds (default: 1.0)
            session_name: Base name for this session (trial numbers will be appended)
        
        Returns:
            Dictionary mapping camera IDs to lists of their output file paths
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Starting {num_trials} trials with {trial_duration}s each")
        logger.info(f"Inter-trial interval: {inter_trial_interval}s")
        
        # Store all output files per camera
        all_outputs: Dict[int, List[Path]] = {cam_id: [] for cam_id in self.camera_ids}
        
        for trial_num in range(1, num_trials + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"TRIAL {trial_num}/{num_trials}")
            logger.info(f"{'='*60}")
            
            try:
                # Start recording for this trial
                if not self.start_recording(session_name=session_name, trial_number=trial_num):
                    logger.error(f"Failed to start trial {trial_num}")
                    continue
                
                # Record for the trial duration
                logger.info(f"Recording trial {trial_num}...")
                time.sleep(trial_duration)
                
                # Stop recording
                output_files = self.stop_recording()
                
                # Store the output files
                for cam_id, filepath in output_files.items():
                    all_outputs[cam_id].append(filepath)
                
                # Inter-trial interval (except after last trial)
                if trial_num < num_trials:
                    logger.info(f"Inter-trial interval: {inter_trial_interval}s")
                    time.sleep(inter_trial_interval)
                
            except KeyboardInterrupt:
                logger.info("\nTrials interrupted by user")
                self.stop_recording()
                break
            except Exception as e:
                logger.error(f"Error during trial {trial_num}: {e}")
                if self.is_recording:
                    self.stop_recording()
                continue
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TRIALS COMPLETE")
        logger.info(f"{'='*60}")
        total_files = sum(len(files) for files in all_outputs.values())
        logger.info(f"Total trials recorded: {total_files // len(self.camera_ids)}")
        logger.info(f"Files saved to: {self.output_dir}")
        
        for cam_id, filepaths in all_outputs.items():
            if filepaths:
                total_size = sum(f.stat().st_size for f in filepaths if f.exists())
                size_mb = total_size / (1024 * 1024)
                logger.info(f"  Camera {cam_id}: {len(filepaths)} files ({size_mb:.1f} MB)")
        
        return all_outputs
    
    def get_status(self) -> Dict[str, any]:
        """Get current recording status"""
        active_cameras = []
        for cam_id, process in self.processes.items():
            if process.poll() is None:  # Process is still running
                active_cameras.append(cam_id)
        
        return {
            "is_recording": self.is_recording,
            "active_cameras": active_cameras,
            "total_cameras": len(self.camera_ids),
            "output_dir": str(self.output_dir)
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup"""
        if self.is_recording:
            self.stop_recording()
        return False


# Platform-specific codec recommendations
def get_recommended_codec() -> str:
    """Get the recommended codec for the current platform"""
    import platform
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return "h264_videotoolbox"
    elif system == "Linux":
        return "libx264"
    elif system == "Windows":
        return "h264_nvenc"  # NVIDIA GPU encoding, fallback to libx264
    else:
        return "libx264"


def get_input_format() -> str:
    """Get the input format for the current platform"""
    import platform
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return "avfoundation"
    elif system == "Linux":
        return "v4l2"
    elif system == "Windows":
        return "dshow"
    else:
        return "v4l2"


# Example usage functions
def record_multicam_simple(camera_ids: List[int], duration: float = 10, output_dir: str = "recordings"):
    """
    Simple function to record multiple cameras for a specified duration
    
    Example:
        record_multicam_simple([0, 1, 2, 3], duration=30)
    """
    with FFmpegMultiCamRecorder(
        camera_ids=camera_ids,
        output_dir=output_dir,
        codec=get_recommended_codec(),
        input_format=get_input_format()
    ) as recorder:
        recorder.record_for_duration(duration)


def main_example():
    """Example usage of the FFmpeg multi-camera recorder"""
    
    # Example 1: Record 4 cameras for 10 seconds
    camera_ids = [0, 1, 2, 3]
    
    with FFmpegMultiCamRecorder(
        camera_ids=camera_ids,
        output_dir="my_recordings",
        framerate=30,
        resolution="1920x1080",
        codec=get_recommended_codec(),
        input_format=get_input_format()
    ) as recorder:
        
        # Optional: List available cameras first
        # recorder.list_cameras()
        
        # Start recording
        if recorder.start_recording(session_name="test_session"):
            
            # Record for 10 seconds
            try:
                print("Recording... Press Ctrl+C to stop")
                time.sleep(10)
            except KeyboardInterrupt:
                print("\nStopping...")
            
            # Stop and get output files
            output_files = recorder.stop_recording()
            
            print("\nRecording complete!")
            print("Output files:")
            for cam_id, filepath in output_files.items():
                print(f"  Camera {cam_id}: {filepath}")


if __name__ == "__main__":
    # Simple quick test
    print("FFmpeg Multi-Camera Recorder")
    print("=" * 50)
    
    # Adjust these camera IDs based on your setup
    cameras = [0, 1, 2, 3]
    
    # Record for 5 seconds
    record_multicam_simple(cameras, duration=5)
