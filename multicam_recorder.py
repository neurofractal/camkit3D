"""
Multi-Camera Synchronized Video Recorder - ROBUST VERSION
No FFmpeg dependency - pure OpenCV with enhanced error handling
"""

import cv2
import numpy as np
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraThread(threading.Thread):
    """Thread for capturing frames from a single camera - ROBUST VERSION"""
    
    def __init__(self, camera_id: int, width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__()
        self.daemon = True  # Daemon thread for automatic cleanup
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=3)  # Small queue - we always grab latest
        self.running = False
        self.recording = False
        self.is_connected = False
        
        # Recording
        self.video_writer = None
        self.frame_count = 0
        self.timestamps = []
        self._lock = threading.Lock()  # Thread-safe operations
        self._stop_event = threading.Event()  # Clean shutdown signal
        
        # Keep last valid frame to avoid black flashes
        self.last_frame = None
        self.last_timestamp = None
        
        # Camera name
        self.camera_name = None
        
    def connect(self) -> bool:
        """Connect to the camera"""
        try:
            # Try DirectShow first (Windows)
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Try to get camera name
            self._detect_camera_name()
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize lag
            
            # Read actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            camera_info = f"Camera {self.camera_id}"
            if self.camera_name:
                camera_info += f" ({self.camera_name})"
            camera_info += f": {actual_width}x{actual_height} @ {actual_fps}fps"
            logger.info(camera_info)
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to camera {self.camera_id}: {e}")
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
            return False
    
    def _detect_camera_name(self):
        """Try to detect camera name (Windows only)"""
        try:
            import platform
            if platform.system() == 'Windows':
                import subprocess
                # Use Windows PowerShell to get camera names
                result = subprocess.run(
                    ['powershell', '-Command', 
                     'Get-PnpDevice -Class Camera | Select-Object FriendlyName | Format-Table -HideTableHeaders'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                    if self.camera_id < len(lines):
                        self.camera_name = lines[self.camera_id]
                        return
        except Exception as e:
            logger.debug(f"Could not detect camera name: {e}")
        
        # Fallback: generic name
        self.camera_name = f"Camera {self.camera_id}"
    
    def start_recording(self, output_path: Path) -> bool:
        """Start recording to file"""
        if not self.is_connected:
            logger.error(f"Camera {self.camera_id} not connected")
            return False
        
        with self._lock:
            try:
                # Use MJPEG codec - most stable for OpenCV
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                
                # Force .avi extension for MJPEG
                if output_path.suffix.lower() != '.avi':
                    output_path = output_path.with_suffix('.avi')
                
                self.video_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    self.fps,
                    (self.width, self.height)
                )
                
                if not self.video_writer or not self.video_writer.isOpened():
                    logger.error(f"Failed to open video writer for camera {self.camera_id}")
                    if self.video_writer:
                        try:
                            self.video_writer.release()
                        except:
                            pass
                    self.video_writer = None
                    return False
                
                self.recording = True
                self.frame_count = 0
                self.timestamps = []
                logger.info(f"Camera {self.camera_id} recording to {output_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error starting recording on camera {self.camera_id}: {e}")
                if self.video_writer:
                    try:
                        self.video_writer.release()
                    except:
                        pass
                    self.video_writer = None
                return False
    
    def stop_recording(self) -> Tuple[int, List[float]]:
        """Stop recording and return stats"""
        with self._lock:
            self.recording = False
            
            # Save stats before cleanup
            frame_count = self.frame_count
            timestamps = self.timestamps.copy()
            
            # Release video writer SAFELY
            if self.video_writer:
                try:
                    # Give a moment for pending writes
                    time.sleep(0.1)
                    
                    # Check if still valid before releasing
                    if self.video_writer.isOpened():
                        self.video_writer.release()
                except Exception as e:
                    logger.warning(f"Error releasing video writer for camera {self.camera_id}: {e}")
                finally:
                    self.video_writer = None
            
            logger.info(f"Camera {self.camera_id} stopped recording: {frame_count} frames")
            
            return frame_count, timestamps
    
    def run(self):
        """Main camera capture loop - ROBUST VERSION"""
        self.running = True
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running and not self._stop_event.is_set():
            if not self.is_connected:
                time.sleep(0.1)
                continue
            
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from camera {self.camera_id} ({consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Camera {self.camera_id} has too many failures, stopping")
                        self.running = False
                        break
                    
                    time.sleep(0.05)
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Record timestamp
                timestamp = time.time()
                
                # If recording, save frame (thread-safe)
                if self.recording:
                    with self._lock:
                        if self.video_writer and self.video_writer.isOpened():
                            try:
                                self.video_writer.write(frame)
                                self.frame_count += 1
                                self.timestamps.append(timestamp)
                            except Exception as e:
                                logger.error(f"Error writing frame for camera {self.camera_id}: {e}")
                                self.recording = False  # Stop recording on write error
                
                # Put frame in queue for display/processing (non-blocking)
                try:
                    self.frame_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    # Clear old frames if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame, timestamp))
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in camera {self.camera_id} loop: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get the latest frame (non-blocking) - always returns most recent"""
        latest_frame = None
        
        # Clear the queue and get the most recent frame
        try:
            while True:
                latest_frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        
        # If we got a new frame, update last_frame cache
        if latest_frame is not None:
            self.last_frame, self.last_timestamp = latest_frame
            return latest_frame
        
        # If queue was empty, return the last valid frame
        if self.last_frame is not None:
            return (self.last_frame, self.last_timestamp)
        
        return None
    
    def stop(self):
        """Stop the camera thread - ROBUST CLEANUP"""
        logger.info(f"Stopping camera {self.camera_id}...")
        
        # Signal thread to stop
        self.running = False
        self._stop_event.set()
        
        # Stop recording if active
        with self._lock:
            self.recording = False
        
        # Give thread time to finish gracefully
        time.sleep(0.3)
        
        # Clean up video writer
        with self._lock:
            if self.video_writer:
                try:
                    if self.video_writer.isOpened():
                        self.video_writer.release()
                except Exception as e:
                    logger.warning(f"Error releasing video writer for camera {self.camera_id}: {e}")
                finally:
                    self.video_writer = None
        
        # Clean up camera capture
        if self.cap:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing camera {self.camera_id}: {e}")
            finally:
                self.cap = None
        
        self.is_connected = False
        logger.info(f"Camera {self.camera_id} stopped successfully")


class MultiCameraRecorder:
    """
    Multi-camera recorder with synchronized recording capability - ROBUST VERSION
    
    Features:
    - Pure OpenCV (no FFmpeg dependency for recording)
    - Enhanced error handling
    - Thread-safe operations
    - Graceful cleanup
    
    Workflow:
    1. Connect to cameras
    2. Record multiple trials
    3. Synchronize offline
    """
    
    def __init__(self, 
                 camera_ids: List[int] = None,
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30,
                 base_output_dir: str = "./recordings"):
        """
        Initialize multi-camera recorder
        
        Args:
            camera_ids: List of camera IDs to use (e.g., [0, 1, 2])
            width: Frame width
            height: Frame height
            fps: Frames per second
            base_output_dir: Base directory for recordings
        """
        self.camera_ids = camera_ids or [0]
        self.width = width
        self.height = height
        self.fps = fps
        self.base_output_dir = Path(base_output_dir)
        
        self.cameras: Dict[int, CameraThread] = {}
        self.is_recording = False
        self.current_trial_name = None
        self.current_trial_dir = None
        
        logger.info(f"MultiCameraRecorder initialized with cameras: {self.camera_ids}")
    
    def connect_cameras(self) -> Dict[int, bool]:
        """
        Connect to all cameras
        
        Returns:
            Dictionary of camera_id -> connection success
        """
        results = {}
        
        for cam_id in self.camera_ids:
            logger.info(f"Connecting to camera {cam_id}...")
            
            camera = CameraThread(cam_id, self.width, self.height, self.fps)
            success = camera.connect()
            
            if success:
                self.cameras[cam_id] = camera
                camera.start()  # Start thread
                time.sleep(0.2)  # Small delay between camera starts
            
            results[cam_id] = success
        
        connected = sum(results.values())
        logger.info(f"Connected to {connected}/{len(self.camera_ids)} cameras")
        
        return results
    
    def start_recording(self, trial_name: Optional[str] = None) -> bool:
        """
        Start recording on all cameras
        
        Args:
            trial_name: Name for this trial (auto-generated if None)
            
        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            logger.warning("Already recording!")
            return False
        
        if not self.cameras:
            logger.error("No cameras connected!")
            return False
        
        # Generate trial name
        if trial_name is None:
            trial_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.current_trial_name = trial_name
        self.current_trial_dir = self.base_output_dir / trial_name / "raw_videos"
        self.current_trial_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting recording: {trial_name}")
        
        # Start recording on all cameras
        success_count = 0
        for cam_id, camera in self.cameras.items():
            output_path = self.current_trial_dir / f"camera_{cam_id}.avi"  # .avi for MJPEG
            if camera.start_recording(output_path):
                success_count += 1
        
        if success_count > 0:
            self.is_recording = True
            logger.info(f"Recording started on {success_count} cameras")
            return True
        else:
            logger.error("Failed to start recording on any camera")
            return False
    
    def stop_recording(self) -> Dict[int, Tuple[int, List[float]]]:
        """
        Stop recording on all cameras
        
        Returns:
            Dictionary of camera_id -> (frame_count, timestamps)
        """
        if not self.is_recording:
            logger.warning("Not currently recording!")
            return {}
        
        logger.info("Stopping recording...")
        
        results = {}
        for cam_id, camera in self.cameras.items():
            try:
                frame_count, timestamps = camera.stop_recording()
                results[cam_id] = (frame_count, timestamps)
            except Exception as e:
                logger.error(f"Error stopping recording for camera {cam_id}: {e}")
                results[cam_id] = (0, [])
        
        # Save metadata
        try:
            self._save_metadata(results)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
        
        self.is_recording = False
        logger.info(f"Recording stopped: {self.current_trial_name}")
        
        return results
    
    def _save_metadata(self, recording_stats: Dict[int, Tuple[int, List[float]]]):
        """Save recording metadata"""
        metadata_path = self.base_output_dir / self.current_trial_name / "metadata.txt"
        
        try:
            with open(metadata_path, 'w') as f:
                f.write(f"Trial: {self.current_trial_name}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Resolution: {self.width}x{self.height}\n")
                f.write(f"FPS: {self.fps}\n")
                f.write(f"Codec: MJPEG (.avi)\n\n")
                
                for cam_id, (frame_count, timestamps) in recording_stats.items():
                    f.write(f"Camera {cam_id}:\n")
                    f.write(f"  Frames: {frame_count}\n")
                    if timestamps and len(timestamps) > 0:
                        f.write(f"  Duration: {timestamps[-1] - timestamps[0]:.2f}s\n")
                    f.write("\n")
            
            # Save timestamps as numpy arrays
            for cam_id, (_, timestamps) in recording_stats.items():
                if timestamps and len(timestamps) > 0:
                    try:
                        np.save(
                            self.base_output_dir / self.current_trial_name / f"camera_{cam_id}_timestamps.npy",
                            np.array(timestamps)
                        )
                    except Exception as e:
                        logger.error(f"Error saving timestamps for camera {cam_id}: {e}")
        except Exception as e:
            logger.error(f"Error in _save_metadata: {e}")
    
    def disconnect_cameras(self):
        """Disconnect all cameras - ROBUST CLEANUP"""
        logger.info("Disconnecting cameras...")
        
        # Stop recording if still active
        if self.is_recording:
            logger.warning("Still recording - stopping recording first")
            try:
                self.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording during disconnect: {e}")
        
        # Give a moment for cleanup
        time.sleep(0.3)
        
        # Stop all camera threads
        for cam_id, camera in list(self.cameras.items()):
            try:
                camera.stop()
            except Exception as e:
                logger.error(f"Error stopping camera {cam_id}: {e}")
        
        # Wait for threads to finish
        for cam_id, camera in list(self.cameras.items()):
            try:
                camera.join(timeout=3.0)
                if camera.is_alive():
                    logger.warning(f"Camera {cam_id} thread did not stop cleanly")
            except Exception as e:
                logger.error(f"Error joining camera {cam_id} thread: {e}")
        
        # Clear camera dictionary
        self.cameras.clear()
        
        # Final cleanup delay
        time.sleep(0.2)
        
        # Force OpenCV cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("All cameras disconnected successfully")
    
    def preview_cameras(self, duration: float = 5.0, preview_scale: float = 0.5, target_fps: int = 15):
        """
        Show preview in combined grid view with smooth playback
        
        Args:
            duration: Preview duration in seconds
            preview_scale: Scale factor for preview (0.5 = half size, faster)
            target_fps: Target display FPS (15 fps recommended for smooth preview)
        """
        if not self.cameras:
            logger.error("No cameras connected!")
            return

        num_cameras = len(self.cameras)
        
        # Camera symbols (ASCII art style that OpenCV can display)
        camera_symbols = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]"]
        camera_colors = [
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 255, 0),   # Green-yellow
            (255, 0, 255),   # Magenta
            (0, 255, 128),   # Aqua
            (255, 255, 0),   # Yellow
            (128, 0, 255),   # Purple
            (255, 128, 255), # Pink
            (0, 128, 255),   # Light blue
        ]

        # Calculate scaled dimensions
        preview_width = int(self.width * preview_scale)
        preview_height = int(self.height * preview_scale)

        # Calculate grid layout
        if num_cameras == 1:
            grid_rows, grid_cols = 1, 1
        elif num_cameras == 2:
            grid_rows, grid_cols = 1, 2
        elif num_cameras == 3:
            grid_rows, grid_cols = 2, 2  # 2x2 with one empty
        elif num_cameras == 4:
            grid_rows, grid_cols = 2, 2
        elif num_cameras <= 6:
            grid_rows, grid_cols = 2, 3
        elif num_cameras <= 9:
            grid_rows, grid_cols = 3, 3
        else:
            grid_rows, grid_cols = 4, 4

        logger.info(f"Showing preview for {duration}s at ~{target_fps}fps (press 'q' to quit)...")
        logger.info(f"Preview resolution: {preview_width}x{preview_height} ({int(preview_scale*100)}% of original)")
        
        start_time = time.time()
        frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        
        # FPS tracking for each camera (actual camera FPS, not display FPS)
        fps_trackers = {cam_id: {'last_time': time.time(), 'frame_count': 0, 'fps': 0.0} 
                       for cam_id in self.cameras.keys()}

        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Frame rate limiting for smooth display
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                last_frame_time = current_time
                grid_frames = []
                
                for idx, cam_id in enumerate(sorted(self.cameras.keys())):
                    camera = self.cameras[cam_id]
                    frame_data = camera.get_frame()
                    
                    if frame_data:
                        frame, timestamp = frame_data
                        
                        # Downsample frame for preview
                        frame_preview = cv2.resize(frame, (preview_width, preview_height), 
                                                  interpolation=cv2.INTER_LINEAR)
                        
                        # Update FPS tracking
                        tracker = fps_trackers[cam_id]
                        tracker['frame_count'] += 1
                        time_diff = current_time - tracker['last_time']
                        
                        if time_diff >= 1.0:  # Update FPS every second
                            tracker['fps'] = tracker['frame_count'] / time_diff
                            tracker['frame_count'] = 0
                            tracker['last_time'] = current_time
                        
                        # Get symbol and color for this camera
                        symbol = camera_symbols[idx] if idx < len(camera_symbols) else f"[{idx+1}]"
                        color = camera_colors[idx % len(camera_colors)]
                        
                        # Get camera name
                        cam_name = camera.camera_name if camera.camera_name else f"Camera {cam_id}"
                        
                        # Add semi-transparent background for text (scaled)
                        overlay = frame_preview.copy()
                        text_bg_width = int(min(preview_width - 10, 450 * preview_scale))
                        text_bg_height = int(120 * preview_scale)  # Increased for extra line
                        cv2.rectangle(overlay, (0, 0), (text_bg_width, text_bg_height), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, frame_preview, 0.3, 0, frame_preview)
                        
                        # Scaled text sizes
                        symbol_font_scale = 1.2 * preview_scale
                        label_font_scale = 0.6 * preview_scale
                        fps_font_scale = 0.55 * preview_scale
                        label_thickness = max(1, int(2 * preview_scale))
                        
                        # Camera symbol (large and colorful)
                        symbol_y = int(40 * preview_scale)
                        cv2.putText(frame_preview, symbol, (10, symbol_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, symbol_font_scale, 
                                  color, label_thickness + 1, cv2.LINE_AA)
                        
                        # Camera name (below symbol)
                        name_x = int(70 * preview_scale) if idx < 9 else int(90 * preview_scale)
                        name_y = int(30 * preview_scale)
                        
                        # Truncate camera name if too long
                        max_name_length = int(35 / preview_scale)
                        if len(cam_name) > max_name_length:
                            cam_name = cam_name[:max_name_length-3] + "..."
                        
                        cv2.putText(frame_preview, cam_name, (name_x, name_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 
                                  (200, 200, 200), label_thickness, cv2.LINE_AA)
                        
                        # FPS display
                        fps_text = f"FPS: {tracker['fps']:.1f}"
                        fps_y = int(55 * preview_scale)
                        cv2.putText(frame_preview, fps_text, (name_x, fps_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, 
                                  (0, 255, 0), label_thickness, cv2.LINE_AA)
                        
                        # Resolution info (actual recording resolution)
                        res_text = f"Rec: {self.width}x{self.height}"
                        res_y = int(78 * preview_scale)
                        cv2.putText(frame_preview, res_text, (name_x, res_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale, 
                                  (150, 150, 255), label_thickness, cv2.LINE_AA)
                        
                        # Preview resolution
                        preview_res_text = f"View: {preview_width}x{preview_height}"
                        preview_res_y = int(98 * preview_scale)
                        cv2.putText(frame_preview, preview_res_text, (name_x, preview_res_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, fps_font_scale * 0.9, 
                                  (100, 100, 100), label_thickness, cv2.LINE_AA)
                        
                        grid_frames.append(frame_preview)
                    else:
                        # Create blank frame if no data (scaled)
                        blank = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
                        
                        # Add "No Signal" message
                        symbol = camera_symbols[idx] if idx < len(camera_symbols) else f"[{idx+1}]"
                        label_font_scale = 0.9 * preview_scale
                        no_signal_font_scale = 1.2 * preview_scale
                        label_thickness = max(1, int(2 * preview_scale))
                        
                        cv2.putText(blank, f"{symbol} No Signal", 
                                  (10, int(35 * preview_scale)),
                                  cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 
                                  (100, 100, 100), label_thickness, cv2.LINE_AA)
                        grid_frames.append(blank)
                
                # Fill remaining grid positions with black frames
                while len(grid_frames) < grid_rows * grid_cols:
                    blank = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
                    grid_frames.append(blank)
                
                # Build grid
                rows = []
                for i in range(grid_rows):
                    start_idx = i * grid_cols
                    end_idx = start_idx + grid_cols
                    row_frames = grid_frames[start_idx:end_idx]
                    rows.append(np.hstack(row_frames))
                
                combined = np.vstack(rows)
                
                # Add global title bar (scaled)
                title_height = int(60 * preview_scale)
                title_bar = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
                title_text = f"Multi-Camera Preview ({int(preview_scale*100)}% scale, ~{target_fps}fps)  |  {num_cameras} Camera{'s' if num_cameras != 1 else ''}  |  Press 'Q' to quit"
                title_font_scale = 0.6 * preview_scale
                title_thickness = max(1, int(2 * preview_scale))
                text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                          title_font_scale, title_thickness)[0]
                text_x = max(10, (combined.shape[1] - text_size[0]) // 2)
                text_y = int(40 * preview_scale)
                cv2.putText(title_bar, title_text, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, 
                          (255, 255, 255), title_thickness, cv2.LINE_AA)
                
                # Combine title bar with grid
                final_display = np.vstack([title_bar, combined])
                
                cv2.imshow("Multi-Camera Preview", final_display)
                
                # Check for quit key (non-blocking)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Preview error: {e}")
        finally:
            cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup"""
        try:
            self.disconnect_cameras()
        except Exception as e:
            logger.error(f"Error in context manager cleanup: {e}")
        return False


# Convenience function for safe usage
def create_recorder(*args, **kwargs) -> MultiCameraRecorder:
    """
    Create a recorder instance - use with context manager for automatic cleanup
    
    Example:
        with create_recorder(camera_ids=[0, 1]) as recorder:
            recorder.connect_cameras()
            recorder.start_recording("test")
            time.sleep(5)
            recorder.stop_recording()
        # Automatic cleanup on exit
    """
    return MultiCameraRecorder(*args, **kwargs)