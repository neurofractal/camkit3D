"""Synchronized multi-camera video recorder for CamKit3D.

Captures frame-locked video from multiple USB cameras for markerless 3D
motion capture, built to record cleanly on standard hardware and to align
precisely with neural data.

Key features:

- Two-phase start/stop. Every camera is prepared (file and encoder opened)
  first, then armed, so all streams begin and end within microseconds of 
  each other rather than staggered by slow file setup.
- Hardware synchronisation. Optional DataPixx digital triggers fire the
  instant capture starts and stops, giving a shared t=0 with external
  recordings such as OPM-MEG, EEG, eye-trackers, or physiology.
- Drop-free capture. A threaded write queue decouples frame grabbing from
  disk I/O, preventing periodic OS buffer flushes from stalling capture and
  dropping frames.
- Pure OpenCV with no FFmpeg dependency, and per-camera fallback logic for
  robust connection across backends.

Author: Dr. Robert Seymour, OHBA, University of Oxford
License: GNU General Public License v3, 2026
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


# ---------------------------------------------------------------------------
# DataPixx trigger helpers (only used when use_datapixx=True)
# ---------------------------------------------------------------------------
_dpx_available = False
try:
    from pypixxlib import _libdpx as dp
    _dpx_available = True
except ImportError:
    logger.info("pypixxlib not available — DataPixx triggers disabled")


def _dpx_send_trigger(trig: int):
    """Send a trigger value via DataPixx digital outputs.

    Args:
        trig: Trigger number (0 = all off, 1/2/3/... = specific trigger line)
    """
    if not _dpx_available:
        return

    if trig == 0:
        val = 0
    else:
        val = 1 << (2 * trig)  # trig 1 → bit 2, trig 2 → bit 4, etc.

    bit_mask = 0xFFFFFF
    dp.DPxSetDoutValue(val, bit_mask)
    dp.DPxWriteRegCache()


def _dpx_open():
    """Open DataPixx connection and ensure triggers start at 0."""
    if not _dpx_available:
        logger.info("DataPixx not available — skipping open")
        return

    dp.DPxOpen()
    if not dp.DPxIsReady():
        logger.warning("DataPixx not ready")
    _dpx_send_trigger(0)
    time.sleep(0.1)
    logger.info("DataPixx connection opened, triggers reset to 0")


def _dpx_close():
    """Ensure triggers are off and close DataPixx connection."""
    if not _dpx_available:
        return

    _dpx_send_trigger(0)
    dp.DPxClose()
    logger.info("DataPixx connection closed")


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
        self.running = False
        self.recording = False
        self.is_connected = False
        
        # Recording
        self.video_writer = None
        self.frame_count = 0
        self.timestamps = []
        self._lock = threading.Lock()  # Thread-safe operations
        self._stop_event = threading.Event()  # Clean shutdown signal
        
        # Write queue: decouples capture from disk I/O so that periodic
        # OS write-buffer flushes (~every 40 frames with MJPG at 720p)
        # cannot stall the capture thread and cause frame drops.
        self._write_queue = queue.Queue(maxsize=120)  # ~4s buffer at 30fps
        self._writer_thread = None
        self._writer_stop = threading.Event()
        
        # Keep last valid frame to avoid black flashes
        self.last_frame = None
        self.last_timestamp = None
        
        # Camera name
        self.camera_name = None
        
    def connect(self) -> bool:
        """Connect to the camera"""
        try:
            # Use default backend (MSMF on Windows) - DirectShow has frame drop issues
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                # Fallback to DirectShow
                self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Try to get camera name
            self._detect_camera_name()
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            # NOTE: Do NOT set CAP_PROP_BUFFERSIZE=1, it causes the driver to
            # discard frames when the app can't read fast enough
            
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
    
    def _writer_loop(self):
        """Dedicated disk-writing thread. Drains the write queue and writes
        frames to the VideoWriter. This runs on its own thread so that
        periodic OS write-buffer flushes (which block for ~10-30ms) never
        stall the capture thread."""
        while not self._writer_stop.is_set():
            try:
                frame, timestamp = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                    self.frame_count += 1
                    self.timestamps.append(timestamp)
            except Exception as e:
                logger.error(f"Writer thread error for camera {self.camera_id}: {e}")
    
    def prepare_recording(self, output_path: Path) -> bool:
        """Phase 1 of starting a recording: do ALL the slow work (open the
        VideoWriter, reset counters, start the writer thread) but DO NOT begin
        capturing frames yet. self.recording stays False until arm_recording().

        Splitting prepare/arm lets MultiCamRecorder open every camera's writer
        first, then flip them all live near-simultaneously, so the cameras
        start within microseconds of each other instead of being staggered by
        the (~100 ms) cost of opening each writer in sequence.
        """
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

                # Clear the write queue and reset counters
                while not self._write_queue.empty():
                    try:
                        self._write_queue.get_nowait()
                    except queue.Empty:
                        break

                self.frame_count = 0
                self.timestamps = []

                # Start the dedicated writer thread. It will sit idle (the
                # capture loop only enqueues frames once self.recording is True),
                # so starting it here costs nothing but removes it from the
                # critical arm path.
                self._writer_stop.clear()
                self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
                self._writer_thread.start()

                logger.info(f"Camera {self.camera_id} prepared, writing to {output_path}")
                return True

            except Exception as e:
                logger.error(f"Error preparing recording on camera {self.camera_id}: {e}")
                if self.video_writer:
                    try:
                        self.video_writer.release()
                    except:
                        pass
                    self.video_writer = None
                return False

    def arm_recording(self) -> None:
        """Phase 2: flip the recording flag live. This is the ONLY thing that
        decides which frames land in the file, and it is a single attribute
        write, so calling it back-to-back across cameras starts them all within
        microseconds. Must be preceded by a successful prepare_recording()."""
        self.recording = True

    def start_recording(self, output_path: Path) -> bool:
        """Convenience: prepare + arm in one call (single-camera use).
        MultiCamRecorder does NOT use this; it calls prepare_recording() on
        every camera first, then arm_recording() on every camera, so the starts
        are simultaneous. Kept for backward compatibility / standalone use."""
        if not self.prepare_recording(output_path):
            return False
        self.arm_recording()
        logger.info(f"Camera {self.camera_id} recording")
        return True
    
    def disarm_recording(self) -> None:
        """Phase 1 of stopping: flip the recording flag off. This is the moment
        frame capture into the file actually ceases. A single attribute write,
        so MultiCamRecorder can disarm all cameras within microseconds of each
        other and fire the stop trigger immediately, BEFORE the slow teardown."""
        self.recording = False

    def finalize_recording(self) -> Tuple[int, List[float]]:
        """Phase 2 of stopping: the slow teardown (join writer thread, drain the
        queue, release the VideoWriter / finalise the .avi). Returns
        (frame_count, timestamps). Safe to call after disarm_recording(); also
        safe if disarm was never called (it just sets recording False again)."""
        with self._lock:
            self.recording = False

            # Signal writer thread to stop and wait for it to drain the queue
            self._writer_stop.set()
            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=5.0)

            # Drain any remaining frames in the write queue
            while not self._write_queue.empty():
                try:
                    frame, timestamp = self._write_queue.get_nowait()
                    if self.video_writer and self.video_writer.isOpened():
                        self.video_writer.write(frame)
                        self.frame_count += 1
                        self.timestamps.append(timestamp)
                except queue.Empty:
                    break
                except Exception as e:
                    logger.warning(f"Error draining write queue for camera {self.camera_id}: {e}")

            # Save stats before cleanup
            frame_count = self.frame_count
            timestamps = self.timestamps.copy()

            # Release video writer SAFELY
            if self.video_writer:
                try:
                    time.sleep(0.05)
                    if self.video_writer.isOpened():
                        self.video_writer.release()
                except Exception as e:
                    logger.warning(f"Error releasing video writer for camera {self.camera_id}: {e}")
                finally:
                    self.video_writer = None

            logger.info(f"Camera {self.camera_id} stopped recording: {frame_count} frames")

            return frame_count, timestamps

    def stop_recording(self) -> Tuple[int, List[float]]:
        """Convenience: disarm + finalize in one call (single-camera use).
        MultiCamRecorder does NOT use this; it disarms every camera first,
        fires the stop trigger, then finalizes each. Kept for compatibility."""
        self.disarm_recording()
        return self.finalize_recording()
    
    def run(self):
        """Main camera capture loop - capture thread never touches disk I/O"""
        self.running = True
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running and not self._stop_event.is_set():
            if not self.is_connected:
                time.sleep(0.1)
                continue
            
            try:
                # grab() is fast and keeps the driver buffer flowing
                grabbed = self.cap.grab()
                
                if not grabbed:
                    consecutive_failures += 1
                    logger.warning(f"Failed to grab frame from camera {self.camera_id} ({consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Camera {self.camera_id} has too many failures, stopping")
                        self.running = False
                        break
                    
                    time.sleep(0.01)
                    continue
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # retrieve() decodes the grabbed frame
                ret, frame = self.cap.retrieve()
                if not ret or frame is None:
                    continue
                
                # Record timestamp
                timestamp = time.time()
                
                # If recording, enqueue frame for the writer thread.
                # The writer thread handles all disk I/O on its own thread,
                # so periodic OS write-buffer flushes cannot stall us here.
                if self.recording:
                    try:
                        self._write_queue.put_nowait((frame, timestamp))
                    except queue.Full:
                        logger.warning(f"Camera {self.camera_id} write queue full - disk too slow")
                
                # Update latest frame for preview (atomic reference swap)
                self.last_frame = frame
                self.last_timestamp = timestamp
                        
            except Exception as e:
                logger.error(f"Error in camera {self.camera_id} loop: {e}")
                time.sleep(0.05)
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Get the latest frame (non-blocking) - returns most recent frame"""
        frame = self.last_frame
        ts = self.last_timestamp
        if frame is not None:
            return (frame, ts)
        return None
    
    def stop(self):
        """Stop the camera thread - ROBUST CLEANUP"""
        logger.info(f"Stopping camera {self.camera_id}...")
        
        # Signal thread to stop
        self.running = False
        self._stop_event.set()
        
        # Stop recording if active
        self.recording = False
        
        # Stop writer thread
        self._writer_stop.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=3.0)
        
        # Give capture thread time to finish gracefully
        time.sleep(0.2)
        
        # Clean up video writer (writer thread is already stopped)
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


class MultiCamRecorder:
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
                 base_output_dir: str = "./recordings",
                 use_datapixx: bool = None,
                 trigger: int = 1):
        """
        Initialize MultiCamRecorder
        
        Args:
            camera_ids: List of camera IDs to use (e.g., [0, 1, 2])
            width: Frame width
            height: Frame height
            fps: Frames per second
            base_output_dir: Base directory for recordings
            use_datapixx: If True, send DataPixx digital triggers on
                          recording start/stop (requires pypixxlib).
                          If None (default), auto-enables when trigger
                          is passed and pypixxlib is available.
            trigger: DataPixx trigger number (1, 2, 3, etc.)
        """
        self.camera_ids = camera_ids or [0]
        self.width = width
        self.height = height
        self.fps = fps
        self.base_output_dir = Path(base_output_dir)
        self.trigger = trigger
        
        # Auto-detect: if use_datapixx not explicitly set, enable when
        # pypixxlib is available (user passing trigger= implies intent)
        if use_datapixx is None:
            self.use_datapixx = _dpx_available
        else:
            self.use_datapixx = use_datapixx
        
        self.cameras: Dict[int, CameraThread] = {}
        self.is_recording = False
        self.current_trial_name = None
        self.current_trial_dir = None
        
        # Open DataPixx connection if enabled
        if self.use_datapixx:
            _dpx_open()
        
        logger.info(f"MultiCamRecorder initialized with cameras: {self.camera_ids}"
                     + (f", DataPixx trigger: {self.trigger}" if self.use_datapixx else ""))
    
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
        
        # --- Phase 1: PREPARE every camera (slow: opens VideoWriters etc.) ---
        # Done sequentially, but no frames are captured yet, so the time taken
        # here does NOT stagger the recordings.
        prepared = []
        for cam_id, camera in self.cameras.items():
            output_path = self.current_trial_dir / f"camera_{cam_id}.avi"  # .avi for MJPEG
            if camera.prepare_recording(output_path):
                prepared.append(camera)
            else:
                logger.error(f"Camera {cam_id} failed to prepare; it will not record this trial")

        if not prepared:
            logger.error("Failed to prepare recording on any camera")
            return False

        # --- Phase 2: ARM every prepared camera as close to simultaneously as
        # possible. arm_recording() is a single attribute write, so this loop
        # flips all cameras live within microseconds of each other. ---
        for camera in prepared:
            camera.arm_recording()

        self.is_recording = True

        # --- Trigger fires IMMEDIATELY after all cameras are live, so the OPM
        # start trigger marks t=0 for the (now simultaneous) recordings. ---
        if self.use_datapixx:
            logger.info(f"Sending DataPixx trigger {self.trigger} pulse (start)")
            _dpx_send_trigger(self.trigger)
            threading.Timer(0.1, _dpx_send_trigger, args=(0,)).start()

        logger.info(f"Recording started on {len(prepared)} cameras")
        return True
    
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

        # --- Phase 1: DISARM every camera near-simultaneously. This is the
        # moment capture into the files actually stops. Single attribute writes,
        # so all cameras cease within microseconds of each other. ---
        for cam_id, camera in self.cameras.items():
            camera.disarm_recording()

        self.is_recording = False

        # --- Trigger fires IMMEDIATELY, right after capture stops, BEFORE the
        # slow teardown, so the OPM stop edge marks when recording truly ended,
        # not when the files finished closing. ---
        if self.use_datapixx:
            logger.info(f"Sending DataPixx trigger {self.trigger} pulse (stop)")
            _dpx_send_trigger(self.trigger)
            threading.Timer(0.1, _dpx_send_trigger, args=(0,)).start()

        # --- Phase 2: FINALIZE every camera (slow: join writer thread, drain
        # queue, release/finalise the .avi). Happens AFTER the trigger, so its
        # duration no longer inflates the trigger-to-trigger interval. ---
        results = {}
        for cam_id, camera in self.cameras.items():
            try:
                frame_count, timestamps = camera.finalize_recording()
                results[cam_id] = (frame_count, timestamps)
            except Exception as e:
                logger.error(f"Error finalizing recording for camera {cam_id}: {e}")
                results[cam_id] = (0, [])

        # Save metadata
        try:
            self._save_metadata(results)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

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
        
        # Close DataPixx connection if it was opened
        if self.use_datapixx:
            _dpx_close()
    
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
                        
                        # Add simple dark background for text (no expensive alpha blend)
                        text_bg_width = int(min(preview_width - 10, 450 * preview_scale))
                        text_bg_height = int(120 * preview_scale)
                        cv2.rectangle(frame_preview, (0, 0), (text_bg_width, text_bg_height), (0, 0, 0), -1)
                        
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
def create_recorder(*args, **kwargs) -> MultiCamRecorder:
    """
    Create a recorder instance - use with context manager for automatic cleanup
    
    Example (without DataPixx):
        with create_recorder(camera_ids=[0, 1]) as recorder:
            recorder.connect_cameras()
            recorder.start_recording("test")
            time.sleep(5)
            recorder.stop_recording()
        # Automatic cleanup on exit
    
    Example (with DataPixx trigger):
        with create_recorder(camera_ids=[0, 1], use_datapixx=True, trigger=1) as recorder:
            recorder.connect_cameras()
            recorder.start_recording("test")   # trigger ON
            time.sleep(5)
            recorder.stop_recording()           # trigger OFF
        # Automatic cleanup + DataPixx close on exit
    """
    return MultiCamRecorder(*args, **kwargs)