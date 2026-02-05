"""
Video Synchronization Module
Based on skelly_synchronize approach using audio cross-correlation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from scipy import signal
from scipy.io import wavfile
import subprocess
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSynchronizer:
    """
    Synchronize multiple videos using audio cross-correlation
    
    This class implements the same synchronization approach as skelly_synchronize:
    - Extract audio from videos
    - Use cross-correlation to find time offsets
    - Trim videos to synchronized start/end points
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize video synchronizer
        
        Args:
            sample_rate: Audio sample rate for synchronization
        """
        self.sample_rate = sample_rate
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True)
            logger.info("FFmpeg found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not found! Please install FFmpeg for audio extraction.")
            raise RuntimeError("FFmpeg is required for video synchronization")
    
    def synchronize_videos(self, 
                          video_dir: Path,
                          output_dir: Optional[Path] = None,
                          method: str = 'audio') -> Dict[str, any]:
        """
        Synchronize all videos in a directory
        
        Args:
            video_dir: Directory containing raw videos
            output_dir: Output directory for synchronized videos
            method: Synchronization method ('audio' or 'brightness')
            
        Returns:
            Dictionary with synchronization results
        """
        video_dir = Path(video_dir)
        
        if output_dir is None:
            output_dir = video_dir.parent / "synchronized_videos"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all video files
        video_files = self._find_video_files(video_dir)
        
        if len(video_files) < 2:
            logger.error("Need at least 2 videos to synchronize!")
            return {}
        
        logger.info(f"Found {len(video_files)} videos to synchronize")
        
        if method == 'audio':
            return self._synchronize_by_audio(video_files, output_dir)
        elif method == 'brightness':
            return self._synchronize_by_brightness(video_files, output_dir)
        else:
            raise ValueError(f"Unknown synchronization method: {method}")
    
    def _find_video_files(self, directory: Path) -> List[Path]:
        """Find all video files in directory"""
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mpeg']
        video_files = []
        
        for ext in extensions:
            video_files.extend(directory.glob(f"*{ext}"))
        
        return sorted(video_files)
    
    def _extract_audio(self, video_path: Path, output_path: Path) -> bool:
        """Extract audio from video using FFmpeg"""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                str(output_path)
            ]
            
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg warning for {video_path.name}: {result.stderr}")
            
            return output_path.exists()
            
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            return False
    
    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            sample_rate, audio_data = wavfile.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Normalize
            audio_data = audio_data.astype(np.float32)
            audio_data /= np.max(np.abs(audio_data))
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return np.array([]), 0
    
    def _compute_offset(self, 
                       audio1: np.ndarray, 
                       audio2: np.ndarray,
                       sample_rate: int) -> float:
        """
        Compute time offset between two audio signals using cross-correlation
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal  
            sample_rate: Audio sample rate
            
        Returns:
            Time offset in seconds (positive means audio2 starts after audio1)
        """
        # Compute cross-correlation
        correlation = signal.correlate(audio1, audio2, mode='full')
        
        # Find peak
        lags = signal.correlation_lags(len(audio1), len(audio2), mode='full')
        peak_idx = np.argmax(correlation)
        lag_samples = lags[peak_idx]
        
        # Convert to time
        offset_seconds = lag_samples / sample_rate
        
        return offset_seconds
    
    def _synchronize_by_audio(self, 
                             video_files: List[Path],
                             output_dir: Path) -> Dict[str, any]:
        """Synchronize videos using audio cross-correlation"""
        logger.info("Synchronizing by audio cross-correlation...")
        
        # Create temp directory for audio files
        temp_dir = output_dir / "temp_audio"
        temp_dir.mkdir(exist_ok=True)
        
        # Extract audio from all videos
        audio_files = []
        audio_data = []
        
        for video_file in video_files:
            audio_path = temp_dir / f"{video_file.stem}.wav"
            
            logger.info(f"Extracting audio from {video_file.name}...")
            if self._extract_audio(video_file, audio_path):
                audio_files.append(audio_path)
                
                # Load audio
                audio, sr = self._load_audio(audio_path)
                audio_data.append(audio)
            else:
                logger.error(f"Failed to extract audio from {video_file.name}")
                return {}
        
        if len(audio_data) < 2:
            logger.error("Not enough valid audio tracks for synchronization")
            return {}
        
        # Compute offsets relative to first video
        logger.info("Computing time offsets...")
        reference_audio = audio_data[0]
        offsets = [0.0]  # First video is reference
        
        for i in range(1, len(audio_data)):
            offset = self._compute_offset(reference_audio, audio_data[i], self.sample_rate)
            offsets.append(offset)
            logger.info(f"  {video_files[i].name}: {offset:.3f}s offset")
        
        # Find earliest start and latest end
        min_offset = min(offsets)
        
        # Adjust all offsets to start at earliest time
        adjusted_offsets = [offset - min_offset for offset in offsets]
        
        # Get video durations
        durations = []
        for video_file in video_files:
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            durations.append(duration)
            cap.release()
        
        # Find common duration (latest start to earliest end)
        end_times = [adjusted_offsets[i] + durations[i] for i in range(len(video_files))]
        common_end = min(end_times)
        
        # Trim videos to synchronized times
        logger.info("Trimming videos to synchronized times...")
        results = {
            'offsets': {},
            'output_files': []
        }
        
        for i, video_file in enumerate(video_files):
            start_time = adjusted_offsets[i]
            duration = common_end - start_time
            
            output_path = output_dir / f"synced_{video_file.name}"
            
            logger.info(f"  {video_file.name}: trim from {start_time:.3f}s, duration {duration:.3f}s")
            
            # Trim video using FFmpeg
            self._trim_video(video_file, output_path, start_time, duration)
            
            results['offsets'][video_file.name] = start_time
            results['output_files'].append(output_path)
        
        # Save synchronization report
        self._save_sync_report(output_dir, results, adjusted_offsets, video_files)
        
        # Clean up temp audio files
        shutil.rmtree(temp_dir)
        
        logger.info(f"Synchronization complete! Videos saved to {output_dir}")
        
        return results
    
    def _trim_video(self, 
                   input_path: Path,
                   output_path: Path,
                   start_time: float,
                   duration: float):
        """Trim video using FFmpeg"""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # Copy codec (fast)
                '-y',
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error trimming {input_path}: {e.stderr}")
    
    def _synchronize_by_brightness(self,
                                   video_files: List[Path],
                                   output_dir: Path,
                                   threshold: float = 50.0) -> Dict[str, any]:
        """
        Synchronize videos by detecting brightness flash
        
        Args:
            video_files: List of video files
            output_dir: Output directory
            threshold: Brightness change threshold
        """
        logger.info("Synchronizing by brightness detection...")
        
        flash_times = []
        
        for video_file in video_files:
            logger.info(f"Detecting flash in {video_file.name}...")
            flash_frame = self._detect_brightness_flash(video_file, threshold)
            
            if flash_frame is not None:
                cap = cv2.VideoCapture(str(video_file))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                flash_time = flash_frame / fps if fps > 0 else 0
                flash_times.append(flash_time)
                logger.info(f"  Flash detected at frame {flash_frame} ({flash_time:.3f}s)")
            else:
                logger.error(f"  No flash detected in {video_file.name}")
                return {}
        
        if len(flash_times) != len(video_files):
            logger.error("Could not detect flash in all videos")
            return {}
        
        # Compute offsets
        min_flash_time = min(flash_times)
        adjusted_offsets = [ft - min_flash_time for ft in flash_times]
        
        # Trim videos
        results = {
            'offsets': {},
            'output_files': []
        }
        
        for i, video_file in enumerate(video_files):
            start_time = adjusted_offsets[i]
            
            output_path = output_dir / f"synced_{video_file.name}"
            
            # For brightness sync, we trim from the flash point
            self._trim_from_point(video_file, output_path, start_time)
            
            results['offsets'][video_file.name] = start_time
            results['output_files'].append(output_path)
        
        self._save_sync_report(output_dir, results, adjusted_offsets, video_files)
        
        logger.info(f"Synchronization complete! Videos saved to {output_dir}")
        
        return results
    
    def _detect_brightness_flash(self, 
                                 video_path: Path,
                                 threshold: float = 50.0) -> Optional[int]:
        """Detect first significant brightness increase"""
        cap = cv2.VideoCapture(str(video_path))
        
        prev_brightness = None
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Compute mean brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if prev_brightness is not None:
                change = brightness - prev_brightness
                
                if change > threshold:
                    cap.release()
                    return frame_num
            
            prev_brightness = brightness
            frame_num += 1
        
        cap.release()
        return None
    
    def _trim_from_point(self,
                        input_path: Path,
                        output_path: Path,
                        start_time: float):
        """Trim video from a specific start point to end"""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start_time),
                '-c', 'copy',
                '-y',
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error trimming {input_path}: {e.stderr}")
    
    def _save_sync_report(self,
                         output_dir: Path,
                         results: Dict,
                         offsets: List[float],
                         video_files: List[Path]):
        """Save synchronization report"""
        report_path = output_dir / "synchronization_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Video Synchronization Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Time Offsets (relative to earliest start):\n")
            for i, video_file in enumerate(video_files):
                f.write(f"  {video_file.name}: {offsets[i]:.3f}s\n")
            
            f.write(f"\nOutput files:\n")
            for output_file in results['output_files']:
                f.write(f"  {output_file.name}\n")
        
        logger.info(f"Synchronization report saved to {report_path}")
