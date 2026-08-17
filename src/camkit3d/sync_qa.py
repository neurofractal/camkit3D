"""Quality assurance for synchronized multi-camera videos in CamKit3D.

After synchronisation, this module verifies that the output videos really are
aligned and gives you tools to inspect them by eye.

Key features:

- Sync validation. Re-opens every synchronised video and checks that frame
  counts, durations, frame rates, and resolutions match across cameras,
  flagging any stream that drifted out of agreement.
- Side-by-side review. Composites all camera feeds into a single labelled
  grid video so misalignment is visible at a glance.
- Offset visualisation. Parses the synchronisation report and plots the
  residual timing offset of each camera.

Author: Dr. Robert Seymour, OHBA, University of Oxford
License: GNU General Public License v3, 2026
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """Analyze and validate synchronized videos"""
    
    def __init__(self):
        pass
    
    def analyze_synchronization(self, video_dir: Path) -> Dict:
        """
        Analyze synchronized videos for quality metrics
        
        Args:
            video_dir: Directory containing synchronized videos
            
        Returns:
            Dictionary with analysis results
        """
        video_dir = Path(video_dir)
        
        # Find all video files
        video_files = sorted(video_dir.glob("*.mp4"))
        
        if len(video_files) < 2:
            logger.error("Need at least 2 videos to analyze")
            return {}
        
        logger.info(f"Analyzing {len(video_files)} videos...")
        
        results = {
            'videos': [],
            'frame_counts': [],
            'durations': [],
            'fps': [],
            'resolutions': [],
        }
        
        # Analyze each video
        for video_file in video_files:
            info = self._get_video_info(video_file)
            
            results['videos'].append(video_file.name)
            results['frame_counts'].append(info['frame_count'])
            results['durations'].append(info['duration'])
            results['fps'].append(info['fps'])
            results['resolutions'].append(info['resolution'])
        
        # Check consistency
        results['sync_quality'] = self._check_sync_quality(results)
        
        # Print report
        self._print_analysis_report(results)
        
        return results
    
    def _get_video_info(self, video_path: Path) -> Dict:
        """Get information about a video file"""
        cap = cv2.VideoCapture(str(video_path))
        
        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        
        info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
        info['resolution'] = (info['width'], info['height'])
        
        cap.release()
        
        return info
    
    def _check_sync_quality(self, results: Dict) -> Dict:
        """Check synchronization quality"""
        quality = {
            'frame_count_match': False,
            'duration_match': False,
            'fps_match': False,
            'resolution_match': False,
        }
        
        # Check if all videos have same properties
        frame_counts = results['frame_counts']
        durations = results['durations']
        fps_values = results['fps']
        resolutions = results['resolutions']
        
        # Frame count should match exactly
        quality['frame_count_match'] = len(set(frame_counts)) == 1
        
        # Duration should be very close (within 0.1s)
        if durations:
            max_dur = max(durations)
            min_dur = min(durations)
            quality['duration_match'] = (max_dur - min_dur) < 0.1
        
        # FPS should match
        quality['fps_match'] = len(set(fps_values)) == 1
        
        # Resolution should match
        quality['resolution_match'] = len(set(resolutions)) == 1
        
        return quality
    
    def _print_analysis_report(self, results: Dict):
        """Print analysis report"""
        print("\n" + "="*70)
        print("VIDEO SYNCHRONIZATION ANALYSIS")
        print("="*70)
        
        print(f"\nVideos analyzed: {len(results['videos'])}")
        
        for i, video in enumerate(results['videos']):
            print(f"\n  {video}:")
            print(f"    Frames: {results['frame_counts'][i]}")
            print(f"    Duration: {results['durations'][i]:.2f}s")
            print(f"    FPS: {results['fps'][i]:.1f}")
            print(f"    Resolution: {results['resolutions'][i][0]}x{results['resolutions'][i][1]}")
        
        print("\n" + "-"*70)
        print("SYNCHRONIZATION QUALITY")
        print("-"*70)
        
        quality = results['sync_quality']
        
        for check, passed in quality.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}")
        
        all_passed = all(quality.values())
        
        print("\n" + "-"*70)
        if all_passed:
            print("✓ All quality checks passed - videos are well synchronized")
        else:
            print("⚠ Some quality checks failed - review synchronization")
        print("="*70 + "\n")
    
    def create_side_by_side_comparison(self,
                                       video_files: List[Path],
                                       output_path: Path,
                                       duration: float = 10.0):
        """
        Create a side-by-side comparison video
        
        Args:
            video_files: List of video files to compare
            output_path: Output video path
            duration: Duration of comparison video in seconds
        """
        if len(video_files) < 2:
            logger.error("Need at least 2 videos")
            return
        
        logger.info(f"Creating side-by-side comparison of {len(video_files)} videos...")
        
        # Open all videos
        caps = [cv2.VideoCapture(str(vf)) for vf in video_files]
        
        # Get properties from first video
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video
        # Arrange videos in a grid
        n_videos = len(video_files)
        grid_cols = int(np.ceil(np.sqrt(n_videos)))
        grid_rows = int(np.ceil(n_videos / grid_cols))
        
        out_width = width * grid_cols
        out_height = height * grid_rows
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
        
        frame_count = 0
        max_frames = int(duration * fps)
        
        while frame_count < max_frames:
            # Read frames from all videos
            frames = []
            all_ok = True
            
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    all_ok = False
                    break
                frames.append(frame)
            
            if not all_ok:
                break
            
            # Create grid
            grid_frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            
            for i, frame in enumerate(frames):
                row = i // grid_cols
                col = i % grid_cols
                
                y_start = row * height
                x_start = col * width
                
                grid_frame[y_start:y_start+height, x_start:x_start+width] = frame
                
                # Add label
                label = f"Camera {i}"
                cv2.putText(grid_frame, label, 
                           (x_start + 10, y_start + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(grid_frame)
            frame_count += 1
        
        # Cleanup
        for cap in caps:
            cap.release()
        out.release()
        
        logger.info(f"Comparison video saved to {output_path}")
    
    def visualize_sync_offsets(self, sync_report_path: Path):
        """
        Visualize synchronization offsets from report
        
        Args:
            sync_report_path: Path to synchronization_report.txt
        """
        # Parse report file
        offsets = {}
        
        with open(sync_report_path, 'r') as f:
            in_offsets = False
            for line in f:
                if 'Time Offsets' in line:
                    in_offsets = True
                    continue
                
                if in_offsets and ':' in line:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        offset = float(parts[1].strip().replace('s', ''))
                        offsets[name] = offset
        
        if not offsets:
            logger.error("No offsets found in report")
            return
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        names = list(offsets.keys())
        values = list(offsets.values())
        
        plt.bar(names, values)
        plt.xlabel('Video')
        plt.ylabel('Time Offset (seconds)')
        plt.title('Video Synchronization Offsets')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = sync_report_path.parent / 'sync_offsets_plot.png'
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        logger.info(f"Offset plot saved to {output_path}")


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sync_qa.py <synchronized_videos_directory>")
        return
    
    video_dir = Path(sys.argv[1])
    
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return
    
    # Analyze synchronization
    analyzer = VideoAnalyzer()
    results = analyzer.analyze_synchronization(video_dir)
    
    # Create side-by-side comparison
    video_files = sorted(video_dir.glob("*.mp4"))
    if len(video_files) >= 2:
        print("\nCreating side-by-side comparison...")
        comparison_path = video_dir / "comparison_video.mp4"
        analyzer.create_side_by_side_comparison(
            video_files,
            comparison_path,
            duration=5.0
        )
    
    # Visualize offsets if report exists
    report_path = video_dir / "synchronization_report.txt"
    if report_path.exists():
        print("\nCreating offset visualization...")
        analyzer.visualize_sync_offsets(report_path)


if __name__ == "__main__":
    main()