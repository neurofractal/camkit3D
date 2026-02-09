import argparse
from multicam_recorder import MultiCameraRecorder

# Set up argument parser
parser = argparse.ArgumentParser(description='Preview multiple cameras')
parser.add_argument('--duration', '-d', type=int, default=10, 
                    help='Duration to display preview in seconds (default: 10)')
parser.add_argument('--fps', '-f', type=int, default=30,
                    help='Target FPS for preview (default: 30)')

args = parser.parse_args()

# Run the recorder with specified duration
recorder = MultiCameraRecorder(camera_ids=[0, 1, 2], fps=args.fps)
recorder.connect_cameras()
recorder.preview_cameras(duration=args.duration, target_fps=args.fps)
recorder.disconnect_cameras()