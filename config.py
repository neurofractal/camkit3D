"""
Configuration file for multi-camera recording experiments
Copy and modify this for your specific experimental setup
"""

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

# List of camera IDs to use
# Run test_system.py to detect available cameras
CAMERA_IDS = [0, 1]

# Video resolution
WIDTH = 1280
HEIGHT = 720

# Frame rate (fps)
FPS = 30

# ============================================================================
# RECORDING CONFIGURATION
# ============================================================================

# Base output directory for all recordings
OUTPUT_DIR = "./recordings"

# Trial naming scheme
# Options: 'timestamp', 'sequential', 'custom'
TRIAL_NAMING = 'sequential'

# Trial name prefix (for sequential naming)
TRIAL_PREFIX = 'trial'

# ============================================================================
# SYNCHRONIZATION CONFIGURATION
# ============================================================================

# Synchronization method
# Options: 'audio', 'brightness'
SYNC_METHOD = 'audio'

# Audio sample rate for synchronization
AUDIO_SAMPLE_RATE = 44100

# Brightness threshold for flash detection (if using brightness method)
BRIGHTNESS_THRESHOLD = 50.0

# ============================================================================
# EXPERIMENT PROTOCOL
# ============================================================================

# Number of trials per session
NUM_TRIALS = 10

# Recording duration per trial (seconds)
TRIAL_DURATION = 30.0

# Inter-trial interval (seconds)
ITI = 5.0

# ============================================================================
# POST-PROCESSING
# ============================================================================

# Auto-synchronize after recording session
AUTO_SYNC = True

# Create quality check reports
CREATE_REPORTS = True

# Create side-by-side comparison videos
CREATE_COMPARISON = True

# Comparison video duration (seconds)
COMPARISON_DURATION = 5.0

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Camera backend (Windows)
# Options: 'dshow', 'auto'
CAMERA_BACKEND = 'dshow'

# Buffer size for frame queue (frames)
FRAME_BUFFER_SIZE = 60

# Enable preview windows during recording
ENABLE_PREVIEW = False

# Save raw timestamps
SAVE_TIMESTAMPS = True

# Codec for video encoding
# Options: 'mp4v', 'h264', 'xvid'
VIDEO_CODEC = 'mp4v'

# ============================================================================
# EXAMPLE CONFIGURATIONS
# ============================================================================

# Example 1: High-quality motion capture
"""
CAMERA_IDS = [0, 1, 2, 3]
WIDTH = 1920
HEIGHT = 1080
FPS = 60
TRIAL_DURATION = 60.0
"""

# Example 2: Long-term monitoring
"""
CAMERA_IDS = [0, 1]
WIDTH = 640
HEIGHT = 480
FPS = 15
TRIAL_DURATION = 300.0  # 5 minutes
"""

# Example 3: Multi-angle capture
"""
CAMERA_IDS = [0, 1, 2]
WIDTH = 1280
HEIGHT = 720
FPS = 30
SYNC_METHOD = 'brightness'  # Use flash for sync
"""
