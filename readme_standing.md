# Standing/Sitting Audio Experiment with Multi-Camera Recording

A PsychoPy-based behavioral experiment for recording standing-to-sitting transitions with synchronized multi-camera video and audio cues.

## Overview

This experiment records participants as they:
1. Stand at the beginning of each trial (cued by `standing.wav`)
2. Sit down when ready (pressing '1' key triggers `sit.wav`)
3. Complete the trial when experimenter presses SPACE

All trials are recorded with synchronized multi-camera video, with timestamps logged for all events.

## Features

- **Multi-camera synchronized recording** - Record from multiple webcams simultaneously
- **Audio cues** - Custom WAV files played at trial start and participant response
- **Precise timing** - Both Unix timestamps and PsychoPy clock times logged
- **Visual countdown** - Inter-trial intervals display remaining time
- **Detailed logging** - CSV file with all event timestamps
- **Flexible configuration** - Command-line arguments for all parameters
- **User-friendly interface** - Clear on-screen instructions and progress updates

## Requirements

### Python Environment

```bash
# Create a conda environment (recommended)
conda create -n psychocam python=3.10
conda activate psychocam

# Install required packages
pip install psychopy numpy sounddevice soundfile
```

### Additional Requirements

1. **Camera access** - One or more webcams with proper permissions
2. **Audio files** - `standing.wav` and `sit.wav` (mono, 48kHz recommended)
3. **MultiCameraRecorder** - The `multicam_recorder.py` module must be in the same directory

### Dependencies

- **PsychoPy** - Stimulus presentation and event handling
- **NumPy** - Numerical operations
- **SoundDevice** - Audio playback
- **SoundFile** - WAV file loading
- **OpenCV** (via MultiCameraRecorder) - Video capture

## Installation

1. Clone or download this repository
2. Ensure `multicam_recorder.py` is in the same directory as `standing_audio.py`
3. Create your audio files (`standing.wav` and `sit.wav`)
4. Install dependencies (see Requirements above)

## Usage

### Basic Usage

Run with default settings (camera 0, 5 trials, output to `./recordings`):

```bash
python standing_audio.py
```

### Common Use Cases

**Single camera, 10 trials:**
```bash
python standing_audio.py --cameras 0 --trials 10
```

**Multiple cameras:**
```bash
python standing_audio.py --cameras 0 1 2 --trials 5
```

**Custom output directory:**
```bash
python standing_audio.py --output ./my_experiment_data
```

**High-resolution recording:**
```bash
python standing_audio.py --width 1920 --height 1080 --fps 60
```

**Custom inter-trial intervals:**
```bash
python standing_audio.py --iti-min 3.0 --iti-max 7.0
```

**Custom audio files:**
```bash
python standing_audio.py --standing-audio cue_stand.wav --sit-audio cue_sit.wav
```

### Full Configuration Example

```bash
python standing_audio.py \
    --cameras 0 1 \
    --trials 20 \
    --output ./experiment_2024 \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --iti-min 4.0 \
    --iti-max 8.0 \
    --standing-audio standing_cue.wav \
    --sit-audio sitting_cue.wav
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cameras` | int(s) | `[0]` | Camera IDs to use (space-separated for multiple) |
| `--trials` | int | `5` | Number of trials to run |
| `--output` | str | `./recordings` | Output directory for recordings and events |
| `--width` | int | `1280` | Camera resolution width |
| `--height` | int | `720` | Camera resolution height |
| `--fps` | int | `30` | Frames per second for video recording |
| `--iti-min` | float | `5.0` | Minimum inter-trial interval (seconds) |
| `--iti-max` | float | `6.0` | Maximum inter-trial interval (seconds) |
| `--standing-audio` | str | `standing.wav` | Path to standing audio cue file |
| `--sit-audio` | str | `sit.wav` | Path to sitting audio cue file |

## Audio File Requirements

Audio files should be:
- **Format**: WAV (uncompressed)
- **Channels**: Mono (stereo will be converted automatically)
- **Sample Rate**: 48kHz (other rates will trigger a warning but still work)
- **Duration**: Keep cues short (0.5-2 seconds recommended)

### Creating Audio Files

You can create simple beep tones using Python:

```python
import numpy as np
import soundfile as sf

# Create a 800 Hz beep for standing cue
fs = 48000
duration = 1.0  # seconds
t = np.linspace(0, duration, int(fs * duration))
standing = 0.3 * np.sin(2 * np.pi * 800 * t)
sf.write('standing.wav', standing, fs)

# Create a 600 Hz beep for sitting cue
sitting = 0.3 * np.sin(2 * np.pi * 600 * t)
sf.write('sit.wav', sitting, fs)
```

Or use any audio editing software (Audacity, Adobe Audition, etc.) to create or import your cues.

## Output Structure

The experiment creates the following directory structure:

```
recordings/                           # Base output directory
├── events_20240209_201056.csv       # Event log with all timestamps
├── trial_001_2024-02-09_20-10-56/   # Trial 1 directory
│   └── raw_videos/
│       ├── camera_0.avi             # Camera 0 video
│       └── camera_1.avi             # Camera 1 video (if multiple cameras)
├── trial_002_2024-02-09_20-11-13/   # Trial 2 directory
│   └── raw_videos/
│       ├── camera_0.avi
│       └── camera_1.avi
└── ...
```

### Events CSV Format

The events CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `trial` | Trial number (1, 2, 3, ...) |
| `camera_ids` | Comma-separated list of camera IDs used |
| `trial_name` | Unique trial identifier with timestamp |
| `trial_start_unix` | Trial start time (Unix timestamp) |
| `trial_start_psychopy` | Trial start time (PsychoPy clock) |
| `standing_audio_unix` | Standing cue playback time (Unix) |
| `standing_audio_psychopy` | Standing cue playback time (PsychoPy) |
| `sit_key_unix` | Time when '1' key was pressed (Unix) |
| `sit_key_psychopy` | Time when '1' key was pressed (PsychoPy) |
| `sit_audio_unix` | Sitting cue playback time (Unix) |
| `sit_audio_psychopy` | Sitting cue playback time (PsychoPy) |
| `stop_space_unix` | Trial stop time (Unix) |
| `stop_space_psychopy` | Trial stop time (PsychoPy) |
| `iti_s` | Inter-trial interval duration (seconds) |

**Note**: If participant doesn't press '1' during a trial, the `sit_key_*` and `sit_audio_*` columns will be empty for that trial.

## Experiment Workflow

### 1. Preparation
- Launch the script with desired parameters
- Audio files are loaded and validated
- Cameras are initialized and connected

### 2. Instructions Screen
Displays experiment overview and controls:
- Trial sequence explanation
- Output directory location
- Keyboard controls

### 3. Trial Sequence

For each trial:

1. **Recording starts** automatically
   - `standing.wav` plays immediately
   - On-screen display shows trial progress and file location
   
2. **Participant stands and performs task**
   - When ready to sit, participant presses '1'
   - `sit.wav` plays as confirmation
   
3. **Experimenter ends trial**
   - Presses SPACE when trial should end
   - Recording stops
   
4. **Inter-trial interval**
   - Countdown timer displays remaining time
   - Duration randomly selected from `iti_range`

### 4. Completion
- Summary of completed trials
- File locations displayed
- Cameras gracefully disconnected

## Keyboard Controls

| Key | Function | Who |
|-----|----------|-----|
| **SPACE** | Start experiment / End current trial | Experimenter |
| **1** | Signal sitting (triggers sit.wav) | Participant |
| **ESC** | Abort experiment (at any time) | Either |

## Troubleshooting

### No Cameras Connected

**Symptoms**: "NO CAMERAS CONNECTED" message on startup

**Solutions**:
1. Check camera IDs are correct (try `--cameras 0` for built-in webcam)
2. Ensure no other applications are using the cameras
3. Grant camera permissions to your terminal/Python
4. On macOS: System Settings → Privacy & Security → Camera
5. On Linux: Check user is in `video` group

### Audio Files Not Found

**Symptoms**: "ERROR: Could not load 'standing.wav'" message

**Solutions**:
1. Ensure `standing.wav` and `sit.wav` are in the same directory as the script
2. Or specify full paths: `--standing-audio /path/to/standing.wav`
3. Check file permissions are readable

### Video Recording Issues

**Symptoms**: Recording fails to start or crashes mid-trial

**Solutions**:
1. Reduce resolution: `--width 640 --height 480`
2. Lower frame rate: `--fps 15`
3. Check available disk space
4. Close other applications using cameras
5. Try fewer cameras

### PsychoPy Window Issues

**Symptoms**: Window doesn't appear or appears on wrong monitor

**Solutions**:
1. The window is not fullscreen by default (set `fullscr=True` in code if needed)
2. Adjust window size in `visual.Window(size=(1000, 700))`
3. Drag window to desired monitor before starting

### High CPU Usage

**Solutions**:
1. Reduce number of cameras
2. Lower resolution and frame rate
3. Close unnecessary applications
4. Check `multicam_recorder.py` is using efficient encoding

## Data Analysis

### Loading Event Data

```python
import pandas as pd

# Load events CSV
events = pd.read_csv('recordings/events_20240209_201056.csv')

# Calculate reaction times (time from standing cue to sitting)
events['reaction_time'] = events['sit_key_psychopy'] - events['standing_audio_psychopy']

# Display summary
print(events[['trial', 'reaction_time']].describe())
```

## Advanced Configuration

### Modifying the Script

The script is designed to be easily modified. Key sections:

**Change visual appearance:**
```python
# Line ~265 - Window setup
win = visual.Window(
    size=(1000, 700),  # Change window size
    fullscr=False,      # Set True for fullscreen
    units="pix",
    color=(0, 0, 0)     # Background color (R, G, B)
)
```

**Change text formatting:**
```python
# Line ~270 - Text stimulus
msg = visual.TextStim(
    win, 
    text="", 
    height=24,          # Font size
    color="white",      # Text color
    wrapWidth=900,      # Text wrapping width
    alignText='center'
)
```

**Add custom events:**
```python
# In the trial loop, add additional keyboard monitoring:
if "2" in keys:
    custom_event_time = clock.getTime()
    # Log this to CSV or process as needed
```

## Technical Details

### Timing Precision

- **PsychoPy clock**: High-precision timer reset at experiment start
- **Unix timestamps**: Absolute time for cross-referencing with other systems
- **Audio latency**: ~5-20ms depending on system (not compensated)
- **Video timestamps**: Each frame has associated timestamp in video metadata

### Video Format

- **Container**: AVI
- **Codec**: System-dependent (usually MJPEG or H264)
- **Resolution**: Configurable via `--width` and `--height`
- **Frame rate**: Configurable via `--fps`

### Thread Safety

The MultiCameraRecorder runs each camera in a separate thread for optimal performance. The main PsychoPy loop runs in the main thread for accurate timing.

## Citation

If you use this code in your research, please cite:

```
[Your Name/Lab] (2024). Standing/Sitting Audio Experiment with Multi-Camera Recording.
GitHub: [your-repo-url]
```

## License

[Specify your license here, e.g., MIT, GPL, etc.]

## Credits

- Built with [PsychoPy](https://www.psychopy.org/)
- Audio playback via [python-sounddevice](https://python-sounddevice.readthedocs.io/)
- Video recording with OpenCV

## Support

For issues, questions, or contributions:
- Open an issue on GitHub: [your-repo-url]
- Email: [your-email]

## Changelog

### Version 1.0.0 (2024-02-09)
- Initial release
- Multi-camera synchronized recording
- Custom audio cue playback
- Real-time countdown display
- Comprehensive event logging
- Command-line interface

## Frequently Asked Questions

**Q: Can I use different audio formats (MP3, M4A, etc.)?**  
A: The script expects WAV files. Convert other formats using tools like `ffmpeg`:
```bash
ffmpeg -i input.mp3 -ar 48000 -ac 1 output.wav
```

**Q: How do I find my camera IDs?**  
A: Run this Python snippet:
```python
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```

**Q: Can I run this without a display (headless)?**  
A: No, PsychoPy requires a display. For headless recording, you'd need to modify the script significantly.

**Q: How much disk space do I need?**  
A: Approximately:
- 1280x720 @ 30fps = ~50-100 MB/minute per camera
- 1920x1080 @ 60fps = ~200-400 MB/minute per camera

**Q: Can participants see the countdown?**  
A: The countdown is only shown during inter-trial intervals, not during active trials.

**Q: What if the participant forgets to press '1'?**  
A: The trial continues normally. The `sit_key_*` columns in the CSV will be empty, but video is still recorded.

---

**Last updated**: February 2024  
**Version**: 1.0.0
