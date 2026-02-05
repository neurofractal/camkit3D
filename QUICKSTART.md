# Quick Start Guide

## Multi-Camera Synchronized Recording System

This system allows you to record synchronized videos from multiple webcams with OFFLINE synchronization - perfect for neuroscience experiments with multiple trials.

---

## 🚀 Getting Started in 3 Steps

### Step 1: Install

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install FFmpeg (required for synchronization)
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg  
# Linux: sudo apt install ffmpeg
```

### Step 2: Test Your Setup

```bash
python test_system.py
```

This will check:
- ✅ Python version
- ✅ Dependencies installed
- ✅ FFmpeg available
- ✅ Cameras detected

### Step 3: Run a Demo

```bash
python demo.py
```

Choose option [1] for a basic demonstration.

---

## 📁 What's Included

| File | Purpose |
|------|---------|
| `multicam_recorder.py` | Core multi-camera recording engine |
| `video_synchronizer.py` | Offline video synchronization (audio/brightness) |
| `demo.py` | Interactive demos showing different workflows |
| `run_experiment.py` | Automated experiment runner |
| `config.py` | Configuration file for your experiments |
| `test_system.py` | System verification tool |
| `post_processing.py` | Quality analysis and comparison tools |
| `README.md` | Full documentation |

---

## 💡 Basic Usage Example

```python
from multicam_recorder import MultiCameraRecorder
from video_synchronizer import VideoSynchronizer
import time

# 1. Setup recorder
recorder = MultiCameraRecorder(
    camera_ids=[0, 1],  # Your cameras
    fps=30
)

# 2. Connect to cameras (once)
recorder.connect_cameras()

# 3. Record Trial 1
recorder.start_recording("trial_001")
time.sleep(10)  # Record for 10 seconds
recorder.stop_recording()

# 4. Record Trial 2  
recorder.start_recording("trial_002")
time.sleep(10)
recorder.stop_recording()

# 5. Disconnect
recorder.disconnect_cameras()

# 6. OFFLINE: Synchronize
sync = VideoSynchronizer()
sync.synchronize_videos("./recordings/trial_001/raw_videos")
sync.synchronize_videos("./recordings/trial_002/raw_videos")
```

---

## 🔬 For Neuroscience Experiments

### Method 1: Manual Control (Recommended)

```bash
python demo.py
# Choose option [3] for custom interactive workflow
# This gives you full control over each recording
```

### Method 2: Automated Protocol

1. Edit `config.py` to set your parameters:
   ```python
   CAMERA_IDS = [0, 1, 2]
   NUM_TRIALS = 10
   TRIAL_DURATION = 30.0
   SYNC_METHOD = 'audio'
   ```

2. Run automated experiment:
   ```bash
   python run_experiment.py
   ```

---

## 📊 Output Structure

```
recordings/
├── trial_001/
│   ├── raw_videos/
│   │   ├── camera_0.mp4
│   │   ├── camera_1.mp4
│   │   └── camera_2.mp4
│   ├── synchronized_videos/
│   │   ├── synced_camera_0.mp4
│   │   ├── synced_camera_1.mp4
│   │   ├── synced_camera_2.mp4
│   │   ├── synchronization_report.txt
│   │   └── comparison.mp4
│   └── metadata.txt
```

---

## 🎯 Key Features

### ✅ Separate Connect/Record/Stop Functions
Perfect for loop recording in experiments:
```python
recorder.connect_cameras()    # Connect once

for trial in range(10):
    recorder.start_recording()  # Start
    # ... experiment ...
    recorder.stop_recording()   # Stop

recorder.disconnect_cameras() # Disconnect once
```

### ✅ OFFLINE Synchronization
- Record all trials first
- Synchronize afterward using audio cross-correlation
- No real-time processing overhead

### ✅ Audio Cross-Correlation Sync
- Most reliable method
- Make a loud clap at trial start
- Works even with slight camera drift

### ✅ Alternative Brightness Sync
- Use a camera flash or bright light
- Visible to all cameras simultaneously
- Good for silent environments

---

## 🔧 Troubleshooting

**No cameras detected?**
- Close other apps using cameras (Zoom, Skype, etc.)
- Try different camera IDs: `[0, 1, 2, 3, 4]`

**FFmpeg not found?**
- Install FFmpeg (see installation instructions)
- Verify: `ffmpeg -version`

**Videos not synchronized?**
- Ensure loud, distinct sound at start (for audio sync)
- Or bright flash visible to all cameras (for brightness sync)
- Check `synchronization_report.txt` for offsets

---

## 📖 Need More Help?

1. Read the full `README.md`
2. Run `python test_system.py` to diagnose issues
3. Try the demos in `demo.py`
4. Check FreeMoCap documentation: https://freemocap.github.io/documentation/

---

## 🎬 Workflow Summary

```
1. SETUP
   └─ Connect cameras (once)

2. RECORD
   ├─ Trial 1: Start → Record → Stop
   ├─ Trial 2: Start → Record → Stop
   └─ Trial N: Start → Record → Stop

3. CLEANUP
   └─ Disconnect cameras

4. SYNCHRONIZE (offline)
   ├─ Synchronize Trial 1
   ├─ Synchronize Trial 2
   └─ Synchronize Trial N

5. ANALYZE
   └─ Quality checks & comparisons
```

---

**Ready? Run `python demo.py` to get started!** 🚀
