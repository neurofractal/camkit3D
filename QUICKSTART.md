# 🚀 CamKit3D Quick Start Guide

**Get up and running with multi-camera synchronized recording in 5 minutes**

---

## 📋 What You'll Learn

1. Install dependencies
2. Test your cameras
3. Record your first synchronized videos
4. Understand the output
5. Run a multi-trial experiment

---

## Step 1: Installation (2 minutes)

### Install Python Dependencies

```bash
pip install opencv-python numpy matplotlib
```

That's it! No FFmpeg required for basic recording.

### Verify Installation

```bash
python -c "import cv2, numpy, matplotlib; print('✓ All dependencies installed')"
```

---

## Step 2: Test Your Cameras (1 minute)

### Find Your Cameras

```python
from multicam_recorder import MultiCameraRecorder

# Test camera IDs 0, 1, 2
recorder = MultiCameraRecorder(camera_ids=[0, 1, 2], fps=30)
results = recorder.connect_cameras()

print("Connected cameras:", [cam_id for cam_id, success in results.items() if success])
recorder.disconnect_cameras()
```

**Expected output:**
```
Camera 0 connected: 1280x720 @ 30fps
Camera 1 connected: 1280x720 @ 30fps
Connected cameras: [0, 1]
```

**Common Issues:**
- If no cameras found, try IDs: `[0, 1, 2, 3, 4]`
- Close other apps using cameras (Zoom, Skype, Teams)

### Preview Your Cameras

```python
recorder = MultiCameraRecorder(camera_ids=[0, 1], fps=30)
recorder.connect_cameras()
recorder.preview_cameras(duration=5)  # Shows preview for 5 seconds
recorder.disconnect_cameras()
```

Press 'q' to exit preview early.

---

## Step 3: Record Your First Video (2 minutes)

### Complete Recording Example

Save this as `my_first_recording.py`:

```python
from multicam_recorder import MultiCameraRecorder
from sync_by_timestamps import synchronize_videos_to_ideal_fps
import time

print("=== CamKit3D First Recording ===\n")

# Step 1: Setup
print("[1] Setting up cameras...")
recorder = MultiCameraRecorder(
    camera_ids=[0, 1],  # YOUR camera IDs here
    fps=30,
    width=1280,
    height=720
)

# Step 2: Connect
print("[2] Connecting to cameras...")
results = recorder.connect_cameras()
print(f"    Connected: {[cam for cam, ok in results.items() if ok]}")

# Step 3: Record
print("[3] Recording for 5 seconds...")
print("    TIP: Make a loud clap or flash a light for sync reference!")
recorder.start_recording("my_first_test")
time.sleep(5)  # Record for 5 seconds
recorder.stop_recording()
print("    ✓ Recording complete!")

# Step 4: Disconnect
print("[4] Disconnecting cameras...")
recorder.disconnect_cameras()

# Step 5: Synchronize
print("[5] Synchronizing videos...")
results = synchronize_videos_to_ideal_fps(
    trial_folder="./recordings/my_first_test",
    target_fps=30.0
)

print(f"\n✓ SUCCESS!")
print(f"✓ Recorded and synchronized {len(results['camera_ids'])} cameras")
print(f"✓ {results['frame_count']} frames at perfect {results['target_fps']} FPS")
print(f"✓ Videos saved to: {results['output_dir']}")

# Quality check
print(f"\nSynchronization Quality:")
for cam_id, metrics in results['sync_metrics'].items():
    print(f"  Camera {cam_id}: {metrics['mean_diff_ms']:.2f}ms mean error (excellent!)")
```

### Run It!

```bash
python my_first_recording.py
```

**What happens:**
1. Connects to cameras (once)
2. Records 5-second video
3. Disconnects cameras
4. Synchronizes videos using timestamps
5. Saves synchronized output

---

## Step 4: Understand the Output

### File Structure

After running, you'll see:

```
recordings/
└── my_first_test/
    ├── raw_videos/
    │   ├── camera_0.avi           ← Original recordings
    │   └── camera_1.avi
    ├── synchronized_videos/
    │   ├── camera_0_synchronized.avi  ← Frame-perfect sync!
    │   └── camera_1_synchronized.avi
    ├── camera_0_timestamps.npy    ← Hardware timestamps
    ├── camera_1_timestamps.npy
    ├── frame_mappings_to_ideal_fps.npz  ← Sync data
    └── metadata.txt               ← Recording info
```

### View the Metadata

```bash
cat recordings/my_first_test/metadata.txt
```

```
Trial: my_first_test
Timestamp: 2026-02-05T14:30:00
Resolution: 1280x720
FPS: 30
Codec: MJPEG (.avi)

Camera 0:
  Frames: 150
  Duration: 5.02s

Camera 1:
  Frames: 148
  Duration: 5.01s
```

**Notice**: Original recordings may have different frame counts due to dropped frames. This is normal and gets fixed during synchronization!

### View Synchronized Videos

The synchronized videos in `synchronized_videos/` will have:
- ✅ **Identical frame counts** (150 frames each)
- ✅ **Perfect 30.0 FPS timing**
- ✅ **Sub-millisecond alignment**

---

## Step 5: Multi-Trial Experiment (Advanced)

### Recording Multiple Trials

Save this as `run_experiment.py`:

```python
from multicam_recorder import MultiCameraRecorder
import time

print("=== Multi-Trial Experiment ===\n")

# Setup once
recorder = MultiCameraRecorder(camera_ids=[0, 1], fps=30)
recorder.connect_cameras()

# Record 10 trials
num_trials = 10

for trial_num in range(1, num_trials + 1):
    print(f"\nTrial {trial_num}/{num_trials}")
    input("Press Enter when ready to record...")
    
    # Record trial
    trial_name = f"experiment_trial_{trial_num:02d}"
    recorder.start_recording(trial_name)
    
    print("Recording... (10 seconds)")
    time.sleep(10)
    
    recorder.stop_recording()
    print(f"✓ Trial {trial_num} saved!")

# Cleanup once
recorder.disconnect_cameras()
print("\n✓ All trials recorded!")
print("Now run batch synchronization...")
```

### Batch Synchronization

After recording all trials, synchronize them:

```python
from sync_by_timestamps import synchronize_videos_to_ideal_fps
from pathlib import Path

print("=== Batch Synchronization ===\n")

# Find all trial folders
trial_folders = sorted(Path("./recordings").glob("experiment_trial_*"))

print(f"Found {len(trial_folders)} trials to synchronize\n")

for i, trial_folder in enumerate(trial_folders, 1):
    print(f"[{i}/{len(trial_folders)}] Synchronizing {trial_folder.name}...")
    
    results = synchronize_videos_to_ideal_fps(
        trial_folder=str(trial_folder),
        target_fps=30.0
    )
    
    # Quick quality check
    avg_error = sum(m['mean_diff_ms'] for m in results['sync_metrics'].values()) / len(results['sync_metrics'])
    print(f"    ✓ Average sync error: {avg_error:.2f}ms\n")

print("✓ All trials synchronized!")
```

---

## 🎯 Key Concepts

### Why Separate Recording and Synchronization?

**Recording Phase:**
- Fast and efficient
- No computational overhead
- Can record many trials quickly
- Camera connection stays stable

**Synchronization Phase:**
- Done offline after all recordings
- Uses saved timestamps for perfect alignment
- Can be re-run with different parameters
- No time pressure

### Understanding Timestamps

Each frame gets a hardware timestamp when captured:

```python
# Recorded automatically
timestamps_cam0 = [0.000, 0.033, 0.067, 0.100, ...]  # Seconds
timestamps_cam1 = [0.001, 0.034, 0.066, 0.099, ...]
```

These timestamps are used to:
1. Create an ideal timing grid (perfect 30 FPS = every 0.03333s)
2. Find the nearest actual frame for each ideal timepoint
3. Write synchronized videos with perfect timing

### What Gets Fixed During Synchronization?

**Original recordings may have:**
- ❌ Different frame counts (148 vs 150 frames)
- ❌ Dropped frames (missing frame #47)
- ❌ Variable frame timing (32ms, 34ms, 33ms, ...)
- ❌ Camera drift (accumulates over time)

**Synchronized videos have:**
- ✅ Identical frame counts
- ✅ No dropped frames in output
- ✅ Perfect frame timing (exactly 33.333ms each)
- ✅ Sub-millisecond alignment between cameras

---

## 📊 Synchronization Analysis

The system provides detailed analysis showing:

![Synchronization Analysis](sync_analysis.png)

**What you see:**
1. **Top Left**: Original frame times showing drift and dropped frames
2. **Top Right**: Synchronized frame times - perfect alignment
3. **Middle**: Timing errors - original massive errors → synchronized <2ms
4. **Bottom Middle**: Frame durations - variable → consistent
5. **Bottom**: Dropped frame detection - clear problem identification

**Key Metrics:**
- **Mean error < 1ms**: Excellent
- **Mean error 1-5ms**: Good for most applications
- **Mean error > 10ms**: Check camera/USB issues

---

## 🎓 Best Practices

### For Recording

1. **Connect cameras once** at session start
2. **Record multiple trials** without disconnecting
3. **Disconnect once** at session end
4. **Use consistent settings** (FPS, resolution) across trials

### For Synchronization

1. **Synchronize after all recordings** (batch processing)
2. **Check quality metrics** for each trial
3. **Keep raw videos** as backup
4. **Document any issues** in trial notes

### For Camera Placement

1. **Use tripods** for stable positioning
2. **Good lighting** improves quality
3. **Avoid backlighting** (windows behind subject)
4. **Test preview** before recording trials

### For USB Connections

1. **Use USB 3.0 ports** when possible
2. **Don't daisy-chain** USB hubs
3. **Dedicated USB controller** for multiple cameras
4. **Disable power saving** on USB ports

---

## 🔧 Common Settings

### 60 FPS High-Speed

```python
recorder = MultiCameraRecorder(
    camera_ids=[0, 1],
    fps=60,
    width=1280,
    height=720
)
```

### Lower Resolution for More Cameras

```python
recorder = MultiCameraRecorder(
    camera_ids=[0, 1, 2, 3],  # 4 cameras
    fps=30,
    width=640,
    height=480
)
```

### High Quality Recording

```python
recorder = MultiCameraRecorder(
    camera_ids=[0, 1],
    fps=30,
    width=1920,
    height=1080
)
```

---

## 🐛 Troubleshooting Quick Reference

### Problem: Camera won't connect

```python
# Try different camera IDs
recorder = MultiCameraRecorder(camera_ids=[0, 1, 2, 3, 4])
```

### Problem: Many dropped frames

```python
# Reduce resolution and FPS
recorder = MultiCameraRecorder(
    camera_ids=[0, 1],
    fps=30,        # Instead of 60
    width=640,     # Instead of 1280
    height=480     # Instead of 720
)
```

### Problem: Synchronization warnings

```python
# Increase threshold if warnings are acceptable
results = synchronize_videos_to_ideal_fps(
    trial_folder="./recordings/my_trial",
    target_fps=30.0,
    max_time_diff_ms=200.0  # Increased from 100ms
)
```

---

## 📚 Next Steps

### Learn More
- Read the full [README.md](README.md) for detailed documentation
- See example synchronization results above
- Explore advanced features (context managers, custom codecs)

### Run Interactive Demos
```bash
python demo.py  # If available in your distribution
```

### Integrate with Your Research
- Combine with stimulus presentation software
- Export timestamps for analysis
- Process synchronized videos with motion capture software

---

## 🎬 Complete Workflow Summary

```
┌─────────────────┐
│  1. Setup       │  camera_ids, fps, resolution
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Connect     │  Connect once per session
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Record      │  Multiple trials
│     Trials      │  No disconnection needed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Disconnect  │  Disconnect once at end
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Synchronize │  Batch processing
│     (Offline)   │  Frame-perfect alignment
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Analysis    │  Quality metrics
│     & Export    │  Visualization
└─────────────────┘
```

---

## 💡 Pro Tips

1. **Record a "test trial" first** to check camera angles and lighting
2. **Document your setup** (camera positions, settings) for reproducibility
3. **Monitor disk space** - video files can be large (1GB per minute at 1080p)
4. **Back up raw videos** before experimenting with sync parameters
5. **Check sync quality** on first few trials before recording all sessions

---

## ✅ Checklist: Your First Recording

- [ ] Python and dependencies installed
- [ ] Cameras detected and working
- [ ] Test preview successful
- [ ] First recording completed
- [ ] Synchronization successful
- [ ] Output files verified
- [ ] Quality metrics checked
- [ ] Ready for multi-trial experiments!

---

**🎉 Congratulations! You're now ready to record synchronized multi-camera videos for your research!**

For questions, issues, or advanced usage, see the full [README.md](README.md) documentation.

---

*CamKit3D - Professional multi-camera recording made simple.*
