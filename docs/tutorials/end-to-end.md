# camkit3D — End-to-End Pipeline

`Record` → `2D Pose` → `3D Triangulation` → `Align` → `Interpolate` → `View` → `Animate` → `Combine`

<!-- ![Pipeline banner](./images/placeholder_banner.png) -->

## Initial setup

> **Do this once**, before your first recording session.

### Buy webcams

- Choose a **wide field of view** — essential for smaller rooms.

### Connect them to your computer

- Plug them in, but **spread them across USB controllers**. Piling every camera onto one controller causes lots of dropped frames.

### Test the connection in camkit3D

- See **Stage 0** below.
- Ask: *do they all connect? Do any drop frames?*

### Mount on steady tripods

- Point the cameras at your participant **from multiple angles and heights**.
- Avoid anything too perpendicular or at very acute angles — those views triangulate poorly.

### Download & print a Charuco board

![Charuco board 5x3](https://raw.githubusercontent.com/neurofractal/camkit3D/main/docs/images/charuco_board_5x3.png)

- **Measure the size of one black square** — you'll need this for calibration.

### Perform a calibration

Full guide: [camkit3D calibration docs](https://github.com/neurofractal/camkit3D/blob/main/docs/calibration.md)

- Wave the Charuco board around **slowly**, rotating through angles so **every camera** gets full coverage.
- Don't get too close to any single camera — focus on the volume where your participant will actually be.

> **Handy terminal code** — record for 90 s, then immediately synchronise the frames:

```bash
cd path_to_camkit3d
python record_and_sync.py 90 --cameras 0 1 2 --countdown 10 --width 1024 --height 576 \
  --output-dir /Users/robertseymour/Documents/recordings
```

Run it through **FreeMoCap**:

`Load recording → Process Data → Calibrate from Active Recording → Charuco Board 5x3 → Run calibration → Copy camera_calibration.toml to your parent directory`

## Stage 0 · Config & imports

Flip `RECORD_NEW` to `False` to re-analyse an existing session instead of recording afresh.

```python
import time, json, math
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

%matplotlib inline

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR    = Path('/Users/robertseymour/Documents/recordings')
CAMERA_IDS  = [0, 1, 2]
FPS         = 30
WIDTH, HEIGHT = 1024, 576
RECORD_SECONDS = 30

RECORD_NEW = True   # False → reuse an existing SESSION_DIR below

# # If RECORD_NEW is False, set this to an existing session folder:
# SESSION_DIR = Path('/Users/robertseymour/Documents/recordings/2026-08-17_13-59-01')

print('Config ready')
```

## Stage 1 · Record & synchronise

Records **~30 s** from all cameras with a countdown. Then resample every camera onto one ideal FPS timeline so the frames line up across views.

### Connect to the cameras

```python
from camkit3d.recorder import MultiCamRecorder

recorder = MultiCamRecorder(
    camera_ids=[0,1,2],  # Your cameras
    fps=30,
    base_output_dir = DATA_DIR,
    width = 1024,
    height= 576
)

# Connect to cameras (once)
recorder.connect_cameras()
```

### Preview before you commit

Use the live preview to check framing, lighting, and that everyone is fully in shot.

```python
recorder.preview_cameras(duration=300.0, target_fps=30)
```

![Live camera preview](./images/placeholder_preview.png)

### Record with a countdown

```python
from camkit3d.sync import synchronize_videos_to_ideal_fps

if RECORD_NEW:
    # 5s countdown before recording
    for s in range(6, 0, -1):
        print(f'\rRecording in {s}...', end='', flush=True)
        time.sleep(1)
    print('\rRecording now!      ')

    timestamp   = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    SESSION_DIR = DATA_DIR / f'recording_{timestamp}'
    recorder.start_recording(str(SESSION_DIR))
    time.sleep(RECORD_SECONDS)
    recorder.stop_recording()
    recorder.disconnect_cameras()
```

### Synchronise to an ideal FPS

```python
if RECORD_NEW:
    # Synchronise all cameras to an ideal FPS
    synchronize_videos_to_ideal_fps(
        trial_folder=str(SESSION_DIR),
        target_fps=float(FPS),
        max_time_diff_ms=50.0,
    )

assert SESSION_DIR is not None, 'Set SESSION_DIR or use RECORD_NEW=True'
SESSION_DIR = Path(SESSION_DIR)
print('Session:', SESSION_DIR)
```

## Stage 2 · 2D pose estimation (MediaPipe)

Runs **MediaPipe** on each synchronised video, then smooths the keypoints and cleans stray face points near the frame edges.

```python
from camkit3d.pose2d import (
    PoseProcessor,
    smooth_keypoints_directory,
    clean_face_points_directory,
)

INPUT_VIDEO_DIR = SESSION_DIR / 'synchronized_videos'
MP_OUTPUT_DIR   = SESSION_DIR / 'mediapipe_output'
MP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

processor = PoseProcessor(
    input_dir=INPUT_VIDEO_DIR,
    output_dir=MP_OUTPUT_DIR,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=2,
    smooth_landmarks=True,
)

# Produces labelled videos + 2D keypoints per camera
metrics_all = processor.process_all_videos(save_labeled_videos=True)
for cam, m in metrics_all.items():
    print(f'{cam}: detection {m.detection_rate:.1%}, mean conf {m.avg_confidence:.3f}')

# ── Clean the 2D keypoints in place ──
data_2d_dir = MP_OUTPUT_DIR / 'data_2d'
smooth_keypoints_directory(input_dir=data_2d_dir, cutoff_freq=4.0, sampling_freq=FPS)
clean_face_points_directory(input_dir=data_2d_dir, frame_width=1920, frame_height=1080)
print('2D keypoints cleaned')
```

![2D pose overlay](./images/placeholder_pose2d.png)

*A labelled frame showing the MediaPipe skeleton drawn over one camera view.*

## Stage 3 · 3D triangulation

> Place the relevant `camera_calibration.toml` in the `SESSION_DIR`.

**DLT triangulation** across cameras using the FreeMoCap calibration. A confidence floor drops genuinely bad detections, soft weighting handles the rest, and a reprojection-error filter removes remaining outliers.

```python
from camkit3d.pose3d import Pose3DProjector

CALIBRATION_FILE = SESSION_DIR / 'camera_calibration.toml'
OUTPUT_3D_DIR    = SESSION_DIR / 'data_3d'
OUTPUT_3D_DIR.mkdir(parents=True, exist_ok=True)

projector = Pose3DProjector(
    calibration_path=CALIBRATION_FILE,
    keypoints_dir=MP_OUTPUT_DIR / 'data_2d',
    skeleton='mediapipe_pose',
    min_cameras_for_triangulation=2,
    soft_conf_floor=0.5,  # Drop anything below 0.5
    reprojection_error_threshold=20.0,
    use_iterative_rejection=True,
)

points_3d, metrics = projector.triangulate_all_frames()

# NaN out keypoints with high mean reprojection error
points_3d = Pose3DProjector.nan_filter_by_reprojection_error(
    points_3d, metrics, per_keypoint=False, error_threshold=15.0,
)

projector.save_3d_data(points_3d, output_path=str(OUTPUT_3D_DIR / 'pose_3d'),
                       metadata={'min_cameras': 2, 'soft_conf_floor': 0.5})

print('3D points:', points_3d.shape, '| mean reproj error:',
      f'{metrics.mean_reprojection_error:.2f}px')
```

## Stage 4 · Align to a body-centred frame

Re-expresses the pose in **upright, person-facing, floor-referenced** coordinates, so downstream measures don't depend on where the cameras happened to sit.

```python
from camkit3d.analysis import align_pose_to_standard_frame, interpolate_nans
from camkit3d.viewer import viewer

points_3d_aligned, R, orient = align_pose_to_standard_frame(points_3d)
print('Aligned:', points_3d_aligned.shape)
```

## Stage 5 · Interpolate NaNs

**PCHIP** fill for bounded gaps. The plot below sanity-checks the interpolation.

```python
points_3d_filled, report = interpolate_nans(
    points_3d_aligned, method='pchip', max_gap_seconds=1.0, fps=FPS,
)

# Compare a representative measure before vs after interpolation
LEFT_SHOULDER, LEFT_WRIST = 11, 15

def wrist_shoulder_distance(points):
    d = np.full(points.shape[0], np.nan)
    for f in range(points.shape[0]):
        s, w = points[f, LEFT_SHOULDER], points[f, LEFT_WRIST]
        if not np.isnan(s).any() and not np.isnan(w).any():
            d[f] = euclidean(s, w)
    return d

dist_original = wrist_shoulder_distance(points_3d_aligned)
dist_filled   = wrist_shoulder_distance(points_3d_filled)

plt.figure(figsize=(12, 4))
plt.plot(dist_filled,   color='red',   lw=1.5, label='Filled')
plt.plot(dist_original, color='black', lw=1.5, label='Original')
plt.xlabel('Frame'); plt.ylabel('Distance (mm)')
plt.title('Left shoulder → left wrist (before/after interpolation)')
plt.grid(alpha=0.3); plt.legend(); plt.show()
```

![Interpolation comparison plot](./images/placeholder_interpolation.png)

*The shoulder→wrist distance plot, black (original) vs red (filled).*

## Stage 6 · Interactive viewer — before & after interpolation

Opens the camkit3D viewer for a visual QC pass.

> **Note:** requires an interactive backend. Switch to `%matplotlib qt` first — the inline backend won't show the live viewer.

```python
# After interpolation (gaps filled)
viewer(points_3d_filled, fps=FPS)
```

## Stage 7 · Render the 3D animation

A front-view, real-time animation of the filled pose, saved straight to **MP4**.

```python
from camkit3d.analysis import animate_3d_pose

animate_3d_pose(
    points_3d_filled,
    show_floor=False,
    view_mode='custom',
    frames_to_animate=points_3d_filled.shape[0],
    rotation_speed=0,
    output_path=SESSION_DIR / 'animations' / 'pose_front.mp4',
    elevation=20, azimuth_start=150,
    keypoint_size=70, line_width=4,
    show_axes=True, fps=FPS, quality='high',
)
print('Animation saved')
```

## Stage 8 · Combine camera videos with the 3D animation

Lays out the **labelled camera grid on the left** and the **3D front view on the right**, written to a single MP4 — handy for talks, figures, and quick QC.

```python
def _open(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {path}')
    return cap

def _info(cap):
    return (cap.get(cv2.CAP_PROP_FPS),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

def _grid_dims(n):
    cols = math.ceil(math.sqrt(n))
    return cols, math.ceil(n / cols)

def _build_grid(cam_frames, n_cols, n_rows, grid_w, grid_h):
    cell_w, cell_h = grid_w // n_cols, grid_h // n_rows
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    n = len(cam_frames)
    for idx in range(n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        last_row_count = n - (n_rows - 1) * n_cols
        col_offset = ((n_cols - last_row_count) * cell_w // 2
                      if row == n_rows - 1 and last_row_count < n_cols else 0)
        x0, y0 = col * cell_w + col_offset, row * cell_h
        if idx >= n or cam_frames[idx] is None:
            continue
        frame = cam_frames[idx]
        fh, fw = frame.shape[:2]
        scale = min(cell_w / fw, cell_h / fh)
        nw, nh = max(1, round(fw * scale)), max(1, round(fh * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        x1, y1 = x0 + (cell_w - nw) // 2, y0 + (cell_h - nh) // 2
        y2, x2 = min(y1 + nh, grid_h), min(x1 + nw, grid_w)
        canvas[y1:y2, x1:x2] = resized[:y2 - y1, :x2 - x1]
    return canvas

def combine_cameras_and_front_view(
    session_dir,
    synced_folder_name='mediapipe_output/labeled_videos',
    front_view_rel='animations/pose_front.mp4',
    out_name='animations/video_pose.mp4',
    camera_glob='camera_*_synchronized_labeled.mp4',
    grid_target_h=1080,
):
    session_dir = Path(session_dir)
    synced_dir  = session_dir / synced_folder_name
    front_path  = session_dir / front_view_rel
    out_path    = session_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam_paths = sorted(synced_dir.glob(camera_glob))
    if not cam_paths:
        raise FileNotFoundError(f'No camera videos in {synced_dir}')
    if not front_path.exists():
        raise FileNotFoundError(f'Missing front view: {front_path}')

    cap_front = _open(front_path)
    caps = [_open(p) for p in cam_paths]
    fps_front, n_front, front_w, front_h = _info(cap_front)
    infos = [_info(c) for c in caps]
    out_fps = fps_front if fps_front and fps_front > 0 else 30.0

    min_frames = min(n_front, *[i[1] for i in infos])
    n_cols, n_rows = _grid_dims(len(caps))

    src_ws = sorted(i[2] for i in infos); src_hs = sorted(i[3] for i in infos)
    median_w, median_h = src_ws[len(src_ws)//2], src_hs[len(src_hs)//2]
    cell_h = max(1, grid_target_h // n_rows)
    cell_w = max(1, round(median_w * (cell_h / median_h)))
    grid_w, grid_h = cell_w * n_cols, cell_h * n_rows
    right_h = grid_h
    right_w = max(1, round(front_w * (right_h / front_h)))
    out_w, out_h = grid_w + right_w, grid_h

    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             out_fps, (out_w, out_h))
    for fi in range(min_frames):
        if fi % 100 == 0:
            print(f'  frame {fi}/{min_frames}')
        cam_frames = []
        for cap in caps:
            ok, f = cap.read()
            cam_frames.append(f if ok and f is not None else None)
        grid = _build_grid(cam_frames, n_cols, n_rows, grid_w, grid_h)
        ok, f_front = cap_front.read()
        f_front = (cv2.resize(f_front, (right_w, right_h), interpolation=cv2.INTER_AREA)
                   if ok and f_front is not None
                   else np.zeros((right_h, right_w, 3), np.uint8))
        writer.write(np.hstack([grid, f_front]))

    writer.release(); cap_front.release()
    for c in caps: c.release()
    print('Saved:', out_path)
    return str(out_path)


result = combine_cameras_and_front_view(SESSION_DIR)
print('Combined video:', result)
```

More docs & the full library: [github.com/neurofractal/camkit3D](https://github.com/neurofractal/camkit3D)