# Camera calibration

3D camera calibration is handled externally to CamKit3D. We recommend Freemocap which provides an accessible GUI-like worlflow to calibration, relying on the well-validated [anipose python package](https://github.com/lambdaloop/anipose). The basic idea is to record synchronised videos using a known Charuco board:

![5x3 ChArUco board](images/charuco_board_5x3.png)

### Freemocap
For detailed instructions see: [https://docs.freemocap.org/documentation/multi-camera-calibration.html](https://docs.freemocap.org/documentation/multi-camera-calibration.html)

**Workflow:**
- Record a multi-camera ~90s video using CamKit3D 
- Sycnronise the videos using CamKit3D
- Open Freemocap --> Point to the sychronised video files
- Perform Calibration --> TOML file

!!! note "Recording a calbiration video"
    It is vital you record a **high-quality calibration video**. Tips:
    - Record at least 90s of data
    - Move the Charuco Board slowly
    - Move the board forwards and back, tilt in all axes

### TOML File

CamKit3D expects a TOML calibration file with intrinsic and extrinsic parameters for each camera. This format is compatible with Anipose and FreeMoCap calibration workflows.

The file contains one `[cam_N]` table per camera plus a `[metadata]` table:

**Per-camera fields**

| Field | Description |
|-------|-------------|
| `name` | Camera identifier (e.g. `camera_0_synchronized`) |
| `size` | Image resolution `[width, height]` in pixels |
| `matrix` | 3×3 intrinsic matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` — focal lengths and principal point in pixels |
| `distortions` | Lens distortion coefficients `[k1, k2, p1, p2, k3]` (radial + tangential) |
| `rotation` | Extrinsic rotation as a Rodrigues (axis-angle) vector, in radians |
| `translation` | Extrinsic translation vector `[x, y, z]` in mm |
| `world_orientation` | 3×3 rotation matrix mapping camera → world coordinates |
| `world_position` | Camera centre in world coordinates `[x, y, z]` in mm |

The first camera (`cam_0`) defines the world origin: zero rotation/translation and an identity `world_orientation`. All other cameras are expressed relative to it. `rotation`/`translation` give the world→camera transform (used for projection), while `world_orientation`/`world_position` give the inverse camera→world pose (used for plotting camera locations in the scene).

**Metadata fields**

| Field | Description |
|-------|-------------|
| `charuco_square_size` | ChArUco square side length in mm (sets the real-world scale) |
| `charuco_board_object` | Board definition (layout, marker dictionary) used during calibration |
| `path_to_recorded_videos` | Source directory of the synchronized videos |
| `date_time_calibrated` | ISO timestamp of calibration |
| `groundplane_calibration` | Whether a ground-plane alignment step was applied |

Units are millimetres throughout (set by `charuco_square_size`), so triangulated 3D coordinates are returned in mm.