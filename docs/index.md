# CamKit3D

<p align="center">
  <a href="https://github.com/neurofractal/camkit3D">
    <img src="images/logo.png" width="250">
  </a>
</p>

**Multi-camera 3D pose estimation pipeline for naturalistic behavioural and neuroscience applications.**

CamKit3D turns a set of USB webcams into a markerless motion-capture system. It handles: recording, temporal synchronisation, 2D pose estimation, triangulation, analysis and visualisation.

The code can be used as a stand-alone package or integrated alongside wearable brain imaging.

![gif](https://raw.githubusercontent.com/neurofractal/camkit3D/main/images/example_RS.gif)

---

## Getting Started
**- Buy webcams**. Worth considering a wide field of view — essential for smaller rooms.

**- Connect them to your computer**. Plug them in, but spread them across USB controllers. Putting every camera onto one controller causes lots of dropped frames.

**- Install camkit3d**. See [Installation](installation.md).

**- Test the connection in camkit3D**. See [Stage 0](tutorials/end-to-end.md) here.

*Do they all connect? Any drop frames?*

**- Mount on steady tripods**. Point the cameras at your participant from multiple angles and heights.

**- Download & print a Charuco board**. [Charuco board 5x3](https://raw.githubusercontent.com/neurofractal/camkit3D/main/docs/images/charuco_board_5x3.png).

Measure the size of one black square — you'll need this for calibration.

**- Perform a calibration**. Full guide: [camkit3D calibration docs](https://github.com/neurofractal/camkit3D/blob/main/docs/calibration.md)

**- End-to-End Pipeline**. Try to follow the [End-to-end Walkthrough](tutorials/end-to-end.md).

## What CamKit3D does

The pipeline is organised as five sequential stages, but you can also use the scripts independently if your data is organised correctly.

| Stage | Module | Purpose |
|---|---|---|
| 1 | `camkit3d.recorder` | Capture synchronised video from multiple USB webcams |
| 2 | `camkit3d.sync` | Align frames across cameras to a common timing grid |
| 3 | `camkit3d.pose2d` | Extract 33 body keypoints per frame with MediaPipe |
| 4 | `camkit3d.pose3d` | Triangulate 2D keypoints into 3D world coordinates |
| 5 | `camkit3d.analysis` | Orient, plot, and animate the 3D skeleton data |

## Acknowledgements

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) (Google) for 2D keypoint detection
- [Anipose](https://anipose.readthedocs.io/) for triangulation methodology
- [FreeMoCap](https://freemocap.org/) for calibration format and inspiration

We also acknowledge Dr. Patricia Cambalova and Dr. Seb Rieger who have provided invaludable experimental support in the Oxford OPM lab.

## Citation

Seymour, R. A., Hill, R., Brookes, M. J., & Woolrich, M. W. (in preparation). *Markerless 3D Motion Capture for Wearable OPM-MEG*.

## License
GNU v3.0 — see [LICENSE](https://github.com/neurofractal/camkit3D/blob/main/LICENSE) for details.
