# Camera calibration

CamKit3D expects a TOML calibration file with intrinsic and extrinsic parameters for each camera. This format is compatible with Anipose and FreeMoCap calibration workflows.

For now, see the [Anipose calibration guide](https://anipose.readthedocs.io/en/latest/calibration.html) for instructions on creating one using a ChArUco board.

<!--
TODO: Add a CamKit3D-specific calibration walkthrough here, e.g.:
- recommended ChArUco board size / dictionary
- how many frames / angles to capture
- how to verify the calibration (reprojection error thresholds)
- example TOML snippet
-->

## What the calibration file contains

Each camera entry provides:

- the intrinsic matrix,
- distortion coefficients,
- extrinsic rotation and translation.

These are consumed in [Stage 4 — Triangulation](../pipeline/triangulation.md).
