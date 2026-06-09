# Installation

```bash
pip install camkit3d
```

Or install from source in editable mode (recommended for development):

```bash
git clone https://github.com/neurofractal/camkit3D
cd camkit3D
pip install -e ".[dev]"
```

## Requirements

- Python ≥ 3.9
- USB 3.0 ports for multi-camera recording
- A TOML camera calibration file (e.g. from [Anipose](https://anipose.readthedocs.io/) or [FreeMoCap](https://freemocap.org/))

!!! tip "Calibration file"
    You need a calibration file before you can triangulate (Stage 4), but not
    before recording (Stage 1). See [Camera Calibration](tutorials/calibration.md)
    for how to create one with a ChArUco board.
