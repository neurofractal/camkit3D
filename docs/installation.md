# Installation

!!! warning "To Follow"
    Rob needs to add this package to pip when he is happy with everything.

```bash
pip install camkit3d
```

Or install from source in editable mode:

```bash
git clone https://github.com/neurofractal/camkit3D
cd camkit3D
pip install -e ".[dev]"
```

## Requirements

- Python ≥ 3.9
- USB 3.0 ports for multi-camera recording
- A TOML camera calibration file (e.g. from [Anipose](https://anipose.readthedocs.io/) or [FreeMoCap](https://freemocap.org/))
