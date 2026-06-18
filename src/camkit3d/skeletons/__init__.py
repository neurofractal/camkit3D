"""
camkit3d.skeletons – skeleton descriptor registry and loader.

A *skeleton descriptor* is a YAML file under ``skeletons/data/`` describing a
pose topology: its landmarks, semantic groups, skeleton connections, colours,
symmetry and metadata. This package discovers those files and loads them into
validated :class:`~camkit3d.skeletons.definition.PoseDefinition` objects.

Adding a new skeleton
---------------------
Drop a new ``<skeleton_id>.yaml`` into ``skeletons/data/``. No code changes
needed — it is discovered automatically and loadable via its filename stem::

    from camkit3d import skeletons
    pose = skeletons.load("mediapipe_pose")
    print(skeletons.available())          # ['mediapipe_pose', ...]

Typical use in the other modules::

    from camkit3d import skeletons
    pose = skeletons.load()               # default skeleton
    edges       = pose.edges              # replaces SKELETON_CONNECTIONS
    edge_colors = pose.edge_colors        # replaces _connection_color()
    thresholds  = pose.confidence_thresholds()   # replaces _per_kp_threshold
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

from .definition import PoseDefinition

__all__ = ["load", "available", "to_json", "PoseDefinition"]

# The skeleton used when load() is called with no argument.
DEFAULT_SKELETON = "mediapipe_pose"

# Folder holding the descriptor YAML files, resolved relative to this file.
_DATA_DIR = Path(__file__).parent / "data"


def available() -> List[str]:
    """Return the sorted skeleton_ids of all discoverable descriptors.

    A skeleton's id is the stem of its ``.yaml``/``.yml`` file in
    ``skeletons/data/``.
    """
    return sorted(p.stem for p in _DATA_DIR.glob("*.y*ml"))


@lru_cache(maxsize=None)
def load(skeleton_id: str = DEFAULT_SKELETON) -> PoseDefinition:
    """Load and validate a skeleton descriptor by id.

    Parameters
    ----------
    skeleton_id : str, optional
        Filename stem of a descriptor in ``skeletons/data/``. Defaults to
        :data:`DEFAULT_SKELETON`.

    Returns
    -------
    PoseDefinition
        Cached, validated definition. Repeated calls return the same object.

    Raises
    ------
    FileNotFoundError
        If no descriptor with that id exists. The message lists what's
        available.
    """
    raw = _read_descriptor(skeleton_id)
    pose = PoseDefinition.from_dict(raw)

    # Guard against id/filename mismatch — keeps the registry honest.
    if pose.skeleton_id != skeleton_id:
        raise ValueError(
            f"descriptor '{skeleton_id}.yaml' declares "
            f"metadata.skeleton_id='{pose.skeleton_id}'; these must match"
        )
    return pose


def to_json(skeleton_id: str = DEFAULT_SKELETON, *, indent: int = 2) -> str:
    """Return a JSON string of the *raw* descriptor.

    Useful for emitting a frozen JSON twin of the YAML (e.g. to embed in the
    self-contained HTML viewer, or to ship to a non-Python consumer).
    """
    return json.dumps(_read_descriptor(skeleton_id), indent=indent)


# ── internal ────────────────────────────────────────────────────────
def _read_descriptor(skeleton_id: str) -> Dict:
    """Read and YAML-parse a descriptor file, or raise FileNotFoundError."""
    for ext in ("yaml", "yml"):
        p = _DATA_DIR / f"{skeleton_id}.{ext}"
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
    raise FileNotFoundError(
        f"no skeleton descriptor '{skeleton_id}' "
        f"(available: {', '.join(available()) or 'none'})"
    )
