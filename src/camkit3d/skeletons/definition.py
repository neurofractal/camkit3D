"""Parsed skeleton descriptor for CamKit3D.

Defines :class:`PoseDefinition`, the in-memory representation of a skeleton
descriptor YAML (see ``skeletons/data/*.yaml``). This is the single source of
truth for skeleton topology, so the recorder, pose2d, pose3d, viewer, and
analysis modules all read landmark names, groups, and connections from one
place rather than hard-coding them.

Key features:

- One descriptor, many skeletons. A YAML file per skeleton (MediaPipe Pose,
  MediaPipe Holistic, and so on) is parsed into a validated object, so
  supporting a new skeleton means adding data, not code.
- Strict validation. On construction it checks that landmark indices are
  contiguous and that every group, connection, symmetry pair, and anatomy
  anchor references a real landmark, catching malformed descriptors early.
- Role-based lookups. Callers ask for semantic roles (face, hand, body) and
  anatomy anchors (left_hip, and so on) rather than raw indices, so code
  written against one skeleton keeps working when the group naming differs.
- Derived accessors. Edges, per-edge colours, per-landmark confidence
  thresholds, and colour-by-index are computed from the descriptor for the
  drawing and triangulation stages.

You normally do not import this module directly. Use
``camkit3d.skeletons.load(...)`` instead, which returns a cached, validated
:class:`PoseDefinition`.

Author: Dr. Robert Seymour, OHBA, University of Oxford
License: GNU General Public License v3, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["PoseDefinition", "Landmark", "Group", "Connection"]


@dataclass(frozen=True)
class Landmark:
    """A single keypoint."""
    index: int
    name: str
    group: str
    side: str  # 'left' | 'right' | 'center'


@dataclass(frozen=True)
class Group:
    """A semantic group of landmarks (used for colour / filtering)."""
    name: str
    indices: Tuple[int, ...]
    color: str
    confidence_threshold: Optional[float] = None


@dataclass(frozen=True)
class Connection:
    """A skeleton edge between two landmark indices."""
    start: int
    end: int
    group: str


@dataclass
class PoseDefinition:
    """In-memory skeleton descriptor.

    Attributes
    ----------
    skeleton_id : str
        Registry key (matches the YAML filename stem).
    metadata : dict
        The full ``metadata`` block from the descriptor.
    landmarks : list of Landmark
        Ordered by ``index`` (0 .. num_landmarks-1).
    groups : dict of str -> Group
    connections : list of Connection
    symmetry : dict
        ``{'midline': [...], 'pairs': [(l, r), ...]}``.
    anatomy : dict of str -> int
        Semantic role -> landmark index (e.g. ``{'left_hip': 23}``). May be
        empty for skeletons that don't define anchors.
    virtual_landmarks : list of dict
    segments : list of dict

    Notes
    -----
    Construct via :func:`camkit3d.skeletons.load`, not directly, so you get
    validation and caching. :meth:`from_dict` is the low-level entry point.
    """

    skeleton_id: str
    metadata: Dict
    landmarks: List[Landmark]
    groups: Dict[str, Group]
    connections: List[Connection]
    symmetry: Dict = field(default_factory=dict)
    anatomy: Dict[str, int] = field(default_factory=dict)
    roles: Dict[str, List[str]] = field(default_factory=dict)
    virtual_landmarks: List[Dict] = field(default_factory=list)
    segments: List[Dict] = field(default_factory=list)

    # ── Construction ────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, raw: Dict) -> "PoseDefinition":
        """Build and validate a PoseDefinition from a parsed YAML/JSON dict."""
        meta = raw.get("metadata", {})
        skeleton_id = meta.get("skeleton_id")
        if not skeleton_id:
            raise ValueError("descriptor missing metadata.skeleton_id")

        landmarks = [
            Landmark(
                index=int(lm["index"]),
                name=lm["name"],
                group=lm.get("group", ""),
                side=lm.get("side", "center"),
            )
            for lm in raw.get("landmarks", [])
        ]
        landmarks.sort(key=lambda lm: lm.index)

        groups = {
            name: Group(
                name=name,
                indices=tuple(g["indices"]),
                color=g["color"],
                confidence_threshold=g.get("confidence_threshold"),
            )
            for name, g in raw.get("groups", {}).items()
        }

        connections = [
            Connection(start=int(c["edge"][0]), end=int(c["edge"][1]),
                       group=c.get("group", ""))
            for c in raw.get("connections", [])
        ]

        sym = raw.get("symmetry", {})
        symmetry = {
            "midline": list(sym.get("midline", [])),
            "pairs": [tuple(p) for p in sym.get("pairs", [])],
        }

        obj = cls(
            skeleton_id=skeleton_id,
            metadata=meta,
            landmarks=landmarks,
            groups=groups,
            connections=connections,
            symmetry=symmetry,
            anatomy={k: int(v) for k, v in raw.get("anatomy", {}).items()},
            roles={k: list(v) for k, v in raw.get("roles", {}).items()},
            virtual_landmarks=raw.get("virtual_landmarks", []),
            segments=raw.get("segments", []),
        )
        obj.validate()
        return obj

    # ── Validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        """Raise ValueError if the descriptor is internally inconsistent."""
        n = self.num_landmarks
        declared = self.metadata.get("num_landmarks")
        if declared is not None and declared != n:
            raise ValueError(
                f"{self.skeleton_id}: metadata.num_landmarks={declared} "
                f"but {n} landmarks listed"
            )

        # Indices must be contiguous 0 .. n-1
        idx = [lm.index for lm in self.landmarks]
        if idx != list(range(n)):
            raise ValueError(
                f"{self.skeleton_id}: landmark indices must be contiguous "
                f"0..{n - 1}, got {idx}"
            )

        # Every group index must reference a real landmark
        for g in self.groups.values():
            bad = [i for i in g.indices if not 0 <= i < n]
            if bad:
                raise ValueError(
                    f"{self.skeleton_id}: group '{g.name}' references "
                    f"out-of-range indices {bad}"
                )

        # Every connection must reference real landmarks + a known group
        for c in self.connections:
            for endpoint in (c.start, c.end):
                if not 0 <= endpoint < n:
                    raise ValueError(
                        f"{self.skeleton_id}: connection [{c.start},{c.end}] "
                        f"references out-of-range index {endpoint}"
                    )
            if c.group and c.group not in self.groups:
                raise ValueError(
                    f"{self.skeleton_id}: connection [{c.start},{c.end}] "
                    f"names unknown group '{c.group}'"
                )

        # Symmetry pairs must reference real landmarks
        for l, r in self.symmetry.get("pairs", []):
            if not (0 <= l < n and 0 <= r < n):
                raise ValueError(
                    f"{self.skeleton_id}: symmetry pair ({l},{r}) out of range"
                )

        # Anatomy anchors must reference real landmarks
        for role, idx in self.anatomy.items():
            if not 0 <= idx < n:
                raise ValueError(
                    f"{self.skeleton_id}: anatomy anchor '{role}'={idx} "
                    f"out of range 0..{n - 1}"
                )

    # ── Derived accessors ───────────────────────────────────────────
    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)

    @property
    def names(self) -> List[str]:
        """Landmark names ordered by index."""
        return [lm.name for lm in self.landmarks]

    def index_of(self, name: str) -> int:
        """Return the landmark index for a given name."""
        for lm in self.landmarks:
            if lm.name == name:
                return lm.index
        raise KeyError(f"{self.skeleton_id}: no landmark named '{name}'")

    def anchor(self, role: str) -> int:
        """Return the landmark index for an anatomy *role* (e.g. 'left_hip').

        Raises
        ------
        KeyError
            If the skeleton declares no anchor for that role. The message
            lists the roles that *are* available.
        """
        try:
            return self.anatomy[role]
        except KeyError:
            raise KeyError(
                f"{self.skeleton_id}: no anatomy anchor '{role}' "
                f"(has: {', '.join(sorted(self.anatomy)) or 'none'})"
            ) from None

    def has_anchors(self, *roles: str) -> bool:
        """True if the skeleton declares anchors for *all* given roles."""
        return all(r in self.anatomy for r in roles)

    def group_indices(self, group: str) -> Tuple[int, ...]:
        """Indices belonging to a named group."""
        return self.groups[group].indices

    def role_indices(self, role: str) -> List[int]:
        """Sorted indices for a semantic role ('face', 'hand', 'body', ...).

        Resolution order:
        1. If the descriptor declares a ``roles`` block, union the groups it
           lists for this role.
        2. Otherwise fall back to name heuristics: a group whose name equals
           the role, contains it (e.g. 'left_hand' for 'hand'), or, for
           'face', the 'face'/'pose_face' groups.

        This lets skeletons with different group naming (mediapipe_pose's
        'hand' vs mediapipe_holistic's 'left_hand'/'right_hand') answer the
        same question without callers hard-coding group names.
        """
        # 1. Explicit roles block
        if role in self.roles:
            idx = set()
            for gname in self.roles[role]:
                if gname in self.groups:
                    idx.update(self.groups[gname].indices)
            return sorted(idx)

        # 2. Name-heuristic fallback
        idx = set()
        for gname, g in self.groups.items():
            if gname == role or role in gname.split("_"):
                idx.update(g.indices)
        return sorted(idx)

    @property
    def face_indices(self) -> List[int]:
        """All landmark indices belonging to the face (role-resolved)."""
        return self.role_indices("face")

    @property
    def hand_indices(self) -> List[int]:
        """All landmark indices belonging to the hands (role-resolved)."""
        return self.role_indices("hand")

    @property
    def body_indices(self) -> List[int]:
        """Indices that are neither face nor hand (role-resolved)."""
        exclude = set(self.face_indices) | set(self.hand_indices)
        return [i for i in range(self.num_landmarks) if i not in exclude]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Connections as a plain list of (start, end) tuples.

        Drop-in replacement for the SKELETON_CONNECTIONS list in viewer.py.
        """
        return [(c.start, c.end) for c in self.connections]

    @property
    def edge_colors(self) -> List[str]:
        """Hex colour per edge, aligned with :attr:`edges`."""
        return [self.groups[c.group].color if c.group in self.groups
                else "#888888" for c in self.connections]

    def color_for_index(self, index: int) -> str:
        """Colour of the (first) group a landmark belongs to."""
        for g in self.groups.values():
            if index in g.indices:
                return g.color
        return "#888888"

    def confidence_thresholds(self, default: float = 0.3) -> np.ndarray:
        """Per-landmark confidence-threshold array of shape (num_landmarks,).

        Mirrors the per-keypoint threshold array built in pose3d.py, driven by
        each group's ``confidence_threshold``. Where a landmark belongs to more
        than one group with a threshold (e.g. hands are in both 'hand' and an
        'arm' group), the groups named 'face' and 'hand' take priority, so the
        stricter face/hand thresholds win. Any remaining ties fall to the
        larger (stricter) threshold.
        """
        thr = np.full(self.num_landmarks, default, dtype=float)

        # Pass 1: broad groups (everything except face/hand); stricter wins
        for name, g in self.groups.items():
            if name in ("face", "hand") or g.confidence_threshold is None:
                continue
            for i in g.indices:
                thr[i] = max(thr[i], g.confidence_threshold) \
                    if thr[i] != default else g.confidence_threshold

        # Pass 2: face then hand override (these are the priority groups)
        for name in ("face", "hand"):
            g = self.groups.get(name)
            if g is None or g.confidence_threshold is None:
                continue
            for i in g.indices:
                thr[i] = g.confidence_threshold

        return thr

    def __str__(self) -> str:
        return (f"<PoseDefinition '{self.skeleton_id}': "
                f"{self.num_landmarks} landmarks, "
                f"{len(self.connections)} connections, "
                f"{len(self.groups)} groups>")