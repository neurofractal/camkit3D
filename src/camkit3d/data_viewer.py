"""Interactive 3D pose data explorer for CamKit3D.

Browses the triangulated output of :mod:`camkit3d.pose3d` across one take or a
whole folder of takes, in the browser. The main window shows every landmark as
its own horizontal trace over time; alongside it sit a 3D skeleton view that
follows the playhead and a keypoint picker for showing or hiding landmarks by
group or individually.

Key features:

- Folder discovery. Point it at a single take (e.g. ``trial_001``) or at a
  folder of takes. It finds every 3D pose array (``.npy`` shaped
  ``(n_frames, n_landmarks, 3)`` inside a folder with "3d" in its name, or with
  a ``pose3d`` JSON sidecar), groups them by take, and skips 2D
  ``*_keypoints.npy`` files. When a take holds several 3D outputs you choose
  which one to view; the newest is the default.
- Gap filling on load. NaNs are filled with
  :func:`camkit3d.analysis.interpolate_nans` (PCHIP by default) the first time
  a trial is opened. Filled samples are marked in the traces so you can see
  exactly what was interpolated. Pass ``interpolate=False`` to view raw data.
- Anatomical alignment on load. The body orientation is detected once, from
  the first trial, with :func:`camkit3d.analysis.align_pose_to_standard_frame`,
  and that same rotation is applied to every trial, so all trials share
  X = medial-lateral (right +), Y = anterior-posterior (forward +),
  Z = superior-inferior (up +). Pass ``align=False`` to keep the calibration
  frame.
- Fast. A small local server loads, interpolates and caches trials once, then
  streams them to the page as raw binary. The page keeps recent trials in
  memory and prefetches neighbours, and draws traces to an off-screen buffer
  so playback only redraws the playhead.
- Skeleton-driven. Landmark names, colours and bones come from the skeleton
  descriptor. Keypoints are picked in non-overlapping groups (face, torso,
  left/right arm, left/right hand, left/right leg) built from each landmark's
  ``group`` and ``side``, so every landmark belongs to exactly one group.

Usage:

    from camkit3d.data_viewer import data_viewer

    data_viewer()                                  # pick a folder
    data_viewer("/path/to/recordings")             # a folder of takes
    data_viewer("/path/to/recordings/trial_001")   # a single take
    data_viewer(all_data, fps=30)                  # arrays already in memory
    data_viewer("/path/to/recordings", interpolate=False, align=False)

From a terminal:

    python -m camkit3d.data_viewer /path/to/recordings [--no-interpolate] [--no-align]

Controls
--------
Play / pause      : space
Step frame        : left / right arrows (shift: 1 second)
Previous / next   : [ and ]  (or up / down arrows)
Choose trials     : type in the trial filter, e.g. 1-4, 7, 10-12
Zoom time         : scroll over the traces, or + / -
Pan time          : horizontal scroll or shift + scroll
Reset zoom        : double-click the traces, or 0
Seek / scrub      : click or drag on the traces or the timeline
Solo a group      : shift-click a group chip

Author: Dr. Robert Seymour, OHBA, University of Oxford
License: GNU General Public License v3, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse

import numpy as np

from camkit3d import skeletons as _skeletons
from camkit3d.skeletons import PoseDefinition

__all__ = ["data_viewer", "DataViewerSession", "find_3d_takes"]


# ════════════════════════════════════════════════════════════════════════════
#  Discovery
# ════════════════════════════════════════════════════════════════════════════

# Folders that never hold 3D output; skipping them keeps scans of large
# recording trees quick.
_SKIP_DIRS = {
    "__pycache__", "node_modules", "raw_videos", "synchronized_videos",
    "labeled_videos", "data_2d",
}
_FPS_KEYS = ("fps", "frame_rate", "target_fps", "sampling_rate")


# Base groups that are not split into left and right in the keypoint picker.
_UNSPLIT_GROUPS = {"face", "head", "torso", "trunk", "pelvis", "spine", "body", "neck"}


def _exclusive_groups(skel: PoseDefinition) -> List[Dict]:
    """Partition landmarks into non-overlapping groups for the picker.

    Each landmark's ``group`` field (face, torso, arm, hand, leg, ...) is split
    by ``side`` into left/right groups, except for midline groups such as face
    and torso. Colours come from the descriptor group of the same name
    (``left_arm``), else the base group (``hand``), else the landmark colour.
    """
    groups: "OrderedDict[str, Dict]" = OrderedDict()
    for lm in skel.landmarks:
        base = lm.group or "other"
        side = lm.side if lm.side in ("left", "right") and base not in _UNSPLIT_GROUPS else None
        key = f"{side}_{base}" if side else base
        if key not in groups:
            if key in skel.groups:
                color = skel.groups[key].color
            elif base in skel.groups:
                color = skel.groups[base].color
            else:
                color = skel.color_for_index(lm.index)
            pretty = base.replace("_", " ")
            label = f"{side.capitalize()} {pretty}" if side else pretty.capitalize()
            groups[key] = {"name": key, "label": label, "side": side,
                           "indices": [], "color": color}
        groups[key]["indices"].append(lm.index)
    return list(groups.values())


def _natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def _is_3d_name(name: str) -> bool:
    return "3d" in name.lower()


def _read_sidecar(npy_path: Path) -> Dict:
    sidecar = npy_path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        with open(sidecar, "r") as fh:
            meta = json.load(fh)
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def _npy_shape(path: Path):
    """Read an .npy header (shape, dtype) without loading the array."""
    try:
        with open(path, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
            else:
                shape, _, dtype = np.lib.format.read_array_header_2_0(fh)
        return shape, dtype
    except Exception:
        return None, None


@dataclass
class _Version:
    label: str
    mtime: float
    n_frames: int
    n_landmarks: int
    fps: Optional[float] = None
    skeleton_id: Optional[str] = None
    path: Optional[Path] = None
    array: Optional[np.ndarray] = None  # in-memory sources


@dataclass
class _Take:
    name: str
    path: Optional[Path]
    versions: List[_Version] = field(default_factory=list)


def find_3d_takes(root: Union[str, Path]) -> List[_Take]:
    """Find 3D pose outputs under *root* and group them by take.

    A file counts as 3D pose output when it is a floating-point ``.npy`` of
    shape ``(n_frames, n_landmarks, 3)``, is not a 2D ``*_keypoints.npy``, and
    either sits inside a folder whose name contains "3d" (``data_3d``,
    ``data_3d_soft``, ...), has "3d" in its own name, or has a JSON sidecar
    written by :meth:`Pose3DProjector.save_3d_data`.

    The take is the folder above the first "3d" folder (so
    ``trial_001/data_3d/pose_3d.npy`` belongs to ``trial_001``). Files not in a
    "3d" folder belong to the folder they sit in. Versions within a take are
    sorted newest first.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    takes: Dict[Path, _Take] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in _SKIP_DIRS]
        npys = [f for f in filenames
                if f.endswith(".npy") and not f.startswith("._")
                and not f.endswith("_keypoints.npy")]
        if not npys:
            continue
        here = Path(dirpath)
        rel_parts = here.relative_to(root).parts
        accepted = []
        for fname in npys:
            path = here / fname
            meta = _read_sidecar(path)
            looks_3d = (_is_3d_name(fname) or any(_is_3d_name(p) for p in rel_parts)
                        or "n_keypoints" in meta)
            if not looks_3d:
                continue
            shape, dtype = _npy_shape(path)
            if (shape is None or len(shape) != 3 or shape[2] != 3 or shape[0] < 2
                    or dtype is None or dtype.kind != "f"):
                continue
            accepted.append((path, shape, meta))

        for path, shape, meta in accepted:
            idx3d = next((i for i, p in enumerate(rel_parts) if _is_3d_name(p)), None)
            take_path = root.joinpath(*rel_parts[:idx3d]) if idx3d is not None else here
            rel_from_take = here.relative_to(take_path).as_posix()
            label = rel_from_take if rel_from_take != "." else path.parent.name
            if len(accepted) > 1:
                label = f"{label}/{path.stem}"
            fps = next((float(meta[k]) for k in _FPS_KEYS
                        if isinstance(meta.get(k), (int, float))), None)
            version = _Version(
                label=label, mtime=path.stat().st_mtime,
                n_frames=int(shape[0]), n_landmarks=int(shape[1]),
                fps=fps, skeleton_id=meta.get("skeleton_id"), path=path,
            )
            if take_path not in takes:
                rel = take_path.relative_to(root).as_posix()
                takes[take_path] = _Take(name=rel if rel != "." else root.name,
                                         path=take_path)
            takes[take_path].versions.append(version)

    out = sorted(takes.values(), key=lambda t: _natural_key(t.name))
    for take in out:
        take.versions.sort(key=lambda v: v.mtime, reverse=True)
    return out


def _takes_from_arrays(source, names: Optional[Sequence[str]]) -> List[_Take]:
    if isinstance(source, np.ndarray):
        if source.ndim == 3:
            arrays = [source]
        elif source.ndim == 4:
            arrays = list(source)
        else:
            raise ValueError(
                "Arrays must be (n_frames, n_landmarks, 3) or "
                f"(n_trials, n_frames, n_landmarks, 3); got {source.shape}")
    else:
        arrays = [np.asarray(a) for a in source]
    takes = []
    for i, arr in enumerate(arrays):
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Trial {i}: expected (n_frames, n_landmarks, 3), got {arr.shape}")
        name = names[i] if names is not None else f"Trial {i}"
        takes.append(_Take(name=str(name), path=None, versions=[_Version(
            label="in memory", mtime=time.time(), n_frames=arr.shape[0],
            n_landmarks=arr.shape[1], array=arr)]))
    return takes


# ════════════════════════════════════════════════════════════════════════════
#  Session: loading, interpolation, cache, HTTP server
# ════════════════════════════════════════════════════════════════════════════

class DataViewerSession:
    """A running data viewer. Returned by :func:`data_viewer`.

    Attributes
    ----------
    url : str
        Address of the viewer; open it in any browser on this machine.
    root : Path or None
        Folder currently being browsed.

    Methods
    -------
    open_browser()
        Open (another) browser tab on the viewer.
    stop()
        Shut the server down and free cached trials.
    """

    def __init__(self, takes, root, skeleton, fps, interpolate, method,
                 max_gap_seconds, cache_mb, verbose, port, fixed_skeleton,
                 align=True):
        self.takes: List[_Take] = takes
        self.root: Optional[Path] = root
        self.skeleton: PoseDefinition = skeleton
        self._fixed_skeleton = fixed_skeleton
        self.fps = float(fps)
        self.interpolate = interpolate
        self.method = method
        self.max_gap_seconds = max_gap_seconds
        self.verbose = verbose
        self._cache: "OrderedDict[tuple, bytes]" = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = int(cache_mb * 1024 * 1024)
        self._cache_lock = threading.Lock()
        self._key_locks: Dict[tuple, threading.Lock] = {}
        self._generation = 0
        self._interp_fn = None
        self._interp_warned = False
        self.align = align
        self._rotation_matrix: Optional[np.ndarray] = None
        self._rotation_ref: Optional[str] = None
        self._rotation_failed = False
        self._rotation_lock = threading.Lock()

        handler = _make_handler(self)
        self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/"
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="camkit3d-data-viewer", daemon=True)
        self._thread.start()

    # ── public ──────────────────────────────────────────────────────
    def open_browser(self):
        webbrowser.open(self.url)

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        with self._cache_lock:
            self._cache.clear()
            self._cache_bytes = 0
        print("CamKit3D data viewer stopped.")

    def __repr__(self):
        where = self.root if self.root is not None else "arrays in memory"
        return (f"<DataViewerSession {self.url} | {len(self.takes)} takes | {where}>")

    # ── folder handling ─────────────────────────────────────────────
    def open_folder(self, folder) -> None:
        takes = find_3d_takes(folder)
        self.root = Path(folder).expanduser().resolve()
        self.takes = takes
        self._generation += 1
        with self._rotation_lock:
            self._rotation_matrix, self._rotation_ref = None, None
            self._rotation_failed = False
        if not self._fixed_skeleton:
            self.skeleton = _skeleton_for(takes)
        with self._cache_lock:
            self._cache.clear()
            self._cache_bytes = 0

    def index_payload(self) -> Dict:
        skel = self.skeleton
        return {
            "generation": self._generation,
            "root": str(self.root) if self.root else None,
            "rootName": self.root.name if self.root else "Arrays in memory",
            "canBrowse": True,
            "interpolate": bool(self.interpolate),
            "method": self.method,
            "maxGapSeconds": self.max_gap_seconds,
            "align": bool(self.align),
            "takes": [{
                "name": t.name,
                "versions": [{
                    "label": v.label,
                    "mtime": v.mtime,
                    "nFrames": v.n_frames,
                    "fps": v.fps or self.fps,
                } for v in t.versions],
            } for t in self.takes],
            "skeleton": {
                "id": skel.skeleton_id,
                "names": skel.names,
                "landmarkGroups": [lm.group for lm in skel.landmarks],
                "groups": [{"name": g.name, "indices": list(g.indices), "color": g.color}
                           for g in skel.groups.values()],
                "edges": [list(e) for e in skel.edges],
                "edgeColors": skel.edge_colors,
                "pointColors": [skel.color_for_index(i) for i in range(skel.num_landmarks)],
                "pairs": [list(p) for p in skel.symmetry.get("pairs", [])],
                "partition": _exclusive_groups(skel),
            },
        }

    # ── trial loading ───────────────────────────────────────────────
    def trial_bytes(self, take_idx: int, version_idx: int) -> bytes:
        take = self.takes[take_idx]
        version = take.versions[version_idx]
        key = (self._generation, take_idx, version_idx)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            lock = self._key_locks.setdefault(key, threading.Lock())
        with lock:  # one load per trial even if requests race
            with self._cache_lock:
                if key in self._cache:
                    return self._cache[key]
            payload = self._build_trial(take, version)
            with self._cache_lock:
                self._cache[key] = payload
                self._cache_bytes += len(payload)
                while self._cache_bytes > self._cache_limit and len(self._cache) > 1:
                    _, old = self._cache.popitem(last=False)
                    self._cache_bytes -= len(old)
            return payload

    def _interpolator(self):
        if self._interp_fn is None and not self._interp_warned:
            try:
                from camkit3d.analysis import interpolate_nans
                self._interp_fn = interpolate_nans
            except ImportError as exc:
                self._interp_warned = True
                print(f"Warning: could not import camkit3d.analysis.interpolate_nans "
                      f"({exc}); showing raw data.")
        return self._interp_fn

    @staticmethod
    def _load_raw(version: _Version) -> np.ndarray:
        if version.array is not None:
            return np.asarray(version.array, dtype=np.float64)
        return np.load(version.path).astype(np.float64, copy=False)

    def _rotation(self) -> Optional[np.ndarray]:
        """Rotation to the standard anatomical frame, detected from the first trial.

        Computed once per folder from the first take's newest 3D output (raw,
        before gap filling), then reused for every trial so they all share one
        frame. Returns None if alignment is off or could not be computed.
        """
        if not self.align or not self.takes:
            return None
        with self._rotation_lock:
            if self._rotation_matrix is not None or self._rotation_failed:
                return self._rotation_matrix
            try:
                from camkit3d.analysis import align_pose_to_standard_frame
            except ImportError as exc:
                print(f"Warning: could not import camkit3d.analysis."
                      f"align_pose_to_standard_frame ({exc}); trials are not aligned.")
                self._rotation_failed = True
                return None
            ref_take = self.takes[0]
            ref_version = ref_take.versions[0]
            try:
                reference = self._load_raw(ref_version)
                _, R, _orient = align_pose_to_standard_frame(reference)
                R = np.asarray(R, dtype=np.float64)
                if R.shape != (3, 3) or not np.all(np.isfinite(R)):
                    raise ValueError(f"expected a finite 3x3 rotation, got shape {R.shape}")
            except Exception as exc:
                print(f"Warning: could not detect body orientation from "
                      f"{ref_take.name} ({type(exc).__name__}: {exc}); trials are not aligned.")
                self._rotation_failed = True
                return None
            self._rotation_matrix = R
            self._rotation_ref = f"{ref_take.name} [{ref_version.label}]"
            print(f"Aligning all trials to the anatomical frame detected from "
                  f"{self._rotation_ref}.\nRotation matrix R:\n{np.array2string(R, precision=4)}")
            return R

    def _build_trial(self, take: _Take, version: _Version) -> bytes:
        t0 = time.perf_counter()
        raw = self._load_raw(version)
        n_frames, n_kp, _ = raw.shape
        if n_kp != self.skeleton.num_landmarks:
            raise ValueError(
                f"{take.name}: {n_kp} landmarks, but skeleton "
                f"'{self.skeleton.skeleton_id}' has {self.skeleton.num_landmarks}")
        fps = version.fps or self.fps

        missing = np.isnan(raw).any(axis=2)
        data, report = raw, None
        interp = self._interpolator() if self.interpolate else None
        if interp is not None and missing.any():
            data, report = interp(raw, method=self.method,
                                  max_gap_seconds=self.max_gap_seconds,
                                  fps=fps, verbose=self.verbose)
            data = np.asarray(data, dtype=np.float64)
        still_missing = np.isnan(data).any(axis=2)

        # Rotate into the shared anatomical frame, keeping NaNs as NaN.
        R = self._rotation()
        if R is not None:
            data = np.einsum("ij,nkj->nki", R, data)
            data[still_missing] = np.nan

        # 0 = measured, 1 = filled by interpolation, 2 = still missing
        state = np.where(still_missing, 2, np.where(missing, 1, 0)).astype(np.uint8)

        valid = data[~still_missing]
        if valid.size:
            # Centre on the middle of the body's spatial extent rather than the
            # median landmark (the 11 face points would pull it upwards).
            lo, hi = np.percentile(valid, [2, 98], axis=0)
            centre = (lo + hi) / 2
            extent = float(np.percentile(np.linalg.norm(valid - centre, axis=1), 98))
            floor_z = float(np.percentile(valid[:, 2], 1))
        else:
            centre, extent, floor_z = np.zeros(3), 1.0, 0.0
        extent = extent if extent > 1e-6 else 1.0

        header = {
            "take": take.name,
            "version": version.label,
            "nFrames": int(n_frames),
            "nLandmarks": int(n_kp),
            "fps": float(fps),
            "centre": [float(c) for c in centre],
            "extent": extent,
            "floorZ": floor_z,
            "nMissingRaw": int(missing.sum()),
            "nFilled": int((state == 1).sum()),
            "nMissing": int((state == 2).sum()),
            "interpolated": interp is not None,
            "aligned": R is not None,
            "alignedTo": self._rotation_ref if R is not None else None,
            "source": str(version.path) if version.path else None,
        }
        if self.verbose or report is not None:
            print(f"Loaded {take.name} [{version.label}]: {n_frames} frames, "
                  f"filled {header['nFilled']}, still missing {header['nMissing']} "
                  f"({time.perf_counter() - t0:.2f} s)"
                  + (f" | {report}" if self.verbose and report is not None else ""))
        return _pack(header, data.astype("<f4", copy=False), state)


def _pack(header: Dict, data: np.ndarray, state: np.ndarray) -> bytes:
    """[uint32 header length][JSON header, padded to 4 bytes][float32 xyz][uint8 state]"""
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hdr += b" " * ((-(4 + len(hdr))) % 4)
    return b"".join([struct.pack("<I", len(hdr)), hdr,
                     np.ascontiguousarray(data, dtype="<f4").tobytes(),
                     np.ascontiguousarray(state, dtype=np.uint8).tobytes()])


def _make_handler(session: DataViewerSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, _HTML_TEMPLATE.encode("utf-8"), "text/html; charset=utf-8")
                elif url.path == "/api/index":
                    self._json(session.index_payload())
                elif url.path == "/api/trial":
                    if int(q.get("gen", session._generation)) != session._generation:
                        self._json({"error": "The folder changed; reload the list."}, 409)
                        return
                    body = session.trial_bytes(int(q["take"]), int(q["version"]))
                    self._send(200, body, "application/octet-stream")
                elif url.path == "/api/browse":
                    self._json(_browse(q.get("path") or (str(session.root) if session.root else None)))
                elif url.path == "/api/open":
                    session.open_folder(q["path"])
                    self._json(session.index_payload())
                elif url.path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # report errors to the page
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def _browse(path: Optional[str]) -> Dict:
    folder = Path(path).expanduser() if path else Path.home()
    if not folder.is_dir():
        folder = Path.home()
    folder = folder.resolve()
    try:
        dirs = sorted((d.name for d in folder.iterdir()
                       if d.is_dir() and not d.name.startswith(".")), key=_natural_key)
    except PermissionError:
        dirs = []
    parts = []
    acc = Path(folder.anchor)
    parts.append({"name": folder.anchor or "/", "path": str(acc)})
    for p in folder.relative_to(folder.anchor).parts:
        acc = acc / p
        parts.append({"name": p, "path": str(acc)})
    return {"path": str(folder), "parent": str(folder.parent), "dirs": dirs,
            "crumbs": parts, "sep": os.sep}


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def _ask_directory() -> Optional[str]:
    """Native folder dialog. Returns None if unavailable or cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        tk_root = tk.Tk()
        tk_root.withdraw()
        try:
            tk_root.attributes("-topmost", True)
        except Exception:
            pass
        chosen = filedialog.askdirectory(
            title="Choose a take, or a folder of takes", mustexist=True)
        tk_root.update()
        tk_root.destroy()
        return chosen or None
    except Exception:
        return None


def _skeleton_for(takes: List[_Take]) -> PoseDefinition:
    for take in takes:
        for v in take.versions:
            if v.skeleton_id:
                try:
                    return _skeletons.load(v.skeleton_id)
                except Exception:
                    pass
    return _skeletons.load()


def data_viewer(
    source=None,
    fps: float = 30,
    interpolate: bool = True,
    method: str = "pchip",
    max_gap_seconds: float = 1.0,
    align: bool = True,
    skeleton=None,
    names: Optional[Sequence[str]] = None,
    folder_dialog: Optional[bool] = None,
    open_browser: bool = True,
    block: Optional[bool] = None,
    port: int = 0,
    cache_mb: float = 1024,
    verbose: bool = False,
) -> DataViewerSession:
    """
    Explore triangulated 3D pose data in the browser.

    Parameters
    ----------
    source : str, Path, ndarray or list of ndarray, optional
        A take folder (e.g. ``trial_001``), a folder of takes, or data already
        in memory: one array ``(n_frames, n_landmarks, 3)``, a stack
        ``(n_trials, n_frames, n_landmarks, 3)``, or a list of arrays. If None,
        you choose a folder (see ``folder_dialog``).
    fps : float, default 30
        Frame rate, used when a file's JSON sidecar does not record one.
    interpolate : bool, default True
        Fill NaN gaps with :func:`camkit3d.analysis.interpolate_nans` when a
        trial is first opened. Set False to view the raw reconstruction.
    method : str, default "pchip"
        Interpolation method passed to ``interpolate_nans``.
    max_gap_seconds : float, default 1.0
        Longest gap to fill; longer gaps stay missing.
    align : bool, default True
        Rotate every trial into a standard anatomical frame (X right, Y
        forward, Z up). The rotation is detected once from the first trial with
        :func:`camkit3d.analysis.align_pose_to_standard_frame` and applied
        unchanged to all trials, so relative orientation between trials is
        preserved. Set False to keep the calibration frame.
    skeleton : str or PoseDefinition, optional
        Skeleton topology. Defaults to the ``skeleton_id`` in the data's JSON
        sidecar, or MediaPipe Pose.
    names : sequence of str, optional
        Trial names for in-memory arrays.
    folder_dialog : bool, optional
        When ``source`` is None: True opens the system folder dialog, False
        uses the folder picker inside the viewer. Defaults to the system dialog
        in scripts and the in-viewer picker in notebooks (Tk dialogs can hang
        notebook kernels on macOS).
    open_browser : bool, default True
        Open the viewer in the default browser.
    block : bool, optional
        Keep Python busy until Ctrl+C. Defaults to True in scripts (so the
        server outlives the call) and False in notebooks.
    port : int, default 0
        Local port; 0 picks a free one.
    cache_mb : float, default 1024
        Memory budget for prepared trials on the Python side.
    verbose : bool, default False
        Print the interpolation report for each trial as it loads.

    Returns
    -------
    DataViewerSession
        Has ``.url`` and ``.stop()``.
    """
    notebook = _in_notebook()
    root = None
    fixed_skeleton = skeleton is not None

    if source is None:
        use_dialog = (not notebook) if folder_dialog is None else folder_dialog
        chosen = _ask_directory() if use_dialog else None
        if chosen:
            source = chosen

    if source is None:
        takes = []
    elif isinstance(source, (str, os.PathLike)):
        root = Path(source).expanduser().resolve()
        takes = find_3d_takes(root)
    else:
        takes = _takes_from_arrays(source, names)

    if skeleton is None:
        skel = _skeleton_for(takes)
    elif isinstance(skeleton, PoseDefinition):
        skel = skeleton
    else:
        skel = _skeletons.load(skeleton)

    session = DataViewerSession(
        takes=takes, root=root, skeleton=skel, fps=fps, interpolate=interpolate,
        method=method, max_gap_seconds=max_gap_seconds, cache_mb=cache_mb,
        verbose=verbose, port=port, fixed_skeleton=fixed_skeleton, align=align)

    n_versions = sum(len(t.versions) for t in takes)
    if source is None:
        print(f"CamKit3D data viewer: choose a folder in the browser. {session.url}")
    else:
        print(f"CamKit3D data viewer: {len(takes)} takes, {n_versions} 3D outputs. "
              f"{session.url}")
    if open_browser:
        session.open_browser()

    if block is None:
        block = not notebook
    if block:
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            session.stop()
    return session


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Explore CamKit3D 3D pose data.")
    parser.add_argument("folder", nargs="?", help="A take, or a folder of takes")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--no-interpolate", action="store_true",
                        help="Show raw data without filling NaN gaps")
    parser.add_argument("--no-align", action="store_true",
                        help="Keep the calibration frame instead of aligning to anatomy")
    parser.add_argument("--method", default="pchip")
    parser.add_argument("--max-gap-seconds", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    data_viewer(args.folder, fps=args.fps, interpolate=not args.no_interpolate,
                align=not args.no_align,
                method=args.method, max_gap_seconds=args.max_gap_seconds,
                port=args.port, verbose=args.verbose, block=True)


# ════════════════════════════════════════════════════════════════════════════
#  HTML template
# ════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CamKit3D data viewer</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --page:#FDF4F6; --panel:#FFFFFF; --edge:#E7E2F1; --edge2:#D9D3E8;
  --ink:#2E2A3E; --muted:#7D778F; --faint:#B3ADC4;
  --accent:#7F7BD0; --accent-soft:#E7E6F8; --accent-ink:#5652A8;
  --peach:#F9DFCF; --mist:#ECEAF2;
  --r-panel:16px; --r-ctl:9px;
  --railW:210px; --sideW:420px; --viewH:55%;
  --font:Arial,'Liberation Sans',Helvetica,sans-serif;
}
html,body{height:100%}
body{background:var(--page);color:var(--ink);font:14px/1.4 var(--font);overflow:hidden;
  display:grid;grid-template-rows:56px 1fr;-webkit-font-smoothing:antialiased}
button,select,input{font:inherit;color:inherit}
button{cursor:pointer;background:none;border:0}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.num{font-variant-numeric:tabular-nums}

/* ── Top bar ───────────────────────────────── */
.top{display:flex;align-items:center;gap:14px;padding:0 16px 0 10px}
.icon{width:34px;height:34px;border-radius:var(--r-ctl);display:grid;place-items:center;color:var(--muted)}
.icon:hover{background:var(--mist);color:var(--ink)}
.icon svg{width:18px;height:18px}
.where{display:flex;align-items:baseline;gap:8px;min-width:0}
.where b{font-weight:700;letter-spacing:-.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:26vw}
.link{color:var(--accent-ink);font-weight:500;font-size:13px;white-space:nowrap}
.link:hover{text-decoration:underline}
.trialnav{display:flex;align-items:center;gap:2px;margin-left:auto;background:var(--panel);
  border:1px solid var(--edge);border-radius:999px;padding:3px}
.trialnav .icon{width:30px;height:30px;border-radius:999px}
.trialname{min-width:150px;text-align:center;font-weight:600;font-size:15px;padding:0 6px;white-space:nowrap}
.trialname small{display:block;font-weight:400;font-size:11px;color:var(--muted);margin-top:-2px}
.ver{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted)}
select{appearance:none;background:var(--panel) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%237D778F' fill='none' stroke-width='1.5'/%3E%3C/svg%3E") no-repeat right 10px center;
  border:1px solid var(--edge);border-radius:var(--r-ctl);padding:6px 28px 6px 10px;font-size:13px;color:var(--ink);max-width:340px}
select:disabled{opacity:.6}
.status{font-size:12px;color:var(--muted);min-width:170px;text-align:right;margin-left:auto}
.status .spin{display:inline-block;width:10px;height:10px;border:2px solid var(--accent-soft);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite;vertical-align:-1px;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── App grid ─────────────────────────────── */
.app{display:grid;grid-template-columns:var(--railW) minmax(320px,1fr) 10px var(--sideW);
  gap:0;padding:0 12px 12px;min-height:0}
.app.norail{grid-template-columns:0 minmax(320px,1fr) 10px var(--sideW)}
.panel{background:var(--panel);border:1px solid var(--edge);border-radius:var(--r-panel);min-height:0;min-width:0;
  display:flex;flex-direction:column;overflow:hidden}

/* ── Trial rail ───────────────────────────── */
.rail{min-height:0;display:flex;flex-direction:column;padding-right:12px;overflow:hidden}
.app.norail .rail{visibility:hidden;padding:0}
.rail input{border:1px solid var(--edge);background:var(--panel);border-radius:var(--r-ctl);padding:7px 10px;font-size:13px}
.filtnote{font-size:12px;color:var(--muted);padding:4px 4px 6px;min-height:8px}
.rail ol{list-style:none;overflow-y:auto;flex:1;padding-bottom:8px;scrollbar-width:thin}
.rail li button{width:100%;text-align:left;padding:8px 10px;border-radius:10px;display:block;border:1px solid transparent}
.rail li button:hover{background:var(--mist)}
.rail li button.on{background:var(--panel);border-color:var(--edge2);box-shadow:0 1px 0 var(--edge)}
.rail li .nm{font-weight:600;font-size:13px;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rail li .sub{font-size:11.5px;color:var(--muted)}
.rail li button.on .nm{color:var(--accent-ink)}
.rail .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--faint);margin-right:6px;vertical-align:1px}
.rail .dot.ready{background:#9ED3B8}

/* ── Traces panel ─────────────────────────── */
.bar{display:flex;align-items:center;gap:12px;padding:10px 12px;border-bottom:1px solid var(--edge);flex-wrap:wrap}
.seg{display:inline-flex;background:var(--mist);border-radius:10px;padding:3px;gap:2px}
.seg button{padding:5px 11px;border-radius:7px;font-size:12.5px;font-weight:500;color:var(--muted);white-space:nowrap}
.seg button:hover{color:var(--ink)}
.seg button.on{background:var(--panel);color:var(--ink);box-shadow:0 1px 2px rgba(60,50,110,.12)}
.seg.small button{padding:4px 9px;font-size:12px}
.check{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:var(--muted);cursor:pointer;user-select:none}
.check input{accent-color:var(--accent);width:14px;height:14px}
.swatch{display:inline-block;width:18px;height:10px;border-radius:3px;vertical-align:-1px}
.hint{margin-left:auto;font-size:12px;color:var(--faint)}
.plot{position:relative;flex:1;min-height:0}
.plot canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
.tip{position:absolute;pointer-events:none;background:rgba(46,42,62,.92);color:#fff;font-size:12px;padding:6px 9px;border-radius:8px;
  white-space:nowrap;opacity:0;transition:opacity .08s;transform:translate(12px,-50%)}
.tip b{font-weight:600}
.tip span{color:#CFCBE6}
.msg{position:absolute;inset:0;display:none;place-items:center;text-align:center;padding:30px;color:var(--muted)}
.msg.show{display:grid}
.msg h2{font-size:18px;color:var(--ink);font-weight:600;margin-bottom:6px}
.msg p{max-width:46ch;margin:0 auto 14px}
.btn{background:var(--accent);color:#fff;border-radius:var(--r-ctl);padding:8px 16px;font-weight:600;font-size:13px}
.btn:hover{background:var(--accent-ink)}
.btn.ghost{background:var(--mist);color:var(--ink)}
.btn.ghost:hover{background:var(--edge)}

.transport{display:flex;align-items:center;gap:12px;padding:10px 14px;border-top:1px solid var(--edge)}
.play{width:40px;height:40px;border-radius:50%;background:var(--accent);color:#fff;display:grid;place-items:center;flex-shrink:0}
.play:hover{background:var(--accent-ink)}
.play svg{width:16px;height:16px}
.time{font-size:13px;font-weight:600;min-width:62px}
.time.dim{color:var(--muted);font-weight:500}
.scrub{position:relative;flex:1;height:26px;cursor:pointer;touch-action:none}
.scrub .track{position:absolute;left:0;right:0;top:11px;height:4px;border-radius:4px;background:var(--mist)}
.scrub .win{position:absolute;top:6px;height:14px;border-radius:5px;background:var(--accent-soft);border:1px solid #D2D0F2}
.scrub .fill{position:absolute;left:0;top:11px;height:4px;border-radius:4px;background:#BDBAEA}
.scrub .knob{position:absolute;top:5px;width:16px;height:16px;border-radius:50%;background:var(--panel);border:3px solid var(--accent);
  transform:translateX(-50%);box-shadow:0 1px 3px rgba(60,50,110,.25)}

/* ── Splitters ────────────────────────────── */
.split{position:relative;touch-action:none}
.split::after{content:"";position:absolute;border-radius:3px;background:transparent;transition:background .12s}
.split.v{cursor:col-resize}
.split.v::after{left:3px;right:3px;top:40%;bottom:40%}
.split.h{cursor:row-resize;height:10px;flex-shrink:0}
.split.h::after{top:3px;bottom:3px;left:40%;right:40%}
.split:hover::after,.split.drag::after{background:var(--edge2)}

/* ── Side column ──────────────────────────── */
.side{display:flex;flex-direction:column;min-height:0;min-width:0}
.view3d{flex:0 0 var(--viewH);position:relative;background:linear-gradient(#FFFCFD,#FBF1F4)}
.view3d canvas{position:absolute;inset:0;width:100%;height:100%;cursor:grab;touch-action:none}
.view3d canvas:active{cursor:grabbing}
.ov{position:absolute;left:10px;right:10px;top:10px;display:flex;gap:8px;align-items:flex-start;pointer-events:none}
.ov>*{pointer-events:auto}
.ov .seg{background:rgba(255,255,255,.85);backdrop-filter:blur(6px);border:1px solid var(--edge)}
.menu{margin-left:auto;position:relative}
.menu summary{list-style:none;cursor:pointer;background:rgba(255,255,255,.85);border:1px solid var(--edge);border-radius:10px;
  padding:5px 11px;font-size:12px;font-weight:500;color:var(--muted)}
.menu summary::-webkit-details-marker{display:none}
.menu[open] summary{color:var(--ink)}
.menu .pop{position:absolute;right:0;top:34px;width:220px;background:var(--panel);border:1px solid var(--edge);border-radius:12px;
  padding:12px;box-shadow:0 8px 24px rgba(60,50,110,.12);display:grid;gap:10px;z-index:5}
.row{display:grid;grid-template-columns:78px 1fr;align-items:center;gap:8px;font-size:12.5px;color:var(--muted)}
input[type=range]{-webkit-appearance:none;appearance:none;height:4px;border-radius:4px;background:var(--mist);outline-offset:6px}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--accent);border:0}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--accent);border:0}
.foot3d{position:absolute;left:12px;bottom:10px;font-size:12px;color:var(--muted);pointer-events:none}
.nogl{position:absolute;inset:0;display:none;place-items:center;text-align:center;padding:20px;color:var(--muted);font-size:13px}

/* ── Keypoint picker ──────────────────────── */
.kps{flex:1}
.kps .bar{gap:8px}
.kps h3{font-size:14px;font-weight:600}
.count{font-size:12px;color:var(--muted)}
.mini{font-size:12px;font-weight:500;color:var(--accent-ink);padding:3px 8px;border-radius:7px}
.mini:hover{background:var(--accent-soft)}
.mini:first-of-type{margin-left:auto}
.chips{display:flex;flex-wrap:wrap;gap:6px;padding:10px 12px;border-bottom:1px solid var(--edge)}
.chip{display:inline-flex;align-items:center;gap:6px;padding:4px 11px 4px 5px;border-radius:999px;font-size:12.5px;font-weight:700;
  border:1px solid var(--edge2);background:var(--panel);color:var(--muted);user-select:none;white-space:nowrap}
.chip i{width:14px;height:14px;border-radius:50%;border:1.5px solid var(--c);display:block}
.chip:hover{border-color:var(--c)}
.chip.all{background:var(--cs);border-color:var(--cs);color:var(--ink)}
.chip.all i{background:var(--c)}
.chip.some{color:var(--ink)}
.chip.some i{background:linear-gradient(90deg,var(--c) 50%,transparent 50%)}
.kpgroups{overflow-y:auto;padding:10px 12px 12px;flex:1;display:grid;grid-template-columns:1fr 1fr;gap:10px;align-content:start;scrollbar-width:thin}
.grp{border:1px solid var(--edge);border-radius:12px;padding:6px 8px 8px;min-width:0}
.grp.left{grid-column:1}.grp.right{grid-column:2}.grp.span{grid-column:1 / -1}
.ghead{display:flex;align-items:center;gap:7px;width:100%;padding:3px 4px;border-radius:7px;font-size:13px;font-weight:700;color:var(--ink);text-align:left}
.ghead:hover{background:var(--mist)}
.ghead i{width:13px;height:13px;border-radius:50%;border:1.5px solid var(--c);flex-shrink:0}
.grp.all .ghead i{background:var(--c)}
.grp.some .ghead i{background:linear-gradient(90deg,var(--c) 50%,transparent 50%)}
.ghead .gc{margin-left:auto;font-size:11.5px;font-weight:400;color:var(--muted)}
.pts{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
.pt{font-size:12px;padding:3px 9px;border-radius:999px;border:1px solid var(--edge2);color:var(--muted);background:var(--panel);user-select:none;white-space:nowrap}
.pt:hover{border-color:var(--c)}
.pt.on{background:var(--cs);border-color:var(--cs);color:var(--ink)}
.pt.hot{box-shadow:0 0 0 2px var(--c)}

/* ── Folder picker ────────────────────────── */
.picker{position:fixed;inset:0;background:rgba(46,42,62,.28);display:grid;place-items:center;z-index:20;padding:20px}
.picker[hidden]{display:none}
.sheet{background:var(--panel);border-radius:18px;width:min(640px,100%);max-height:min(640px,90vh);display:flex;flex-direction:column;
  box-shadow:0 20px 60px rgba(46,42,62,.25)}
.sheet header{padding:18px 20px 10px}
.sheet h2{font-size:18px;font-weight:600}
.sheet p{color:var(--muted);font-size:13px;margin-top:2px}
.crumbs{display:flex;flex-wrap:wrap;gap:2px;padding:0 20px 10px;font-size:13px}
.crumbs button{padding:2px 6px;border-radius:6px;color:var(--accent-ink)}
.crumbs button:hover{background:var(--accent-soft)}
.crumbs span{color:var(--faint)}
.pathrow{display:flex;gap:8px;padding:0 20px 10px}
.pathrow input{flex:1;border:1px solid var(--edge);border-radius:var(--r-ctl);padding:7px 10px;font-size:13px}
.dirs{list-style:none;overflow-y:auto;border-top:1px solid var(--edge);border-bottom:1px solid var(--edge);flex:1;min-height:180px;padding:6px 10px}
.dirs button{width:100%;text-align:left;padding:7px 10px;border-radius:8px;display:flex;gap:10px;align-items:center;font-size:13.5px}
.dirs button:hover{background:var(--mist)}
.dirs svg{width:16px;height:16px;color:#A9A2D6;flex-shrink:0}
.sheet footer{display:flex;gap:8px;justify-content:flex-end;padding:14px 20px;align-items:center}
.sheet footer .err{margin-right:auto;color:#B3566A;font-size:13px}

/* ── Shortcuts ────────────────────────────── */
.keys{position:relative}
.keys .pop{right:0;left:auto;top:42px;width:250px;font-size:12.5px;gap:6px}
.keys .pop div{display:flex;justify-content:space-between;gap:10px;color:var(--muted)}
.keys .pop kbd{font:inherit;font-weight:600;color:var(--ink);background:var(--mist);border-radius:5px;padding:0 6px}

@media (max-width:980px){
  body{overflow:auto;grid-template-rows:auto 1fr}
  .top{flex-wrap:wrap;padding:8px 10px}
  .status{min-width:0}
  .app,.app.norail{grid-template-columns:1fr;grid-auto-rows:minmax(420px,auto)}
  .rail{max-height:180px;padding-right:0}
  .split.v{display:none}
  .side{min-height:640px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>

<header class="top">
  <button class="icon" id="railBtn" title="Show or hide the trial list" aria-label="Show or hide the trial list">
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><rect x="2.5" y="3.5" width="15" height="13" rx="2.5"/><path d="M7.5 3.5v13"/></svg>
  </button>
  <div class="where"><b id="rootName">CamKit3D</b><button class="link" id="openBtn">Open folder</button></div>
  <div class="trialnav">
    <button class="icon" id="prevTrial" title="Previous trial  [" aria-label="Previous trial">
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4l-6 6 6 6"/></svg></button>
    <div class="trialname" id="trialName">No trial</div>
    <button class="icon" id="nextTrial" title="Next trial  ]" aria-label="Next trial">
      <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 4l6 6-6 6"/></svg></button>
  </div>
  <label class="ver">3D output <select id="verSel" disabled></select></label>
  <div class="status num" id="status"></div>
  <details class="menu keys"><summary>Keys</summary>
    <div class="pop">
      <div>Play or pause <kbd>Space</kbd></div>
      <div>Step a frame <kbd>← →</kbd></div>
      <div>Step a second <kbd>Shift ← →</kbd></div>
      <div>Previous, next trial <kbd>[ ]</kbd></div>
      <div>Zoom time <kbd>+ −</kbd></div>
      <div>Reset zoom <kbd>0</kbd></div>
      <div>Camera views <kbd>1 2 3 4</kbd></div>
      <div>Solo a group <kbd>Shift-click</kbd></div>
      <div>Choose trials <kbd>1-4, 7</kbd></div>
    </div>
  </details>
</header>

<main class="app" id="app">
  <aside class="rail" id="rail">
    <input type="search" id="trialFilter" placeholder="Trials, e.g. 1-4, 7" title="Type trial numbers (1, 2, 5), ranges (1-4, 6-10), or part of a name" spellcheck="false">
    <div class="filtnote" id="filtNote"></div>
    <ol id="trialList"></ol>
  </aside>

  <section class="panel">
    <div class="bar">
      <div class="seg" id="sigSeg">
        <button data-sig="disp" title="Distance of each landmark from its median position">Displacement</button>
        <button data-sig="speed" title="Speed in mm per second">Speed</button>
        <button data-sig="x" title="X position, centred per landmark">X</button>
        <button data-sig="y" title="Y position, centred per landmark">Y</button>
        <button data-sig="z" title="Z position, centred per landmark">Z</button>
        <button data-sig="xyz" title="X, Y and Z overlaid">XYZ</button>
      </div>
      <label class="check" title="Scale every row the same way, so amplitudes are comparable"><input type="checkbox" id="shared"> Same scale</label>
      <label class="check" title="Draw samples filled by interpolation as a lighter line. Missing samples are gaps in the line."><input type="checkbox" id="gaps" checked> <span class="swatch" style="background:#CFCBEA;height:3px"></span> Mark filled</label>
      <span class="hint" id="hint">Scroll to zoom, drag to scrub</span>
    </div>
    <div class="plot" id="plotWrap">
      <canvas id="plot"></canvas>
      <div class="tip num" id="tip"></div>
      <div class="msg" id="msg"></div>
    </div>
    <div class="transport">
      <button class="play" id="play" title="Play or pause (space)" aria-label="Play">
        <svg id="playIcon" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>
      </button>
      <span class="time num" id="timeLbl">0:00.00</span>
      <div class="scrub" id="scrub" aria-label="Timeline">
        <div class="track"></div><div class="win" id="scrubWin"></div><div class="fill" id="scrubFill"></div><div class="knob" id="scrubKnob"></div>
      </div>
      <span class="time dim num" id="durLbl">0:00.00</span>
      <div class="seg small" id="speedSeg">
        <button data-speed="0.25">¼×</button><button data-speed="0.5">½×</button><button data-speed="1" class="on">1×</button><button data-speed="2">2×</button>
      </div>
    </div>
  </section>

  <div class="split v" id="splitV" title="Drag to resize"></div>

  <section class="side">
    <div class="panel view3d" id="view3d">
      <canvas id="c3d"></canvas>
      <div class="ov">
        <div class="seg small" id="camSeg">
          <button data-view="front">Front</button><button data-view="side">Side</button><button data-view="top">Top</button><button data-view="angle" class="on">Angle</button>
        </div>
        <details class="menu"><summary>Display</summary>
          <div class="pop">
            <label class="row">Landmarks <input type="range" id="kpSize" min="10" max="200" value="70"></label>
            <label class="row">Bones <input type="range" id="boneSize" min="1" max="20" value="8" step="0.5"></label>
            <label class="check"><input type="checkbox" id="showAxes"> Axes</label>
            <label class="check"><input type="checkbox" id="showFloor" checked> Floor</label>
            <label class="check"><input type="checkbox" id="lockCentre" checked> Centre each trial</label>
          </div>
        </details>
      </div>
      <div class="foot3d num" id="frameLbl"></div>
      <div class="nogl" id="nogl">The 3D view needs internet access to load Three.js. Traces still work.</div>
    </div>
    <div class="split h" id="splitH" title="Drag to resize"></div>
    <div class="panel kps">
      <div class="bar">
        <h3>Keypoints</h3><span class="count num" id="kpCount"></span>
        <button class="mini" id="kpAll">All</button><button class="mini" id="kpNone">None</button><button class="mini" id="kpInvert">Invert</button>
      </div>
      <div class="chips" id="chips"></div>
      <div class="kpgroups" id="kpGroups"></div>
    </div>
  </section>
</main>

<div class="picker" id="picker" hidden>
  <div class="sheet" role="dialog" aria-modal="true" aria-labelledby="pickTitle">
    <header><h2 id="pickTitle">Open a folder</h2>
      <p>Choose a single take, like trial_001, or a folder that holds many takes.</p></header>
    <div class="crumbs" id="crumbs"></div>
    <div class="pathrow"><input id="pathInput" spellcheck="false" aria-label="Folder path"><button class="btn ghost" id="pathGo">Go</button></div>
    <ul class="dirs" id="dirs"></ul>
    <footer><span class="err" id="pickErr"></span><button class="btn ghost" id="pickCancel">Cancel</button><button class="btn" id="pickOpen">Open this folder</button></footer>
  </div>
</div>

<script type="importmap">
{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}
</script>

<script type="module">
// ═════════════════════════════════════════════════════════════════════
//  Utilities
// ═════════════════════════════════════════════════════════════════════
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fmtTime = s => { s = Math.max(0, s); const m = Math.floor(s / 60); return m + ':' + (s - m * 60).toFixed(2).padStart(5, '0'); };
const prettyName = n => n.replace(/_/g, ' ');

function hexToHsl(hex) {
  hex = hex.replace('#', ''); if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
  const r = parseInt(hex.slice(0, 2), 16) / 255, g = parseInt(hex.slice(2, 4), 16) / 255, b = parseInt(hex.slice(4, 6), 16) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b); let h = 0, s = 0; const l = (mx + mn) / 2;
  if (mx !== mn) { const d = mx - mn; s = l > .5 ? d / (2 - mx - mn) : d / (mx + mn);
    h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; h /= 6; }
  return [h * 360, s * 100, l * 100];
}
// Re-tone a descriptor colour: same hue, controlled lightness/saturation.
const tone = (hex, L, sMax = 60, sMin = 28) => { const [h, s] = hexToHsl(hex); return `hsl(${h.toFixed(0)} ${clamp(s, sMin, sMax).toFixed(0)}% ${L}%)`; };
function cssToHex(css) { const c = document.createElement('canvas').getContext('2d'); c.fillStyle = css; return c.fillStyle; }

async function getJSON(url) {
  const r = await fetch(url); const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText); return j;
}

// ═════════════════════════════════════════════════════════════════════
//  State
// ═════════════════════════════════════════════════════════════════════
const S = {
  index: null, skel: null, take: -1, verByTake: new Map(),
  trial: null, t: 0, playing: false, speed: 1,
  win: [0, 1], sig: 'disp', shared: false, gaps: true,
  vis: null, hover: -1, token: 0,
};
let tracesDirty = true, needsDraw = true;
const markTraces = () => { tracesDirty = true; needsDraw = true; };

// Colours per landmark, derived once per skeleton.
let INK = [], SOFT = [], FILL = [], BONE = [], ROW_ORDER = [], GROUP_OF = null;

// ═════════════════════════════════════════════════════════════════════
//  Trial cache (browser side) with neighbour prefetch
// ═════════════════════════════════════════════════════════════════════
const CACHE_LIMIT = 700 * 1024 * 1024;
const cache = new Map();   // key -> {promise, bytes, used}

function parseTrial(buf) {
  const dv = new DataView(buf);
  const hlen = dv.getUint32(0, true);
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  const n = hdr.nFrames * hdr.nLandmarks, off = 4 + hlen;
  return { ...hdr, pos: new Float32Array(buf, off, n * 3), state: new Uint8Array(buf, off + n * 12, n), sigs: {}, bytes: buf.byteLength };
}

function trialKey(t, v) { return `${S.index.generation}:${t}:${v}`; }

function loadTrial(t, v) {
  const key = trialKey(t, v);
  let e = cache.get(key);
  if (e) { e.used = performance.now(); return e.promise; }
  e = { used: performance.now(), bytes: 0 };
  e.promise = fetch(`/api/trial?gen=${S.index.generation}&take=${t}&version=${v}`)
    .then(async r => { if (!r.ok) { let m = r.statusText; try { m = (await r.json()).error; } catch {} throw new Error(m); } return r.arrayBuffer(); })
    .then(buf => { const tr = parseTrial(buf); tr.key = key; tr.takeIdx = t; e.bytes = tr.bytes; evict(key); if (S.index && key.startsWith(S.index.generation + ':')) markRailReady(t); return tr; })
    .catch(err => { cache.delete(key); throw err; });
  cache.set(key, e);
  return e.promise;
}

function evict(keep) {
  let total = 0; for (const e of cache.values()) total += e.bytes;
  if (total <= CACHE_LIMIT) return;
  const entries = [...cache.entries()].filter(([k, e]) => k !== keep && e.bytes && (!S.trial || k !== S.trial.key)).sort((a, b) => a[1].used - b[1].used);
  for (const [k, e] of entries) { if (total <= CACHE_LIMIT) break; total -= e.bytes; cache.delete(k); }
}

function prefetchNeighbours(t) {
  const L = navList(), pos = L.indexOf(t);
  const around = pos >= 0 ? [L[pos + 1], L[pos - 1], L[pos + 2]] : [t + 1, t - 1];
  around.forEach((j, i) => {
    if (j === undefined || j < 0 || j >= S.index.takes.length) return;
    setTimeout(() => { if (S.index && j < S.index.takes.length) loadTrial(j, S.verByTake.get(j) ?? 0).catch(() => {}); }, 60 + i * 120);
  });
}

// ═════════════════════════════════════════════════════════════════════
//  Index, rail, version picker
// ═════════════════════════════════════════════════════════════════════
async function applyIndex(index) {
  S.index = index; S.take = -1; S.trial = null; S.verByTake.clear(); cache.clear();
  $('rootName').textContent = index.rootName;
  $('rootName').title = index.root || '';
  const skelChanged = !S.skel || S.skel.id !== index.skeleton.id || S.skel.names.length !== index.skeleton.names.length;
  if (skelChanged) { S.skel = index.skeleton; buildSkeleton(); }
  buildRail();
  computeFilter();
  if (!index.takes.length) {
    showMsg(index.root ? 'No 3D pose files here' : 'Open a folder to begin',
      index.root ? `Nothing in ${index.rootName} looks like CamKit3D 3D output. The viewer looks for .npy files shaped (frames, landmarks, 3) inside folders with "3d" in their name, such as data_3d.`
                 : 'Choose a single take, or a folder that holds many takes.', 'Open folder');
    $('trialName').innerHTML = 'No trial'; $('verSel').innerHTML = ''; $('verSel').disabled = true; setStatus('');
    return;
  }
  selectTrial(navList()[0] ?? 0);
}

function buildRail() {
  const list = $('trialList'); list.innerHTML = '';
  const takes = S.index.takes;
  takes.forEach((tk, i) => {
    const li = document.createElement('li');
    const b = document.createElement('button');
    const v0 = tk.versions[0];
    const dur = fmtTime((v0.nFrames - 1) / v0.fps);
    b.innerHTML = `<span class="nm"></span><span class="sub num"><span class="dot"></span>${dur}${tk.versions.length > 1 ? `, ${tk.versions.length} outputs` : ''}</span>`;
    b.querySelector('.nm').textContent = tk.name;
    b.title = tk.name;
    b.onclick = () => selectTrial(i);
    li.dataset.name = tk.name.toLowerCase();
    li.appendChild(b); list.appendChild(li);
  });
}
function markRailReady(t) { const li = $('trialList').children[t]; if (li) li.querySelector('.dot')?.classList.add('ready'); }

// Trial numbers used by the filter: the last number in each name (trial_004 -> 4)
// when every take has a distinct one, otherwise the 1-based position in the list.
function trialNumbers() {
  const takes = S.index.takes;
  const nums = takes.map(t => { const m = t.name.match(/(\d+)(?!.*\d)/); return m ? +m[1] : null; });
  const ok = nums.every(n => n !== null) && new Set(nums).size === nums.length;
  return ok ? nums : takes.map((_, i) => i + 1);
}
// "1,2,3" / "1-4" / "1-3, 7, 10-12" -> Set of numbers; null if it is not a number selection.
function parseSelection(q) {
  if (!/^[\d\s,;\-–]+$/.test(q)) return null;
  const out = new Set();
  for (let part of q.split(/[,;]/)) {
    part = part.trim().replace(/[-–]\s*$/, '').replace(/^[-–]\s*/, '');
    if (!part) continue;
    const m = part.match(/^(\d+)\s*[-–]\s*(\d+)$/);
    if (m) { let a = +m[1], b = +m[2]; if (a > b) [a, b] = [b, a]; if (b - a <= 100000) for (let i = a; i <= b; i++) out.add(i); }
    else if (/^\d+$/.test(part)) out.add(+part);
  }
  return out;
}
function navList() { return S.shown ?? (S.index ? S.index.takes.map((_, i) => i) : []); }
function computeFilter() {
  const q = $('trialFilter').value.trim(), takes = S.index?.takes || [];
  let match = null;
  if (q) {
    const sel = parseSelection(q), all = takes.map((_, i) => i);
    if (sel) { const nums = trialNumbers(); match = all.filter(i => sel.has(nums[i])); }
    else { const ql = q.toLowerCase(); match = all.filter(i => takes[i].name.toLowerCase().includes(ql)); }
  }
  S.shown = match;
  const set = match ? new Set(match) : null;
  [...$('trialList').children].forEach((li, i) => li.hidden = set ? !set.has(i) : false);
  $('filtNote').textContent = !match ? '' : match.length ? `${match.length} of ${takes.length} trials` : 'No trials match';
  return match;
}
function applyFilter() {
  if (!S.index) return;
  const match = computeFilter();
  if (match && match.length && !match.includes(S.take)) selectTrial(match[0]); else updateNav();
}
$('trialFilter').addEventListener('input', applyFilter);
$('trialFilter').addEventListener('keydown', e => {
  if (e.key === 'Escape') { e.target.value = ''; applyFilter(); e.target.blur(); }
  if (e.key === 'Enter') e.target.blur();
});
function stepTrial(d) {
  const L = navList(); if (!L.length) return;
  const pos = L.indexOf(S.take);
  let target;
  if (pos >= 0) target = L[clamp(pos + d, 0, L.length - 1)];
  else target = d > 0 ? (L.find(i => i > S.take) ?? L[L.length - 1]) : ([...L].reverse().find(i => i < S.take) ?? L[0]);
  if (target !== S.take) selectTrial(target);
}
function updateNav() {
  const takes = S.index?.takes; if (!takes || S.take < 0) return;
  const L = navList(), pos = L.indexOf(S.take);
  $('trialName').innerHTML = `<span></span><small class="num"></small>`;
  $('trialName').firstChild.textContent = takes[S.take].name;
  $('trialName').lastChild.textContent = pos >= 0 ? `${pos + 1} of ${L.length}${S.shown ? ' chosen' : ''}` : `not in the chosen trials`;
  $('prevTrial').disabled = pos <= 0 && !(pos < 0 && L.some(i => i < S.take));
  $('nextTrial').disabled = (pos < 0 ? !L.some(i => i > S.take) : pos === L.length - 1);
}

function fmtDate(sec) {
  const d = new Date(sec * 1000);
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) + ', ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });
}

async function selectTrial(t, v) {
  const takes = S.index?.takes; if (!takes || !takes.length) return;
  t = clamp(t, 0, takes.length - 1);
  if (v === undefined) v = S.verByTake.get(t) ?? 0;
  S.verByTake.set(t, v);
  if (S.take === t && S.trial && S.trial.key === trialKey(t, v)) return;
  S.take = t;
  const tk = takes[t], token = ++S.token;

  // Immediate UI feedback
  [...$('trialList').children].forEach((li, i) => li.firstChild.classList.toggle('on', i === t));
  $('trialList').children[t]?.firstChild.scrollIntoView({ block: 'nearest' });
  updateNav();
  const sel = $('verSel'); sel.innerHTML = '';
  tk.versions.forEach((ver, j) => { const o = document.createElement('option'); o.value = j;
    o.textContent = tk.versions.length > 1 ? `${ver.label} (${j === 0 ? 'newest, ' : ''}${fmtDate(ver.mtime)})` : ver.label; o.title = ver.label; sel.appendChild(o); });
  sel.value = v; sel.disabled = tk.versions.length < 2;

  const slow = setTimeout(() => { if (token === S.token) setStatus('<span class="spin"></span>Loading ' + escapeHtml(tk.name)); }, 120);
  try {
    const tr = await loadTrial(t, v);
    clearTimeout(slow);
    if (token !== S.token) return;
    const wasPlaying = S.playing;
    S.trial = tr; S.t = 0; S.win = [0, tr.nFrames - 1];
    hideMsg(); setTrialScene(tr); markTraces(); updateTransport();
    const pctF = 100 * tr.nFilled / (tr.nFrames * tr.nLandmarks), pctM = 100 * tr.nMissing / (tr.nFrames * tr.nLandmarks);
    setStatus(`${tr.nFrames} frames at ${+tr.fps.toFixed(2)} fps${tr.aligned ? ', aligned' : ''}<br>${tr.interpolated ? `${pctF.toFixed(1)}% filled, ` : (S.index.interpolate ? '' : 'raw, ')}${pctM.toFixed(1)}% missing`);
    $('status').title = tr.aligned ? `Rotated to the anatomical frame detected from ${tr.alignedTo}` : (S.index.align ? 'Not aligned (see the Python console)' : 'Calibration frame (align=False)');
    const dirs = tr.aligned ? ['right +', 'forward +', 'up +'] : null;
    ['x', 'y', 'z'].forEach((a, i) => { $('sigSeg').querySelector(`[data-sig="${a}"]`).title = `${a.toUpperCase()} position${dirs ? ` (${dirs[i]})` : ''}, centred per landmark`; });
    S.playing = wasPlaying; setPlayIcon();
    prefetchNeighbours(t);
  } catch (err) {
    clearTimeout(slow);
    if (token !== S.token) return;
    S.trial = null; markTraces();
    showMsg(`Couldn't open ${tk.name}`, err.message, null);
    setStatus('');
  }
}
$('verSel').onchange = e => selectTrial(S.take, +e.target.value);
$('prevTrial').onclick = () => stepTrial(-1);
$('nextTrial').onclick = () => stepTrial(1);
$('railBtn').onclick = () => { $('app').classList.toggle('norail'); needsDraw = true; };

function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function setStatus(html) { $('status').innerHTML = html; }
function showMsg(title, body, action) {
  const m = $('msg'); m.innerHTML = `<div><h2></h2><p></p>${action ? `<button class="btn">${action}</button>` : ''}</div>`;
  m.querySelector('h2').textContent = title; m.querySelector('p').textContent = body;
  if (action) m.querySelector('button').onclick = () => openPicker();
  m.classList.add('show');
}
function hideMsg() { $('msg').classList.remove('show'); }

// ═════════════════════════════════════════════════════════════════════
//  Keypoint picker
// ═════════════════════════════════════════════════════════════════════
function buildSkeleton() {
  const sk = S.skel, nK = sk.names.length, parts = sk.partition;
  S.vis = new Uint8Array(nK).fill(1);
  GROUP_OF = new Int16Array(nK);
  parts.forEach((g, gi) => g.indices.forEach(i => GROUP_OF[i] = gi));
  const colorOf = k => parts[GROUP_OF[k]].color;
  INK = Array.from({ length: nK }, (_, k) => tone(colorOf(k), 56, 48, 30));
  SOFT = Array.from({ length: nK }, (_, k) => tone(colorOf(k), 80, 70, 40));
  FILL = Array.from({ length: nK }, (_, k) => tone(colorOf(k), 82, 45, 25));
  BONE = sk.edgeColors.map(c => cssToHex(tone(c, 76, 62, 36)));
  ROW_ORDER = parts.flatMap(g => g.indices);

  // One block per exclusive group; left groups in the left column, right in the right.
  const stripSide = n => n.split('_').filter(p => p !== 'left' && p !== 'right').join(' ') || prettyName(n);
  const chips = $('chips'); chips.innerHTML = '';
  parts.forEach((g, gi) => {
    const b = document.createElement('button'); b.className = 'chip'; b.dataset.g = gi;
    b.style.setProperty('--c', tone(g.color, 60, 60, 35));
    b.style.setProperty('--cs', tone(g.color, 86, 70, 40));
    b.innerHTML = '<i></i><span></span>'; b.lastChild.textContent = g.label;
    b.title = `Show or hide ${g.label.toLowerCase()}. Shift-click to show only ${g.label.toLowerCase()}.`;
    b.onclick = e => toggleGroup(g, e);
    chips.appendChild(b);
  });
  const wrap = $('kpGroups'); wrap.innerHTML = '';
  parts.forEach((g, gi) => {
    const box = document.createElement('div');
    box.className = 'grp ' + (g.side || 'span'); box.dataset.g = gi;
    box.style.setProperty('--c', tone(g.color, 60, 60, 35));
    box.style.setProperty('--cs', tone(g.color, 86, 70, 40));
    const head = document.createElement('button'); head.className = 'ghead';
    head.innerHTML = '<i></i><span class="gl"></span><span class="gc num"></span>';
    head.querySelector('.gl').textContent = g.label;
    head.title = 'Click to show or hide this group. Shift-click to show only this group.';
    head.onclick = e => toggleGroup(g, e);
    const pts = document.createElement('div'); pts.className = 'pts';
    g.indices.forEach(k => {
      const b = document.createElement('button'); b.className = 'pt'; b.dataset.k = k;
      b.textContent = g.side ? stripSide(sk.names[k]) : prettyName(sk.names[k]);
      b.title = prettyName(sk.names[k]);
      b.onclick = () => { S.vis[k] ^= 1; refreshKp(); };
      b.onmouseenter = () => setHover(k); b.onmouseleave = () => setHover(-1);
      pts.appendChild(b);
    });
    box.append(head, pts); wrap.appendChild(box);
  });
  build3DSkeleton();
  refreshKp();
}

function toggleGroup(g, e) {
  if (e && (e.shiftKey || e.altKey)) { S.vis.fill(0); g.indices.forEach(i => S.vis[i] = 1); }
  else { const allOn = g.indices.every(i => S.vis[i]); g.indices.forEach(i => S.vis[i] = allOn ? 0 : 1); }
  refreshKp();
}

function refreshKp() {
  const sk = S.skel, n = S.vis.reduce((a, b) => a + b, 0);
  $('kpCount').textContent = `${n} of ${sk.names.length}`;
  $('kpGroups').querySelectorAll('.pt').forEach(p => p.classList.toggle('on', !!S.vis[+p.dataset.k]));
  $('kpGroups').querySelectorAll('.grp').forEach(box => {
    const g = sk.partition[+box.dataset.g], on = g.indices.filter(i => S.vis[i]).length;
    box.classList.toggle('all', on === g.indices.length); box.classList.toggle('some', on > 0 && on < g.indices.length);
    box.querySelector('.gc').textContent = `${on}/${g.indices.length}`;
  });
  $('chips').querySelectorAll('.chip').forEach(c => {
    const g = sk.partition[+c.dataset.g], on = g.indices.filter(i => S.vis[i]).length;
    c.classList.toggle('all', on === g.indices.length); c.classList.toggle('some', on > 0 && on < g.indices.length);
  });
  markTraces(); pose3dDirty = true;
}
$('kpAll').onclick = () => { S.vis.fill(1); refreshKp(); };
$('kpNone').onclick = () => { S.vis.fill(0); refreshKp(); };
$('kpInvert').onclick = () => { S.vis.forEach((v, i) => S.vis[i] = v ? 0 : 1); refreshKp(); };

function setHover(k) {
  if (S.hover === k) return; S.hover = k; needsDraw = true; pose3dDirty = true;
  $('kpGroups').querySelectorAll('.pt').forEach(p => p.classList.toggle('hot', +p.dataset.k === k));
}

// ═════════════════════════════════════════════════════════════════════
//  Signals for the trace view
// ═════════════════════════════════════════════════════════════════════
function robustRange(arr, nF, nK, k, pLo, pHi) {
  const step = Math.max(1, Math.floor(nF / 2500)), vals = [];
  for (let f = 0; f < nF; f += step) { const v = arr[f * nK + k]; if (v === v) vals.push(v); }
  if (!vals.length) return [0, 1];
  vals.sort((a, b) => a - b);
  return [vals[Math.floor(pLo * (vals.length - 1))], vals[Math.floor(pHi * (vals.length - 1))]];
}

function channel(tr, c) {
  const key = 'c' + c; if (tr.sigs[key]) return tr.sigs[key];
  const { nFrames: nF, nLandmarks: nK, pos } = tr, out = new Float32Array(nF * nK);
  // centre each landmark on its median
  const med = new Float32Array(nK);
  for (let k = 0; k < nK; k++) {
    const step = Math.max(1, Math.floor(nF / 2500)), vals = [];
    for (let f = 0; f < nF; f += step) { const v = pos[(f * nK + k) * 3 + c]; if (v === v) vals.push(v); }
    vals.sort((a, b) => a - b); med[k] = vals.length ? vals[vals.length >> 1] : 0;
  }
  for (let f = 0; f < nF; f++) for (let k = 0; k < nK; k++) out[f * nK + k] = pos[(f * nK + k) * 3 + c] - med[k];
  return (tr.sigs[key] = out);
}

function signal(tr, mode) {
  if (tr.sigs[mode]) return tr.sigs[mode];
  const { nFrames: nF, nLandmarks: nK, pos, fps } = tr;
  let chans, nonneg = false;
  if (mode === 'x' || mode === 'y' || mode === 'z') chans = [channel(tr, 'xyz'.indexOf(mode))];
  else if (mode === 'xyz') chans = [channel(tr, 0), channel(tr, 1), channel(tr, 2)];
  else if (mode === 'disp') {
    const cx = channel(tr, 0), cy = channel(tr, 1), cz = channel(tr, 2), out = new Float32Array(nF * nK);
    for (let i = 0; i < out.length; i++) out[i] = Math.hypot(cx[i], cy[i], cz[i]);
    chans = [out]; nonneg = true;
  } else {
    const out = new Float32Array(nF * nK);
    for (let f = 0; f < nF; f++) {
      const a = Math.max(0, f - 1), b = Math.min(nF - 1, f + 1), dt = (b - a) / fps;
      for (let k = 0; k < nK; k++) {
        const ia = (a * nK + k) * 3, ib = (b * nK + k) * 3;
        out[f * nK + k] = Math.hypot(pos[ib] - pos[ia], pos[ib + 1] - pos[ia + 1], pos[ib + 2] - pos[ia + 2]) / dt;
      }
    }
    chans = [out]; nonneg = true;
  }
  const lo = new Float32Array(nK), hi = new Float32Array(nK);
  for (let k = 0; k < nK; k++) {
    let a = Infinity, b = -Infinity;
    for (const ch of chans) { const [l, h] = robustRange(ch, nF, nK, k, .005, .995); a = Math.min(a, l); b = Math.max(b, h); }
    if (nonneg) a = 0;
    if (!(b > a)) { b = a + 1; }
    lo[k] = a; hi[k] = b;
  }
  const unit = mode === 'speed' ? 'mm/s' : 'mm';
  return (tr.sigs[mode] = { chans, lo, hi, nonneg, unit });
}

// ═════════════════════════════════════════════════════════════════════
//  Trace view: off-screen render + cheap overlay per animation frame
// ═════════════════════════════════════════════════════════════════════
const plot = $('plot'), pctx = plot.getContext('2d');
const off = document.createElement('canvas'), octx = off.getContext('2d');
const XYZ_INK = ['hsl(352 55% 64%)', 'hsl(152 34% 50%)', 'hsl(220 55% 62%)'];
const XYZ_FILL = ['hsl(352 50% 85%)', 'hsl(152 30% 80%)', 'hsl(220 50% 85%)'];
let L = { W: 0, H: 0, dpr: 1, gx: 0, pw: 0, top: 8, ph: 0, rows: [], rowH: 0 };

function layout() {
  const r = $('plotWrap').getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
  const W = Math.max(10, Math.floor(r.width)), H = Math.max(10, Math.floor(r.height));
  if (W !== L.W || H !== L.H || dpr !== L.dpr) {
    plot.width = off.width = W * dpr; plot.height = off.height = H * dpr;
    L.W = W; L.H = H; L.dpr = dpr; markTraces();
  }
  L.rows = S.vis ? ROW_ORDER.filter(k => S.vis[k]) : [];
  L.gx = 128; L.top = 8; L.axis = 24; L.pw = Math.max(10, L.W - L.gx - 14); L.ph = Math.max(10, L.H - L.top - L.axis);
  L.rowH = L.rows.length ? L.ph / L.rows.length : L.ph;
}
const fToX = f => L.gx + (f - S.win[0]) / Math.max(1e-9, S.win[1] - S.win[0]) * L.pw;
const xToF = x => S.win[0] + (x - L.gx) / L.pw * (S.win[1] - S.win[0]);

function niceStep(secPerPx) {
  const target = secPerPx * 110, steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200];
  return steps.find(s => s >= target) || 3600;
}

function rowScale(sg, k) {
  if (!S.shared) return [sg.lo[k], sg.hi[k]];
  if (!sg._shared) sg._shared = new Map();
  const rowsKey = L.rows.join(',');
  let sh = sg._shared.get(rowsKey);
  if (!sh) {
    let span = 0, top = 0;
    for (const r of L.rows) { span = Math.max(span, sg.hi[r] - sg.lo[r]); top = Math.max(top, sg.hi[r]); }
    sh = { span, top }; sg._shared.set(rowsKey, sh);
  }
  if (sg.nonneg) return [0, sh.top];
  const c = (sg.lo[k] + sg.hi[k]) / 2; return [c - sh.span / 2, c + sh.span / 2];
}

function renderTraces() {
  tracesDirty = false; layout();
  const c = octx, { dpr, W, H, gx, pw, top, ph, rows, rowH } = L;
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const tr = S.trial; if (!tr || !S.skel) return;
  const { nFrames: nF, nLandmarks: nK, fps, state } = tr;
  const sg = signal(tr, S.sig);
  const [w0, w1] = S.win, fpp = (w1 - w0) / pw;

  // Time grid
  const step = niceStep(fpp / fps);
  c.strokeStyle = '#EEEBF4'; c.lineWidth = 1; c.fillStyle = '#9A94AC'; c.font = '11.5px Arial, Liberation Sans, Helvetica, sans-serif'; c.textAlign = 'center'; c.textBaseline = 'top';
  c.beginPath();
  for (let s = Math.ceil(w0 / fps / step) * step; s * fps <= w1 + 1e-6; s += step) {
    const x = Math.round(fToX(s * fps)) + .5; c.moveTo(x, top); c.lineTo(x, top + ph);
    c.fillText(step < 1 ? s.toFixed(step < 0.1 ? 2 : 1) + ' s' : (s >= 60 ? fmtTime(s).replace(/\.00$/, '') : s.toFixed(0) + ' s'), x, top + ph + 6);
  }
  c.stroke();
  if (!rows.length) {
    c.fillStyle = '#9A94AC'; c.font = '14px Arial, Liberation Sans, Helvetica, sans-serif'; c.textBaseline = 'middle';
    c.fillText('No keypoints selected. Pick some on the right.', gx + pw / 2, top + ph / 2); return;
  }

  const f0 = Math.max(0, Math.floor(w0)), f1 = Math.min(nF - 1, Math.ceil(w1));
  const pad = Math.min(4, rowH * .12);
  const lw = rowH < 10 ? 1 : rowH < 24 ? 1.25 : 1.5;

  for (let r = 0; r < rows.length; r++) {
    const k = rows[r], y0 = top + r * rowH, yTop = y0 + pad, yH = rowH - 2 * pad;
    const [lo, hi] = rowScale(sg, k), inv = 1 / (hi - lo);
    const yOf = v => clamp(yTop + (1 - (v - lo) * inv) * yH, y0, y0 + rowH);
    const mark = S.gaps;
    sg.chans.forEach((ch, ci) => {
      // Two paths: measured samples in the full colour, interpolated ones lighter.
      // Missing samples break the line.
      const pN = new Path2D(), pF = mark ? new Path2D() : pN;
      let lx = 0, ly = 0, lk = -1;
      const seg = (x, y, kind) => {
        if (lk >= 0) { const path = (kind === 1 || lk === 1) ? pF : pN; path.moveTo(lx, ly); path.lineTo(x, y); }
        lx = x; ly = y; lk = kind;
      };
      if (fpp <= 1.5) {
        for (let f = f0; f <= f1; f++) {
          const i = f * nK + k, v = ch[i];
          if (v !== v) { lk = -1; continue; }
          seg(fToX(f), yOf(v), state[i]);
        }
      } else {
        // min/max envelope per pixel column
        for (let px = 0; px < pw; px++) {
          const fa = Math.max(0, Math.floor(w0 + px * fpp)), fb = Math.min(nF - 1, Math.floor(w0 + (px + 1) * fpp));
          let mn = Infinity, mx = -Infinity, measured = false;
          for (let f = fa; f <= fb; f++) { const i = f * nK + k, v = ch[i]; if (v === v) { if (v < mn) mn = v; if (v > mx) mx = v; if (state[i] === 0) measured = true; } }
          if (mn === Infinity) { lk = -1; continue; }
          const x = gx + px + .5, kind = measured ? 0 : 1;
          seg(x, yOf(mx), kind); seg(x, yOf(mn), kind);
        }
      }
      c.lineWidth = lw; c.lineCap = 'round';
      c.strokeStyle = sg.chans.length > 1 ? XYZ_INK[ci] : INK[k]; c.stroke(pN);
      if (mark) { c.strokeStyle = sg.chans.length > 1 ? XYZ_FILL[ci] : FILL[k]; c.stroke(pF); }
    });
  }

  // Labels
  if (rowH >= 7) {
    const fs = clamp(rowH * .62, 8.5, 12.5);
    c.font = `500 ${fs}px Arial, Liberation Sans, Helvetica, sans-serif`; c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let r = 0; r < rows.length; r++) {
      const k = rows[r], y = top + (r + .5) * rowH;
      c.fillStyle = '#5E5873'; c.fillText(prettyName(S.skel.names[k]), gx - 16, y, gx - 22);
      c.fillStyle = SOFT[k]; c.beginPath(); c.arc(gx - 8, y, Math.min(3.5, rowH * .28), 0, 6.283); c.fill();
    }
  } else {
    c.fillStyle = '#9A94AC'; c.font = '12px Arial, Liberation Sans, Helvetica, sans-serif'; c.textAlign = 'right'; c.textBaseline = 'middle';
    c.fillText(`${rows.length} rows`, gx - 10, top + ph / 2);
    c.fillText('hover for names', gx - 10, top + ph / 2 + 16);
  }
  c.strokeStyle = '#E7E2F1'; c.beginPath(); c.moveTo(gx + .5, top); c.lineTo(gx + .5, top + ph); c.stroke();
}

let mouse = { x: -1, y: -1, inside: false };
function drawPlot() {
  needsDraw = false;
  const c = pctx, { dpr, W, H, gx, pw, top, ph, rows, rowH } = L;
  c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, plot.width, plot.height); c.drawImage(off, 0, 0);
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const tr = S.trial; if (!tr) return;

  // Hovered row
  const hr = rows.indexOf(S.hover);
  if (hr >= 0) {
    const yc = clamp(top + (hr + .5) * rowH, top + 8, top + ph - 8), bh = Math.max(rowH, 16);
    c.fillStyle = '#FFFFFF'; c.fillRect(0, yc - bh / 2, gx - 1, bh);
    c.fillStyle = '#2E2A3E'; c.font = `700 ${clamp(rowH * .62, 11, 12.5)}px Arial, Liberation Sans, Helvetica, sans-serif`;
    c.textAlign = 'right'; c.textBaseline = 'middle';
    c.fillText(prettyName(S.skel.names[S.hover]), gx - 16, yc, gx - 22);
    c.fillStyle = '#7F7BD0'; c.beginPath(); c.arc(gx - 8, yc, 3.5, 0, 6.283); c.fill();
  }
  // Hover crosshair
  if (mouse.inside && mouse.x >= gx) {
    c.strokeStyle = 'rgba(46,42,62,.18)'; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(mouse.x + .5, top); c.lineTo(mouse.x + .5, top + ph); c.stroke(); c.setLineDash([]);
  }
  // Playhead
  const f = S.t * tr.fps;
  if (f >= S.win[0] - 1 && f <= S.win[1] + 1) {
    const x = Math.round(fToX(f)) + .5;
    c.strokeStyle = '#7F7BD0'; c.lineWidth = 2; c.beginPath(); c.moveTo(x, top); c.lineTo(x, top + ph); c.stroke();
    c.fillStyle = '#7F7BD0'; c.beginPath(); c.moveTo(x - 5, top - 2); c.lineTo(x + 5, top - 2); c.lineTo(x, top + 5); c.fill();
  }
}

function updateTip() {
  const tip = $('tip'), tr = S.trial;
  if (!tr || !mouse.inside || S.hover < 0 || mouse.x < L.gx) { tip.style.opacity = 0; return; }
  const f = clamp(Math.round(xToF(mouse.x)), 0, tr.nFrames - 1), k = S.hover, sg = signal(tr, S.sig), i = f * tr.nLandmarks + k;
  const st = tr.state[i], vals = sg.chans.map(ch => ch[i]);
  const valTxt = vals.every(v => v !== v) ? 'missing' : vals.map(v => (v !== v ? '–' : v.toFixed(sg.unit === 'mm/s' ? 0 : 1))).join(', ') + ' ' + sg.unit;
  tip.innerHTML = `<b></b> <span>${(f / tr.fps).toFixed(2)} s, frame ${f}</span><br>${valTxt}${st === 1 ? ' <span>(filled)</span>' : ''}`;
  tip.querySelector('b').textContent = prettyName(S.skel.names[k]);
  const flip = mouse.x > L.W - 220;
  tip.style.left = mouse.x + 'px'; tip.style.top = mouse.y + 'px';
  tip.style.transform = flip ? 'translate(calc(-100% - 12px),-50%)' : 'translate(12px,-50%)';
  tip.style.opacity = 1;
}

// Interaction on traces
function setTimeFromFrame(f) { const tr = S.trial; if (!tr) return; S.t = clamp(f, 0, tr.nFrames - 1) / tr.fps; needsDraw = true; pose3dDirty = true; updateTransport(); }
function setWindow(a, b) {
  const tr = S.trial; if (!tr) return; const n = tr.nFrames - 1, minSpan = Math.min(n, 10);
  let span = clamp(b - a, minSpan, n); a = clamp(a, 0, n - span); S.win = [a, a + span]; markTraces(); updateTransport();
}
let dragging = false;
plot.addEventListener('pointerdown', e => {
  if (!S.trial || e.button !== 0) return; const x = e.offsetX; if (x < L.gx) return;
  dragging = true; plot.setPointerCapture(e.pointerId); setTimeFromFrame(xToF(x));
});
plot.addEventListener('pointermove', e => {
  mouse = { x: e.offsetX, y: e.offsetY, inside: true };
  if (dragging) setTimeFromFrame(xToF(clamp(e.offsetX, L.gx, L.gx + L.pw)));
  const r = Math.floor((e.offsetY - L.top) / L.rowH);
  setHover(e.offsetY >= L.top && r >= 0 && r < L.rows.length ? L.rows[r] : -1);
  needsDraw = true; updateTip();
});
plot.addEventListener('pointerup', () => dragging = false);
plot.addEventListener('pointerleave', () => { mouse.inside = false; setHover(-1); needsDraw = true; updateTip(); });
plot.addEventListener('dblclick', () => { if (S.trial) setWindow(0, S.trial.nFrames - 1); });
plot.addEventListener('wheel', e => {
  if (!S.trial) return; e.preventDefault();
  const span = S.win[1] - S.win[0];
  const dx = e.shiftKey ? e.deltaY : e.deltaX, dy = e.shiftKey ? 0 : e.deltaY;
  if (Math.abs(dx) > Math.abs(dy)) { const sh = dx / L.pw * span; setWindow(S.win[0] + sh, S.win[1] + sh); }
  else {
    const fx = clamp(xToF(Math.max(e.offsetX, L.gx)), 0, S.trial.nFrames - 1);
    const z = Math.exp(dy * (e.ctrlKey ? .01 : .0025)), ns = span * z;
    setWindow(fx - (fx - S.win[0]) * ns / span, fx + (S.win[1] - fx) * ns / span);
  }
  updateTip();
}, { passive: false });

// Timeline scrubber
function updateTransport() {
  const tr = S.trial; if (!tr) { $('timeLbl').textContent = fmtTime(0); $('durLbl').textContent = fmtTime(0); return; }
  const n = Math.max(1, tr.nFrames - 1), f = S.t * tr.fps;
  $('timeLbl').textContent = fmtTime(S.t); $('durLbl').textContent = fmtTime(n / tr.fps);
  $('scrubKnob').style.left = (100 * f / n) + '%'; $('scrubFill').style.width = (100 * f / n) + '%';
  const full = S.win[0] <= 0 && S.win[1] >= n;
  const w = $('scrubWin'); w.style.display = full ? 'none' : 'block';
  w.style.left = (100 * S.win[0] / n) + '%'; w.style.width = (100 * (S.win[1] - S.win[0]) / n) + '%';
  $('frameLbl').textContent = `Frame ${Math.round(f)} of ${n}`;
}
{
  const sc = $('scrub'); let drag = false;
  const seek = e => { const tr = S.trial; if (!tr) return; const r = sc.getBoundingClientRect();
    const f = clamp((e.clientX - r.left) / r.width, 0, 1) * (tr.nFrames - 1); setTimeFromFrame(f); keepInWindow(true); };
  sc.addEventListener('pointerdown', e => { drag = true; sc.setPointerCapture(e.pointerId); seek(e); });
  sc.addEventListener('pointermove', e => drag && seek(e));
  sc.addEventListener('pointerup', () => drag = false);
}

function keepInWindow(centre) {
  const tr = S.trial; if (!tr) return; const n = tr.nFrames - 1, span = S.win[1] - S.win[0];
  if (span >= n) return; const f = S.t * tr.fps;
  if (f < S.win[0] || f > S.win[0] + span * .92) {
    const a = centre ? f - span / 2 : f - span * .08; setWindow(a, a + span);
  }
}

// ═════════════════════════════════════════════════════════════════════
//  Playback
// ═════════════════════════════════════════════════════════════════════
function setPlayIcon() {
  $('playIcon').innerHTML = S.playing ? '<rect x="3.5" y="2.5" width="3" height="11" rx="1"/><rect x="9.5" y="2.5" width="3" height="11" rx="1"/>' : '<path d="M4 2.5v11l9-5.5z"/>';
  $('play').setAttribute('aria-label', S.playing ? 'Pause' : 'Play');
}
function togglePlay() {
  if (!S.trial) return;
  if (!S.playing && S.t * S.trial.fps >= S.trial.nFrames - 1.01) S.t = 0;
  S.playing = !S.playing; setPlayIcon();
}
$('play').onclick = togglePlay;
$('speedSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return; S.speed = +b.dataset.speed;
  $('speedSeg').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
});
$('sigSeg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return; S.sig = b.dataset.sig;
  $('sigSeg').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); markTraces();
  $('hint').innerHTML = S.sig === 'xyz'
    ? XYZ_INK.map((c, i) => `<span class="check" style="cursor:default;margin-left:8px"><span class="swatch" style="background:${c};height:3px"></span> ${'XYZ'[i]}</span>`).join('')
    : 'Scroll to zoom, drag to scrub';
});
$('sigSeg').querySelector('[data-sig="disp"]').classList.add('on');
$('shared').onchange = e => { S.shared = e.target.checked; markTraces(); };
$('gaps').onchange = e => { S.gaps = e.target.checked; markTraces(); };

document.addEventListener('keydown', e => {
  if (e.target.matches('input:not([type=checkbox]):not([type=range]),select,textarea') || !$('picker').hidden) return;
  const tr = S.trial;
  switch (e.key) {
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft': case 'ArrowRight': {
      if (!tr) break; e.preventDefault(); const d = (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? tr.fps : 1);
      setTimeFromFrame(Math.round(S.t * tr.fps) + d); keepInWindow(false); break; }
    case '[': case 'ArrowUp': case 'PageUp': e.preventDefault(); stepTrial(-1); break;
    case ']': case 'ArrowDown': case 'PageDown': e.preventDefault(); stepTrial(1); break;
    case 'Home': setTimeFromFrame(0); keepInWindow(true); break;
    case 'End': if (tr) { setTimeFromFrame(tr.nFrames - 1); keepInWindow(true); } break;
    case '+': case '=': case '-': case '_': {
      if (!tr) break; const f = S.t * tr.fps, span = (S.win[1] - S.win[0]) * (e.key === '+' || e.key === '=' ? .6 : 1 / .6);
      setWindow(f - span / 2, f + span / 2); break; }
    case '0': if (tr) setWindow(0, tr.nFrames - 1); break;
    case '1': case '2': case '3': case '4': setView(['front', 'side', 'top', 'angle'][+e.key - 1]); break;
  }
});

// ═════════════════════════════════════════════════════════════════════
//  3D view (Three.js loaded lazily so traces work offline)
// ═════════════════════════════════════════════════════════════════════
let THREE = null, OrbitControls = null;
try {
  THREE = await import('three');
  ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));
} catch (err) { $('nogl').style.display = 'grid'; }

let R3 = null, scene, cam, ctrl, kpIM, boneMeshes = [], axGroup, grid, root3 = null;
let kpScale = 70 / 100 * 0.015, boneScale = 8 / 10 * 0.006, pose3dDirty = true, lastPoseKey = '';
// Scene = (X, Z, -Y): with aligned data the person faces scene -z, so "Front" looks at their face.
const VIEWS = { front: [3.1416, 1.5708, 3.1], side: [1.5708, 1.5708, 3.1], top: [0, 0.15, 3.4], angle: [2.3562, 1.2, 3.2] };
let sceneCtr = [0, 0, 0], sceneS = 1, sceneSet = false;

function init3D() {
  if (!THREE) return;
  const canvas = $('c3d');
  R3 = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: false });
  R3.setPixelRatio(Math.min(devicePixelRatio, 2)); R3.setClearColor(0x000000, 0);
  scene = new THREE.Scene();
  cam = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
  ctrl = new OrbitControls(cam, canvas); ctrl.enableDamping = true; ctrl.dampingFactor = .12; ctrl.rotateSpeed = .7; ctrl.panSpeed = .5;
  ctrl.addEventListener('start', () => $('camSeg').querySelectorAll('button').forEach(b => b.classList.remove('on')));
  scene.add(new THREE.HemisphereLight(0xffffff, 0xE6E0F4, 1.1));
  const dl = new THREE.DirectionalLight(0xffffff, .9); dl.position.set(3, 5, 4); scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0xC9C2F0, .35); dl2.position.set(-3, 2, -3); scene.add(dl2);

  axGroup = new THREE.Group();
  // X (red) right, Y (green) forward, Z (blue) up, drawn in data directions
  [[1, 0, 0, 0xE59AA6], [0, 0, -1, 0x8FC9A8], [0, 1, 0, 0x93AEE3]].forEach(([x, y, z, col]) => {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(x, y, z).multiplyScalar(.5)]);
    axGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col })));
  });
  axGroup.visible = false; scene.add(axGroup);

  grid = new THREE.GridHelper(4, 16, 0xD8D1EA, 0xE8E3F3); grid.material.transparent = true; grid.material.opacity = .9; scene.add(grid);
  root3 = new THREE.Group(); scene.add(root3);
  setView('angle');
  new ResizeObserver(resize3D).observe($('view3d')); resize3D();
}
function resize3D() {
  if (!R3) return; const el = $('view3d'), w = el.clientWidth, h = el.clientHeight; if (!w || !h) return;
  R3.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix();
}
function setView(name) {
  if (!R3) return; const [t, p, d] = VIEWS[name];
  cam.position.set(d * Math.sin(p) * Math.sin(t), d * Math.cos(p), d * Math.sin(p) * Math.cos(t));
  ctrl.target.set(0, 0, 0); ctrl.update();
  $('camSeg').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.view === name));
}
$('camSeg').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setView(b.dataset.view); });

function build3DSkeleton() {
  if (!R3 || !S.skel) return;
  root3.clear(); boneMeshes = [];
  const nK = S.skel.names.length;
  kpIM = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 18, 14), new THREE.MeshStandardMaterial({ roughness: .5, metalness: 0 }), nK);
  kpIM.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  for (let k = 0; k < nK; k++) kpIM.setColorAt(k, new THREE.Color(cssToHex(tone(S.skel.partition[GROUP_OF[k]].color, 60, 58, 34))));
  kpIM.instanceColor.needsUpdate = true; root3.add(kpIM);
  const geo = new THREE.CylinderGeometry(1, 1, 1, 10, 1);
  S.skel.edges.forEach((e, i) => {
    const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: new THREE.Color(BONE[i]), roughness: .6 }));
    m.visible = false; root3.add(m); boneMeshes.push(m);
  });
  pose3dDirty = true;
}

function setTrialScene(tr) {
  if (!R3) return;
  if ($('lockCentre').checked || !sceneSet) {
    sceneCtr = tr.centre; sceneS = 1 / tr.extent; sceneSet = true;
    grid.position.y = (tr.floorZ - sceneCtr[2]) * sceneS;
  }
  pose3dDirty = true;
}

const _o = THREE ? new THREE.Object3D() : null, _a = THREE ? new THREE.Vector3() : null, _b = THREE ? new THREE.Vector3() : null,
      _d = THREE ? new THREE.Vector3() : null, _up = THREE ? new THREE.Vector3(0, 1, 0) : null, pts = [];

function updatePose() {
  const tr = S.trial; if (!R3 || !kpIM) return;
  const nK = S.skel.names.length;
  if (!tr) { kpIM.visible = false; boneMeshes.forEach(m => m.visible = false); return; }
  kpIM.visible = true;
  const ff = clamp(S.t * tr.fps, 0, tr.nFrames - 1), f0 = Math.floor(ff), f1 = Math.min(f0 + 1, tr.nFrames - 1), a = ff - f0;
  const key = `${tr.key}|${ff.toFixed(3)}|${S.hover}|${kpScale}|${boneScale}`;
  if (!pose3dDirty && key === lastPoseKey) return;
  lastPoseKey = key; pose3dDirty = false;
  const P = tr.pos;
  for (let k = 0; k < nK; k++) {
    const i0 = (f0 * nK + k) * 3, i1 = (f1 * nK + k) * 3;
    let x = P[i0], y = P[i0 + 1], z = P[i0 + 2];
    const x1 = P[i1], y1 = P[i1 + 1], z1 = P[i1 + 2];
    if (x === x && x1 === x1) { x += (x1 - x) * a; y += (y1 - y) * a; z += (z1 - z) * a; }
    else if (x !== x) { x = x1; y = y1; z = z1; }
    const ok = x === x && S.vis[k];
    // Data Z is up; data Y (forward) points to scene -z. A rotation, not a mirror.
    pts[k] = ok ? [(x - sceneCtr[0]) * sceneS, (z - sceneCtr[2]) * sceneS, -(y - sceneCtr[1]) * sceneS] : null;
    if (ok) { _o.position.set(pts[k][0], pts[k][1], pts[k][2]); _o.scale.setScalar(kpScale * (k === S.hover ? 2 : 1)); }
    else _o.scale.setScalar(0);
    _o.updateMatrix(); kpIM.setMatrixAt(k, _o.matrix);
  }
  kpIM.instanceMatrix.needsUpdate = true;
  S.skel.edges.forEach(([s, e], i) => {
    const m = boneMeshes[i], ps = pts[s], pe = pts[e];
    if (!ps || !pe) { m.visible = false; return; }
    _a.set(ps[0], ps[1], ps[2]); _b.set(pe[0], pe[1], pe[2]); _d.subVectors(_b, _a);
    const len = _d.length(); if (len < 1e-6) { m.visible = false; return; }
    m.visible = true; m.position.lerpVectors(_a, _b, .5); m.quaternion.setFromUnitVectors(_up, _d.normalize()); m.scale.set(boneScale, len, boneScale);
  });
}
$('kpSize').oninput = e => { kpScale = +e.target.value / 100 * 0.015; };
$('boneSize').oninput = e => { boneScale = +e.target.value / 10 * 0.006; };
$('showAxes').onchange = e => { if (axGroup) axGroup.visible = e.target.checked; };
$('lockCentre').onchange = e => { if (e.target.checked && S.trial) setTrialScene(S.trial); };
$('showFloor').onchange = e => { if (grid) grid.visible = e.target.checked; };

// ═════════════════════════════════════════════════════════════════════
//  Splitters
// ═════════════════════════════════════════════════════════════════════
function dragSplit(el, onMove) {
  el.addEventListener('pointerdown', e => {
    el.setPointerCapture(e.pointerId); el.classList.add('drag');
    const mv = ev => { onMove(ev); needsDraw = true; };
    const up = () => { el.classList.remove('drag'); el.removeEventListener('pointermove', mv); el.removeEventListener('pointerup', up); };
    el.addEventListener('pointermove', mv); el.addEventListener('pointerup', up);
  });
}
dragSplit($('splitV'), e => { const w = clamp(window.innerWidth - e.clientX - 12, 280, window.innerWidth * .6); document.documentElement.style.setProperty('--sideW', w + 'px'); });
dragSplit($('splitH'), e => { const side = document.querySelector('.side').getBoundingClientRect();
  const h = clamp(e.clientY - side.top - 5, 160, side.height - 180); document.documentElement.style.setProperty('--viewH', h + 'px'); });
new ResizeObserver(() => { layout(); markTraces(); }).observe($('plotWrap'));

// ═════════════════════════════════════════════════════════════════════
//  Folder picker
// ═════════════════════════════════════════════════════════════════════
let pickPath = null;
async function browse(path) {
  $('pickErr').textContent = '';
  try {
    const d = await getJSON('/api/browse' + (path ? '?path=' + encodeURIComponent(path) : ''));
    pickPath = d.path; $('pathInput').value = d.path;
    const cr = $('crumbs'); cr.innerHTML = '';
    d.crumbs.forEach((c, i) => {
      if (i > 1) { const s = document.createElement('span'); s.textContent = '/'; cr.appendChild(s); }
      const b = document.createElement('button'); b.textContent = c.name; b.onclick = () => browse(c.path); cr.appendChild(b);
    });
    const ul = $('dirs'); ul.innerHTML = '';
    if (!d.dirs.length) { const li = document.createElement('li'); li.style.cssText = 'padding:10px;color:var(--muted);font-size:13px'; li.textContent = 'No subfolders.'; ul.appendChild(li); }
    d.dirs.forEach(name => {
      const li = document.createElement('li'), b = document.createElement('button');
      b.innerHTML = '<svg viewBox="0 0 20 20" fill="currentColor"><path d="M2 5.5A1.5 1.5 0 013.5 4h4l2 2h7A1.5 1.5 0 0118 7.5v7a1.5 1.5 0 01-1.5 1.5h-13A1.5 1.5 0 012 14.5z"/></svg><span></span>';
      b.lastChild.textContent = name; b.onclick = () => browse(d.path + (d.path.endsWith(d.sep) ? '' : d.sep) + name);
      li.appendChild(b); ul.appendChild(li);
    });
    ul.scrollTop = 0;
  } catch (err) { $('pickErr').textContent = err.message; }
}
function openPicker() { $('picker').hidden = false; browse(S.index?.root || null); }
function closePicker() { $('picker').hidden = true; }
$('openBtn').onclick = openPicker;
$('pickCancel').onclick = closePicker;
$('picker').addEventListener('click', e => { if (e.target === $('picker')) closePicker(); });
$('picker').addEventListener('keydown', e => { if (e.key === 'Escape') closePicker(); });
$('pathGo').onclick = () => browse($('pathInput').value.trim());
$('pathInput').addEventListener('keydown', e => { if (e.key === 'Enter') browse(e.target.value.trim()); });
$('pickOpen').onclick = async () => {
  const btn = $('pickOpen'); btn.disabled = true; btn.textContent = 'Searching for 3D files';
  try { const idx = await getJSON('/api/open?path=' + encodeURIComponent(pickPath)); closePicker(); S.playing = false; setPlayIcon(); await applyIndex(idx); }
  catch (err) { $('pickErr').textContent = err.message; }
  finally { btn.disabled = false; btn.textContent = 'Open this folder'; }
};

// ═════════════════════════════════════════════════════════════════════
//  Main loop
// ═════════════════════════════════════════════════════════════════════
let lastTs = 0;
function frame(ts) {
  requestAnimationFrame(frame);
  const dt = lastTs ? Math.min(.1, (ts - lastTs) / 1000) : 0; lastTs = ts;
  const tr = S.trial;
  if (S.playing && tr) {
    S.t += dt * S.speed;
    const dur = (tr.nFrames - 1) / tr.fps;
    if (S.t >= dur) S.t = 0;
    keepInWindow(false); needsDraw = true; updateTransport();
  }
  if (tracesDirty) renderTraces();
  if (needsDraw) drawPlot();
  if (R3) { ctrl.update(); updatePose(); R3.render(scene, cam); }
}

init3D();
layout();
try {
  const idx = await getJSON('/api/index');
  await applyIndex(idx);
  if (!idx.root && !idx.takes.length) openPicker();
} catch (err) {
  showMsg('The viewer server is not running', 'Start it again from Python with data_viewer().', null);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    _main()
