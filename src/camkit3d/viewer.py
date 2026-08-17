"""GPU-accelerated 3D pose viewer for CamKit3D.

Renders reconstructed 3D pose sequences as an interactive skeleton animation,
with no matplotlib dependency.

Key features:

- Self-contained output. Writes a single standalone HTML file with an
  embedded Three.js (r160) scene and opens it in the default browser, so the
  result is shareable and needs no server or Python runtime to replay.
- Works everywhere. Behaves identically from terminal scripts and Jupyter
  notebooks, with orbit/zoom controls, selectable initial views, and light or
  dark backgrounds.
- Skeleton-driven. Reads its topology from the skeleton descriptor
  (MediaPipe Pose by default), so the same viewer draws any supported
  skeleton without code changes.

Usage:

    from camkit3d.viewer import viewer
    viewer(points_3d_aligned, fps=30, keypoint_size=70, line_width=8)

Author: Dr. Robert Seymour, OHBA, University of Oxford
License: GNU General Public License v3, 2026
"""

import json
import tempfile
import webbrowser
import numpy as np
from pathlib import Path

from camkit3d import skeletons as _skeletons
from camkit3d.skeletons import PoseDefinition


# ── Skeleton topology ───────────────────────────────────────────────────
# Loaded from the skeleton descriptor (MediaPipe Pose by default). Pass a
# different skeleton id / PoseDefinition to viewer() to change it.
_DEFAULT_SKELETON = _skeletons.load()


def _resolve_skeleton(skeleton):
    if skeleton is None:
        return _DEFAULT_SKELETON
    if isinstance(skeleton, PoseDefinition):
        return skeleton
    return _skeletons.load(skeleton)


def viewer(
    points_3d,
    fps=30,
    keypoint_size=70,
    line_width=8,
    initial_view="front",
    dark_mode=False,
    bg_color=None,
    show_axes=True,
    output_path=None,
    window_title="CamKit3D Pose Viewer",
    skeleton=None,
):
    """
    Launch a smooth, GPU-accelerated 3D pose viewer in the browser.

    Controls
    --------
    Orbit        : left-click drag
    Zoom         : scroll wheel / pinch
    Pan          : right-click drag / two-finger drag
    Play/Pause   : spacebar or button
    Scrub        : drag timeline
    Step frame   : left/right arrow keys
    Speed        : 0.1x - 3x slider
    Landmark size: live slider
    Bone width   : live slider
    Dark/Light   : toggle button
    Axes         : toggle button
    Screenshot   : button (downloads PNG)
    View presets : Front / Side / Top / Free buttons

    Parameters
    ----------
    points_3d : np.ndarray
        Shape (n_frames, n_keypoints, 3). NaN keypoints are hidden.
    fps : int
        Playback frame rate.
    keypoint_size : float
        Initial landmark sphere scale (range 1-200 in the UI).
    line_width : float
        Initial bone thickness scale (range 0.5-20 in the UI).
    initial_view : str
        'front', 'right', 'back', 'top', or 'diagonal'.
    dark_mode : bool
        Start dark (True) or light (False).
    bg_color : str, optional
        Light-mode background colour as a hex string (e.g. "#ffffff" or
        "e0f0ff"). Overrides the default light background. Ignored in dark
        mode. If None, uses the default light background.
    show_axes : bool
        Show labelled XYZ axes on startup.
    output_path : str or Path, optional
        Save the HTML here. If None, uses a temp file.
    window_title : str
        Browser tab title.
    skeleton : str or PoseDefinition, optional
        Skeleton topology (edges + per-edge colours). Defaults to MediaPipe
        Pose. Pass a skeleton id (e.g. "mediapipe_pose") or a loaded
        PoseDefinition to render a different skeleton.

    Returns
    -------
    str : path to the generated HTML file.
    """
    data = np.asarray(points_3d, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected (n_frames, n_keypoints, 3), got {data.shape}")

    n_frames, n_kp, _ = data.shape

    # Subsample very long recordings
    max_embed = 36_000
    step = 1
    effective_fps = fps
    if n_frames > max_embed:
        step = int(np.ceil(n_frames / max_embed))
        data = data[::step]
        n_frames = data.shape[0]
        effective_fps = max(1, fps // step)

    # Scene normalisation
    flat = data.reshape(-1, 3)
    valid = flat[~np.isnan(flat).any(axis=1)]
    centre = np.nanmedian(valid, axis=0).tolist()
    extent = float(np.nanpercentile(np.linalg.norm(valid - centre, axis=1), 98))
    if extent < 1e-6:
        extent = 1.0

    # Serialise frames
    def pack(frame):
        out = []
        for kp in frame:
            if np.any(np.isnan(kp)):
                out.append(None)
            else:
                out.append([round(float(kp[0]), 2),
                            round(float(kp[1]), 2),
                            round(float(kp[2]), 2)])
        return out

    frames_json = json.dumps(
        [pack(data[i]) for i in range(n_frames)], separators=(",", ":")
    )
    skel = _resolve_skeleton(skeleton)
    conns_json = json.dumps([list(e) for e in skel.edges], separators=(",", ":"))
    colors_json = json.dumps(skel.edge_colors, separators=(",", ":"))

    # Camera presets
    views = {
        "front":    {"t": 0,      "p": 1.5708, "d": 2.8},
        "right":    {"t": 1.5708, "p": 1.5708, "d": 2.8},
        "back":     {"t": 3.1416, "p": 1.5708, "d": 2.8},
        "top":      {"t": 0,      "p": 0.15,   "d": 3.2},
        "diagonal": {"t": 0.7854, "p": 1.1,    "d": 3.0},
    }
    iv = views.get(initial_view, views["front"])

    # Light-mode background colour override
    if bg_color is None:
        light_bg_js = "0xf2f2f5"
    else:
        hexstr = str(bg_color).lstrip("#").strip()
        if len(hexstr) != 6 or any(c not in "0123456789abcdefABCDEF" for c in hexstr):
            raise ValueError(f"bg_color must be a 6-digit hex string, got {bg_color!r}")
        light_bg_js = "0x" + hexstr

    # Build HTML
    replacements = {
        "__TITLE__":      window_title,
        "__FRAMES__":     frames_json,
        "__CONNS__":      conns_json,
        "__COLORS__":     colors_json,
        "__CENTRE__":     json.dumps(centre),
        "__EXTENT__":     str(extent),
        "__N_FRAMES__":   str(n_frames),
        "__N_KP__":       str(n_kp),
        "__FPS__":        str(effective_fps),
        "__KP_SIZE__":    str(keypoint_size),
        "__LINE_WIDTH__": str(line_width),
        "__INIT_THETA__": str(iv["t"]),
        "__INIT_PHI__":   str(iv["p"]),
        "__INIT_DIST__":  str(iv["d"]),
        "__DARK_MODE__":  "true" if dark_mode else "false",
        "__LIGHT_BG__":   light_bg_js,
        "__SHOW_AXES__":  "true" if show_axes else "false",
    }

    html = _HTML_TEMPLATE
    for key, val in replacements.items():
        html = html.replace(key, val)

    # Write and open
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        html_path = str(out)
    else:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", prefix="pose_viewer_", delete=False,
            mode="w", encoding="utf-8",
        )
        tmp.write(html)
        tmp.close()
        html_path = tmp.name

    print(f"Viewer saved: {html_path}")
    webbrowser.open(Path(html_path).as_uri())

    return html_path


# ════════════════════════════════════════════════════════════════════════════
#  HTML template: sidebar layout, Three.js r160 + OrbitControls via CDN
# ════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

:root {
  --bg:       #f2f2f5;
  --surface:  #ffffff;
  --surface2: #e9e9ee;
  --border:   #d0d0d8;
  --text:     #1a1a2e;
  --dim:      #6e6e82;
  --accent:   #2ba89e;
  --accent-g: rgba(43,168,158,0.15);
  --radius:   6px;
  --font:     'Outfit', sans-serif;
  --mono:     'JetBrains Mono', monospace;
  --scene-bg: #f2f2f5;
  --kp-color: 0x2C3E50;
  --ax-color: #666;
}

.dark {
  --bg:       #0c0c10;
  --surface:  #16161c;
  --surface2: #1e1e26;
  --border:   #2a2a36;
  --text:     #d4d4dc;
  --dim:      #6e6e82;
  --accent:   #4ECDC4;
  --accent-g: rgba(78,205,196,0.2);
  --scene-bg: #0c0c10;
  --kp-color: 0xe0e0e6;
  --ax-color: #999;
}

body {
  background:var(--bg);
  color:var(--text);
  font-family:var(--font);
  overflow:hidden;
  height:100vh;
  display:flex;
  flex-direction:row;
}

/* ── Viewport (canvas + timeline at bottom) ────────── */
.viewport {
  flex:1;
  display:flex;
  flex-direction:column;
  min-width:0;
}
#c { display:block; flex:1; cursor:grab; }
#c:active { cursor:grabbing; }

.timeline-bar {
  background:var(--surface);
  border-top:1px solid var(--border);
  padding:8px 14px;
  display:flex;
  align-items:center;
  gap:10px;
}
.timeline-bar input[type=range] { flex:1; }

/* ── Sidebar ───────────────────────────────────────── */
.sidebar {
  width: 240px;
  min-width: 240px;
  background:var(--surface);
  border-left:1px solid var(--border);
  display:flex;
  flex-direction:column;
  overflow-y:auto;
  user-select:none;
}

.sb-section {
  padding:14px 16px;
  border-bottom:1px solid var(--border);
}
.sb-section:last-child { border-bottom:none; }

.sb-title {
  font-size:10px;
  font-weight:600;
  text-transform:uppercase;
  letter-spacing:0.08em;
  color:var(--dim);
  margin-bottom:10px;
}

/* ── Shared controls ───────────────────────────────── */
.lbl {
  font-family:var(--mono);
  font-size:11px;
  color:var(--dim);
  min-width:60px;
  text-align:center;
}

input[type=range] {
  -webkit-appearance:none; appearance:none;
  width:100%; height:5px; border-radius:3px;
  background:var(--surface2);
  outline:none; cursor:pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance:none;
  width:14px;height:14px;border-radius:50%;
  background:var(--accent);
  box-shadow:0 0 6px var(--accent-g);
  cursor:pointer; transition:transform .1s;
}
input[type=range]::-webkit-slider-thumb:hover { transform:scale(1.3); }
input[type=range]::-moz-range-thumb {
  width:14px;height:14px;border-radius:50%;
  background:var(--accent);border:none;cursor:pointer;
}

.slider-row {
  display:flex;
  align-items:center;
  gap:8px;
  margin-bottom:8px;
}
.slider-row label {
  font-size:11px;
  font-weight:500;
  color:var(--text);
  min-width:60px;
}
.slider-row .val {
  font-family:var(--mono);
  font-size:11px;
  color:var(--dim);
  min-width:32px;
  text-align:right;
}
.slider-row input[type=range] { flex:1; }

/* Buttons */
.b {
  font-family:var(--font); font-size:11px; font-weight:500;
  padding:6px 0; border-radius:var(--radius);
  border:1px solid var(--border); background:var(--surface2);
  color:var(--text); cursor:pointer; transition:all .12s;
  text-align:center; width:100%;
}
.b:hover { background:var(--border); }
.b.on { background:var(--accent); color:#fff; border-color:var(--accent); }

.btn-row {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:6px;
  margin-bottom:6px;
}
.btn-row.three { grid-template-columns:1fr 1fr 1fr; }
.btn-row.four { grid-template-columns:1fr 1fr 1fr 1fr; }

/* Play button */
.bp {
  width:36px;height:36px;padding:0;
  display:flex;align-items:center;justify-content:center;
  font-size:16px; border-radius:50%;
  background:var(--accent); color:#fff; border:none; cursor:pointer;
  flex-shrink:0;
}
.bp:hover { box-shadow:0 0 12px var(--accent-g); }
</style>
</head>
<body>

<!-- Left: viewport + timeline -->
<div class="viewport">
  <canvas id="c"></canvas>
  <div class="timeline-bar">
    <button class="bp" id="pp" title="Space">&#9654;</button>
    <span class="lbl" id="tL">0:00.00</span>
    <input type="range" id="tR" min="0" max="__N_FRAMES__" value="0" step="1">
    <span class="lbl" id="fL">0 / __N_FRAMES__</span>
  </div>
</div>

<!-- Right: sidebar controls -->
<div class="sidebar">

  <!-- Playback -->
  <div class="sb-section">
    <div class="sb-title">Playback</div>
    <div class="slider-row">
      <label>Speed</label>
      <input type="range" id="sR" min="0" max="5" step="1" value="3">
      <span class="val" id="sL">1x</span>
    </div>
  </div>

  <!-- Appearance -->
  <div class="sb-section">
    <div class="sb-title">Appearance</div>
    <div class="slider-row">
      <label>Landmarks</label>
      <input type="range" id="kvR" min="1" max="200" value="__KP_SIZE__" step="1">
      <span class="val" id="kvV">__KP_SIZE__</span>
    </div>
    <div class="slider-row">
      <label>Bones</label>
      <input type="range" id="lwR" min="1" max="20" value="__LINE_WIDTH__" step="0.5">
      <span class="val" id="lwV">__LINE_WIDTH__</span>
    </div>
    <div class="btn-row" style="margin-top:4px;">
      <button class="b" id="bgLight">Light</button>
      <button class="b" id="bgDark">Dark</button>
    </div>
  </div>

  <!-- View -->
  <div class="sb-section">
    <div class="sb-title">Camera</div>
    <div class="btn-row four">
      <button class="b" id="v1">Front</button>
      <button class="b" id="v2">Side</button>
      <button class="b" id="v3">Top</button>
      <button class="b" id="v4">Free</button>
    </div>
  </div>

  <!-- Tools -->
  <div class="sb-section">
    <div class="sb-title">Tools</div>
    <div class="btn-row">
      <button class="b" id="axBtn">Axes: On</button>
      <button class="b" id="ssBtn">Screenshot</button>
    </div>
  </div>

  <!-- Info -->
  <div class="sb-section" style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;">
    <div style="font-family:var(--mono);font-size:10px;color:var(--dim);line-height:1.6;">
      <div>Orbit: left drag</div>
      <div>Pan: right drag</div>
      <div>Zoom: scroll</div>
      <div>Step: ← → keys</div>
      <div>Play: spacebar</div>
    </div>
  </div>

</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── DATA ────────────────────────────────────────────────────────────
const FDATA = __FRAMES__;
const CO = __CONNS__;
const CC = __COLORS__;
const CTR = __CENTRE__;
const EXT = __EXTENT__;
const NF = __N_FRAMES__;
const NK = __N_KP__;
const BASE_FPS = __FPS__;
const SPEEDS = [0.1, 0.25, 0.5, 1, 1.5, 2];
let spdIdx = 3;
let pFps = BASE_FPS;
const S = 1.0 / EXT;

// ── RENDERER ────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const R = new THREE.WebGLRenderer({ canvas, antialias:true, preserveDrawingBuffer:true, alpha:true });
R.setPixelRatio(Math.min(devicePixelRatio, 2));

let isDark = __DARK_MODE__;
const LIGHT_BG = __LIGHT_BG__;
function sceneColor() { return isDark ? 0x0c0c10 : LIGHT_BG; }
function kpColor()    { return isDark ? 0xe0e0e6 : 0x2C3E50; }
R.setClearColor(sceneColor(), 1);

const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(50, 1, 0.01, 500);

// ── CONTROLS ────────────────────────────────────────────────────────
const ctrl = new OrbitControls(cam, canvas);
ctrl.enableDamping = true;
ctrl.dampingFactor = 0.12;
ctrl.rotateSpeed = 0.7;
ctrl.panSpeed = 0.5;
ctrl.zoomSpeed = 1.2;
ctrl.target.set(0, 0, 0);

function setView(t, p, d) {
  const x = d * Math.sin(p) * Math.sin(t);
  const y = d * Math.cos(p);
  const z = d * Math.sin(p) * Math.cos(t);
  cam.position.set(x, y, z);
  ctrl.target.set(0, 0, 0);
  ctrl.update();
}
setView(__INIT_THETA__, __INIT_PHI__, __INIT_DIST__);

// ── LIGHTING ────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dl1 = new THREE.DirectionalLight(0xffffff, 0.6);
dl1.position.set(3, 5, 4); scene.add(dl1);
const dl2 = new THREE.DirectionalLight(0x8899cc, 0.25);
dl2.position.set(-3, -2, -3); scene.add(dl2);

// ── 3D AXES (proper labelled X/Y/Z) ────────────────────────────────
const axGroup = new THREE.Group();
const axLen = 0.5;
const axColors = { x: 0xff4444, y: 0x44cc44, z: 0x4488ff };

function makeAxisLine(dir, color) {
  const pts = [new THREE.Vector3(0,0,0), dir.clone().multiplyScalar(axLen)];
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
  return new THREE.Line(geo, mat);
}

axGroup.add(makeAxisLine(new THREE.Vector3(1,0,0), axColors.x));
axGroup.add(makeAxisLine(new THREE.Vector3(0,1,0), axColors.y));
axGroup.add(makeAxisLine(new THREE.Vector3(0,0,1), axColors.z));

// Axis cones (arrowheads)
function makeCone(dir, color) {
  const geo = new THREE.ConeGeometry(0.02, 0.06, 8);
  const mat = new THREE.MeshBasicMaterial({ color });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(dir.clone().multiplyScalar(axLen));
  // Orient cone along direction
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(new THREE.Vector3(0,1,0), dir.clone().normalize());
  mesh.quaternion.copy(q);
  return mesh;
}
axGroup.add(makeCone(new THREE.Vector3(1,0,0), axColors.x));
axGroup.add(makeCone(new THREE.Vector3(0,1,0), axColors.y));
axGroup.add(makeCone(new THREE.Vector3(0,0,1), axColors.z));

// Axis labels (sprites)
function makeLabel(text, pos, color) {
  const cnv = document.createElement('canvas');
  cnv.width = 64; cnv.height = 32;
  const ctx = cnv.getContext('2d');
  ctx.font = 'bold 24px Outfit, sans-serif';
  ctx.fillStyle = '#' + color.toString(16).padStart(6, '0');
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 16);
  const tex = new THREE.CanvasTexture(cnv);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.position.copy(pos);
  sprite.scale.set(0.12, 0.06, 1);
  return sprite;
}
axGroup.add(makeLabel('X', new THREE.Vector3(axLen + 0.06, 0, 0), axColors.x));
axGroup.add(makeLabel('Y', new THREE.Vector3(0, axLen + 0.06, 0), axColors.y));
axGroup.add(makeLabel('Z', new THREE.Vector3(0, 0, axLen + 0.06), axColors.z));

axGroup.visible = __SHOW_AXES__;
scene.add(axGroup);

// ── SKELETON MESHES ─────────────────────────────────────────────────
let kpScale = __KP_SIZE__ / 100 * 0.015;
let boneScale = __LINE_WIDTH__ / 10 * 0.006;

const kpGeo = new THREE.SphereGeometry(1, 16, 12);
const kpMat = new THREE.MeshStandardMaterial({
  color: kpColor(), roughness: 0.35, metalness: 0.05
});
const kpIM = new THREE.InstancedMesh(kpGeo, kpMat, NK);
kpIM.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(kpIM);

const boneMeshes = [];
const upVec = new THREE.Vector3(0, 1, 0);
for (let i = 0; i < CO.length; i++) {
  const m = new THREE.MeshStandardMaterial({
    color: new THREE.Color(CC[i]), roughness: 0.4, metalness: 0.05
  });
  const g = new THREE.CylinderGeometry(1, 1, 1, 8, 1);
  const mesh = new THREE.Mesh(g, m);
  mesh.visible = false;
  scene.add(mesh);
  boneMeshes.push(mesh);
}

// ── FRAME RENDERING ─────────────────────────────────────────────────
const _d = new THREE.Object3D();
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _q = new THREE.Quaternion();

function toScene(pt) {
  return [(pt[0]-CTR[0])*S, (pt[2]-CTR[2])*S, (pt[1]-CTR[1])*S];
}

function renderFrame(fi) {
  const fr = FDATA[fi]; if (!fr) return;

  for (let k = 0; k < NK; k++) {
    const p = fr[k];
    if (p === null) {
      _d.scale.set(0,0,0);
    } else {
      const s = toScene(p);
      _d.position.set(s[0], s[1], s[2]);
      _d.scale.setScalar(kpScale);
    }
    _d.updateMatrix();
    kpIM.setMatrixAt(k, _d.matrix);
  }
  kpIM.instanceMatrix.needsUpdate = true;

  for (let i = 0; i < CO.length; i++) {
    const sp = fr[CO[i][0]], ep = fr[CO[i][1]];
    const mesh = boneMeshes[i];
    if (sp === null || ep === null) { mesh.visible = false; continue; }
    const sa = toScene(sp), sb = toScene(ep);
    _a.set(sa[0],sa[1],sa[2]);
    _b.set(sb[0],sb[1],sb[2]);
    _dir.subVectors(_b, _a);
    const len = _dir.length();
    if (len < 1e-6) { mesh.visible = false; continue; }
    mesh.visible = true;

    mesh.position.lerpVectors(_a, _b, 0.5);
    _dir.normalize();
    _q.setFromUnitVectors(upVec, _dir);
    mesh.quaternion.copy(_q);
    mesh.scale.set(boneScale, len, boneScale);
  }
}

// ── PLAYBACK ────────────────────────────────────────────────────────
let cf = 0, playing = false, lastT = 0, acc = 0;
const tR = document.getElementById('tR');
const tL = document.getElementById('tL');
const fL = document.getElementById('fL');
const pp = document.getElementById('pp');
const totalS = NF / BASE_FPS;

function fmt(s) {
  const m = Math.floor(s/60);
  return m + ':' + (s-m*60).toFixed(2).padStart(5,'0');
}
function upLabels() {
  const t = cf / BASE_FPS;
  tL.textContent = fmt(t) + ' / ' + fmt(totalS);
  fL.textContent = cf + ' / ' + (NF-1);
  tR.value = cf;
}

// ── RESIZE ──────────────────────────────────────────────────────────
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const pr = R.getPixelRatio();
  if (canvas.width !== w*pr || canvas.height !== h*pr) {
    R.setSize(w, h, false);
    cam.aspect = w/h;
    cam.updateProjectionMatrix();
  }
}

// ── LOOP ────────────────────────────────────────────────────────────
function loop(ts) {
  requestAnimationFrame(loop);
  resize();
  ctrl.update();

  if (playing) {
    const dt = lastT ? (ts - lastT) : 0;
    lastT = ts;
    acc += dt;
    const iv = 1000 / pFps;
    while (acc >= iv) { acc -= iv; cf++; if (cf >= NF) cf = 0; }
    upLabels();
  } else {
    lastT = 0; acc = 0;
  }

  renderFrame(cf);
  R.render(scene, cam);
}
requestAnimationFrame(loop);
renderFrame(0);
upLabels();

// ── UI WIRING ───────────────────────────────────────────────────────

// Play/Pause
pp.onclick = () => {
  playing = !playing;
  pp.innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
};

// Timeline
tR.addEventListener('input', () => { cf = +tR.value; upLabels(); });

// Keyboard
document.addEventListener('keydown', e => {
  if (e.code === 'Space')           { e.preventDefault(); pp.click(); }
  else if (e.code === 'ArrowLeft')  { e.preventDefault(); cf = Math.max(0, cf-1); upLabels(); }
  else if (e.code === 'ArrowRight') { e.preventDefault(); cf = Math.min(NF-1, cf+1); upLabels(); }
});

// Speed
const sR = document.getElementById('sR');
const sL = document.getElementById('sL');
sR.oninput = () => {
  spdIdx = +sR.value;
  const m = SPEEDS[spdIdx];
  pFps = BASE_FPS * m;
  sL.textContent = m + 'x';
};

// Views
document.getElementById('v1').onclick = () => setView(0, 1.5708, 2.8);
document.getElementById('v2').onclick = () => setView(1.5708, 1.5708, 2.8);
document.getElementById('v3').onclick = () => setView(0, 0.15, 3.2);
document.getElementById('v4').onclick = () => setView(0.7854, 1.1, 3.0);

// Keypoint size
const kvR = document.getElementById('kvR');
const kvV = document.getElementById('kvV');
kvR.oninput = () => {
  kvV.textContent = kvR.value;
  kpScale = +kvR.value / 100 * 0.015;
};

// Bone width
const lwR = document.getElementById('lwR');
const lwV = document.getElementById('lwV');
lwR.oninput = () => {
  lwV.textContent = lwR.value;
  boneScale = +lwR.value / 10 * 0.006;
};

// Theme toggle
function applyTheme() {
  document.body.classList.toggle('dark', isDark);
  R.setClearColor(sceneColor(), 1);
  kpMat.color.set(kpColor());
}
document.getElementById('bgDark').onclick  = () => { isDark = true;  applyTheme(); };
document.getElementById('bgLight').onclick = () => { isDark = false; applyTheme(); };
applyTheme();

// Axes toggle
const axBtn = document.getElementById('axBtn');
axBtn.textContent = axGroup.visible ? 'Axes: On' : 'Axes: Off';
axBtn.classList.toggle('on', axGroup.visible);
axBtn.onclick = () => {
  axGroup.visible = !axGroup.visible;
  axBtn.textContent = axGroup.visible ? 'Axes: On' : 'Axes: Off';
  axBtn.classList.toggle('on', axGroup.visible);
};

// Screenshot
document.getElementById('ssBtn').onclick = () => {
  // Render with a fully transparent background so the saved PNG has no
  // scene backdrop (only keypoints/bones are plotted).
  R.setClearColor(0x000000, 0);
  R.render(scene, cam);
  canvas.toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'pose_frame_' + String(cf).padStart(5,'0') + '.png';
    a.click();
    URL.revokeObjectURL(a.href);
    // Restore the on-screen background.
    R.setClearColor(sceneColor(), 1);
    R.render(scene, cam);
  });
};

</script>
</body>
</html>"""