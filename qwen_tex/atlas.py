"""Control atlases for Qwen Image 2.1 and atlas <-> views conversion (pure numpy + the moderngl renderer).

Atlas layout (same as UltraTex): 2 rows x 3 columns of square tiles,
    row 1: front, left, right      row 2: back, top, bottom
View order everywhere else: 0 front, 1 left, 2 back, 3 right, 4 top, 5 bottom.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .ultratex_core.prep import render_conditions
from .ultratex_core.render import GLRenderer, GRID_ORDER, VIEW_NAMES, rig_for_view, view_matrix

CONTROL_MODES = ["depth", "grayscale", "normal"]

# key light in camera space (camera looks down -Z, +Y up): from the upper left, slightly in front
_LIGHT = np.array([-0.35, 0.45, 0.82])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)

LAYOUT_PROMPT = (
    "A 2x3 grid of six views of the same single object on a pure white background, rendered with flat, even, "
    "shadowless lighting (albedo texture). Top row, left to right: front view, left side view, right side view. "
    "Bottom row, left to right: back view, top view, bottom view. Every view shows the same object with exactly "
    "the same colors, materials and details as the reference image, consistent across all six views."
)


def _resize(arr: np.ndarray, size: int, resample=Image.Resampling.BILINEAR) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    return np.asarray(Image.fromarray(arr).resize((size, size), resample))


def view_masks(prep, size: int) -> np.ndarray:
    """(6, size, size) bool silhouettes of prep's rig."""
    return np.stack([_resize(prep.normal_rgba[v, ..., 3], size) > 127 for v in range(6)])


def render_control(prep, size: int, mode: str) -> np.ndarray:
    """(6, size, size, 3) uint8 control images in view order.

    depth: near = white, far = dark, background black (Depth Anything convention).
    grayscale: Lambert-shaded clay render on white.
    normal: the UltraTex world-normal colours on white.
    """
    if mode not in CONTROL_MODES:
        raise ValueError(f"unknown control mode {mode!r}")
    if mode == "normal":
        out = np.full((6, size, size, 3), 255, np.uint8)
        for v in range(6):
            tile = _resize(prep.normal_rgba[v], size)
            fg = tile[..., 3] > 127
            out[v][fg] = tile[..., :3][fg]
        return out

    out = np.zeros((6, size, size, 3), np.uint8)
    rig = None if np.allclose(prep.rig, np.eye(3)) else prep.rig
    with GLRenderer(prep.vertices, prep.normals, prep.faces) as renderer:
        for v in range(6):
            rig_v = rig_for_view(rig, v)
            _, nrm, dep = renderer.render_view(v, size, rig_v)
            fg = dep[..., 3] > 0
            if mode == "depth":
                d = dep[..., 0]
                img = np.zeros((size, size), np.float32)
                if fg.any():
                    lo, hi = np.percentile(d[fg], [0.5, 99.5])
                    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                    img = np.where(fg, 1.0 - 0.85 * t, 0.0)
            else:
                mv = view_matrix(v, rig_v)
                n = np.nan_to_num(nrm[..., :3]) @ mv[:3, :3].T
                n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
                facing = np.where(n[..., 2:3] < 0, -n, n)  # flipped / back faces still shade like front faces
                lambert = np.clip(facing @ _LIGHT, 0.0, 1.0)
                img = np.where(fg, 0.22 + 0.7 * lambert, 1.0)
            out[v] = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)[..., None]
    return out


def to_atlas(tiles: np.ndarray) -> np.ndarray:
    """(6, s, s, ...) in view order -> (2s, 3s, ...) atlas."""
    return np.concatenate([np.concatenate([tiles[v] for v in row], axis=1) for row in GRID_ORDER], axis=0)


def from_atlas(atlas: np.ndarray, tile: int | None = None) -> np.ndarray:
    """(H, W, C) atlas -> (6, tile, tile, C) in view order. The atlas may be any size (e.g. a decoded image
    whose sides were rounded): each cell is cut at H/2 x W/3 and resized to a square tile."""
    h, w = atlas.shape[:2]
    if tile is None:
        tile = int(round(min(h / 2, w / 3)))
    out = [None] * 6
    for r, row in enumerate(GRID_ORDER):
        for c, v in enumerate(row):
            y0, y1 = round(r * h / 2), round((r + 1) * h / 2)
            x0, x1 = round(c * w / 3), round((c + 1) * w / 3)
            cell = atlas[y0:y1, x0:x1]
            if cell.shape[0] != tile or cell.shape[1] != tile:
                f = cell.astype(np.float32)
                cell = np.stack([np.asarray(Image.fromarray(f[..., k]).resize((tile, tile), Image.Resampling.BICUBIC))
                                 for k in range(f.shape[-1])], axis=-1)
            out[v] = cell
    return np.stack(out)


def tile_size_for(target_pixels: int) -> int:
    """Largest multiple of 16 such that 6 tiles have about target_pixels pixels."""
    return max(256, int(np.sqrt(target_pixels / 6)) // 16 * 16)


def view_labels() -> str:
    return " | ".join(VIEW_NAMES[v] for row in GRID_ORDER for v in row)


__all__ = ["CONTROL_MODES", "LAYOUT_PROMPT", "render_control", "to_atlas", "from_atlas", "view_masks",
           "render_conditions", "tile_size_for", "view_labels"]
