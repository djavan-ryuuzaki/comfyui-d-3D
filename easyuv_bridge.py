"""D3D - EasyUV Unwrap: calls the EasyUV unwrapper (https://github.com/lburdier/EasyUV) through its C API.

EasyUV is a separate program (GPL-3.0-or-later) that the user installs; this file contains no EasyUV code.
It loads ``easyuv.dll`` (Windows) / ``libeasyuv.so`` (Linux) at run time, passes our body-part cut lines as
forced seams and lets EasyUV make every island a disk, flatten it (LSCM + ARAP), re-split what stretches and
pack. With ``learn`` on it uses EasyUV's self-learning (several settings tried in parallel, the best kept and
remembered in EasyUV's own learning file, shared with the EasyUV app and Blender addon).

If the library is not found the node falls back to "D3D - Character UV".
"""
from __future__ import annotations

import ctypes
import glob
import logging
import os

import numpy as np
import torch

try:
    from .character_uv import (body_part_labels, body_part_seams, character_uv, PART_NAMES)
    from .lowpoly_tools import _batch_items, _progress_text, repack_uvs, UP_AXES
except ImportError:  # loaded outside the package (tests)
    import importlib.util as _ilu
    import sys as _sys

    _here = os.path.dirname(os.path.abspath(__file__))

    def _load(name):
        if name in _sys.modules:
            return _sys.modules[name]
        spec = _ilu.spec_from_file_location(name, os.path.join(_here, name + ".py"))
        mod = _ilu.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _lp = _load("lowpoly_tools")
    _cu = _load("character_uv")
    body_part_labels, body_part_seams, character_uv, PART_NAMES = (
        _cu.body_part_labels, _cu.body_part_seams, _cu.character_uv, _cu.PART_NAMES)
    _batch_items, _progress_text, repack_uvs, UP_AXES = (
        _lp._batch_items, _lp._progress_text, _lp.repack_uvs, _lp.UP_AXES)

API_VERSION = 3
SHAPES = ["auto", "hard-surface", "cylinder", "organic"]


class _Params(ctypes.Structure):
    _fields_ = [("angle_deg", ctypes.c_float), ("max_distortion", ctypes.c_float),
                ("relax_iterations", ctypes.c_uint32), ("margin", ctypes.c_float),
                ("density_mode", ctypes.c_uint32), ("texture_size", ctypes.c_uint32),
                ("target_density", ctypes.c_float), ("quality", ctypes.c_uint32),
                ("rotate_islands", ctypes.c_uint32), ("priority", ctypes.c_float), ("shape", ctypes.c_uint32)]


class _Stats(ctypes.Structure):
    _fields_ = [("charts", ctypes.c_uint32), ("flipped", ctypes.c_uint32), ("seams", ctypes.c_uint32),
                ("overflow", ctypes.c_uint32), ("mean_distortion", ctypes.c_float),
                ("max_distortion", ctypes.c_float), ("density_px_per_unit", ctypes.c_float),
                ("coverage", ctypes.c_float), ("time_ms", ctypes.c_float)]


class _LearnInfo(ctypes.Structure):
    _fields_ = [("angle_deg", ctypes.c_float), ("max_distortion", ctypes.c_float),
                ("relax_iterations", ctypes.c_uint32), ("trials", ctypes.c_uint32), ("score", ctypes.c_float),
                ("default_score", ctypes.c_float), ("confidence", ctypes.c_float), ("experiences", ctypes.c_uint32)]


_LIB = {}


def _candidates(user_path):
    names = ["easyuv.dll"] if os.name == "nt" else ["libeasyuv.so", "easyuv.so", "libeasyuv.dylib"]
    out = []
    for p in (user_path, os.environ.get("EASYUV_DLL", "")):
        if p:
            p = os.path.expandvars(os.path.expanduser(p))
            out += [os.path.join(p, n) for n in names] if os.path.isdir(p) else [p]
    here = os.path.dirname(os.path.abspath(__file__))
    out += [os.path.join(here, "bin", n) for n in names]
    appdata = os.environ.get("APPDATA", "")
    if appdata:  # EasyUV Blender extension
        for n in names:
            out += sorted(glob.glob(os.path.join(appdata, "Blender Foundation", "Blender", "*", "extensions", "*",
                                                 "easyuv", n)), reverse=True)
    return out


def load_easyuv(user_path=""):
    """Returns (library or None, path or reason)."""
    tried = []
    for path in _candidates(user_path):
        if not os.path.isfile(path):
            tried.append(path)
            continue
        if path in _LIB:
            return _LIB[path], path
        try:
            lib = ctypes.CDLL(path)
        except OSError as e:
            return None, f"could not load {path}: {e}"
        P, f32, u32 = ctypes.POINTER, ctypes.c_float, ctypes.c_uint32
        lib.euv_api_version.restype = u32
        if lib.euv_api_version() != API_VERSION:
            return None, f"{path}: EasyUV API {lib.euv_api_version()}, this node needs {API_VERSION}"
        common = [P(f32), u32, P(u32), u32, P(u32), u32, P(u32), u32, P(_Params)]
        outs = [P(f32), P(u32), P(u32), u32, P(u32), P(_Stats)]
        lib.euv_unwrap.restype = ctypes.c_int32
        lib.euv_unwrap.argtypes = common + outs
        lib.euv_unwrap_learned.restype = ctypes.c_int32
        lib.euv_unwrap_learned.argtypes = common + [ctypes.c_char_p, u32] + outs + [P(_LearnInfo)]
        _LIB[path] = lib
        return lib, path
    return None, "easyuv library not found (looked in: " + "; ".join(tried[:6]) + ")"


def _ptr(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def easyuv_unwrap(lib, v, f, seams, params, learn=True, trials=6):
    """Run EasyUV on a triangle mesh. Returns (corner uv [nf,3,2] v-up, face chart [nf], stats dict)."""
    pos = np.ascontiguousarray(v, np.float32).reshape(-1)
    nf = len(f)
    sizes = np.full(nf, 3, np.uint32)
    corners = np.ascontiguousarray(f, np.uint32).reshape(-1)
    sm = np.ascontiguousarray(seams, np.uint32).reshape(-1)
    out_uv = np.zeros(len(corners) * 2, np.float32)
    out_chart = np.zeros(nf, np.uint32)
    n_out = ctypes.c_uint32(0)
    st = _Stats()
    args = [_ptr(pos, ctypes.c_float), len(pos) // 3, _ptr(sizes, ctypes.c_uint32), nf,
            _ptr(corners, ctypes.c_uint32), len(corners),
            _ptr(sm, ctypes.c_uint32) if len(sm) else None, len(sm) // 2, ctypes.byref(params)]
    outs = [_ptr(out_uv, ctypes.c_float), _ptr(out_chart, ctypes.c_uint32), None, 0, ctypes.byref(n_out),
            ctypes.byref(st)]
    info = {}
    if learn:
        li = _LearnInfo()
        r = lib.euv_unwrap_learned(*args, None, int(trials), *outs, ctypes.byref(li))
        info = {"learned_angle": round(li.angle_deg, 1), "learned_distortion": round(li.max_distortion, 3),
                "learned_relax": li.relax_iterations, "trials": li.trials, "score": round(li.score, 3),
                "default_score": round(li.default_score, 3), "confidence": round(li.confidence, 2),
                "experiences": li.experiences}
    else:
        r = lib.euv_unwrap(*args, *outs)
    if r != 0:
        raise RuntimeError(f"EasyUV failed (code {r})")
    stats = {"charts": st.charts, "flipped": st.flipped, "seams": st.seams, "overflow": st.overflow,
             "mean_distortion": round(st.mean_distortion, 4), "max_distortion": round(st.max_distortion, 3),
             "coverage": round(st.coverage, 3), "time_ms": round(st.time_ms, 1)}
    stats.update(info)
    return out_uv.reshape(nf, 3, 2), out_chart, stats


def split_by_corner_uv(v, f, corner_uv, face_chart):
    """One vertex per (vertex, chart, uv). Returns (vertices, faces, uv)."""
    nf = len(f)
    q = np.round(corner_uv.reshape(-1, 2).astype(np.float64) * 2 ** 22).astype(np.int64)
    key = np.stack([f.reshape(-1).astype(np.int64), np.repeat(face_chart.astype(np.int64), 3), q[:, 0], q[:, 1]], 1)
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    sv = v[uniq[:, 0]]
    suv = corner_uv.reshape(-1, 2)[first]
    return sv, inv.reshape(nf, 3), suv


class D3DEasyUVUnwrap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "body_part_seams": ("BOOLEAN", {"default": True, "tooltip": "Force our body-part cut lines (head front/back, torso, arms, hands, legs) as seams; EasyUV adds the rest."}),
                "learn": ("BOOLEAN", {"default": False, "tooltip": "EasyUV self-learning: try several settings in parallel, keep the best and remember it (EasyUV's learning file)."}),
                "trials": ("INT", {"default": 6, "min": 2, "max": 64}),
                "priority": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = fewer seams, 1 = less distortion (learning score)."}),
                "seam_angle": ("FLOAT", {"default": 179.0, "min": 10.0, "max": 180.0, "step": 1.0, "tooltip": "Edges sharper than this become seams. 179 = no crease seams, only ours and the ones EasyUV needs (66 is EasyUV's default for hard-surface props; on a decimated character it cuts every fold)."}),
                "max_distortion": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01}),
                "relax_iterations": ("INT", {"default": 4, "min": 0, "max": 50, "tooltip": "ARAP iterations after LSCM."}),
                "shape": (SHAPES, {"default": "organic"}),
                "quality": ("INT", {"default": 2, "min": 0, "max": 3}),
                "resolution": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "padding": ("INT", {"default": 8, "min": 0, "max": 64}),
                "head_texel_scale": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Repack with more texels on the head (1.0 keeps EasyUV's own packing)."}),
                "hands_texel_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 4.0, "step": 0.05}),
                "head_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01}),
                "hands_size": ("FLOAT", {"default": 0.1, "min": 0.02, "max": 0.3, "step": 0.01}),
                "front_axis": (["+Z", "-Z"], {"default": "+Z"}),
                "up_axis": (UP_AXES, {"default": "auto"}),
                "library_path": ("STRING", {"default": "", "tooltip": "easyuv.dll (or its folder). Empty: EASYUV_DLL, comfyui-d-3D/bin, then the EasyUV Blender extension."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "unwrap"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("UV unwrap with EasyUV (installed separately), with our body-part cut lines as forced seams and "
                   "EasyUV's self-learning. Falls back to D3D - Character UV when EasyUV is not installed.")

    def unwrap(self, mesh, body_part_seams, learn, trials, priority, seam_angle, max_distortion, relax_iterations,
               shape, quality, resolution, padding, head_texel_scale, hands_texel_scale, head_size, hands_size,
               front_axis, up_axis, library_path, unique_id=None):
        from comfy_api.latest import Types

        n, item = _batch_items(mesh)
        if n != 1:
            logging.warning("[D3DEasyUVUnwrap] batch of meshes: only the first one is unwrapped")
        v, f = item(0)[:2]
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        lib, where = load_easyuv(library_path)
        if lib is None:
            logging.warning(f"[D3DEasyUVUnwrap] {where}; using D3D - Character UV instead")
            sv, sf, uv, info = character_uv(v, f, resolution, padding, head_texel_scale, hands_texel_scale,
                                            head_size, hands_size, up_axis, front_axis)
            text = f"EasyUV not found, used Character UV: {info['islands']} islands, {info['flipped']} flipped"
        else:
            wv, wf, adj, body, lab, full = body_part_labels(v, f, up_axis, head_size, hands_size, front_axis)
            seams = body_part_seams(wf, adj, full) if body_part_seams else np.zeros((0, 2), np.int64)
            p = _Params(angle_deg=seam_angle, max_distortion=max_distortion, relax_iterations=relax_iterations,
                        margin=float(padding) / float(resolution), density_mode=0, texture_size=resolution,
                        target_density=0.0, quality=quality, rotate_islands=1, priority=priority,
                        shape=SHAPES.index(shape))
            corner_uv, chart, st = easyuv_unwrap(lib, wv, wf, seams, p, learn, trials)
            sv, sf, uv = split_by_corner_uv(wv, wf, corner_uv, chart)
            repacked = head_texel_scale > 1.0 or hands_texel_scale > 1.0
            if repacked:
                density = body.reg.density(torch.from_numpy(sv).float(), float(head_texel_scale),
                                           float(hands_texel_scale)).numpy()
                uv, _, _ = repack_uvs(sv, sf, uv.astype(np.float32), density, resolution, padding)
            text = (f"EasyUV: {st['charts']} islands, {st['flipped']} flipped, mean distortion "
                    f"{st['mean_distortion']}, {st['time_ms']:.0f} ms | {len(seams)} body-part seam edges")
            if learn:
                text += (f" | learned: angle {st['learned_angle']}, distortion {st['learned_distortion']}, "
                         f"ARAP {st['learned_relax']}, score {st['score']} vs default {st['default_score']}, "
                         f"{st['experiences']} experiences")
            if repacked:
                text += " | repacked with head/hands texel boost"
            sv, sf, uv = sv.astype(np.float32), sf.astype(np.int64), np.asarray(uv, np.float32)
        uv_gltf = uv.copy()
        uv_gltf[:, 1] = 1.0 - uv_gltf[:, 1]
        out = Types.MESH(vertices=torch.from_numpy(np.ascontiguousarray(sv, np.float32))[None],
                         faces=torch.from_numpy(np.ascontiguousarray(sf, np.int64))[None],
                         uvs=torch.from_numpy(np.ascontiguousarray(uv_gltf, np.float32))[None])
        logging.info(f"[D3DEasyUVUnwrap] {text}")
        _progress_text(unique_id, text)
        return (out, text)


NODE_CLASS_MAPPINGS = {"D3DEasyUVUnwrap": D3DEasyUVUnwrap}
NODE_DISPLAY_NAME_MAPPINGS = {"D3DEasyUVUnwrap": "D3D - EasyUV Unwrap (body-part seams + learning)"}
