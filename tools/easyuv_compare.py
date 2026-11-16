"""Compare EasyUV (with/without our body-part seams, with/without learning) against D3D - Character UV.

usage: python easyuv_compare.py OUT_DIR mesh1.glb [mesh2.glb ...]
Run with ComfyUI's python from the ComfyUI folder (so comfy_extras imports). Needs easyuv.dll found by
easyuv_bridge.load_easyuv (EASYUV_DLL, comfyui-d-3D/bin, or the EasyUV Blender extension)."""
import os
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import importlib.util as ilu  # noqa: E402


def load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = ilu.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lp = load("lowpoly_tools")
cu = load("character_uv")
eb = load("easyuv_bridge")

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
lib, where = eb.load_easyuv(os.environ.get("EASYUV_DLL", ""))
print("easyuv:", where)
for path in sys.argv[2:]:
    name = os.path.splitext(os.path.basename(path))[0]
    s = trimesh.load(path, force="mesh", process=False)
    g = trimesh.Trimesh(s.vertices, s.faces, process=True)
    v, f = np.asarray(g.vertices, np.float64), np.asarray(g.faces, np.int64)
    print(f"\n== {name}: {len(f)} faces, genus {(2 - g.euler_number) // 2}")
    t = time.time()
    sv, sf, uv, info = cu.character_uv(v, f)
    ov = cu._self_overlap(sv.astype(np.float64), sf, uv.astype(np.float64))
    print(f"CharacterUV: {info['islands']} islands, {info['flipped']} flipped, overlap {ov:.4f}, {time.time() - t:.1f}s")
    np.savez(os.path.join(out_dir, f"{name}_charuv.npz"), v=sv, f=sf, uv=uv)
    if lib is None:
        continue
    wv, wf, adj, body, lab, full = cu.body_part_labels(v, f)
    seams = cu.body_part_seams(wf, adj, full)
    for tag, use_seams, learn in (("easyuv", False, False), ("easyuv_seams", True, False),
                                  ("easyuv_seams_learn", True, True)):
        p = eb._Params(angle_deg=66.0, max_distortion=0.12, relax_iterations=4, margin=8 / 2048, density_mode=0,
                       texture_size=2048, target_density=0.0, quality=2, rotate_islands=1, priority=0.5, shape=3)
        t = time.time()
        cuv, ch, st = eb.easyuv_unwrap(lib, wv, wf, seams if use_seams else np.zeros((0, 2)), p, learn, 6)
        ev, ef, euv = eb.split_by_corner_uv(wv, wf, cuv, ch)
        ov = cu._self_overlap(ev.astype(np.float64), ef, euv.astype(np.float64))
        print(f"{tag}: {st} | overlap {ov:.4f} | {time.time() - t:.1f}s")
        np.savez(os.path.join(out_dir, f"{name}_{tag}.npz"), v=ev, f=ef, uv=euv)
