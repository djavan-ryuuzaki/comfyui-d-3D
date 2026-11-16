"""Sweep EasyUV seam angle / distortion tolerance (with our body-part seams) on GLB files.
usage: python easyuv_sweep.py OUT_DIR mesh.glb [...]  (same setup as easyuv_compare.py)"""
import os
import sys
import time

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tools"))
from easyuv_compare_lib import load  # noqa: E402

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
    wv, wf, adj, body, lab, full = cu.body_part_labels(np.asarray(g.vertices, np.float64), np.asarray(g.faces, np.int64))
    seams = cu.body_part_seams(wf, adj, full)
    print(f"\n== {name}")
    for angle in (66.0, 100.0, 130.0, 179.0):
        for dist in (0.12, 0.25, 0.4):
            for shape in (0, 3):
                p = eb._Params(angle_deg=angle, max_distortion=dist, relax_iterations=4, margin=8 / 2048,
                               density_mode=0, texture_size=2048, target_density=0.0, quality=2,
                               rotate_islands=1, priority=0.5, shape=shape)
                cuv, ch, st = eb.easyuv_unwrap(lib, wv, wf, seams, p, False, 6)
                ev, ef, euv = eb.split_by_corner_uv(wv, wf, cuv, ch)
                ov = cu._self_overlap(ev.astype(np.float64), ef, euv.astype(np.float64))
                print(f"angle {angle:5.0f} dist {dist:.2f} shape {eb.SHAPES[shape]:7s}: {st['charts']:4d} islands, "
                      f"{st['flipped']} flipped, mean dist {st['mean_distortion']:.3f}, max {st['max_distortion']:.3f}, "
                      f"coverage {st['coverage']:.3f}, overlap {ov:.4f}")
                if angle == 130.0 and dist == 0.25 and shape == 3:
                    np.savez(os.path.join(out_dir, f"{name}_easyuv_a130_d025.npz"), v=ev, f=ef, uv=euv)
