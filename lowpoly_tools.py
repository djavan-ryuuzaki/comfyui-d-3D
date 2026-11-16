"""Game-ready low-poly character tools for ComfyUI's native MESH type (ComfyUI 0.45+).

Four nodes that sit around the native DecimateMesh / UnwrapMesh / ApplyTextureToMesh chain:

* D3DCharacterDecimate  - QEM decimation that keeps more triangles on the head (face + hair) and hands.
* D3DCharacterQuadRemesh - Instant Meshes quads, smaller on the head and hands.
* D3DMeshCleanup        - weld, drop non-manifold / duplicate / degenerate faces and specks, fill small holes,
                          fix winding, and report (or refuse) a mesh that is not watertight.
* D3DUVTexelBoost       - repacks the UV islands of an unwrapped mesh so the head and hands get more texels.
* D3DGameMaterial       - zeroes stray metallic on cloth / skin / leather and turns off doubleSided for export.

Head and hands are found from the shape alone, assuming a standing character in T- or A-pose: the head is a
sphere at the top centre, the hands are the two ends of the arm span.
"""
from __future__ import annotations

import copy
import logging
import math
from contextlib import contextmanager

import numpy as np
import torch

UP_AXES = ["auto", "Y", "Z"]


# ---------------------------------------------------------------------------------------------- regions

def _smooth(t):
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class _Regions:
    """Head / hand masks in a frame fixed from the input vertices, so they stay put while QEM moves verts."""

    def __init__(self, verts, up_axis="auto", head_size=0.2, hands_size=0.1):
        v = torch.as_tensor(verts).detach().float().cpu()
        lo, hi = v.min(0).values, v.max(0).values
        ext = hi - lo
        if up_axis == "Y":
            up = 1
        elif up_axis == "Z":
            up = 2
        else:
            up = 1 if float(ext[1]) >= float(ext[2]) else 2
        self.up = up
        self.lat = 0
        self.depth = 3 - up  # the remaining axis of {1, 2}
        h = max(float(ext[up]), 1e-6)
        top = float(hi[up])
        near_top = v[v[:, up] > top - 0.03 * h]
        if near_top.shape[0] == 0:
            near_top = v
        self.r = max(head_size, 1e-3) * h * 0.5
        c = torch.zeros(3)
        c[self.lat] = near_top[:, self.lat].median()
        c[self.depth] = near_top[:, self.depth].median()
        c[up] = top - self.r
        self.head_center = c
        self.xmin, self.xmax = float(lo[self.lat]), float(hi[self.lat])
        self.width = max(self.xmax - self.xmin, 1e-6)
        self.hands_size = max(hands_size, 1e-3)

    def head(self, verts):
        c = self.head_center.to(verts.device, verts.dtype)
        d = torch.linalg.norm(verts - c, dim=-1)
        return _smooth(1.0 - (d - self.r) / (0.35 * self.r))

    def hands(self, verts):
        x = verts[:, self.lat]
        span = self.hands_size * self.width
        ramp = 0.25 * span
        left = _smooth((self.xmin + span - x) / ramp)
        right = _smooth((x - (self.xmax - span)) / ramp)
        return torch.maximum(left, right)

    def density(self, verts, head_scale, hands_scale):
        """1 on the body, head_scale on the head, hands_scale on the hands, smooth in between."""
        return 1.0 + (head_scale - 1.0) * self.head(verts) + (hands_scale - 1.0) * self.hands(verts)


# ---------------------------------------------------------------------------------------------- helpers

def _batch_items(mesh):
    from comfy_extras.nodes_save_3d import get_mesh_batch_item

    if isinstance(mesh.vertices, list):
        return len(mesh.vertices), lambda i: (mesh.vertices[i], mesh.faces[i], None, None, None)
    if mesh.vertices.ndim == 2:
        return 1, lambda i: (mesh.vertices, mesh.faces, mesh.vertex_colors, mesh.uvs, mesh.normals)
    return int(mesh.vertices.shape[0]), lambda i: get_mesh_batch_item(mesh, i)


def _pack_geometry(verts_list, faces_list):
    """Geometry-only MESH (no UVs/textures) from per-item vertex/face tensors."""
    from comfy_api.latest import Types

    if len(verts_list) == 1:
        return Types.MESH(vertices=verts_list[0][None].float(), faces=faces_list[0][None].long())
    from comfy_extras.nodes_save_3d import pack_variable_mesh_batch
    return pack_variable_mesh_batch(verts_list, faces_list)


@contextmanager
def _weighted_quadrics(regions, head_detail, hands_detail):
    """Scale the per-vertex quadric of the native QEM by the region weight while it runs.

    A heavier quadric makes collapsing that vertex cost more, so QEM spends its triangle budget there.
    """
    import comfy_extras.mesh3d.postprocess.qem_decimate as qd

    original = qd._build_quadrics

    def weighted(verts, faces, cfg, b_edges):
        Q = original(verts, faces, cfg, b_edges)
        w = regions.density(verts, head_detail, hands_detail).to(Q.dtype)
        return Q * w.view(-1, *([1] * (Q.ndim - 1)))

    qd._build_quadrics = weighted
    try:
        yield
    finally:
        qd._build_quadrics = original


def _progress_text(unique_id, text):
    if not unique_id:
        return
    try:
        from server import PromptServer
        PromptServer.instance.send_progress_text(text, unique_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------- decimate

class D3DCharacterDecimate:
    """QEM decimation (the native DecimateMesh algorithm) that keeps more detail on the head and hands."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "High-poly character, usually from RemeshMesh."}),
                "target_face_count": ("INT", {"default": 12000, "min": 100, "max": 5_000_000, "step": 100,
                                              "tooltip": "Total triangles. The budget is the same; it is only spent differently."}),
                "head_detail": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 100.0, "step": 0.5,
                                          "tooltip": "How much more a collapse on the head (face + hair) costs. 1 = same as the native DecimateMesh."}),
                "hands_detail": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 100.0, "step": 0.5,
                                           "tooltip": "Same for the hands (the two ends of the arm span)."}),
                "head_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01,
                                        "tooltip": "Head height as a fraction of the character height (0.2 suits a stylised head, 0.13 a realistic one)."}),
                "hands_size": ("FLOAT", {"default": 0.1, "min": 0.02, "max": 0.3, "step": 0.01,
                                         "tooltip": "Hand length as a fraction of the arm span."}),
                "up_axis": (UP_AXES, {"default": "auto", "tooltip": "auto picks Y or Z, whichever is taller. TRELLIS/Pixal3D meshes are Y up."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "decimate"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("Decimates like the native DecimateMesh (QEM, midpoint) but weights the quadrics on the head and "
                   "hands, so face, hair and fingers keep more triangles for the same total.")

    def decimate(self, mesh, target_face_count, head_detail, hands_detail, head_size, hands_size, up_axis,
                 unique_id=None):
        import comfy.model_management
        from comfy_extras.mesh3d.postprocess.qem_decimate import QEMConfig, qem_simplify

        device = comfy.model_management.get_torch_device()
        n, item = _batch_items(mesh)
        out_v, out_f, info = [], [], []
        for i in range(n):
            v, f = item(i)[:2]
            regions = _Regions(v, up_axis, head_size, hands_size)
            n_in = int(f.shape[0])
            if n_in > target_face_count:
                with _weighted_quadrics(regions, float(head_detail), float(hands_detail)):
                    rv, rf, _ = qem_simplify(v.to(device).float(), f.to(device).long(), int(target_face_count),
                                             config=QEMConfig())
                v, f = rv.cpu(), rf.cpu()
            v, f = v.float().cpu(), f.long().cpu()
            out_v.append(v)
            out_f.append(f)
            info.append(_region_stats(regions, v, f, n_in))
        text = "\n".join(info)
        logging.info(f"[D3DCharacterDecimate] {text}")
        _progress_text(unique_id, text)
        return (_pack_geometry(out_v, out_f), text)


def _region_stats(regions, v, f, n_in):
    cent = v[f].mean(1)
    head = regions.head(cent) > 0.5
    hands = regions.hands(cent) > 0.5
    nf = max(int(f.shape[0]), 1)
    return (f"faces {n_in} -> {int(f.shape[0])} | head {int(head.sum())} ({100 * int(head.sum()) / nf:.0f}%), "
            f"hands {int(hands.sum())} ({100 * int(hands.sum()) / nf:.0f}%)")


# ---------------------------------------------------------------------------------------------- cleanup

def _face_areas(v, f):
    return 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)


def _edge_table(f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    face_of = np.tile(np.arange(len(f)), 3)
    key = np.sort(e, axis=1)
    uniq, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e, face_of, uniq, inv.reshape(-1), counts


def _weld(v, f, eps_rel):
    diag = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
    q = np.round(v / max(eps_rel * diag, 1e-12)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return v[first], inv.reshape(-1)[f]


def _collapse_short_edges(v, f, eps_rel):
    """Merge the ends of every edge shorter than eps_rel x diagonal (clusters move to their mean).

    UnwrapMesh welds vertices closer than its weld_distance; doing it here first, with a radius at least as
    large, means the unwrap cannot turn faces into slivers later and open holes the cleanup never saw.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    diag = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
    eps = eps_rel * diag
    merged = 0
    for _ in range(3):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
        short = e[np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1) < eps]
        if len(short) == 0:
            break
        g = coo_matrix((np.ones(len(short), np.int8), (short[:, 0], short[:, 1])), shape=(len(v), len(v)))
        n, label = connected_components(g, directed=False)
        counts = np.bincount(label, minlength=n).astype(np.float64)
        nv = np.zeros((n, 3))
        np.add.at(nv, label, v)
        merged += len(v) - n
        v = nv / counts[:, None]
        f = label[f]
        ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        f = f[ok]
    return v, f, merged


def _drop_bad_faces(v, f):
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
    f = f[ok]
    area = _face_areas(v, f)
    scale = float(np.linalg.norm(v.max(0) - v.min(0))) ** 2 or 1.0
    f = f[area > 1e-14 * scale]
    if len(f) == 0:
        return f
    # duplicates: the same triangle wound both ways is a fold (two quads back to back) and both copies go;
    # copies wound the same way collapse to one
    order = np.argsort(f, axis=1)
    even = np.array([1, -1])[((order[:, 0] > order[:, 1]).astype(int) + (order[:, 0] > order[:, 2]) + (order[:, 1] > order[:, 2])) % 2]
    _, first, inv = np.unique(np.sort(f, axis=1), axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    net = np.bincount(inv, weights=even, minlength=len(first))
    keep_group = net != 0
    keep = first[keep_group]
    out = f[np.sort(keep)].copy()
    # a surviving group keeps the winding of its majority
    flip_groups = np.zeros(len(first), bool)
    flip_groups[keep_group] = np.sign(net[keep_group]) != even[first[keep_group]]
    flip = flip_groups[inv[np.sort(keep)]]
    out[flip] = out[flip][:, ::-1]
    return out


def _drop_nonmanifold(v, f):
    """Edges used by more than two faces keep their two largest faces."""
    _, face_of, _, inv, counts = _edge_table(f)
    bad_edges = np.nonzero(counts > 2)[0]
    if bad_edges.size == 0:
        return f, 0
    area = _face_areas(v, f)
    # how well each face is stitched in: its edges that are shared by exactly two faces
    nf = len(f)
    manifold_edges = np.bincount(face_of, weights=(counts[inv] == 2).astype(np.float64), minlength=nf)
    drop = set()
    slots = np.nonzero(np.isin(inv, bad_edges))[0]
    by_edge = {}
    for s in slots:
        by_edge.setdefault(int(inv[s]), []).append(int(face_of[s]))
    for faces in by_edge.values():
        # keep the two faces best connected to the rest (fins and flaps go), larger first on ties
        faces = sorted(set(faces), key=lambda k: (-manifold_edges[k], -area[k]))
        drop.update(faces[2:])
    keep = np.ones(len(f), bool)
    keep[list(drop)] = False
    return f[keep], int(bad_edges.size)


def _split_sheets(v, f, sep_rel=5e-4):
    """Split surfaces that touch along an edge or at a vertex into separate closed sheets (ComfyUI's own
    _split_touching_sheets), then push each copy of a split vertex a little along its own sheet's normal so
    UnwrapMesh's weld does not glue them back. Returns (v, f, number of split vertices) or None if unavailable."""
    try:
        from comfy_extras.mesh3d.postprocess.qem_decimate import _split_touching_sheets
    except Exception:
        return None
    _, _, _, _, counts = _edge_table(f)
    if not (counts > 2).any():
        return v, f, 0
    nf, src = _split_touching_sheets(torch.from_numpy(v), torch.from_numpy(f))
    nf = nf.numpy().astype(np.int64)
    src = src.numpy()
    nv = v[src]
    dup = np.bincount(src, minlength=len(v))[src] > 1
    if dup.any():
        fn = np.cross(nv[nf[:, 1]] - nv[nf[:, 0]], nv[nf[:, 2]] - nv[nf[:, 0]])
        vn = np.zeros_like(nv)
        for k in range(3):
            np.add.at(vn, nf[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)
        diag = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
        nv[dup] += vn[dup] * sep_rel * diag
    return nv, nf, int(dup.sum())


def _drop_specks(f, nv, min_faces):
    """Remove pieces of fewer than min_faces faces. Pieces are joined through shared edges, not shared vertices,
    so a lone triangle hanging off the surface by its corners also counts as a speck."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(f) == 0:
        return f, 0
    _, face_of, _, inv, counts = _edge_table(f)
    order = np.argsort(inv, kind="stable")
    si, sf = inv[order], face_of[order]
    same = si[1:] == si[:-1]
    a, b = sf[:-1][same], sf[1:][same]
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(len(f), len(f)))
    _, label = connected_components(g, directed=False)
    sizes = np.bincount(label)
    keep = sizes[label] >= max(min_faces, 1)
    return f[keep], int((sizes < max(min_faces, 1)).sum())


def _boundary_loops(boundary):
    """Split directed boundary edges into closed loops; a pinched vertex (two loops touching) is walked twice."""
    out_edges = {}
    for k, (a, _) in enumerate(boundary):
        out_edges.setdefault(int(a), []).append(k)
    used = np.zeros(len(boundary), bool)
    loops, broken = [], 0
    for k0 in range(len(boundary)):
        if used[k0]:
            continue
        start = int(boundary[k0, 0])
        loop, k = [], k0
        while True:
            used[k] = True
            loop.append(int(boundary[k, 0]))
            cur = int(boundary[k, 1])
            if cur == start:
                loops.append(loop)
                break
            nxt = [j for j in out_edges.get(cur, []) if not used[j]]
            if not nxt:
                broken += 1
                break
            k = nxt[0]
    return loops, broken


def _ear_clip(v, loop):
    """Triangulate a boundary loop by cutting the sharpest convex corner first (no centre pole).

    Faces run against the loop direction, so they match the winding of the surrounding faces.
    """
    loop = list(loop)
    p = v[loop]
    n = np.zeros(3)
    for k in range(len(loop)):  # Newell normal of the loop
        a, b = p[k], p[(k + 1) % len(loop)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    n /= max(np.linalg.norm(n), 1e-20)
    tris = []
    while len(loop) > 3:
        m = len(loop)
        best, best_score = 0, None
        for k in range(m):
            a, b, c = v[loop[k - 1]], v[loop[k]], v[loop[(k + 1) % m]]
            d1, d2 = a - b, c - b
            cosang = np.dot(d1, d2) / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-20)
            convex = np.dot(np.cross(b - a, c - b), n) > 0
            score = (0 if convex else 1, -cosang)  # convex first, then the sharpest corner
            if best_score is None or score < best_score:
                best, best_score = k, score
        prev, cur, nxt = loop[best - 1], loop[best], loop[(best + 1) % m]
        tris.append([nxt, cur, prev])
        loop.pop(best)
    tris.append([loop[2], loop[1], loop[0]])
    return tris


def _fill_holes(v, f, max_edges):
    """Fill boundary loops of at most max_edges edges by ear clipping."""
    e, _, _, inv, counts = _edge_table(f)
    boundary = e[counts[inv] == 1]  # directed as in their face
    if len(boundary) == 0 or max_edges <= 0:
        return v, f, 0, 0
    loops, broken = _boundary_loops(boundary)
    new_v, new_f = [], []
    filled, left = 0, broken
    nv = len(v)
    remap = None
    for loop in loops:
        if len(loop) > max_edges or len(loop) < 3:
            left += 1
            continue
        if len(loop) == 3:
            p = v[loop]
            sides = np.linalg.norm(p[[1, 2, 0]] - p, axis=1)
            area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
            if area < 1e-3 * sides.max() ** 2:
                # a slit (T-junction): a triangle there would be a sliver; weld the middle vertex instead
                k = int(np.argmax(sides))  # longest side runs loop[k] -> loop[k+1]; the middle vertex is opposite
                mid, a, b = loop[(k + 2) % 3], loop[k], loop[(k + 1) % 3]
                tgt = a if np.linalg.norm(v[mid] - v[a]) <= np.linalg.norm(v[mid] - v[b]) else b
                if remap is None:
                    remap = np.arange(len(v))
                remap[mid] = tgt
                filled += 1
                continue
        new_f.extend(_ear_clip(v, loop))
        filled += 1
    if remap is not None:
        f = remap[f]
    if new_v:
        v = np.concatenate([v, np.asarray(new_v, v.dtype)], axis=0)
    if new_f:
        f = np.concatenate([f, np.asarray(new_f, f.dtype)], axis=0)
    return v, f, filled, left


def _drop_unused(v, f):
    used = np.zeros(len(v), bool)
    used[f.reshape(-1)] = True
    remap = np.cumsum(used) - 1
    return v[used], remap[f]


def cleanup_mesh(v, f, weld_eps_rel=5e-4, min_component_faces=20, max_hole_edges=64, passes=4):
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    n_in = len(f)
    v, f = _weld(v, f, 1e-7)
    merged = 0
    if weld_eps_rel > 0:
        v, f, merged = _collapse_short_edges(v, f, weld_eps_rel)
    stats = {"nonmanifold_fixed": 0, "specks": 0, "holes_filled": 0, "merged": merged}
    for _ in range(passes):
        f = _drop_bad_faces(v, f)
        split = _split_sheets(v, f, sep_rel=max(weld_eps_rel, 5e-4))
        if split is not None and split[2] > 0:
            v, f, nm = split
            f = _drop_bad_faces(v, f)
        else:
            nm = 0
        f, nm2 = _drop_nonmanifold(v, f)  # whatever the split could not separate
        nm += nm2
        f, sp = _drop_specks(f, len(v), min_component_faces)
        v, f, filled, _ = _fill_holes(v, f, max_hole_edges)
        stats["nonmanifold_fixed"] += nm
        stats["specks"] += sp
        stats["holes_filled"] += filled
        if nm == 0 and filled == 0 and sp == 0:
            break
    f = _drop_bad_faces(v, f)
    v, f = _drop_unused(v, f)

    try:
        import trimesh
        tm = trimesh.Trimesh(v, f, process=False)
        trimesh.repair.fix_normals(tm, multibody=True)
        f = np.asarray(tm.faces, np.int64)
    except Exception as e:  # trimesh is a dependency of this pack, but keep the node usable without it
        logging.warning(f"[D3DMeshCleanup] winding fix skipped: {e}")

    _, _, _, inv, counts = _edge_table(f)
    stats.update(
        faces_in=n_in, faces_out=int(len(f)),
        boundary_edges=int((counts == 1).sum()),
        nonmanifold_edges=int((counts > 2).sum()),
    )
    stats["watertight"] = stats["boundary_edges"] == 0 and stats["nonmanifold_edges"] == 0
    stats["max_valence"] = int(np.bincount(f.reshape(-1)).max()) if len(f) else 0
    return v.astype(np.float32), f, stats


class D3DMeshCleanup:
    """Watertight gate for the low-poly mesh: run it after decimation and before UnwrapMesh."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Low-poly mesh after DecimateMesh / Quad Remesh, before UnwrapMesh. UVs and textures are dropped."}),
                "max_hole_edges": ("INT", {"default": 64, "min": 0, "max": 2048,
                                           "tooltip": "Fill holes whose border has up to this many edges. 0 = do not fill."}),
                "min_component_faces": ("INT", {"default": 20, "min": 0, "max": 100000,
                                                "tooltip": "Remove loose pieces smaller than this (floating specks)."}),
                "weld_epsilon": ("FLOAT", {"default": 5e-4, "min": 0.0, "max": 0.01, "step": 1e-5,
                                           "tooltip": "Merge-by-distance radius as a fraction of the mesh diagonal. Keep it at least "
                                                      "as large as UnwrapMesh weld_distance (0.0002), or the unwrap can weld slivers into new holes."}),
                "if_not_watertight": (["warn", "error"], {"default": "warn",
                                                          "tooltip": "error stops the run instead of saving a mesh with holes or non-manifold edges."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "report")
    FUNCTION = "clean"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("Merge by distance, removes faces on edges shared by more than two faces, duplicate/degenerate faces "
                   "and loose specks, fills small holes, fixes winding and reports whether the mesh is watertight.")

    def clean(self, mesh, max_hole_edges, min_component_faces, weld_epsilon, if_not_watertight, unique_id=None):
        n, item = _batch_items(mesh)
        out_v, out_f, lines, leaky = [], [], [], []
        for i in range(n):
            v, f = item(i)[:2]
            cv, cf, s = cleanup_mesh(v.detach().cpu().numpy(), f.detach().cpu().numpy(),
                                     weld_eps_rel=weld_epsilon, min_component_faces=min_component_faces,
                                     max_hole_edges=max_hole_edges)
            out_v.append(torch.from_numpy(cv))
            out_f.append(torch.from_numpy(cf))
            line = (f"faces {s['faces_in']} -> {s['faces_out']} | watertight: {'yes' if s['watertight'] else 'NO'} | "
                    f"non-manifold fixed {s['nonmanifold_fixed']}, holes filled {s['holes_filled']}, "
                    f"specks removed {s['specks']}, verts merged {s['merged']} | max valence {s['max_valence']} | left: {s['boundary_edges']} border edges, "
                    f"{s['nonmanifold_edges']} non-manifold edges")
            lines.append(line)
            if not s["watertight"]:
                leaky.append(i)
        report = "\n".join(lines)
        logging.info(f"[D3DMeshCleanup] {report}")
        _progress_text(unique_id, report)
        if leaky and if_not_watertight == "error":
            raise RuntimeError(f"D3DMeshCleanup: mesh is not watertight.\n{report}\n"
                               "Raise max_hole_edges, or raise RemeshMesh 'fill', or set if_not_watertight = warn.")
        return (_pack_geometry(out_v, out_f), report)


# ---------------------------------------------------------------------------------------------- quad remesh

class _RadialMap:
    """Monotone radial stretch around `center`: x`scale` inside r0, back to identity at r0 * (scale + 1).

    Instant Meshes only does a uniform quad size. Remeshing the stretched shape and mapping the quads back
    gives smaller quads where the shape was stretched, with no folds since the radial profile is monotone.
    """

    def __init__(self, center, r0, scale):
        from scipy.interpolate import PchipInterpolator

        self.c = np.asarray(center, np.float64)
        r0 = max(float(r0), 1e-6)
        self.R = r0 * (scale + 1.0)
        knots_x = [0.0, r0, self.R, 2.0 * self.R]
        knots_y = [0.0, scale * r0, self.R, 2.0 * self.R]
        self.rs = np.linspace(0.0, self.R, 4096)
        self.fs = PchipInterpolator(knots_x, knots_y)(self.rs)

    def _apply(self, p, src, dst):
        d = p - self.c
        r = np.linalg.norm(d, axis=1)
        inside = r < self.R
        r_new = r.copy()
        r_new[inside] = np.interp(r[inside], src, dst)
        return self.c + d * (r_new / np.maximum(r, 1e-12))[:, None]

    def forward(self, p):
        return self._apply(p, self.rs, self.fs)

    def inverse(self, p):
        return self._apply(p, self.fs, self.rs)


def _load_quad_mesh():
    try:
        from . import quad_mesh
        return quad_mesh
    except ImportError:
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location("d3d_quad_mesh", os.path.join(os.path.dirname(__file__), "quad_mesh.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def character_quad_remesh(v, f, target_quads, head_scale, hands_scale, head_size, hands_size, up_axis,
                          smooth_iterations=2, crease_angle=0.0):
    qm = _load_quad_mesh()
    v = np.asarray(v, np.float64)
    regions = _Regions(v, up_axis, head_size, hands_size)
    vt = torch.from_numpy(v).float()
    hands = regions.hands(vt).numpy() > 0.5
    maps = []
    p = v.copy()
    if head_scale > 1.0:
        maps.append(_RadialMap(regions.head_center.numpy(), regions.r, head_scale))
        p = maps[-1].forward(p)
    if hands_scale > 1.0:
        mid = 0.5 * (regions.xmin + regions.xmax)
        for side in (v[:, regions.lat] < mid, v[:, regions.lat] >= mid):
            sel = hands & side
            if sel.sum() < 8:
                continue
            # The whole hand must sit in the uniform part of the stretch (radius r0). Past r0 the map
            # compresses radially, so fingertips there got longer quads than the body and Instant Meshes
            # fused the fingers.
            c = p[sel].mean(0)
            r0 = max(0.5 * regions.hands_size * regions.width, 1.05 * float(np.linalg.norm(p[sel] - c, axis=1).max()))
            maps.append(_RadialMap(c, r0, hands_scale))
            p = maps[-1].forward(p)
    qv, quads = qm.remesh_quads(p.astype(np.float32), f, int(target_quads), float(crease_angle),
                                int(smooth_iterations), False, True)
    qv, quads = qm.clean_quads(qv, quads)
    qv = qv.astype(np.float64)
    for m in reversed(maps):
        qv = m.inverse(qv)
    tris = qm.quads_to_triangle_pairs(quads)
    cent = torch.from_numpy(qv[tris].mean(1)).float()
    head_share = float((regions.head(cent) > 0.5).float().mean())
    hands_share = float((regions.hands(cent) > 0.5).float().mean())
    return qv.astype(np.float32), tris, len(quads), head_share, hands_share


class D3DCharacterQuadRemesh:
    """Instant Meshes quad retopology with smaller quads on the head and hands (native MESH in and out)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Watertight high-poly character, usually from RemeshMesh."}),
                "target_quads": ("INT", {"default": 6000, "min": 100, "max": 500000, "step": 100,
                                         "tooltip": "About how many quads (each one is 2 triangles in the GLB)."}),
                "head_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.05,
                                         "tooltip": "Quads per unit length on the head relative to the body. 1.5 = about 2x the quads on the head (~30% of the budget on a stylised character)."}),
                "hands_scale": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 5.0, "step": 0.05,
                                          "tooltip": "Quads per unit length on each hand relative to the body. Fingers need 2.5 or more to stay separate (2.5 = about 18% of the budget on the hands)."}),
                "head_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01}),
                "hands_size": ("FLOAT", {"default": 0.1, "min": 0.02, "max": 0.3, "step": 0.01}),
                "smooth_iterations": ("INT", {"default": 2, "min": 0, "max": 10}),
                "crease_angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 180.0, "step": 1.0}),
                "up_axis": (UP_AXES, {"default": "auto"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("Quad retopology (Instant Meshes) with denser quads on the head and hands. Output is geometry only, "
                   "quads stored as triangle pairs; follow it with D3D Mesh Cleanup, which closes the holes Instant Meshes leaves.")

    def remesh(self, mesh, target_quads, head_scale, hands_scale, head_size, hands_size, smooth_iterations,
               crease_angle, up_axis, unique_id=None):
        n, item = _batch_items(mesh)
        out_v, out_f, lines = [], [], []
        for i in range(n):
            v, f = item(i)[:2]
            qv, tris, nq, hs, hd = character_quad_remesh(
                v.detach().cpu().numpy(), f.detach().cpu().numpy().astype(np.uint32), target_quads,
                float(head_scale), float(hands_scale), head_size, hands_size, up_axis,
                smooth_iterations, crease_angle)
            out_v.append(torch.from_numpy(qv))
            out_f.append(torch.from_numpy(tris.astype(np.int64)))
            lines.append(f"{nq} quads | head {100 * hs:.0f}% of the faces, hands {100 * hd:.0f}%")
        info = "\n".join(lines)
        logging.info(f"[D3DCharacterQuadRemesh] {info}")
        _progress_text(unique_id, info)
        return (_pack_geometry(out_v, out_f), info)


# ---------------------------------------------------------------------------------------------- UV texel boost

def _uv_islands(f, nv):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(nv, nv))
    n, label = connected_components(g, directed=False)
    return n, label


def repack_uvs(v, f, uv, density, resolution, padding):
    """Repack UV islands with the native packer; island texel density is multiplied by `density`."""
    from comfy_extras.mesh3d.uv_unwrap import pack as uv_pack

    nv = len(v)
    _, vlabel = _uv_islands(f, nv)
    flabel = vlabel[f[:, 0]]
    used = np.unique(f.reshape(-1))
    # islands that own faces, renumbered 0..n-1
    isl_ids, flabel = np.unique(flabel, return_inverse=True)
    n = len(isl_ids)
    remap_isl = -np.ones(vlabel.max() + 1, np.int64)
    remap_isl[isl_ids] = np.arange(n)
    vis = remap_isl[vlabel[used]]

    order_v = np.argsort(vis, kind="stable")
    verts_sorted = used[order_v]
    vert_counts = np.bincount(vis, minlength=n)
    vert_off = np.zeros(n + 1, np.int64)
    np.cumsum(vert_counts, out=vert_off[1:])
    local = np.empty(nv, np.int64)
    local[verts_sorted] = np.arange(len(verts_sorted)) - vert_off[vis[order_v]]

    order_f = np.argsort(flabel, kind="stable")
    face_counts = np.bincount(flabel, minlength=n)
    face_off = np.zeros(n + 1, np.int64)
    np.cumsum(face_counts, out=face_off[1:])
    faces_local = local[f[order_f]]

    uvs_cat = uv[verts_sorted].astype(np.float64)
    a, b, c = (uv[f[:, k]].astype(np.float64) for k in range(3))
    uv_area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    area3 = _face_areas(v.astype(np.float64), f)
    fdens = density[f].mean(1)
    a3 = np.bincount(flabel, weights=area3 * fdens * fdens, minlength=n)
    auv = np.bincount(flabel, weights=uv_area, minlength=n)
    a3 = np.maximum(a3, 1e-20)
    auv = np.maximum(auv, 1e-20)

    tpu = math.sqrt(float(resolution) ** 2 * 0.62 / float(a3.sum()))
    px, py, sw, th, sc, chh, aw, ah = uv_pack.pack_bitmap_concat(
        uvs_cat, vert_off, faces_local, face_off, a3, auv, texels_per_unit=tpu, padding_texels=int(padding))
    packed = uv_pack.apply_placements_concat(uvs_cat, vert_off, px, py, sw, th, sc, chh, aw, ah)
    out = uv.copy()
    out[verts_sorted] = packed
    return out, n, (aw, ah)


class D3DUVTexelBoost:
    """Run right after UnwrapMesh: same islands, repacked so the head and hands get more of the texture."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Output of UnwrapMesh. Feed the result to every bake and to ApplyTextureToMesh."}),
                "head_texel_scale": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 6.0, "step": 0.1,
                                               "tooltip": "Texel density on the head (face + hair) relative to the body. 2 = twice the resolution."}),
                "hands_texel_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 6.0, "step": 0.1}),
                "head_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01,
                                        "tooltip": "Head height as a fraction of the character height. Use the same value as in the decimate node."}),
                "hands_size": ("FLOAT", {"default": 0.1, "min": 0.02, "max": 0.3, "step": 0.01}),
                "resolution": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256,
                                       "tooltip": "Texture resolution the atlas is packed for (same as UnwrapMesh / bakes)."}),
                "padding": ("INT", {"default": 2, "min": 0, "max": 16, "tooltip": "Texels between islands."}),
                "up_axis": (UP_AXES, {"default": "auto"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "boost"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("Repacks the UV islands from UnwrapMesh with the native packer, giving the head and hands more texels "
                   "per metre. Geometry, seams and island shapes do not change.")

    def boost(self, mesh, head_texel_scale, hands_texel_scale, head_size, hands_size, resolution, padding, up_axis,
              unique_id=None):
        if mesh.uvs is None:
            raise ValueError("D3DUVTexelBoost: mesh has no UVs. Connect it after UnwrapMesh.")
        n, item = _batch_items(mesh)
        new_uvs = mesh.uvs.clone()
        lines = []
        for i in range(n):
            v_t, f_t, _, uv_t, _ = item(i)
            v = v_t.detach().float().cpu().numpy()
            f = f_t.detach().long().cpu().numpy()
            uv = uv_t.detach().float().cpu().numpy().copy()
            uv[:, 1] = 1.0 - uv[:, 1]  # native MESH stores glTF v-down; pack in v-up like UnwrapMesh does
            regions = _Regions(v, up_axis, head_size, hands_size)
            vt = torch.from_numpy(v)
            density = regions.density(vt, float(head_texel_scale), float(hands_texel_scale)).numpy()
            head_before = _uv_share(uv, f, regions.head(vt).numpy())
            out, n_isl, _ = repack_uvs(v, f, uv, density, resolution, padding)
            head_after = _uv_share(out, f, regions.head(vt).numpy())
            out[:, 1] = 1.0 - out[:, 1]
            t = torch.from_numpy(out.astype(np.float32)).to(new_uvs)
            if new_uvs.ndim == 3:
                new_uvs[i, :t.shape[0]] = t
            else:
                new_uvs[:t.shape[0]] = t
            lines.append(f"{n_isl} islands | head uses {100 * head_before:.0f}% -> {100 * head_after:.0f}% of the used UV area")
        info = "\n".join(lines)
        logging.info(f"[D3DUVTexelBoost] {info}")
        _progress_text(unique_id, info)
        out_mesh = copy.copy(mesh)
        out_mesh.uvs = new_uvs
        return (out_mesh, info)


def _uv_share(uv, f, vmask):
    a, b, c = (uv[f[:, k]].astype(np.float64) for k in range(3))
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    m = vmask[f].mean(1) > 0.5
    return float(area[m].sum() / max(area.sum(), 1e-20))


# ---------------------------------------------------------------------------------------------- material

class D3DGameMaterial:
    """Last step before saving: clean metallic and set the glTF material flags for a game engine."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Textured mesh (after ApplyTextureToMesh / MeshSmoothNormals)."}),
                "metallic_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                 "tooltip": "Metallic below this becomes 0 (cloth, skin, leather); above it is kept (buckles, blades). 0 = keep the map as is."}),
                "double_sided": ("BOOLEAN", {"default": False,
                                             "tooltip": "glTF doubleSided. Off for a closed character (backface culling, correct normals in engines)."}),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "apply"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = "Zeroes metallic below a threshold (soft edge of 0.1) and sets glTF doubleSided."

    def apply(self, mesh, metallic_threshold, double_sided):
        out = copy.copy(mesh)
        mr = getattr(mesh, "metallic_roughness", None)
        if mr is not None and metallic_threshold > 0:
            mr = mr.clone()
            m = mr[..., 2]
            soft = 0.1
            keep = ((m - (metallic_threshold - soft)) / soft).clamp(0.0, 1.0)
            mr[..., 2] = m * keep
            out.metallic_roughness = mr
        mat = dict(mesh.material) if isinstance(getattr(mesh, "material", None), dict) else {}
        mat["double_sided"] = bool(double_sided)
        out.material = mat
        return (out,)


NODE_CLASS_MAPPINGS = {
    "D3DCharacterDecimate": D3DCharacterDecimate,
    "D3DMeshCleanup": D3DMeshCleanup,
    "D3DCharacterQuadRemesh": D3DCharacterQuadRemesh,
    "D3DUVTexelBoost": D3DUVTexelBoost,
    "D3DGameMaterial": D3DGameMaterial,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "D3DCharacterDecimate": "D3D - Character Decimate (head/hands detail)",
    "D3DMeshCleanup": "D3D - Mesh Cleanup (watertight)",
    "D3DCharacterQuadRemesh": "D3D - Character Quad Remesh (head/hands detail)",
    "D3DUVTexelBoost": "D3D - UV Texel Boost (head/hands)",
    "D3DGameMaterial": "D3D - Game Material (metallic/doubleSided)",
}
