"""Artist-style UVs for game characters, and two texture fixes that go with them (ComfyUI 0.45+ native MESH).

* D3DCharacterUV      - seams by body part (neck, shoulders, wrists, waist, hips; head split behind the ears,
                        limbs split along their length), each island flattened with LSCM and packed with
                        ComfyUI's own packer. Typically 15-30 islands instead of the 300+ of UnwrapMesh.
* D3DTexturePadding   - fills the texels around every UV island with the nearest island colour, so bilinear
                        filtering and mipmaps do not pull dark background into the seams.
* D3DAlbedoTopBlend   - uses a second albedo (TRELLIS) on top of the head, where the six QwenTex views see
                        little, colour-matched to the main albedo and blended softly.

Body parts come from the shape alone: a standing character in T- or A-pose, facing +Z or -Z (front_axis).
"""
from __future__ import annotations

import copy
import logging
import math

import numpy as np
import torch

try:
    from .lowpoly_tools import _Regions, _batch_items, _progress_text, repack_uvs, UP_AXES, _weld
except ImportError:  # loaded outside the package (tests)
    import importlib.util
    import os
    _spec = importlib.util.spec_from_file_location("d3d_lowpoly_tools", os.path.join(os.path.dirname(__file__), "lowpoly_tools.py"))
    _lp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_lp)
    _Regions, _batch_items, _progress_text, repack_uvs, UP_AXES, _weld = (
        _lp._Regions, _lp._batch_items, _lp._progress_text, _lp.repack_uvs, _lp.UP_AXES, _lp._weld)

PART_NAMES = ["head_front", "head_back", "torso_front", "torso_back", "arm_l", "arm_r", "hand_l", "hand_r",
              "leg_l", "leg_r"]


# ---------------------------------------------------------------------------------------------- topology helpers

def _edges(f):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    face_of = np.tile(np.arange(len(f)), 3)
    key = np.sort(e, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e, face_of, inv.reshape(-1), counts


def _face_adjacency(f):
    """(a, b) pairs of faces sharing an edge."""
    _, face_of, inv, _ = _edges(f)
    order = np.argsort(inv, kind="stable")
    si, sf = inv[order], face_of[order]
    same = si[1:] == si[:-1]
    return sf[:-1][same], sf[1:][same]


def _components(face_ids, adj_a, adj_b, n_faces):
    """Connected components (through shared edges) of a subset of faces. Returns a list of index arrays."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    inside = np.zeros(n_faces, bool)
    inside[face_ids] = True
    keep = inside[adj_a] & inside[adj_b]
    local = -np.ones(n_faces, np.int64)
    local[face_ids] = np.arange(len(face_ids))
    a, b = local[adj_a[keep]], local[adj_b[keep]]
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(len(face_ids), len(face_ids)))
    n, label = connected_components(g, directed=False)
    return [face_ids[label == k] for k in range(n)]


def _is_disk(f):
    """Topological disk: Euler characteristic 1 and a single boundary loop."""
    used = np.unique(f)
    _, _, inv, counts = _edges(f)
    V, E, F = len(used), len(counts), len(f)
    if V - E + F != 1:
        return False
    e, _, inv, counts = _edges(f)
    b = e[counts[inv] == 1]
    if len(b) == 0:
        return False
    # one loop: walking the boundary from any vertex must visit every boundary vertex
    nxt = {}
    for a, c in b:
        if int(a) in nxt:
            return False  # pinched boundary
        nxt[int(a)] = int(c)
    start = int(b[0, 0])
    cur, steps = nxt.get(start), 1
    while cur is not None and cur != start and steps <= len(b):
        cur = nxt.get(cur)
        steps += 1
    return cur == start and steps == len(b)


# ---------------------------------------------------------------------------------------------- LSCM

def lscm(v, f):
    """Least-squares conformal map of a disk-like patch. Returns (uv [V,2], flipped fraction, area distortion)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    used, lf = np.unique(f, return_inverse=True)
    lf = lf.reshape(-1, 3)
    p = v[used].astype(np.float64)
    nv, nf = len(p), len(lf)
    p0, p1, p2 = p[lf[:, 0]], p[lf[:, 1]], p[lf[:, 2]]
    e1 = p1 - p0
    n = np.cross(e1, p2 - p0)
    area2 = np.linalg.norm(n, axis=1)
    ok = area2 > 1e-20
    x_ax = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    y_ax = np.cross(n / np.maximum(area2[:, None], 1e-20), x_ax)
    q = np.zeros((nf, 3, 2))
    q[:, 1, 0] = np.linalg.norm(e1, axis=1)
    q[:, 2, 0] = np.einsum("ij,ij->i", p2 - p0, x_ax)
    q[:, 2, 1] = np.einsum("ij,ij->i", p2 - p0, y_ax)
    s = 1.0 / np.sqrt(np.maximum(area2, 1e-20))
    rows, cols, wr, wi = [], [], [], []
    for j in range(3):
        k, l = (j + 1) % 3, (j + 2) % 3
        wr.append((q[:, l, 0] - q[:, k, 0]) * s)
        wi.append((q[:, l, 1] - q[:, k, 1]) * s)
        rows.append(np.arange(nf))
        cols.append(lf[:, j])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    wr, wi = np.concatenate(wr), np.concatenate(wi)
    keep = ok[rows]
    rows, cols, wr, wi = rows[keep], cols[keep], wr[keep], wi[keep]

    # pins: the two boundary vertices farthest apart
    e, _, inv, counts = _edges(lf)
    bverts = np.unique(e[counts[inv] == 1])
    if len(bverts) < 2:
        bverts = np.arange(nv)
    bp = p[bverts]
    i0 = int(np.argmax(np.linalg.norm(bp - bp.mean(0), axis=1)))
    i1 = int(np.argmax(np.linalg.norm(bp - bp[i0], axis=1)))
    pin = [int(bverts[i0]), int(bverts[i1])]
    pin_uv = np.array([[0.0, 0.0], [float(np.linalg.norm(p[pin[1]] - p[pin[0]])) or 1.0, 0.0]])

    # real system [Wr -Wi; Wi Wr] [u; v] = 0, with the pinned unknowns moved to the right-hand side
    R = np.concatenate([rows, rows, rows + nf, rows + nf])
    C = np.concatenate([cols, cols + nv, cols, cols + nv])
    D = np.concatenate([wr, -wi, wi, wr])
    A = coo_matrix((D, (R, C)), shape=(2 * nf, 2 * nv)).tocsc()
    fixed = np.array([pin[0], pin[1], pin[0] + nv, pin[1] + nv])
    fixed_val = np.array([pin_uv[0, 0], pin_uv[1, 0], pin_uv[0, 1], pin_uv[1, 1]])
    free = np.setdiff1d(np.arange(2 * nv), fixed)
    Af, Ap = A[:, free], A[:, fixed]
    rhs = -(Ap @ fixed_val)
    sol = spsolve((Af.T @ Af).tocsc(), Af.T @ rhs)
    x = np.zeros(2 * nv)
    x[free] = sol
    x[fixed] = fixed_val
    uv = np.stack([x[:nv], x[nv:]], axis=1)
    if not np.all(np.isfinite(uv)):
        return None, 1.0, 1e9
    sa = _signed_uv_area(uv, lf)
    if sa.sum() < 0:  # mirrored solution
        uv[:, 0] = -uv[:, 0]
        sa = -sa
    flipped = float((sa <= 0).mean())
    a3 = 0.5 * area2
    ratio = np.log(np.maximum(sa, 1e-30) / np.maximum(a3, 1e-30))
    w = a3 / max(a3.sum(), 1e-30)
    mean = float((ratio * w).sum())
    distortion = float(np.sqrt((w * (ratio - mean) ** 2).sum()))
    out = np.zeros((len(v), 2))
    out[used] = uv
    return out, flipped, distortion


def arap(v, f, uv0, iters=12):
    """As-rigid-as-possible refinement of a disk parameterisation (Liu et al. 2008, local/global).

    LSCM is conformal, so narrow parts (hair strands, fingers, the far side of a long seam) shrink
    exponentially; ARAP pulls every triangle back toward its true size and shape."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    used, lf = np.unique(f, return_inverse=True)
    lf = lf.reshape(-1, 3)
    p = v[used].astype(np.float64)
    u = uv0[used].astype(np.float64).copy()
    nv, nf = len(p), len(lf)
    p0, p1, p2 = p[lf[:, 0]], p[lf[:, 1]], p[lf[:, 2]]
    e1 = p1 - p0
    n = np.cross(e1, p2 - p0)
    area2 = np.linalg.norm(n, axis=1)
    x_ax = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    y_ax = np.cross(n / np.maximum(area2[:, None], 1e-20), x_ax)
    q = np.zeros((nf, 3, 2))
    q[:, 1, 0] = np.linalg.norm(e1, axis=1)
    q[:, 2, 0] = np.einsum("ij,ij->i", p2 - p0, x_ax)
    q[:, 2, 1] = np.einsum("ij,ij->i", p2 - p0, y_ax)
    # cotangent of the angle opposite each edge (i -> j), edge k = (k, k+1), opposite corner k+2
    pairs = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    cot = np.zeros((nf, 3))
    for k, (i, j, o) in enumerate(pairs):
        a = q[:, i] - q[:, o]
        b = q[:, j] - q[:, o]
        cr = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        cot[:, k] = (a * b).sum(1) / np.maximum(cr, 1e-20)
    scale = max(float(np.median(np.abs(cot))), 1e-6)
    cot = np.clip(cot, 1e-3 * scale, 1e3 * scale)
    rows, cols, vals = [], [], []
    for k, (i, j, _) in enumerate(pairs):
        vi, vj, w = lf[:, i], lf[:, j], cot[:, k]
        rows += [vi, vj, vi, vj]
        cols += [vi, vj, vj, vi]
        vals += [w, w, -w, -w]
    L = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nv, nv)).tocsc()
    pin = 0
    keep = np.arange(nv) != pin
    try:
        solver = splu(L[keep][:, keep].tocsc())
    except Exception:
        return uv0
    Lp = L[keep][:, [pin]]
    for _ in range(iters):
        # local: best rotation per triangle
        S = np.zeros((nf, 2, 2))
        for k, (i, j, _) in enumerate(pairs):
            du = u[lf[:, i]] - u[lf[:, j]]
            dx = q[:, i] - q[:, j]
            S += cot[:, k, None, None] * du[:, :, None] * dx[:, None, :]
        U, _, Vt = np.linalg.svd(S)
        R = U @ Vt
        bad = np.linalg.det(R) < 0
        if bad.any():
            U[bad, :, 1] *= -1
            R[bad] = U[bad] @ Vt[bad]
        # global: Poisson solve
        b = np.zeros((nv, 2))
        for k, (i, j, _) in enumerate(pairs):
            dx = q[:, i] - q[:, j]
            r = cot[:, k, None] * np.einsum("tab,tb->ta", R, dx)
            np.add.at(b, lf[:, i], r)
            np.add.at(b, lf[:, j], -r)
        rhs = b[keep] - Lp @ u[[pin]]
        new = np.zeros_like(u)
        new[pin] = u[pin]
        new[keep] = np.stack([solver.solve(rhs[:, 0]), solver.solve(rhs[:, 1])], 1)
        if not np.all(np.isfinite(new)):
            break
        u = new
    out = np.zeros((len(v), 2))
    out[used] = u
    return out


def _self_overlap(v, f, uv):
    """Area fraction of faces whose UV centroid falls inside another face of the same island."""
    used, lf = np.unique(f, return_inverse=True)
    lf = lf.reshape(-1, 3)
    t = uv[used][lf]                      # [n,3,2]
    p = v[used]
    a3 = 0.5 * np.linalg.norm(np.cross(p[lf[:, 1]] - p[lf[:, 0]], p[lf[:, 2]] - p[lf[:, 0]]), axis=1)
    c = t.mean(1)
    lo, hi = t.min(1), t.max(1)
    n = len(lf)
    hit = np.zeros(n, bool)
    a, b, cc = t[:, 0], t[:, 1], t[:, 2]
    det = (b[:, 1] - cc[:, 1]) * (a[:, 0] - cc[:, 0]) + (cc[:, 0] - b[:, 0]) * (a[:, 1] - cc[:, 1])
    okt = np.abs(det) > 1e-18
    for s0 in range(0, n, 512):
        q = c[s0:s0 + 512]                                   # [m,2]
        inb = ((q[:, None, 0] >= lo[None, :, 0]) & (q[:, None, 0] <= hi[None, :, 0]) &
               (q[:, None, 1] >= lo[None, :, 1]) & (q[:, None, 1] <= hi[None, :, 1]) & okt[None])
        inb[np.arange(len(q)), np.arange(s0, s0 + len(q))] = False
        qi, ti = np.nonzero(inb)
        if len(qi) == 0:
            continue
        px, py = q[qi, 0], q[qi, 1]
        l0 = ((b[ti, 1] - cc[ti, 1]) * (px - cc[ti, 0]) + (cc[ti, 0] - b[ti, 0]) * (py - cc[ti, 1])) / det[ti]
        l1 = ((cc[ti, 1] - a[ti, 1]) * (px - cc[ti, 0]) + (a[ti, 0] - cc[ti, 0]) * (py - cc[ti, 1])) / det[ti]
        l2 = 1.0 - l0 - l1
        inside = (l0 > 1e-9) & (l1 > 1e-9) & (l2 > 1e-9)
        hit[s0 + qi[inside]] = True
    return float(a3[hit].sum() / max(a3.sum(), 1e-30))


def _uv_quality(v, f, uv):
    """(flipped fraction by area, area distortion: weighted std of log(uv area / 3D area))."""
    used, lf = np.unique(f, return_inverse=True)
    lf = lf.reshape(-1, 3)
    p = v[used]
    a3 = 0.5 * np.linalg.norm(np.cross(p[lf[:, 1]] - p[lf[:, 0]], p[lf[:, 2]] - p[lf[:, 0]]), axis=1)
    sa = _signed_uv_area(uv[used], lf)
    w = a3 / max(a3.sum(), 1e-30)
    flipped = float(w[sa <= 0].sum())
    ok = sa > 0
    if not ok.any():
        return 1.0, 1e9
    r = np.log(np.maximum(sa, 1e-30) / np.maximum(a3, 1e-30))
    wo = w * ok
    mean = float((r * wo).sum() / wo.sum())
    return flipped, float(np.sqrt((wo * (r - mean) ** 2).sum() / wo.sum()))


def _signed_uv_area(uv, f):
    a, b, c = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


# ---------------------------------------------------------------------------------------------- body parts

class _Body:
    """Body-part labels per face for a standing character."""

    def __init__(self, v, f, up_axis="auto", head_size=0.2, hands_size=0.1, front_axis="+Z"):
        self.reg = _Regions(v, up_axis, head_size, hands_size)
        r = self.reg
        self.up, self.lat, self.dep = r.up, r.lat, r.depth
        lo, hi = v.min(0), v.max(0)
        self.H = max(float(hi[self.up] - lo[self.up]), 1e-6)
        self.y0 = float(lo[self.up])
        self.cx = float(r.head_center[self.lat])
        self.front = 1.0 if front_axis == "+Z" else -1.0
        y = (v[:, self.up] - self.y0) / self.H
        x = v[:, self.lat] - self.cx
        # torso half width from a slice at belly height, where the arms of a T/A pose are not
        # (per-slice p90 of |x| over several slices; a low percentile of those ignores the
        # slices where arms or hands reach out)
        widths = []
        for yb in np.arange(0.30, 0.62, 0.04):
            band = (y > yb) & (y < yb + 0.04)
            if band.sum() > 10:
                widths.append(float(np.percentile(np.abs(x[band]), 90)))
        self.w_t = float(np.percentile(widths, 25)) if widths else 0.15 * self.H
        self.neck = float((r.head_center[self.up] - self.y0) / self.H) - 0.85 * r.r / self.H
        self.crotch = self._crotch(x, y)

    def _crotch(self, x, y):
        """Highest height where the middle column is empty between the legs."""
        best = 0.45
        tol = 0.012
        for yb in np.arange(0.15, 0.62, 0.01):
            sel = np.abs(y - yb) < tol
            if sel.sum() < 10:
                continue
            xs = x[sel]
            near = np.abs(xs) < 0.12 * self.w_t
            legs = (np.abs(xs) > 0.25 * self.w_t) & (np.abs(xs) < 1.2 * self.w_t)
            if near.sum() == 0 and (xs[legs] < 0).any() and (xs[legs] > 0).any():
                best = float(yb)
        return best

    def labels(self, v, f, adj):
        cent = v[f].mean(1)
        y = (cent[:, self.up] - self.y0) / self.H
        x = cent[:, self.lat] - self.cx
        hands = self.reg.hands(torch.from_numpy(cent).float()).numpy() > 0.5
        lab = np.full(len(f), 2, np.int64)  # torso
        head = y > self.neck
        arm = (~head) & (np.abs(x) > 1.08 * self.w_t) & (y > self.crotch + 0.05)
        leg = (~head) & (~arm) & (y < self.crotch)
        lab[head] = 0
        lab[arm] = np.where(x[arm] < 0, 4, 5)
        lab[arm & hands] = np.where(x[arm & hands] < 0, 6, 7)
        lab[leg] = np.where(x[leg] < 0, 8, 9)
        return _smooth_labels(lab, adj, len(f))

    def split_front_back(self, v, f, faces, part):
        """Second cut that makes each body part a disk: head behind the ears, torso at the sides, limbs along their length."""
        cent = v[f[faces]].mean(1)
        if part == 0:  # head: the face island reaches past the ears; the back island holds the back of the skull
            c = self.reg.head_center.numpy()
            z = (cent[:, self.dep] - c[self.dep]) * self.front
            return faces[z >= -0.35 * self.reg.r], faces[z < -0.35 * self.reg.r]
        if part == 2:
            pts = v[np.unique(f[faces])]
            zc = float(np.median(pts[:, self.dep]))
            z = (cent[:, self.dep] - zc) * self.front
            return faces[z >= 0], faces[z < 0]
        # limbs: plane through the limb axis; arms/hands split top/bottom, legs front/back
        pts = v[np.unique(f[faces])]
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        axis = vt[0]
        depth = np.zeros(3)
        depth[self.dep] = 1.0
        if part in (8, 9):
            n = depth - axis * np.dot(axis, depth)
        else:
            n = np.cross(axis, depth)
        n /= max(np.linalg.norm(n), 1e-12)
        s = (cent - mean) @ n
        return faces[s >= 0], faces[s < 0]


def _smooth_labels(lab, adj, nf, iters=8):
    a, b = adj
    for _ in range(iters):
        votes = np.zeros((nf, int(lab.max()) + 1))
        np.add.at(votes, (a, lab[b]), 1)
        np.add.at(votes, (b, lab[a]), 1)
        best = votes.argmax(1)
        strong = votes[np.arange(nf), best] >= 2
        changed = strong & (best != lab) & (votes[np.arange(nf), lab] <= 1)
        if not changed.any():
            break
        lab = np.where(changed, best, lab)
    return lab


def _split_by_plane(v, f, faces):
    cent = v[f[faces]].mean(1)
    mean = cent.mean(0)
    _, _, vt = np.linalg.svd(cent - mean, full_matrices=False)
    s = (cent - mean) @ vt[0]
    return faces[s >= 0], faces[s < 0]


def _grow_disk(f, faces, adj_a, adj_b, n_faces):
    """Grow one topological disk over a connected patch, face by face from its deepest face.

    A face joins through the edge it shares with the grown region. Its third corner reuses the region's copy
    of that vertex only when the face fills a notch (its second edge is also on the region boundary);
    otherwise the corner gets a new copy, which cuts a seam along the face's other edges. The region stays a
    disk at every step, so holes and handles (hair strands, ears, armpits) become seams instead of forcing the
    patch to be chopped up. Returns (global vertex id per local vertex, local faces [k,3], face ids in order)."""
    import heapq

    inside = np.zeros(n_faces, bool)
    inside[faces] = True
    keep = inside[adj_a] & inside[adj_b]
    a, b = adj_a[keep], adj_b[keep]
    nbr = [[] for _ in range(n_faces)]
    for x, y in zip(a.tolist(), b.tolist()):
        nbr[x].append(y)
        nbr[y].append(x)
    # depth from the patch boundary: seed in the middle so the cuts end up near the edges
    deg = np.zeros(n_faces, np.int64)
    np.add.at(deg, a, 1)
    np.add.at(deg, b, 1)
    depth = {int(t): 0 for t in faces if deg[t] < 3}
    frontier = list(depth)
    if not frontier:
        frontier = [int(faces[0])]
        depth[frontier[0]] = 0
    while frontier:
        nxt = []
        for t in frontier:
            for u in nbr[t]:
                if u not in depth:
                    depth[u] = depth[t] + 1
                    nxt.append(u)
        frontier = nxt
    seed = max(depth, key=depth.get)

    gid = []          # global vertex of each local vertex
    copies = {}       # global vertex -> local copies
    lf = {}           # face -> local corners
    ecount = {}       # local undirected edge -> faces using it

    def new_copy(g):
        gid.append(g)
        copies.setdefault(g, []).append(len(gid) - 1)
        return len(gid) - 1

    def add(t, corners):
        lf[t] = corners
        for i in range(3):
            e = (min(corners[i], corners[(i + 1) % 3]), max(corners[i], corners[(i + 1) % 3]))
            ecount[e] = ecount.get(e, 0) + 1

    add(seed, [new_copy(int(g)) for g in f[seed]])
    heap, order, tick = [], [seed], 0
    for u in nbr[seed]:
        heapq.heappush(heap, (1, tick, u, seed))
        tick += 1
    dist = {seed: 0}
    def try_add(t, via, d, force):
        g = [int(x) for x in f[t]]
        gv = [int(x) for x in f[via]]
        lv = lf[via]
        # the shared edge, with the region's local copies
        loc = {gv[i]: lv[i] for i in range(3)}
        if sum(x in loc for x in g) != 2:
            return False
        k = next(i for i in range(3) if g[i] not in loc)
        ga, gb = g[(k + 1) % 3], g[(k + 2) % 3]
        la, lb = loc[ga], loc[gb]
        if ecount.get((min(la, lb), max(la, lb)), 0) != 1:
            return False
        gc = g[k]
        lc = None
        for c in copies.get(gc, []):
            e1 = ecount.get((min(lb, c), max(lb, c)), 0)
            e2 = ecount.get((min(c, la), max(c, la)), 0)
            if (e1 == 1) != (e2 == 1) and e1 <= 1 and e2 <= 1:  # notch: exactly one more boundary edge
                lc = c
                break
        if lc is None:
            if copies.get(gc) and not force:
                return None  # would cut a seam: wait, the region may reach it as a notch later
            lc = new_copy(gc)
        corners = [0, 0, 0]
        corners[k], corners[(k + 1) % 3], corners[(k + 2) % 3] = lc, la, lb
        add(t, corners)
        order.append(t)
        return True

    deferred = {}
    while True:
        while heap:
            d, _, t, via = heapq.heappop(heap)
            if t in lf:
                continue
            r = try_add(t, via, d, False)
            if r is None:
                if t not in deferred or deferred[t][0] > d:
                    deferred[t] = (d, via)
                continue
            if r:
                deferred.pop(t, None)
                for u in nbr[t]:
                    if u not in lf:
                        heapq.heappush(heap, (d + 1, tick, u, t))
                        tick += 1
        deferred = {t: dv for t, dv in deferred.items() if t not in lf}
        if not deferred:
            break
        # nothing grows without a cut: cut at the nearest waiting face, then retry the others
        t = min(deferred, key=lambda x: deferred[x][0])
        d, via = deferred.pop(t)
        if try_add(t, via, d, True):
            for u in nbr[t]:
                if u not in lf:
                    heapq.heappush(heap, (d + 1, tick, u, t))
                    tick += 1
        for x, (dx, vx) in deferred.items():
            heapq.heappush(heap, (dx, tick, x, vx))
            tick += 1
        deferred = {}
    order = np.array(order, np.int64)
    return np.array(gid, np.int64), np.array([lf[int(t)] for t in order], np.int64), order


def _compact(gid, lfc):
    used, inv = np.unique(lfc, return_inverse=True)
    return gid[used], inv.reshape(-1, 3)


def _param(v, gid, lfc, limit):
    """LSCM + ARAP of a grown disk. Returns (uv over gid, ok)."""
    vg = v[gid]
    uv, _, _ = lscm(vg, lfc)
    if uv is None:
        return None, False
    uv = arap(vg, lfc, uv)
    flipped, dist = _uv_quality(vg, lfc, uv)
    if flipped > 0.003 or dist > limit:
        return uv, False
    return uv, _self_overlap(vg, lfc, uv) <= 0.003


def _flatten(v, f, faces, adj, n_faces, depth, max_distortion, out, hand=False):
    """Grow a disk over each connected piece and flatten it (LSCM + ARAP).

    If it folds over itself or stretches too much, keep the largest central part of the disk that flattens
    cleanly (the disk is grown from the middle, so every prefix of the growth order is a disk too) and
    flatten the rest again; only when that central part is small is the piece split in two."""
    limit = max_distortion * (2.5 if hand else 1.0)
    for comp in _components(faces, adj[0], adj[1], n_faces):
        if len(comp) < 3:
            out.append((comp, f[comp].reshape(-1).copy(), np.arange(3 * len(comp)).reshape(-1, 3), None))
            continue
        gid, lfc, order = _grow_disk(f, comp, adj[0], adj[1], n_faces)
        uv, good = _param(v, gid, lfc, limit)
        if good or depth >= 6 or len(comp) < 48:
            out.append((order, gid, lfc, uv))
            continue
        # largest clean prefix of the growth order (binary search)
        lo, hi, best = max(len(comp) // 4, 24), len(comp) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            g2, l2 = _compact(gid, lfc[:mid])
            uv2, ok = _param(v, g2, l2, limit)
            if ok:
                best = (mid, g2, l2, uv2)
                lo = mid + 1
            else:
                hi = mid - 1
        if best is not None:
            k, g2, l2, uv2 = best
            out.append((order[:k], g2, l2, uv2))
            _flatten(v, f, order[k:], adj, n_faces, depth + 1, max_distortion, out, hand)
            continue
        a, b = _split_by_plane(v, f, comp)
        if len(a) == 0 or len(b) == 0:
            out.append((order, gid, lfc, uv))
            continue
        _flatten(v, f, a, adj, n_faces, depth + 1, max_distortion, out, hand)
        _flatten(v, f, b, adj, n_faces, depth + 1, max_distortion, out, hand)


def _absorb_small(v, f, adj, islands, n_faces, min_faces=20):
    """Unfold the faces of tiny islands onto a neighbouring big island, one triangle at a time, wherever the
    unfolded triangle lands on free space. Removes the slivers that a noisy mesh leaves after cutting."""
    isl = [[np.asarray(o, np.int64).tolist(), np.asarray(g, np.int64).tolist(),
            np.asarray(l, np.int64).tolist(), None if u is None else np.asarray(u, np.float64).tolist()]
           for o, g, l, u in islands]
    owner = -np.ones(n_faces, np.int64)
    pos = -np.ones(n_faces, np.int64)
    for k, (o, _, _, _) in enumerate(isl):
        owner[o] = k
        pos[o] = np.arange(len(o))
    nbr = [[] for _ in range(n_faces)]
    for x, y in zip(adj[0].tolist(), adj[1].tolist()):
        nbr[x].append(y)
        nbr[y].append(x)
    ecount = {}

    def edges_of(k):
        if k not in ecount:
            d = {}
            for c in isl[k][2]:
                for i in range(3):
                    e = (min(c[i], c[(i + 1) % 3]), max(c[i], c[(i + 1) % 3]))
                    d[e] = d.get(e, 0) + 1
            ecount[k] = d
        return ecount[k]

    def free(k, tri):
        u = np.asarray(isl[k][3])
        t = u[np.asarray(isl[k][2])]
        c = tri.mean(0)
        d1 = tri[1] - tri[0]
        d2 = tri[2] - tri[0]
        if d1[0] * d2[1] - d1[1] * d2[0] <= 0:
            return False
        # new centroid inside an existing triangle, or an existing centroid inside the new one
        def inside(pts, tr):
            a, b, cc = tr[..., 0, :], tr[..., 1, :], tr[..., 2, :]
            det = (b[..., 1] - cc[..., 1]) * (a[..., 0] - cc[..., 0]) + (cc[..., 0] - b[..., 0]) * (a[..., 1] - cc[..., 1])
            det = np.where(np.abs(det) < 1e-30, 1e-30, det)
            l0 = ((b[..., 1] - cc[..., 1]) * (pts[..., 0] - cc[..., 0]) + (cc[..., 0] - b[..., 0]) * (pts[..., 1] - cc[..., 1])) / det
            l1 = ((cc[..., 1] - a[..., 1]) * (pts[..., 0] - cc[..., 0]) + (a[..., 0] - cc[..., 0]) * (pts[..., 1] - cc[..., 1])) / det
            return (l0 > 1e-9) & (l1 > 1e-9) & (1 - l0 - l1 > 1e-9)
        if inside(np.broadcast_to(c, (len(t), 2)), t).any():
            return False
        if inside(t.mean(1), np.broadcast_to(tri, (len(t), 3, 2))).any():
            return False
        # also the edge midpoints of the new triangle
        mids = (tri + np.roll(tri, -1, 0)) / 2
        for m in mids:
            if inside(np.broadcast_to(m, (len(t), 2)), t).any():
                return False
        return True

    for _ in range(4):
        moved = False
        small = [k for k, (o, _, _, u) in enumerate(isl) if 0 < len(o) < min_faces]
        for k in small:
            queue = list(isl[k][0])
            stuck = 0
            while queue and stuck < len(queue):
                t = queue.pop(0)
                placed = False
                for nb in nbr[t]:
                    j = owner[nb]
                    if j == k or j < 0 or len(isl[j][0]) < min_faces or isl[j][3] is None:
                        continue
                    gt = [int(x) for x in f[t]]
                    gn = [int(x) for x in f[nb]]
                    cn = isl[j][2][pos[nb]]
                    loc = {gn[i]: cn[i] for i in range(3)}
                    if sum(x in loc for x in gt) != 2:
                        continue
                    kk = next(i for i in range(3) if gt[i] not in loc)
                    ga, gb = gt[(kk + 1) % 3], gt[(kk + 2) % 3]
                    la, lb = loc[ga], loc[gb]
                    if edges_of(j).get((min(la, lb), max(la, lb)), 0) != 1:
                        continue
                    uvj = isl[j][3]
                    ua, ub = np.asarray(uvj[la]), np.asarray(uvj[lb])
                    pa, pb, pc = v[ga], v[gb], v[gt[kk]]
                    e3 = pb - pa
                    L3 = float(np.linalg.norm(e3))
                    L2 = float(np.linalg.norm(ub - ua))
                    if L3 < 1e-12 or L2 < 1e-12:
                        continue
                    sp = float(np.dot(pc - pa, e3)) / (L3 * L3)
                    h = float(np.linalg.norm(np.cross(e3, pc - pa))) / L3 * (L2 / L3)
                    eu = (ub - ua) / L2
                    perp = np.array([-eu[1], eu[0]])
                    uc = ua + sp * (ub - ua) + h * perp
                    tri = np.zeros((3, 2))
                    tri[kk], tri[(kk + 1) % 3], tri[(kk + 2) % 3] = uc, ua, ub
                    if not free(j, tri):
                        continue
                    # accept: new copy of the third vertex
                    isl[j][1].append(gt[kk])
                    uvj.append(uc.tolist())
                    lc = len(isl[j][1]) - 1
                    corners = [0, 0, 0]
                    corners[kk], corners[(kk + 1) % 3], corners[(kk + 2) % 3] = lc, la, lb
                    isl[j][2].append(corners)
                    isl[j][0].append(t)
                    d = edges_of(j)
                    for i in range(3):
                        e = (min(corners[i], corners[(i + 1) % 3]), max(corners[i], corners[(i + 1) % 3]))
                        d[e] = d.get(e, 0) + 1
                    owner[t] = j
                    pos[t] = len(isl[j][0]) - 1
                    placed = True
                    moved = True
                    break
                if placed:
                    stuck = 0
                else:
                    queue.append(t)
                    stuck += 1
            # whatever is left stays a (smaller) island of its own
            rest = [t for t in isl[k][0] if owner[t] == k]
            if len(rest) != len(isl[k][0]):
                if rest:
                    sub = np.asarray(rest, np.int64)
                    gid = f[sub].reshape(-1)
                    isl[k] = [rest, gid.tolist(), np.arange(3 * len(rest)).reshape(-1, 3).tolist(), None]
                    pos[sub] = np.arange(len(rest))
                else:
                    isl[k] = [[], [], [], None]
                ecount.pop(k, None)
        if not moved:
            break
    out = []
    for o, g, l, u in isl:
        if not o:
            continue
        out.append((np.asarray(o, np.int64), np.asarray(g, np.int64), np.asarray(l, np.int64).reshape(-1, 3),
                    None if u is None else np.asarray(u, np.float64)))
    return out


def _merge_fragments(lab, adj, nf, min_keep=0):
    """Every label keeps only its largest connected piece; the other pieces join the label they touch most.
    Removes the specks and enclosed islets that would make holes in a body-part patch."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a, b = adj
    for _ in range(6):
        same = lab[a] == lab[b]
        g = coo_matrix((np.ones(int(same.sum()), np.int8), (a[same], b[same])), shape=(nf, nf))
        n, comp = connected_components(g, directed=False)
        size = np.bincount(comp, minlength=n)
        clab = np.zeros(n, np.int64)
        clab[comp] = lab
        best = np.zeros(int(lab.max()) + 1, np.int64)
        bestsize = np.zeros(int(lab.max()) + 1, np.int64)
        for c in np.argsort(-size):
            if bestsize[clab[c]] == 0:
                bestsize[clab[c]] = size[c]
                best[clab[c]] = c
        minor = np.ones(n, bool)
        minor[best[bestsize > 0]] = False
        if not minor.any():
            break
        # most common neighbouring label of each minor piece
        cross = comp[a] != comp[b]
        ca, cb = comp[a][cross], comp[b][cross]
        la, lb = lab[a][cross], lab[b][cross]
        votes = {}
        for x, y in ((ca, lb), (cb, la)):
            m = minor[x]
            for c, l in zip(x[m].tolist(), y[m].tolist()):
                votes.setdefault(c, {}).setdefault(l, 0)
                votes[c][l] += 1
        changed = False
        # small pieces first, so an islet merges into what surrounds it
        for c in sorted(votes, key=lambda c: size[c]):
            tgt = max(votes[c], key=votes[c].get)
            if tgt != clab[c]:
                lab[comp == c] = tgt
                changed = True
        if not changed:
            break
    return lab


def character_uv(v, f, resolution=2048, padding=8, head_texel_scale=1.4, hands_texel_scale=1.3,
                 head_size=0.2, hands_size=0.1, up_axis="auto", front_axis="+Z", max_distortion=0.6):
    """Returns (vertices split at seams, faces, uv v-up [0,1], info dict)."""
    v = np.asarray(v, np.float64)
    f = np.asarray(f, np.int64)
    v, f = _weld(v, f, 1e-7)
    nf = len(f)
    adj = _face_adjacency(f)
    body = _Body(v, f, up_axis, head_size, hands_size, front_axis)
    lab = body.labels(v, f, adj)

    # final labels: part x side (A/B), then each label is one connected piece without islets
    full = np.full(nf, -1, np.int64)
    for part in range(len(PART_NAMES)):
        faces = np.nonzero(lab == part)[0]
        if len(faces) == 0:
            continue
        for side, a in enumerate(body.split_front_back(v, f, faces, part)):
            full[a] = 2 * part + side
    # straighten the cut lines (no single-triangle teeth), then one connected piece per label
    full = _merge_fragments(_smooth_labels(full, adj, nf, iters=12), adj, nf)

    islands = []
    for key in np.unique(full):
        faces = np.nonzero(full == key)[0]
        part = int(key) // 2
        _flatten(v, f, faces, adj, nf, 0, max_distortion, islands, hand=PART_NAMES[part].startswith("hand"))
    islands = _absorb_small(v, f, adj, islands, nf)

    # split vertices per island
    new_v, new_f, new_uv = [], np.zeros_like(f), []
    base = 0
    fallback = 0
    for faces, gid, lfc, uv in islands:
        if uv is None:  # degenerate speck: a flat projection is good enough
            fallback += 1
            pts = v[gid]
            c = pts - pts.mean(0)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            uv = c @ vt[:2].T
        new_v.append(v[gid])
        new_uv.append(uv)
        new_f[faces] = lfc + base
        base += len(gid)
    sv = np.concatenate(new_v)
    suv = np.concatenate(new_uv)

    regions = body.reg
    st = torch.from_numpy(sv).float()
    density = regions.density(st, float(head_texel_scale), float(hands_texel_scale)).numpy()
    packed, n_isl, _ = repack_uvs(sv, new_f, suv.astype(np.float32), density, resolution, padding)
    sa = _signed_uv_area(packed.astype(np.float64), new_f)
    info = {
        "islands": int(n_isl),
        "flipped": int((sa <= 0).sum()),
        "fallback": fallback,
        "neck": round(body.neck, 3), "crotch": round(body.crotch, 3),
        "parts": {PART_NAMES[k]: int((lab == k).sum()) for k in range(len(PART_NAMES)) if (lab == k).any()},
    }
    return sv.astype(np.float32), new_f, packed.astype(np.float32), info


class D3DCharacterUV:
    """Replaces UnwrapMesh for characters: few islands, seams where an artist would put them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Low-poly character (after Mesh Cleanup). Triangles or quads stored as triangle pairs."}),
                "resolution": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256,
                                       "tooltip": "Texture size the atlas is packed for (same as the bakes)."}),
                "padding": ("INT", {"default": 8, "min": 1, "max": 64,
                                    "tooltip": "Texels between islands. 8 at 2048 leaves room for Texture Padding and mipmaps."}),
                "head_texel_scale": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 6.0, "step": 0.1,
                                               "tooltip": "Texel density on the head relative to the body."}),
                "hands_texel_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 6.0, "step": 0.1}),
                "head_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01}),
                "hands_size": ("FLOAT", {"default": 0.1, "min": 0.02, "max": 0.3, "step": 0.01}),
                "front_axis": (["+Z", "-Z"], {"default": "+Z",
                                              "tooltip": "Which way the character faces. TRELLIS / Pixal3D characters face +Z."}),
                "max_distortion": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 3.0, "step": 0.05,
                                             "tooltip": "Area stretch allowed per island (log-area std). Lower = more, smaller islands."}),
                "up_axis": (UP_AXES, {"default": "auto"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "unwrap"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("Character UVs with seams by body part: rings at neck, shoulders, wrists, waist and hips, the head "
                   "split behind the ears so the face is one island, limbs split along their length. Each island is "
                   "flattened with LSCM (split again if it stretches) and packed with ComfyUI's packer.")

    def unwrap(self, mesh, resolution, padding, head_texel_scale, hands_texel_scale, head_size, hands_size,
               front_axis, max_distortion, up_axis, unique_id=None):
        from comfy_api.latest import Types

        n, item = _batch_items(mesh)
        if n != 1:
            logging.warning("[D3DCharacterUV] batch of meshes: only the first one is unwrapped")
        v, f = item(0)[:2]
        sv, sf, uv, info = character_uv(v.detach().cpu().numpy(), f.detach().cpu().numpy(), resolution, padding,
                                        head_texel_scale, hands_texel_scale, head_size, hands_size, up_axis,
                                        front_axis, max_distortion)
        uv_gltf = uv.copy()
        uv_gltf[:, 1] = 1.0 - uv_gltf[:, 1]
        out = Types.MESH(vertices=torch.from_numpy(sv)[None], faces=torch.from_numpy(sf)[None],
                         uvs=torch.from_numpy(uv_gltf)[None])
        text = (f"{info['islands']} islands, {info['flipped']} flipped faces | neck at {info['neck']:.2f}, "
                f"crotch at {info['crotch']:.2f} of the height | " +
                ", ".join(f"{k} {c}" for k, c in info["parts"].items()))
        logging.info(f"[D3DCharacterUV] {text}")
        _progress_text(unique_id, text)
        return (out, text)


# ---------------------------------------------------------------------------------------------- raster helpers

def _raster(uv_px, f, H, W, values=None):
    """Rasterize triangles in pixel space. Returns coverage mask [H,W] and, with values [V,C], interpolated values."""
    mask = np.zeros((H, W), bool)
    out = None if values is None else np.zeros((H, W, values.shape[1]), np.float32)
    tri = uv_px[f]
    x0 = np.clip(np.floor(tri[:, :, 0].min(1)).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil(tri[:, :, 0].max(1)).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(tri[:, :, 1].min(1)).astype(int), 0, H - 1)
    y1 = np.clip(np.ceil(tri[:, :, 1].max(1)).astype(int), 0, H - 1)
    for t in range(len(f)):
        xs = np.arange(x0[t], x1[t] + 1) + 0.5
        ys = np.arange(y0[t], y1[t] + 1) + 0.5
        if len(xs) == 0 or len(ys) == 0:
            continue
        gx, gy = np.meshgrid(xs, ys)
        (ax, ay), (bx, by), (cx, cy) = tri[t]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / det
        l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / det
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        yy, xx = np.nonzero(inside)
        yy += y0[t]
        xx += x0[t]
        mask[yy, xx] = True
        if out is not None:
            vals = values[f[t]]
            out[yy, xx] = (l0[inside][:, None] * vals[0] + l1[inside][:, None] * vals[1] + l2[inside][:, None] * vals[2])
    return mask, out


def _mesh_uv_for_images(mesh):
    """UVs exactly as the native bakes and ApplyTextureToMesh use them (glTF v-down, fitted into [0,1])."""
    from comfy_extras.nodes_mesh_postprocess import _normalize_uvs_to_unit

    n, item = _batch_items(mesh)
    v, f, _, uv, _ = item(0)
    if uv is None:
        raise ValueError("mesh has no UVs")
    uv = _normalize_uvs_to_unit(uv.detach().cpu().numpy())
    return v.detach().float().cpu().numpy(), f.detach().long().cpu().numpy(), uv


def _dilate(img, mask, pixels):
    """Copy the nearest covered texel into uncovered texels up to `pixels` away."""
    from scipy import ndimage

    if pixels <= 0 or mask.all() or not mask.any():
        return img
    dist, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    fill = (~mask) & (dist <= pixels)
    out = img.copy()
    out[fill] = img[iy[fill], ix[fill]]
    return out


class D3DTexturePadding:
    """Pad every texture of a textured MESH: texels around each UV island take the nearest island colour."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Mesh after ApplyTextureToMesh."}),
                "pixels": ("INT", {"default": 16, "min": 1, "max": 256,
                                   "tooltip": "How far to grow each island, in texels of the largest map."}),
                "edge_texels": ("INT", {"default": 1, "min": 0, "max": 4,
                                        "tooltip": "Also replace this many texels inside the island border, which often hold a mix of background and colour."}),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "pad"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("Dilates base color, metallic/roughness/AO and normal maps past the UV island borders, so seams do "
                   "not show dark lines in engines (bilinear filtering, mipmaps).")

    def pad(self, mesh, pixels, edge_texels):
        from scipy import ndimage

        v, f, uv = _mesh_uv_for_images(mesh)
        out = copy.copy(mesh)
        maps = [(name, getattr(mesh, name, None)) for name in ("texture", "metallic_roughness", "normal_map", "emissive")]
        maps = [(name, t) for name, t in maps if t is not None]
        if not maps:
            return (out,)
        max_w = max(int(t.shape[2]) for _, t in maps)
        cache = {}
        for name, t in maps:
            arr = t.detach().float().cpu().numpy()
            B, H, W = arr.shape[0], arr.shape[1], arr.shape[2]
            if (H, W) not in cache:
                px = np.stack([uv[:, 0] * W, uv[:, 1] * H], axis=1)
                mask, _ = _raster(px, f, H, W)
                if edge_texels > 0:
                    inner = ndimage.binary_erosion(mask, iterations=int(edge_texels), border_value=0)
                    # islands thinner than the erosion keep their own texels
                    far = ndimage.distance_transform_edt(~inner) > edge_texels + 1.5 if inner.any() else np.ones_like(mask)
                    mask = inner | (mask & far)
                cache[(H, W)] = mask
            mask = cache[(H, W)]
            grow = int(round(pixels * W / max_w)) + int(edge_texels)
            res = np.stack([_dilate(arr[k], mask, grow) for k in range(B)])
            setattr(out, name, torch.from_numpy(res).to(t.dtype))
        return (out,)


class D3DAlbedoTopBlend:
    """Top of the head from a second albedo (TRELLIS), the rest from the main albedo (QwenTex)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "The same UV mesh that both albedos were baked on."}),
                "albedo": ("IMAGE", {"tooltip": "Main albedo (QwenTex)."}),
                "top_albedo": ("IMAGE", {"tooltip": "Albedo for the top of the head (BakeTextureFromVoxel base color)."}),
                "start_angle": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 90.0, "step": 1.0,
                                          "tooltip": "Surfaces facing up more than this (degrees above horizontal) on the head use top_albedo."}),
                "softness": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 60.0, "step": 1.0,
                                       "tooltip": "Blend width in degrees."}),
                "match_color": ("BOOLEAN", {"default": True,
                                            "tooltip": "Shift top_albedo to the mean and contrast of the main albedo on the same area of hair, so the join does not show."}),
                "head_size": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01}),
                "up_axis": (UP_AXES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("albedo", "mask")
    FUNCTION = "blend"
    CATEGORY = "Trellis2Wrapper/lowpoly"
    DESCRIPTION = ("Fixes the top of the head, which the six QwenTex views barely see: uses top_albedo there, colour "
                   "matched and blended by how much the surface faces up.")

    def blend(self, mesh, albedo, top_albedo, start_angle, softness, match_color, head_size, up_axis):
        v, f, uv = _mesh_uv_for_images(mesh)
        regions = _Regions(v, up_axis, head_size, 0.1)
        vt = torch.from_numpy(v).float()
        head = regions.head(vt).numpy()
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        vn = np.zeros_like(v)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-20)
        elev = np.degrees(np.arcsin(np.clip(vn[:, regions.up], -1.0, 1.0)))
        t = np.clip((elev - (start_angle - softness * 0.5)) / softness, 0.0, 1.0)
        w = (t * t * (3 - 2 * t)) * head
        a = albedo.detach().float().cpu().numpy()
        b = top_albedo.detach().float().cpu().numpy()
        H, W = a.shape[1], a.shape[2]
        if b.shape[1:3] != a.shape[1:3]:
            b = torch.nn.functional.interpolate(torch.from_numpy(b).permute(0, 3, 1, 2), size=(H, W), mode="bilinear",
                                                align_corners=False).permute(0, 2, 3, 1).numpy()
        px = np.stack([uv[:, 0] * W, uv[:, 1] * H], axis=1)
        cover, mw = _raster(px, f, H, W, w[:, None].astype(np.float32))
        mw = _dilate(mw, cover, 8)[..., 0]
        out = a.copy()
        for i in range(a.shape[0]):
            bi = b[min(i, b.shape[0] - 1)][..., :3].copy()
            if match_color:
                sel = mw > 0.05
                if sel.sum() > 50:
                    ma, sa = a[i][..., :3][sel].mean(0), a[i][..., :3][sel].std(0)
                    mb, sb = bi[sel].mean(0), bi[sel].std(0)
                    bi = np.clip((bi - mb) * (sa / np.maximum(sb, 1e-4)) + ma, 0.0, 1.0)
            out[i][..., :3] = a[i][..., :3] * (1 - mw[..., None]) + bi * mw[..., None]
        return (torch.from_numpy(out).to(albedo.dtype), torch.from_numpy(mw[None].astype(np.float32)))


NODE_CLASS_MAPPINGS = {
    "D3DCharacterUV": D3DCharacterUV,
    "D3DTexturePadding": D3DTexturePadding,
    "D3DAlbedoTopBlend": D3DAlbedoTopBlend,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "D3DCharacterUV": "D3D - Character UV (seams by body part)",
    "D3DTexturePadding": "D3D - Texture Padding (dilate islands)",
    "D3DAlbedoTopBlend": "D3D - Albedo Top Blend (top of head)",
}
