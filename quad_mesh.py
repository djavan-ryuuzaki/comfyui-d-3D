"""Quad retopology (Instant Meshes) and quad-preserving OBJ export.

Quad meshes travel through the rest of the pack as ordinary triangle meshes:
quad i is stored as the two consecutive triangles 2*i = (a, b, c) and
2*i + 1 = (a, c, d). Trellis2 - Mesh Texturing keeps face and vertex order when
the mesh already has UVs, so the export node can merge the pairs back into quads.
"""
import os

import numpy as np


def remesh_quads(vertices, faces, target_quads, crease_angle=0.0, smooth_iterations=2,
                 align_to_boundaries=False, deterministic=True):
    """Retopologize a triangle mesh into a pure quad mesh with Instant Meshes.

    Returns (vertices [V,3] float32, quads [Q,4] int64).
    """
    import pynanoinstantmeshes as pim

    # Instant Meshes extracts a quad-dominant mesh and then subdivides it once into
    # a pure quad mesh, which multiplies the face count by about 4.
    vertex_count = max(int(target_quads) // 4, 16)
    out_vertices, out_quads = pim.remesh(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
        vertex_count,
        rosy=4,
        posy=4,
        creaseAngle=float(crease_angle),
        align_to_boundaries=bool(align_to_boundaries),
        smooth_iter=int(smooth_iterations),
        deterministic=bool(deterministic),
    )
    out_vertices = np.array(out_vertices, dtype=np.float32)
    out_quads = np.array(out_quads, dtype=np.int64)
    if out_quads.ndim != 2 or out_quads.shape[1] != 4:
        raise Exception(f'Instant Meshes returned faces with shape {out_quads.shape}, expected quads')
    return out_vertices, out_quads


def quads_to_triangle_pairs(quads):
    """(a, b, c, d) -> (a, b, c), (a, c, d), stored as consecutive triangles."""
    return np.stack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=1).reshape(-1, 3)


def _best_diagonal(vertices, quads):
    """Start each quad at the corner whose diagonal gives the fatter pair of triangles.

    xatlas tends to put a near-degenerate triangle in its own chart, which would
    cut the quad along its diagonal.
    """
    def min_area(q):
        p = vertices[q]
        t1 = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        t2 = np.linalg.norm(np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=1)
        return np.minimum(t1, t2)

    rotated = quads[:, [1, 2, 3, 0]]
    use_rotated = min_area(rotated) > min_area(quads)
    return np.where(use_rotated[:, None], rotated, quads)


def _split_quads(tri_uv_idx):
    t0 = tri_uv_idx[0::2]
    t1 = tri_uv_idx[1::2]
    return (t0[:, 0] != t1[:, 0]) | (t0[:, 2] != t1[:, 1])


def _run_xatlas(positions, triangles, uvs, resolution, padding):
    import xatlas

    atlas = xatlas.Atlas()
    if uvs is None:
        atlas.add_mesh(positions, triangles)
        chart_options = xatlas.ChartOptions()
    else:
        atlas.add_mesh(positions, triangles, uvs=uvs)
        chart_options = xatlas.ChartOptions()
        chart_options.use_input_mesh_uvs = True
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    pack_options.padding = int(padding)
    pack_options.bilinear = True
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmap, indices, out_uvs = atlas[0]
    vmap = np.asarray(vmap, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.shape[0] != triangles.shape[0] or not np.array_equal(vmap[indices], triangles):
        raise Exception('xatlas changed the face order; cannot keep the quads together')
    return vmap, indices, np.asarray(out_uvs, dtype=np.float32)


def unwrap_quads(vertices, quads, resolution=2048, padding=2, max_rounds=4):
    """UV-unwrap a quad mesh so that no UV seam runs across a quad's diagonal.

    xatlas only knows triangles, so after a first pass a few quads end up with
    their two triangles in different charts. Each of those triangles is unfolded
    into the chart of its partner, and another xatlas pass repacks the charts.
    xatlas may still split a few charts again, so this repeats a few rounds.

    Returns (vertices [N,3], triangles [2Q,3], uvs [N,2], fixed, remaining) where
    vertices are duplicated along UV seams, triangles come in quad pairs, and
    remaining counts quads still split by a seam (exported as two triangles).
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    quads = _best_diagonal(vertices, quads)
    triangles = quads_to_triangle_pairs(quads).astype(np.uint32)

    vmap, tri_uv, uvs = _run_xatlas(vertices, triangles, None, resolution, padding)

    fixed = 0
    for _ in range(max_rounds):
        split = np.nonzero(_split_quads(tri_uv))[0]
        if len(split) == 0:
            break
        fixed += len(split)
        t0 = tri_uv[2 * split]
        a, b, c = vertices[vmap[t0[:, 0]]], vertices[vmap[t0[:, 1]]], vertices[vmap[t0[:, 2]]]
        d = vertices[quads[split, 3]]
        ua, ub, uc = uvs[t0[:, 0]], uvs[t0[:, 1]], uvs[t0[:, 2]]

        # Lay triangle (a, c, d) flat next to the edge a-c of triangle (a, b, c),
        # on the side opposite to b, keeping its 3D proportions.
        e = c - a
        e_len2 = np.maximum((e * e).sum(axis=1), 1e-20)
        along = ((d - a) * e).sum(axis=1) / e_len2
        height = np.linalg.norm((d - a) - along[:, None] * e, axis=1) / np.sqrt(e_len2)
        eu = uc - ua
        perp = np.stack([-eu[:, 1], eu[:, 0]], axis=1)
        b_side = eu[:, 0] * (ub - ua)[:, 1] - eu[:, 1] * (ub - ua)[:, 0]
        side = np.where(b_side > 0, -1.0, 1.0)
        ud = ua + along[:, None] * eu + (side * height)[:, None] * perp

        new_ids = np.arange(len(uvs), len(uvs) + len(split))
        vmap = np.concatenate([vmap, quads[split, 3]])
        uvs = np.concatenate([uvs, ud.astype(np.float32)])
        tri_uv[2 * split + 1] = np.stack([t0[:, 0], t0[:, 2], new_ids], axis=1)

        # Repack the charts as they now are in UV space.
        vmap2, tri_uv, uvs = _run_xatlas(
            np.ascontiguousarray(vertices[vmap]),
            np.ascontiguousarray(tri_uv, dtype=np.uint32),
            np.ascontiguousarray(uvs),
            resolution, padding,
        )
        vmap = vmap[vmap2]

    remaining = int(_split_quads(tri_uv).sum())
    return vertices[vmap], tri_uv, uvs, fixed, remaining


def triangle_pairs_to_faces(faces):
    """Merge consecutive triangle pairs back into quads.

    Returns a list of faces (each a tuple of 3 or 4 indices) and the number of
    pairs that did not look like a quad split and were kept as triangles.
    """
    faces = np.asarray(faces, dtype=np.int64)
    n_pairs = len(faces) // 2
    t0 = faces[0:2 * n_pairs:2]
    t1 = faces[1:2 * n_pairs:2]
    is_quad = (t0[:, 0] == t1[:, 0]) & (t0[:, 2] == t1[:, 1])

    out = []
    for i in range(n_pairs):
        if is_quad[i]:
            out.append((t0[i, 0], t0[i, 1], t0[i, 2], t1[i, 2]))
        else:
            out.append(tuple(t0[i]))
            out.append(tuple(t1[i]))
    if len(faces) % 2:
        out.append(tuple(faces[-1]))
    return out, int(n_pairs - is_quad.sum())


def _corners(face):
    # A quad with a repeated corner collapses into a triangle.
    seen = []
    for v in face:
        if v not in seen:
            seen.append(v)
    return seen


def write_quad_obj(mesh, obj_path, material_name):
    """Write a trimesh.Trimesh as an OBJ, merging triangle pairs into quads.

    Positions are welded, so UV seams do not cut the mesh apart. Textures from
    Trellis2 - Mesh Texturing are written next to the OBJ and referenced by an MTL
    (base color as map_Kd, metallic/roughness with the PBR extension).
    Returns (quad count, triangle count).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    positions, v_index = np.unique(vertices, axis=0, return_inverse=True)
    v_index = v_index.reshape(-1)

    visual = getattr(mesh, 'visual', None)
    uvs = getattr(visual, 'uv', None) if visual is not None and visual.kind == 'texture' else None
    material = getattr(visual, 'material', None) if uvs is not None else None
    base_color = None
    metallic_roughness = None
    if material is not None:
        base_color = getattr(material, 'baseColorTexture', None)
        if base_color is None:
            base_color = getattr(material, 'image', None)
        metallic_roughness = getattr(material, 'metallicRoughnessTexture', None)
    # trimesh gives untextured UV meshes a 2x2 placeholder image.
    if base_color is not None and max(base_color.size) <= 2:
        base_color = None
    colors = None
    if visual is not None and visual.kind == 'vertex':
        vertex_colors = np.asarray(visual.vertex_colors)[:, :3] / 255.0
        colors = np.zeros((len(positions), 3))
        colors[v_index] = vertex_colors

    if uvs is not None:
        uvs = np.asarray(uvs, dtype=np.float64)
        uv_values, uv_index = np.unique(uvs, axis=0, return_inverse=True)
        uv_index = uv_index.reshape(-1)

    merged, kept_triangles = triangle_pairs_to_faces(faces)

    base = os.path.splitext(os.path.basename(obj_path))[0]
    folder = os.path.dirname(obj_path)
    mtl_file = None
    if base_color is not None or metallic_roughness is not None:
        mtl_file = f'{base}.mtl'
        lines = [f'newmtl {material_name}', 'Kd 1.000000 1.000000 1.000000', 'Ka 0.000000 0.000000 0.000000',
                 'Ks 0.000000 0.000000 0.000000', 'illum 1']
        if base_color is not None:
            base_color.save(os.path.join(folder, f'{base}_basecolor.png'))
            lines.append(f'map_Kd {base}_basecolor.png')
        if metallic_roughness is not None:
            mr = np.asarray(metallic_roughness.convert('RGB'))
            from PIL import Image
            Image.fromarray(mr[..., 1]).save(os.path.join(folder, f'{base}_roughness.png'))
            Image.fromarray(mr[..., 2]).save(os.path.join(folder, f'{base}_metallic.png'))
            lines.append(f'map_Pr {base}_roughness.png')
            lines.append(f'map_Pm {base}_metallic.png')
        with open(os.path.join(folder, mtl_file), 'w') as f:
            f.write('\n'.join(lines) + '\n')

    quad_count = 0
    triangle_count = 0
    with open(obj_path, 'w') as f:
        f.write('# Quad mesh exported by comfyui-d-3D\n')
        if mtl_file is not None:
            f.write(f'mtllib {mtl_file}\n')
        if colors is not None:
            f.write(''.join(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n'
                            for p, c in zip(positions, colors)))
        else:
            f.write(''.join(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n' for p in positions))
        if uvs is not None:
            f.write(''.join(f'vt {t[0]:.6f} {t[1]:.6f}\n' for t in uv_values))
        if mtl_file is not None:
            f.write(f'usemtl {material_name}\n')
        f.write('s 1\n')
        rows = []
        for face in merged:
            corners = _corners(face)
            if len(corners) < 3:
                continue
            if len(corners) == 4:
                quad_count += 1
            else:
                triangle_count += 1
            if uvs is not None:
                rows.append('f ' + ' '.join(f'{v_index[v] + 1}/{uv_index[v] + 1}' for v in corners) + '\n')
            else:
                rows.append('f ' + ' '.join(f'{v_index[v] + 1}' for v in corners) + '\n')
        f.write(''.join(rows))

    if kept_triangles:
        print(f'Export Quad Mesh: {kept_triangles} triangle pairs were not quads (was the mesh changed after Quad Remesh?)')
    return quad_count, triangle_count
