"""UV layout export and painted texture re-application.

Used by the "Trellis2 - UV Layout (for painting)" and "Trellis2 - Apply Painted
Texture" nodes. Pure numpy / OpenCV / PIL so it can be tested without a GPU.

Conventions: trimesh UVs have v pointing up (OpenGL), image row 0 is the top,
so a UV (u, v) lands on pixel (u * W, (1 - v) * H). This matches how trimesh
and the GLB exporter sample the textures written by Trellis2 - Mesh Texturing.
"""
import numpy as np
import cv2
from PIL import Image

_SHIFT = 4  # sub-pixel bits for OpenCV drawing
_SCALE = 1 << _SHIFT


def parse_color(value, default=(0, 255, 0)):
    """'#RRGGBB', 'RRGGBB' or 'r,g,b' -> (r, g, b)."""
    s = str(value).strip()
    try:
        if ',' in s:
            parts = [int(float(p)) for p in s.split(',')[:3]]
            return tuple(int(np.clip(p, 0, 255)) for p in parts)
        s = s.lstrip('#')
        if len(s) == 3:
            s = ''.join(c * 2 for c in s)
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def mesh_uvs(mesh):
    visual = getattr(mesh, 'visual', None)
    uv = getattr(visual, 'uv', None) if visual is not None and visual.kind == 'texture' else None
    if uv is None or len(uv) != len(mesh.vertices):
        raise Exception('This mesh has no UVs. Unwrap it first (Trellis2 - UnWrap Trimesh, '
                        'UV Unwrap and Rasterize, Mesh Texturing or Quad Remesh).')
    return np.asarray(uv, dtype=np.float64)


def mesh_textures(mesh):
    """Return (base_color PIL or None, metallic_roughness PIL or None, material or None)."""
    visual = getattr(mesh, 'visual', None)
    material = getattr(visual, 'material', None) if visual is not None and visual.kind == 'texture' else None
    if material is None:
        return None, None, None
    base = getattr(material, 'baseColorTexture', None)
    if base is None:
        base = getattr(material, 'image', None)
    # trimesh gives untextured UV meshes a tiny placeholder image.
    if base is not None and max(base.size) <= 2:
        base = None
    mr = getattr(material, 'metallicRoughnessTexture', None)
    return base, mr, material


def _uv_to_px(uv, width, height):
    px = np.empty_like(uv)
    px[:, 0] = uv[:, 0] * width
    px[:, 1] = (1.0 - uv[:, 1]) * height
    return px


def _unique_edges(edges, n_vertices, return_counts=False):
    e = np.sort(edges, axis=1)
    key = e[:, 0] * np.int64(n_vertices) + e[:, 1]
    res = np.unique(key, return_counts=return_counts)
    key = res[0] if return_counts else res
    out = np.stack([key // n_vertices, key % n_vertices], axis=1)
    return (out, res[1]) if return_counts else out


def uv_edges(faces, show_quads):
    """Return (all_edges, boundary_edges) as (N, 2) index arrays in UV-vertex space."""
    faces = np.asarray(faces, dtype=np.int64)
    n = int(faces.max()) + 1 if len(faces) else 1
    # Boundaries come from the triangles: an edge used once is an island border (seam).
    tri_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    uniq, counts = _unique_edges(tri_edges, n, return_counts=True)
    boundary = uniq[counts == 1]
    if not show_quads:
        return uniq, boundary

    # Quads are stored as consecutive triangles (a, b, c), (a, c, d): drop the a-c diagonal.
    n_pairs = len(faces) // 2
    t0 = faces[0:2 * n_pairs:2]
    t1 = faces[1:2 * n_pairs:2]
    is_quad = (t0[:, 0] == t1[:, 0]) & (t0[:, 2] == t1[:, 1])
    q0, q1 = t0[is_quad], t1[is_quad]
    quad_edges = np.concatenate([q0[:, [0, 1]], q0[:, [1, 2]], q1[:, [1, 2]], q1[:, [2, 0]]], axis=0)
    rest = np.concatenate([t0[~is_quad], t1[~is_quad], faces[2 * n_pairs:]], axis=0)
    rest_edges = np.concatenate([rest[:, [0, 1]], rest[:, [1, 2]], rest[:, [2, 0]]], axis=0)
    all_edges = _unique_edges(np.concatenate([quad_edges, rest_edges], axis=0), n)
    return all_edges, boundary


def rasterize_islands(uv, faces, width, height, grow=1):
    """uint8 mask (255 inside a UV island) at width x height."""
    px = _uv_to_px(uv, width, height)
    tris = np.round(px[np.asarray(faces, dtype=np.int64)]).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    # drawContours fills every contour on its own, so overlaps never cancel out.
    chunk = 200000
    for i in range(0, len(tris), chunk):
        cv2.drawContours(mask, list(tris[i:i + chunk].reshape(-1, 3, 1, 2)), -1, 255, thickness=-1)
    if grow > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=grow)
    return mask


def draw_edges(uv, edges, width, height, thickness):
    """Anti-aliased line mask (float 0..1) for the given edges."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    if len(edges) == 0:
        return canvas.astype(np.float32)
    px = np.round(_uv_to_px(uv, width, height) * _SCALE).astype(np.int32)
    segs = px[edges]  # (N, 2, 2)
    chunk = 200000
    for i in range(0, len(segs), chunk):
        cv2.polylines(canvas, list(segs[i:i + chunk].reshape(-1, 2, 1, 2)), False, 255,
                      thickness=max(1, int(thickness)), lineType=cv2.LINE_AA, shift=_SHIFT)
    return canvas.astype(np.float32) / 255.0


def checker(width, height, cells=16):
    cell = max(1, min(width, height) // cells)
    yy, xx = np.mgrid[0:height, 0:width]
    c = ((xx // cell + yy // cell) % 2).astype(np.float32)
    v = (0.55 + 0.25 * c)[..., None]
    return np.repeat(v, 3, axis=2)


def build_uv_layout(mesh, resolution, background, line_color, seam_color, line_thickness,
                    seam_thickness, line_opacity, show_quads, dim_outside):
    """Return dict of float32 arrays in 0..1:
    layout (H,W,3), texture (H,W,3), lines (H,W) [all edges + seams], islands (H,W),
    and the PIL base color / metallic-roughness of the mesh (or None)."""
    uv = mesh_uvs(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    base, mr, _ = mesh_textures(mesh)

    if resolution <= 0:
        width, height = base.size if base is not None else (2048, 2048)
    else:
        width = height = int(resolution)

    if base is not None:
        tex = np.asarray(base.convert('RGB').resize((width, height), Image.LANCZOS), dtype=np.float32) / 255.0
    else:
        tex = None

    if background == 'texture' and tex is not None:
        bg = tex.copy()
    elif background == 'white':
        bg = np.ones((height, width, 3), np.float32)
    elif background == 'black':
        bg = np.zeros((height, width, 3), np.float32)
    elif background == 'gray':
        bg = np.full((height, width, 3), 0.5, np.float32)
    else:  # checker, or 'texture' without a texture
        bg = checker(width, height)

    islands = rasterize_islands(uv, faces, width, height).astype(np.float32) / 255.0

    layout = bg
    if dim_outside > 0:
        layout = layout * (1.0 - dim_outside * (1.0 - islands[..., None]))

    all_edges, boundary = uv_edges(faces, show_quads)
    lines = np.zeros((height, width), np.float32)
    if line_thickness > 0:  # 0 = only the seams
        lines = draw_edges(uv, all_edges, width, height, line_thickness) * line_opacity
        col = np.asarray(line_color, np.float32) / 255.0
        layout = layout * (1.0 - lines[..., None]) + col * lines[..., None]

    seams = np.zeros_like(lines)
    if seam_thickness > 0 and len(boundary):
        seams = draw_edges(uv, boundary, width, height, seam_thickness) * line_opacity
        scol = np.asarray(seam_color, np.float32) / 255.0
        layout = layout * (1.0 - seams[..., None]) + scol * seams[..., None]

    if tex is None:
        tex = np.ones((height, width, 3), np.float32)

    return {
        'layout': np.clip(layout, 0, 1).astype(np.float32),
        'texture': tex,
        'lines': np.maximum(lines, seams).astype(np.float32),
        'edge_lines': lines.astype(np.float32),
        'seam_lines': seams.astype(np.float32),
        'islands': islands,
        'base_color': base,
        'metallic_roughness': mr,
        'size': (width, height),
    }


def lines_rgba(edge_lines, seam_lines, line_color, seam_color):
    """RGBA uint8 image of the UV wireframe on a transparent background (for an editor layer)."""
    h, w = edge_lines.shape
    rgb = np.zeros((h, w, 3), np.float32)
    a = np.maximum(edge_lines, seam_lines)
    lc = np.asarray(line_color, np.float32) / 255.0
    sc = np.asarray(seam_color, np.float32) / 255.0
    with np.errstate(invalid='ignore', divide='ignore'):
        wsum = edge_lines + seam_lines
        rgb = np.where(wsum[..., None] > 0,
                       (edge_lines[..., None] * lc + seam_lines[..., None] * sc) / np.maximum(wsum, 1e-6)[..., None],
                       0)
    seam_on = seam_lines > 0
    rgb[seam_on] = sc
    out = np.concatenate([rgb, a[..., None]], axis=-1)
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def bleed(color, mask, pixels):
    """Extend colors from inside `mask` (bool) outward by `pixels` (<=0: whole image).

    Prevents dark seams where the texture is sampled across island borders.
    """
    if mask.all() or not mask.any():
        return color
    src = np.where(mask, 0, 255).astype(np.uint8)  # zeros = known pixels
    dist, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(src == 0)
    # labels number the zero pixels in row-major scan order, starting at 1.
    lut_y = np.concatenate([[0], ys])
    lut_x = np.concatenate([[0], xs])
    ny = lut_y[labels]
    nx = lut_x[labels]
    filled = color[ny, nx]
    region = ~mask
    if pixels > 0:
        region &= dist <= pixels
    out = color.copy()
    out[region] = filled[region]
    return out


def apply_painted_texture(mesh, painted_rgb, bleed_pixels=8, layer_alpha=None, keep_alpha=True,
                          metallic_roughness_rgb=None, alpha_mode=None, double_sided=None):
    """Return (new trimesh, base color PIL RGBA, metallic roughness PIL or None).

    painted_rgb: float (H,W,3) 0..1, the full painted texture.
    layer_alpha: optional float (H,W) 0..1 opacity of the painted layer over the mesh's
                 current texture (1 = painted, 0 = keep original).
    """
    import trimesh as Trimesh

    uv = mesh_uvs(mesh)
    base, mr, material = mesh_textures(mesh)
    h, w = painted_rgb.shape[:2]

    color = np.clip(painted_rgb, 0, 1).astype(np.float32)
    if layer_alpha is not None:
        a = np.clip(layer_alpha, 0, 1).astype(np.float32)
        if a.shape != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        if base is not None:
            orig = np.asarray(base.convert('RGB').resize((w, h), Image.LANCZOS), np.float32) / 255.0
        else:
            orig = np.ones((h, w, 3), np.float32)
        color = orig * (1.0 - a[..., None]) + color * a[..., None]

    if bleed_pixels != 0:
        islands = rasterize_islands(uv, mesh.faces, w, h, grow=0) > 0
        color = bleed(color, islands, bleed_pixels)

    rgb8 = (np.clip(color, 0, 1) * 255).round().astype(np.uint8)
    if keep_alpha and base is not None and base.mode in ('RGBA', 'LA'):
        alpha = np.asarray(base.getchannel('A').resize((w, h), Image.LANCZOS))
    else:
        alpha = np.full((h, w), 255, np.uint8)
    base_out = Image.fromarray(np.concatenate([rgb8, alpha[..., None]], axis=-1), 'RGBA')

    if metallic_roughness_rgb is not None:
        mr_out = Image.fromarray((np.clip(metallic_roughness_rgb, 0, 1) * 255).round().astype(np.uint8)[..., :3])
    else:
        mr_out = mr

    kwargs = dict(
        baseColorTexture=base_out,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=mr_out,
        metallicFactor=getattr(material, 'metallicFactor', None) if material is not None and mr_out is mr else None,
        roughnessFactor=getattr(material, 'roughnessFactor', None) if material is not None and mr_out is mr else None,
        alphaMode=alpha_mode or getattr(material, 'alphaMode', None) or 'OPAQUE',
        doubleSided=double_sided if double_sided is not None else bool(getattr(material, 'doubleSided', False) or False),
    )
    if mr_out is not None:
        if kwargs['metallicFactor'] is None:
            kwargs['metallicFactor'] = 1.0
        if kwargs['roughnessFactor'] is None:
            kwargs['roughnessFactor'] = 1.0
    else:
        kwargs.pop('metallicRoughnessTexture')
        kwargs['metallicFactor'] = 0.0
        kwargs['roughnessFactor'] = 1.0 if kwargs['roughnessFactor'] is None else kwargs['roughnessFactor']
    new_material = Trimesh.visual.material.PBRMaterial(**kwargs)

    out = mesh.copy()
    out.visual = Trimesh.visual.TextureVisuals(uv=uv.copy(), material=new_material)
    return out, base_out, mr_out
