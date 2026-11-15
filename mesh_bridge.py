"""Bridge between ComfyUI's native MESH type and this pack's TRIMESH type.

ComfyUI 0.45+ ships native mesh nodes (RemeshMesh, DecimateMesh, UnwrapMesh, BakeTextureFromVoxel,
BakeNormalMapFromMesh, BakeAmbientOcclusion, ApplyTextureToMesh, ...) that work on MESH, a batch of
tensors. The TRELLIS.2 / QwenTex / Quad / UV paint nodes here work on a trimesh.Trimesh. These two
nodes convert between them without moving the geometry, so voxel bakes and texture bakes from either
side line up.

UV convention: native MESH stores glTF UVs (v grows downwards, as written to the GLB; see the
"UV y flipped vs trimesh" line in ComfyUI's UnwrapMesh). trimesh stores v upwards. The texture
images themselves are the same in both (row 0 = top of the image), so only v is flipped.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _item(t, index, count=None):
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        t = t[index]
    elif t.ndim >= 3 or (t.ndim == 2 and count is not None and t.shape[0] != count):
        t = t[index]
    if count is not None:
        t = t[:count]
    return t


def _image_to_pil(img, mode="RGB"):
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]
        arr = img.detach().float().cpu().numpy()
    else:
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    arr = (arr[..., :3] * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert(mode)


def _pil_to_image(pil):
    if pil is None:
        return None
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def native_to_trimesh(mesh, index=0, keep_textures=True):
    import trimesh as Trimesh

    vc = getattr(mesh, "vertex_counts", None)
    fc = getattr(mesh, "face_counts", None)
    nv = int(vc[index]) if vc is not None else None
    nf = int(fc[index]) if fc is not None else None

    verts = _item(mesh.vertices, index, nv).detach().float().cpu().numpy()
    faces = _item(mesh.faces, index, nf).detach().long().cpu().numpy()
    uvs = _item(getattr(mesh, "uvs", None), index, nv)
    colors = _item(getattr(mesh, "vertex_colors", None), index, nv)
    normals = _item(getattr(mesh, "normals", None), index, nv)

    kwargs = {}
    if normals is not None:
        kwargs["vertex_normals"] = normals.detach().float().cpu().numpy()
    out = Trimesh.Trimesh(vertices=verts, faces=faces, process=False, **kwargs)

    if uvs is not None:
        uv = uvs.detach().float().cpu().numpy().copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        base = mr = None
        if keep_textures:
            tex = getattr(mesh, "texture", None)
            base = _image_to_pil(tex[index] if tex is not None and tex.ndim == 4 else tex, "RGBA")
            mrt = getattr(mesh, "metallic_roughness", None)
            mr = _image_to_pil(mrt[index] if mrt is not None and mrt.ndim == 4 else mrt)
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=base,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=mr,
            metallicFactor=1.0 if mr is not None else 0.0,
            roughnessFactor=1.0,
        )
        out.visual = Trimesh.visual.TextureVisuals(uv=uv, material=material)
    elif colors is not None:
        c = np.clip(colors.detach().float().cpu().numpy(), 0.0, 1.0)
        if c.shape[1] == 3:
            c = np.concatenate([c, np.ones((len(c), 1), np.float32)], axis=1)
        out.visual = Trimesh.visual.ColorVisuals(out, vertex_colors=(c * 255 + 0.5).astype(np.uint8))
    return out


def _single(tm):
    import trimesh as Trimesh

    if isinstance(tm, Trimesh.Scene):
        geoms = list(tm.dump()) if hasattr(tm, "dump") else list(tm.geometry.values())
        if len(geoms) == 1:
            return geoms[0]
        return Trimesh.util.concatenate(geoms)
    return tm


def trimesh_to_native(tm, keep_textures=True):
    from comfy_api.latest import Types

    tm = _single(tm)
    verts = torch.from_numpy(np.ascontiguousarray(tm.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.ascontiguousarray(tm.faces).astype(np.int64))
    kwargs = {}
    visual = getattr(tm, "visual", None)
    if visual is not None and visual.kind == "texture" and getattr(visual, "uv", None) is not None \
            and len(visual.uv) == len(verts):
        uv = np.asarray(visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        kwargs["uvs"] = torch.from_numpy(uv)[None]
        mat = getattr(visual, "material", None)
        if keep_textures and mat is not None:
            base = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if base is not None:
                kwargs["texture"] = _pil_to_image(base)
            mr = getattr(mat, "metallicRoughnessTexture", None)
            if mr is not None:
                kwargs["metallic_roughness"] = _pil_to_image(mr)
    elif visual is not None and visual.kind == "vertex":
        c = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        kwargs["vertex_colors"] = torch.from_numpy(c)[None]
    try:
        n = np.asarray(tm.vertex_normals, dtype=np.float32)
        if n.shape == tuple(verts.shape):
            kwargs["normals"] = torch.from_numpy(np.ascontiguousarray(n))[None]
    except Exception:
        pass
    return Types.MESH(vertices=verts[None], faces=faces[None], **kwargs)


class D3DNativeMeshToTrimesh:
    """Native ComfyUI MESH -> TRIMESH for the TRELLIS.2 / QwenTex / Quad / UV paint nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH", {"tooltip": "Native ComfyUI mesh (VaeDecodeShapeTrellis, RemeshMesh, DecimateMesh, UnwrapMesh, ApplyTextureToMesh...)."}),
                "keep_textures": ("BOOLEAN", {"default": True, "tooltip": "Carry the base color and metallic/roughness textures over when the mesh has UVs."}),
            },
            "optional": {
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "Which mesh of a batch to convert."}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "convert"
    CATEGORY = "Trellis2Wrapper/bridge"
    DESCRIPTION = "Converts a native ComfyUI MESH to TRIMESH. Geometry is not moved; UVs are flipped from glTF (v down) to trimesh (v up)."

    def convert(self, mesh, keep_textures, batch_index=0):
        return (native_to_trimesh(mesh, batch_index, keep_textures),)


class D3DTrimeshToNativeMesh:
    """TRIMESH -> native ComfyUI MESH for RemeshMesh, DecimateMesh, Bake*, ApplyTextureToMesh, SaveGLB..."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "Mesh from the TRELLIS.2 / QwenTex / Quad Remesh / UV paint nodes."}),
                "keep_textures": ("BOOLEAN", {"default": True, "tooltip": "Carry the base color and metallic/roughness textures over when the mesh has UVs."}),
            },
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "convert"
    CATEGORY = "Trellis2Wrapper/bridge"
    DESCRIPTION = "Converts a TRIMESH to a native ComfyUI MESH. Geometry is not moved; UVs are flipped from trimesh (v up) to glTF (v down)."

    def convert(self, trimesh, keep_textures):
        return (trimesh_to_native(trimesh, keep_textures),)


NODE_CLASS_MAPPINGS = {
    "D3DNativeMeshToTrimesh": D3DNativeMeshToTrimesh,
    "D3DTrimeshToNativeMesh": D3DTrimeshToNativeMesh,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "D3DNativeMeshToTrimesh": "D3D - Native Mesh -> Trimesh",
    "D3DTrimeshToNativeMesh": "D3D - Trimesh -> Native Mesh",
}
