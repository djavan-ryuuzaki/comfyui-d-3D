"""D3D QwenTex nodes: texture a mesh with Qwen Image 2.1, using UltraTex's mesh prep and bake.

Workflow (stock ComfyUI nodes in brackets):

  mesh (TRIMESH) + reference image
    -> QwenTex Prep -> prep
    -> QwenTex Control Atlas -> control_atlas, mask_atlas, prompt, width/height
         [Load Model Patch: qwen_image_2.1_fun_controlnet_union]
         [Apply Fun ControlNet (image = control_atlas)]           on the Qwen Image 2.1 model
         [Text Encode Qwen Image 2.1 (image_1 = reference)]       positive / negative
         [Empty latent of width x height]  -> [KSampler] -> [VAE Decode] -> atlas
    -> QwenTex Split Atlas (atlas -> 6 views)
    -> QwenTex Bake Texture -> GLB (output/3D/qwentex)

Second pass (4 more views at 45/135/225/315 deg): QwenTex Rotate Rig -> QwenTex Render Views
(inpaint_image + mask for Apply Fun ControlNet, and the control atlas of the rotated rig) -> sampler ->
Split Atlas (rotated prep) -> Bake (prep_2 / albedo_views_2).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch
import trimesh as trimesh_lib
from PIL import Image

import folder_paths

try:  # ComfyUI's 3D file type (Preview 3D input)
    from comfy_api.latest import Types
    File3D = Types.File3D
except (ImportError, AttributeError):
    File3D = None

from . import atlas as atlas_core
from .ultratex_core import bake as bake_core
from .ultratex_core import prep as prep_core

log = logging.getLogger("D3D.QwenTex")

CATEGORY = "D3D/QwenTex"
PREP_TYPE = "QWENTEX_PREP"
BAKE_TYPE = "QWENTEX_BAKE"

VIEWS_2_CHOICES = {
    "all": None,
    "no top/bottom": [0, 1, 2, 3],
    "front + back": [0, 2],
    "front only": [0],
    "back only": [2],
}


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return np.clip(image.detach().cpu().float().numpy() * 255 + 0.5, 0, 255).astype(np.uint8)


def _to_image(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32) / 255.0)


def _resolve_path(path: str) -> str:
    path = path.strip().strip('"')
    for c in (path, os.path.join(folder_paths.get_input_directory(), path), os.path.join(folder_paths.get_output_directory(), path)):
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(f"QwenTex: mesh file not found: {path!r} (absolute, or relative to the input/output folders)")


def _rig(prep):
    return None if np.allclose(prep.rig, np.eye(3)) else prep.rig


def _same_mesh(a, b) -> bool:
    return a.vertices.shape == b.vertices.shape and np.array_equal(a.uvs, b.uvs)


def _views_for_bake(images: torch.Tensor, prep):
    if images.shape[0] != 6:
        raise ValueError(f"QwenTex Bake: expected 6 views (QwenTex Split Atlas), got {images.shape[0]}")
    views = images[..., :3].cpu().numpy().astype(np.float32)
    return views, atlas_core.view_masks(prep, views.shape[1]), _rig(prep)


# =========================================================================== prep
class D3DQwenTexPrep:
    """Normalises the mesh, keeps (or makes) its UVs, renders the 6 canonical views and aligns the reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {"tooltip": "The object (ideally its front) on a transparent background, or with reference_mask."}),
                "render_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256, "tooltip": "Resolution of the internal renders (silhouettes used by the bake). 2048 is plenty for tiles up to 1024."}),
                "uv_mode": (["keep_existing", "comfy_gpu", "xatlas"], {"default": "keep_existing", "tooltip": "keep_existing: the mesh's own UVs (TRELLIS.2 Mesh Texturing / UnWrap output); falls back to comfy_gpu when the mesh has none. comfy_gpu: ComfyUI's GPU unwrapper. xatlas: CPU, slow on dense meshes."}),
                "atlas_size": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 512, "tooltip": "Texture size the new UV charts are packed for (ignored with keep_existing)."}),
                "reference_alignment": (["fit_front_silhouette", "center"], {"tooltip": "fit_front_silhouette: crop the reference to its mask and fit it on the front view's silhouette."}),
                "mask_meaning": (["background (LoadImage)", "foreground"], {"tooltip": "LoadImage's MASK is 1 on transparent pixels; background-removal nodes usually output 1 on the object."}),
            },
            "optional": {
                "trimesh": ("TRIMESH", {"tooltip": "Mesh from the TRELLIS.2 nodes. Takes priority over mesh_path."}),
                "mesh_path": ("STRING", {"default": "", "tooltip": "Mesh file (glb/obj/ply), absolute or relative to input/output. Used when no TRIMESH is connected."}),
                "reference_mask": ("MASK",),
            },
        }

    RETURN_TYPES = (PREP_TYPE, "IMAGE", "IMAGE")
    RETURN_NAMES = ("prep", "normal_views", "reference")
    FUNCTION = "prep"
    CATEGORY = CATEGORY
    DESCRIPTION = "Mesh + reference image -> QwenTex prep (UltraTex 6-view camera rig)."

    def prep(self, reference_image, render_size, uv_mode, atlas_size, reference_alignment, mask_meaning,
             trimesh=None, mesh_path="", reference_mask=None):
        if trimesh is not None:
            mesh_obj = trimesh
        elif mesh_path and mesh_path.strip():
            mesh_obj = trimesh_lib.load(_resolve_path(mesh_path), process=False)
        else:
            raise ValueError("QwenTex Prep: connect a TRIMESH or set mesh_path")
        img = _to_uint8(reference_image[0])
        rgb = img[..., :3]
        if reference_mask is not None:
            m = reference_mask[0].cpu().numpy()
            if m.shape != rgb.shape[:2]:
                m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(rgb.shape[1::-1], Image.Resampling.BILINEAR)) / 255.0
            alpha = 1.0 - m if mask_meaning.startswith("background") else m
        elif img.shape[-1] == 4:
            alpha = img[..., 3] / 255.0
        else:
            raise ValueError("QwenTex Prep: the reference has no alpha; connect reference_mask (LoadImage's MASK or a background-removal mask)")
        rgba = np.dstack([rgb, np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)])
        prep = prep_core.run_prep(mesh_obj, rgba, render_size, atlas_size, uv_mode, reference_alignment)

        size = min(render_size, 1024)
        normals = atlas_core.render_control(prep, size, "normal")
        ref = Image.fromarray(prep.reference_rgba).resize((size, size), Image.Resampling.LANCZOS)
        white = Image.new("RGB", ref.size, (255, 255, 255))
        white.paste(ref, mask=ref.getchannel("A"))
        return (prep, _to_image(normals), _to_image(np.asarray(white))[None])


# =========================================================================== control atlas
class D3DQwenTexControlAtlas:
    """Renders the 2x3 control atlas that guides Qwen Image 2.1 (Fun ControlNet) for the prep's rig."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": (PREP_TYPE,),
                "tile_size": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "tooltip": "Size of each of the 6 views. The atlas is 3*tile x 2*tile (768 -> 2304x1536). 1024 gives a 3072x2048 atlas: sharper, much slower."}),
                "control": (atlas_core.CONTROL_MODES, {"default": "depth", "tooltip": "depth: Fun ControlNet depth (near = white). grayscale: shaded clay render (Fun ControlNet grayscale / colourise). normal: world-normal colours (not a trained Fun ControlNet condition; use as a reference image)."}),
            },
            "optional": {
                "extra_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Appended to the layout prompt, e.g. a short description of the object."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("control_atlas", "mask_atlas", "reference_tile", "prompt", "width", "height")
    OUTPUT_TOOLTIPS = (
        "Control image for Apply Fun ControlNet (image).",
        "1 on the object, 0 on the background.",
        "The aligned reference at tile size, on white: connect to Text Encode Qwen Image 2.1 image_1.",
        "Layout prompt describing the 2x3 grid: connect to the text encoder's prompt.",
        "Atlas width: use for the empty latent.",
        "Atlas height: use for the empty latent.",
    )
    FUNCTION = "make"
    CATEGORY = CATEGORY

    def make(self, prep, tile_size, control, extra_prompt=""):
        tiles = atlas_core.render_control(prep, tile_size, control)
        masks = atlas_core.view_masks(prep, tile_size)
        grid = atlas_core.to_atlas(tiles)
        mask = atlas_core.to_atlas(masks).astype(np.float32)
        ref = Image.fromarray(prep.reference_rgba).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        white = Image.new("RGB", ref.size, (255, 255, 255))
        white.paste(ref, mask=ref.getchannel("A"))
        prompt = atlas_core.LAYOUT_PROMPT + ((" " + extra_prompt.strip()) if extra_prompt and extra_prompt.strip() else "")
        h, w = grid.shape[:2]
        return (_to_image(grid)[None], torch.from_numpy(mask)[None], _to_image(np.asarray(white))[None], prompt, w, h)


# =========================================================================== split
class D3DQwenTexSplitAtlas:
    """Cuts the generated 2x3 atlas into the 6 views (view order) and whitens everything outside the silhouettes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": (PREP_TYPE, {"tooltip": "The prep (or rotated rig) the atlas was generated for."}),
                "atlas": ("IMAGE", {"tooltip": "VAE Decode output (one 2x3 atlas)."}),
                "clean_background": ("BOOLEAN", {"default": True, "tooltip": "Paint pixels outside the mesh silhouette white (what the bake expects)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("views", "masks", "atlas_clean")
    FUNCTION = "split"
    CATEGORY = CATEGORY

    def split(self, prep, atlas, clean_background):
        arr = atlas[0, ..., :3].cpu().float().numpy()
        views = atlas_core.from_atlas(arr)
        tile = views.shape[1]
        masks = atlas_core.view_masks(prep, tile)
        if clean_background:
            views = np.where(masks[..., None], views, 1.0)
        views = np.clip(views, 0.0, 1.0).astype(np.float32)
        grid = atlas_core.to_atlas(views)
        return (torch.from_numpy(views), torch.from_numpy(masks.astype(np.float32)), torch.from_numpy(grid)[None])


# =========================================================================== bake
class D3DQwenTexBake:
    """Back-projects the views into a UV texture (UltraTex two-band fusion) and saves a textured GLB."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": (PREP_TYPE,),
                "albedo_views": ("IMAGE", {"tooltip": "6 views from QwenTex Split Atlas."}),
                "texture_size": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 512}),
                "filename_prefix": ("STRING", {"default": "3D/qwentex/mesh"}),
                "view_weight_power": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 16.0, "step": 0.5, "tooltip": "Blend weight = cos(normal, view)^power."}),
                "best_view_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "legacy fusion only: 0 = weighted blend, 1 = best view only."}),
                "edge_feather_px": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 64.0, "step": 1.0, "tooltip": "Down-weight pixels near silhouettes / occlusion edges."}),
            },
            "optional": {
                "prep_2": (PREP_TYPE, {"tooltip": "Rotated rig of a second pass (QwenTex Rotate Rig)."}),
                "albedo_views_2": ("IMAGE", {"tooltip": "6 views generated for prep_2."}),
                "views_2_use": (list(VIEWS_2_CHOICES), {"default": "no top/bottom", "tooltip": "Which second-pass views to bake."}),
                "fusion": (["two_band", "legacy"], {"default": "two_band"}),
                "detail_sharpness": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 128.0, "step": 1.0, "tooltip": "two_band: how strongly fine detail comes from the best view only."}),
                "exposure_match": ("BOOLEAN", {"default": True, "tooltip": "two_band: per-view colour gains (+-15%) to remove brightness seams."}),
                "hole_fill": (["mesh", "nearest"], {"default": "mesh"}),
                "views_2_weight": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Below 1 the second-pass views paint only where they see clearly better than the first pass."}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "STRING", "FILE_3D_GLB", BAKE_TYPE)
    RETURN_NAMES = ("trimesh", "albedo_texture", "preview", "glb_path", "model_3d", "bake_state")
    FUNCTION = "bake"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def bake(self, prep, albedo_views, texture_size, filename_prefix, view_weight_power, best_view_mix, edge_feather_px,
             prep_2=None, albedo_views_2=None, views_2_use="no top/bottom", fusion="two_band", detail_sharpness=32.0,
             exposure_match=True, hole_fill="mesh", views_2_weight=0.6):
        settings = bake_core.BakeSettings(
            power=view_weight_power, best_view_mix=best_view_mix, edge_px=edge_feather_px, fusion=fusion,
            detail_power=float(detail_sharpness), exposure_match=bool(exposure_match), hole_fill=hole_fill,
            extra_view_bias=float(views_2_weight),
        )
        sets = [_views_for_bake(albedo_views, prep)]
        if prep_2 is not None and albedo_views_2 is not None:
            if not _same_mesh(prep, prep_2):
                raise ValueError("QwenTex Bake: prep_2 must come from QwenTex Rotate Rig on the same prep")
            sets.append((*_views_for_bake(albedo_views_2, prep_2), VIEWS_2_CHOICES.get(views_2_use)))

        baker = bake_core.Baker(prep, texture_size)
        try:
            albedo_np, confidence = baker.bake(sets, settings)
            albedo = bake_core.to_pil(albedo_np)
            preview = baker.preview(np.asarray(albedo).astype(np.float32) / 255.0)
        finally:
            baker.release()

        mesh = bake_core.textured_mesh(prep, albedo, None)
        out_dir = folder_paths.get_output_directory()
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        name = f"{filename}_{counter:05}_"
        glb_file = os.path.join(full_folder, name + ".glb")
        mesh.export(glb_file)
        albedo.save(os.path.join(full_folder, name + "_albedo.png"))
        rel = os.path.join(subfolder, name + ".glb").replace("\\", "/")
        log.info("QwenTex: saved %s", rel)
        state = {"albedo": albedo_np, "confidence": confidence, "uvs": prep.uvs}
        model_3d = File3D(glb_file, file_format="glb") if File3D is not None else None
        return {"ui": {"3d": [{"filename": name + ".glb", "subfolder": subfolder, "type": "output"}]},
                "result": (mesh, _to_image(np.asarray(albedo))[None], torch.from_numpy(preview), rel, model_3d, state)}


# =========================================================================== second pass
class D3DQwenTexRotateRig:
    """Same mesh and UVs seen through a rotated camera rig (45 deg: side views at 45/135/225/315)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": (PREP_TYPE,),
                "azimuth": ("FLOAT", {"default": 45.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "elevation": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0, "tooltip": "Side cameras seen from above (> 0) or below (< 0)."}),
            }
        }

    RETURN_TYPES = (PREP_TYPE, "IMAGE")
    RETURN_NAMES = ("prep", "normal_views")
    FUNCTION = "rotate"
    CATEGORY = CATEGORY

    def rotate(self, prep, azimuth, elevation):
        rotated = prep_core.rotate_rig(prep, azimuth, elevation, per_view=True)
        return (rotated, _to_image(atlas_core.render_control(rotated, min(rotated.render_size, 1024), "normal")))


class D3DQwenTexRenderViews:
    """Re-projects the first pass onto a rotated rig: the inpaint image + mask for the second pass.

    Connect inpaint_atlas -> Apply Fun ControlNet inpaint_image and regenerate_mask -> its mask
    (1 = regenerate). Regions the first pass saw head-on are kept, the rest is repainted.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prep": (PREP_TYPE, {"tooltip": "The rotated rig (QwenTex Rotate Rig)."}),
                "source_prep": (PREP_TYPE, {"tooltip": "The prep of the first pass."}),
                "source_views": ("IMAGE", {"tooltip": "The 6 first-pass views (QwenTex Split Atlas)."}),
                "tile_size": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16, "tooltip": "Same tile size as the second pass's control atlas."}),
                "keep_above": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.01, "step": 0.01, "tooltip": "Keep texels the first pass saw at least this head-on (cosine; 0.75 ~ 41 deg). 1.01 = repaint everything."}),
            },
            "optional": {
                "bake_state": (BAKE_TYPE, {"tooltip": "Use a first QwenTex Bake's texture instead of baking source_views again."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("inpaint_atlas", "regenerate_mask", "init_views", "keep_mask")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, prep, source_prep, source_views, tile_size, keep_above, bake_state=None):
        if not _same_mesh(prep, source_prep):
            raise ValueError("QwenTex Render Views: prep must be QwenTex Rotate Rig applied to source_prep")
        baker = bake_core.Baker(prep, max(2048, tile_size))
        try:
            if bake_state is not None:
                if not np.array_equal(bake_state["uvs"], prep.uvs):
                    raise ValueError("QwenTex Render Views: bake_state comes from a different mesh / UV layout")
                tex, conf_tex = bake_state["albedo"], bake_state["confidence"]
            else:
                tex, conf_tex = baker.bake([_views_for_bake(source_views, source_prep)], bake_core.BakeSettings())
            views, covers, conf = baker.render_views(tex, conf_tex, tile_size, _rig(prep))
        finally:
            baker.release()
        keep = (conf >= keep_above) & covers
        silhouettes = atlas_core.view_masks(prep, tile_size)
        regen = silhouettes & ~keep
        views = np.where(silhouettes[..., None], views, 1.0).astype(np.float32)
        return (torch.from_numpy(atlas_core.to_atlas(views))[None],
                torch.from_numpy(atlas_core.to_atlas(regen).astype(np.float32))[None],
                torch.from_numpy(views), torch.from_numpy(keep.astype(np.float32)))


NODE_CLASS_MAPPINGS = {
    "D3DQwenTexPrep": D3DQwenTexPrep,
    "D3DQwenTexControlAtlas": D3DQwenTexControlAtlas,
    "D3DQwenTexSplitAtlas": D3DQwenTexSplitAtlas,
    "D3DQwenTexBake": D3DQwenTexBake,
    "D3DQwenTexRotateRig": D3DQwenTexRotateRig,
    "D3DQwenTexRenderViews": D3DQwenTexRenderViews,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "D3DQwenTexPrep": "QwenTex - Prep (mesh + reference)",
    "D3DQwenTexControlAtlas": "QwenTex - Control Atlas (Fun ControlNet)",
    "D3DQwenTexSplitAtlas": "QwenTex - Split Atlas (6 views)",
    "D3DQwenTexBake": "QwenTex - Bake Texture (GLB)",
    "D3DQwenTexRotateRig": "QwenTex - Rotate Rig (second pass)",
    "D3DQwenTexRenderViews": "QwenTex - Render Views (second pass inpaint)",
}
