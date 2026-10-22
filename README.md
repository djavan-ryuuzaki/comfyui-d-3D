# comfyui-d-3D

Fork of [visualbruno/ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) that also runs on the
**Comfy-Org int8 / bf16 single-file models** (the ones used by ComfyUI's native TRELLIS.2 / Pixal3D nodes) and ships
wheels for **PyTorch 2.12 + CUDA 13.2** on Windows (Python 3.13) and Linux (Python 3.12 and 3.13).

## What's new in this fork

- New node **"Trellis2 - LoadModel (Comfy-Org int8 / bf16 files)"**. It returns the same `TRELLIS2PIPELINE`, so every other node of the pack works unchanged.
  - `diffusion_model` (models/diffusion_models): `trellis_2_int8_convrot`, `pixal3d_int8_convrot`, `pixal3d_multiview_int8_convrot` or their `_bf16` versions. TRELLIS.2 vs Pixal3D is detected from the file; multi-view is detected from the file name (or forced with `pixal3d_multiview`).
  - `shape_vae` / `texture_vae` (models/vae): `trellis_2_shape_vae_bf16`, `trellis_2_texture_vae_bf16`.
  - `dino_model` (models/clip_vision): `dino_v3_vit_l` or `dino_v3_L_naf_fp32`, or `auto` to download facebook/dinov3 as before.
  - int8 ConvRot layers stay int8 in VRAM (~1.3 GB per denoiser) and run through `comfy_kitchen`, which ships with ComfyUI.
- Works with transformers 5.x and huggingface_hub 1.x.
- Comfy-Org ships a single texture denoiser, so the `512` texture stage reuses it (same as the Pixal3D path). ReconViaGen and the FP8 models still need the original `Trellis2 - LoadModel` node.

Model files: [Comfy-Org/TRELLIS.2](https://huggingface.co/Comfy-Org/TRELLIS.2) and [Comfy-Org/Pixal3D](https://huggingface.co/Comfy-Org/Pixal3D).

### Wheels for Python 3.13 / Torch 2.12 / CUDA 13.2 (Windows)

In `wheels/Windows/Torch2120/CUDA 13.2`, built for sm_75 to sm_120 (RTX 20xx to RTX 50xx):

```bash
cd ComfyUI_windows_portable
python_embeded\python.exe -m pip install "ComfyUI\custom_nodes\comfyui-d-3D\wheels\Windows\Torch2120\CUDA 13.2\flex_gemm-1.0.0-cp313-cp313-win_amd64.whl"
python_embeded\python.exe -m pip install "ComfyUI\custom_nodes\comfyui-d-3D\wheels\Windows\Torch2120\CUDA 13.2\cumesh-1.0-cp313-cp313-win_amd64.whl"
python_embeded\python.exe -m pip install "ComfyUI\custom_nodes\comfyui-d-3D\wheels\Windows\Torch2120\CUDA 13.2\o_voxel-0.0.1-cp313-cp313-win_amd64.whl"
python_embeded\python.exe -m pip install "ComfyUI\custom_nodes\comfyui-d-3D\wheels\Windows\Torch2120\CUDA 13.2\nvdiffrast-0.4.0-cp313-cp313-win_amd64.whl"
python_embeded\python.exe -m pip install "ComfyUI\custom_nodes\comfyui-d-3D\wheels\Windows\Torch2120\CUDA 13.2\nvdiffrec_render-0.0.0-cp313-cp313-win_amd64.whl"
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\comfyui-d-3D\requirements.txt
```

Sources, commits and the patches needed for CUDA 13.2 are in `wheels/Windows/Torch2120/CUDA 13.2/BUILD.md`.

### Wheels for Python 3.12 and 3.13 / Torch 2.12 / CUDA 13.2 (Linux)

In `wheels/Linux/Torch2120/CUDA 13.2`, same packages and GPUs as Windows, for Python 3.12 (`cp312`) and 3.13 (`cp313`).
They work with torch 2.12.x `+cu130` or `+cu132`. Install the ones that match your Python:

```bash
cd ComfyUI/custom_nodes/comfyui-d-3D
PYTAG=cp$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')
pip install "wheels/Linux/Torch2120/CUDA 13.2/"*-$PYTAG-$PYTAG-*.whl
pip install -r requirements.txt
```

#### mmartial/comfyui-nvidia-docker (Docker / Kubernetes)

That image installs torch at startup into the venv at `/comfy/mnt/venv` (Python 3.12 from Ubuntu 24.04) and picks the
latest torch for cu130 by default. Pin torch 2.12 with this environment variable on the container or pod:

```
TORCH_LOCK=torch==2.12.0 torchvision==0.27.0
```

Restart the container so `init.bash` reinstalls torch, then install the `cp312` wheels inside the venv. To do it on
every start, save this as `/comfy/mnt/user_script.bash` (the `run` folder). `--force-reinstall` matters: an older
flex_gemm / cumesh / o_voxel with the same version number may already be in the venv, and pip would keep it (an old
flex_gemm fails with `Unsupported data type` in `hashmap_insert_3d_idx_as_val_cuda`). The constraint file comes from
`TORCH_LOCK` and keeps pip from replacing torch:

```bash
#!/bin/bash
source /comfy/mnt/venv/bin/activate
cd /comfy/mnt/ComfyUI/custom_nodes/comfyui-d-3D || exit 1  # with BASE_DIRECTORY: <BASE_DIRECTORY>/custom_nodes/comfyui-d-3D
pip3 install --force-reinstall --no-deps "wheels/Linux/Torch2120/CUDA 13.2/"*-cp312-cp312-*.whl
pip3 install --constraint /comfy/mnt/torch_lock.txt -r requirements.txt
exit 0  # 0 = let init.bash start ComfyUI normally
```

Build details are in `wheels/Linux/Torch2120/CUDA 13.2/BUILD.md`.

---

# 🌀 ComfyUI Wrapper for [https://github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)

---

<img width="883" height="566" alt="{09272892-57D6-4EB8-B27B-6B875916982A}" src="https://github.com/user-attachments/assets/a7788f13-141c-4072-9143-b8b1ee1ead2a" />

---

<img width="980" height="579" alt="{F6FE6B7B-94B7-44C6-8C89-02E7C81EBF7E}" src="https://github.com/user-attachments/assets/ad27111c-beb8-48ef-8613-c533a3a5cacd" />

---

## 📋 Changelog

| Date | Description |
| --- | --- |
| **2026-09-08** | Pixal3D: Added node "Mesh Texturing Pixal3D MultiView"<br>"Export Mesh" node exports model_3d compatible with new "Preview 3D" node  |
| **2026-09-07** | Add support for Pixal3D MultiView |
| **2026-07-31** | Added new nodes "Smooth Mesh with PyMeshlab" and "Smooth Trimesh with PyMeshlab" |
| **2026-06-02** | Added new node "Render MultiView (Nvdiffrast)"<br>Thanks GiusTex |
| **2026-05-26** | Added new nodes used for the projection<br>Check the example Projection_Blender_Qwen_XViews |
| **2026-05-22** | Added FOV Custom MoGe Camera node |
| **2026-05-20** | Fixed Pixal3D<br>The node "Mesh with Voxel Advanced Generator" is compatible with Pixal3D |
| **2026-05-13** | Added support for Pixal3D-T model<br>It's not compatible with all nodes<br>Check in the folder example_workflows |
| **2026-04-20** | Recreated all Workflows |
| **2026-04-20** | Added node "Sparse MultiView Generator"<br>Added node "ImageCond MultiView Generator"<br>Added node "Shape MultiView Generator"<br>Added node "Shape Cascade MultiView Generator"<br>Added node "Tex Slat MultiView Generator" |
| **2026-04-20** | Added node "Fill Holes Nicely with Meshlib"<br>Fixed DinoV3 Features Extractor |
| **2026-04-06** | Added new "DINO-Lock" functionality<br>New fill_holes in Sparse Generator<br>Thanks to Easymode on Discord |
| **2026-04-05** | Added node "Extract Images from Video"<br>Can be used with "Sparse Generator with ReconViaGen" |
| **2026-04-04** | Added node "Sparse Generator with ReconViaGen" |
| **2026-04-01** | Added node "Voxel to Mesh"<br>It replaces Remeshing to make watertight mesh |
| **2026-03-21** | Added node "Projection HighPoly to LowPoly"<br>Added node "Render MultiView" |
| **2026-03-17** | Added Inpainting Choice NS and TELEA |
| **2026-03-14** | Added Experimental node "Projection MultiView Texturing"<br>Check in example_workflows folder |
| **2026-03-08** | Updated CuMesh wheels for Torch 2.7, 2.8 and Linux<br>You can use the node "Fill Holes with Cumesh" |
| **2026-03-07** | Added "Heun" sampler<br>Added the node "Mesh with Voxel Cascade Generator" |
| **2026-03-05** | Added "RK4" and "RK5" samplers<br>Processing is much slower, so reduce the number of steps |
| **2026-03-04** | Sparse Structure Resolution supported up to 128<br>Experimental for "cascade" pipelines only<br>Can increase the details |
| **2026-02-27** | Added the Wheels for Windows Python 3.13, Torch 2.10.0, CUDA 13.1 |
| **2026-02-26** | Added FP8 models<br>Added "sdpa" and "flash_attn_3" for the backend |
| **2026-02-21** | Fixed "Vertical lines" bug |
| **2026-02-17** | Disabled Triton Cache (trying to fix vertical lines bug)<br>Fixed "Weld Vertices"<br>Added "Reconstruct Mesh with Quad" node |
| **2026-02-13** | Added the node "Weld Vertices"<br>Added the resolution 1536 for "Mesh Texturing" |
| **2026-02-12** | Added the node "Mesh With Voxel Multi-View Generator" |
|| Added the node "Mesh Texturing Multi-View" |
|| Added new example workflows |
| **2026-02-10** | Improved progress bar when filling holes with meshlib |
| **2026-02-09** | Fixed "Mesh Texturing" node<br>"mesh_cluster_threshold_cone_half_angle_rad" was not used |
| **2026-02-08** | Fixed "Fill Holes" node progress bar<br>Updated Cumesh package<br>Added "Remesh with Quad" node<br>Added "Batch Simplify Mesh and Export" node|
| **2026-02-07** | Updated Cumesh package<br>Improved "Remesh" node when removing inner layer|
| **2026-02-02** | Added node "Smooth Normals"<br>Useful for "Low Poly" mesh to remove the "blocky" aspect|
|| Added "remove_background" parameter for "PreProcess Image" node<br>Using rembg package|
| **2026-01-30** | Updated Cumesh, updated nodes, updated workflows|
||PostProcess UnWrap and Rasterize : removed fill_holes_max_perimeter <br> using fill holes from Meshlib|
||Remesh : added "remove_inner_faces" -> same algorithm as "Reconstruct Mesh"|
||Mesh Texturing: added "mesh_cluster_threshold_cone_half_angle_rad"|
| **2026-01-29** |Updated cumesh -> Remesh and Reconstruct made by chunk|
| **2026-01-28** |Added the node "Fill Holes With Meshlib"|
||Trying to fix caching issue|
| **2026-01-27** |Added the node "Trellis2ReconstructMesh"|
||"Multiple Images" support for "Mesh Refiner" node|
| **2026-01-21** |Added a "Continue" node|
||Added the option "bake_on_vertices" for "Mesh Texturing" node|
||Added "padding" option for "Preprocess Image" node|
| **2026-01-20** |Added node "Simplify Trimesh"|
||Fixed crash with "remove_infinite_vertices" in "PostProcess Mesh" node|
||Fixed texture generation|
| **2026-01-19** |Updated Cumesh|
| **2026-01-12** |Can pass multiple images to "Mesh Texturing" node (experimental)|
||Applied latest fixes from Microsoft|
| **2026-01-05** |Implemented "Tiled" Decoder|
||Updated Cumesh and O_voxel|

---

## REQUIREMENTS ##

You need to have access to facebook dinov3 models in order to use Trellis.2

[https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

Clone the repository in ComfyUI models folder under "facebook/dinov3-vitl16-pretrain-lvd1689m"

So in ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m

To use **TencentARC/Pixal3D-T** model, it's required to install **natten** package : https://github.com/SHI-Labs/NATTEN

---

## ⚙️ Installation Guide

> Tested on **Windows 11** with **Python 3.11** and **Torch = 2.7.0 + cu128**.

### 1. Install Wheels

#### For a standard python environment:

**If you use Torch v2.7.0:**
```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/cumesh-1.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.8.0:**
```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/cumesh-1.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

---

#### For ComfyUI Portable:

**If you use Torch v2.7.0:**
```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\cumesh-1.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.8.0:**
```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\cumesh-1.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

---

**Check the folder wheels for the other versions**

---

### 2. Custom Build

#### o_voxel

Use my own version of Trellis.2 here: https://github.com/visualbruno/TRELLIS.2

#### Cumesh 

Use my own version of Cumesh here: https://github.com/visualbruno/CuMesh

### FlexGEMM

Use my own version of FlexGEMM here: https://github.com/visualbruno/FlexGEMM

### natten (only used for TencentARC/Pixal3D-T model)

https://github.com/SHI-Labs/NATTEN

---

### 3. Requirements.txt

#### For a standard python environment:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-Trellis2/requirements.txt
```

---

#### For ComfyUI Portable:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Trellis2\requirements.txt
```

## 🙏 Acknowledgements

Discord community

"Blackwell Fix" from https://github.com/ThatButters/trellis2-blackwell-fix
