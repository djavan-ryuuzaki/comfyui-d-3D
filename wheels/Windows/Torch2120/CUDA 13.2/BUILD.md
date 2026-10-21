# Wheels for Python 3.13 / PyTorch 2.12 / CUDA 13.2 (Windows x64)

| Wheel | Source | Commit | Patches |
|---|---|---|---|
| `flex_gemm-1.0.0-cp313-cp313-win_amd64.whl` | https://github.com/visualbruno/FlexGEMM | `db388bd17c34abc697792aa718dca780734c2783` | none |
| `cumesh-1.0-cp313-cp313-win_amd64.whl` | https://github.com/visualbruno/CuMesh | `d10e54c30ddd03d11472c1431693f985501c7966` | `build/patches/cumesh.patch` |
| `o_voxel-0.0.1-cp313-cp313-win_amd64.whl` | https://github.com/visualbruno/TRELLIS.2 (`o-voxel` folder) | `65d1e13b4a92296036044df0633242bb9e95abf6` | none |
| `nvdiffrast-0.4.0-cp313-cp313-win_amd64.whl` | https://github.com/NVlabs/nvdiffrast (tag `v0.4.0`) | `253ac4fcea7de5f396371124af597e6cc957bfae` | none |
| `nvdiffrec_render-0.0.0-cp313-cp313-win_amd64.whl` | https://github.com/JeffreyXiang/nvdiffrec (branch `renderutils`) | `b296927cc7fd01c2ac1087c8065c4d7248f72da4` | none |

o_voxel also needs `plyfile`, `easydict`, `trimesh`, `zstandard` and `tqdm` at runtime; they are in the node's `requirements.txt` or already shipped with ComfyUI.

## Build environment

- Visual Studio 2022 Build Tools, MSVC 14.44 (x64, `vcvars64.bat`)
- CUDA 13.2 (`CUDA_HOME` / `CUDA_PATH`)
- Python 3.13 (ComfyUI portable `python_embeded`), torch 2.12.0+cu132, ninja, setuptools, wheel
- `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0+PTX`
- `CL=/Zc:preprocessor`: the CCCL headers of CUDA 13 refuse MSVC's traditional preprocessor (`fatal error C1189`).
- `DISTUTILS_USE_SDK=1`, `MAX_JOBS=8`

## cumesh patch

- CCCL 3.2 added `cub::DeviceScan::ExclusiveSum(in, out, num_items, env)`, which now captures the in-place call
  `ExclusiveSum(temp, bytes, data, num_items)`. All 16 such calls are rewritten to `ExclusiveSum(temp, bytes, data, data, num_items)`
  (still in place). `build/fix_inplace_scan.py <src dir>` does this rewrite.
- `setup.py`: the `_cubvh` extension looks for Eigen in `third_party/cubvh/third_party/eigen`, which is not in the repo;
  the repo root (which vendors `Eigen/`) is added to `include_dirs`.

## Rebuild

```bash
git clone --recursive https://github.com/visualbruno/CuMesh cumesh
git -C cumesh checkout d10e54c30ddd03d11472c1431693f985501c7966
git -C cumesh apply "path\to\build\patches\cumesh.patch"
set PYTHON=C:\path\to\ComfyUI_windows_portable\python_embeded\python.exe
build\build.bat cumesh C:\path\to\wheels
```

The other packages build the same way without patches (for o_voxel pass the `o-voxel` subfolder of the TRELLIS.2 clone).
