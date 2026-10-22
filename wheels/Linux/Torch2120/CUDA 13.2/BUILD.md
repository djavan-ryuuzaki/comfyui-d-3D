# Wheels for Python 3.12 and 3.13 / PyTorch 2.12 / CUDA 13.2 (Linux x86_64)

The `cp312` wheels are built the same way as the `cp313` ones listed below (same names with `cp312`).

Same sources and commits as the Windows build in `wheels/Windows/Torch2120/CUDA 13.2`:

| Wheel | Source | Commit | Patches |
|---|---|---|---|
| `flex_gemm-1.0.0-cp313-cp313-linux_x86_64.whl` | https://github.com/visualbruno/FlexGEMM | `db388bd17c34abc697792aa718dca780734c2783` | none |
| `cumesh-1.0-cp313-cp313-linux_x86_64.whl` | https://github.com/visualbruno/CuMesh | `d10e54c30ddd03d11472c1431693f985501c7966` | `wheels/Windows/Torch2120/CUDA 13.2/build/patches/cumesh.patch` |
| `o_voxel-0.0.1-cp313-cp313-linux_x86_64.whl` | https://github.com/visualbruno/TRELLIS.2 (`o-voxel` folder) | `65d1e13b4a92296036044df0633242bb9e95abf6` | none |
| `nvdiffrast-0.4.0-cp313-cp313-linux_x86_64.whl` | https://github.com/NVlabs/nvdiffrast (tag `v0.4.0`) | `253ac4fcea7de5f396371124af597e6cc957bfae` | none |
| `nvdiffrec_render-0.0.0-cp313-cp313-linux_x86_64.whl` | https://github.com/JeffreyXiang/nvdiffrec (branch `renderutils`) | `b296927cc7fd01c2ac1087c8065c4d7248f72da4` | none |

Built for sm_75, sm_80, sm_86, sm_89, sm_90 and sm_120 (+ PTX for sm_120), i.e. RTX 20xx to RTX 50xx.
The extensions link `libcudart.so.13`, `libc10*` and `libtorch*`, so they load with any torch 2.12.x CUDA 13 build
(`+cu130` from PyPI or `+cu132` from download.pytorch.org). They do not work with other torch versions.

## Build environment

- Ubuntu 24.04, GCC 13.3, Python 3.13 (`cp313`) and Python 3.12 (`cp312`, one venv per Python)
- torch 2.12.0 (PyPI, `+cu130`), ninja, setuptools, wheel
- nvcc 13.2.86 (`nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-nvvm`, `nvidia-cuda-cccl`, `nvidia-cuda-runtime` 13.2.86
  from PyPI, laid over the CUDA 13 libraries that torch installs). A regular CUDA 13.2 toolkit works the same way.
- `TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0+PTX`
- torch warns about the minor version mismatch (13.2 vs 13.0); that is expected.
- nvdiffrec links `-lcuda`. Without a driver installed, put the toolkit's `lib64/stubs/libcuda.so` in the linker path.

## Rebuild

```bash
git clone --recursive https://github.com/visualbruno/CuMesh cumesh
git -C cumesh checkout d10e54c30ddd03d11472c1431693f985501c7966
git -C cumesh apply "path/to/wheels/Windows/Torch2120/CUDA 13.2/build/patches/cumesh.patch"
PYTHON=/path/to/venv/bin/python CUDA_HOME=/usr/local/cuda-13.2 ./build/build.sh cumesh /path/to/wheels
```

The other packages build the same way without patches (for o_voxel pass the `o-voxel` subfolder of the TRELLIS.2 clone).
