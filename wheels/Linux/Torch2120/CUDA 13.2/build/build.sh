#!/usr/bin/env bash
# usage: build_linux.sh <source dir> <output dir>
# PYTHON: python with torch 2.12 installed; CUDA_HOME: CUDA 13.2 toolkit
set -euo pipefail
PYTHON=${PYTHON:-python3}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.2}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.5;8.0;8.6;8.9;9.0;12.0+PTX"}
export MAX_JOBS=${MAX_JOBS:-4}
cd "$1"
"$PYTHON" -m pip wheel . --no-build-isolation --no-deps -w "$2" -v
