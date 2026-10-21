@echo off
rem usage: build.bat <source dir> <output dir>
rem Set PYTHON to the python.exe that has torch installed, e.g. <ComfyUI_windows_portable>\python_embeded\python.exe
if "%PYTHON%"=="" set "PYTHON=python"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if "%TORCH_CUDA_ARCH_LIST%"=="" set "TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0+PTX"
set DISTUTILS_USE_SDK=1
set MAX_JOBS=8
rem CUDA 13 CCCL refuses MSVC's traditional preprocessor
set "CL=/Zc:preprocessor"
cd /d %1
"%PYTHON%" -m pip wheel . --no-build-isolation --no-deps -w %2 -v
echo EXIT %ERRORLEVEL%
