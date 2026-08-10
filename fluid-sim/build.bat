@echo off
REM Build the CUDA 2D stable-fluids solver for Turing (RTX 2080, sm_75).
setlocal
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
where cl.exe >nul 2>&1 || call %VCVARS% >nul
if not exist build mkdir build
%NVCC% -O3 -arch=sm_75 -lineinfo --ptxas-options=-v ^
    -o build\fluid.exe main.cu fluid_kernels.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo Built build\fluid.exe
