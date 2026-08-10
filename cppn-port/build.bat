@echo off
REM GlassOS CPPN -- CUDA port. Build for Turing (RTX 2080 = sm_75).
setlocal
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
set CLBIN="C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

echo [1/2] nvcc cppn.cu  (--ptxas-options=-v gives register/occupancy data)
%NVCC% -O3 -arch=sm_75 -lineinfo --ptxas-options=-v -ccbin %CLBIN% -o cppn.exe cppn.cu
if errorlevel 1 exit /b 1

echo [2/2] nvcc cpu_ref.cpp  (host-only reference, double precision)
%NVCC% -O3 -x c++ -ccbin %CLBIN% -o cpu_ref.exe cpu_ref.cpp
if errorlevel 1 exit /b 1

echo done: cppn.exe, cpu_ref.exe
