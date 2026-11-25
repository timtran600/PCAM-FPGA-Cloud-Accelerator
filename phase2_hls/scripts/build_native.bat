@echo off
rem build_native.bat - native (non-HLS) build of the whole-network csim
rem testbench. Same sources and include paths as run_hls.tcl, compiled
rem directly so the full test split runs without csim harness overhead.
rem
rem   build_native.bat
rem   build\tb_native.exe [n_tiles] [artifacts_dir]
rem
rem Works from any working directory: every path is resolved from this
rem script's location (%~dp0).
rem
rem Compiler: cl if it is already on PATH (Developer prompt), otherwise
rem the MinGW g++ 8.3.0 that ships with Vitis HLS 2023.2 - the same
rem compiler csim uses. Output goes to scripts\build\tb_native.exe.

setlocal

set "HERE=%~dp0"
set "SRC=%HERE%..\src"
set "TB=%HERE%..\tb"
set "ART=%HERE%..\..\artifacts"
set "HLS_INC=C:\Xilinx\Vitis_HLS\2023.2\include"
set "MINGW_BIN=C:\Xilinx\Vitis_HLS\2023.2\tps\mingw\8.3.0\win64.o\nt\bin"
set "OUTDIR=%HERE%build"

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

where cl >nul 2>nul
if %ERRORLEVEL%==0 goto build_cl

if not exist "%MINGW_BIN%\g++.exe" (
    echo ERROR: no cl on PATH and no g++ at %MINGW_BIN%
    exit /b 1
)
set "PATH=%MINGW_BIN%;%PATH%"
echo Compiler: %MINGW_BIN%\g++.exe
g++ --version | findstr /b "g++"
rem -static: no dependency on libstdc++/libgcc/winpthread DLLs at runtime.
g++ -O2 -std=c++14 -static ^
    -I"%HLS_INC%" -I"%SRC%" -I"%ART%\weights" ^
    "%SRC%\pcam_infer.cpp" "%TB%\tb_top.cpp" ^
    -o "%OUTDIR%\tb_native.exe"
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)
goto done

:build_cl
echo Compiler: cl
cl /nologo /O2 /EHsc /std:c++14 ^
    /I"%HLS_INC%" /I"%SRC%" /I"%ART%\weights" ^
    "%SRC%\pcam_infer.cpp" "%TB%\tb_top.cpp" ^
    /Fo"%OUTDIR%\\" /Fe"%OUTDIR%\tb_native.exe"
if errorlevel 1 (
    echo BUILD FAILED
    exit /b 1
)

:done
echo Built %OUTDIR%\tb_native.exe
endlocal
