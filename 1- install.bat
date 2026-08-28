@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CosyVoice3 Easy GUI - Frozen CUDA 12.8 Installer

set "UV_VERSION=0.11.33"
set "PYTHON_VERSION=3.11.15"
set "LOCK_CUTOFF=2026-08-23T19:00:00Z"
set "TORCH_VERSION=2.8.0+cu128"
set "TORCHAUDIO_VERSION=2.8.0+cu128"
set "CUDA_RUNTIME=12.8"
set "CTRANSLATE2_VERSION=4.8.1"
set "ORT_VERSION=1.26.0"
set "TRITON_VERSION=3.4.0.post21"
set "FA_VERSION=2.8.3"
set "TRT_VERSION=10.13.3.9"
set "FA_WHEEL_NAME=flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl"
set "FA_WHEEL_URL=https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/%FA_WHEEL_NAME%"

set "UV_DIR=%CD%\.runtime\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "UV_PYTHON_INSTALL_DIR=%CD%\.runtime\python"
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"
set "UV_CACHE_DIR=%CD%\.runtime\uv-cache"
set "UV_NO_CACHE=1"
set "UV_LINK_MODE=copy"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "CUDA_MODULE_LOADING=LAZY"
rem Never let a machine-wide CUDA Toolkit override the project-local CUDA/cuDNN DLLs.
rem These variables are cleared only inside this BAT process.
set "CUDA_PATH="
set "CUDA_HOME="
set "TMP=%CD%\.runtime\temp"
set "TEMP=%CD%\.runtime\temp"
set "HF_HOME=%CD%\.runtime\cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "MODELSCOPE_CACHE=%CD%\.runtime\cache\modelscope"
set "GRADIO_TEMP_DIR=%CD%\.runtime\gradio-temp"
set "TORCH_EXTENSIONS_DIR=%CD%\.runtime\torch-extensions"
set "TRITON_CACHE_DIR=%CD%\.runtime\triton-cache"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\.runtime\torchinductor-cache"

for %%D in (
  ".runtime"
  ".runtime\temp"
  ".runtime\cache"
  "%UV_DIR%"
  "%HF_HOME%"
  "%HF_XET_CACHE%"
  "%MODELSCOPE_CACHE%"
  "%GRADIO_TEMP_DIR%"
  "%TORCH_EXTENSIONS_DIR%"
  "%TRITON_CACHE_DIR%"
  "%TORCHINDUCTOR_CACHE_DIR%"
  "models"
  "voices"
  "outputs"
  "logs"
  "config"
  "training\projects"
  "training\datasets"
  "training\outputs"
  "accelerator_wheels"
) do if not exist "%%~D" mkdir "%%~D"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [ERROR] NVIDIA driver / nvidia-smi was not detected.
  goto :fail
)
echo [1/8] NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo [2/8] Checking project-local uv %UV_VERSION%...
if not exist "%UV_EXE%" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $zip=Join-Path $env:TEMP 'cosyvoice-uv.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip; Expand-Archive -Force $zip '%UV_DIR%'; Remove-Item -Force $zip"
  if errorlevel 1 goto :fail
)
for /f "tokens=2" %%V in ('"%UV_EXE%" --version') do set "CURRENT_UV=%%V"
if /I not "%CURRENT_UV%"=="%UV_VERSION%" (
  echo [uv] Replacing project-local uv %CURRENT_UV% with %UV_VERSION%...
  del /q "%UV_EXE%" >nul 2>nul
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $zip=Join-Path $env:TEMP 'cosyvoice-uv.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip; Expand-Archive -Force $zip '%UV_DIR%'; Remove-Item -Force $zip"
  if errorlevel 1 goto :fail
)
"%UV_EXE%" --version || goto :fail

echo [3/8] Checking project-local Python %PYTHON_VERSION%...
set "CURRENT_PYTHON="
if exist "%PY_EXE%" for /f "delims=" %%V in ('"%PY_EXE%" -c "import sys; print('.'.join(map(str,sys.version_info[:3])))" 2^>nul') do set "CURRENT_PYTHON=%%V"
if /I "%CURRENT_PYTHON%"=="%PYTHON_VERSION%" (
  echo Project-local Python %PYTHON_VERSION% already present.
) else (
  "%UV_EXE%" python install %PYTHON_VERSION% --no-cache --no-bin --no-registry || goto :fail
)

echo [4/8] Validating / creating a clean canonical uv.lock...
set "LOCK_PY_EXE="
for /f "usebackq delims=" %%P in (`"%UV_EXE%" python find %PYTHON_VERSION%`) do set "LOCK_PY_EXE=%%P"
if not defined LOCK_PY_EXE (
  echo [ERROR] Could not locate project-managed Python %PYTHON_VERSION%.
  goto :fail
)
"%LOCK_PY_EXE%" -c "import platform, sys; assert sys.platform == 'win32'; assert platform.machine() == 'AMD64', platform.machine()" || (
  echo [ERROR] This frozen runtime supports Windows x64 / AMD64 only.
  goto :fail
)
"%UV_EXE%" lock --check --python "%LOCK_PY_EXE%" >nul 2>&1
if errorlevel 1 (
  echo [lock] Existing uv.lock is stale, universal, or missing.
  echo [lock] Deleting it and resolving a fresh Windows-only CUDA 12.8 lock from pyproject.toml...
  if exist "uv.lock" del /f /q "uv.lock"
  "%UV_EXE%" lock --python "%LOCK_PY_EXE%" --exclude-newer "%LOCK_CUTOFF%" --no-cache || goto :fail
) else (
  echo Canonical Windows-only uv.lock is already current. No dependency resolution needed.
)
"%LOCK_PY_EXE%" tools\verify_lock_contract.py || goto :fail

echo [5/8] Synchronizing the complete frozen CUDA core from uv.lock...
"%UV_EXE%" sync --frozen --no-cache --python %PYTHON_VERSION% || goto :fail
if not exist "%PY_EXE%" goto :fail
set "PATH=%CD%\.venv\Lib\site-packages\torch\lib;%PATH%"
"%PY_EXE%" -c "import importlib.metadata as m, torch, onnxruntime as ort; assert torch.__version__=='%TORCH_VERSION%'; assert torch.version.cuda=='%CUDA_RUNTIME%'; assert m.version('torchaudio')=='%TORCHAUDIO_VERSION%'; assert m.version('onnxruntime-gpu')=='%ORT_VERSION%'; assert m.version('onnxruntime')=='%ORT_VERSION%'; assert ort.__version__=='%ORT_VERSION%'; assert m.version('ctranslate2')=='%CTRANSLATE2_VERSION%'; assert m.version('triton-windows')=='%TRITON_VERSION%'; assert torch.backends.cudnn.version() and torch.backends.cudnn.version()//10000==9" || goto :fail

echo [6/8] Checking exact FlashAttention %FA_VERSION% Windows wheel...
"%PY_EXE%" -c "import importlib.metadata as m; v=m.version('flash-attn'); assert v and v.split('+')[0]=='%FA_VERSION%'; import flash_attn; from flash_attn import flash_attn_func" >nul 2>&1
if errorlevel 1 call :install_flash
if errorlevel 1 goto :fail

echo [7/8] Checking exact TensorRT CUDA 12 runtime %TRT_VERSION%...
"%PY_EXE%" -c "import importlib.metadata as m; assert m.version('tensorrt-cu12-bindings')=='%TRT_VERSION%'; assert m.version('tensorrt-cu12-libs')=='%TRT_VERSION%'; import tensorrt as trt; assert trt.Builder(trt.Logger(trt.Logger.ERROR))" >nul 2>&1
if errorlevel 1 (
  rem NVIDIA publishes the Windows bindings/libs through its TensorRT packaging path;
  rem keep this exact post-sync exception rather than contaminating the canonical uv graph.
  "%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps --reinstall "tensorrt-cu12-libs==%TRT_VERSION%" "tensorrt-cu12-bindings==%TRT_VERSION%" "tensorrt-cu12==%TRT_VERSION%" || goto :fail
)

echo [8/8] Verifying CUDA, ONNX GPU, Faster-Whisper and all accelerators...
"%PY_EXE%" tools\verify_runtime.py --require-cuda --require-asr-cuda --require-onnx-cuda || goto :fail
"%PY_EXE%" tools\verify_accelerators.py --strict --compile-smoke || goto :fail

echo.
echo Environment ready.
echo Frozen core: uv.lock / Python %PYTHON_VERSION% / PyTorch %TORCH_VERSION% / CUDA %CUDA_RUNTIME% / cuDNN 9.
echo ONNX Runtime GPU: %ORT_VERSION% with local metadata shim for Faster-Whisper.
echo CTranslate2: %CTRANSLATE2_VERSION%.
echo Triton Windows: %TRITON_VERSION%.
echo FlashAttention: %FA_VERSION% exact external wheel.
echo TensorRT CUDA 12: %TRT_VERSION% exact external NVIDIA package set.
echo Global CUDA_PATH/CUDA_HOME were ignored only for this installer process.
echo Run "2- run.bat". Model files download automatically on first use.
pause
exit /b 0

:install_flash
rem A cancelled/locked install can leave only flash_attn*.pyc and an empty
rem .dist-info directory. Remove that exact distribution before reinstalling
rem the known-good wheel so uv cannot treat the partial install as current.
"%UV_EXE%" pip uninstall --python "%PY_EXE%" --yes flash-attn >nul 2>&1
for %%F in ("accelerator_wheels\%FA_WHEEL_NAME%") do if exist "%%~fF" (
  echo [FlashAttention] Using local wheel: %%~nxF
  "%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps --reinstall "%%~fF"
  exit /b %errorlevel%
)
echo [FlashAttention] Downloading fixed wheel from kingbri1/flash-attention...
"%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps --reinstall "%FA_WHEEL_URL%"
exit /b %errorlevel%

:fail
echo.
echo [ERROR] Installation, clean lock generation or accelerator verification failed.
echo The system CUDA Toolkit and global Python installation were not modified.
pause
exit /b 1
