@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CosyVoice3 Easy GUI

set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"
set "UV_CACHE_DIR=%CD%\.runtime\uv-cache"
set "UV_NO_CACHE=1"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"
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
set "PYTHONPATH=%CD%;%CD%\third_party\Matcha-TTS"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "CUDA_MODULE_LOADING=LAZY"
rem Keep triton-windows on its bundled CUDA 12.8 toolchain for this process.
set "CUDA_PATH="
set "CUDA_HOME="

for %%D in (
  ".runtime\temp"
  ".runtime\cache"
  "%HF_HOME%"
  "%HF_XET_CACHE%"
  "%GRADIO_TEMP_DIR%"
  "models"
  "voices"
  "outputs"
  "logs"
  "config"
  "training\projects"
  "training\datasets"
  "training\outputs"
) do if not exist "%%~D" mkdir "%%~D"

if not exist "%PY_EXE%" (
  echo [ERROR] Project environment not installed. Run "1- install.bat" first.
  pause
  exit /b 1
)
"%PY_EXE%" --version >nul 2>&1 || (
  echo [ERROR] The project Python is broken. Run "1- install.bat" to repair it.
  pause
  exit /b 1
)

rem Put project-local PyTorch CUDA/cuDNN DLLs first. ONNX Runtime also uses an
rem explicit torch/lib preload in Python, so a global CUDA Toolkit cannot win.
set "PATH=%CD%\.venv\Lib\site-packages\torch\lib;%PATH%"

"%PY_EXE%" tools\verify_runtime.py --require-cuda --require-asr-cuda --require-onnx-cuda
if errorlevel 1 (
  echo [ERROR] The frozen CUDA/ONNX/Faster-Whisper runtime failed verification.
  echo The detailed report above identifies the failing import or accelerator.
  echo Run "1- install.bat" to repair it.
  pause
  exit /b 1
)
"%PY_EXE%" tools\verify_accelerators.py --strict --compile-smoke
if errorlevel 1 (
  echo [ERROR] One or more fixed accelerators are missing or incompatible.
  echo The detailed report above identifies the package or probe that failed.
  echo Run "1- install.bat" to repair them.
  pause
  exit /b 1
)

"%PY_EXE%" "%CD%\easy_gui.py"
if errorlevel 1 (
  echo.
  echo [ERROR] CosyVoice3 Easy GUI exited with an error.
  pause
)
