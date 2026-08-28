from __future__ import annotations

import os
import site
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

_DLL_HANDLES: list[Any] = []
_PREPARED = False


def _prepend_path(path: Path) -> None:
    current = os.environ.get("PATH", "")
    value = str(path)
    entries = current.split(os.pathsep) if current else []
    if value not in entries:
        os.environ["PATH"] = value + (os.pathsep + current if current else "")


def prepare_local_cuda_runtime() -> dict[str, str]:
    """Prefer CUDA/cuDNN DLLs bundled with the project-local PyTorch wheel.

    This prevents ONNX Runtime, CTranslate2 and TensorRT from accidentally
    resolving an incompatible CUDA/cuDNN major from a machine-wide toolkit.
    The NVIDIA display driver remains system-provided, as required by CUDA.
    """
    global _PREPARED
    details: dict[str, str] = {}
    if _PREPARED:
        return details

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        details["torch_lib"] = str(torch_lib)
        if os.name == "nt" and torch_lib.is_dir():
            _prepend_path(torch_lib)
            try:
                _DLL_HANDLES.append(os.add_dll_directory(str(torch_lib)))
            except (AttributeError, FileNotFoundError, OSError):
                pass

        # TensorRT's Windows pip package keeps its native DLLs in a local
        # site-packages directory. Register it explicitly when present so the
        # system PATH is never required for TensorRT itself.
        if os.name == "nt":
            for base in site.getsitepackages():
                for name in ("tensorrt_libs", "tensorrt_cu12_libs"):
                    trt_lib = Path(base) / name
                    if not trt_lib.is_dir():
                        continue
                    details[f"{name}_dir"] = str(trt_lib)
                    _prepend_path(trt_lib)
                    try:
                        _DLL_HANDLES.append(os.add_dll_directory(str(trt_lib)))
                    except (AttributeError, FileNotFoundError, OSError):
                        pass
    except Exception as exc:
        details["torch_error"] = str(exc)
        return details

    # ONNX Runtime still needs its MSVC/native bootstrap even when importing
    # PyTorch has already loaded CUDA/cuDNN. Hide only ORT's redundant
    # "Skip loading ... since torch is imported" informational message.
    try:
        import onnxruntime as ort

        if os.name == "nt" and "torch_lib" in details and hasattr(ort, "preload_dlls"):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=details["torch_lib"])
            details["onnx_preload"] = "torch/lib"
    except Exception as exc:
        details["onnx_preload_error"] = str(exc)

    _PREPARED = True
    return details


def onnx_providers(prefer_cuda: bool = True, device_id: int = 0):
    prepare_local_cuda_runtime()
    import torch
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if prefer_cuda and torch.cuda.is_available() and "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", {"device_id": int(device_id)}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]
