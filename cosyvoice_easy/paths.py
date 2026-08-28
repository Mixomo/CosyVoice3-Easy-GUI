from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
CONFIG = ROOT / "config"
MODELS = ROOT / "models"
VOICES = ROOT / "voices"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
TRAINING = ROOT / "training"
PROJECTS = TRAINING / "projects"
DATASETS = TRAINING / "datasets"
TRAINING_OUTPUTS = TRAINING / "outputs"
CACHE = RUNTIME / "cache"
TEMP = RUNTIME / "temp"

APP_DIRS = (RUNTIME, CONFIG, MODELS, VOICES, OUTPUTS, LOGS, PROJECTS, DATASETS, TRAINING_OUTPUTS, CACHE, TEMP)


def ensure_layout() -> None:
    # The bundled PyTorch/ONNX/Triton stack is self-contained.  A machine-wide
    # CUDA Toolkit (especially CUDA 13 next to the pinned CUDA 12.8 wheels)
    # can win DLL resolution when the GUI is launched directly instead of via
    # ``2- run.bat``.  Remove those process-local hints; the CUDA runtime
    # helper will register the project-local PyTorch DLL directory instead.
    os.environ.pop("CUDA_PATH", None)
    os.environ.pop("CUDA_HOME", None)
    # Do not inherit an opt-in CUDA-graph setting from another Torch project.
    # CosyVoice streams variable-length inputs, which are unsafe to capture
    # in CUDA graphs on this Windows runtime.
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        entry for entry in path_entries
        if "nvidia gpu computing toolkit" not in entry.lower()
    )
    for path in APP_DIRS:
        path.mkdir(parents=True, exist_ok=True)
    local_env = {
        "HF_HOME": CACHE / "huggingface",
        "HF_HUB_CACHE": CACHE / "huggingface" / "hub",
        "HF_XET_CACHE": CACHE / "huggingface" / "xet",
        "MODELSCOPE_CACHE": CACHE / "modelscope",
        "XDG_CACHE_HOME": CACHE,
        "TMP": TEMP,
        "TEMP": TEMP,
        "TORCH_EXTENSIONS_DIR": RUNTIME / "torch-extensions",
        "TRITON_CACHE_DIR": RUNTIME / "triton-cache",
        "TORCHINDUCTOR_CACHE_DIR": RUNTIME / "torchinductor-cache",
    }
    # Keep compiler/temp artifacts inside the application.  ``setdefault``
    # is intentionally retained for model caches so an explicitly configured
    # HF/ModelScope cache remains usable, but compiler backends must not
    # inherit a machine-wide temporary directory that may be read-only (for
    # example ``C:\\tc\\em`` on some Windows installations).
    local_only = {
        "TMP",
        "TEMP",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    }
    for key, value in local_env.items():
        value.mkdir(parents=True, exist_ok=True)
        if key in local_only:
            os.environ[key] = str(value)
        else:
            os.environ.setdefault(key, str(value))

    # Qwen's causal-mask helper uses Tensor.item(), which intentionally causes
    # a TorchDynamo graph break for variable-length autoregressive inputs.  The
    # multiline diagnostic is harmless but overwhelms the Windows console.
    # Keep capture_scalar_outputs disabled for compiler stability.
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo.variables.tensor").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r"Graph break from `Tensor\.item\(\)`.*",
        category=UserWarning,
        module=r"torch\._dynamo.*",
    )
