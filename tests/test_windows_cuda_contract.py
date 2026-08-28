import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_windows_cuda_matrix():
    matrix = json.loads((ROOT / "config" / "runtime_windows_cuda128.json").read_text(encoding="utf-8"))
    assert matrix["python"] == "3.11.15"
    assert matrix["cuda_runtime"] == "12.8"
    assert matrix["cudnn_major"] == 9
    assert matrix["global_cuda_toolkit_required"] is False
    assert matrix["packages"]["torch"] == "2.8.0+cu128"
    assert matrix["packages"]["torchaudio"] == "2.8.0+cu128"
    assert matrix["packages"]["onnxruntime-gpu"] == "1.26.0"
    assert matrix["packages"]["ctranslate2"] == "4.8.1"
    assert matrix["packages"]["triton-windows"] == "3.4.0.post21"
    assert matrix["packages"]["tensorrt-cu12"] == "10.13.3.9"


def test_installer_uses_canonical_lock_then_exact_external_exceptions():
    bat = (ROOT / "1- install.bat").read_text(encoding="utf-8")
    assert "lock --check" in bat
    assert "--exclude-newer" in bat
    assert "tools\\verify_lock_contract.py" in bat
    assert "sync --frozen --no-cache --python" in bat
    assert "--no-install-package onnxruntime" not in bat
    assert "Applying the frozen CUDA 12.8 core overlay" not in bat
    assert "torch==%TORCH_VERSION%" not in bat
    assert "onnxruntime-gpu==%ORT_VERSION%" not in bat
    assert "ctranslate2==%CTRANSLATE2_VERSION%" not in bat
    assert "triton-windows==%TRITON_VERSION%" not in bat
    assert "tensorrt-cu12==%TRT_VERSION%" in bat
    assert "--compile-smoke" in bat


def test_launchers_isolate_global_cuda_hints():
    install = (ROOT / "1- install.bat").read_text(encoding="utf-8")
    run = (ROOT / "2- run.bat").read_text(encoding="utf-8")
    for bat in (install, run):
        assert 'set "CUDA_PATH="' in bat
        assert 'set "CUDA_HOME="' in bat
        assert ".venv\\Lib\\site-packages\\torch\\lib" in bat


def test_onnx_runtime_uses_shared_cuda_provider_helper():
    frontend = (ROOT / "cosyvoice" / "cli" / "frontend.py").read_text(encoding="utf-8")
    helpers = (ROOT / "cosyvoice" / "utils" / "onnx.py").read_text(encoding="utf-8")
    export = (ROOT / "cosyvoice" / "bin" / "export_onnx.py").read_text(encoding="utf-8")
    runtime = (ROOT / "cosyvoice" / "utils" / "cuda_runtime.py").read_text(encoding="utf-8")
    assert "onnx_providers(prefer_cuda=True)" in frontend
    assert "onnx_providers(prefer_cuda=True" in helpers
    assert "onnx_providers(prefer_cuda=True)" in export
    assert '"CUDAExecutionProvider"' in runtime
    assert "preload_dlls" in runtime
