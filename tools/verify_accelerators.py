from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosyvoice.utils.cuda_runtime import prepare_local_cuda_runtime
from cosyvoice_easy.paths import ensure_layout
from cosyvoice_easy.ui_helpers import runtime_capabilities

ensure_layout()

EXPECTED = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "triton-windows": "3.4.0.post21",
    "flash-attn": "2.8.3",
    "onnxruntime-gpu": "1.26.0",
    "ctranslate2": "4.8.1",
    "faster-whisper": "1.2.1",
    "onnx": "1.16.0",
    "tensorrt-cu12": "10.13.3.9",
    "tensorrt-cu12-bindings": "10.13.3.9",
    "tensorrt-cu12-libs": "10.13.3.9",
}


def installed_version(name: str) -> str:
    try:
        # A partially interrupted wheel install can leave a ``.dist-info``
        # directory without a Version field.  ``importlib.metadata`` returns
        # None in that case; normalize it so the verifier reports a useful
        # missing/broken package instead of crashing on ``.split()``.
        value = metadata.version(name)
        return str(value) if value else "missing"
    except (metadata.PackageNotFoundError, TypeError, ValueError):
        return "missing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--compile-smoke", action="store_true")
    args = parser.parse_args()

    prepare_local_cuda_runtime()
    versions = {name: installed_version(name) for name in EXPECTED}
    capabilities = runtime_capabilities(deep_probe=True)
    report = {"versions": versions, "capabilities": capabilities}
    errors: list[str] = []

    try:
        matrix = json.loads((ROOT / "config" / "runtime_windows_cuda128.json").read_text(encoding="utf-8"))
        report["runtime_matrix"] = matrix
        if matrix.get("cuda_runtime") != "12.8" or matrix.get("packages", {}).get("torch") != "2.8.0+cu128":
            errors.append("runtime matrix: unexpected CUDA/PyTorch pin")
    except Exception as exc:
        errors.append(f"runtime matrix: {exc}")

    for name, expected in EXPECTED.items():
        actual = versions[name]
        if name in {"torch", "torchaudio", "flash-attn"}:
            if actual == "missing" or actual.split("+")[0] != expected:
                errors.append(f"{name}: expected base {expected}, got {actual}")
        elif actual != expected:
            errors.append(f"{name}: expected {expected}, got {actual}")

    try:
        import torch
        if torch.__version__ != "2.8.0+cu128":
            errors.append(f"torch build: expected 2.8.0+cu128, got {torch.__version__}")
        if torch.version.cuda != "12.8":
            errors.append(f"torch CUDA: expected 12.8, got {torch.version.cuda}")
        if not torch.cuda.is_available():
            errors.append("torch CUDA device unavailable")
        cudnn = torch.backends.cudnn.version()
        if not cudnn or int(cudnn) // 10000 != 9:
            errors.append(f"cuDNN: expected major 9, got {cudnn}")
    except Exception as exc:
        errors.append(f"torch probe: {exc}")

    for key in ("flash_attention", "tensorrt", "torch_compile", "onnx_cuda", "faster_whisper_cuda"):
        if not capabilities.get(key):
            errors.append(f"{key}: unavailable")

    if args.compile_smoke and capabilities.get("torch_compile"):
        try:
            import torch

            def _smoke(x):
                return torch.sin(x) + x

            try:
                torch._inductor.config.triton.cudagraphs = False
            except Exception:
                pass
            _smoke = torch.compile(_smoke, mode="default")

            x = torch.ones(64, device="cuda")
            y = _smoke(x)
            torch.cuda.synchronize()
            if y.shape != x.shape:
                raise RuntimeError("unexpected torch.compile output shape")
            report["torch_compile_smoke"] = "ok"
        except Exception as exc:
            errors.append(f"torch.compile smoke: {exc}")

    report["errors"] = errors
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
