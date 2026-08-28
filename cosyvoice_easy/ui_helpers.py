from __future__ import annotations

import importlib.util
import os
import secrets
from pathlib import Path

from .paths import ROOT, ensure_layout
from .instructions import MARKER, instruction_body, normalize_instruction

NONE = "None"

LANGUAGE_NAMES = {
    "Auto": "Auto-detect",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ru": "Russian",
}
LANGUAGE_CHOICES = [(label, code) for code, label in LANGUAGE_NAMES.items()]


def resolve_seed(seed: int | float | str | None, random_seed: bool) -> int:
    if random_seed:
        return secrets.randbelow(2_147_483_647)
    try:
        return max(0, min(2_147_483_647, int(seed or 0)))
    except (TypeError, ValueError):
        return 0


def language_instruction(instruction: str, language: str) -> str:
    """Build the CosyVoice3 natural-language prefix and append its control marker."""
    base = instruction_body(instruction) or "You are a helpful assistant."
    code = str(language or "Auto")
    if code != "Auto":
        name = LANGUAGE_NAMES.get(code, code)
        base = f"{base} Please speak in {name}.".strip()
    return normalize_instruction(base)


def play_completion_chime() -> None:
    """Play the shared Easy GUI completion sound without blocking the UI."""
    if os.name != "nt":
        return
    try:
        import winsound
    except ImportError:
        return
    chime = ROOT / "assets" / "chime.wav"
    if not chime.is_file():
        return
    try:
        winsound.PlaySound(
            str(chime),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except Exception:
        pass


def runtime_capabilities(deep_probe: bool = False) -> dict[str, object]:
    """Return verified optional accelerator capabilities for the frozen Windows stack."""
    # Make this helper safe to call from a standalone verifier as well as the
    # GUI.  In particular, PyTorch Inductor must not inherit a read-only
    # machine-wide TEMP/Triton cache from the parent shell.
    ensure_layout()
    result: dict[str, object] = {
        "flash_attention": False,
        "tensorrt": False,
        "torch_compile": False,
        "onnx_cuda": False,
        "faster_whisper_cuda": False,
        "torch_version": "unknown",
        "cuda_version": "unknown",
        "torch_compile_detail": "not probed",
    }
    try:
        from cosyvoice.utils.cuda_runtime import prepare_local_cuda_runtime
        prepare_local_cuda_runtime()
        import torch

        result["torch_version"] = str(torch.__version__)
        result["cuda_version"] = str(torch.version.cuda or "none")
        cuda_ok = bool(torch.cuda.is_available())
        result["flash_attention"] = bool(cuda_ok and importlib.util.find_spec("flash_attn") is not None)
        result["tensorrt"] = bool(cuda_ok and importlib.util.find_spec("tensorrt") is not None)
        result["torch_compile"] = bool(
            cuda_ok and hasattr(torch, "compile") and importlib.util.find_spec("triton") is not None
        )
        try:
            import onnxruntime as ort
            result["onnx_cuda"] = bool(cuda_ok and "CUDAExecutionProvider" in ort.get_available_providers())
        except Exception:
            pass
        if deep_probe:
            try:
                if result["flash_attention"]:
                    import flash_attn  # noqa: F401
                    from flash_attn import flash_attn_func  # noqa: F401
            except Exception:
                result["flash_attention"] = False
            try:
                if result["tensorrt"]:
                    import tensorrt as trt
                    result["tensorrt"] = bool(trt.Builder(trt.Logger(trt.Logger.ERROR)))
            except Exception:
                result["tensorrt"] = False
            try:
                import ctranslate2
                result["faster_whisper_cuda"] = bool(ctranslate2.get_cuda_device_count() > 0)
            except Exception:
                result["faster_whisper_cuda"] = False
            if result["torch_compile"]:
                try:
                    def _compile_probe(x):
                        return torch.sin(x) + x

                    try:
                        torch._inductor.config.triton.cudagraphs = False
                        torch._inductor.config.triton.cudagraph_trees = False
                    except Exception:
                        pass
                    compiled_probe = torch.compile(
                        _compile_probe,
                        mode="default",
                    )
                    probe = torch.ones(8, device="cuda")
                    output = compiled_probe(probe)
                    torch.cuda.synchronize()
                    if tuple(output.shape) != tuple(probe.shape):
                        raise RuntimeError("unexpected torch.compile probe output shape")
                    result["torch_compile_detail"] = "smoke test passed"
                except Exception as exc:
                    result["torch_compile"] = False
                    result["torch_compile_detail"] = f"smoke test failed: {exc}"
            elif cuda_ok:
                result["torch_compile_detail"] = "unavailable (PyTorch/Triton check failed)"
    except Exception:
        pass
    return result


def capability_summary(capabilities: dict[str, object]) -> str:
    def state(key: str) -> str:
        return "available" if capabilities.get(key) else "unavailable"

    return (
        f"TensorRT: {state('tensorrt')} · FlashAttention 2: {state('flash_attention')} · "
        f"torch.compile: {state('torch_compile')} · ONNX CUDA: {state('onnx_cuda')} · "
        f"PyTorch: {capabilities.get('torch_version', 'unknown')} / CUDA {capabilities.get('cuda_version', 'unknown')} · "
        f"torch.compile probe: {capabilities.get('torch_compile_detail', 'not probed')}"
    )
