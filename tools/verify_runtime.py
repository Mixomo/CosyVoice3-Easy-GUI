from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.utils.cuda_runtime import onnx_providers, prepare_local_cuda_runtime
from cosyvoice_easy.paths import ensure_layout
ensure_layout()

from cosyvoice_easy.runtime import DEFAULT_MODEL, model_complete


def _onnx_cuda_smoke() -> tuple[bool, str]:
    try:
        prepare_local_cuda_runtime()
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper

        providers = onnx_providers(prefer_cuda=True)
        names = [item[0] if isinstance(item, tuple) else item for item in providers]
        if not names or names[0] != "CUDAExecutionProvider":
            return False, f"CUDAExecutionProvider unavailable: {ort.get_available_providers()}"
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node], "cuda-smoke",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = min(model.ir_version, 10)
        session = ort.InferenceSession(model.SerializeToString(), providers=providers)
        active = session.get_providers()
        result = session.run(None, {"x": np.asarray([1.0], dtype=np.float32)})[0]
        ok = bool(active and active[0] == "CUDAExecutionProvider" and float(result[0]) == 1.0)
        return ok, f"providers={active}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--require-asr-cuda", action="store_true")
    parser.add_argument("--require-onnx-cuda", action="store_true")
    args = parser.parse_args()

    prepare_local_cuda_runtime()
    report = {
        "model_complete": model_complete(args.model_dir),
        "model_dir": str(Path(args.model_dir)),
    }
    imports = ["gradio", "faster_whisper", "peft", "torch", "torchaudio", "onnxruntime", "tensorrt", "cosyvoice.cli.cosyvoice"]
    errors = []
    try:
        flash_version = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        flash_version = None
    report["flash_attention_version"] = flash_version or "missing"
    if not flash_version:
        errors.append(
            "flash-attn metadata is missing or incomplete; close running Python processes and run 1- install.bat"
        )
    for module in imports:
        try:
            __import__(module)
        except Exception as exc:
            errors.append(f"{module}: {exc}")
    report["import_errors"] = errors

    asr_cuda = False
    cuda_available = False
    try:
        import torch
        import ctranslate2
        cuda_available = torch.cuda.is_available()
        asr_cuda = ctranslate2.get_cuda_device_count() > 0
        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cudnn"] = torch.backends.cudnn.version()
        report["gpu"] = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    except Exception as exc:
        report["faster_whisper_cuda_error"] = str(exc)

    onnx_cuda, onnx_detail = _onnx_cuda_smoke()
    report["faster_whisper_cuda"] = asr_cuda
    report["cuda_available"] = cuda_available
    report["onnx_cuda"] = onnx_cuda
    report["onnx_cuda_detail"] = onnx_detail

    failed = (
        bool(errors)
        or (args.require_cuda and not cuda_available)
        or (args.require_model and not model_complete(args.model_dir))
        or (args.require_asr_cuda and not asr_cuda)
        or (args.require_onnx_cuda and not onnx_cuda)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
