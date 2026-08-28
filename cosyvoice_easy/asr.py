from __future__ import annotations

import gc
import os
import threading
from pathlib import Path

from cosyvoice.utils.cuda_runtime import prepare_local_cuda_runtime

_LOCK = threading.RLock()
_MODEL = None
_MODEL_NAME = ""


def transcribe(audio_path: str, model_name: str = "large-v3", language: str = "Auto", batch_size: int = 1) -> tuple[str, str]:
    global _MODEL, _MODEL_NAME
    if not audio_path:
        raise ValueError("Select audio to transcribe.")
    with _LOCK:
        if _MODEL is None or _MODEL_NAME != model_name:
            unload()
            # CTranslate2 4.8 uses CUDA 12/cuDNN 9. Prefer the CUDA/cuDNN
            # DLLs bundled by project-local PyTorch 2.8 before importing it.
            prepare_local_cuda_runtime()
            import torch
            from faster_whisper import WhisperModel
            import ctranslate2
            if not torch.cuda.is_available() or ctranslate2.get_cuda_device_count() < 1:
                raise RuntimeError("Faster-Whisper CUDA is unavailable. Run the CUDA runtime verifier; CPU fallback is intentionally disabled.")
            _MODEL = WhisperModel(model_name, device="cuda", compute_type="float16")
            _MODEL_NAME = model_name
        lang = None if not language or language == "Auto" else language
        if int(batch_size or 1) > 1:
            from faster_whisper import BatchedInferencePipeline
            segments, info = BatchedInferencePipeline(model=_MODEL).transcribe(
                audio_path, language=lang, vad_filter=True, beam_size=5, batch_size=int(batch_size)
            )
        else:
            segments, info = _MODEL.transcribe(audio_path, language=lang, vad_filter=True, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, getattr(info, "language", lang or "unknown")


def unload() -> None:
    global _MODEL, _MODEL_NAME
    _MODEL = None
    _MODEL_NAME = ""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
