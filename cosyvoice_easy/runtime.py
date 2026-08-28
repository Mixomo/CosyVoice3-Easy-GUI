from __future__ import annotations

import gc
import importlib.metadata as metadata
import importlib.util
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .audio import concat_audio, save_wav, split_long_text
from .console import log
from .paths import CACHE, MODELS, OUTPUTS, ROOT, ensure_layout
from .schemas import InferenceRequest
from .storage import append_jsonl, read_json
from .voices import load_voice

DEFAULT_MODEL = MODELS / "Fun-CosyVoice3-0.5B-2512"
MODEL_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_COSYVOICE3_PROMPT = "You are a helpful assistant.<|endofprompt|>"

# Direct runtime calls (smoke tests, workers and the GUI) all retain caches
# inside the application, not in a user-profile Hugging Face directory.
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "huggingface" / "hub"))
ensure_layout()


def condition_cross_lingual_text(text: str) -> str:
    """Add the mandatory CosyVoice3 prompt delimiter to cross-lingual text."""
    value = str(text or "")
    if "<|endofprompt|>" in value:
        return value
    return DEFAULT_COSYVOICE3_PROMPT + value


def model_complete(model_dir: str | Path = DEFAULT_MODEL, variant: str = "Base") -> bool:
    base = Path(model_dir)
    required = ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx")
    complete = all((base / name).is_file() for name in required) and (base / "CosyVoice-BlankEN").is_dir()
    if variant == "RL":
        complete = complete and (base / "llm.rl.pt").is_file()
    return complete


def download_model(model_dir: str | Path = DEFAULT_MODEL) -> str:
    from huggingface_hub import snapshot_download

    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_REPO, local_dir=destination, resume_download=True)
    if not model_complete(destination):
        raise RuntimeError("Download completed but required CosyVoice3 files are missing.")
    return str(destination)


def ensure_model_available(model_dir: str | Path = DEFAULT_MODEL, variant: str = "Base") -> str:
    """Ensure the bundled default model exists, downloading it on demand when needed."""
    selected = Path(model_dir).resolve()
    variant = "RL" if variant == "RL" else "Base"
    if model_complete(selected, variant):
        return str(selected)
    if selected != DEFAULT_MODEL.resolve():
        raise FileNotFoundError(
            "The selected custom CosyVoice3 model directory is incomplete. "
            "Use the bundled default model path or repair the custom directory manually."
        )
    log("[model] Fun-CosyVoice3-0.5B-2512 is missing or incomplete; downloading/repairing on demand.")
    download_model(selected)
    if not model_complete(selected, variant):
        missing = "llm.rl.pt" if variant == "RL" else "required model files"
        raise FileNotFoundError(f"The downloaded CosyVoice3 model is still missing {missing}.")
    return str(selected)


def resolve_flash_attention(enabled: bool, fp16: bool = True) -> tuple[bool, str]:
    """Return only a verified Transformers FlashAttention 2 configuration."""
    if not enabled:
        return False, "FlashAttention 2 disabled; using the official SDPA path."
    if not fp16:
        return False, "FlashAttention 2 requires FP16; using the official SDPA path."
    if importlib.util.find_spec("flash_attn") is None:
        return False, "FlashAttention 2 is not installed; using the official SDPA path."
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "FlashAttention 2 requires CUDA; using the official SDPA path."
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False, "FlashAttention 2 requires an Ampere-or-newer GPU; using the official SDPA path."
        version = metadata.version("flash-attn")
        if not version:
            return False, "FlashAttention 2 installation metadata is incomplete; using the official SDPA path."
        from flash_attn import flash_attn_func  # noqa: F401
    except Exception as exc:
        return False, f"FlashAttention 2 probe failed ({exc}); using the official SDPA path."
    return True, "FlashAttention 2 enabled."


def compile_backbone(model, enabled: bool, mode: str) -> tuple[object, bool, str]:
    """Compile only CosyVoice3's Qwen backbone with a clean eager fallback."""
    if not enabled:
        return model, False, "torch.compile disabled; using eager execution."
    try:
        import torch
        if not hasattr(torch, "compile"):
            return model, False, "torch.compile is unavailable in this PyTorch build; using eager execution."
        selected_mode = mode if mode in {"default", "reduce-overhead"} else "default"
        # reduce-overhead enables CUDA graphs internally. Variable-length
        # streaming calls are unsafe with CUDA-graph capture on Windows, so
        # compile through the stable Inductor path there.
        effective_mode = "default" if os.name == "nt" and selected_mode == "reduce-overhead" else selected_mode
        # CosyVoice streams generation from a background thread. Inductor's
        # CUDA-graph tree manager is thread-local and raises an assertion in
        # that path, so keep the compiled kernels but disable CUDA graphs.
        try:
            torch._inductor.config.triton.cudagraphs = False
            torch._inductor.config.triton.cudagraph_trees = False
        except Exception:
            pass
        # PyTorch 2.8 rejects passing ``mode`` and ``options`` together.  The
        # Inductor setting above disables CUDA graphs while retaining the
        # selected compile mode.
        # CosyVoice feeds variable-length token windows on every decoding
        # step. Dynamic shape guards avoid static Triton specializations (and
        # the Windows integer-overflow path seen with those kernels).
        compiled = torch.compile(model, mode=effective_mode, dynamic=True)
        mode_note = f" (Windows-safe effective mode: {effective_mode})" if effective_mode != selected_mode else ""
        return _RuntimeCompileFallback(compiled, model), True, f"torch.compile enabled ({selected_mode}){mode_note}."
    except Exception as exc:
        return model, False, f"torch.compile setup failed ({exc}); using eager execution."


class _RuntimeCompileFallback(torch.nn.Module):
    """Call Inductor until its first real runtime failure, then stay eager."""

    def __init__(self, compiled, eager) -> None:
        super().__init__()
        self._compiled = compiled
        self._eager = eager
        self._use_eager = False
        self._reported = False

    def __getattr__(self, name):
        # Preserve Qwen/PEFT attribute access expected by the surrounding
        # CosyVoice model while keeping the callable dispatch under control.
        try:
            return super().__getattr__(name)
        except AttributeError:
            eager = object.__getattribute__(self, "_modules").get("_eager")
            if eager is None:
                raise
            return getattr(eager, name)

    def forward(self, *args, **kwargs):
        if self._use_eager:
            return self._eager(*args, **kwargs)
        try:
            return self._compiled(*args, **kwargs)
        except Exception as exc:
            self._use_eager = True
            if not self._reported:
                log(f"torch.compile runtime failure; falling back to eager execution: {exc}", "WARN")
                self._reported = True
            return self._eager(*args, **kwargs)



class EngineManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine = None
        self._signature: tuple = ()
        self._cancel = threading.Event()

    def load(self, model_dir: str, fp16: bool, load_trt: bool, adapter_dir: str = "", variant: str = "Base",
             use_flash_attention: bool = False, use_torch_compile: bool = False, compile_mode: str = "default") -> str:
        variant = "RL" if variant == "RL" else "Base"
        flash_enabled, flash_message = resolve_flash_attention(use_flash_attention, fp16)
        selected_compile_mode = compile_mode if compile_mode in {"default", "reduce-overhead"} else "default"
        signature = (str(Path(model_dir).resolve()), bool(fp16), bool(load_trt), str(Path(adapter_dir).resolve()) if adapter_dir else "", variant,
                     flash_enabled, bool(use_torch_compile), selected_compile_mode)
        with self._lock:
            if self._engine is not None and signature == self._signature:
                return "CosyVoice3 is already loaded."
            ensure_model_available(model_dir, variant)
            self.unload()
            matcha = ROOT / "third_party" / "Matcha-TTS"
            if str(matcha) not in sys.path:
                sys.path.insert(0, str(matcha))
            from cosyvoice.cli.cosyvoice import AutoModel
            previous_attention = os.environ.get("COSYVOICE_ATTN_IMPLEMENTATION")
            if flash_enabled:
                os.environ["COSYVOICE_ATTN_IMPLEMENTATION"] = "flash_attention_2"
            else:
                os.environ.pop("COSYVOICE_ATTN_IMPLEMENTATION", None)
            try:
                self._engine = AutoModel(model_dir=model_dir, fp16=fp16, load_trt=load_trt, load_vllm=False)
                if self._engine.__class__.__name__ != "CosyVoice3":
                    self.unload()
                    raise TypeError("Only Fun-CosyVoice3 models are accepted by this application.")
            except Exception:
                if load_trt:
                    self._engine = AutoModel(model_dir=model_dir, fp16=fp16, load_trt=False, load_vllm=False)
                    load_trt = False
                    signature = (signature[0], signature[1], False, *signature[3:])
                else:
                    raise
            finally:
                if previous_attention is None:
                    os.environ.pop("COSYVOICE_ATTN_IMPLEMENTATION", None)
                else:
                    os.environ["COSYVOICE_ATTN_IMPLEMENTATION"] = previous_attention
            if variant == "RL":
                self._load_rl_weights(Path(model_dir) / "llm.rl.pt")
            if adapter_dir:
                self._apply_adapter(adapter_dir)
                log(f"LoRA adapter loaded: {Path(adapter_dir).resolve()}")
            else:
                log("LoRA adapter: none selected; using the base model.")
            backbone = self._engine.model.llm.llm.model
            compiled, compiled_enabled, compile_message = compile_backbone(backbone, use_torch_compile, selected_compile_mode)
            if compiled_enabled:
                self._engine.model.llm.llm.model = compiled
            if use_torch_compile and compiled_enabled:
                log(f"torch.compile verifier: ACTIVE ({selected_compile_mode}); CUDA graphs disabled; compiled callable={callable(compiled)}")
            elif use_torch_compile:
                log(f"torch.compile verifier: FALLBACK to eager ({selected_compile_mode}). {compile_message}", "WARN")
            else:
                log("torch.compile verifier: DISABLED; using eager execution.")
            self._signature = signature
            return f"Loaded CosyVoice3 {variant} ({'FP16' if fp16 else 'FP32'}, TensorRT={'on' if load_trt else 'off'}). {flash_message} {compile_message}"

    def unload(self) -> str:
        with self._lock:
            self._engine = None
            self._signature = ()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return "Model unloaded and GPU cache released."

    def cancel(self) -> str:
        self._cancel.set()
        log("Stop requested for the active synthesis.", "WARN")
        return "Stop requested. The current synthesis will stop at the next safe chunk."

    def generate(self, request: InferenceRequest, model_dir: str, fp16: bool, load_trt: bool,
                 adapter_dir: str = "", variant: str = "Base", use_flash_attention: bool = False,
                 use_torch_compile: bool = False, compile_mode: str = "default", progress=None) -> tuple[str, str]:
        if adapter_dir:
            self._validate_adapter(adapter_dir, model_dir, variant)
        if progress is not None:
            progress(0.10, desc="Checking CosyVoice3 model files")
        self.load(model_dir, fp16, load_trt, adapter_dir, variant, use_flash_attention, use_torch_compile, compile_mode)
        if progress is not None:
            progress(0.22, desc="Runtime loaded; preparing reference voice")
        self._cancel.clear()
        voice = load_voice(request.voice) if request.voice else None
        reference = request.reference_audio or (voice.audio if voice else "")
        prompt_text = request.prompt_text or (voice.transcript if voice else "")
        if not reference:
            raise ValueError("Select a saved voice or reference audio.")
        text = request.text.strip()
        chunks = split_long_text(text, request.chunk_mode)
        if not chunks:
            raise ValueError("Target text is empty.")
        import torch
        torch.manual_seed(int(request.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(request.seed))
        pieces: list[tuple[int, np.ndarray]] = []
        started = time.time()
        with self._lock:
            for chunk_index, chunk in enumerate(chunks, 1):
                if progress is not None:
                    base = 0.26 + 0.64 * ((chunk_index - 1) / max(len(chunks), 1))
                    progress(base, desc=f"Generating chunk {chunk_index}/{len(chunks)}")
                if self._cancel.is_set():
                    raise RuntimeError("Synthesis cancelled by the user.")
                generation = {
                    "top_k": int(request.top_k),
                    "top_p": float(request.top_p),
                    "temperature": float(request.temperature),
                    "ras_window": int(request.ras_window),
                    "ras_repetition_threshold": float(request.ras_repetition_threshold),
                    "min_token_text_ratio": float(request.min_token_text_ratio),
                    "max_token_text_ratio": float(request.max_token_text_ratio),
                    "flow_steps": int(request.flow_steps),
                    "flow_temperature": float(request.flow_temperature),
                }
                common = {
                    "stream": False,
                    "speed": float(request.speed),
                    "text_frontend": bool(request.text_frontend),
                    "language": str(request.language or "Auto"),
                    **generation,
                }
                if request.mode == "Cross-lingual":
                    iterator = self._engine.inference_cross_lingual(condition_cross_lingual_text(chunk), reference, **common)
                elif request.mode == "Instruct":
                    instruction = request.instruction.strip() or DEFAULT_COSYVOICE3_PROMPT
                    if "<|endofprompt|>" not in instruction:
                        instruction += "<|endofprompt|>"
                    iterator = self._engine.inference_instruct2(chunk, instruction, reference, **common)
                else:
                    if not prompt_text.strip():
                        raise ValueError("Zero-shot mode requires the reference transcript.")
                    conditioned = prompt_text.strip()
                    if "<|endofprompt|>" not in conditioned:
                        conditioned = "You are a helpful assistant.<|endofprompt|>" + conditioned
                    iterator = self._engine.inference_zero_shot(chunk, conditioned, reference, **common)
                try:
                    # The upstream wetext parser uses an assertion for empty
                    # FST input.  Convert it here as well because inference
                    # methods are generators and the exception is raised only
                    # while they are consumed, outside the iterator call.
                    samples = [item["tts_speech"].detach().float().cpu().numpy().reshape(-1) for item in iterator]
                except AssertionError as exc:
                    raise ValueError(
                        "Text Frontend could not normalize this block (wetext received an "
                        "empty segment; this commonly happens with very long text). Split the "
                        "text using 'Paragraph/Sentence Auto' or disable Text Frontend."
                    ) from exc
                if samples:
                    pieces.append((self._engine.sample_rate, np.concatenate(samples)))
        if progress is not None:
            progress(0.92, desc="Merging generated audio")
        sr, audio = concat_audio(pieces, request.gap_seconds)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = save_wav(OUTPUTS / f"cosyvoice3-{stamp}.wav", sr, audio)
        elapsed = time.time() - started
        log(f"Generated {len(audio) / sr:.2f}s using {variant} CosyVoice3 in {elapsed:.2f}s.")
        log(f"Inference verifier: {len(chunks)} chunk(s), {elapsed:.2f}s wall time; torch.compile={'requested' if use_torch_compile else 'disabled'}.")
        entry = {**asdict(request), "output": output, "sample_rate": sr, "duration": len(audio) / sr,
                 "elapsed": elapsed, "created_at": datetime.now(timezone.utc).isoformat()}
        append_jsonl(OUTPUTS / "history.jsonl", entry)
        if progress is not None:
            progress(1.0, desc="Inference complete")
        return output, f"Generated {len(audio) / sr:.2f}s in {elapsed:.2f}s ({len(chunks)} chunk(s))."

    @staticmethod
    def _validate_adapter(adapter_dir: str, model_dir: str, variant: str = "Base") -> None:
        adapter = Path(adapter_dir)
        metadata = read_json(adapter / "adapter_metadata.json", {})
        if not metadata:
            metadata = read_json(adapter.parent / "adapter_metadata.json", {})
        if not metadata:
            raise ValueError("Adapter metadata is missing.")
        expected = Path(model_dir).name
        actual = Path(str(metadata.get("base_model", ""))).name
        if actual and actual != expected:
            raise ValueError(f"Adapter base model '{actual}' does not match '{expected}'.")
        trained_variant = str(metadata.get("base_variant", "")).strip()
        if trained_variant and trained_variant != ("RL" if variant == "RL" else "Base"):
            raise ValueError(
                f"Adapter was trained on the {trained_variant} checkpoint; select the matching runtime checkpoint."
            )

    def _apply_adapter(self, adapter_dir: str) -> None:
        from peft import PeftModel

        encoder = self._engine.model.llm.llm
        if not hasattr(encoder, "model"):
            raise RuntimeError("Unexpected CosyVoice3 Qwen encoder structure.")
        peft_model = PeftModel.from_pretrained(encoder.model, adapter_dir, is_trainable=False)
        peft_model.eval()
        qwen = peft_model.model
        if not hasattr(qwen, "embed_tokens") and hasattr(qwen, "model"):
            object.__setattr__(qwen, "embed_tokens", qwen.model.embed_tokens)
        encoder.model = peft_model

    def _load_rl_weights(self, checkpoint: Path) -> None:
        import torch

        state = torch.load(checkpoint, map_location=self._engine.model.device, weights_only=True)
        self._engine.model.llm.load_state_dict(state, strict=True)
        self._engine.model.llm.to(self._engine.model.device).eval()


ENGINE = EngineManager()


def clean_app_cache() -> str:
    removed = 0
    root = CACHE.resolve()
    for child in list(root.iterdir()):
        resolved = child.resolve()
        if resolved.parent != root:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    return f"Removed {removed} app-cache entries. Models, voices and outputs were preserved."
