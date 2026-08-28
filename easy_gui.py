from __future__ import annotations

import html
import json
import logging
import os
import re
import shutil
import sys
import time
import warnings


class _IgnoreUvicornContentLength(logging.Filter):
    def filter(self, record):
        return "Too much data for declared Content-Length" not in record.getMessage()


logging.getLogger("uvicorn.error").addFilter(_IgnoreUvicornContentLength())
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Quiet known third-party lifecycle warnings that do not affect this pinned
# runtime. Application and accelerator warnings remain visible.
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=r".*torch\.nn\.utils\.weight_norm.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"In 2\.9, this function's implementation will be changed.*", category=UserWarning)

ROOT = Path(__file__).resolve().parent
MATCHA = ROOT / "third_party" / "Matcha-TTS"
for item in (ROOT, MATCHA):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from cosyvoice_easy.paths import CONFIG, DATASETS, OUTPUTS, PROJECTS, TRAINING_OUTPUTS, VOICES, ensure_layout

ensure_layout()
# Load the project-local Torch CUDA/cuDNN DLLs before Gradio (and its optional
# integrations) can import any native audio/runtime extension.  On Windows,
# importing Gradio first can make ONNX Runtime or Triton resolve a conflicting
# system DLL and make the capability probe report false negatives.
from cosyvoice.utils.cuda_runtime import prepare_local_cuda_runtime
prepare_local_cuda_runtime()

import gradio as gr

from cosyvoice_easy import asr
from cosyvoice_easy.datasets import (clone_project, create_project, delete_project, extract_features,
                                      delete_training_project, list_dataset_projects, list_training_projects, list_projects, prepare_dataset, save_project, scan_source, transcribe_missing)
from cosyvoice_easy.runtime import DEFAULT_MODEL, ENGINE, ensure_model_available
from cosyvoice_easy.schemas import DialogueLine, InferenceRequest, TrainingConfig
from cosyvoice_easy.storage import append_jsonl, atomic_json, read_json
from cosyvoice_easy.training import (adapter_choices, adapter_dropdown_choices, resume_checkpoint_choices, launch as launch_training, progress_snapshot as training_progress_snapshot,
                                     prepare_tensorboard_project, restart_tensorboard, start_tensorboard, status as training_status, stop as stop_training)
from cosyvoice_easy.voices import delete_voice, list_voices, load_voice, save_voice
from cosyvoice_easy.audio import CHUNK_CHOICES, concat_audio, save_wav, safe_name
from cosyvoice_easy.console import html_view, log
from cosyvoice_easy.instructions import (instruction_body, list_instructions, load_instruction, normalize_instruction,
                                         save_instruction, delete_instruction)
from cosyvoice_easy.ui_helpers import (LANGUAGE_CHOICES, NONE, capability_summary, language_instruction,
                                       play_completion_chime, resolve_seed, runtime_capabilities)

log("CosyVoice3 Easy GUI initialized.")

SETTINGS = CONFIG / "ui_settings.json"
LANGUAGES = [value for _, value in LANGUAGE_CHOICES]
ASR_LANGUAGES = ["Auto-detect", "zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"]
ASR_MODELS = ["large-v3", "medium", "small", "base"]
MAX_DIALOGUE_ROWS = 12
AUTO_PROFILE = "Auto · dataset-aware"
AUTOTUNE_PROFILES = [
    "≈ 0–30 min · Conservative (r8)",
    "≈ 30–180 min · Validated default (r16)",
    "≈ 180+ min / 4+ speakers · Higher capacity (r32)",
]
LEGACY_AUTOTUNE_PROFILES = {
    "12–16 GB (Conservative)": AUTOTUNE_PROFILES[0],
    "24 GB RTX 3090/4090": AUTOTUNE_PROFILES[1],
    "32 GB+": AUTOTUNE_PROFILES[2],
    "Small dataset / conservative (r8)": AUTOTUNE_PROFILES[0],
    "Validated default (r16)": AUTOTUNE_PROFILES[1],
    "Large diverse dataset / higher capacity (r32)": AUTOTUNE_PROFILES[2],
    AUTO_PROFILE: AUTOTUNE_PROFILES[1],
}

CSS = """
/* Native Gradio-first layout: spacing and width corrections only. */
html, body, #root { width:100%; min-width:0; }
.gradio-container { max-width:none !important; width:100% !important; margin:0 auto !important; padding:18px 28px 28px !important; }
.tabs, .tabitem, .gradio-container .block, .gradio-container .form { width:100% !important; max-width:none !important; }
.title-section { border-bottom:1px solid var(--border-color-primary); margin-bottom:8px; padding-bottom:8px; align-items:center !important; }
.title-section .prose, .title-section h1 { margin:0 !important; padding:0 !important; }
.title-section button { min-height:36px !important; white-space:nowrap; }
.tab-subtitle { opacity:.78; margin:0 0 12px 0 !important; padding:0 !important; }
.section-heading { margin:10px 0 2px !important; }
.global-toolbar, .project-strip {
    padding:4px 2px !important; border:0 !important; border-radius:0 !important; background:transparent !important; box-shadow:none !important;
}
.project-strip { padding-bottom:12px !important; margin-bottom:10px !important; border-bottom:1px solid var(--border-color-primary) !important; }
.global-toolbar { margin:4px 0 10px !important; }
.global-toolbar button { min-height:38px !important; }
.runtime-row, .workflow-actions, .dialogue-toolbar { gap:10px !important; align-items:end !important; }
.compact-button { min-width:52px !important; max-width:110px !important; }
.medium-control { max-width:420px !important; }
.small-control { max-width:260px !important; }
.dialogue-turn-card {
    padding:14px 14px 12px !important; margin:6px 0 12px !important;
    border:1px solid var(--border-color-primary) !important; border-radius:10px !important;
    background:transparent !important; box-shadow:none !important;
}
.dialogue-turn-title { margin:0 0 6px 0 !important; opacity:.86; }
.dialogue-actions { gap:7px !important; align-items:center !important; }
.dialogue-actions button { min-width:88px !important; padding-left:10px !important; padding-right:10px !important; white-space:nowrap !important; }
.audio-safe-space { overflow:visible !important; padding-bottom:18px !important; border:0 !important; box-shadow:none !important; }
.output-clean, .output-clean > div, .output-clean .wrap { border:0 !important; box-shadow:none !important; }
.compact-status { margin:0 0 8px 0 !important; padding:0 !important; min-height:0 !important; }
.console-accordion, .console-accordion > div { border-radius:8px !important; }
.training-progress-card { padding:10px 12px; border:1px solid var(--border-color-primary); border-radius:9px; background:transparent; }
.training-progress-head { display:flex; justify-content:space-between; gap:16px; margin-bottom:8px; }
.training-progress-track { width:100%; height:10px; border-radius:999px; background:var(--background-fill-secondary); overflow:hidden; }
.training-progress-fill { height:100%; border-radius:999px; background:var(--button-primary-background-fill); transition:width .25s ease; }
.training-progress-meta { margin-top:8px; opacity:.82; font-size:.92em; }
.instruction-help { opacity:.82; margin-top:2px !important; }
footer { display:none !important; }
@media(max-width:900px){ .gradio-container{padding:10px!important}.title-section button{min-width:100% !important;} .medium-control,.small-control{max-width:none!important;} }
"""


def save_settings(**values):
    current = read_json(SETTINGS, {})
    current.update(values)
    atomic_json(SETTINGS, current)


def voice_choices() -> list[str]:
    return [NONE, *list_voices()]


def adapter_values():
    return [NONE, *adapter_dropdown_choices()]

def instruction_choices() -> list[str]:
    return [NONE, *list_instructions()]


def load_instruction_ui(name: str):
    if not name or name == NONE:
        return "", "", "No saved instruction selected."
    record = load_instruction(name)
    if not record:
        return "", "", "The selected instruction is unavailable. Refresh the library."
    return name, instruction_body(record.get("text", "")), f"Loaded instruction '{name}'."


def refresh_instruction_choices_ui(*current_values):
    choices = instruction_choices()
    return [gr.update(choices=choices, value=value if value in choices else NONE) for value in current_values]


def instruction_text_for_choice(name: str) -> str:
    if not name or name == NONE:
        return ""
    record = load_instruction(name)
    return instruction_body(record.get("text", "")) if record else ""


def save_instruction_ui(name, text):
    saved, message = save_instruction(name, text)
    choices = instruction_choices()
    return gr.update(choices=choices, value=saved), saved, instruction_body(normalize_instruction(text)), message


def delete_instruction_ui(name):
    message = delete_instruction(name)
    return gr.update(choices=instruction_choices(), value=NONE), "", "", message


def mode_visibility(mode: str):
    zero = mode == "Zero-shot"
    instruct = mode == "Instruct"
    return gr.update(visible=zero), gr.update(visible=instruct)


def instruction_source_visibility(source: str):
    return gr.update(visible=source == "Custom")


def training_length_mode_ui(mode: str):
    return gr.update(visible=mode == "Steps"), gr.update(visible=mode == "Epochs")


def resolve_dataset_instruction(source: str, custom_text: str) -> str:
    if source == "Custom":
        if not str(custom_text or "").strip():
            raise ValueError("Enter a custom training instruction or choose Standard.")
        return normalize_instruction(custom_text)
    return normalize_instruction("You are a helpful assistant.")


def voice_values(name: str):
    if not name or name == NONE:
        return "", None, None, "", "", "Auto", "No saved voice selected."
    record = load_voice(name)
    if not record:
        return "", None, None, "", "", "Auto", "The selected voice is unavailable. Refresh the library."
    language = record.language if record.language in LANGUAGES else "Auto"
    return name, record.audio, record.audio, record.transcript, record.transcript, language, f"Loaded '{name}'."


def inference_voice_values(name: str):
    if not name or name == NONE:
        return None, "", "No saved voice selected; use a reference override or select a voice."
    record = load_voice(name)
    if not record:
        return None, "", "The selected voice is unavailable. Refresh the library."
    return record.audio, record.transcript, f"Loaded '{name}' with its saved transcript."


def save_voice_ui(name, audio, transcript, detected_language):
    language = detected_language if detected_language in LANGUAGES else "Auto"
    saved, path = save_voice(name, audio, transcript, language, "")
    choices = voice_choices()
    play_completion_chime()
    return gr.update(choices=choices, value=saved), saved, path, path, transcript, f"Saved voice '{saved}'."


def delete_voice_ui(name):
    if not name or name == NONE:
        return gr.update(choices=voice_choices(), value=NONE), "", None, None, "", "Select a saved voice to delete."
    delete_voice(name)
    return gr.update(choices=voice_choices(), value=NONE), "", None, None, "", f"Deleted '{name}'."


def transcribe_ui(audio, model, language, batch_size=1, progress=gr.Progress(track_tqdm=False)):
    progress(0.08, desc="Loading Faster-Whisper")
    selected_language = "Auto" if not language or language == "Auto-detect" else language
    progress(0.30, desc="Transcribing reference audio")
    text, detected = asr.transcribe(audio, model, selected_language, batch_size=int(batch_size or 1))
    progress(1.0, desc="Transcription complete")
    play_completion_chime()
    normalized = detected if detected in LANGUAGES else "Auto"
    return text, normalized, f"Transcription complete; detected language: {detected}."


def transcribe_inference_ui(audio, model, language, batch_size=1, progress=gr.Progress(track_tqdm=False)):
    text, detected, message = transcribe_ui(audio, model, language, batch_size, progress)
    return text, message


def sidecar_transcript_ui(audio, current_text=""):
    """Load a same-basename .txt transcript when direct audio has one."""
    if not audio:
        return current_text or "", "No reference audio selected."
    try:
        path = Path(str(audio))
        sidecar = path.with_suffix(".txt")
        if sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8-sig", errors="replace").strip()
            if text:
                return text, f"Loaded sidecar transcript: {sidecar.name}."
    except Exception as exc:
        log(f"Could not inspect reference transcript sidecar: {exc}", "WARN")
    return current_text or "", "Reference audio loaded. Use its saved/sidecar transcript or Faster-Whisper when needed."


def running_button_updates(running: bool):
    return gr.update(interactive=not running), gr.update(interactive=running)


def generate_ui(text, mode, voice, ref_audio, prompt, instruction, language, seed, random_seed,
                speed, text_frontend, top_k, top_p, temperature, ras_window, ras_threshold,
                min_token_ratio, max_token_ratio, flow_steps, flow_temperature, chunk_mode, gap,
                model_dir, model_variant, fp16, trt, flash_attention, torch_compile, adapter,
                progress=gr.Progress(track_tqdm=False)):
    def friendly_error(exc):
        """Turn backend/parser failures into a useful status message in the GUI."""
        detail = str(exc).strip()
        if isinstance(exc, AssertionError) or "TokenParser" in detail or "len(input)" in detail:
            return (
                "Text Frontend could not process this block (wetext received an empty segment; "
                "this commonly happens with very long text). Split the text using "
                "'Paragraph/Sentence Auto' or disable Text Frontend and try again."
            )
        return detail or "No se pudo generar el audio. Revisa el texto y la referencia de voz."

    compile_mode = "default"
    actual_seed = resolve_seed(seed, random_seed)
    selected_voice = "" if not voice or voice == NONE else voice
    selected_adapter = "" if not adapter or adapter == NONE else adapter
    if float(max_token_ratio) <= float(min_token_ratio):
        raise ValueError("Max Token/Text Ratio must be greater than Min Token/Text Ratio.")
    selected_instruction = instruction or ""
    if mode == "Instruct":
        selected_instruction = language_instruction(selected_instruction, language)
    log(f"Starting {mode} inference ({model_variant} checkpoint, seed={actual_seed}).")
    request = InferenceRequest(
        text=text, voice=selected_voice, reference_audio=ref_audio or "", prompt_text=prompt or "",
        instruction=selected_instruction, language=language, mode=mode,
        seed=actual_seed, speed=float(speed), chunk_mode=chunk_mode if chunk_mode in CHUNK_CHOICES else "None",
        gap_seconds=float(gap), text_frontend=bool(text_frontend), top_k=int(top_k), top_p=float(top_p),
        temperature=float(temperature), ras_window=int(ras_window), ras_repetition_threshold=float(ras_threshold),
        min_token_text_ratio=float(min_token_ratio), max_token_text_ratio=float(max_token_ratio),
        flow_steps=int(flow_steps), flow_temperature=float(flow_temperature),
    )
    save_settings(inference=asdict(request), model_variant=model_variant, fp16=fp16, trt=trt,
                  flash_attention=flash_attention, torch_compile=torch_compile, compile_mode=compile_mode,
                  adapter=selected_adapter)
    try:
        output, status = ENGINE.generate(
            request, model_dir, fp16, trt, selected_adapter, model_variant, bool(flash_attention),
            bool(torch_compile), compile_mode, progress=progress,
        )
    except (AssertionError, ValueError) as exc:
        message = friendly_error(exc)
        log(f"Inference could not be completed: {message}", "WARN")
        return None, f"❌ {message}"
    status += f" Seed: {actual_seed}."
    play_completion_chime()
    return output, status

def dialogue_visibility(count):
    return [gr.update(visible=index < int(count)) for index in range(MAX_DIALOGUE_ROWS)]


def dialogue_mode_visibility(mode: str):
    return gr.update(visible=mode == "Instruct")


def _dialogue_row_updates(rows):
    updates = []
    for mode, speaker, adapter, text, instruction in rows:
        updates.extend([mode, speaker, adapter, text, gr.update(value=instruction, visible=mode == "Instruct")])
    return updates


def dialogue_row_action(action: str, index: int, count: int, *values):
    """Index/FireRed-style row editing with mode-aware CosyVoice3 turn fields."""
    count = max(1, min(MAX_DIALOGUE_ROWS, int(count)))
    rows = [list(values[offset:offset + 5]) for offset in range(0, len(values), 5)]
    rows += [["Zero-shot", NONE, NONE, "", ""] for _ in range(MAX_DIALOGUE_ROWS - len(rows))]
    index = max(0, min(MAX_DIALOGUE_ROWS - 1, int(index)))
    if action == "add" and count < MAX_DIALOGUE_ROWS:
        rows.insert(index + 1, ["Zero-shot", NONE, NONE, "", ""])
        rows = rows[:MAX_DIALOGUE_ROWS]
        count += 1
    elif action == "clone" and count < MAX_DIALOGUE_ROWS:
        rows.insert(index + 1, list(rows[index]))
        rows = rows[:MAX_DIALOGUE_ROWS]
        count += 1
    elif action == "delete" and count > 1:
        rows.pop(index)
        rows.append(["Zero-shot", NONE, NONE, "", ""])
        count -= 1
    elif action == "clear":
        rows[index] = ["Zero-shot", NONE, NONE, "", ""]
    elif action in {"up", "down"}:
        other = index - 1 if action == "up" else index + 1
        if 0 <= other < count:
            rows[index], rows[other] = rows[other], rows[index]
    rows = rows[:MAX_DIALOGUE_ROWS]
    while len(rows) < MAX_DIALOGUE_ROWS:
        rows.append(["Zero-shot", NONE, NONE, "", ""])
    visible = [gr.update(visible=i < count) for i in range(MAX_DIALOGUE_ROWS)]
    return [count, *visible, *_dialogue_row_updates(rows)]


def dialogue_toolbar_action(action: str, count: int, *values):
    rows = [list(values[offset:offset + 5]) for offset in range(0, len(values), 5)]
    rows += [["Zero-shot", NONE, NONE, "", ""] for _ in range(MAX_DIALOGUE_ROWS - len(rows))]
    count = max(1, min(MAX_DIALOGUE_ROWS, int(count)))
    if action == "reset":
        count = 2
        rows = [["Zero-shot", NONE, NONE, "", ""] for _ in range(MAX_DIALOGUE_ROWS)]
    elif action == "clear":
        for index in range(count):
            rows[index] = ["Zero-shot", NONE, NONE, "", ""]
    elif action == "compact":
        kept = [row for row in rows[:count] if str(row[3] or "").strip()]
        count = max(1, len(kept))
        rows = kept + [["Zero-shot", NONE, NONE, "", ""] for _ in range(MAX_DIALOGUE_ROWS - len(kept))]
    visible = [gr.update(visible=i < count) for i in range(MAX_DIALOGUE_ROWS)]
    rows = rows[:MAX_DIALOGUE_ROWS]
    return [count, *visible, *_dialogue_row_updates(rows)]

def refresh_voice_ui(current=None):
    choices = voice_choices()
    selected = current if current in choices else NONE
    return gr.update(choices=choices, value=selected), gr.update(choices=choices, value=selected)


def refresh_dialogue_voices(*current_values):
    choices = voice_choices()
    return [gr.update(choices=choices, value=value if value in choices else NONE) for value in current_values]


def refresh_adapter_ui(current=None):
    choices = adapter_values()
    values = [item[1] if isinstance(item, tuple) else item for item in choices]
    selected = current if current in values else NONE
    return gr.update(choices=choices, value=selected)


def refresh_dialogue_adapters(*current_values):
    choices = adapter_values()
    values = [item[1] if isinstance(item, tuple) else item for item in choices]
    return [gr.update(choices=choices, value=value if value in values else NONE) for value in current_values]


def refresh_resume_ui(project, model_variant, current=NONE):
    choices = [NONE, *resume_checkpoint_choices(project, model_variant)]
    values = [item[1] if isinstance(item, tuple) else item for item in choices]
    selected = current if current in values else NONE
    return gr.update(choices=choices, value=selected)


def load_resume_config_ui(checkpoint):
    if not checkpoint or checkpoint == NONE:
        return *(gr.update() for _ in range(11)), "New run selected; training will start from the chosen Base/RL checkpoint."
    run_dir = Path(checkpoint).parent
    data = read_json(run_dir / "run.json", {})
    if not data:
        return *(gr.update() for _ in range(11)), f"Resume checkpoint selected: {Path(checkpoint).name}."
    variant = data.get("model_variant", "Base")
    return (
        gr.update(value=variant),
        gr.update(value=int(data.get("rank", 16))),
        gr.update(value=int(data.get("alpha", 64))),
        gr.update(value=float(data.get("dropout", 0.05))),
        gr.update(value=float(data.get("learning_rate", 5e-5))),
        gr.update(value=data.get("training_mode", "Steps")),
        gr.update(value=int(data.get("steps", 1500))),
        gr.update(value=int(data.get("save_every_steps", 250))),
        gr.update(value=int(data.get("epochs", 20))),
        gr.update(value=int(data.get("save_every_epochs", 5))),
        gr.update(value=int(data.get("seed", 1234))),
        f"Loaded resume configuration from {Path(checkpoint).name}. Resume loads adapter weights as a warm start.",
    )


def normalize_autotune_profile(profile):
    value = LEGACY_AUTOTUNE_PROFILES.get(str(profile), str(profile))
    return value if value in AUTOTUNE_PROFILES else AUTOTUNE_PROFILES[1]


def training_dataset_statistics(project):
    metadata = read_json(_project_file(project), {})
    ui = metadata.get("dataset_ui", {}) if isinstance(metadata.get("dataset_ui", {}), dict) else {}
    rows = ui.get("analyzed_rows", []) if isinstance(ui.get("analyzed_rows", []), list) else []
    if not rows:
        manifest = Path(str(metadata.get("selected_manifest", "") or ""))
        if not manifest.is_file():
            manifest = DATASETS / str(project or "") / "manifest.jsonl"
        if manifest.is_file():
            rows = []
            for line in manifest.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        rows.append(record)
                except json.JSONDecodeError:
                    continue
    total_seconds = 0.0
    valid = 0
    speakers = set()
    train_items = cv_items = 0
    from cosyvoice_easy.audio import wav_info
    for row in rows:
        if row.get("error"):
            continue
        duration = float(row.get("duration", 0.0) or 0.0)
        if duration <= 0 and row.get("audio"):
            try:
                duration = float(wav_info(row["audio"])["duration"])
            except Exception:
                duration = 0.0
        if duration <= 0:
            continue
        valid += 1
        total_seconds += duration
        speakers.add(str(row.get("speaker", "speaker") or "speaker"))
        if row.get("split") == "cv":
            cv_items += 1
        elif row.get("split") == "train":
            train_items += 1
    if valid and not (train_items or cv_items):
        cv_items = max(1, round(valid * int(ui.get("cv_percent", 10) or 10) / 100))
        train_items = valid - cv_items
    train_ratio = train_items / valid if valid else 0.0
    return {
        "items": valid, "speakers": len(speakers), "minutes": total_seconds / 60.0,
        "mean_seconds": total_seconds / valid if valid else 0.0,
        "train_items": train_items, "cv_items": cv_items, "train_minutes": total_seconds / 60.0 * train_ratio,
    }


def lora_vram_report(profile, rank, alpha, dropout, learning_rate, epochs, dataset=None, requested_profile=None):
    """Explain the exact adapter-state math and evidence-calibrated VRAM boundary."""
    layers, hidden, kv_width = 24, 896, 128
    params_per_rank_per_layer = (hidden + hidden) + (hidden + kv_width) * 2 + (hidden + hidden)
    trainable = layers * params_per_rank_per_layer * int(rank)
    base_params = 506_000_000
    checkpoint_mib = trainable * 4 / 2**20
    training_state_mib = trainable * 16 / 2**20
    frozen_gib = base_params * 4 / 2**30
    observed_peak_gb = 6.96
    r16_state_gb = (layers * params_per_rank_per_layer * 16 * 16) / 1e9
    projected_peak_gb = observed_peak_gb + (trainable * 16 / 1e9) - r16_state_gb
    gpu_line = "CUDA device unavailable while calculating; physical VRAM was not read."
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            gpu_line = f"{props.name}: {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB currently free."
    except Exception as exc:
        gpu_line = f"CUDA VRAM query unavailable: {exc}"
    scale = float(alpha) / max(int(rank), 1)
    dataset = dataset or {}
    minutes = float(dataset.get("minutes", 0.0) or 0.0)
    if dataset.get("items"):
        exposure = float(dataset.get("train_minutes", 0.0)) * int(epochs)
        dataset_block = f"""**Dataset used by AutoTune:** {dataset['items']:,} valid clips · **{minutes:.1f} minutes** total · {dataset['speakers']} speaker(s) · mean clip {dataset['mean_seconds']:.1f}s · Train/CV {dataset['train_items']}/{dataset['cv_items']}.

**Epoch estimate:** `{int(epochs)}` maximum epochs × {dataset['train_minutes']:.1f} train minutes = **{exposure:.0f} train-minutes of maximum exposure**. Training runs to the selected target. Dynamic batching prevents a trustworthy optimizer-step estimate from minutes alone.
"""
    else:
        dataset_block = "**Dataset used by AutoTune:** no valid duration metadata was found. The profile uses the validated r16 fallback and 20 epochs. Analyze/prepare the project, then press AutoTune again."
    report = f"""### AutoTune calculation — {profile}

{dataset_block}

**Selected values:** rank `{int(rank)}`, alpha `{int(alpha)}` (scale `α/r = {scale:.2f}`), dropout `{float(dropout):.2f}`, learning rate `{float(learning_rate):.2g}`, maximum epochs `{int(epochs)}` with CV patience 3.

**Detected GPU:** {gpu_line}

| Memory component | Calculation | Result |
|---|---:|---:|
| Trainable LoRA parameters | `24 layers × 5,632 × r` | **{trainable:,}** ({100 * trainable / base_params:.3f}% of ~506M) |
| Adapter checkpoint | parameters × 4-byte FP32 | **{checkpoint_mib:.1f} MiB** |
| LoRA weights + gradients + Adam states | parameters × (4 + 4 + 8 bytes) | **{training_state_mib:.1f} MiB** |
| Frozen ~506M-parameter LLM weights | parameters × 4-byte FP32 | **{frozen_gib:.2f} GiB** |
| Evidence-calibrated peak | 6.96 GB at r16 + adapter-state delta | **~{projected_peak_gb:.2f} GB** |

**Why this is not a VRAM tier:** changing r8 → r32 changes adapter training state by only ~50 MiB. Peak VRAM is dominated by frozen weights, CUDA/DDP workspace and activations from the dynamic microbatch (`max_frames_in_batch=2000`, token limit 200). Gradient accumulation is fixed at 2 and improves effective batch size, but does not reduce the memory of each microbatch.

**Practical boundary:** no official CosyVoice3 LoRA minimum is published. The closest matching documented run measured **6.96 GB peak at r16** on an RTX 3090 Ti. This GUI uses native single-GPU DDP rather than that run's DeepSpeed Stage 2, so the number is a calibration point, not a guarantee. A **12 GB GPU is a reasonable conservative floor** for the current pipeline; 24/32 GB does not by itself justify a higher rank. Choose rank for dataset diversity/capacity and validate with CV loss.

**Sources:** [official CosyVoice3 model/config](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) · [matching r16 LoRA run and measured peak](https://github.com/instavar/cosyvoice3-lora-finetuning) · [PEFT LoRA rank and target-module semantics](https://huggingface.co/docs/peft/en/package_reference/lora)
"""
    # Keep the GUI report compact: the useful recommendation/calculation ends
    # immediately before the verbose detected-GPU memory table.
    return report.split("**Detected GPU:**", 1)[0].rstrip()


def _nice_save_interval(target: int, unit: str = "steps") -> int:
    """Choose a round cadence that divides the selected training target."""
    candidates = (1000, 500, 250, 100, 50) if unit == "steps" else (25, 20, 10, 5, 2, 1)
    ceiling = max(1, target // 4)
    return next((value for value in candidates if value <= ceiling and target % value == 0), 1)


def coherent_checkpoint_cadence_ui(steps, save_steps, epochs, save_epochs):
    steps, epochs = max(1, int(steps)), max(1, int(epochs))
    step_divisors = [value for value in range(50, steps + 1, 50) if steps % value == 0]
    epoch_divisors = [value for value in range(1, epochs + 1) if epochs % value == 0]
    fixed_steps = min(step_divisors or [steps], key=lambda value: abs(value - int(save_steps or 250)))
    fixed_epochs = min(epoch_divisors, key=lambda value: abs(value - int(save_epochs or 5)))
    return gr.update(value=fixed_steps), gr.update(value=fixed_epochs)


def autotune_training_ui(project, profile, rank, alpha, dropout, learning_rate, epochs):
    """Apply an evidence-based single-GPU LoRA capacity profile."""
    presets = {
        AUTOTUNE_PROFILES[0]: (8, 32, 0.10, 1e-5),
        AUTOTUNE_PROFILES[1]: (16, 64, 0.08, 1.5e-5),
        AUTOTUNE_PROFILES[2]: (32, 128, 0.05, 1e-5),
    }
    requested = AUTO_PROFILE if profile == AUTO_PROFILE else normalize_autotune_profile(profile)
    stats = training_dataset_statistics(project)
    selected = requested
    if requested == AUTO_PROFILE:
        if stats["items"] and stats["minutes"] < 30 and stats["speakers"] <= 1:
            selected = AUTOTUNE_PROFILES[0]
        elif stats["minutes"] >= 180 and stats["speakers"] >= 4:
            selected = AUTOTUNE_PROFILES[2]
        else:
            selected = AUTOTUNE_PROFILES[1]
    epochs_value = 30 if stats["items"] and stats["minutes"] < 30 else (25 if stats["items"] and stats["minutes"] < 120 else 20)
    steps_value = 1500 if stats["items"] and stats["minutes"] < 30 else (2500 if stats["items"] and stats["minutes"] < 120 else 4000)
    if selected == AUTOTUNE_PROFILES[2]:
        steps_value = max(steps_value, 5000)
    save_steps_value = _nice_save_interval(steps_value, "steps")
    save_epochs_value = _nice_save_interval(epochs_value, "epochs")
    values = (*presets[selected], epochs_value)
    log(f"Applied LoRA AutoTune profile: {selected}.")
    report = lora_vram_report(selected, *values, dataset=stats, requested_profile=requested)
    report = report.replace("**Selected values:**", f"**Training objective:** Steps = **{steps_value:,} optimizer updates**, checkpoint every **{save_steps_value:,} steps** ({steps_value // save_steps_value} exact intervals) plus a mandatory final save.\n\n**Selected values:**")
    return gr.update(value="Steps"), gr.update(value=steps_value), gr.update(value=save_steps_value), *[gr.update(value=value) for value in values], gr.update(value=save_epochs_value), gr.update(value=2), report


def autotune_dataset_ui(project, rank, alpha, dropout, learning_rate, epochs):
    return autotune_training_ui(project, AUTO_PROFILE, rank, alpha, dropout, learning_rate, epochs)


def save_training_length_ui(project, mode, steps, save_steps, epochs, save_epochs):
    if not project:
        return
    path = _project_file(project)
    metadata = read_json(path, {"name": project})
    metadata.setdefault("training_ui", {})
    fixed_steps, fixed_epochs = coherent_checkpoint_cadence_ui(steps, save_steps, epochs, save_epochs)
    metadata["training_ui"].update({"training_mode": mode if mode in {"Steps", "Epochs"} else "Steps",
                                    "steps": int(steps or 1500), "save_every_steps": int(fixed_steps["value"]),
                                    "epochs": int(epochs or 20), "save_every_epochs": int(fixed_epochs["value"])})
    atomic_json(path, metadata)


def clear_outputs_ui():
    removed = 0
    for path in OUTPUTS.glob("*.wav"):
        path.unlink(missing_ok=True)
        removed += 1
    history = OUTPUTS / "history.jsonl"
    if history.is_file():
        history.unlink()
        removed += 1
    log(f"Removed {removed} generated output file(s); voice and model data were preserved.")
    return f"Removed {removed} generated output file(s)."


def clear_voices_ui():
    removed = 0
    for path in VOICES.iterdir():
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
            removed += 1
    log(f"Removed {removed} saved reference sample(s).", "WARN")
    return f"Removed {removed} saved reference sample(s)."


def _project_file(project: str) -> Path:
    return PROJECTS / str(project or "").strip() / "project.json"


def _persist_project_audio(project: str, audio_path: str | None, stem: str) -> str:
    """Copy transient Gradio audio into the project so project reloads remain valid."""
    if not project or not audio_path:
        return ""
    source = Path(str(audio_path))
    if not source.is_file():
        # Preserve an already-stable project path if it still exists; otherwise clear it.
        return str(source) if source.is_file() else ""
    assets = PROJECTS / str(project) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() if source.suffix else ".wav"
    target = assets / f"{stem}{suffix}"
    try:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return str(target)


def _dataset_ui_payload(folder, language, transcribe_missing, asr_model, asr_language, asr_batch,
                        instruction_source, instruction_custom, cv_percent, manifest, rows_json):
    try:
        rows = json.loads(rows_json or "[]")
    except json.JSONDecodeError:
        rows = []
    return {
        "source_folder": str(folder or ""),
        "language": str(language or "Auto"),
        "transcribe_missing": bool(transcribe_missing),
        "whisper_model": str(asr_model or "large-v3"),
        "whisper_language": str(asr_language or "Auto-detect"),
        "whisper_batch_size": int(asr_batch or 4),
        "instruction_mode": str(instruction_source or "Standard (Recommended)"),
        "custom_instruction": str(instruction_custom or ""),
        "cv_percent": int(cv_percent or 10),
        "manifest": str(manifest or ""),
        "analyzed_rows": rows if isinstance(rows, list) else [],
    }


def _training_ui_payload(model_variant, vram_preset, resume_checkpoint, epochs, seed, rank, alpha, dropout, lr,
                         eval_audio, eval_transcript, eval_text, eval_language,
                         eval_asr_model, eval_asr_language, eval_asr_batch):
    return {
        "base_variant": "RL" if model_variant == "RL" else "Base",
        "vram_preset": normalize_autotune_profile(vram_preset),
        "resume_checkpoint": "" if not resume_checkpoint or str(resume_checkpoint).strip().lower() in {"none", ""} else str(resume_checkpoint),
        "epochs": int(epochs or 20),
        "seed": int(seed or 1234),
        "rank": int(rank or 16),
        "alpha": int(alpha or 64),
        "dropout": float(dropout or 0.05),
        "learning_rate": float(lr or 5e-5),
        "eval": {
            "audio": str(eval_audio or ""),
            "transcript": str(eval_transcript or "").strip(),
            "text": str(eval_text or "").strip(),
            "language": str(eval_language or "Auto"),
            "whisper_model": str(eval_asr_model or "large-v3"),
            "whisper_language": str(eval_asr_language or "Auto-detect"),
            "whisper_batch_size": int(eval_asr_batch or 4),
        },
    }


def load_project_ui(project):
    """Restore every Dataset Preparation control saved with an existing project."""
    choices = list_dataset_projects()
    project_name = str(project or "").strip()
    project_file = _project_file(project_name)
    if project_name and not project_file.is_file():
        selected = choices[0] if choices else None
        return (
            gr.update(choices=choices, value=selected), "", "Auto", False, "large-v3", "Auto-detect", 4,
            "Standard (Recommended)", "", gr.update(visible=False), 10, "", "[]",
            f"New project name '{project_name}' entered. Press Create Project to create it."
        )
    metadata = read_json(project_file, {}) if project_name else {}
    ui = metadata.get("dataset_ui", {}) if isinstance(metadata.get("dataset_ui", {}), dict) else {}
    manifest = str(ui.get("manifest") or metadata.get("selected_manifest", ""))
    rows = ui.get("analyzed_rows", []) if isinstance(ui.get("analyzed_rows", []), list) else []
    # Backward compatibility: reconstruct rows from a prepared manifest when old projects do not have analyzed_rows.
    if not rows and manifest:
        manifest_path = Path(manifest)
        if manifest_path.is_file():
            for line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                    rows.append({
                        "audio": record.get("audio", ""),
                        "transcript": record.get("text", record.get("transcript", "")),
                        "speaker": record.get("speaker", ""),
                        "duration": record.get("duration", 0.0),
                        "error": record.get("error", ""),
                    })
                except json.JSONDecodeError:
                    continue
    language = ui.get("language", metadata.get("language", "Auto"))
    if language not in LANGUAGES:
        language = "Auto"
    instruction_mode = ui.get("instruction_mode", "Standard (Recommended)")
    if instruction_mode not in {"Standard (Recommended)", "Custom"}:
        instruction_mode = "Standard (Recommended)"
    project_update = gr.update(choices=choices, value=project_name if project_name in choices else (choices[0] if choices else None))
    return (
        project_update,
        ui.get("source_folder") or metadata.get("source_folder") or metadata.get("dataset_dir", ""),
        language,
        bool(ui.get("transcribe_missing", False)),
        ui.get("whisper_model", "large-v3") if ui.get("whisper_model", "large-v3") in ASR_MODELS else "large-v3",
        ui.get("whisper_language", "Auto-detect") if ui.get("whisper_language", "Auto-detect") in ASR_LANGUAGES else "Auto-detect",
        int(ui.get("whisper_batch_size", 4)),
        instruction_mode,
        ui.get("custom_instruction", ""),
        gr.update(visible=instruction_mode == "Custom"),
        int(ui.get("cv_percent", 10)),
        manifest,
        json.dumps(rows, ensure_ascii=False),
        f"Loaded all Dataset Preparation settings for '{project_name}'." if project_name else "No project selected.",
    )


def dialogue_generate(count, model_dir, model_variant, fp16, trt, flash_attention, torch_compile,
                      seed, random_seed, language, turn_silence, default_instruction,
                      chunk_mode, speed, text_frontend, top_k, top_p, temperature, ras_window, ras_threshold,
                      min_token_ratio, max_token_ratio, flow_steps, flow_temperature, *values,
                      progress=gr.Progress(track_tqdm=False)):
    compile_mode = "default"
    count = int(count)
    actual_seed = resolve_seed(seed, random_seed)
    if float(max_token_ratio) <= float(min_token_ratio):
        raise ValueError("Max Token/Text Ratio must be greater than Min Token/Text Ratio.")
    log(f"Starting dialogue generation with {count} visible row(s), seed={actual_seed}.")
    generated = []
    rendered = 0
    saved_lines = []
    for index in range(count):
        offset = index * 5
        row_mode, speaker, row_adapter, text, row_instruction = values[offset:offset + 5]
        if not str(text or "").strip():
            continue
        progress(0.05 + 0.82 * (index / max(count, 1)), desc=f"Generating dialogue turn {index + 1}/{count}")
        selected_speaker = "" if not speaker or speaker == NONE else speaker
        selected_adapter = "" if not row_adapter or row_adapter == NONE else row_adapter
        mode = row_mode if row_mode in {"Zero-shot", "Cross-lingual", "Instruct"} else "Zero-shot"
        instruction = str(row_instruction or "").strip() or str(default_instruction or "").strip()
        selected_instruction = language_instruction(instruction, language) if mode == "Instruct" else ""
        request = InferenceRequest(
            text=text, voice=selected_speaker, instruction=selected_instruction, language=language, mode=mode,
            seed=actual_seed + index, speed=float(speed), chunk_mode=chunk_mode if chunk_mode in CHUNK_CHOICES else "None",
            gap_seconds=0.0, text_frontend=bool(text_frontend), top_k=int(top_k), top_p=float(top_p),
            temperature=float(temperature), ras_window=int(ras_window), ras_repetition_threshold=float(ras_threshold),
            min_token_text_ratio=float(min_token_ratio), max_token_text_ratio=float(max_token_ratio),
            flow_steps=int(flow_steps), flow_temperature=float(flow_temperature),
        )
        output, _ = ENGINE.generate(
            request, model_dir, fp16, trt, selected_adapter, model_variant, bool(flash_attention),
            bool(torch_compile), compile_mode,
        )
        from cosyvoice_easy.audio import load_audio
        sr, audio = load_audio(output)
        generated.append((sr, audio))
        rendered += 1
        saved_lines.append(asdict(DialogueLine(selected_speaker or NONE, str(text), str(row_instruction or ""), mode, selected_adapter)))
    if rendered == 0:
        raise ValueError("Add text to at least one dialogue row.")
    progress(0.92, desc="Merging dialogue audio")
    sr, audio = concat_audio(generated, max(0.0, float(turn_silence)))
    output = save_wav(ROOT / "outputs" / "dialogue-latest.wav", sr, audio)
    atomic_json(CONFIG / "last_dialogue.json", {
        "lines": saved_lines,
        "defaults": {"language": language, "turn_silence": float(turn_silence), "instruction": default_instruction or ""},
    })
    append_jsonl(OUTPUTS / "history.jsonl", {"mode": "Dialogue", "output": output,
                                             "created_at": datetime.now(timezone.utc).isoformat()})
    progress(1.0, desc="Dialogue complete")
    play_completion_chime()
    return output, f"Generated {rendered} dialogue turn(s). Seed: {actual_seed}."

def browse_folder_ui(current=""):
    """Open a native Windows folder picker for dataset source audio."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(initialdir=current or str(ROOT), title="Select audio dataset folder")
        root.destroy()
        return selected or current or ""
    except Exception as exc:
        log(f"Dataset folder picker unavailable: {exc}", "WARN")
        return current or ""


def scan_ui(folder, transcribe_enabled, model, asr_language, _batch_size=8, progress=gr.Progress(track_tqdm=False)):
    progress(0.08, desc="Analyzing source audio")
    log(f"Analyzing dataset source folder: {folder}")
    rows, message = scan_source(folder)
    if transcribe_enabled:
        progress(0.35, desc="Transcribing missing transcripts")
        selected_language = "Auto" if not asr_language or asr_language == "Auto-detect" else asr_language
        log(f"Transcribing missing dataset rows with Faster-Whisper {model} (CUDA).")
        pending = [row for row in rows if not row.get("transcript") and not row.get("error")]
        total_pending = len(pending)
        def report_transcription(index, total, row, detected):
            fraction = 0.35 + (0.60 * index / max(1, total))
            filename = Path(row["audio"]).name
            text = str(row.get("transcript", "")).strip()
            progress(fraction, desc=f"Whisper {index}/{total}: {filename}")
            log(f"[Whisper {index}/{total_pending}] {filename} [{detected}]: {text}")
        rows, asr_message = transcribe_missing(
            rows, model, selected_language, batch_size=int(_batch_size or 1), on_item=report_transcription
        )
        message = f"{message} {asr_message}"
    valid = sum(bool(row.get("audio") and row.get("transcript") and not row.get("error")) for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    progress(1.0, desc="Source analysis complete")
    if transcribe_enabled:
        play_completion_chime()
    detail = f"{message} Ready pairs: {valid}. Errors: {errors}."
    return json.dumps(rows, ensure_ascii=False), detail


def prepare_ui(project, rows_json, language, cv_percent, instruction_source, instruction_custom,
               progress=gr.Progress(track_tqdm=False)):
    progress(0.10, desc="Validating analyzed source")
    instruction = resolve_dataset_instruction(instruction_source, instruction_custom)
    log(f"Preparing dataset project '{project}' with {instruction_source.lower()} conditioning.")
    valid_rows = [row for row in json.loads(rows_json or "[]") if row.get("audio") and row.get("transcript") and not row.get("error")]
    def report_preparation(index, total, row, utterance):
        progress(0.10 + 0.82 * index / max(1, total), desc=f"Preparing {index}/{total}: {Path(row['audio']).name}")
        log(f"[Dataset {index}/{total}] prepared {Path(row['audio']).name} -> {utterance}")
    result = prepare_dataset(project, valid_rows, language, cv_percent, instruction, on_item=report_preparation)
    progress(1.0, desc="Train / CV preparation complete")
    play_completion_chime()
    return result


def extract_features_ui(project, model_dir, progress=gr.Progress(track_tqdm=False)):
    progress(0.05, desc="Checking CosyVoice3 model files")
    ensure_model_available(model_dir)
    progress(0.12, desc="Loading CosyVoice3 feature extractors")
    def report_stage(completed, total, split, stage, state):
        progress(0.12 + 0.84 * completed / max(1, total), desc=f"{split.upper()} · {stage} · {state}")
        log(f"[Features {completed}/{total}] {split}: {stage} {state}")
    result = extract_features(project, model_dir, on_stage=report_stage)
    progress(1.0, desc="Feature extraction complete")
    play_completion_chime()
    return result


def _write_dataset_project_state(project, folder, language, transcribe_missing, asr_model, asr_language, asr_batch,
                                 instruction_source, instruction_custom, cv_percent, manifest, rows_json):
    path = _project_file(project)
    metadata = read_json(path, {"name": project})
    metadata.update({"source_folder": str(folder or ""), "language": str(language or "Auto"),
                     "updated_at": datetime.now(timezone.utc).isoformat()})
    metadata["dataset_ui"] = _dataset_ui_payload(
        folder, language, transcribe_missing, asr_model, asr_language, asr_batch,
        instruction_source, instruction_custom, cv_percent, manifest, rows_json,
    )
    atomic_json(path, metadata)


def create_project_ui(name, folder, language, transcribe_missing, asr_model, asr_language, asr_batch,
                      instruction_source, instruction_custom, cv_percent, manifest, rows_json,
                      model_variant, vram_preset, resume_checkpoint, epochs, seed, rank, alpha, dropout, lr,
                      eval_audio, eval_transcript, eval_text, eval_language, eval_asr_model, eval_asr_language, eval_asr_batch):
    project, message = create_project(name, folder, language)
    _write_dataset_project_state(project, folder, language, transcribe_missing, asr_model, asr_language, asr_batch,
                                 instruction_source, instruction_custom, cv_percent, manifest, rows_json)
    path = _project_file(project)
    metadata = read_json(path, {"name": project})
    stable_eval_audio = _persist_project_audio(project, eval_audio, "eval_reference") if eval_audio else ""
    training_ui = _training_ui_payload(
        model_variant, vram_preset, resume_checkpoint, epochs, seed, rank, alpha, dropout, lr,
        stable_eval_audio, eval_transcript, eval_text, eval_language, eval_asr_model, eval_asr_language, eval_asr_batch,
    )
    metadata["training_ui"] = training_ui
    metadata["training_eval"] = dict(training_ui["eval"])
    metadata["training_base_variant"] = training_ui["base_variant"]
    atomic_json(path, metadata)
    choices = list_dataset_projects()
    return gr.update(choices=choices, value=project), gr.update(choices=choices, value=project), message


def save_project_ui(project, folder, language, transcribe_missing, asr_model, asr_language, asr_batch,
                    instruction_source, instruction_custom, cv_percent, manifest, rows_json):
    message = save_project(project, folder, language)
    _write_dataset_project_state(project, folder, language, transcribe_missing, asr_model, asr_language, asr_batch,
                                 instruction_source, instruction_custom, cv_percent, manifest, rows_json)
    choices = list_dataset_projects()
    return gr.update(choices=choices, value=project), gr.update(choices=choices, value=project), message


def delete_project_ui(project):
    message = delete_project(project)
    dataset_choices = list_dataset_projects()
    training_choices = list_training_projects()
    return gr.update(choices=dataset_choices, value=(dataset_choices[0] if dataset_choices else None)), gr.update(choices=training_choices, value=(training_choices[0] if training_choices else None)), message


def delete_training_project_ui(project):
    message = delete_training_project(project)
    choices = list_training_projects()
    return gr.update(choices=choices, value=(choices[0] if choices else None)), message


def clean_tensorboard_runs_ui(project):
    if not project:
        return "Select a training project first."
    root = (TRAINING_OUTPUTS / safe_name(str(project))).resolve()
    base = TRAINING_OUTPUTS.resolve()
    if root.parent != base or not root.is_dir():
        return f"No training directory found for '{project}'."
    removed = 0
    for folder in [p for p in root.rglob("tensorboard") if p.is_dir()]:
        shutil.rmtree(folder)
        removed += 1
    return f"Cleaned {removed} TensorBoard run folder(s) for '{project}'. Checkpoints and adapters were preserved."


def _next_incremental_project_name(project: str) -> str:
    base = re.sub(r"-\d+$", "", str(project or "").strip()) or "voice-project"
    occupied = set()
    for name in list_projects():
        metadata = read_json(PROJECTS / name / "project.json", {})
        if metadata.get("dataset_deleted") and metadata.get("training_deleted"):
            continue
        match = re.fullmatch(re.escape(base) + r"-(\d+)", name)
        if match:
            occupied.add(int(match.group(1)))
    for index in range(2, 1000):
        candidate = f"{base}-{index}"
        if index not in occupied:
            return candidate
    raise RuntimeError("Could not allocate an incremental project name.")


def clone_project_ui(project):
    if not project:
        raise ValueError("Select an existing project before cloning it.")
    cloned, message = clone_project(project, _next_incremental_project_name(project))
    return gr.update(choices=list_dataset_projects(), value=cloned), gr.update(choices=list_training_projects(), value=cloned), message


def save_training_project_ui(project, folder, language, transcribe_missing, dataset_asr_model, dataset_asr_language,
                             dataset_asr_batch, instruction_source, instruction_custom, cv_percent, manifest, rows_json,
                             model_variant, vram_preset, resume_checkpoint, epochs, seed, rank, alpha, dropout, lr,
                             eval_audio, eval_transcript, eval_text, eval_language,
                             eval_asr_model, eval_asr_language, eval_asr_batch):
    if not project:
        return gr.update(), gr.update(), "Select a training project before saving it.", "Select a training project before saving it."
    save_project(project, folder, language)
    _write_dataset_project_state(project, folder, language, transcribe_missing, dataset_asr_model, dataset_asr_language,
                                 dataset_asr_batch, instruction_source, instruction_custom, cv_percent, manifest, rows_json)
    path = _project_file(project)
    metadata = read_json(path, {"name": project})
    stable_eval_audio = _persist_project_audio(project, eval_audio, "eval_reference") if eval_audio else ""
    training_ui = _training_ui_payload(
        model_variant, vram_preset, resume_checkpoint, epochs, seed, rank, alpha, dropout, lr,
        stable_eval_audio, eval_transcript, eval_text, eval_language, eval_asr_model, eval_asr_language, eval_asr_batch,
    )
    metadata["training_ui"] = training_ui
    # Backward-compatible keys consumed by older patches/tools.
    metadata["training_eval"] = dict(training_ui["eval"])
    metadata["training_base_variant"] = training_ui["base_variant"]
    atomic_json(path, metadata)
    dataset_update = gr.update(choices=list_dataset_projects(), value=project)
    training_update = gr.update(choices=list_training_projects(), value=project)
    return dataset_update, training_update, f"Saved every Dataset + Training field for '{project}'.", f"Saved training/evaluation settings for '{project}'."


def save_training_eval_ui(project, eval_audio, eval_transcript, eval_text, eval_language, model_variant):
    """Backward-compatible narrow save used immediately before launch; full project Save uses save_training_project_ui."""
    if not project:
        return "Select a training project before saving evaluation settings."
    path = _project_file(project)
    metadata = read_json(path, {"name": project})
    stable_eval_audio = _persist_project_audio(project, eval_audio, "eval_reference") if eval_audio else ""
    eval_data = {
        "audio": stable_eval_audio,
        "transcript": str(eval_transcript or "").strip(),
        "text": str(eval_text or "").strip(),
        "language": str(eval_language or "Auto"),
    }
    metadata["training_eval"] = eval_data
    metadata["training_base_variant"] = "RL" if model_variant == "RL" else "Base"
    if isinstance(metadata.get("training_ui"), dict):
        metadata["training_ui"]["eval"] = {**metadata["training_ui"].get("eval", {}), **eval_data}
        metadata["training_ui"]["base_variant"] = metadata["training_base_variant"]
    atomic_json(path, metadata)
    return f"Saved training/evaluation settings for '{project}'."


def load_training_project_ui(project):
    prepare_tensorboard_project(project)
    metadata = read_json(_project_file(project), {})
    ui = metadata.get("training_ui", {}) if isinstance(metadata.get("training_ui", {}), dict) else {}
    legacy_eval = metadata.get("training_eval", {}) if isinstance(metadata.get("training_eval", {}), dict) else {}
    eval_data = ui.get("eval", legacy_eval) if isinstance(ui.get("eval", legacy_eval), dict) else legacy_eval
    variant = ui.get("base_variant", metadata.get("training_base_variant", "Base"))
    if variant not in {"Base", "RL"}:
        variant = "Base"
    preset = normalize_autotune_profile(ui.get("vram_preset", AUTOTUNE_PROFILES[1]))
    resume_value = str(ui.get("resume_checkpoint", "") or "")
    resume_choices = [NONE, *resume_checkpoint_choices(project, variant)]
    valid_resume_paths = {str(item[1]) for item in resume_choices if isinstance(item, tuple) and len(item) == 2}
    resume = resume_value if resume_value and resume_value in valid_resume_paths and Path(resume_value).is_dir() else NONE
    resume_update = gr.update(choices=resume_choices, value=resume)
    eval_audio = str(eval_data.get("audio", "") or "")
    if eval_audio and not Path(eval_audio).is_file():
        eval_audio = ""
    return (
        eval_audio or None,
        eval_data.get("transcript", ""),
        eval_data.get("text", "This is a CosyVoice3 training preview."),
        eval_data.get("language", "Auto") if eval_data.get("language", "Auto") in LANGUAGES else "Auto",
        variant,
        preset,
        resume_update,
        ui.get("training_mode", "Steps") if ui.get("training_mode", "Steps") in {"Steps", "Epochs"} else "Steps",
        int(ui.get("steps", 1500)),
        int(ui.get("save_every_steps", 250)),
        int(ui.get("epochs", 20)),
        int(ui.get("save_every_epochs", 5)),
        int(ui.get("seed", 1234)),
        int(ui.get("rank", 16)),
        int(ui.get("alpha", 64)),
        float(ui.get("dropout", 0.05)),
        float(ui.get("learning_rate", 5e-5)),
        eval_data.get("whisper_model", "large-v3") if eval_data.get("whisper_model", "large-v3") in ASR_MODELS else "large-v3",
        eval_data.get("whisper_language", "Auto-detect") if eval_data.get("whisper_language", "Auto-detect") in ASR_LANGUAGES else "Auto-detect",
        int(eval_data.get("whisper_batch_size", 4)),
    )


def launch_training_ui(project, model_dir, model_variant, resume_checkpoint, rank, alpha, dropout, lr, grad_accumulation, mode, steps, save_steps, epochs, save_epochs, eval_reference, eval_reference_text, eval_text, seed):
    if not project:
        raise ValueError("Select a prepared training project first.")
    variant = "RL" if model_variant == "RL" else "Base"
    ensure_model_available(model_dir, variant)
    dataset = DATASETS / project
    resume_value = "" if not resume_checkpoint or str(resume_checkpoint).strip().lower() in {"none", ""} else str(resume_checkpoint)
    fixed_steps, fixed_epochs = coherent_checkpoint_cadence_ui(steps, save_steps, epochs, save_epochs)
    if resume_value:
        output = Path(resume_value).parent
    else:
        output_root = TRAINING_OUTPUTS / project
        output_root.mkdir(parents=True, exist_ok=True)
        # One project owns one run directory. Presets and LoRA rank must not
        # create confusing nested names; cloned projects provide PL-2, etc.
        output = output_root
        if output.exists():
            for child in output.iterdir():
                if child.name == "project.json":
                    continue
                if child.is_dir() or child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    # Easy GUI policy: CV is diagnostic only; training runs to the selected target.
    config = TrainingConfig(
        project=project, model_dir=model_dir, dataset_dir=str(dataset), output_dir=str(output),
        model_variant=variant, rank=int(rank), alpha=int(alpha), dropout=float(dropout), learning_rate=float(lr),
        training_mode=mode if mode in {"Steps", "Epochs"} else "Steps", steps=int(steps),
        save_every_steps=int(fixed_steps["value"]), epochs=int(epochs), save_every_epochs=int(fixed_epochs["value"]),
        patience=3, grad_accumulation=max(1, int(grad_accumulation)), resume=resume_value, eval_reference=str(eval_reference or ""), eval_reference_text=str(eval_reference_text or ""), eval_text=str(eval_text or ""), seed=int(seed), deterministic=False,
        guarded_checkpoints=False, resume_keep_last=3,
    )
    project_file = PROJECTS / project / "project.json"
    metadata = read_json(project_file, {"name": project})
    metadata["training"] = asdict(config)
    metadata["training_base_variant"] = variant
    atomic_json(project_file, metadata)
    return launch_training(config)



_TRAIN_WAS_RUNNING = False
_TRAIN_CHIMED_EXIT = None


def unload_all_ui():
    model_message = ENGINE.unload()
    asr.unload()
    log("Unloaded CosyVoice3 and Faster-Whisper; GPU cache released.")
    return f"{model_message} Faster-Whisper unloaded."


def _format_duration(seconds):
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def training_progress_html(snapshot=None):
    snapshot = snapshot or training_progress_snapshot()
    pct = min(100.0, max(0.0, float(snapshot.get("pct", 0.0) or 0.0)))
    total = int(snapshot.get("total_epochs", 0) or 0)
    complete = int(snapshot.get("completed_epochs", 0) or 0)
    current = int(snapshot.get("current_epoch", 0) or 0)
    loss = snapshot.get("loss")
    loss_text = "--" if loss is None else f"{float(loss):.5f}"
    state = "Running" if snapshot.get("running") else ("Complete" if snapshot.get("exit_code") == 0 else "Idle")
    unit = snapshot.get("unit", "Epochs")
    return f'''<div class="training-progress-card">
<div class="training-progress-head"><b>{html.escape(state)}</b><span>{pct:.1f}%</span></div>
<div class="training-progress-track"><div class="training-progress-fill" style="width:{pct:.2f}%"></div></div>
<div class="training-progress-meta">{unit[:-1] if unit.endswith('s') else unit} {(complete if unit == 'Steps' else (current if current else complete))}/{total or '--'} · Completed {complete} · Loss {loss_text} · Elapsed {_format_duration(snapshot.get('elapsed'))} · ETA {_format_duration(snapshot.get('eta'))}</div>
</div>'''


def training_poll_ui():
    global _TRAIN_WAS_RUNNING, _TRAIN_CHIMED_EXIT
    state, _tail = training_status()
    snap = training_progress_snapshot()
    running = bool(snap.get("running"))
    exit_code = snap.get("exit_code")
    if _TRAIN_WAS_RUNNING and not running and exit_code is not None and _TRAIN_CHIMED_EXIT != exit_code:
        play_completion_chime()
        _TRAIN_CHIMED_EXIT = exit_code
    if running:
        _TRAIN_CHIMED_EXIT = None
    _TRAIN_WAS_RUNNING = running
    return state, training_progress_html(snap), gr.update(interactive=not running), gr.update(interactive=running)

def build_ui() -> gr.Blocks:
    settings = read_json(SETTINGS, {})
    voices = voice_choices()
    projects = list_dataset_projects()
    training_projects = list_training_projects()
    instructions = instruction_choices()
    model_default = str(DEFAULT_MODEL)
    caps = runtime_capabilities(deep_probe=True)
    log(f"Runtime verifier: {capability_summary(caps)}")
    flash_value = bool(settings.get("flash_attention", False) and caps["flash_attention"])
    trt_value = bool(settings.get("trt", False) and caps["tensorrt"])
    compile_value = bool(settings.get("torch_compile", False) and caps["torch_compile"])

    with gr.Blocks(css=CSS, theme=gr.themes.Default(), title="CosyVoice3 Easy GUI") as demo:
        model_dir = gr.State(model_default)

        with gr.Row(elem_classes="title-section"):
            with gr.Column(scale=7, min_width=420):
                gr.Markdown("# 🎙️ [CosyVoice3 Easy GUI](https://github.com/Mixomo/CosyVoice3-Easy-GUI)")
                gr.Markdown("Inference, reusable voice/instruction libraries, dataset preparation and LoRA training · [official CosyVoice3 project](https://github.com/FunAudioLLM/CosyVoice)", elem_classes="tab-subtitle")
            with gr.Column(scale=3, min_width=480):
                with gr.Row():
                    top_unload = gr.Button("🧹 Unload All Models", size="sm", variant="secondary")
                    top_clear_outputs = gr.Button("🗑️ Clear Outputs", size="sm", variant="secondary")
                    top_clear_voices = gr.Button("🗑️ Clear Samples", size="sm", variant="stop")
        # Hidden callback target for top-bar maintenance actions.
        global_status = gr.Markdown("", visible=False)
        with gr.Tabs(elem_classes="tabs"):
            # -----------------------------------------------------------------
            # PREP SAMPLES
            # -----------------------------------------------------------------
            with gr.Tab("🎙️ Prep Samples"):
                gr.Markdown("*Build reusable voice references and natural-language instruction presets before inference or training.*", elem_classes="tab-subtitle")
                with gr.Accordion("📖 Quick Guide", open=False):
                    gr.Markdown("""### Voice Library
1. Load or record a clean reference clip. If the backend-visible audio path has a same-name `.txt`, the GUI reuses it automatically; browser uploads may be copied to a temporary path.
2. If no transcript is available, open **Faster-Whisper Transcription** and transcribe locally on CUDA. Correct the text before saving.
3. Enter a short **Voice Name** and save. Selecting a saved voice later restores its audio and transcript automatically.

### Instruction Library
CosyVoice3 accepts ordinary natural-language performance directions. Save reusable instructions such as *Speak softly and warmly*, *Use Argentine Spanish*, or *Read this as an excited sports announcer*. Do not type `<|endofprompt|>`; the GUI appends that internal token.

These libraries are shared by TTS / Voice Clone and Dialogue Builder.""")

                gr.Markdown("### 📚 Voice Library", elem_classes="section-heading")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        voice_select = gr.Dropdown(voices, value=NONE, label="Saved Voice",
                                                   info="Select a reusable reference. Its name, audio and transcript are restored automatically.")
                        with gr.Row():
                            voice_refresh = gr.Button("↻", variant="secondary")
                            voice_delete = gr.Button("🗑️ Delete", variant="stop")
                        voice_preview = gr.Audio(label="Library Preview", interactive=False,
                                                 waveform_options=gr.WaveformOptions(show_recording_waveform=True))
                        voice_saved_text = gr.Textbox(label="Saved Transcript", interactive=False, lines=5,
                                                      info="Read-only copy stored with the selected voice.")
                    with gr.Column(scale=2, min_width=500):
                        voice_audio = gr.Audio(label="Reference Audio · 30 seconds of audio maximum", sources=["upload", "microphone"], type="filepath",
                                               waveform_options=gr.WaveformOptions(show_recording_waveform=True))
                        with gr.Accordion("🛰️ Faster-Whisper Transcription (Optional)", open=False):
                            gr.Markdown("Use this only when the reference has no trustworthy transcript. Auto-detect is convenient, but an explicit language can improve transcription consistency.")
                            with gr.Row():
                                prep_asr_model = gr.Dropdown(ASR_MODELS, value="large-v3", label="Whisper Model",
                                                             info="Larger models are slower but generally more accurate.", elem_classes="medium-control")
                                prep_asr_language = gr.Dropdown(ASR_LANGUAGES, value="Auto-detect", label="Whisper Language",
                                                                info="Choose an explicit language when Auto-detect guesses incorrectly.", elem_classes="medium-control")
                                prep_asr_batch = gr.Slider(1, 16, 4, 1, label="Batch Size",
                                                           info="Higher values can improve throughput but use more VRAM.")
                            prep_transcribe = gr.Button("📝 Transcribe Reference Audio", variant="secondary")
                        voice_transcript = gr.Textbox(label="Reference Transcript", lines=5,
                                                      placeholder="Exact words spoken in the reference audio.",
                                                      info="Required for zero-shot cloning. Correct Whisper mistakes before saving.")
                        voice_detected_language = gr.State("Auto")
                        voice_name = gr.Textbox(label="Voice Name", placeholder="e.g. narrator_warm",
                                                info="Short reusable name used in inference and Dialogue Builder.")
                        voice_save = gr.Button("💾 Save Voice", variant="primary")
                        voice_status = gr.Markdown("Ready.", elem_classes="compact-status")

                gr.Markdown("### 🧭 Instruction Library", elem_classes="section-heading")
                gr.Markdown(
                    "Instructions are free-form CosyVoice3 conditioning text for **language/accent, emotion, pacing, loudness and speaking style**. "
                    "Selecting a preset fills the instruction editor immediately so it can be reviewed or adjusted before use.", elem_classes="instruction-help"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        instruction_select = gr.Dropdown(instructions, value=NONE, label="Saved Instruction",
                                                         info="Load a saved instruction into the editor.")
                        with gr.Row():
                            instruction_refresh = gr.Button("↻", variant="secondary")
                            instruction_delete = gr.Button("🗑️ Delete", variant="stop")
                    with gr.Column(scale=2, min_width=500):
                        instruction_name = gr.Textbox(label="Instruction Name", placeholder="e.g. warm_spanish_storyteller",
                                                      info="Reusable preset name.")
                        instruction_text = gr.Textbox(
                            label="Natural-language Instruction", lines=5,
                            placeholder="Example: Speak in a warm, intimate tone, slightly slower than normal, with clear Spanish pronunciation.",
                            info="Write normal prose. The required CosyVoice3 end-of-prompt marker is added internally."
                        )
                        instruction_save = gr.Button("💾 Save Instruction", variant="primary")
                        instruction_status = gr.Markdown("No saved instruction selected.", elem_classes="compact-status")

            # -----------------------------------------------------------------
            # INFERENCE
            # -----------------------------------------------------------------
            with gr.Tab("🔊 Inference"):
                gr.Markdown("*Configure the runtime once, then synthesize a single utterance or build a multi-turn dialogue.*", elem_classes="tab-subtitle")
                with gr.Accordion("📖 Quick Guide", open=False):
                    gr.Markdown("""### Choose the mode
- **Zero-shot** — best for ordinary voice cloning. Requires reference audio **and the exact transcript spoken in that audio**.
- **Cross-lingual** — clones the reference voice without requiring its transcript. Useful when source and target languages differ.
- **Instruct** — uses reference audio plus a free-form instruction for accent/language, emotion, pacing, loudness or style.

### Reference audio and transcript
Choose a **Saved Voice** or load audio directly. Saved voices restore their transcript automatically. When a directly loaded path still exposes a same-name `.txt`, the GUI reuses it; browser uploads may be temporary, so the optional Faster-Whisper panel is the reliable fallback.

### Runtime language
**Inference Language** is applied as an explicit natural-language control in **Instruct** mode. Zero-shot and Cross-lingual use CosyVoice3's native text/reference language behavior, so `Auto` is normally correct there.

### Generation
Write the speech in **Target Speech**, keep generation parameters at their defaults first, and only tune decoding/RAS/flow controls when solving a specific artifact or stability issue. Long-text chunking is optional and has no character-count limit.""")

                gr.Markdown("### ⚡ Runtime", elem_classes="section-heading")
                with gr.Row(elem_classes="runtime-row"):
                    model_variant = gr.Radio(
                        ["Base", "RL"], value=settings.get("model_variant", "Base"), label="Checkpoint", scale=2, min_width=220,
                        info="Base is the standard released checkpoint. RL uses the released post-trained llm.rl.pt weights."
                    )
                    infer_language = gr.Dropdown(
                        LANGUAGE_CHOICES, value="Auto", label="Inference Language", scale=2, min_width=220, elem_classes="small-control",
                        info="Explicit language control is injected in Instruct mode. Auto is normally correct for Zero-shot/Cross-lingual."
                    )
                gr.Markdown("### ⚙️ Acceleration engines", elem_classes="section-heading")
                with gr.Row(elem_classes="runtime-row"):
                    fp16 = gr.Checkbox(True, label="FP16", info="Use half precision on CUDA to reduce VRAM and improve throughput.")
                    trt = gr.Checkbox(trt_value, label="TensorRT", interactive=caps["tensorrt"],
                                      info="May improve inference speed by running supported synthesis components with TensorRT.")
                    flash_attention = gr.Checkbox(flash_value, label="FlashAttention 2", interactive=caps["flash_attention"],
                                                  info="May improve inference speed and reduce attention memory use on supported NVIDIA GPUs.")
                    torch_compile = gr.Checkbox(compile_value, label="torch.compile / Inductor", interactive=caps["torch_compile"],
                                                info="May improve inference speed after the first run while Inductor builds optimized kernels.")
                gr.Markdown(capability_summary(caps), elem_classes="compact-status")

                with gr.Tabs():
                    # ---------------------------------------------------------
                    # Single inference
                    # ---------------------------------------------------------
                    with gr.Tab("TTS / Voice Clone"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=1, min_width=360):
                                mode = gr.Radio(
                                    ["Zero-shot", "Cross-lingual", "Instruct"], value="Zero-shot", label="Generation Mode",
                                    info="Controls which CosyVoice3 inference API is used and which fields below are relevant."
                                )
                                with gr.Row():
                                    infer_adapter = gr.Dropdown(
                                        adapter_values(), value=settings.get("adapter", NONE) or NONE,
                                        label="LoRA Adapter", scale=4,
                                        info="Optional adapter for this single inference. Leave None for the selected Base/RL checkpoint."
                                    )
                                    infer_adapter_refresh = gr.Button("↻", size="sm", variant="secondary", scale=0, min_width=52)
                                with gr.Row():
                                    infer_voice = gr.Dropdown(
                                        voices, value=NONE, label="Saved Voice", scale=4,
                                        info="Selecting a saved voice fills its reference audio and saved transcript automatically."
                                    )
                                    infer_voice_refresh = gr.Button("↻", size="sm", variant="secondary", scale=0, min_width=52)
                                infer_audio = gr.Audio(
                                    label="Reference Audio · 30 seconds of audio maximum", sources=["upload", "microphone"], type="filepath",
                                    waveform_options=gr.WaveformOptions(show_recording_waveform=True)
                                )
                                with gr.Accordion("🛰️ Faster-Whisper Transcription (Optional)", open=False):
                                    gr.Markdown("Use this when loading reference audio directly and no accurate transcript is available. The resulting text fills **Reference Transcript**.")
                                    with gr.Row():
                                        infer_asr_model = gr.Dropdown(ASR_MODELS, value="large-v3", label="Whisper Model",
                                                                      info="Larger models are generally more accurate but use more VRAM.", elem_classes="medium-control")
                                        infer_asr_language = gr.Dropdown(ASR_LANGUAGES, value="Auto-detect", label="Whisper Language",
                                                                         info="Set explicitly if Auto-detect chooses the wrong language.", elem_classes="medium-control")
                                        infer_asr_batch = gr.Slider(1, 16, 4, 1, label="Batch Size",
                                                                    info="Higher values can improve throughput but increase VRAM usage.")
                                    infer_transcribe = gr.Button("📝 Transcribe Reference Audio", variant="secondary")
                                with gr.Column(visible=True) as prompt_controls:
                                    infer_prompt = gr.Textbox(
                                        label="Reference Transcript", lines=5,
                                        placeholder="Exact words spoken in the reference audio.",
                                        info="Zero-shot only. Must match the reference speech as closely as possible."
                                    )
                                with gr.Column(visible=False) as instruct_controls:
                                    with gr.Row():
                                        infer_instruction_library = gr.Dropdown(
                                            instructions, value=NONE, label="Instruction Preset", scale=4,
                                            info="Loading a preset immediately fills the instruction editor below."
                                        )
                                        infer_instruction_refresh = gr.Button("↻", size="sm", variant="secondary", scale=0, min_width=52)
                                    infer_instruction = gr.Textbox(
                                        label="Natural-language Instruction", lines=5,
                                        placeholder="Example: Speak softly, with a warm emotional tone and measured pacing.",
                                        info="Instruct mode only. Write ordinary language/accent/emotion/speed/volume/style directions."
                                    )
                                infer_status = gr.Markdown("Ready.", elem_classes="compact-status")

                            with gr.Column(scale=2, min_width=560):
                                target_text = gr.Textbox(
                                    label="Target Speech", lines=11, placeholder="Enter the text to synthesize…",
                                    info="Main text CosyVoice3 will speak. Pronunciation inpainting tokens should normally be placed inline here."
                                )
                                with gr.Accordion("❓ Pronunciation / Control Markup Guide", open=False):
                                    gr.Markdown("""CosyVoice3 does **not** accept arbitrary bracket tags. Its tokenizer contains a finite, explicit vocabulary. Unknown tags should be treated as ordinary text and may be spoken literally or behave unpredictably.

### Supported vocal / style tokens
These are the user-facing control tokens explicitly registered by the CosyVoice3 tokenizer:

`[breath]` · `[quick_breath]` · `[noise]` · `[laughter]` · `<laughter>...</laughter>` · `[cough]` · `[clucking]` · `[accent]` · `[hissing]` · `[sigh]` · `[vocalized-noise]` · `[lipsmack]` · `[mn]` · `<strong>...</strong>`

Place an event **inline where it should happen**, for example `Hello [breath] everyone.` or `I <strong>really</strong> mean it.` The small **Append Control Tokens** box above is only a convenience for a trailing event such as `[breath]`; it is not required.

### English pronunciation inpainting
English uses the tokenizer's finite **CMU / ARPAbet** inventory. Consonants include `[B] [CH] [D] [DH] [F] [G] [HH] [JH] [K] [L] [M] [N] [NG] [P] [R] [S] [SH] [T] [TH] [V] [W] [Y] [Z] [ZH]`. Vowels use CMU symbols with optional stress `0/1/2`, for example `[AH0]`, `[IY1]`, `[ER0]`, `[OW1]`, `[AE2]`, `[AA1]`. Supported vowel families are `AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW`.

Pronunciation tokens belong **inside Target Speech at the word/location being corrected**; do not append a phoneme string at the end unless you actually want it spoken there.

### Chinese pronunciation inpainting
Chinese uses a finite set of bracketed **Pinyin initials/finals and tone-marked finals** registered by the tokenizer, e.g. `[j][ǐ]`, `[zh]`, `[ang]`, `[iǎo]`, `[uǒ]`. The official CosyVoice3 feature is pronunciation inpainting with Chinese Pinyin and English CMU phonemes.

Example pattern: `报道[j][ǐ]予好评。`

### Not user markup
Tokens such as `<|im_start|>`, `<|im_end|>`, `<|endofprompt|>`, `<|endofsystem|>` and `<|endoftext|>` are tokenizer/system delimiters managed internally by the GUI. Do **not** type them into normal speech text.

For ordinary TTS, use plain text and leave markup empty.""")
                                with gr.Row():
                                    seed = gr.Number(0, precision=0, label="Seed", scale=1,
                                                     info="Reproduce a generation when Random Seed is disabled.")
                                    random_seed = gr.Checkbox(True, label="Random Seed", scale=1,
                                                              info="Generate a fresh seed for every request.")
                                    speed = gr.Slider(0.25, 4.0, 1.0, 0.05, label="Speed", scale=3,
                                                      info="Post-generation speech-rate control used by the CosyVoice3 inference path.")

                                with gr.Accordion("⚙️ Advanced Generation Parameters", open=False):
                                    gr.Markdown("These are low-level CosyVoice3 decoding controls. Defaults mirror the released eager/RAS/flow configuration; change one variable at a time when troubleshooting generation quality.")
                                    with gr.Row():
                                        top_k = gr.Slider(1, 100, 25, 1, label="Top-K", info="Limits LLM sampling to the K most likely next speech tokens.")
                                        top_p = gr.Slider(0.05, 1.0, 0.8, 0.01, label="Top-P", info="Nucleus-sampling probability mass.")
                                        temperature = gr.Slider(0.1, 2.0, 1.0, 0.05, label="LLM Temperature", info="Higher values increase token sampling variability.")
                                    with gr.Row():
                                        ras_window = gr.Slider(1, 50, 10, 1, label="RAS Window", info="History window used by Repetition Aware Sampling.")
                                        ras_threshold = gr.Slider(0.01, 1.0, 0.1, 0.01, label="RAS Repetition Threshold", info="Sensitivity of repetition suppression inside the RAS window.")
                                        text_frontend = gr.Checkbox(True, label="Text Frontend", info="Apply CosyVoice text normalization/segmentation before tokenization.")
                                    with gr.Row():
                                        min_token_ratio = gr.Slider(0.5, 10.0, 2.0, 0.5, label="Min Token / Text Ratio", info="Lower bound used when deriving allowed speech-token length from input text.")
                                        max_token_ratio = gr.Slider(5.0, 60.0, 20.0, 1.0, label="Max Token / Text Ratio", info="Upper bound controlling the longest speech-token sequence allowed for the text.")
                                    with gr.Row():
                                        flow_steps = gr.Slider(1, 50, 10, 1, label="Flow Steps", info="Number of flow-matching decoding steps. More steps can cost more compute.")
                                        flow_temperature = gr.Slider(0.1, 2.0, 1.0, 0.05, label="Flow Temperature", info="Controls stochasticity/noise in flow generation.")

                                with gr.Accordion("📚 Long Text / Chunking", open=False):
                                    with gr.Row():
                                        chunk_mode = gr.Dropdown(
                                            CHUNK_CHOICES, value="None", label="Chunking Rule", scale=2,
                                            info="Split only on the selected linguistic boundary. None sends the text as one request.", elem_classes="medium-control"
                                        )
                                        chunk_gap = gr.Slider(
                                            0.0, 2.0, 0.15, 0.05, label="Silence Between Chunks (s)", scale=3,
                                            info="Inserted only when multiple generated chunks are merged."
                                        )
                                    gr.Markdown(
                                        "There is **no arbitrary character limit**. Generation length is controlled by CosyVoice3's token/text ratios and model limits; chunking only chooses safe linguistic split points.",
                                        elem_classes="instruction-help"
                                    )
                                with gr.Row(elem_classes="workflow-actions"):
                                    generate = gr.Button("▶ Generate", variant="primary", interactive=True)
                                    infer_stop = gr.Button("⏹ Stop", variant="stop", interactive=False)
                                infer_output = gr.Audio(label="Generated Audio", interactive=False, elem_classes="audio-safe-space")

                    # ---------------------------------------------------------
                    # Dialogue Builder
                    # ---------------------------------------------------------
                    with gr.Tab("Dialogue Builder"):
                        with gr.Accordion("📖 Quick Guide", open=False):
                            gr.Markdown("""1. Set **Target Language**, **Silence Between Turns**, chunking and the default instruction once in **Dialogue Defaults**.
2. Choose **Zero-shot**, **Cross-lingual** or **Instruct** independently for every turn, then select its saved **Speaker** and write the speech.
3. Zero-shot uses the speaker's stored transcript; Cross-lingual needs only its audio; Instruct reveals an optional per-turn instruction editor.
4. Leave **Instruction Override** empty in Instruct mode to inherit the shared default instruction and target language.
5. Use **Add / Clone / Up / Down / Clear / Delete** to edit rows, then generate the complete dialogue. The shared silence slider is inserted between rendered turns.""")

                        gr.Markdown("### 🎛️ Dialogue Defaults", elem_classes="section-heading")
                        with gr.Row():
                            dialogue_count = gr.Slider(1, MAX_DIALOGUE_ROWS, 2, 1, label="Visible Turns", scale=2,
                                                       info="Number of dialogue rows shown below; empty rows are skipped during generation.")
                            dialogue_language = gr.Dropdown(LANGUAGE_CHOICES, value="Auto", label="Target Language", scale=2,
                                                            info="Shared by all turns. Auto leaves language control to the reference/model.")
                            dialogue_turn_silence = gr.Slider(0.0, 3.0, 0.25, 0.05, label="Silence Between Turns (s)", scale=3,
                                                              info="Fixed pause inserted between generated turns when the final dialogue is merged.")
                            dialogue_chunk_mode = gr.Dropdown(CHUNK_CHOICES, value="None", label="Chunking Rule", scale=2,
                                                              info="Optional linguistic chunking applied inside each turn.", elem_classes="medium-control")
                        with gr.Row():
                            dialogue_seed = gr.Number(0, precision=0, label="Base Seed", scale=1,
                                                      info="Base seed; each rendered turn receives a deterministic offset.")
                            dialogue_random_seed = gr.Checkbox(True, label="Random Seed", scale=1,
                                                               info="Generate a new base seed for each complete dialogue.")
                            dialogue_instruction_library = gr.Dropdown(instructions, value=NONE, label="Default Instruction Preset", scale=2,
                                                                        info="Loading a preset fills the shared instruction editor below.", elem_classes="medium-control")
                            dialogue_instruction_refresh = gr.Button("↻", size="sm", variant="secondary", scale=0, min_width=52)
                        dialogue_default_instruction = gr.Textbox(
                            label="Default Natural-language Instruction (Optional)", lines=3,
                            placeholder="Applied to every turn unless that row provides an override.",
                            info="Free-form style/language/emotion direction. Leave empty for ordinary zero-shot dialogue when Target Language is Auto."
                        )

                        with gr.Accordion("⚙️ Dialogue Generation Parameters", open=False):
                            with gr.Row():
                                dialogue_speed = gr.Slider(0.25, 4.0, 1.0, 0.05, label="Speed", info="Shared speech-rate control for all rendered turns.")
                                dialogue_top_k = gr.Slider(1, 100, 25, 1, label="Top-K", info="Shared LLM top-K sampling limit for all turns.")
                                dialogue_top_p = gr.Slider(0.05, 1.0, 0.8, 0.01, label="Top-P", info="Shared nucleus-sampling probability mass.")
                                dialogue_temperature = gr.Slider(0.1, 2.0, 1.0, 0.05, label="LLM Temperature", info="Shared LLM sampling variability.")
                            with gr.Row():
                                dialogue_ras_window = gr.Slider(1, 50, 10, 1, label="RAS Window", info="History window for Repetition Aware Sampling.")
                                dialogue_ras_threshold = gr.Slider(0.01, 1.0, 0.1, 0.01, label="RAS Repetition Threshold", info="Shared repetition-suppression sensitivity.")
                                dialogue_text_frontend = gr.Checkbox(True, label="Text Frontend", info="Normalize/segment each turn with the CosyVoice text frontend before tokenization.")
                            with gr.Row():
                                dialogue_min_ratio = gr.Slider(0.5, 10.0, 2.0, 0.5, label="Min Token / Text Ratio", info="Shared minimum allowed speech-token length relative to text.")
                                dialogue_max_ratio = gr.Slider(5.0, 60.0, 20.0, 1.0, label="Max Token / Text Ratio", info="Shared maximum allowed speech-token length relative to text.")
                                dialogue_flow_steps = gr.Slider(1, 50, 10, 1, label="Flow Steps", info="Shared number of flow-matching decoding steps.")
                                dialogue_flow_temperature = gr.Slider(0.1, 2.0, 1.0, 0.05, label="Flow Temperature", info="Shared flow-generation stochasticity.")

                        with gr.Row(elem_classes="dialogue-toolbar"):
                            dialogue_reset = gr.Button("↺ Reset", variant="secondary")
                            dialogue_clear_all = gr.Button("🧹 Clear Text", variant="secondary")
                            dialogue_compact = gr.Button("↕ Remove Empty", variant="secondary")
                            dialogue_refresh_voices = gr.Button("↻ Refresh Voices", variant="secondary")
                            dialogue_refresh_adapters = gr.Button("↻ Refresh Adapters", variant="secondary")

                        dialogue_groups = []
                        dialogue_inputs = []
                        dialogue_speakers = []
                        dialogue_adapters = []
                        dialogue_modes = []
                        dialogue_instruction_fields = []
                        dialogue_actions = []
                        for index in range(MAX_DIALOGUE_ROWS):
                            with gr.Column(visible=index < 2, elem_classes="dialogue-turn-card") as group:
                                gr.Markdown(f"#### Turn {index + 1}", elem_classes="dialogue-turn-title")
                                with gr.Row(equal_height=False):
                                    row_mode = gr.Dropdown(
                                        ["Zero-shot", "Cross-lingual", "Instruct"], value="Zero-shot",
                                        label="Mode", scale=2, min_width=190,
                                        info="Zero-shot uses the saved transcript; Cross-lingual uses audio only; Instruct enables directions."
                                    )
                                    speaker = gr.Dropdown(voices, value=NONE, label="Speaker", scale=2, min_width=220, info="Saved voice reference used for this turn.")
                                    row_adapter = gr.Dropdown(
                                        adapter_values(), value=NONE, label="LoRA Adapter", scale=3, min_width=260,
                                        info="Optional trained adapter applied only to this turn.", elem_classes="medium-control"
                                    )
                                    text = gr.Textbox(label="Speech", lines=4, scale=7, placeholder=f"Text for turn {index + 1}…", info="Text spoken by this turn.")
                                instruction = gr.Textbox(label="Instruction Override (Optional)", lines=2, visible=False,
                                                         placeholder="Leave empty to inherit Dialogue Defaults.",
                                                         info="Instruct mode only. This turn uses the override; empty inherits the shared instruction.")
                                with gr.Row(elem_classes="dialogue-actions"):
                                    add = gr.Button("➕ Add", size="sm", variant="secondary")
                                    clone = gr.Button("📋 Clone", size="sm", variant="secondary")
                                    up = gr.Button("⬆ Up", size="sm", variant="secondary")
                                    down = gr.Button("⬇ Down", size="sm", variant="secondary")
                                    clear = gr.Button("🧹 Clear", size="sm", variant="secondary")
                                    delete = gr.Button("🗑 Delete", size="sm", variant="stop")
                            dialogue_groups.append(group)
                            dialogue_inputs.extend([row_mode, speaker, row_adapter, text, instruction])
                            dialogue_speakers.append(speaker)
                            dialogue_adapters.append(row_adapter)
                            dialogue_modes.append(row_mode)
                            dialogue_instruction_fields.append(instruction)
                            dialogue_actions.append((add, clone, up, down, clear, delete))

                        with gr.Row(elem_classes="workflow-actions"):
                            dialogue_generate_btn = gr.Button("▶ Generate Dialogue", variant="primary", interactive=True)
                            dialogue_stop = gr.Button("⏹ Stop", variant="stop", interactive=False)
                        dialogue_output = gr.Audio(label="Dialogue Output", interactive=False, elem_classes="audio-safe-space")
                        dialogue_status = gr.Markdown("Ready.", elem_classes="compact-status")

            # -----------------------------------------------------------------
            # DATASET
            # -----------------------------------------------------------------
            with gr.Tab("📂 Dataset Preparation"):
                gr.Markdown("*Turn source audio + transcripts into a reproducible Train/CV CosyVoice3 dataset, then extract the ONNX features/parquet lists required by LoRA training.*", elem_classes="tab-subtitle")
                with gr.Accordion("📖 Quick Guide", open=False):
                    gr.Markdown("""### 1 · Project and source
Select an existing **Project Name** or type a new name and press **Create Project**. **Clone Project** never asks for a second name: it creates the next incremental project automatically (`name-02`, `name-03`, ...).

Choose the folder containing audio. Same-basename `.txt` files are used as transcripts. If some clips have no sidecar text, enable the optional Faster-Whisper pass.

### 2 · Analyze, then prepare
**Analyze Source** discovers files and validates audio/transcript pairs without writing the final dataset. Review the status message before continuing.

**Training Instruction** should normally stay **Standard** for ordinary voice adaptation. Choose **Custom** only if the entire dataset intentionally represents one consistent instructed style/accent/emotion.

**CV Split (%)** is the validation holdout: `10%` means approximately `90% Train / 10% CV`. CV data is not used for gradient updates; it is logged as a diagnostic.

### 3 · Build training artifacts
**Prepare Train / CV** normalizes audio and writes the CosyVoice mappings/manifest. **Extract Features + Parquet** then runs the official embedding/speech-token ONNX pipeline. Training should start only after this final step succeeds.""")

                gr.Markdown("### 📁 Project", elem_classes="section-heading")
                with gr.Row(elem_classes="project-strip"):
                    dataset_project = gr.Dropdown(
                        projects, value=None, label="Project Name", scale=4, allow_custom_value=True,
                        info="Select an existing project or type a new name. Create uses this exact name; Clone creates the next numbered copy automatically."
                    )
                    project_create = gr.Button("＋ Create Project", variant="secondary", scale=1)
                    project_save = gr.Button("💾 Save Project", variant="secondary", scale=1)
                    project_clone = gr.Button("📋 Clone Project", variant="secondary", scale=1)
                    project_delete = gr.Button("🗑 Delete", variant="stop", scale=1)

                gr.Markdown("### 1 · Source Analysis", elem_classes="section-heading")
                with gr.Row():
                    source_folder = gr.Textbox(
                        label="Source Audio Folder", placeholder=r"J:\datasets\my_voice", scale=6,
                        info="Folder scanned recursively for WAV/FLAC/MP3/M4A/OGG files and optional same-name .txt transcripts."
                    )
                    dataset_browse = gr.Button("📂 Browse", variant="secondary", scale=1, min_width=110)
                    dataset_language = gr.Dropdown(
                        LANGUAGE_CHOICES, value="Auto", label="Dataset Language", scale=2,
                        info="Metadata language written into prepared records. Use Auto for mixed-language datasets.", elem_classes="medium-control"
                    )
                with gr.Accordion("🛰️ Transcribe Missing Text with Faster-Whisper (Optional)", open=False):
                    dataset_transcribe_missing = gr.Checkbox(
                        False, label="Transcribe clips without sidecar .txt",
                        info="Only missing transcripts are generated; existing .txt files are preserved."
                    )
                    with gr.Row():
                        dataset_asr = gr.Dropdown(ASR_MODELS, value="large-v3", label="Whisper Model",
                                                  info="Model used only for missing dataset transcripts.", elem_classes="medium-control")
                        dataset_asr_language = gr.Dropdown(ASR_LANGUAGES, value="Auto-detect", label="Whisper Language",
                                                           info="Choose explicitly if language auto-detection is unreliable.", elem_classes="medium-control")
                        dataset_asr_batch = gr.Slider(1, 16, 4, 1, label="Batch Size",
                                                      info="Higher values increase transcription throughput and VRAM use.")

                gr.Markdown("### 2 · Training Conditioning", elem_classes="section-heading")
                gr.Markdown(
                    "For normal speaker adaptation leave **Standard** selected. CosyVoice3 internally stores the neutral training prefix and the GUI manages the required end-of-prompt token. "
                    "Use **Custom** only when every utterance in this project intentionally shares the same instruction.", elem_classes="instruction-help"
                )
                dataset_instruction_source = gr.Radio(
                    ["Standard (Recommended)", "Custom"], value="Standard (Recommended)", label="Training Instruction",
                    info="Standard is appropriate for ordinary voice LoRA training. Custom applies one instruction to every dataset utterance."
                )
                with gr.Column(visible=False) as dataset_instruction_custom_group:
                    dataset_instruction_custom = gr.Textbox(
                        label="Custom Training Instruction", lines=4,
                        placeholder="Example: Speak in a restrained documentary narration style.",
                        info="Free-form CosyVoice3 instruction applied uniformly to every Train and CV record."
                    )
                cv_percent = gr.Slider(
                    1, 30, 10, 1, label="CV Split (%) — Validation Holdout",
                    info="Percentage reserved for cross-validation. Example: 10% ≈ 90% Train / 10% CV; at least one item is always reserved for CV."
                )

                gr.Markdown("### 3 · Build CosyVoice3 Dataset", elem_classes="section-heading")
                with gr.Row(elem_classes="workflow-actions"):
                    scan_btn = gr.Button("1. Analyze Source", variant="secondary")
                    prepare_btn = gr.Button("2. Prepare Train / CV", variant="primary")
                    extract_btn = gr.Button("3. Extract Features + Parquet", variant="secondary")
                dataset_manifest = gr.Textbox(
                    label="Prepared Manifest", interactive=False,
                    placeholder="Created after Prepare Train / CV.",
                    info="Path to the generated manifest used to track the prepared project."
                )
                dataset_status = gr.Markdown("Create/select a project, choose a source folder, then analyze it.")
                dataset_rows_state = gr.State("[]")

            # -----------------------------------------------------------------
            # TRAINING
            # -----------------------------------------------------------------
            with gr.Tab("🚀 LoRA Training"):
                gr.Markdown("*Train a single-GPU CosyVoice3 LoRA from a prepared project, with the same project/AutoTune/start-stop hierarchy used by the Easy GUI family.*", elem_classes="tab-subtitle")
                with gr.Accordion("📖 Quick Guide", open=False):
                    gr.Markdown("""### 1 · Select the prepared project
Choose a **Project Name** that already completed **Extract Features + Parquet**. Use **Clone Project** when you want a new incremental experiment without overwriting the current project metadata.

### 2 · Choose the training base checkpoint
**Base** trains the LoRA on the released standard `llm.pt`. **RL** trains on the released post-trained `llm.rl.pt`. This is independent of inference LoRA selection; there is no visible warm-start-adapter control.

### 3 · Resume or start clean
Leave **Resume Checkpoint = None** for a new run. To warm-start from a previous adapter, explicitly select one of the listed checkpoints. The selected adapter weights are loaded into a fresh optimizer run.

### 4 · Tune only the useful LoRA parameters
Use **AutoTune** as a conservative starting point for the dataset, then adjust Steps/Epochs, Rank, Alpha, Dropout or Learning Rate only when needed. CV evaluates the run but never stops it early.

### 5 · Evaluation reference
The optional **Eval Reference + Whisper** panel stores a stable reference clip, transcript and preview text with the project so trained adapters can be compared consistently. The reference is not part of the training loss.

### 6 · Start / Stop
**Start Training** locks while the external training process is running and **Stop** becomes available. When the process finishes or stops, the buttons automatically return to their idle state. TensorBoard reads the selected project's training logs.""")

                gr.Markdown("### 📁 Training Project", elem_classes="section-heading")
                with gr.Row(elem_classes="project-strip"):
                    train_project = gr.Dropdown(
                        training_projects, value=None, label="Project Name", scale=4,
                        info="Prepared dataset project used for training."
                    )
                    train_base_variant = gr.Radio(
                        ["Base", "RL"], value="Base", label="Training Base Checkpoint", scale=2,
                        info="Base = llm.pt. RL = released llm.rl.pt post-trained checkpoint. The resulting adapter records this base variant."
                    )
                    train_vram_preset = gr.Dropdown(
                        AUTOTUNE_PROFILES, value=AUTOTUNE_PROFILES[1], label="Manual Dataset Preset", scale=2,
                        info="Manual capacity profile by approximate prepared-audio minutes. AutoTune calculates the profile directly from the selected project."
                    )
                    autotune_btn = gr.Button("⚙ AutoTune", variant="secondary", scale=1)
                    train_project_save = gr.Button("💾 Save Project", variant="secondary", scale=1)
                    train_project_clone = gr.Button("📋 Clone Project", variant="secondary", scale=1)
                    with gr.Column(scale=1, min_width=180, elem_classes="project-actions-column"):
                        train_project_delete = gr.Button("🗑 Delete", variant="stop", scale=1)
                        clean_tensorboard_btn = gr.Button("🧹 Clean TensorBoard", variant="secondary", scale=1, size="sm")
                autotune_status = gr.Markdown(
                    "Press AutoTune to apply the profile and display the complete parameter/VRAM calculation.",
                    elem_classes="compact-status"
                )

                gr.Markdown("### ↩️ Training Resume", elem_classes="section-heading")
                with gr.Row():
                    train_resume = gr.Dropdown(
                        [NONE], value=NONE,
                        label="Resume Checkpoint", scale=4,
                        info="None starts a clean run. Selecting a checkpoint explicitly loads its adapter weights as a warm start."
                    )
                    train_resume_refresh = gr.Button("↻", size="sm", variant="secondary", scale=0, min_width=52, elem_classes="compact-button")
                train_resume_status = gr.Markdown("None = start a new training run.", elem_classes="compact-status")

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=420):
                        gr.Markdown("### 📦 Training Setup", elem_classes="section-heading")
                        train_length_mode = gr.Radio(["Steps", "Epochs"], value="Steps", label="Training Length")
                        with gr.Group(visible=True) as train_steps_group:
                            train_steps = gr.Slider(100, 20000, 1500, 100, label="Training Steps",
                                                    info="Exact optimizer updates after gradient accumulation. AutoTune targets this mode.")
                            train_save_steps = gr.Slider(50, 5000, 250, 50, label="Save Every (Steps)",
                                                         info="Round optimizer-step checkpoint cadence. The final target is always saved too.")
                        with gr.Group(visible=False) as train_epochs_group:
                            train_epochs = gr.Slider(1, 500, 20, 1, label="Epochs",
                                                     info="Maximum dataset passes; training runs to the selected target.")
                            train_save_epochs = gr.Slider(1, 100, 5, 1, label="Save Every (Epochs)",
                                                          info="Checkpoint cadence in complete dataset passes. Early/final stop is always saved too.")
                        train_seed = gr.Number(
                            1234, precision=0, label="Training Seed",
                            info="Seed recorded with the run for reproducibility. Strict deterministic CUDA mode stays internal/off for normal performance."
                        )
                        gr.Markdown(
                            "**Managed internally:** CV metrics are logged without early stopping · single-GPU checkpoints · adapter resume only when explicitly selected.",
                            elem_classes="instruction-help"
                        )
                    with gr.Column(scale=1, min_width=420):
                        gr.Markdown("### ⚙️ LoRA Hyperparameters", elem_classes="section-heading")
                        with gr.Row():
                            train_rank = gr.Slider(4, 128, 16, 4, label="LoRA Rank",
                                                   info="Adapter rank/capacity. Higher values increase trainable parameters and VRAM use.")
                            train_alpha = gr.Slider(8, 256, 64, 8, label="LoRA Alpha",
                                                    info="LoRA scaling factor; normally tuned together with Rank.")
                        with gr.Row():
                            train_dropout = gr.Slider(0, .3, .05, .01, label="Dropout",
                                                      info="Regularization inside LoRA layers. Small values such as 0.05 are typical.")
                            train_lr = gr.Slider(1e-6, 5e-4, 5e-5, 1e-6, label="Learning Rate",
                                                 info="Optimizer learning rate for LoRA training. Lower values are safer for small datasets.")
                        train_grad_accum = gr.Slider(1, 8, 2, 1, label="Gradient Accumulation",
                                                     info="Micro-batches accumulated before each optimizer update. Higher values increase effective batch size and reduce update frequency.")

                with gr.Accordion("🎧 Eval Reference + Faster-Whisper", open=False):
                    gr.Markdown(
                        "Store one stable reference and preview sentence with the project for repeatable post-training comparisons. "
                        "This audio/transcript is **not** used to compute Train or CV loss.", elem_classes="instruction-help"
                    )
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            gr.Markdown("**Eval Reference Audio** · Clean reference voice used when comparing trained adapters.", elem_classes="instruction-help")
                            train_eval_ref = gr.Audio(
                                type="filepath", label="Eval Reference Audio · 30 seconds of audio maximum", sources=["upload", "microphone"],
                                elem_classes="audio-safe-space"
                            )
                            train_eval_ref_text = gr.Textbox(
                                label="Eval Reference Transcript", lines=4,
                                placeholder="Exact words spoken in Eval Reference Audio.",
                                info="Can be typed manually, loaded from a sidecar .txt, or generated with Faster-Whisper."
                            )
                        with gr.Column(scale=1):
                            train_eval_text = gr.Textbox(
                                value="This is a CosyVoice3 training preview.", label="Eval Text", lines=4,
                                info="Fixed sentence to synthesize when comparing checkpoints/adapters later."
                            )
                            train_eval_language = gr.Dropdown(
                                LANGUAGE_CHOICES, value="Auto", label="Eval Language",
                                info="Language used for the evaluation preview; Auto leaves control to the model/reference."
                            )
                    with gr.Row():
                        train_eval_asr_model = gr.Dropdown(
                            ASR_MODELS, value="large-v3", label="Whisper Model",
                            info="Local Faster-Whisper model used only to transcribe the eval reference.", elem_classes="medium-control"
                        )
                        train_eval_asr_language = gr.Dropdown(
                            ASR_LANGUAGES, value="Auto-detect", label="Whisper Language",
                            info="Set explicitly if Auto-detect chooses the wrong language.", elem_classes="medium-control"
                        )
                        train_eval_asr_batch = gr.Slider(
                            1, 16, 4, 1, label="Batch Size",
                            info="Transcription batch size; larger values use more VRAM."
                        )
                    train_eval_transcribe = gr.Button("🛰️ Transcribe Eval Audio", variant="secondary")
                    train_eval_status = gr.Markdown("Evaluation reference is optional.", elem_classes="compact-status")

                with gr.Row(elem_classes="global-toolbar"):
                    train_start = gr.Button("🚀 Start Training", variant="primary", interactive=True)
                    train_stop = gr.Button("⏹ Stop", variant="stop", interactive=False)
                    train_tensorboard = gr.Button("📊 TensorBoard", variant="secondary")
                    train_reload_tensorboard = gr.Button("↻ Reload TensorBoard", variant="secondary")
                train_progress = gr.HTML(value=training_progress_html())
                train_state = gr.Textbox(label="Training Status", value="Idle.", interactive=False)
                train_timer = gr.Timer(1.0)

        with gr.Accordion("🖥️ Console", open=True, elem_classes="console-accordion"):
            console = gr.HTML(value=html_view(), elem_id="cosyvoice-console")
            console_timer = gr.Timer(0.5)

        # ---------------------------------------------------------------------
        # Voice + Instruction libraries
        # ---------------------------------------------------------------------
        voice_select.change(
            voice_values, voice_select,
            [voice_name, voice_preview, voice_audio, voice_saved_text, voice_transcript, voice_detected_language, voice_status],
            queue=False,
        )
        voice_refresh.click(refresh_voice_ui, voice_select, [voice_select, infer_voice], queue=False)
        infer_voice_refresh.click(refresh_voice_ui, infer_voice, [voice_select, infer_voice], queue=False)
        voice_audio.change(sidecar_transcript_ui, [voice_audio, voice_transcript], [voice_transcript, voice_status], queue=False)
        prep_transcribe.click(
            transcribe_ui, [voice_audio, prep_asr_model, prep_asr_language, prep_asr_batch],
            [voice_transcript, voice_detected_language, voice_status],
        )
        voice_save.click(
            save_voice_ui, [voice_name, voice_audio, voice_transcript, voice_detected_language],
            [voice_select, voice_name, voice_preview, voice_audio, voice_saved_text, voice_status],
        )
        voice_delete.click(
            delete_voice_ui, voice_select,
            [voice_select, voice_name, voice_preview, voice_audio, voice_saved_text, voice_status],
        )

        instruction_targets = [instruction_select, infer_instruction_library, dialogue_instruction_library]
        instruction_select.change(
            load_instruction_ui, instruction_select, [instruction_name, instruction_text, instruction_status], queue=False
        )
        instruction_save.click(
            save_instruction_ui, [instruction_name, instruction_text],
            [instruction_select, instruction_name, instruction_text, instruction_status], queue=False,
        ).then(refresh_instruction_choices_ui, instruction_targets, instruction_targets, queue=False)
        instruction_delete.click(
            delete_instruction_ui, instruction_select,
            [instruction_select, instruction_name, instruction_text, instruction_status], queue=False,
        ).then(refresh_instruction_choices_ui, instruction_targets, instruction_targets, queue=False)
        instruction_refresh.click(refresh_instruction_choices_ui, instruction_targets, instruction_targets, queue=False)
        infer_instruction_refresh.click(refresh_instruction_choices_ui, instruction_targets, instruction_targets, queue=False)
        dialogue_instruction_refresh.click(refresh_instruction_choices_ui, instruction_targets, instruction_targets, queue=False)
        infer_instruction_library.change(instruction_text_for_choice, infer_instruction_library, infer_instruction, queue=False)
        dialogue_instruction_library.change(instruction_text_for_choice, dialogue_instruction_library, dialogue_default_instruction, queue=False)

        # ---------------------------------------------------------------------
        # Inference
        # ---------------------------------------------------------------------
        mode.change(mode_visibility, mode, [prompt_controls, instruct_controls], queue=False)
        infer_voice.change(inference_voice_values, infer_voice, [infer_audio, infer_prompt, infer_status], queue=False)
        infer_audio.change(sidecar_transcript_ui, [infer_audio, infer_prompt], [infer_prompt, infer_status], queue=False)
        infer_transcribe.click(
            transcribe_inference_ui, [infer_audio, infer_asr_model, infer_asr_language, infer_asr_batch],
            [infer_prompt, infer_status],
        )
        infer_adapter_refresh.click(refresh_adapter_ui, infer_adapter, infer_adapter, queue=False)
        infer_run = generate.click(
            lambda: running_button_updates(True), outputs=[generate, infer_stop], queue=False,
        ).then(
            generate_ui,
            [target_text, mode, infer_voice, infer_audio, infer_prompt, infer_instruction, infer_language,
             seed, random_seed, speed, text_frontend, top_k, top_p, temperature, ras_window, ras_threshold,
             min_token_ratio, max_token_ratio, flow_steps, flow_temperature, chunk_mode, chunk_gap,
             model_dir, model_variant, fp16, trt, flash_attention, torch_compile, infer_adapter],
            [infer_output, infer_status],
        )
        infer_run.then(lambda: running_button_updates(False), outputs=[generate, infer_stop], queue=False)
        infer_stop.click(ENGINE.cancel, outputs=infer_status, queue=False).then(
            lambda: gr.update(interactive=False), outputs=infer_stop, queue=False
        )

        # ---------------------------------------------------------------------
        # Dialogue Builder
        # ---------------------------------------------------------------------
        dialogue_count.change(dialogue_visibility, dialogue_count, dialogue_groups)
        dialogue_refresh_voices.click(refresh_dialogue_voices, dialogue_speakers, dialogue_speakers, queue=False)
        dialogue_refresh_adapters.click(refresh_dialogue_adapters, dialogue_adapters, dialogue_adapters, queue=False)
        for row_mode, instruction in zip(dialogue_modes, dialogue_instruction_fields):
            row_mode.change(dialogue_mode_visibility, row_mode, instruction, queue=False)
        dialogue_run = dialogue_generate_btn.click(
            lambda: running_button_updates(True), outputs=[dialogue_generate_btn, dialogue_stop], queue=False,
        ).then(
            dialogue_generate,
            [dialogue_count, model_dir, model_variant, fp16, trt, flash_attention, torch_compile,
             dialogue_seed, dialogue_random_seed, dialogue_language, dialogue_turn_silence, dialogue_default_instruction,
             dialogue_chunk_mode, dialogue_speed, dialogue_text_frontend, dialogue_top_k, dialogue_top_p, dialogue_temperature,
             dialogue_ras_window, dialogue_ras_threshold, dialogue_min_ratio, dialogue_max_ratio,
             dialogue_flow_steps, dialogue_flow_temperature, *dialogue_inputs],
            [dialogue_output, dialogue_status],
        )
        dialogue_run.then(lambda: running_button_updates(False), outputs=[dialogue_generate_btn, dialogue_stop], queue=False)
        dialogue_stop.click(ENGINE.cancel, outputs=dialogue_status, queue=False).then(
            lambda: gr.update(interactive=False), outputs=dialogue_stop, queue=False
        )
        dialogue_action_outputs = [dialogue_count, *dialogue_groups, *dialogue_inputs]
        dialogue_reset.click(lambda count, *values: dialogue_toolbar_action("reset", count, *values),
                             [dialogue_count, *dialogue_inputs], dialogue_action_outputs, queue=False)
        dialogue_clear_all.click(lambda count, *values: dialogue_toolbar_action("clear", count, *values),
                                 [dialogue_count, *dialogue_inputs], dialogue_action_outputs, queue=False)
        dialogue_compact.click(lambda count, *values: dialogue_toolbar_action("compact", count, *values),
                               [dialogue_count, *dialogue_inputs], dialogue_action_outputs, queue=False)
        action_names = ("add", "clone", "up", "down", "clear", "delete")
        for row_index, buttons in enumerate(dialogue_actions):
            for action, button in zip(action_names, buttons):
                button.click(lambda count, *values, _action=action, _index=row_index:
                             dialogue_row_action(_action, _index, count, *values),
                             [dialogue_count, *dialogue_inputs], dialogue_action_outputs, queue=False)

        # ---------------------------------------------------------------------
        # Dataset workflow
        # ---------------------------------------------------------------------
        dataset_instruction_source.change(
            instruction_source_visibility, dataset_instruction_source, dataset_instruction_custom_group, queue=False
        )
        dataset_project_inputs = [
            dataset_project, source_folder, dataset_language, dataset_transcribe_missing, dataset_asr, dataset_asr_language,
            dataset_asr_batch, dataset_instruction_source, dataset_instruction_custom, cv_percent, dataset_manifest, dataset_rows_state,
        ]
        training_project_inputs = [
            train_base_variant, train_vram_preset, train_resume, train_epochs, train_seed, train_rank, train_alpha, train_dropout, train_lr,
            train_eval_ref, train_eval_ref_text, train_eval_text, train_eval_language,
            train_eval_asr_model, train_eval_asr_language, train_eval_asr_batch,
        ]
        full_project_inputs = [*dataset_project_inputs, *training_project_inputs]
        train_full_project_inputs = [train_project, *dataset_project_inputs[1:], *training_project_inputs]
        dataset_project_outputs = [
            train_project, source_folder, dataset_language, dataset_transcribe_missing, dataset_asr, dataset_asr_language,
            dataset_asr_batch, dataset_instruction_source, dataset_instruction_custom, dataset_instruction_custom_group,
            cv_percent, dataset_manifest, dataset_rows_state, dataset_status,
        ]
        project_create.click(
            create_project_ui, full_project_inputs,
            [dataset_project, train_project, dataset_status],
        )
        project_save.click(
            save_training_project_ui, full_project_inputs,
            [dataset_project, train_project, dataset_status, train_eval_status], queue=False,
        ).then(save_training_length_ui, [train_project, train_length_mode, train_steps, train_save_steps, train_epochs, train_save_epochs], queue=False)
        project_clone.click(clone_project_ui, dataset_project, [dataset_project, train_project, dataset_status])
        project_delete.click(delete_project_ui, dataset_project, [dataset_project, train_project, dataset_status])
        dataset_project.change(load_project_ui, dataset_project, dataset_project_outputs, queue=False)
        dataset_project.change(
            load_training_project_ui, dataset_project,
            [train_eval_ref, train_eval_ref_text, train_eval_text, train_eval_language, train_base_variant,
             train_vram_preset, train_resume, train_length_mode, train_steps, train_save_steps, train_epochs, train_save_epochs, train_seed, train_rank, train_alpha, train_dropout, train_lr,
             train_eval_asr_model, train_eval_asr_language, train_eval_asr_batch], queue=False,
        )
        train_project.change(
            load_project_ui, train_project,
            [dataset_project, source_folder, dataset_language, dataset_transcribe_missing, dataset_asr, dataset_asr_language,
             dataset_asr_batch, dataset_instruction_source, dataset_instruction_custom, dataset_instruction_custom_group,
             cv_percent, dataset_manifest, dataset_rows_state, dataset_status], queue=False,
        )
        dataset_browse.click(browse_folder_ui, source_folder, source_folder, queue=False)
        scan_btn.click(
            scan_ui, [source_folder, dataset_transcribe_missing, dataset_asr, dataset_asr_language, dataset_asr_batch],
            [dataset_rows_state, dataset_status],
        )
        prepare_btn.click(
            prepare_ui,
            [dataset_project, dataset_rows_state, dataset_language, cv_percent, dataset_instruction_source, dataset_instruction_custom],
            [dataset_manifest, dataset_status],
        )
        extract_btn.click(extract_features_ui, [dataset_project, model_dir], dataset_status)

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------
        train_project.change(
            load_training_project_ui, train_project,
            [train_eval_ref, train_eval_ref_text, train_eval_text, train_eval_language, train_base_variant,
             train_vram_preset, train_resume, train_length_mode, train_steps, train_save_steps, train_epochs, train_save_epochs, train_seed, train_rank, train_alpha, train_dropout, train_lr,
             train_eval_asr_model, train_eval_asr_language, train_eval_asr_batch], queue=False,
        )
        train_length_mode.change(training_length_mode_ui, train_length_mode, [train_steps_group, train_epochs_group], queue=False)
        for cadence_control in (train_steps, train_save_steps, train_epochs, train_save_epochs):
            cadence_control.change(
                coherent_checkpoint_cadence_ui,
                [train_steps, train_save_steps, train_epochs, train_save_epochs],
                [train_save_steps, train_save_epochs], queue=False,
            )
        train_project.change(
            refresh_resume_ui, [train_project, train_base_variant, train_resume], train_resume, queue=False,
        )
        train_base_variant.change(
            refresh_resume_ui, [train_project, train_base_variant, train_resume], train_resume, queue=False,
        )
        train_resume_refresh.click(
            refresh_resume_ui, [train_project, train_base_variant, train_resume], train_resume, queue=False,
        )
        train_resume.change(
            load_resume_config_ui, train_resume,
            [train_base_variant, train_rank, train_alpha, train_dropout, train_lr, train_length_mode, train_steps,
             train_save_steps, train_epochs, train_save_epochs, train_seed, train_resume_status],
            queue=False,
        )
        train_eval_ref.change(
            sidecar_transcript_ui, [train_eval_ref, train_eval_ref_text], [train_eval_ref_text, train_eval_status], queue=False,
        )
        train_eval_transcribe.click(
            transcribe_inference_ui,
            [train_eval_ref, train_eval_asr_model, train_eval_asr_language, train_eval_asr_batch],
            [train_eval_ref_text, train_eval_status],
        )
        train_run = train_start.click(
            lambda: running_button_updates(True), outputs=[train_start, train_stop], queue=False,
        ).then(
            save_training_project_ui, train_full_project_inputs,
            [dataset_project, train_project, dataset_status, train_eval_status], queue=False,
        ).then(
            save_training_length_ui, [train_project, train_length_mode, train_steps, train_save_steps, train_epochs, train_save_epochs], queue=False,
        ).then(
            launch_training_ui,
            [train_project, model_dir, train_base_variant, train_resume, train_rank, train_alpha, train_dropout, train_lr, train_grad_accum,
             train_length_mode, train_steps, train_save_steps, train_epochs, train_save_epochs,
             train_eval_ref, train_eval_ref_text, train_eval_text, train_seed],
            train_state,
        )
        train_stop.click(stop_training, outputs=train_state, queue=False).then(
            lambda: gr.update(interactive=False), outputs=train_stop, queue=False
        )
        train_tensorboard.click(start_tensorboard, train_project, train_state)
        train_reload_tensorboard.click(restart_tensorboard, train_project, train_state)
        train_timer.tick(
            training_poll_ui, outputs=[train_state, train_progress, train_start, train_stop],
            queue=False, show_progress="hidden",
        )
        autotune_btn.click(
            autotune_dataset_ui,
            [train_project, train_rank, train_alpha, train_dropout, train_lr, train_epochs],
            [train_length_mode, train_steps, train_save_steps, train_rank, train_alpha, train_dropout, train_lr, train_epochs, train_save_epochs, train_grad_accum, autotune_status], queue=False,
        )
        train_vram_preset.change(
            autotune_training_ui,
            [train_project, train_vram_preset, train_rank, train_alpha, train_dropout, train_lr, train_epochs],
            [train_length_mode, train_steps, train_save_steps, train_rank, train_alpha, train_dropout, train_lr, train_epochs, train_save_epochs, train_grad_accum, autotune_status], queue=False,
        )
        train_project_save.click(
            save_training_project_ui, train_full_project_inputs,
            [dataset_project, train_project, dataset_status, train_eval_status], queue=False,
        ).then(save_training_length_ui, [train_project, train_length_mode, train_steps, train_save_steps, train_epochs, train_save_epochs], queue=False)
        train_project_clone.click(clone_project_ui, train_project, [dataset_project, train_project, dataset_status], queue=False)
        train_project_delete.click(
            delete_training_project_ui, train_project, [train_project, train_resume_status], queue=False,
        )
        clean_tensorboard_btn.click(
            clean_tensorboard_runs_ui, train_project, train_resume_status, queue=False,
        )

        # ---------------------------------------------------------------------
        # Global utilities
        # ---------------------------------------------------------------------
        top_unload.click(unload_all_ui, outputs=global_status, queue=False)
        top_clear_outputs.click(clear_outputs_ui, outputs=global_status, queue=False)
        top_clear_voices.click(clear_voices_ui, outputs=global_status, queue=False).then(
            refresh_voice_ui, voice_select, [voice_select, infer_voice], queue=False,
        ).then(
            lambda: ("", None, None, "", "", "Auto"),
            outputs=[voice_name, voice_preview, voice_audio, voice_saved_text, voice_transcript, voice_detected_language], queue=False,
        ).then(
            refresh_dialogue_voices, dialogue_speakers, dialogue_speakers, queue=False,
        )
        console_timer.tick(html_view, outputs=console, queue=False, show_progress="hidden")
    return demo

if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=int(os.environ.get("COSYVOICE_PORT", "7860")), inbrowser=True
    )
