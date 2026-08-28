from __future__ import annotations

from pathlib import Path

from easy_gui import sidecar_transcript_ui

ROOT = Path(__file__).resolve().parents[1]
GUI = (ROOT / "easy_gui.py").read_text(encoding="utf-8")
TRAINING = (ROOT / "cosyvoice_easy" / "training.py").read_text(encoding="utf-8")
RUNTIME = (ROOT / "cosyvoice_easy" / "runtime.py").read_text(encoding="utf-8")


def test_reference_and_instruction_libraries_autofill_editors():
    assert "voice_select.change(" in GUI
    assert "[voice_name, voice_preview, voice_audio, voice_saved_text, voice_transcript" in GUI
    assert "instruction_select.change(" in GUI
    assert "[instruction_name, instruction_text, instruction_status]" in GUI
    assert "infer_instruction_library.change(instruction_text_for_choice" in GUI
    assert "dialogue_instruction_library.change(instruction_text_for_choice" in GUI


def test_direct_inference_has_optional_whisper_and_runtime_language():
    assert 'label="Inference Language"' in GUI
    assert 'infer_transcribe = gr.Button("📝 Transcribe Reference Audio"' in GUI
    assert "infer_asr_model" in GUI and "infer_asr_language" in GUI
    assert 'label="Reference Language"' not in GUI


def test_instruction_notes_removed_and_dataset_conditioning_is_simple():
    assert "instruction_notes" not in GUI
    assert 'dataset_instruction_source = gr.Radio(' in GUI
    assert '["Standard (Recommended)", "Custom"]' in GUI
    assert "dataset_instruction_library" not in GUI


def test_pronunciation_markup_has_inline_guide_and_example():
    assert "Pronunciation / Control Markup Guide" in GUI
    assert "Hello [breath] everyone." in GUI
    assert "报道[j][ǐ]予好评。" in GUI
    assert "Append Control Tokens (Optional)" not in GUI


def test_project_clone_name_is_automatic_and_incremental():
    assert "Clone Name" not in GUI
    assert "New / Clone Name" not in GUI
    assert "_next_incremental_project_name" in GUI
    assert 'f"{base}-{index:02d}"' in GUI
    assert "clone_project_ui, dataset_project" in GUI
    assert "clone_project_ui, train_project" in GUI


def test_training_surface_hides_internal_safety_switches_and_exposes_base_variant():
    for obsolete_label in ["CV Patience", "Guarded Checkpoints", "Deterministic Training", "Warm-start Adapter", "Keep Last Checkpoints"]:
        assert f'label="{obsolete_label}"' not in GUI
    assert 'label="Training Base Checkpoint"' in GUI
    assert "CV patience = 3 validation epochs" in GUI
    assert "single-GPU checkpoints" in GUI


def test_base_and_rl_training_select_the_matching_checkpoint_and_adapter_contract():
    assert 'checkpoint = model / ("llm.rl.pt" if variant == "RL" else "llm.pt")' in TRAINING
    assert '"base_variant": variant' in TRAINING
    assert "Adapter was trained on the" in RUNTIME


def test_training_has_eval_reference_and_whisper_panel():
    assert "Eval Reference + Faster-Whisper" in GUI
    assert 'label="Eval Reference Audio"' in GUI
    assert 'label="Eval Reference Transcript"' in GUI
    assert 'train_eval_transcribe = gr.Button("🛰️ Transcribe Eval Audio"' in GUI


def test_start_stop_controls_use_mutually_exclusive_interactive_states():
    assert 'generate = gr.Button("▶ Generate", variant="primary", interactive=True)' in GUI
    assert 'infer_stop = gr.Button("⏹ Stop", variant="stop", interactive=False)' in GUI
    assert 'dialogue_generate_btn = gr.Button("▶ Generate Dialogue", variant="primary", interactive=True)' in GUI
    assert 'dialogue_stop = gr.Button("⏹ Stop", variant="stop", interactive=False)' in GUI
    assert 'train_start = gr.Button("🚀 Start Training", variant="primary", interactive=True)' in GUI
    assert 'train_stop = gr.Button("⏹ Stop", variant="stop", interactive=False)' in GUI
    assert "running_button_updates(True)" in GUI
    assert "training_poll_ui, outputs=[train_state, train_log, train_progress, train_start, train_stop]" in GUI


def test_cv_split_has_plain_language_explanation():
    assert "CV Split (%) — Validation Holdout" in GUI
    assert "90% Train / 10% CV" in GUI


def test_sidecar_helper_loads_accessible_transcript(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not-an-audio-test-fixture")
    audio.with_suffix(".txt").write_text("Exact transcript", encoding="utf-8")
    text, status = sidecar_transcript_ui(str(audio), "")
    assert text == "Exact transcript"
    assert "sidecar transcript" in status
