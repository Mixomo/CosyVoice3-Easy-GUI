from pathlib import Path

from cosyvoice_easy.instructions import instruction_body, normalize_instruction
from easy_gui import mode_visibility, resolve_dataset_instruction


ROOT = Path(__file__).resolve().parents[1]
GUI = (ROOT / "easy_gui.py").read_text(encoding="utf-8")


def test_native_first_gui_removes_obsolete_visual_workflow():
    assert "Discovered Samples" not in GUI
    assert "Discovered Dataset" not in GUI
    assert "gr.Dataframe(" not in GUI
    assert "Native Streaming Iterator" not in GUI
    assert "Max Characters per Chunk" not in GUI
    assert "chunk_chars" not in GUI
    assert "theme=gr.themes.Default()" in GUI


def test_dataset_workflow_is_explicit_and_instruction_is_human_facing():
    assert "1. Analyze Source" in GUI
    assert "2. Prepare Train / CV" in GUI
    assert "3. Extract Features + Parquet" in GUI
    assert "Standard (Recommended)" in GUI
    assert "Custom Training Instruction" in GUI
    assert "dataset_instruction_library" not in GUI
    standard = resolve_dataset_instruction("Standard (Recommended)", "")
    assert standard == "You are a helpful assistant.<|endofprompt|>"


def test_instruction_library_hides_internal_marker_from_editor():
    value = normalize_instruction("Speak warmly and slowly.")
    assert value.endswith("<|endofprompt|>")
    assert instruction_body(value) == "Speak warmly and slowly."
    assert "Do not type `<|endofprompt|>`" in GUI


def test_mode_specific_controls_are_dynamic():
    zero_prompt, zero_instruct = mode_visibility("Zero-shot")
    cross_prompt, cross_instruct = mode_visibility("Cross-lingual")
    instruct_prompt, instruct_panel = mode_visibility("Instruct")
    assert zero_prompt["visible"] is True and zero_instruct["visible"] is False
    assert cross_prompt["visible"] is False and cross_instruct["visible"] is False
    assert instruct_prompt["visible"] is False and instruct_panel["visible"] is True


def test_generation_and_dialogue_defaults_are_exposed_without_row_language_pause():
    for label in [
        "Top-K", "Top-P", "LLM Temperature", "RAS Window", "RAS Repetition Threshold",
        "Min Token / Text Ratio", "Max Token / Text Ratio", "Flow Steps", "Flow Temperature", "Text Frontend",
    ]:
        assert label in GUI
    assert 'dialogue_language = gr.Dropdown' in GUI
    assert 'dialogue_turn_silence = gr.Slider' in GUI
    assert 'dialogue_inputs.extend([row_mode, speaker, row_adapter, text, instruction])' in GUI
    assert 'label="Instruction Override (Optional)"' in GUI


def test_width_constrained_runtime_controls_are_present():
    assert 'label="LoRA Adapter"' in GUI and 'elem_classes="medium-control"' in GUI
    assert 'label="torch.compile Mode"' in GUI and 'elem_classes="small-control"' in GUI


def test_compile_mode_is_not_decorative():
    runtime = (ROOT / "cosyvoice_easy" / "runtime.py").read_text(encoding="utf-8")
    assert '{"default", "reduce-overhead", "max-autotune"}' in runtime
    assert 'torch.compile(model, mode=selected_mode)' in runtime
