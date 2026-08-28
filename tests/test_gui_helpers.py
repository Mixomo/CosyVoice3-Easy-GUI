from __future__ import annotations

from cosyvoice_easy.console import html_view, log
from cosyvoice_easy.runtime import resolve_flash_attention
from cosyvoice_easy.training import start_tensorboard
from easy_gui import autotune_training_ui, build_ui, dialogue_row_action, dialogue_toolbar_action


def _values(count=2):
    rows = []
    for index in range(12):
        rows.extend(["Zero-shot", f"voice{index}", "None", f"line{index}", ""])
    return [count, *rows]


def test_dialogue_clone_and_delete_keep_fixed_state_shape():
    result = dialogue_row_action("clone", 0, *_values())
    assert result[0] == 3
    assert len(result) == 1 + 12 + 12 * 5
    result = dialogue_row_action("delete", 1, *result)
    assert result[0] == 2
    assert len(result) == 1 + 12 + 12 * 5


def test_dialogue_compact_removes_empty_lines():
    values = _values(3)
    values[1 + 1 * 5 + 3] = ""
    result = dialogue_toolbar_action("compact", 3, *values[1:])
    assert result[0] == 2


def test_dialogue_toolbar_reset_keeps_two_rows():
    result = dialogue_toolbar_action("reset", 5, *_values(5)[1:])
    assert result[0] == 2
    assert len(result) == 1 + 12 + 12 * 5


def test_console_html_escapes_log_content():
    log("<script>alert('x')</script>", "TEST")
    rendered = html_view()
    assert "&lt;script&gt;" in rendered
    assert "<script>alert('x')</script>" not in rendered


def test_autotune_returns_all_visible_training_controls():
    result = autotune_training_ui("", "Validated default (r16)", 4, 8, .2, 1e-4, 1)
    assert len(result) == 10
    assert result[1]["value"] % result[2]["value"] == 0
    assert "2,162,688" in result[-1]
    assert "6.96 GB" in result[-1]


def test_tensorboard_requires_project():
    assert "Select a training project" in start_tensorboard("")


def test_flash_attention_disabled_uses_official_sdpa_path():
    enabled, message = resolve_flash_attention(False)
    assert not enabled
    assert "SDPA" in message


def test_index_moss_workflow_tab_hierarchy_is_preserved():
    app = build_ui()
    labels = [getattr(component, "label", None) for component in app.blocks.values()
              if component.__class__.__name__ == "Tab"]
    assert labels == [
        "🎙️ Prep Samples", "🔊 Inference", "TTS / Voice Clone", "Dialogue Builder",
        "📂 Dataset Preparation", "🚀 LoRA Training",
    ]


def test_family_none_sentinel_and_language_instruction_contract():
    from cosyvoice_easy.ui_helpers import NONE, language_instruction, resolve_seed
    from easy_gui import adapter_values, voice_choices

    assert voice_choices()[0] == NONE
    assert adapter_values()[0] == NONE
    prompt = language_instruction("You are a calm narrator.<|endofprompt|>", "es")
    assert "Please speak in Spanish." in prompt
    assert prompt.endswith("<|endofprompt|>")
    assert resolve_seed(1234, False) == 1234


def test_manual_model_lifecycle_buttons_are_removed():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "easy_gui.py"
    text = source.read_text(encoding="utf-8")
    assert "Download / Repair Model" not in text
    assert 'gr.Button("Load Model"' not in text
    assert "🧹 Unload All Models" in text
    assert 'chunk_mode = gr.Dropdown(CHUNK_CHOICES, value="None", label="Chunking Rule"' in text


def test_launcher_uses_on_demand_model_download():
    from pathlib import Path

    launcher = (Path(__file__).resolve().parents[1] / "2- run.bat").read_text(encoding="utf-8")
    assert "tools\\ensure_model.py" not in launcher
    assert "easy_gui.py" in launcher


def test_project_selectors_start_empty_so_first_selection_loads_state():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "easy_gui.py").read_text(encoding="utf-8")
    assert 'dataset_project = gr.Dropdown(\n                        projects, value=None' in source
    assert 'train_project = gr.Dropdown(\n                        projects, value=None' in source
    assert '[NONE], value=NONE' in source
