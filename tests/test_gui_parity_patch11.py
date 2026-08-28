from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import easy_gui
from cosyvoice_easy import datasets
from cosyvoice_easy.audio import save_wav

ROOT = Path(__file__).resolve().parents[1]
GUI = (ROOT / "easy_gui.py").read_text(encoding="utf-8")
TOKENIZER = (ROOT / "cosyvoice" / "tokenizer" / "tokenizer.py").read_text(encoding="utf-8")


def test_project_roundtrip_restores_dataset_training_and_eval_audio(tmp_path: Path, monkeypatch):
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(datasets, "PROJECTS", projects)
    monkeypatch.setattr(easy_gui, "PROJECTS", projects)

    project, _ = datasets.create_project("roundtrip", str(tmp_path / "source"), "es")
    eval_audio = tmp_path / "temporary-upload.wav"
    save_wav(eval_audio, 24000, np.zeros(1200, dtype=np.float32))
    rows = json.dumps([{"audio": "a.wav", "transcript": "hola", "speaker": "spk", "duration": 1.0, "error": ""}])

    easy_gui.save_training_project_ui(
        project, str(tmp_path / "source"), "es", True, "medium", "es", 7,
        "Custom", "Speak warmly.", 17, str(tmp_path / "manifest.jsonl"), rows,
        "RL", "≈ 180+ min / 4+ speakers · Higher capacity (r32)", easy_gui.NONE, 77, 4321, 32, 128, 0.12, 0.000123,
        str(eval_audio), "texto eval", "frase de evaluación", "es", "small", "es", 6,
    )

    metadata = json.loads((projects / project / "project.json").read_text(encoding="utf-8"))
    assert metadata["dataset_ui"]["whisper_batch_size"] == 7
    assert metadata["dataset_ui"]["instruction_mode"] == "Custom"
    assert metadata["dataset_ui"]["cv_percent"] == 17
    assert metadata["dataset_ui"]["analyzed_rows"][0]["transcript"] == "hola"
    assert metadata["training_ui"]["base_variant"] == "RL"
    assert metadata["training_ui"]["vram_preset"] == "≈ 180+ min / 4+ speakers · Higher capacity (r32)"
    assert metadata["training_ui"]["rank"] == 32
    assert metadata["training_ui"]["alpha"] == 128
    assert metadata["training_ui"]["eval"]["whisper_model"] == "small"
    stable_audio = Path(metadata["training_ui"]["eval"]["audio"])
    assert stable_audio.is_file()
    assert stable_audio.parent == projects / project / "assets"

    dataset_values = easy_gui.load_project_ui(project)
    assert dataset_values[1] == str(tmp_path / "source")
    assert dataset_values[2] == "es"
    assert dataset_values[3] is True
    assert dataset_values[4] == "medium"
    assert dataset_values[5] == "es"
    assert dataset_values[6] == 7
    assert dataset_values[7] == "Custom"
    assert dataset_values[8] == "Speak warmly."
    assert dataset_values[10] == 17
    assert json.loads(dataset_values[12])[0]["transcript"] == "hola"

    training_values = easy_gui.load_training_project_ui(project)
    assert Path(training_values[0]) == stable_audio
    assert training_values[1] == "texto eval"
    assert training_values[2] == "frase de evaluación"
    assert training_values[3] == "es"
    assert training_values[4] == "RL"
    assert training_values[5] == "≈ 180+ min / 4+ speakers · Higher capacity (r32)"
    assert training_values[7] == "Steps"
    assert training_values[8] == 1500
    assert training_values[9] == 77
    assert training_values[10] == 4321
    assert training_values[11] == 32
    assert training_values[12] == 128
    assert training_values[13] == 0.12
    assert training_values[14] == 0.000123
    assert training_values[15] == "small"
    assert training_values[16] == "es"
    assert training_values[17] == 6


def test_project_selection_from_either_tab_restores_both_surfaces():
    assert "dataset_project.change(\n            load_training_project_ui" in GUI
    assert "train_project.change(\n            load_project_ui" in GUI
    assert "save_training_project_ui, full_project_inputs" in GUI
    assert "save_training_project_ui, train_full_project_inputs" in GUI


def test_markup_guide_matches_finite_cosyvoice3_tokenizer_vocabulary():
    for token in [
        "[breath]", "[quick_breath]", "[noise]", "[laughter]", "[cough]", "[clucking]",
        "[accent]", "[hissing]", "[sigh]", "[vocalized-noise]", "[lipsmack]", "[mn]",
        "<strong>", "</strong>", "<laughter>", "</laughter>", "[AA1]", "[ZH]", "[j]", "[ǐ]",
    ]:
        assert token in TOKENIZER
        assert token in GUI
    assert "does **not** accept arbitrary bracket tags" in GUI
    assert "CMU / ARPAbet" in GUI
    assert "Pinyin" in GUI
