from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from cosyvoice_easy.audio import CHUNK_CHOICES, safe_name, save_wav, split_long_text, split_text
from cosyvoice_easy.schemas import InferenceRequest, TrainingConfig
from cosyvoice_easy.storage import atomic_json, read_json
from cosyvoice_easy import voices
from cosyvoice_easy.runtime import condition_cross_lingual_text


def test_atomic_json_round_trip(tmp_path: Path):
    target = tmp_path / "state.json"
    atomic_json(target, {"text": "Español 日本語", "value": 4})
    assert read_json(target, {}) == {"text": "Español 日本語", "value": 4}
    assert not list(tmp_path.glob("*.tmp"))


def test_audio_and_safe_names(tmp_path: Path):
    output = tmp_path / "tone.wav"
    save_wav(output, 24000, np.zeros(2400, dtype=np.float32))
    with wave.open(str(output), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnframes() == 2400
    assert safe_name(" ../Voice:*? ") == "Voice"


def test_long_text_chunking_preserves_text():
    chunks = split_text("First sentence. Second sentence! Third sentence?", 20)
    assert len(chunks) >= 2
    assert "First sentence." in chunks[0]


def test_schema_defaults_are_cosyvoice3_safe():
    request = InferenceRequest(text="hello")
    assert "<|endofprompt|>" in request.instruction
    training = TrainingConfig("p", "m", "d", "o")
    assert training.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert training.steps % training.save_every_steps == 0
    assert training.epochs % training.save_every_epochs == 0


def test_cross_lingual_text_gets_required_cosyvoice3_prompt_marker():
    conditioned = condition_cross_lingual_text("Hola mundo")
    assert conditioned == "You are a helpful assistant.<|endofprompt|>Hola mundo"
    assert condition_cross_lingual_text(conditioned) == conditioned


def test_all_easy_gui_chunking_rules_are_exposed():
    assert CHUNK_CHOICES == ["None", "Paragraph/Sentence Auto", "Periods", "Paragraphs", "Lines", "Speaker turns"]
    assert split_long_text("A. B. C.", "Periods") == ["A.", "B.", "C."]
    assert split_long_text("A\nB", "Lines") == ["A", "B"]


def test_flat_voice_library_accepts_wav_with_txt_or_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(voices, "VOICES", tmp_path)
    (tmp_path / "txt_voice.wav").write_bytes(b"RIFF")
    (tmp_path / "txt_voice.txt").write_text("Transcript from text", encoding="utf-8")
    (tmp_path / "json_voice.wav").write_bytes(b"RIFF")
    atomic_json(tmp_path / "json_voice.json", {"Type": "Sample", "Text": "Transcript from JSON"})
    (tmp_path / "ignored.wav").write_bytes(b"RIFF")

    assert voices.list_voices() == ["json_voice", "txt_voice"]
    assert voices.load_voice("txt_voice").transcript == "Transcript from text"
    record = voices.load_voice("json_voice")
    assert record.audio == str(tmp_path / "json_voice.wav")
    assert record.transcript == "Transcript from JSON"
