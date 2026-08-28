from pathlib import Path

from cosyvoice_easy.runtime import model_complete


def test_model_complete_rejects_partial_directory(tmp_path: Path):
    (tmp_path / "cosyvoice3.yaml").write_text("x", encoding="utf-8")
    assert model_complete(tmp_path) is False


def test_rl_variant_requires_rl_checkpoint(tmp_path: Path):
    for name in ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "CosyVoice-BlankEN").mkdir()
    assert model_complete(tmp_path, "Base") is True
    assert model_complete(tmp_path, "RL") is False
    (tmp_path / "llm.rl.pt").write_bytes(b"x")
    assert model_complete(tmp_path, "RL") is True


def test_default_model_load_path_contains_on_demand_repair():
    import inspect
    from cosyvoice_easy.runtime import EngineManager, ensure_model_available

    load_source = inspect.getsource(EngineManager.load)
    ensure_source = inspect.getsource(ensure_model_available)
    assert "ensure_model_available(model_dir, variant)" in load_source
    assert "download_model(selected)" in ensure_source
    assert "downloading/repairing on demand" in ensure_source
