from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUI = (ROOT / "easy_gui.py").read_text(encoding="utf-8")
TRAINING = (ROOT / "cosyvoice_easy" / "training.py").read_text(encoding="utf-8")


def test_training_resume_surface_exists_and_defaults_to_none():
    assert 'label="Resume Checkpoint"' in GUI
    assert 'value=NONE' in GUI
    assert '### ↩️ Training Resume' in GUI
    assert 'train_resume_refresh' in GUI


def test_resume_is_real_trajectory_continuation_not_adapter_warm_start():
    assert 'command += ["--lora-checkpoint", config.resume]' in TRAINING


def test_resume_selection_restores_run_hyperparameters():
    assert 'load_resume_config_ui' in GUI
    for name in ['train_rank', 'train_alpha', 'train_dropout', 'train_lr', 'train_epochs', 'train_seed']:
        assert name in GUI
    assert 'loads adapter weights as a warm start' in GUI


def test_inference_adapter_discovery_includes_epoch_checkpoints():
    assert 'def adapter_dropdown_choices()' in TRAINING
    assert 'resume_epoch_' in TRAINING
    assert 'resume_step_' in TRAINING
    assert 'directory.name == "initial-adapter"' in TRAINING
    assert 'adapter_dropdown_choices()' in GUI
