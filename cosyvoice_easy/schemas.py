from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class VoiceRecord:
    name: str
    audio: str
    transcript: str = ""
    language: str = "Auto"
    notes: str = ""
    created_at: str = ""

    def dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferenceRequest:
    text: str
    voice: str = ""
    reference_audio: str = ""
    prompt_text: str = ""
    instruction: str = "You are a helpful assistant.<|endofprompt|>"
    language: str = "Auto"
    mode: str = "Zero-shot"
    seed: int = 0
    speed: float = 1.0
    chunk_mode: str = "None"
    gap_seconds: float = 0.15
    text_frontend: bool = True
    top_k: int = 25
    top_p: float = 0.8
    temperature: float = 1.0
    ras_window: int = 10
    ras_repetition_threshold: float = 0.1
    min_token_text_ratio: float = 2.0
    max_token_text_ratio: float = 20.0
    flow_steps: int = 10
    flow_temperature: float = 1.0


@dataclass(slots=True)
class DialogueLine:
    speaker: str
    text: str
    instruction: str = ""
    mode: str = "Zero-shot"
    adapter: str = ""


@dataclass(slots=True)
class TrainingConfig:
    project: str
    model_dir: str
    dataset_dir: str
    output_dir: str
    model_variant: str = "Base"
    rank: int = 16
    alpha: int = 64
    dropout: float = 0.05
    learning_rate: float = 5e-5
    training_mode: str = "Steps"
    steps: int = 1500
    save_every_steps: int = 250
    epochs: int = 20
    save_every_epochs: int = 5
    patience: int = 3
    batch_size: int = 1
    grad_accumulation: int = 2
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    resume: str = ""
    eval_reference: str = ""
    eval_reference_text: str = ""
    eval_text: str = ""
    seed: int = 1234
    deterministic: bool = False
    guarded_checkpoints: bool = False
    resume_keep_last: int = 3
