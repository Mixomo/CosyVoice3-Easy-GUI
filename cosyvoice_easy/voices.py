from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from .audio import prepare_audio, safe_name
from .paths import VOICES
from .schemas import VoiceRecord
from .storage import atomic_json, read_json


def list_voices() -> list[str]:
    names = {p.name for p in VOICES.iterdir() if p.is_dir() and (p / "voice.json").is_file()}
    names.update(p.stem for p in VOICES.glob("*.wav") if _flat_transcript_path(p) is not None)
    return sorted(names, key=str.casefold)


def _flat_transcript_path(audio: Path) -> Path | None:
    """Return the existing TXT/JSON sidecar for a flat WAV voice."""
    txt = audio.with_suffix(".txt")
    if txt.is_file():
        return txt
    metadata = audio.with_suffix(".json")
    return metadata if metadata.is_file() else None


def _flat_transcript(sidecar: Path) -> str:
    if sidecar.suffix.lower() == ".txt":
        return sidecar.read_text(encoding="utf-8-sig", errors="replace").strip()
    data = read_json(sidecar, {})
    if not isinstance(data, dict):
        return ""
    # Existing voice packs use `Text`; native records use `transcript`.
    for key in ("transcript", "Transcript", "text", "Text", "prompt_text", "prompt"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_voice(name: str) -> VoiceRecord | None:
    if not name:
        return None
    native_record = VOICES / name / "voice.json"
    data = read_json(native_record, None)
    if isinstance(data, dict):
        allowed = VoiceRecord.__dataclass_fields__.keys()
        return VoiceRecord(**{key: data.get(key, "") for key in allowed})

    audio = VOICES / f"{name}.wav"
    sidecar = _flat_transcript_path(audio)
    if not audio.is_file() or sidecar is None:
        return None
    metadata = read_json(sidecar, {}) if sidecar.suffix.lower() == ".json" else {}
    language = metadata.get("language", metadata.get("Language", "Auto")) if isinstance(metadata, dict) else "Auto"
    notes = metadata.get("notes", metadata.get("Notes", "")) if isinstance(metadata, dict) else ""
    return VoiceRecord(name, str(audio), _flat_transcript(sidecar), str(language or "Auto"), str(notes or ""), "")


def save_voice(name: str, audio_path: str, transcript: str, language: str, notes: str,
               start: float = 0.0, end: float = 0.0, normalize: bool = True) -> tuple[str, str]:
    if not audio_path:
        raise ValueError("Select or record reference audio first.")
    slug = safe_name(name, Path(audio_path).stem)
    folder = VOICES / slug
    folder.mkdir(parents=True, exist_ok=True)
    prepared = prepare_audio(audio_path, folder / "reference.wav", start, end, normalize)
    record = VoiceRecord(slug, prepared, transcript.strip(), language, notes.strip(), datetime.now(timezone.utc).isoformat())
    atomic_json(folder / "voice.json", record.dict())
    return slug, prepared


def delete_voice(name: str) -> None:
    target = (VOICES / safe_name(name)).resolve()
    root = VOICES.resolve()
    if target.parent != root or not target.exists():
        raise ValueError("Voice does not exist.")
    shutil.rmtree(target)
