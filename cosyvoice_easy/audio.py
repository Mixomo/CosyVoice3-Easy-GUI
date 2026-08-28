from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np


def safe_name(value: str, fallback: str = "item") -> str:
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", (value or "").strip()).strip(" ._")
    return value[:80] or fallback


def load_audio(path: str | Path, target_sr: int = 24000, mono: bool = True) -> tuple[int, np.ndarray]:
    import librosa

    audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite samples.")
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    return sr, audio


def save_wav(path: str | Path, sample_rate: int, audio: np.ndarray) -> str:
    import soundfile as sf

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(output, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return str(output)


def prepare_audio(source: str | Path, destination: str | Path, start: float = 0.0,
                  end: float = 0.0, normalize: bool = True, target_sr: int = 24000) -> str:
    sr, audio = load_audio(source, target_sr)
    first = max(0, int(float(start or 0) * sr))
    last = int(float(end or 0) * sr) if float(end or 0) > 0 else len(audio)
    audio = audio[first:max(first + 1, min(last, len(audio)))]
    if normalize:
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-8:
            audio = audio * (0.95 / peak)
    return save_wav(destination, sr, audio)


def concat_audio(items: list[tuple[int, np.ndarray]], gap_seconds: float = 0.0) -> tuple[int, np.ndarray]:
    if not items:
        raise ValueError("No audio was generated.")
    sr = items[0][0]
    merged: list[np.ndarray] = []
    for item_sr, audio in items:
        if item_sr != sr:
            import librosa
            audio = librosa.resample(np.asarray(audio), orig_sr=item_sr, target_sr=sr)
        merged.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if gap_seconds > 0:
            merged.append(np.zeros(int(sr * gap_seconds), dtype=np.float32))
    return sr, np.concatenate(merged[:-1] if gap_seconds > 0 else merged)


def wav_info(path: str | Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sr = handle.getframerate()
        return {"sample_rate": sr, "channels": handle.getnchannels(), "duration": frames / max(sr, 1), "frames": frames}


CHUNK_CHOICES = ["None", "Paragraph/Sentence Auto", "Periods", "Paragraphs", "Lines", "Speaker turns"]


def split_long_text(text: str, mode: str = "None") -> list[str]:
    """Easy-GUI long-form rules shared with Index/FireRed, without a character ceiling.

    CosyVoice3 still applies its own tokenizer/token-ratio limits inside each request.
    These rules only define explicit semantic boundaries chosen by the user.
    """
    text = (text or "").strip()
    if not text:
        return []
    mode = mode if mode in CHUNK_CHOICES else "None"
    if mode == "None":
        return [text]
    if mode == "Paragraphs":
        return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()] or [text]
    if mode == "Lines":
        return [line.strip() for line in text.splitlines() if line.strip()] or [text]
    if mode == "Periods":
        normalized = re.sub(r"\s+", " ", text)
        return [part.strip() for part in re.split(r"(?<=\.)\s+", normalized) if part.strip()] or [text]
    if mode == "Speaker turns":
        chunks: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if re.match(r"^\s*\[?SPEAKER\d+\]?", clean, flags=re.IGNORECASE) and current:
                chunks.append("\n".join(current).strip())
                current = [clean]
            else:
                current.append(clean)
        if current:
            chunks.append("\n".join(current).strip())
        return chunks or [text]

    # Paragraph/Sentence Auto: semantic boundaries only, never a max-character threshold.
    chunks: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()] or [text]
    for paragraph in paragraphs:
        normalized = re.sub(r"[ \t]+", " ", paragraph).strip()
        sentences = [part.strip() for part in re.split(r"(?<=[.!?…。！？;；])\s*", normalized) if part.strip()]
        chunks.extend(sentences or [normalized])
    return chunks or [text]


def split_text(text: str, max_chars: int) -> list[str]:
    """Backward-compatible wrapper retained for older project helpers/tests."""
    return split_long_text(text, "None") if int(max_chars or 0) <= 0 else split_long_text(text, "Paragraph/Sentence Auto")
