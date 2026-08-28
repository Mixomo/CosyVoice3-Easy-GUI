from __future__ import annotations

from datetime import datetime, timezone

from .audio import safe_name
from .paths import CONFIG
from .storage import atomic_json, read_json

LIBRARY = CONFIG / "instruction_library.json"
MARKER = "<|endofprompt|>"


def normalize_instruction(text: str) -> str:
    """Return a CosyVoice3 instruction prefix with exactly one end-of-prompt marker."""
    value = (text or "").strip()
    if MARKER in value:
        value = value.replace(MARKER, " ").strip()
    value = " ".join(value.split())
    if not value:
        value = "You are a helpful assistant."
    return f"{value}{MARKER}"


def instruction_body(text: str) -> str:
    return (text or "").replace(MARKER, " ").strip()


def _records() -> dict[str, dict[str, str]]:
    data = read_json(LIBRARY, {})
    return data if isinstance(data, dict) else {}


def list_instructions() -> list[str]:
    return sorted(_records(), key=str.casefold)


def load_instruction(name: str) -> dict[str, str] | None:
    if not name:
        return None
    record = _records().get(name)
    return record if isinstance(record, dict) else None


def save_instruction(name: str, text: str, notes: str = "") -> tuple[str, str]:
    body = instruction_body(text)
    if not body:
        raise ValueError("Enter a natural-language instruction before saving it.")
    key = safe_name(name, body[:48])
    records = _records()
    records[key] = {
        "name": key,
        "text": normalize_instruction(body),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(LIBRARY, records)
    return key, f"Saved instruction '{key}'."


def delete_instruction(name: str) -> str:
    records = _records()
    if name not in records:
        return "Select a saved instruction to delete."
    del records[name]
    atomic_json(LIBRARY, records)
    return f"Deleted instruction '{name}'."
