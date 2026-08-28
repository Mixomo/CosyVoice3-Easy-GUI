from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .asr import transcribe
from .audio import prepare_audio, safe_name, wav_info
from .paths import DATASETS, PROJECTS, ROOT, TRAINING_OUTPUTS
from .instructions import normalize_instruction
from .storage import atomic_json

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def list_projects() -> list[str]:
    return sorted(p.name for p in PROJECTS.iterdir() if (p / "project.json").is_file())


def list_dataset_projects() -> list[str]:
    return [name for name in list_projects() if not json.loads((PROJECTS / name / "project.json").read_text(encoding="utf-8")).get("dataset_deleted", False)]


def list_training_projects() -> list[str]:
    return [name for name in list_projects() if not json.loads((PROJECTS / name / "project.json").read_text(encoding="utf-8")).get("training_deleted", False)]


def create_project(name: str, source_folder: str = "", language: str = "Auto") -> tuple[str, str]:
    project = safe_name(name, "voice-project")
    folder = PROJECTS / project
    if (folder / "project.json").is_file():
        raise ValueError(f"Project '{project}' already exists. Select it or use Clone Project.")
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {"name": project, "source_folder": source_folder, "language": language,
                "created_at": datetime.now(timezone.utc).isoformat(), "selected_manifest": ""}
    atomic_json(folder / "project.json", metadata)
    return project, f"Project '{project}' is ready."


def save_project(project: str, source_folder: str, language: str) -> str:
    name = safe_name(project)
    path = PROJECTS / name / "project.json"
    if not path.is_file():
        raise ValueError(f"Project '{name}' does not exist. Use Create Project first.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update({"source_folder": source_folder, "language": language, "updated_at": datetime.now(timezone.utc).isoformat()})
    atomic_json(path, metadata)
    return f"Project '{name}' saved."


def delete_project(project: str) -> str:
    name = safe_name(project)
    folder = (PROJECTS / name).resolve()
    if folder.parent != PROJECTS.resolve() or not folder.is_dir():
        return f"Project '{name}' does not exist."
    metadata = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    metadata["dataset_deleted"] = True
    atomic_json(folder / "project.json", metadata)
    dataset_folder = (DATASETS / name).resolve()
    if dataset_folder.parent == DATASETS.resolve() and dataset_folder.is_dir():
        shutil.rmtree(dataset_folder)
    return f"Dataset component '{name}' deleted. Training component was preserved."


def delete_training_project(project: str) -> str:
    """Delete only a project's LoRA runs/TensorBoard, never its dataset."""
    name = safe_name(project)
    root = (TRAINING_OUTPUTS / name).resolve()
    if root.parent != TRAINING_OUTPUTS.resolve() or not root.is_dir():
        message = f"No training outputs exist for '{name}'. Dataset was preserved."
    else:
        shutil.rmtree(root)
        message = f"Training outputs for '{name}' deleted. Dataset project was preserved."
    metadata_path = PROJECTS / name / "project.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["training_deleted"] = True
        atomic_json(metadata_path, metadata)
    return message


def clone_project(project: str, new_name: str) -> tuple[str, str]:
    source = (PROJECTS / safe_name(project)).resolve()
    target = (PROJECTS / safe_name(new_name)).resolve()
    if source.parent != PROJECTS.resolve() or target.parent != PROJECTS.resolve() or not (source / "project.json").is_file():
        raise ValueError("Select an existing project and enter a new project name.")
    if target.exists():
        raise ValueError(f"Project '{target.name}' already exists.")
    shutil.copytree(source, target)
    source_dataset = (DATASETS / source.name).resolve()
    target_dataset = (DATASETS / target.name).resolve()
    if source_dataset.is_dir():
        shutil.copytree(source_dataset, target_dataset)
    metadata = json.loads((target / "project.json").read_text(encoding="utf-8"))
    metadata["name"] = target.name
    metadata["dataset_deleted"] = False
    metadata["training_deleted"] = False
    if metadata.get("dataset_dir"):
        metadata["dataset_dir"] = str(target_dataset)
    if metadata.get("training"):
        metadata["training"]["project"] = target.name
        metadata["training"]["dataset_dir"] = str(target_dataset)
        metadata["training"]["output_dir"] = str(TRAINING_OUTPUTS / target.name)
    atomic_json(target / "project.json", metadata)
    return target.name, f"Project '{source.name}' cloned as '{target.name}' with prepared dataset assets."


def scan_source(source_folder: str) -> tuple[list[dict], str]:
    source = Path(source_folder)
    if not source.is_dir():
        raise ValueError("Select an existing source folder.")
    rows = []
    for audio in sorted(p for p in source.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
        transcript_path = audio.with_suffix(".txt")
        transcript = transcript_path.read_text(encoding="utf-8-sig").strip() if transcript_path.is_file() else ""
        try:
            info = wav_info(audio) if audio.suffix.lower() == ".wav" else {"duration": 0.0}
            error = ""
        except Exception as exc:
            info, error = {"duration": 0.0}, str(exc)
        rows.append({"audio": str(audio.resolve()), "transcript": transcript,
                     "speaker": safe_name(audio.parent.name, "speaker"), "duration": info["duration"], "error": error})
    missing = sum(not row["transcript"] for row in rows)
    return rows, f"Found {len(rows)} audio file(s); {missing} transcript(s) missing."


def transcribe_missing(rows: list[dict], model: str, language: str, batch_size: int = 1, on_item=None) -> tuple[list[dict], str]:
    """Fill missing transcripts, optionally reporting each completed audio item."""
    completed = 0
    pending = [row for row in rows if not row.get("transcript") and not row.get("error")]
    total = len(pending)
    for row in rows:
        if not row.get("transcript") and not row.get("error"):
            row["transcript"], detected = transcribe(row["audio"], model, language, batch_size=batch_size)
            completed += 1
            if on_item is not None:
                on_item(completed, total, row, detected)
    return rows, f"Transcribed {completed} missing item(s)."


def prepare_dataset(project: str, rows: list[dict], language: str, cv_percent: int,
                    instruction: str = "You are a helpful assistant.<|endofprompt|>", on_item=None) -> tuple[str, str]:
    if not project:
        raise ValueError("Select a dataset project.")
    instruction = normalize_instruction(instruction)
    valid = [row for row in rows if row.get("audio") and row.get("transcript") and not row.get("error")]
    if len(valid) < 2:
        raise ValueError("At least two valid audio/transcript pairs are required.")
    destination = DATASETS / safe_name(project)
    audio_dir = destination / "audio"
    destination.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    random.Random(0).shuffle(valid)
    cv_count = max(1, round(len(valid) * max(1, min(int(cv_percent), 50)) / 100))
    records = []
    for index, row in enumerate(valid):
        utterance = f"{safe_name(project)}_{index:06d}"
        output_audio = prepare_audio(row["audio"], audio_dir / f"{utterance}.wav")
        records.append({"id": utterance, "audio": output_audio, "text": row["transcript"].strip(),
                        "speaker": safe_name(row.get("speaker", project)), "language": language,
                        "instruct": instruction, "split": "cv" if index < cv_count else "train"})
        if on_item is not None:
            on_item(index + 1, len(valid), row, utterance)
    for split in ("train", "cv"):
        split_dir = destination / split
        split_dir.mkdir(exist_ok=True)
        selected = [record for record in records if record["split"] == split]
        mappings = {
            "wav.scp": [f"{r['id']} {r['audio']}" for r in selected],
            "text": [f"{r['id']} {r['text']}" for r in selected],
            "utt2spk": [f"{r['id']} {r['speaker']}" for r in selected],
            "instruct": [f"{r['id']} {r['instruct']}" for r in selected],
        }
        speakers: dict[str, list[str]] = {}
        for record in selected:
            speakers.setdefault(record["speaker"], []).append(record["id"])
        mappings["spk2utt"] = [f"{speaker} {' '.join(utts)}" for speaker, utts in speakers.items()]
        for filename, lines in mappings.items():
            (split_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = destination / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    project_file = PROJECTS / safe_name(project) / "project.json"
    metadata = json.loads(project_file.read_text(encoding="utf-8")) if project_file.is_file() else {"name": project}
    metadata.update({"selected_manifest": str(manifest), "dataset_dir": str(destination), "items": len(records),
                     "train_items": len(records) - cv_count, "cv_items": cv_count})
    atomic_json(project_file, metadata)
    return str(manifest), f"Prepared {len(records)} item(s): {len(records)-cv_count} train / {cv_count} CV. Run feature extraction next."


def extract_features(project: str, model_dir: str, on_stage=None) -> str:
    dataset = DATASETS / safe_name(project)
    model = Path(model_dir)
    if not dataset.is_dir() or not (model / "campplus.onnx").is_file():
        raise ValueError("Prepared dataset or complete model is missing.")
    stages = [(split, stage) for split in ("train", "cv") for stage in ("embeddings", "speech tokens", "parquet")]
    completed = 0
    for split in ("train", "cv"):
        source = dataset / split
        commands = [
            [sys.executable, str(ROOT / "tools" / "extract_embedding.py"), "--dir", str(source), "--onnx_path", str(model / "campplus.onnx")],
            [sys.executable, str(ROOT / "tools" / "extract_speech_token.py"), "--dir", str(source), "--onnx_path", str(model / "speech_tokenizer_v3.onnx")],
            [sys.executable, str(ROOT / "tools" / "make_parquet_list.py"), "--num_utts_per_parquet", "1000", "--num_processes", "1",
             "--src_dir", str(source), "--des_dir", str(source / "parquet")],
        ]
        (source / "parquet").mkdir(exist_ok=True)
        for stage, command in zip(("embeddings", "speech tokens", "parquet"), commands):
            if on_stage is not None:
                on_stage(completed, len(stages), split, stage, "starting")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            subprocess.run(command, cwd=ROOT, check=True, env=env, creationflags=flags)
            if stage == "parquet":
                data_list = source / "parquet" / "data.list"
                missing = [line.strip() for line in data_list.read_text(encoding="utf-8").splitlines()
                           if line.strip() and not Path(line.strip()).is_file()]
                if missing:
                    raise RuntimeError(f"Parquet extraction completed with missing artifact: {missing[0]}")
            completed += 1
            if on_stage is not None:
                on_stage(completed, len(stages), split, stage, "complete")
    return f"Features and parquet lists prepared for '{project}'."
