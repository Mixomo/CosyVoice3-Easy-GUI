from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cosyvoice_easy import datasets
from cosyvoice_easy.audio import save_wav


def test_scan_and_prepare_mappings(tmp_path: Path, monkeypatch):
    source = tmp_path / "source" / "speaker"
    source.mkdir(parents=True)
    for index in range(3):
        save_wav(source / f"line{index}.wav", 24000, np.zeros(2400, dtype=np.float32))
        (source / f"line{index}.txt").write_text(f"Text {index}", encoding="utf-8")
    projects = tmp_path / "projects"
    prepared = tmp_path / "datasets"
    monkeypatch.setattr(datasets, "PROJECTS", projects)
    monkeypatch.setattr(datasets, "DATASETS", prepared)
    project, _ = datasets.create_project("Test Voice", str(source), "en")
    rows, _ = datasets.scan_source(str(source))
    manifest, _ = datasets.prepare_dataset(project, rows, "en", 33)
    records = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert {record["split"] for record in records} == {"train", "cv"}
    for split in ("train", "cv"):
        assert (prepared / project / split / "wav.scp").is_file()
        assert (prepared / project / split / "instruct").is_file()
