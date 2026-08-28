from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "uv.lock"
PROFILE = ROOT / "config" / "runtime_windows_cuda128.json"

EXPECTED = {
    "torch": "2.8.0+cu128",
    "torchaudio": "2.8.0+cu128",
    "onnxruntime-gpu": "1.26.0",
    "onnxruntime": "1.26.0",
    "ctranslate2": "4.8.1",
    "triton-windows": "3.4.0.post21",
}


def _load_lock() -> dict:
    if not LOCK.is_file():
        raise RuntimeError("uv.lock is missing")
    with LOCK.open("rb") as fh:
        return tomllib.load(fh)


def _packages(lock: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for package in lock.get("package", []):
        result.setdefault(str(package.get("name", "")).lower(), []).append(package)
    return result



def _canonical_and_marker(marker: str) -> tuple[str, ...]:
    """Canonicalize the simple conjunction emitted by uv for the Windows lock."""
    return tuple(sorted(part.strip() for part in marker.split(" and ") if part.strip()))


def _contains_equivalent_marker(markers: object, expected: str) -> bool:
    if not isinstance(markers, list):
        return False
    expected_parts = _canonical_and_marker(expected)
    return any(
        isinstance(marker, str) and _canonical_and_marker(marker) == expected_parts
        for marker in markers
    )

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen Windows CUDA 12.8 uv lock contract.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    report: dict[str, object] = {}
    try:
        lock = _load_lock()
        packages = _packages(lock)
        report["requires_python"] = lock.get("requires-python")

        expected_env = "sys_platform == 'win32' and platform_machine == 'AMD64'"
        marker_fields = {
            key: lock.get(key)
            for key in ("resolution-markers", "supported-markers", "required-markers")
            if lock.get(key) is not None
        }
        report["lock_markers"] = marker_fields
        marker_blob = json.dumps(marker_fields, ensure_ascii=False)
        if "sys_platform != 'win32'" in marker_blob:
            errors.append("uv.lock: non-Windows resolution fork is present")
        supported = lock.get("supported-markers")
        if isinstance(supported, list) and not _contains_equivalent_marker(supported, expected_env):
            errors.append(f"uv.lock: expected semantically equivalent supported marker {expected_env!r}, found {supported!r}")
        required = lock.get("required-markers")
        if isinstance(required, list) and not _contains_equivalent_marker(required, expected_env):
            errors.append(f"uv.lock: expected semantically equivalent required marker {expected_env!r}, found {required!r}")

        for name, expected in EXPECTED.items():
            candidates = packages.get(name, [])
            versions = sorted({str(item.get("version", "")) for item in candidates})
            report[name] = versions
            if expected not in versions:
                errors.append(f"{name}: expected {expected}, found {versions or ['missing']}")

        # The onnxruntime dependency must be our metadata-only local shim, never
        # the CPU registry distribution that collides with onnxruntime-gpu.
        ort_entries = packages.get("onnxruntime", [])
        if ort_entries:
            if not any(
                isinstance(item.get("source"), dict)
                and any(key in item["source"] for key in ("directory", "editable", "virtual", "path"))
                for item in ort_entries
            ):
                errors.append("onnxruntime: expected the local GPU compatibility shim, found registry package")
        else:
            errors.append("onnxruntime: local GPU compatibility shim missing")

        torch_entries = packages.get("torch", [])
        if not any("cu128" in json.dumps(item.get("source", {})) for item in torch_entries):
            errors.append("torch: CUDA 12.8 PyTorch index source missing")

        with PROFILE.open("r", encoding="utf-8") as fh:
            profile = json.load(fh)
        report["lock_profile"] = profile.get("lock_profile")
        report["lock_cutoff"] = profile.get("lock_cutoff")
        if profile.get("lock_profile") != 3:
            errors.append("runtime profile: expected lock_profile=3")
    except Exception as exc:
        errors.append(str(exc))

    report["errors"] = errors
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif errors:
        for error in errors:
            print(f"[LOCK ERROR] {error}")
    else:
        print("Frozen Windows CUDA 12.8 uv.lock contract: OK")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
