from pathlib import Path
import json
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_owns_cuda_core():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    assert "ctranslate2==4.8.1" in deps
    assert "onnxruntime-gpu==1.26.0" in deps
    assert "torch==2.8.0+cu128; sys_platform == 'win32' and platform_machine == 'AMD64'" in deps
    assert "torchaudio==2.8.0+cu128; sys_platform == 'win32' and platform_machine == 'AMD64'" in deps
    assert any(item.startswith("triton-windows==3.4.0.post21") for item in deps)
    assert data["tool"]["uv"]["environments"] == ["sys_platform == 'win32' and platform_machine == 'AMD64'"]
    assert data["tool"]["uv"]["required-environments"] == ["sys_platform == 'win32' and platform_machine == 'AMD64'"]
    assert data["tool"]["uv"]["sources"]["onnxruntime"]["path"] == "vendor/onnxruntime-gpu-shim"
    assert data["tool"]["uv"]["index"][0]["url"].endswith("/cu128")


def test_onnxruntime_shim_is_metadata_only():
    shim = tomllib.loads((ROOT / "vendor" / "onnxruntime-gpu-shim" / "pyproject.toml").read_text(encoding="utf-8"))
    assert shim["project"]["name"] == "onnxruntime"
    assert shim["project"]["version"] == "1.26.0"
    assert shim["project"]["dependencies"] == ["onnxruntime-gpu==1.26.0"]
    assert shim["tool"]["setuptools"]["packages"] == []


def test_installer_regenerates_stale_lock_cleanly():
    text = (ROOT / "1- install.bat").read_text(encoding="utf-8")
    low = text.lower()
    assert 'set "uv_version=0.11.33"' in low
    assert "lock --check" in low
    assert 'del /f /q "uv.lock"' in low
    assert "--exclude-newer" in low
    assert "sync --frozen" in low
    assert "verify_lock_contract.py" in low
    assert "backup" not in low
    assert "migrat" not in low
    assert "--no-install-package torch" not in low


def test_runtime_profile_declares_lock_policy_v3():
    profile = json.loads((ROOT / "config" / "runtime_windows_cuda128.json").read_text(encoding="utf-8"))
    assert profile["lock_profile"] == 3
    assert profile["lock_cutoff"] == "2026-08-23T19:00:00Z"
    assert profile["packages"]["torch"] == "2.8.0+cu128"
    assert "metadata-only" in profile["onnxruntime_dependency_policy"].lower()
    assert profile["uv_version"] == "0.11.33"
    assert profile["supported_environments"] == ["sys_platform == 'win32' and platform_machine == 'AMD64'"]


def test_clean_install_validates_lock_with_managed_python_before_venv_exists():
    bat = (ROOT / "1- install.bat").read_text(encoding="utf-8")
    assert "python find %PYTHON_VERSION%" in bat
    assert "platform.machine() == 'AMD64'" in bat
    assert 'set "LOCK_PY_EXE=' in bat
    assert '"%LOCK_PY_EXE%" tools\\verify_lock_contract.py' in bat
    pre_sync = bat.split('echo [5/8] Synchronizing', 1)[0]
    assert '"%PY_EXE%" tools\\verify_lock_contract.py' not in pre_sync


def test_lock_verifier_rejects_old_universal_forks():
    text = (ROOT / "tools" / "verify_lock_contract.py").read_text(encoding="utf-8")
    assert "resolution-markers" in text
    assert "supported-markers" in text
    assert "required-markers" in text
    assert "sys_platform != 'win32'" in text
    assert "platform_machine == 'AMD64'" in text


def test_lock_verifier_accepts_uv_canonical_marker_order():
    text = (ROOT / "tools" / "verify_lock_contract.py").read_text(encoding="utf-8")
    assert "_canonical_and_marker" in text
    assert "_contains_equivalent_marker" in text
    # uv 0.11.33 canonicalizes the conjunction in this order. The verifier
    # must compare semantics, not literal token order.
    ns = {"__file__": str(ROOT / "tools" / "verify_lock_contract.py"), "__name__": "lock_verifier_test"}
    exec(compile(text, str(ROOT / "tools" / "verify_lock_contract.py"), "exec"), ns)
    expected = "sys_platform == 'win32' and platform_machine == 'AMD64'"
    actual = ["platform_machine == 'AMD64' and sys_platform == 'win32'"]
    assert ns["_contains_equivalent_marker"](actual, expected)
