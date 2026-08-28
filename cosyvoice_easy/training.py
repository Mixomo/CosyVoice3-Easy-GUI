from __future__ import annotations

import json
import os
import signal
import re
import time
import webbrowser
import subprocess
import socket
import sys
import threading
import atexit
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .paths import ROOT, TRAINING_OUTPUTS
from .schemas import TrainingConfig
from .storage import atomic_json
from .audio import safe_name

_LOCK = threading.RLock()
_PROCESS: subprocess.Popen | None = None
_LOG = ""
_TENSORBOARD: subprocess.Popen | None = None
_TENSORBOARD_LOGDIR: str | None = None
_TENSORBOARD_PORT: int | None = None
_STARTED_MONOTONIC: float | None = None
_ACTIVE_CONFIG: TrainingConfig | None = None
_LOG_THREAD: threading.Thread | None = None


def close_tensorboard() -> None:
    """Stop the app-owned TensorBoard child when the GUI process exits."""
    global _TENSORBOARD, _TENSORBOARD_LOGDIR, _TENSORBOARD_PORT
    process = _TENSORBOARD
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
    _TENSORBOARD = None
    _TENSORBOARD_LOGDIR = None
    _TENSORBOARD_PORT = None


atexit.register(close_tensorboard)


def _pump_training_output(process: subprocess.Popen, log_path: Path) -> None:
    """Tee the child stream to the persistent log, parent CMD and embedded console."""
    stream = process.stdout
    if stream is None:
        return
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        for line in iter(stream.readline, ""):
            # Some native Windows libraries emit UTF-16-ish console fragments
            # into an otherwise UTF-8 pipe. Remove the interleaved artifacts
            # before persisting/mirroring them.
            clean_line = line.replace("\x00", "").replace("\ufffd", "")
            handle.write(clean_line)
            handle.flush()
            # easy_gui installs a stdout mirror, so this single write reaches
            # both the parent terminal and the embedded Gradio console.
            sys.stdout.write(clean_line)
            sys.stdout.flush()
    stream.close()


def adapter_choices() -> list[str]:
    """Return every inference-ready trained adapter/checkpoint, not only final runs."""
    found: set[str] = set()
    for config_path in TRAINING_OUTPUTS.rglob("adapter_config.json"):
        directory = config_path.parent
        # initial-adapter is the epoch-zero seed artifact, not a trained checkpoint.
        if directory.name == "initial-adapter":
            continue
        if (directory / "adapter_model.safetensors").is_file():
            found.add(str(directory))
    return sorted(found, key=lambda value: value.lower())


def adapter_dropdown_choices() -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for raw in adapter_choices():
        path = Path(raw)
        try:
            rel = path.relative_to(TRAINING_OUTPUTS)
            parts = rel.parts
            if path.name.startswith("resume_epoch_"):
                epoch = int(path.name.rsplit("_", 1)[-1]) + 1
                label = f"{' / '.join(parts[:-1])} / epoch {epoch:06d}"
            elif path.name.startswith("resume_step_"):
                step = int(path.name.rsplit("_", 1)[-1])
                label = f"{' / '.join(parts[:-1])} / step {step:06d}"
            else:
                label = " / ".join(parts)
        except Exception:
            label = path.name
        choices.append((label, raw))
    return choices


def resume_checkpoint_choices(project: str | None = None, model_variant: str | None = None) -> list[tuple[str, str]]:
    """Return the newest resumable guarded checkpoint from every run for a project."""
    if not project:
        return []
    root = TRAINING_OUTPUTS / safe_name(str(project))
    if not root.is_dir():
        return []
    wanted = "RL" if model_variant == "RL" else "Base"
    choices: list[tuple[str, str]] = []
    # Current GUI runs store checkpoints directly under outputs/<project>;
    # retain support for the older nested outputs/<project>/<run> layout.
    direct = [p for p in root.iterdir() if p.is_dir() and (p / "adapter_config.json").is_file()]
    run_dirs = [root] if direct else [p for p in root.iterdir() if p.is_dir()]
    for run_dir in sorted(run_dirs, key=lambda p: p.name.lower()):
        metadata_path = run_dir / "adapter_metadata.json"
        variant = "Base"
        if metadata_path.is_file():
            try:
                variant = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("base_variant", "Base"))
            except Exception:
                pass
        if variant != wanted:
            continue
        checkpoints = [p for p in run_dir.iterdir() if p.is_dir() and (re.fullmatch(r"resume_(?:epoch|step)_\d{6}", p.name) or re.fullmatch(r"checkpoint-\d{6}", p.name) or p.name == "final_adapter") and (p / "adapter_config.json").is_file()]
        if not checkpoints:
            continue
        def checkpoint_step(path: Path) -> int:
            if path.name.startswith("checkpoint-"):
                return int(path.name.rsplit("-", 1)[-1])
            try:
                return int(json.loads((path / "training-state.json").read_text(encoding="utf-8")).get("completed_step", 0))
            except Exception:
                return 0
        checkpoints.sort(key=checkpoint_step)
        for checkpoint in checkpoints:
            step = checkpoint_step(checkpoint)
            if checkpoint.name == "final_adapter":
                label = f"{run_dir.name} / final_adapter"
            else:
                label = f"{run_dir.name} / step {step:06d}"
            choices.append((label, str(checkpoint)))
    return choices


def launch(config: TrainingConfig) -> str:
    global _PROCESS, _LOG, _LOG_THREAD, _STARTED_MONOTONIC, _ACTIVE_CONFIG
    with _LOCK:
        if _PROCESS is not None and _PROCESS.poll() is None:
            raise RuntimeError("A training process is already running.")
        dataset = Path(config.dataset_dir)
        model = Path(config.model_dir)
        train_list = dataset / "train" / "parquet" / "data.list"
        cv_list = dataset / "cv" / "parquet" / "data.list"
        if not train_list.is_file() or not cv_list.is_file():
            raise FileNotFoundError("Extract dataset features before starting LoRA training.")
        if not (model / "llm.pt").is_file():
            raise FileNotFoundError("The selected base model is incomplete.")
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "run.json", {**asdict(config), "base_model": str(model),
                    "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"})
        variant = "RL" if config.model_variant == "RL" else "Base"
        checkpoint = model / ("llm.rl.pt" if variant == "RL" else "llm.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"The selected {variant} training checkpoint is missing: {checkpoint.name}")
        atomic_json(output / "adapter_metadata.json", {"base_model": str(model), "base_variant": variant, "project": config.project,
                    "target_modules": config.target_modules, "rank": config.rank, "alpha": config.alpha,
                    "dropout": config.dropout, "format": "peft-cosyvoice3"})
        log_path = output / "training.log"
        command = [sys.executable, str(ROOT / "training" / "lora_tools" / "train_cosyvoice3_lora.py"),
                   "--train_engine", "single_gpu", "--model", "llm",
                   "--config", str(ROOT / "examples" / "libritts" / "cosyvoice3" / "conf" / "cosyvoice3.yaml"),
                   "--train_data", str(train_list), "--cv_data", str(cv_list),
                   "--qwen_pretrain_path", str(model / "CosyVoice-BlankEN"),
                   "--checkpoint", str(checkpoint), "--model_dir", str(output),
                   "--tensorboard_dir", str(output / "tensorboard"),
                   "--num_workers", "0", "--prefetch", "10", "--use_amp",
                   "--lora-r", str(config.rank), "--lora-alpha", str(config.alpha),
                   "--lora-dropout", str(config.dropout), "--lora-target-modules", ",".join(config.target_modules),
                   "--learning_rate", str(config.learning_rate), "--max_epoch", str(config.epochs),
                   "--max_steps", str(config.steps if config.training_mode == "Steps" else 0),
                   "--save-every-steps", str(config.save_every_steps if config.training_mode == "Steps" else 0),
                   "--save-every-epochs", str(config.save_every_epochs if config.training_mode == "Epochs" else 0),
                   "--accum-grad", str(config.grad_accumulation)]
        if config.eval_reference and config.eval_text:
            command += ["--eval-reference", config.eval_reference, "--eval-reference-text", config.eval_reference_text,
                        "--eval-text", config.eval_text]
        command += ["--seed", str(config.seed), "--resume-keep-last", str(config.resume_keep_last)]
        if config.deterministic:
            command.append("--deterministic")
        if config.resume:
            command += ["--lora-checkpoint", config.resume]
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "third_party" / "Matcha-TTS"),
                    "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
                    "TORCH_CPP_LOG_LEVEL": "ERROR", "RANK": "0", "LOCAL_RANK": "0",
                    "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1",
                    # Dataset processors import cosyvoice.utils.onnx before the
                    # trainer parses CLI arguments, so this must be inherited.
                    "onnx_path": str(model)})
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        _PROCESS = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=creationflags,
        )
        _LOG = str(log_path)
        _LOG_THREAD = threading.Thread(
            target=_pump_training_output, args=(_PROCESS, log_path),
            name="cosyvoice-training-log-tee", daemon=True,
        )
        _LOG_THREAD.start()
        _STARTED_MONOTONIC = time.monotonic()
        _ACTIVE_CONFIG = config
        return f"LoRA training started (PID {_PROCESS.pid})."


def stop() -> str:
    global _PROCESS
    with _LOCK:
        if _PROCESS is None or _PROCESS.poll() is not None:
            return "No training process is running."
        if os.name == "nt":
            _PROCESS.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            _PROCESS.terminate()
        return "Stop requested; the current checkpoint is preserved."


def status() -> tuple[str, str]:
    with _LOCK:
        if _PROCESS is None:
            state = "Idle."
        elif _PROCESS.poll() is None:
            state = f"Running (PID {_PROCESS.pid})."
        else:
            state = f"Finished with exit code {_PROCESS.returncode}."
        tail = ""
        if _LOG and Path(_LOG).is_file():
            lines = Path(_LOG).read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-80:])
        return state, tail



def progress_snapshot() -> dict[str, object]:
    """Return an epoch-level progress snapshot for the Easy GUI progress card."""
    with _LOCK:
        process = _PROCESS
        config = _ACTIVE_CONFIG
        log_path = Path(_LOG) if _LOG else None
        started = _STARTED_MONOTONIC

    steps_mode = config is not None and config.training_mode == "Steps"
    total = max(1, int(config.steps if steps_mode else config.epochs)) if config is not None else 0
    latest_epoch = -1
    completed_epoch = -1
    latest_step = 0
    latest_loss = None
    if log_path is not None and log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for match in re.finditer(r"Epoch\s+(\d+)\s+TRAIN info", text):
            latest_epoch = max(latest_epoch, int(match.group(1)))
        for match in re.finditer(r"Epoch\s+(\d+)\s+Step\s+(\d+)\s+CV info[^\n]*", text):
            completed_epoch = max(completed_epoch, int(match.group(1)))
            latest_step = max(latest_step, int(match.group(2)))
            line = match.group(0)
            loss_match = re.search(r"(?:loss|llm_loss)\s+([0-9.eE+-]+)", line)
            if loss_match:
                try:
                    latest_loss = float(loss_match.group(1))
                except ValueError:
                    pass
        for match in re.finditer(r"resume_epoch_(\d{6})", text):
            completed_epoch = max(completed_epoch, int(match.group(1)))
        for match in re.finditer(r"resume_step_(\d{6})", text):
            latest_step = max(latest_step, int(match.group(1)))
        for match in re.finditer(r"Optimizer step\s+(\d+)", text):
            latest_step = max(latest_step, int(match.group(1)))

    running = process is not None and process.poll() is None
    exit_code = None if process is None or running else process.returncode
    completed = latest_step if steps_mode else max(0, completed_epoch + 1)
    if total:
        completed = min(completed, total)
    pct = (completed / total * 100.0) if total else 0.0
    target_reached = completed >= total if total else False
    if not running and exit_code == 0 and total and target_reached:
        pct = 100.0
        completed = total
    elapsed = max(0.0, time.monotonic() - started) if started is not None else 0.0
    eta = None
    if running and completed > 0 and total > completed:
        eta = elapsed / completed * (total - completed)
    return {
        "running": running,
        "exit_code": exit_code,
        "current_epoch": max(0, latest_epoch + 1),
        "completed_epochs": completed,
        "total_epochs": total,
        "step": latest_step,
        "unit": "Steps" if steps_mode else "Epochs",
        "loss": latest_loss,
        "pct": pct,
        "elapsed": elapsed,
        "eta": eta,
    }


def start_tensorboard(project: str) -> str:
    global _TENSORBOARD, _TENSORBOARD_LOGDIR, _TENSORBOARD_PORT
    if not project:
        return "Select a training project before opening TensorBoard."
    project = safe_name(str(project))
    path = TRAINING_OUTPUTS / project
    path.mkdir(parents=True, exist_ok=True)
    port = 6006
    url = "http://127.0.0.1:6006"
    if _TENSORBOARD is not None and _TENSORBOARD.poll() is None:
        if _TENSORBOARD_LOGDIR != str(path.resolve()):
            _TENSORBOARD.terminate()
            _TENSORBOARD.wait(timeout=5)
            _TENSORBOARD = None
        else:
            webbrowser.open(url)
            return f"TensorBoard is already running (PID {_TENSORBOARD.pid}) at {url}."
    _free_tensorboard_port(port)
    if not _port_available(port):
        port = _next_tensorboard_port(port + 1)
        url = f"http://127.0.0.1:{port}"
    # Give the run an explicit project label so TensorBoard cannot reuse the
    # generic "tensorboard" name/cache from a previously selected project.
    command = [sys.executable, "-m", "tensorboard.main", "--logdir_spec", f"{project}:{path}", "--port", str(port), "--host", "127.0.0.1", "--reload_interval", "1"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    _TENSORBOARD = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    _TENSORBOARD_LOGDIR = str(path.resolve())
    _TENSORBOARD_PORT = port
    time.sleep(1.5)
    if _TENSORBOARD.poll() is not None:
        # A TensorBoard left by an older GUI may own 6006 and be impossible to
        # terminate due to Windows process permissions. Never report that old
        # server as the selected project; use the next local port instead.
        port = _next_tensorboard_port(port + 1)
        url = f"http://127.0.0.1:{port}"
        command = [sys.executable, "-m", "tensorboard.main", "--logdir_spec", f"{project}:{path}", "--port", str(port), "--host", "127.0.0.1", "--reload_interval", "1"]
        _TENSORBOARD = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        _TENSORBOARD_PORT = port
        time.sleep(0.35)
    webbrowser.open(url)
    return f"TensorBoard started at {url} (PID {_TENSORBOARD.pid})."


def prepare_tensorboard_project(project: str) -> None:
    """Select a project's logdir without starting TensorBoard."""
    global _TENSORBOARD, _TENSORBOARD_LOGDIR
    if not project:
        return
    path = (TRAINING_OUTPUTS / safe_name(str(project))).resolve()
    if _TENSORBOARD is not None and _TENSORBOARD.poll() is None and _TENSORBOARD_LOGDIR != str(path):
        _TENSORBOARD.terminate()
        try:
            _TENSORBOARD.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _TENSORBOARD.kill()
    _TENSORBOARD = None
    _TENSORBOARD_LOGDIR = str(path)


def _free_tensorboard_port(port: int = 6006) -> None:
    """Stop a stale TensorBoard left by a previous GUI process."""
    if os.name != "nt":
        return
    try:
        output = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"], text=True,
            encoding="mbcs", errors="replace",
        )
        pids = set()
        for line in output.splitlines():
            fields = line.split()
            if (len(fields) >= 5 and fields[0].upper() == "TCP"
                    and fields[1].endswith(f":{port}")
                    and fields[3].upper() == "LISTENING"):
                pids.add(fields[4])
        for pid in pids:
            subprocess.run(["taskkill", "/PID", pid, "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _next_tensorboard_port(start: int = 6006) -> int:
    port = start
    while not _port_available(port):
        port += 1
    return port


def restart_tensorboard(project: str) -> str:
    """Restart the app-owned TensorBoard viewer for a different project.

    The reference GUIs expose both *Open* and *Reload* actions.  Keeping the
    process handle here makes reload deterministic and prevents a stale logdir
    from being shown after switching projects.
    """
    global _TENSORBOARD, _TENSORBOARD_LOGDIR
    if _TENSORBOARD is not None and _TENSORBOARD.poll() is None:
        if os.name == "nt":
            _TENSORBOARD.terminate()
        else:
            _TENSORBOARD.terminate()
        _TENSORBOARD_LOGDIR = None
        try:
            _TENSORBOARD.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _TENSORBOARD.kill()
    _TENSORBOARD = None
    return start_tensorboard(project)
