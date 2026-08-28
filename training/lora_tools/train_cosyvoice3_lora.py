#!/usr/bin/env python3
from __future__ import print_function

import argparse
import datetime
import inspect
import math
import logging
import os
import platform
import random
import subprocess
import sys
import warnings
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=r"In 2\.9, this function's implementation will be changed to use torchaudio\.load_with_torchcodec.*", category=UserWarning)

try:
    import deepspeed
except ImportError:
    deepspeed = None
import numpy as np
import torch
import torch.distributed as dist
import yaml
from hyperpyyaml import load_hyperpyyaml
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from torch.distributed.elastic.multiprocessing.errors import record


def distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

from cosyvoice_resume_contract import (
    INITIAL_ADAPTER_NAME,
    OPTIMIZER_STATE_NAME,
    ResumeContractError,
    SCHEDULER_STATE_NAME,
    acquire_output_lock,
    build_contract,
    normalize_adapter_config,
    prune_owned_checkpoints,
    publish_checkpoint,
    publish_initial_adapter,
    require_fresh_output,
    validate_checkpoint,
)

from cosyvoice.utils.executor import Executor
from cosyvoice.utils.scheduler import (
    WarmupLR,
    WarmupPolicy,
    NoamHoldAnnealing,
    ConstantLR,
)
from cosyvoice.utils.train_utils import (
    check_modify_and_save_config,
    init_dataset_and_dataloader,
    init_distributed,
    init_summarywriter,
    wrap_cuda_model,
)

try:
    from peft import LoraConfig, PeftModel, get_peft_model

    try:
        from peft import TaskType
    except ImportError:  # pragma: no cover
        TaskType = None
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "peft is required for LoRA training. Install with: pip install peft"
    ) from exc


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def data_list_files(path: str) -> list[Path]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ResumeContractError(f"Data list must not be a symlink: {source}")
    source = source.resolve(strict=True)
    files = [source]
    for line_number, raw in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        artifact = Path(value).expanduser()
        if not artifact.is_absolute():
            artifact = source.parent / artifact
        if artifact.is_symlink():
            raise ResumeContractError(f"{source}:{line_number}: artifact is a symlink")
        artifact = artifact.resolve(strict=True)
        if not artifact.is_file():
            raise ResumeContractError(f"{source}:{line_number}: artifact is not a file")
        files.append(artifact)
    if len(files) == 1:
        raise ResumeContractError(f"Data list has no artifacts: {source}")
    return files


def guarded_contract(args, configs, initial_adapter: Path) -> dict:
    source_root = Path(__file__).resolve().parent
    train_conf = configs["train_conf"]
    training_config = {
        "max_epoch": int(train_conf["max_epoch"]),
        "max_steps": max(0, int(args.max_steps or 0)),
        "save_every_steps": max(0, int(args.save_every_steps or 0)),
        "save_every_epochs": max(1, int(args.save_every_epochs or 1)),
        "train_engine": args.train_engine,
        "model": args.model,
        "num_workers": args.num_workers,
        "prefetch": args.prefetch,
        "pin_memory": args.pin_memory,
        "use_amp": args.use_amp,
        "early_stop_on_cv_overfit": False,
        "resume_keep_last": args.resume_keep_last,
        "timeout": args.timeout,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if args.deterministic
            else None
        ),
        "optim": train_conf.get("optim"),
        "optim_conf": train_conf.get("optim_conf"),
        "scheduler": train_conf.get("scheduler"),
        "scheduler_conf": train_conf.get("scheduler_conf"),
        "accum_grad": train_conf.get("accum_grad"),
        "grad_clip": train_conf.get("grad_clip"),
        "cv_monitor": train_conf.get("cv_monitor"),
        "cv_higher_is_better": train_conf.get("cv_higher_is_better"),
        "cv_min_delta": train_conf.get("cv_min_delta"),
        "cv_patience": train_conf.get("cv_patience"),
        "cv_warmup": train_conf.get("cv_warmup"),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_modules": parse_list(args.lora_target_modules),
            "modules_to_save": parse_list(args.lora_modules_to_save),
            "unfreeze": parse_list(args.lora_unfreeze),
        },
    }
    runtime = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "deepspeed": package_version("deepspeed"),
        "peft": package_version("peft"),
        "transformers": package_version("transformers"),
        "cuda": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "world_size": distributed_world_size(),
    }
    return build_contract(
        output_dir=args.model_dir,
        base_checkpoint=args.checkpoint,
        config_file=args.config,
        qwen_pretrain=args.qwen_pretrain_path or None,
        data_files={
            "train": data_list_files(args.train_data),
            "cross_validation": data_list_files(args.cv_data),
        },
        source_files=(
            Path(__file__),
            source_root / "cosyvoice_resume_contract.py",
            source_root / "distributed_early_stop.py",
            Path(inspect.getfile(Executor)),
            Path(inspect.getfile(init_dataset_and_dataloader)),
        ),
        training_config=training_config,
        runtime=runtime,
        initial_adapter=initial_adapter,
    )


def monitor_state(info_dict: dict) -> dict:
    keys = (
        "best_cv_epoch",
        "cv_higher_is_better",
        "cv_metric",
        "cv_min_delta",
        "cv_monitor",
        "cv_monitor_used",
        "cv_no_improve_epochs",
        "cv_overfit_flag",
        "cv_patience",
        "cv_warmup",
    )
    state = {key: info_dict[key] for key in keys if key in info_dict}
    for key, value in info_dict.items():
        if key.startswith("best_cv_"):
            state[key] = value
    return state


def canonicalize_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).copy()
    if isinstance(value, dict):
        return {key: canonicalize_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(canonicalize_state(item) for item in value)
    return deepcopy(value)


def save_runtime_state(path: Path, optimizer, scheduler, scaler) -> None:
    state = {
        "optimizer": canonicalize_state(optimizer.state_dict()),
        "scheduler": canonicalize_state(scheduler.state_dict()),
        "scaler": (
            canonicalize_state(scaler.state_dict()) if scaler is not None else None
        ),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    outputs = {
        path: state,
        path.parent / OPTIMIZER_STATE_NAME: state["optimizer"],
        path.parent / SCHEDULER_STATE_NAME: state["scheduler"],
    }
    for output, value in outputs.items():
        torch.save(value, output)
        with output.open("rb") as handle:
            os.fsync(handle.fileno())


def configure_reproducibility(args) -> None:
    if args.seed < 0 or args.seed > 2**63 - 1:
        raise ValueError("--seed must be between 0 and 2^63 - 1")
    if args.deterministic:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {None, ":4096:8"}:
            raise ValueError(
                "Deterministic mode requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def restore_runtime_state(path: Path, optimizer, scheduler, scaler) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "optimizer",
        "scheduler",
        "scaler",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ResumeContractError("Guarded runtime state is incomplete")
    if scaler is not None and state["scaler"] is None:
        raise ResumeContractError("Guarded checkpoint omits the configured AMP scaler")
    if scaler is None and state["scaler"] is not None:
        raise ResumeContractError(
            "Guarded checkpoint contains unexpected AMP scaler state"
        )
    if (
        torch.cuda.is_available()
        and len(state["cuda_rng"]) != torch.cuda.device_count()
    ):
        raise ResumeContractError(
            "Guarded checkpoint CUDA RNG topology does not match runtime"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def get_args():
    parser = argparse.ArgumentParser(
        description="CosyVoice3 LoRA fine-tuning (LLM only)."
    )
    parser.add_argument(
        "--train_engine",
        default="single_gpu",
        choices=["single_gpu"],
        help="Native single-GPU training engine",
    )
    parser.add_argument(
        "--model", required=True, help="model which will be trained (use llm)"
    )
    parser.add_argument("--ref_model", required=False, help="ref model used in dpo")
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--train_data", required=True, help="train data file")
    parser.add_argument("--cv_data", required=True, help="cv data file")
    parser.add_argument(
        "--qwen_pretrain_path", required=False, help="qwen pretrain path"
    )
    parser.add_argument("--checkpoint", help="checkpoint model (full model weights)")
    parser.add_argument(
        "--model_dir", required=True, help="save LoRA checkpoints to this dir"
    )
    parser.add_argument(
        "--tensorboard_dir", default="tensorboard", help="tensorboard log dir"
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Explicitly override train_conf.max_epoch after loading the full model config",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Explicitly override train_conf.optim_conf.lr after loading the full model config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Training seed bound into guarded continuation state",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Enable strict deterministic CUDA controls for byte-exact evidence",
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="num of subprocess workers for reading",
    )
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Use pinned memory buffers",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=False,
        help="Use automatic mixed precision training",
    )
    parser.add_argument(
        "--dpo",
        action="store_true",
        default=False,
        help="Use Direct Preference Optimization",
    )
    parser.add_argument(
        "--timeout", default=60, type=int, help="timeout (in seconds) of cosyvoice_join"
    )
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many optimizer updates; 0 uses max_epoch")
    parser.add_argument("--save-every-steps", type=int, default=0, help="Publish a resumable checkpoint every N optimizer updates")
    parser.add_argument("--save-every-epochs", type=int, default=1, help="Publish a resumable checkpoint every N completed epochs")
    parser.add_argument("--cv_patience", type=int, default=3, help="Validation epochs without improvement before stopping")
    parser.add_argument("--eval-reference", default="", help="Reference wav for checkpoint audio evaluation")
    parser.add_argument("--eval-reference-text", default="", help="Transcript of the evaluation reference wav")
    parser.add_argument("--eval-text", default="", help="Text synthesized after each checkpoint")
    parser.add_argument("--accum-grad", type=int, default=0, help="Gradient accumulation batches (0 uses config)")
    # LoRA options
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-bias",
        default="none",
        choices=["none", "all", "lora_only"],
        help="LoRA bias config",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target module names",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default="",
        help="Comma-separated module names to keep trainable alongside LoRA",
    )
    parser.add_argument(
        "--lora-unfreeze",
        default="",
        help="Comma-separated parameter name prefixes to unfreeze",
    )
    parser.add_argument(
        "--lora-checkpoint",
        default="",
        help="Adapter-only warm start; not optimizer trajectory resume",
    )
    parser.add_argument(
        "--guarded-checkpoints",
        action="store_true",
        default=False,
        help="Publish content-bound native single-GPU checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="Exact newest resume_epoch_NNNNNN or resume_step_NNNNNN guarded checkpoint",
    )
    parser.add_argument(
        "--trust-resume-state",
        action="store_true",
        default=False,
        help="Acknowledge trusted pickle-capable optimizer and RNG state",
    )
    parser.add_argument(
        "--trust-model-checkpoint",
        action="store_true",
        default=False,
        help="Acknowledge trusted pickle-capable base llm.pt state",
    )
    parser.add_argument(
        "--resume-keep-last",
        type=int,
        default=3,
        help="Number of owned guarded epoch checkpoints to retain",
    )
    args = parser.parse_args()
    return args


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def freeze_all_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_by_prefix(model, prefixes: list[str]):
    if not prefixes:
        return
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad = True


def apply_lora_to_cosyvoice3(model, args):
    if not hasattr(model, "llm"):
        raise RuntimeError("Expected CosyVoice3LM with .llm attribute")
    encoder = model.llm
    if not hasattr(encoder, "model"):
        raise RuntimeError("Expected Qwen2Encoder with .model attribute")
    base = encoder.model

    lora_targets = parse_list(args.lora_target_modules)
    modules_to_save = parse_list(args.lora_modules_to_save) or None

    if args.lora_checkpoint:
        peft_model = PeftModel.from_pretrained(
            base, args.lora_checkpoint, is_trainable=True
        )
    else:
        lora_kwargs = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": lora_targets,
            "bias": args.lora_bias,
            "modules_to_save": modules_to_save,
        }
        if TaskType is not None:
            lora_kwargs["task_type"] = TaskType.CAUSAL_LM
        lora_cfg = LoraConfig(**lora_kwargs)
        peft_model = get_peft_model(base, lora_cfg)
    encoder.model = peft_model

    # After PEFT wrapping, the upstream CosyVoice3LM forward path accesses
    # encoder.model.model.embed_tokens, expecting Qwen2Model. But now:
    #   encoder.model = PeftModel
    #   encoder.model.model = Qwen2ForCausalLM  (not Qwen2Model)
    #   encoder.model.model.model = Qwen2Model   (has embed_tokens)
    # Fix: proxy embed_tokens on Qwen2ForCausalLM so the upstream path works.
    qwen2_causal = peft_model.model  # Qwen2ForCausalLM
    if not hasattr(qwen2_causal, "embed_tokens") and hasattr(qwen2_causal, "model"):
        qwen2_causal.embed_tokens = qwen2_causal.model.embed_tokens

    return peft_model


def init_optimizer_and_scheduler_lora(args, configs, model, gan):
    if gan is True:
        raise RuntimeError("LoRA training script supports LLM only (gan=False).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after applying LoRA.")

    if configs["train_conf"]["optim"] == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, **configs["train_conf"]["optim_conf"]
        )
    elif configs["train_conf"]["optim"] == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, **configs["train_conf"]["optim_conf"]
        )
    else:
        raise ValueError("unknown optimizer: " + configs["train_conf"])

    # The upstream CosyVoice SFT config uses constantlr. For this LoRA GUI
    # recipe that leaves LR flat for the whole run; use a short warmup followed
    # by cosine decay over the requested optimizer steps instead.
    if args.train_engine == "single_gpu":
        total_updates = max(1, int(args.max_steps or (args.max_epoch * 100)))
        warmup_updates = max(1, min(total_updates // 20, 100))
        peak_lr = float(optimizer.param_groups[0]["lr"])
        def schedule(step):
            if step < warmup_updates:
                return max(1e-3, (step + 1) / warmup_updates)
            progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return model, optimizer, scheduler, None, None

    if configs["train_conf"]["scheduler"] == "warmuplr":
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs["train_conf"]["scheduler_conf"])
    elif configs["train_conf"]["scheduler"] == "warmupconstant":
        scheduler_type = WarmupPolicy
        scheduler = WarmupPolicy(optimizer, **configs["train_conf"]["scheduler_conf"])
    elif configs["train_conf"]["scheduler"] == "NoamHoldAnnealing":
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(
            optimizer, **configs["train_conf"]["scheduler_conf"]
        )
    elif configs["train_conf"]["scheduler"] == "constantlr":
        scheduler_type = ConstantLR
        scheduler = ConstantLR(optimizer)
    else:
        raise ValueError("unknown scheduler: " + configs["train_conf"])

    if args.train_engine == "deepspeed":
        if deepspeed is None:
            raise RuntimeError("DeepSpeed is not supported by the native Windows Easy GUI runtime.")
        if scheduler_type is ConstantLR:

            def scheduler_fn(opt):
                return scheduler_type(opt)
        else:

            def scheduler_fn(opt):
                return scheduler_type(opt, **configs["train_conf"]["scheduler_conf"])

        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=None,
            lr_scheduler=scheduler_fn,
            model_parameters=trainable_params,
        )

    return model, optimizer, scheduler, None, None


def save_lora_checkpoint(model, model_name, info_dict):
    rank = int(os.environ.get("RANK", 0))
    model_dir = info_dict["model_dir"]
    if rank != 0:
        return

    os.makedirs(model_dir, exist_ok=True)
    tag_dir = os.path.join(model_dir, model_name)
    info_path = os.path.join(model_dir, f"{model_name}.yaml")
    guarded = bool(info_dict.get("guarded_checkpoints", False))
    if guarded and (os.path.lexists(tag_dir) or os.path.lexists(info_path)):
        raise ResumeContractError(
            f"Refusing to overwrite or adopt LoRA export: {model_name}"
        )
    os.makedirs(tag_dir, exist_ok=not guarded)

    base_model = unwrap_model(model)
    if not hasattr(base_model, "llm") or not hasattr(base_model.llm, "model"):
        raise RuntimeError("Expected CosyVoice3LM with Qwen2Encoder for LoRA saving")

    peft_model = base_model.llm.model
    if hasattr(peft_model, "save_pretrained"):
        peft_model.save_pretrained(tag_dir)
        normalize_adapter_config(tag_dir)
    else:
        raise RuntimeError("LoRA model missing save_pretrained; did you apply PEFT?")

    info_dict = dict(info_dict)
    info_dict["save_time"] = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(info_path, "w", encoding="utf-8") as fout:
        yaml.dump(info_dict, fout)
    logging.info("[Rank %s] LoRA checkpoint saved to %s", rank, tag_dir)


def evaluate_checkpoint_audio(args, writer, adapter_dir, step):
    """Synthesize one fixed evaluation utterance and publish it to TensorBoard."""
    if writer is None or not args.eval_reference or not args.eval_text:
        return
    out_dir = Path(args.model_dir) / ".eval_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"checkpoint-{int(step):06d}.wav"
    prompt_text = str(args.eval_reference_text or "").strip()
    if prompt_text and "<|endofprompt|>" not in prompt_text:
        prompt_text += "<|endofprompt|>"
    command = [sys.executable, str(Path(__file__).with_name("infer_cosyvoice3_lora.py")),
               "--pretrained-dir", str(Path(args.checkpoint).parent),
               "--lora-dir", str(adapter_dir), "--prompt-wav", str(args.eval_reference),
               "--prompt-text", prompt_text, "--text", str(args.eval_text),
               "--out-wav", str(out_wav)]
    try:
        result = subprocess.run(
            command, cwd=str(Path(__file__).parents[2]), check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            # A malformed/too-short reference can make HiFiGAN reject a
            # tiny feature tensor. This must never abort the training run.
            raise RuntimeError(detail[-1] if detail else f"inference exit code {result.returncode}")
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(out_wav))
        writer.add_audio("eval/checkpoint", waveform, global_step=int(step), sample_rate=sample_rate)
        writer.flush()
        logging.info("Evaluation audio written to TensorBoard for step %s", step)
    except Exception as exc:
        logging.warning("Checkpoint audio evaluation skipped at step %s: %s", step, exc)


@record
def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s"
    )
    configure_reproducibility(args)

    if args.model != "llm":
        raise RuntimeError("LoRA script expects --model llm (CosyVoice3 LLM).")
    if args.dpo:
        raise RuntimeError("LoRA script does not support DPO training.")

    gan = False
    override_dict = {
        k: None for k in ["llm", "flow", "hift", "hifigan"] if k != args.model
    }
    with open(args.config, "r", encoding="utf-8") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={**override_dict, "qwen_pretrain_path": args.qwen_pretrain_path},
        )
    runtime_args = {
        key: value
        for key, value in vars(args).items()
        if key not in {"max_epoch", "learning_rate"}
    }
    configs["train_conf"].update(runtime_args)
    if args.accum_grad > 0:
        configs["train_conf"]["accum_grad"] = max(1, args.accum_grad)
    configs["train_conf"]["cv_patience"] = max(1, int(args.cv_patience))
    if args.max_epoch is not None:
        if args.max_epoch < 1:
            raise ValueError("--max_epoch must be positive")
        configs["train_conf"]["max_epoch"] = args.max_epoch
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning_rate must be positive")
        configs["train_conf"]["optim_conf"]["lr"] = args.learning_rate

    init_distributed(args)
    guarded = bool(args.guarded_checkpoints)
    output_lock_handle = None
    contract = None
    resume_checkpoint = None
    resume_state = None
    if guarded:
        if args.train_engine != "single_gpu" or distributed_world_size() != 1:
            raise ResumeContractError(
                "Guarded continuation supports one native single-GPU process only; DeepSpeed and "
                "multi-rank state need a collective protocol"
            )
        if args.num_workers != 0:
            raise ResumeContractError(
                "Guarded continuation requires --num_workers 0 because worker RNG and "
                "iterator state are not persisted"
            )
        if not args.checkpoint or not args.trust_model_checkpoint:
            raise ResumeContractError(
                "Guarded training requires --checkpoint and --trust-model-checkpoint"
            )
        if args.lora_checkpoint:
            raise ResumeContractError(
                "--lora-checkpoint is an adapter-only warm start and cannot be combined "
                "with guarded trajectory continuation"
            )
        output_lock_handle = acquire_output_lock(args.model_dir)
        if not args.resume_from:
            require_fresh_output(args.model_dir)
            if args.trust_resume_state:
                raise ResumeContractError(
                    "--trust-resume-state is valid only with an exact --resume-from"
                )
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = (
        init_dataset_and_dataloader(args, configs, gan, args.dpo)
    )
    configs = check_modify_and_save_config(args, configs)
    if guarded and args.resume_from:
        initial_adapter = Path(args.model_dir) / INITIAL_ADAPTER_NAME
        contract = guarded_contract(args, configs, initial_adapter)
        resume_checkpoint, resume_state = validate_checkpoint(
            args.resume_from,
            output_dir=args.model_dir,
            expected_contract=contract,
            trust_resume_state=args.trust_resume_state,
            world_size=distributed_world_size(),
            train_engine=args.train_engine,
        )
        args.lora_checkpoint = str(resume_checkpoint)
    writer = init_summarywriter(args)

    # build model
    model = configs[args.model]
    start_step, start_epoch = 0, -1

    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            if not guarded and "step" in state_dict:
                start_step = state_dict["step"]
            if not guarded and "epoch" in state_dict:
                start_epoch = state_dict["epoch"]
        else:
            logging.warning("checkpoint %s does not exist", args.checkpoint)

    # freeze base and apply LoRA
    freeze_all_params(model)
    peft_model = apply_lora_to_cosyvoice3(model, args)
    unfreeze_by_prefix(model, parse_list(args.lora_unfreeze))

    if guarded and resume_checkpoint is None:
        initial_adapter = publish_initial_adapter(
            output_dir=args.model_dir,
            adapter_saver=peft_model.save_pretrained,
        )
        contract = guarded_contract(args, configs, initial_adapter)

    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    # push model to device / ddp or ds
    model = wrap_cuda_model(args, model)

    # init optimizer and scheduler for LoRA params
    model, optimizer, scheduler, optimizer_d, scheduler_d = (
        init_optimizer_and_scheduler_lora(args, configs, model, gan)
    )

    # patch executor save_model to LoRA saver
    import cosyvoice.utils.executor as executor_module

    executor_module.save_model = save_lora_checkpoint

    info_dict = deepcopy(configs["train_conf"])
    info_dict["disable_cv_checkpoint"] = True
    info_dict["step"] = start_step
    info_dict["epoch"] = start_epoch
    info_dict["lora"] = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "bias": args.lora_bias,
        "target_modules": parse_list(args.lora_target_modules),
        "modules_to_save": parse_list(args.lora_modules_to_save),
        "unfreeze": parse_list(args.lora_unfreeze),
        "checkpoint": args.lora_checkpoint or "",
    }
    info_dict["guarded_checkpoints"] = guarded
    info_dict["max_steps"] = max(0, int(args.max_steps or 0))
    info_dict["save_every_steps"] = max(0, int(args.save_every_steps or 0))

    executor = Executor(gan=gan, ref_model=None, dpo_loss=None)
    executor.step = start_step

    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
    if resume_checkpoint is not None and resume_state is not None:
        start_step = int(resume_state["completed_step"])
        start_epoch = int(resume_state["completed_epoch"])
        executor.step = start_step
        info_dict["step"] = start_step
        info_dict["epoch"] = start_epoch
        info_dict.update(resume_state.get("monitor_state", {}))
        restore_runtime_state(
            resume_checkpoint / "runtime-state.pt",
            optimizer,
            scheduler,
            scaler,
        )
    logging.info("start step %s start epoch %s", start_step, start_epoch)

    last_published_step = start_step

    def publish_progress(completed_epoch: int, completed_step: int, unit: str) -> None:
        nonlocal last_published_step
        if completed_step == last_published_step:
            return
        if not guarded:
            name = f"checkpoint-{completed_step:06d}"
            save_lora_checkpoint(model, name, info_dict)
            evaluate_checkpoint_audio(args, writer, Path(args.model_dir) / name, completed_step)
            logging.info("Checkpoint saved to %s", Path(args.model_dir) / name)
            last_published_step = completed_step
            return
        assert contract is not None
        if distributed_rank() == 0:
            published = publish_checkpoint(
                output_dir=args.model_dir,
                completed_epoch=completed_epoch,
                completed_step=completed_step,
                contract=contract,
                adapter_saver=lambda directory: unwrap_model(model).llm.model.save_pretrained(directory),
                runtime_state_saver=lambda path: save_runtime_state(path, optimizer, scheduler, scaler),
                monitor_state=monitor_state(info_dict),
                checkpoint_unit=unit,
            )
            logging.info("Guarded %s checkpoint published to %s", unit, published)
            prune_owned_checkpoints(args.model_dir, keep_last=args.resume_keep_last, expected_contract=contract)
        if dist.is_initialized():
            dist.barrier()
        last_published_step = completed_step

    if info_dict["save_every_steps"] > 0:
        info_dict["step_checkpoint_callback"] = lambda epoch, step: publish_progress(epoch, step, "step")

    # In Steps mode, epochs are only the container loop. Do not let the GUI's
    # cosmetic epoch limit terminate a run before its requested step target.
    epoch_limit = int(info_dict["max_epoch"])
    if int(args.max_steps or 0) > 0:
        epoch_limit = max(epoch_limit, int(args.max_steps) + start_epoch + 2)
    for epoch in range(start_epoch + 1, epoch_limit):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        if dist.is_initialized():
            dist.barrier()
            group_join = dist.new_group(
                backend="gloo", timeout=datetime.timedelta(seconds=args.timeout)
            )
        else:
            group_join = None
        executor.train_one_epoc(
            model,
            optimizer,
            scheduler,
            train_data_loader,
            cv_data_loader,
            writer,
            info_dict,
            scaler,
            group_join,
        )
        if group_join is not None:
            dist.destroy_process_group(group_join)
        reached_max_steps = info_dict["max_steps"] > 0 and int(executor.step) >= info_dict["max_steps"]
        final_epoch = epoch + 1 >= info_dict["max_epoch"]
        if info_dict["max_steps"] <= 0:
            cadence = max(1, int(args.save_every_epochs or 1))
            if (epoch + 1) % cadence == 0 or final_epoch:
                publish_progress(epoch, int(executor.step), "epoch")
        elif reached_max_steps:
            # Mandatory final save when the target is not an exact cadence multiple.
            publish_progress(epoch, int(executor.step), "step")
        if reached_max_steps:
            if distributed_rank() == 0:
                logging.info("Reached max_steps=%s at epoch %s.", info_dict["max_steps"], epoch)
            break
    if not guarded and int(executor.step) > 0 and distributed_rank() == 0:
        info_dict["step"] = int(executor.step)
        save_lora_checkpoint(model, "final_adapter", info_dict)
        evaluate_checkpoint_audio(args, writer, Path(args.model_dir) / "final_adapter", int(executor.step))
        logging.info("Final adapter saved to %s", Path(args.model_dir) / "final_adapter")
    if output_lock_handle is not None:
        output_lock_handle.close()


if __name__ == "__main__":
    main()
