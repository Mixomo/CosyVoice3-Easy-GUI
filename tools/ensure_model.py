from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosyvoice_easy.paths import ensure_layout
ensure_layout()
from cosyvoice_easy.runtime import DEFAULT_MODEL, download_model, model_complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if model_complete(args.model_dir):
        print(f"[model] Complete: {args.model_dir}")
        return 0
    if args.check_only:
        print(f"[model] Incomplete: {args.model_dir}")
        return 1
    print("[model] Downloading or repairing Fun-CosyVoice3-0.5B-2512. Hugging Face shows file progress, speed and ETA.")
    download_model(args.model_dir)
    print(f"[model] Ready: {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
