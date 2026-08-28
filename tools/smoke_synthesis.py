from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice_easy.runtime import DEFAULT_MODEL, ENGINE
from cosyvoice_easy.schemas import InferenceRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--reference", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--text", default="CosyVoice3 Easy GUI is ready.")
    args = parser.parse_args()
    output, status = ENGINE.generate(InferenceRequest(text=args.text, reference_audio=args.reference,
                                                      prompt_text=args.transcript, seed=0), args.model_dir, True, False)
    print(status)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
