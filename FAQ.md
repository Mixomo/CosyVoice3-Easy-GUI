## ModuleNotFoundError: No module named 'matcha'

Matcha-TTS is a third_party module. Please check `third_party` directory. If there is no `Matcha-TTS`, execute `git submodule update --init --recursive`.

run `export PYTHONPATH=third_party/Matcha-TTS` if you want to use `from cosyvoice.cli.cosyvoice import CosyVoice` in python script.

## cannot find resource.zip or cannot unzip resource.zip

Please make sure you have git-lfs installed. Execute

```sh
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
```

## Do I need to uninstall or change my global CUDA Toolkit?

No. The Windows Easy GUI runtime uses the CUDA/cuDNN DLLs shipped inside its project-local PyTorch 2.8/cu128 environment. `1- install.bat` and `2- run.bat` clear `CUDA_PATH` / `CUDA_HOME` only inside their own process and place `.venv\Lib\site-packages\torch\lib` first for DLL resolution. The system NVIDIA display driver is still required.

## Why may `uv.lock` be regenerated during development?

Patch 06 intentionally removes lock migration. If `uv lock --check` reports that the current lock is stale, universal, or from an older dependency contract, `1- install.bat` deletes it and resolves a completely new Windows x64-only CUDA 12.8 lock from `pyproject.toml`. The resolver uses a fixed cutoff and the result is validated before `uv sync --frozen`.

### Should I delete `uv.lock` manually before installing?

No manual action is needed. In the current development build the installer deletes a stale lock automatically. Once a clean lock has been generated successfully, it is reused unchanged. The production package should ship that already-clean lock so end users do not resolve dependencies at all.

### Why is there a local package called `onnxruntime` if the project uses `onnxruntime-gpu`?

It is a metadata-only compatibility shim. Faster-Whisper requires the distribution name `onnxruntime`; the shim satisfies that requirement while delegating the real runtime to `onnxruntime-gpu==1.26.0`. It contains no `onnxruntime` Python package and therefore does not overwrite the GPU module.

## What is Natural-language Instruction?

It is an ordinary free-form CosyVoice3 control prompt. You can describe accent/language, emotion, pace, volume or delivery style in normal language. The GUI adds CosyVoice3's internal `<|endofprompt|>` separator automatically. Selecting a saved Instruction Library preset copies its text into the editable instruction field so it can be adjusted before generation.

## What is Append Control Tokens / pronunciation markup?

Leave it empty for ordinary TTS. CosyVoice3 does **not** accept an unlimited family of arbitrary tags. Its tokenizer explicitly registers a finite set of vocal/style tokens (`[breath]`, `[quick_breath]`, `[noise]`, `[laughter]`, `<laughter>...</laughter>`, `[cough]`, `[clucking]`, `[accent]`, `[hissing]`, `[sigh]`, `[vocalized-noise]`, `[lipsmack]`, `[mn]`, `<strong>...</strong>`) plus finite English CMU/ARPAbet and Chinese Pinyin pronunciation-token inventories. Pronunciation tokens must be placed inline at the location being corrected. The in-app guide now lists the supported control family and explains both pronunciation systems.

## What does CV Split (%) mean?

It is the **validation holdout percentage**. A value of 10% means roughly 90% of usable samples go to Train and 10% to CV/validation. It is not the percentage used for training.

## What is Training Base Checkpoint: Base / RL?

It chooses which official CosyVoice3 checkpoint the LoRA is trained on: Base uses `llm.pt`; RL uses `llm.rl.pt`. This is not a warm-start adapter option. Patch 09 records the selected base variant in adapter metadata and prevents loading that adapter on the other variant.

## Where did CV Patience, Guarded Checkpoints, Deterministic Training and Warm-start Adapter go?

They are no longer normal GUI choices. The Easy GUI manages the training safety policy internally: CV patience defaults to 3, guarded checkpoints are enabled, three recent guarded checkpoints are retained, deterministic byte-exact mode is off for normal use, and adapter-to-adapter warm start is disabled. This keeps the public workflow aligned with the sibling Easy GUIs while preserving safe trainer behavior.

## What is Training Eval Reference?

It is an optional stable audio/text reference saved with the training project for later listening comparisons. Faster-Whisper can fill its transcript. It is not used to calculate the training/CV loss.

## What exactly does Save Project restore?

The project is now a complete Dataset + Training snapshot. Loading it from either Project Name selector restores every persisted GUI value from both tabs, including analyzed dataset rows/transcripts, Whisper settings, CV split/instruction settings, Base/RL and LoRA hyperparameters, Resume selection, and the full Eval Reference panel. Eval audio is copied into the project's own `assets` folder so it survives Gradio temporary-file cleanup.
