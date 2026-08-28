"""Project-local Windows CUDA DLL bootstrap.

Python imports sitecustomize automatically when this project root is on
sys.path. Keep the bootstrap deliberately small and tolerant so utility
commands can still run before the full environment is installed.
"""
from __future__ import annotations

try:
    from cosyvoice.utils.cuda_runtime import prepare_local_cuda_runtime
    prepare_local_cuda_runtime()
except Exception:
    pass
