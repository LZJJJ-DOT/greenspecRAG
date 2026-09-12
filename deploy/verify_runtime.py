"""Fail fast if the runtime image loses a required local inference capability."""
from __future__ import annotations

import sqlite3

import jieba
import torch


def main() -> None:
    options = {row[0] for row in sqlite3.connect(":memory:").execute("pragma compile_options")}
    if "ENABLE_FTS5" not in options:
        raise RuntimeError("SQLite FTS5 is not enabled")
    if jieba.__version__ != "0.42.1":
        raise RuntimeError(f"Expected jieba 0.42.1, got {jieba.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the rag runtime container")
    print(
        {
            "sqlite_fts5": True,
            "jieba_version": jieba.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
