"""Download the PRD-required models and preserve their resolved revisions."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "bge-base-zh-v1.5": ("BAAI/bge-base-zh-v1.5", os.getenv("MODEL_BGE_REVISION", "main")),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", os.getenv("MODEL_RERANKER_REVISION", "main")),
}
MODEL_ROOT = Path("/models")


def main() -> None:
    api = HfApi()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }
    for local_name, (repo_id, revision) in MODELS.items():
        info = api.model_info(repo_id=repo_id, revision=revision)
        destination = MODEL_ROOT / local_name
        snapshot_download(repo_id=repo_id, revision=info.sha, local_dir=destination)
        manifest["models"][local_name] = {
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": info.sha,
            "local_path": str(destination),
        }
        print(f"downloaded {repo_id}@{info.sha}", flush=True)
    (MODEL_ROOT / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
