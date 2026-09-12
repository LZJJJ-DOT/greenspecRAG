"""Verify the local BGE embedding runtime before building a Qdrant collection."""
from __future__ import annotations

import json
import os

from greenspec_rag.retrieval.hybrid import SentenceTransformerEmbedder


def main() -> None:
    embedder = SentenceTransformerEmbedder(
        os.getenv("RAG_BGE_MODEL_PATH", "/models/bge-base-zh-v1.5"),
        device=os.getenv("RAG_EMBEDDING_DEVICE") or None,
    )
    vector = embedder.embed_query("建筑节能与可再生能源利用")
    print(json.dumps({"model_id": embedder.model_id, "device": embedder.device, "dimension": embedder.dimension, "normalization": round(sum(value * value for value in vector), 8), "query_instruction": "enabled"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
