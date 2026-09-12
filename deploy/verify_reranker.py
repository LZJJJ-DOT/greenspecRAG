"""Verify local bge-reranker-v2-m3 loading, device, bounded input, and scoring."""
from __future__ import annotations

import json
import os

from greenspec_rag.retrieval.evidence import BGEReranker


def main() -> None:
    reranker = BGEReranker(os.getenv("RAG_RERANKER_MODEL_PATH", "/models/bge-reranker-v2-m3"), device=os.getenv("RAG_RERANKER_DEVICE", "cuda"))
    score = reranker.score("建筑节能要求", ["建筑节能设计应符合规范要求"])[0]
    print(json.dumps({"model_id": reranker.model_id, "device": reranker.device, "batch_size": reranker.batch_size, "query_max_tokens": 128, "document_max_tokens": 384, "timeout_seconds": reranker.timeout_seconds, "smoke_score": score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
