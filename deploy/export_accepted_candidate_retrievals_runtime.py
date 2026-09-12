"""Freeze actual active-Hybrid retrievals for accepted evaluation candidates.

This runs inside rag-api's image. It constructs the same CompleteRetriever as
the HTTP route, but keeps the local embedding and reranker models resident for
the complete batch; no degraded fallback is allowed.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from greenspec_rag.retrieval.evidence import BGEReranker, CompleteRetriever, EvidencePackBuilder
from greenspec_rag.retrieval.hybrid import HybridRetriever, SentenceTransformerEmbedder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def request_for(candidate: dict[str, Any], top_k: int) -> dict[str, Any]:
    standard_ids = [] if candidate.get("category") == "insufficient_evidence" else list(candidate.get("expected_standard_ids") or [])
    return {"query": candidate["question"], "project_profile": {}, "filters": {"standard_ids": standard_ids, "clause_nos": [], "as_of_date": None, "must_be_current": False, "include_commentary": False}, "top_k": top_k, "allow_degraded": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-id", help="Export only one accepted candidate; useful for a failed-record retry.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    active = json.loads((root / "data/indexes/hybrid/active.json").read_text(encoding="utf-8"))
    device = os.environ.get("RAG_EMBEDDING_DEVICE")
    qdrant = QdrantClient(url=os.environ.get("RAG_QDRANT_URL", "http://qdrant:6333"))
    embedder = SentenceTransformerEmbedder(os.environ.get("RAG_BGE_MODEL_PATH", "/models/bge-base-zh-v1.5"), device=device)
    hybrid = HybridRetriever(hybrid_manifest_path=root / active["manifest_path"], qdrant_client=qdrant, embedder=embedder, registry_path=root / "data/registry/standard_registry.json")
    reranker = BGEReranker(os.environ.get("RAG_RERANKER_MODEL_PATH", "/models/bge-reranker-v2-m3"), device=device or "cuda")
    complete = CompleteRetriever(hybrid=hybrid, reranker=reranker, evidence_builder=EvidencePackBuilder(registry_path=root / "data/registry/standard_registry.json", relations_path=root / "data/registry/clause_relations.jsonl"))
    run_id = f"accepted_candidate_export_{uuid.uuid4().hex}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    records: list[dict[str, Any]] = []
    candidates = read_jsonl(args.accepted)
    if args.candidate_id:
        candidates = [candidate for candidate in candidates if candidate.get("candidate_id") == args.candidate_id]
        if len(candidates) != 1:
            raise RuntimeError(f"accepted candidate not found exactly once: {args.candidate_id}")
    with partial.open("w", encoding="utf-8") as handle:
        for index, candidate in enumerate(candidates, start=1):
            request = request_for(candidate, args.top_k)
            try:
                result, pack = complete.retrieve(request, request_id=f"{run_id}_{candidate['candidate_id']}", retrieval_run_id=f"{run_id}_{candidate['candidate_id']}")
                error = None
            except Exception as exc:
                result, pack, error = None, None, {"code": type(exc).__name__, "message": str(exc)}
            record = {"schema_version": "greenspec.accepted_candidate_retrieval.v1", "exported_at": datetime.now(timezone.utc).isoformat(), "export_run_id": run_id, "execution_path": "active_hybrid_runtime_same_as_rag_api", "candidate_id": candidate["candidate_id"], "question": candidate["question"], "category": candidate.get("category"), "difficulty_label": candidate.get("difficulty_label"), "required_evidence_ids": candidate.get("required_evidence_ids", []), "supporting_evidence_ids": candidate.get("supporting_evidence_ids", []), "forbidden_near_misses": candidate.get("forbidden_near_misses", []), "request": request, "retrieval": {"request_id": (result or {}).get("request_id") or (pack or {}).get("request_id"), "retrieval_run_id": (result or {}).get("retrieval_run_id") or (pack or {}).get("retrieval_run_id"), "index_manifest_id": (result or {}).get("index_manifest_id") or (pack or {}).get("index_manifest_id") or hybrid.manifest["index_manifest_id"], "top_k_items": result.get("items", []) if result else [], "filters_applied": result.get("filters_applied") if result else None, "warnings": result.get("warnings", []) if result else [], "degraded_modes": result.get("degraded_modes", []) if result else [], "error": error}, "evidence_pack": pack}
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}] {candidate['candidate_id']}: {'ok' if pack else error['code']}", flush=True)
    partial.replace(args.output)
    failures = sum(record["evidence_pack"] is None for record in records)
    print(json.dumps({"output": str(args.output), "case_count": len(records), "evidence_pack_count": len(records) - failures, "failed_count": failures, "index_manifest_id": hybrid.manifest["index_manifest_id"]}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
