"""Create immutable, candidate-only Evidence Pack review material.

This tool deliberately reviews the RAG sidecar output rather than a GreenSpec
answer.  It never writes ``data/indexes/hybrid/active.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:  # Supports both ``python -m deploy...`` and direct container execution.
    from deploy.run_evaluation import _load_cases
except ModuleNotFoundError:  # pragma: no cover - direct runtime-container path
    from run_evaluation import _load_cases

from greenspec_rag.retrieval.evidence import BGEReranker, CompleteRetriever, EvidencePackBuilder
from greenspec_rag.retrieval.hybrid import HybridRetriever, SentenceTransformerEmbedder


RAW_PDF_GLOBS = {
    "GB_55015_2021": "GB55015-2021*.pdf",
    "GB_T_50378_2019": "GBT50378-2019*.pdf",
}
ANNOTATION_FIELDS = {
    "citation_accuracy": None,
    "citation_completeness": None,
    "evidence_pack_supports_question": None,
    "needs_manual_review": None,
    "annotator": None,
    "annotated_at": None,
    "notes": None,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_sources(raw_root: Path) -> dict[str, dict[str, str]]:
    sources: dict[str, dict[str, str]] = {}
    for standard_id, pattern in RAW_PDF_GLOBS.items():
        files = list(raw_root.glob(pattern))
        if len(files) != 1:
            raise ValueError(f"expected exactly one original PDF for {standard_id}, found {len(files)}")
        source = files[0]
        sources[standard_id] = {"path": str(source), "sha256": _sha256(source)}
    return sources


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate per-question Evidence Pack review artifacts for a candidate hybrid manifest.")
    parser.add_argument("--hybrid-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path("data/registry/standard_registry.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--reranker-model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/evidence_pack_review_runs"))
    args = parser.parse_args()
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - runtime image gate
        raise SystemExit("qdrant-client is required; run this in the GreenSpec runtime container") from exc

    manifest = json.loads(args.hybrid_manifest.read_text(encoding="utf-8"))
    cases = [case for case in _load_cases(args.questions) if case["gold_evidence_ids"]]
    if len(cases) < 20:
        raise ValueError("at least 20 positive questions with gold evidence IDs are required")
    raw_sources = _raw_sources(args.raw_root)
    qdrant = QdrantClient(url=args.qdrant_url)
    retriever = HybridRetriever(
        hybrid_manifest_path=args.hybrid_manifest,
        qdrant_client=qdrant,
        embedder=SentenceTransformerEmbedder(args.embedding_model_path, device=args.device),
        registry_path=args.registry,
    )
    reranker = BGEReranker(args.reranker_model_path, device=args.device)
    evidence_builder = EvidencePackBuilder(
        registry_path=args.registry,
        relations_path=args.registry.parent / "clause_relations.jsonl",
        # The runtime package deliberately contains only Python modules; the
        # contract stays in the mounted project workspace for review runs.
        schema_path=Path("contracts/schemas/evidence_pack.schema.json"),
    )
    complete = CompleteRetriever(hybrid=retriever, reranker=reranker, evidence_builder=evidence_builder)

    review_run_id = f"pack_review_{uuid.uuid4().hex}"
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=args.output_root))
    try:
        annotations = []
        case_index = []
        for case in cases:
            retrieval_run_id = f"{review_run_id}_{case['eval_id']}"
            _, pack = complete.retrieve(case["request"], request_id=None, retrieval_run_id=retrieval_run_id)
            pack_file = f"packs/{case['eval_id']}.json"
            (temporary / "packs").mkdir(exist_ok=True)
            _write_json(temporary / pack_file, pack)
            relevant_sources = {
                standard_id: raw_sources[standard_id]
                for standard_id in case["expected_standard_ids"]
                if standard_id in raw_sources
            }
            annotations.append({
                "review_run_id": review_run_id,
                "review_target": "evidence_pack",
                "eval_id": case["eval_id"],
                "retrieval_run_id": retrieval_run_id,
                "index_manifest_id": manifest["index_manifest_id"],
                "evidence_pack_file": pack_file,
                "gold_evidence_ids": case["gold_evidence_ids"],
                "required_evidence": case["required_evidence"],
                "raw_pdf_sources": relevant_sources,
                "annotation": dict(ANNOTATION_FIELDS),
            })
            case_index.append({"eval_id": case["eval_id"], "question": case["question"], "request": case["request"], "evidence_pack_file": pack_file})
        with (temporary / "review_annotations.jsonl").open("w", encoding="utf-8") as handle:
            for annotation in annotations:
                handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
        _write_json(temporary / "review_manifest.json", {
            "review_run_id": review_run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "review_target": "evidence_pack",
            "index_manifest_id": manifest["index_manifest_id"],
            "hybrid_manifest_path": str(args.hybrid_manifest),
            "qdrant_collection": manifest["qdrant_collection"],
            "case_count": len(cases),
            "raw_pdf_sources": raw_sources,
            "annotation_file": "review_annotations.jsonl",
            "case_index": case_index,
            "rules": {
                "citation_accuracy": "Every cited locator must identify the correct standard, version, clause/table/formula and PDF page, and the cited text must support the Evidence Pack item.",
                "citation_completeness": "The Evidence Pack must contain sufficient normative evidence for the question, including required evidence and all material conditions, values, units and exceptions.",
                "evidence_pack_supports_question": "The reviewed Evidence Pack, without any external answer-generation inference, supports a response to the frozen question.",
            },
        })
        final = args.output_root / review_run_id
        os.replace(temporary, final)
    except Exception:
        for child in temporary.rglob("*"):
            if child.is_file():
                child.unlink()
        if temporary.exists():
            for child in sorted(temporary.rglob("*"), reverse=True):
                if child.is_dir():
                    child.rmdir()
            temporary.rmdir()
        raise
    print(json.dumps({"review_run_id": review_run_id, "review_path": str(final), "case_count": len(cases), "index_manifest_id": manifest["index_manifest_id"], "publication_status": "pending_human_evidence_pack_review"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
