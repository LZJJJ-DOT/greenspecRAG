"""Compare BM25, dense, and RRF hybrid retrieval against frozen gold IDs.

The command is deliberately candidate-only: it never changes the production
``data/indexes/hybrid/active.json`` pointer.  Publishing requires a separate,
post-review operation after this report passes its retrieval gate.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:  # Supports both ``python -m deploy...`` and direct script execution.
    from deploy.run_evaluation import _load_cases
    from deploy.prepare_ragas_retrieval_dataset import load_frozen_cases, load_suite
except ModuleNotFoundError:  # pragma: no cover - direct runtime-container path
    from run_evaluation import _load_cases
    from prepare_ragas_retrieval_dataset import load_frozen_cases, load_suite
from greenspec_rag.retrieval.hybrid import HybridRetriever, SentenceTransformerEmbedder


MODES = ("bm25", "dense", "hybrid")
HIT_AT = 10
RETRIEVAL_THRESHOLD = 0.85
TRACE_FIELDS = {"bm25_rank", "dense_rank", "bm25_score", "dense_score", "rrf_score", "rrf_k"}
PARENT_CAP_EXEMPT = {"normative_table", "normative_formula"}
SPECIAL_CONTENT_TYPES = ("normative_table", "normative_formula")


def _top10_request(request: dict) -> dict:
    """Keep every frozen request intact while obtaining the required Hit@10."""
    evaluated = copy.deepcopy(request)
    evaluated["top_k"] = max(HIT_AT, int(evaluated["top_k"]))
    return evaluated


def _hit(result: dict, gold: set[str]) -> tuple[bool, list[str]]:
    returned = [str(item["evidence_id"]) for item in result["items"][:HIT_AT]]
    return bool(gold.intersection(returned)), returned


def _locator_is_legal(item: dict) -> bool:
    """Require complete, ordered PDF and printed-page citation locators."""
    fields = ("pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end")
    if not all(isinstance(item.get(field), int) and item[field] >= 1 for field in fields):
        return False
    return item["pdf_page_start"] <= item["pdf_page_end"] and item["printed_page_start"] <= item["printed_page_end"]


def _retrieval_trace(result: dict) -> list[dict]:
    """Persist a replayable, metadata-only ranked trace for audit and review."""
    fields = (
        "evidence_id",
        "clause_id",
        "parent_id",
        "content_type",
        "source_id",
        "pdf_page_start",
        "pdf_page_end",
        "printed_page_start",
        "printed_page_end",
    )
    return [
        {
            **{field: item.get(field) for field in fields},
            "rank": rank,
            "trace": item.get("trace", {}),
        }
        for rank, item in enumerate(result["items"][:HIT_AT], start=1)
    ]


def _integrity(result: dict, retriever: HybridRetriever) -> dict:
    """Verify the Day 4 invariants without recording source text in the report."""
    items = result["items"]
    clause_ids = [str(item["clause_id"]) for item in items]
    parent_counts: dict[str, int] = {}
    for item in items:
        parent_id = item.get("parent_id")
        if parent_id and item.get("content_type") not in PARENT_CAP_EXEMPT:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
    removed = result["filters_applied"]["fusion_trace"]["diversity_removed"]
    protected_parent_cap_removals = []
    for removal in removed:
        if removal.get("reason") != "parent_cap":
            continue
        node = retriever._load_node(str(removal["clause_id"]))
        if node.get("content_type") in PARENT_CAP_EXEMPT:
            protected_parent_cap_removals.append(str(removal["clause_id"]))
    checks = {
        "unique_clause_ids": len(clause_ids) == len(set(clause_ids)),
        "trace_complete": all(TRACE_FIELDS.issubset(set(item.get("trace", {}))) for item in items),
        "page_locators_legal": all(_locator_is_legal(item) for item in items[:HIT_AT]),
        "ordinary_parent_cap": all(count <= 3 for count in parent_counts.values()),
        "table_formula_parent_cap_exempt": not protected_parent_cap_removals,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "diversity_removed_count": len(removed),
        "protected_parent_cap_removals": protected_parent_cap_removals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a candidate hybrid manifest without publishing it.")
    parser.add_argument("--hybrid-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path("data/registry/standard_registry.json"))
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/mode_runs"))
    parser.add_argument("--suite", type=Path, default=Path("data/evaluation/regression_suite.v1.json"))
    args = parser.parse_args()

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - runtime image gate
        raise SystemExit("qdrant-client is required; run this in the GreenSpec runtime container") from exc

    suite = load_suite(args.suite)
    frozen_cases = load_frozen_cases(suite)
    validated_cases = _load_cases(args.questions)
    if validated_cases != frozen_cases:
        raise SystemExit("schema-validated questions differ from the frozen regression suite")
    cases = [case for case in validated_cases if case["gold_evidence_ids"]]
    if len(cases) < 20:
        raise SystemExit("at least 20 positive cases with gold evidence IDs are required")

    retriever = HybridRetriever(
        hybrid_manifest_path=args.hybrid_manifest,
        qdrant_client=QdrantClient(url=args.qdrant_url),
        embedder=SentenceTransformerEmbedder(args.model_path, device=args.device),
        registry_path=args.registry,
    )
    counts = {mode: 0 for mode in MODES}
    special_counts = {content_type: {"cases": 0, "hits": 0} for content_type in SPECIAL_CONTENT_TYPES}
    locator_metrics = {"checked": 0, "valid": 0, "invalid": []}
    case_rows = []
    integrity_failures = []
    for case in cases:
        gold = set(case["gold_evidence_ids"])
        expected_content_type = str(case.get("expected_content_type", ""))
        if expected_content_type in special_counts:
            special_counts[expected_content_type]["cases"] += 1
        row = {"eval_id": case["eval_id"], "gold_evidence_ids": sorted(gold), "modes": {}}
        request = _top10_request(case["request"])
        for mode in MODES:
            result = retriever.retrieve(request, mode=mode, require_degraded=False, include_reranker_unavailable=False)
            hit, returned = _hit(result, gold)
            integrity = _integrity(result, retriever)
            counts[mode] += int(hit)
            if mode == "hybrid":
                if expected_content_type in special_counts:
                    special_counts[expected_content_type]["hits"] += int(hit)
                for item in result["items"][:HIT_AT]:
                    locator_metrics["checked"] += 1
                    if _locator_is_legal(item):
                        locator_metrics["valid"] += 1
                    else:
                        locator_metrics["invalid"].append({"eval_id": case["eval_id"], "evidence_id": str(item["evidence_id"])})
            row["modes"][mode] = {
                "hit_at_10": hit,
                "returned_evidence_ids": returned,
                "candidate_counts": {
                    "bm25": result["filters_applied"]["bm25_candidate_count"],
                    "dense": result["filters_applied"]["dense_candidate_count"],
                },
                "retrieval_trace": _retrieval_trace(result),
                "integrity": integrity,
            }
            if not integrity["passed"]:
                integrity_failures.append({"eval_id": case["eval_id"], "mode": mode, **integrity})
        case_rows.append(row)

    total = len(cases)
    rates = {mode: counts[mode] / total for mode in MODES}
    hybrid_non_regression = counts["hybrid"] >= max(counts["bm25"], counts["dense"])
    integrity_gate_passed = not integrity_failures
    retrieval_gate_passed = hybrid_non_regression and rates["hybrid"] >= RETRIEVAL_THRESHOLD and integrity_gate_passed
    report = {
        "evaluation_run_id": f"mode_eval_{uuid.uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_manifest_id": retriever.manifest["index_manifest_id"],
        "hybrid_manifest_path": str(args.hybrid_manifest),
        "qdrant_collection": retriever.manifest["qdrant_collection"],
        "canonical_sha256": retriever.manifest["canonical_sha256"],
        "embedding": retriever.manifest["embedding"],
        "case_count": total,
        "frozen_suite_id": suite["suite_id"],
        "frozen_question_sha256": suite["question_sha256"],
        "frozen_diagnostic_case_count": suite["diagnostic_case_count"],
        "hit_at_10": {mode: {"hits": counts[mode], "rate": rates[mode]} for mode in MODES},
        "deterministic_metrics": {
            **{f"{content_type}_hit_at_10_hybrid": {**values, "rate": (values["hits"] / values["cases"] if values["cases"] else None)} for content_type, values in special_counts.items()},
            "hybrid_page_locator_legality": {**locator_metrics, "rate": (locator_metrics["valid"] / locator_metrics["checked"] if locator_metrics["checked"] else None)},
        },
        "hybrid_non_regression": hybrid_non_regression,
        "integrity_gate_passed": integrity_gate_passed,
        "integrity_failures": integrity_failures,
        "retrieval_gate_passed": retrieval_gate_passed,
        "publication_status": "retrieval_eligible_pending_citation_review" if retrieval_gate_passed else "experimental",
        "unresolved": ([] if retrieval_gate_passed else ["hybrid must reach Hit@10 >= 0.85, not underperform either single retriever, and satisfy all Day 4 integrity checks"])
        + ["Citation Accuracy and Citation Completeness require run-bound human review before baseline publication"],
        "cases": case_rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{report['evaluation_run_id']}.json"
    temporary = output.with_name(f".{output.name}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"evaluation_run_id": report["evaluation_run_id"], "report_path": str(output), "hit_at_10": report["hit_at_10"], "retrieval_gate_passed": retrieval_gate_passed, "publication_status": report["publication_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
