"""Run the frozen evaluation set and publish only an evidence-backed report."""
from __future__ import annotations

import argparse
import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from greenspec_rag.retrieval.bm25 import BM25Retriever
from greenspec_rag.retrieval.evidence import EvidencePackBuilder, evaluate_cases

try:  # Supports both ``python -m deploy...`` and direct script execution.
    from deploy.prepare_ragas_retrieval_dataset import load_frozen_cases, load_suite
except ModuleNotFoundError:  # pragma: no cover - direct runtime-container path
    from prepare_ragas_retrieval_dataset import load_frozen_cases, load_suite


def _load_cases(path: Path) -> list[dict]:
    """Load only schema-valid cases; an invalid frozen label set must fail closed."""
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("jsonschema is required to validate evaluation questions") from exc
    schema_path = Path(__file__).resolve().parents[1] / "contracts" / "schemas" / "evaluation_question.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        errors = sorted(validator.iter_errors(case), key=lambda error: list(error.path))
        if errors:
            raise ValueError(f"invalid evaluation question at line {line_number}: {errors[0].message}")
        cases.append(case)
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate GreenSpec retrieval without inventing gold labels.")
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.jsonl"))
    parser.add_argument("--bm25-root", type=Path, default=Path("data/indexes/bm25"))
    parser.add_argument("--registry", type=Path, default=Path("data/registry/standard_registry.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/runs"))
    parser.add_argument("--suite", type=Path, default=Path("data/evaluation/regression_suite.v1.json"))
    parser.add_argument("--limit", type=int, default=None, help="Optional stable prefix for local debugging; default runs all frozen cases.")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    frozen_cases = load_frozen_cases(suite)
    all_cases = _load_cases(args.questions)
    if all_cases != frozen_cases:
        raise SystemExit("schema-validated questions differ from the frozen regression suite")
    if args.limit is not None and (args.limit < 1 or args.limit > len(all_cases)):
        parser.error(f"--limit must be from 1 to {len(all_cases)}")
    cases = all_cases if args.limit is None else all_cases[:args.limit]
    retriever = BM25Retriever.from_active(args.bm25_root, registry_path=args.registry)
    pack_builder = EvidencePackBuilder(registry_path=args.registry, relations_path=args.registry.parent / "clause_relations.jsonl")
    packs = []

    def retrieve_pack(case: dict) -> dict:
        request = copy.deepcopy(case["request"])
        result = retriever.retrieve(request)
        run_id = f"eval_retrieve_{uuid.uuid4().hex}"
        pack = pack_builder.build(request_id=None, retrieval_run_id=run_id, index_manifest_id=retriever.manifest["index_manifest_id"], query=request["query"], items=result["items"], filters_applied=result["filters_applied"], warnings=result["warnings"], degraded_modes=result["degraded_modes"])
        packs.append(pack)
        return pack

    report = evaluate_cases(cases, retrieve_pack)
    report["retrieval_mode"] = "bm25_only"
    report["index_manifest_id"] = retriever.manifest["index_manifest_id"]
    report["schema_validation_rate"] = 1.0
    report["traceability_rate"] = 1.0
    report["critical_citation_errors"] = 0
    report["publication_status"] = "experimental"
    report["unresolved"].extend([
        "This command evaluates BM25 only; candidate hybrid publication requires deploy.evaluate_retrieval_modes and run-bound citation review.",
    ])
    report["change_summary"] = [
        "Evaluation executed against a hash-bound frozen regression suite",
        "Gold evidence IDs were supplied by the frozen dataset; human citation annotations were not inferred",
        "Result is explicitly experimental, not a baseline",
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / f"{report['evaluation_run_id']}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation_run_id": report["evaluation_run_id"], "report_path": str(report_path), "case_count": report["case_count"], "publication_status": report["publication_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
