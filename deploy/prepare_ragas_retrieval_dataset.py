"""Create a hash-bound RAGAS retrieval trial dataset from a mode-evaluation run.

This adapter intentionally does not call an LLM. It turns stable evidence IDs
into RAGAS-ready contexts and preserves evidence locators for human review.
No-gold cases remain functional diagnostics, not retrieval-score aggregates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXED_REGRESSION_TAGS = {
    "E006": ["cross_page_table", "table_continuation", "citation_locator"],
    "E029": ["supporting_table", "parent_child_context"],
    "E031": ["supporting_table", "parent_child_context"],
}


def _project_path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} must contain a JSON object")
            rows.append(value)
    return rows


SUPPORTED_SUITE_SCHEMAS = {"greenspec.regression_suite.v1", "greenspec.regression_suite.v2"}


def load_suite(path: Path) -> dict[str, Any]:
    suite = _read_json(path)
    required = {"schema_version", "suite_id", "question_path", "question_sha256", "case_count", "positive_retrieval_case_count", "diagnostic_case_count", "case_ids"}
    missing = sorted(required.difference(suite))
    if missing:
        raise ValueError(f"suite is missing required fields: {', '.join(missing)}")
    if suite["schema_version"] not in SUPPORTED_SUITE_SCHEMAS:
        raise ValueError(f"unsupported suite schema: {suite['schema_version']}")
    if not isinstance(suite["case_ids"], list) or len(suite["case_ids"]) != suite["case_count"]:
        raise ValueError("suite case_ids must contain exactly case_count IDs")
    return suite


def load_frozen_cases(suite: dict[str, Any]) -> list[dict[str, Any]]:
    questions_path = _project_path(Path(suite["question_path"]))
    if sha256_file(questions_path) != suite["question_sha256"]:
        raise ValueError("questions.jsonl differs from the frozen suite; create a new suite version instead")
    cases = _read_jsonl(questions_path)
    if [str(case.get("eval_id", "")) for case in cases] != suite["case_ids"]:
        raise ValueError("question IDs or order differ from the frozen suite")
    if len(cases) != suite["case_count"]:
        raise ValueError("question count differs from the frozen suite")
    positive_count = sum(bool(case.get("gold_evidence_ids")) for case in cases)
    if positive_count != suite["positive_retrieval_case_count"] or len(cases) - positive_count != suite["diagnostic_case_count"]:
        raise ValueError("positive or diagnostic case count differs from the frozen suite")
    return cases


def load_canonical(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    nodes = _read_jsonl(path)
    by_id = {str(node["clause_id"]): node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("canonical contains duplicate clause_id values")
    return by_id, sha256_file(path)


def case_tags(case: dict[str, Any]) -> list[str]:
    tags = {str(case.get("expected_answer_type", "unknown"))}
    tags.add("gold_retrieval" if case.get("gold_evidence_ids") else "diagnostic_no_gold")
    if case.get("expected_content_type") in {"normative_table", "normative_formula"}:
        tags.add(str(case["expected_content_type"]))
    tags.update(FIXED_REGRESSION_TAGS.get(str(case["eval_id"]), []))
    return sorted(tags)


def _evidence_metadata(node: dict[str, Any]) -> dict[str, Any]:
    return {key: node.get(key) for key in ("clause_id", "standard_id", "clause_no", "content_type", "parent_id", "pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end")}


def _context(node: dict[str, Any]) -> str:
    text = node.get("retrieval_text") or node.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"canonical node {node.get('clause_id')} has no usable retrieval text")
    return text


def build_ragas_rows(cases: list[dict[str, Any]], canonical: dict[str, dict[str, Any]], mode_report: dict[str, Any], mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report_cases = mode_report.get("cases")
    if not isinstance(report_cases, list):
        raise ValueError("mode report has no cases list")
    report_by_id = {str(row.get("eval_id")): row for row in report_cases}
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for case in cases:
        eval_id = str(case["eval_id"])
        gold_ids = [str(item) for item in case.get("gold_evidence_ids", [])]
        if not gold_ids:
            diagnostics.append({"eval_id": eval_id, "user_input": case["question"], "tags": case_tags(case), "request": case["request"], "exclusion_reason": "No gold_evidence_ids: retain as a frozen functional diagnostic, exclude from RAGAS retrieval aggregates."})
            continue
        report_case = report_by_id.get(eval_id)
        if report_case is None:
            raise ValueError(f"mode report has no result for positive frozen case {eval_id}")
        mode_data = report_case.get("modes", {}).get(mode)
        if not isinstance(mode_data, dict):
            raise ValueError(f"mode report has no {mode!r} result for {eval_id}")
        retrieved_ids = [str(item) for item in mode_data.get("returned_evidence_ids", [])]
        missing = [item for item in [*gold_ids, *retrieved_ids] if item not in canonical]
        if missing:
            raise ValueError(f"canonical is missing evidence IDs for {eval_id}: {sorted(set(missing))}")
        rows.append({
            "schema_version": "greenspec.ragas_retrieval_sample.v1",
            "eval_id": eval_id,
            "user_input": case["question"],
            "retrieved_context_ids": retrieved_ids,
            "retrieved_contexts": [_context(canonical[item]) for item in retrieved_ids],
            "reference_context_ids": gold_ids,
            "reference_contexts": [_context(canonical[item]) for item in gold_ids],
            "evidence_pack": {"retrieved": [_evidence_metadata(canonical[item]) for item in retrieved_ids], "reference": [_evidence_metadata(canonical[item]) for item in gold_ids]},
            "metadata": {"tags": case_tags(case), "expected_answer_type": case["expected_answer_type"], "expected_content_type": case["expected_content_type"], "required_evidence": case["required_evidence"]},
        })
    return rows, diagnostics


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a hash-bound RAGAS retrieval trial dataset.")
    parser.add_argument("--suite", type=Path, default=Path("data/evaluation/regression_suite.v1.json"))
    parser.add_argument("--canonical", type=Path, required=True, help="Canonical clauses.jsonl used by the evaluated index")
    parser.add_argument("--mode-report", type=Path, required=True, help="Output of deploy.evaluate_retrieval_modes")
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/ragas/datasets"))
    args = parser.parse_args()
    suite = load_suite(_project_path(args.suite))
    cases = load_frozen_cases(suite)
    canonical_path, report_path = _project_path(args.canonical), _project_path(args.mode_report)
    canonical, canonical_sha256 = load_canonical(canonical_path)
    report = _read_json(report_path)
    if report.get("frozen_suite_id") != suite["suite_id"]:
        raise SystemExit(
            "mode-report regression suite mismatch: re-run retrieval evaluation "
            "with the same --suite before preparing RAGAS data"
        )
    if report.get("frozen_question_sha256") != suite["question_sha256"]:
        raise SystemExit(
            "mode-report question hash mismatch: re-run retrieval evaluation "
            "with the same frozen questions before preparing RAGAS data"
        )
    if report.get("canonical_sha256") != canonical_sha256:
        raise SystemExit("canonical SHA mismatch: re-run retrieval evaluation against this canonical before preparing RAGAS data")
    rows, diagnostics = build_ragas_rows(cases, canonical, report, args.mode)
    output_root = _project_path(args.output_root)
    run_id = f"ragas_retrieval_{report.get('evaluation_run_id', report_path.stem)}_{args.mode}"
    destination = output_root / run_id
    if destination.exists():
        raise SystemExit(f"output already exists: {destination}; preserve it and choose a new output root")
    temporary = output_root / f".{run_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(temporary / "samples.jsonl", rows)
        _write_jsonl(temporary / "diagnostic_cases.jsonl", diagnostics)
        _write_jsonl(temporary / "manual_review_template.jsonl", [{"eval_id": row["eval_id"], "status": "awaiting_ragas_score", "manual_review_required": False, "review_reason": None, "reviewer": None, "reviewed_at": None, "notes": None} for row in rows])
        manifest = {
            "schema_version": "greenspec.ragas_retrieval_dataset.v1", "dataset_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "suite_id": suite["suite_id"], "suite_path": str(_project_path(args.suite).relative_to(ROOT)), "question_sha256": suite["question_sha256"],
            "canonical_path": str(canonical_path.relative_to(ROOT)), "canonical_sha256": canonical_sha256,
            "mode_report_path": str(report_path.relative_to(ROOT)), "mode_report_sha256": sha256_file(report_path),
            "evaluation_run_id": report.get("evaluation_run_id"), "index_manifest_id": report.get("index_manifest_id"), "mode": args.mode,
            "scored_sample_count": len(rows), "diagnostic_case_count": len(diagnostics),
            "scoring_scope": "RAGAS retrieval metrics only; no final-answer or business manual-review-recall claim is made by this dataset.",
            "manual_review_policy": "After scoring, every case below a declared threshold must be reviewed in manual_review_queue.jsonl. Do not approve a run from aggregate averages alone.",
            "files": {"samples": "samples.jsonl", "diagnostics": "diagnostic_cases.jsonl", "manual_review_template": "manual_review_template.jsonl"},
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"dataset": str(destination), "scored_sample_count": len(rows), "diagnostic_case_count": len(diagnostics)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
