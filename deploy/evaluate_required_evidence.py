"""Score frozen retrieval traces without issuing another retrieval request.

The report keeps two deliberately separate measurements:

* ID-based Context Recall and MRR use the existing ``gold_evidence_ids``;
* Required-evidence Recall uses only requirements explicitly approved in the
  companion annotation file.  Legacy one-ID gold is never silently promoted
  to an "all required evidence" label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HIT_AT = 10


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _questions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("eval_id"):
            rows[str(row["eval_id"])] = row
    return rows


def _annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    return {
        str(row["eval_id"]): row
        for row in payload.get("annotations", [])
        if row.get("eval_id")
    }


def _ranked_ids(case: dict[str, Any]) -> list[str]:
    trace = case.get("modes", {}).get("hybrid", {}).get("retrieval_trace", [])
    return [
        str(item["evidence_id"])
        for item in trace[:HIT_AT]
        if item.get("evidence_id")
    ]


def _approved_requirements(
        question: dict[str, Any],
        annotation: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    """Return only manually approved requirements.

    Five frozen v2 questions already contain one explicit ``required_evidence``
    entry and one exact gold ID.  That one-to-one mapping is preserved as a
    legacy explicit label; all other cases remain unlabelled until a reviewer
    supplies accepted evidence IDs in the companion file.
    """
    if annotation and annotation.get("status") == "approved":
        requirements = annotation.get("requirements", [])
        if isinstance(requirements, list):
            return requirements, "approved_annotation"

    required = question.get("required_evidence", []) or []
    gold = question.get("gold_evidence_ids", []) or []
    if required and len(required) == len(gold):
        return [
            {
                "requirement_id": f"REQ-{index:03d}",
                "kind": source.get("kind"),
                "value": source.get("value"),
                "acceptable_evidence_ids": [str(gold[index - 1])],
            }
            for index, source in enumerate(required, start=1)
        ], "legacy_explicit_one_to_one"

    return [], "needs_manual_annotation"


def _reciprocal_rank(ranked_ids: list[str], accepted_ids: set[str]) -> float:
    for rank, evidence_id in enumerate(ranked_ids, start=1):
        if evidence_id in accepted_ids:
            return 1.0 / rank
    return 0.0


def build_report(
        *,
        suite_path: Path,
        questions_path: Path,
        mode_report_path: Path,
        annotations_path: Path | None,
) -> dict[str, Any]:
    suite = _load_json(suite_path)
    questions = _questions(questions_path)
    mode_report = _load_json(mode_report_path)
    annotations = _annotations(annotations_path)

    cases = mode_report.get("cases", [])
    rows = []
    gold_total = gold_retrieved = 0
    gold_mrr_values: list[float] = []
    required_total = required_retrieved = 0
    required_case_count = required_case_complete = 0
    required_mrr_values: list[float] = []

    for mode_case in cases:
        eval_id = str(mode_case.get("eval_id", ""))
        question = questions.get(eval_id)
        if not question:
            raise ValueError(f"mode report case {eval_id!r} is absent from questions")

        ranked_ids = _ranked_ids(mode_case)
        gold_ids = [str(value) for value in question.get("gold_evidence_ids", [])]
        gold_set = set(gold_ids)
        gold_hits = sorted(gold_set.intersection(ranked_ids))

        if gold_ids:
            gold_total += len(gold_set)
            gold_retrieved += len(gold_hits)
            gold_mrr_values.append(_reciprocal_rank(ranked_ids, gold_set))

        requirements, label_source = _approved_requirements(
            question,
            annotations.get(eval_id),
        )
        requirement_rows = []
        if requirements:
            required_case_count += 1
            case_complete = True
            first_required_ranks: list[int] = []
            for requirement in requirements:
                accepted = {
                    str(value)
                    for value in requirement.get("acceptable_evidence_ids", [])
                    if value
                }
                if not accepted:
                    raise ValueError(
                        f"{eval_id} has an approved requirement without acceptable_evidence_ids"
                    )
                matching_ranks = [
                    rank
                    for rank, evidence_id in enumerate(ranked_ids, start=1)
                    if evidence_id in accepted
                ]
                hit = bool(matching_ranks)
                required_total += 1
                required_retrieved += int(hit)
                case_complete = case_complete and hit
                if matching_ranks:
                    first_required_ranks.append(min(matching_ranks))
                requirement_rows.append(
                    {
                        "requirement_id": requirement.get("requirement_id"),
                        "kind": requirement.get("kind"),
                        "value": requirement.get("value"),
                        "acceptable_evidence_ids": sorted(accepted),
                        "hit_at_10": hit,
                        "first_rank": min(matching_ranks) if matching_ranks else None,
                    }
                )
            required_case_complete += int(case_complete)
            required_mrr_values.append(
                1.0 / min(first_required_ranks) if first_required_ranks else 0.0
            )

        rows.append(
            {
                "eval_id": eval_id,
                "ranked_evidence_ids": ranked_ids,
                "gold_evidence_ids": sorted(gold_set),
                "id_context_recall_hit_ids": gold_hits,
                "first_gold_rank": next(
                    (rank for rank, value in enumerate(ranked_ids, start=1) if value in gold_set),
                    None,
                ),
                "required_evidence_label_source": label_source,
                "required_evidence": requirement_rows,
            }
        )

    positive_case_count = sum(
        1 for question in questions.values() if question.get("gold_evidence_ids")
    )
    annotation_coverage = required_case_count / positive_case_count if positive_case_count else None
    return {
        "schema_version": "greenspec.required_evidence_metrics.v1",
        "evaluation_run_id": f"required_evidence_{uuid.uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Frozen hybrid top-10 trace only. ID metrics use legacy gold IDs; "
            "Required-evidence metrics include only explicitly approved requirements."
        ),
        "frozen_inputs": {
            "suite_id": suite.get("suite_id"),
            "suite_path": str(suite_path),
            "suite_sha256": _sha256(suite_path),
            "questions_path": str(questions_path),
            "questions_sha256": _sha256(questions_path),
            "mode_report_path": str(mode_report_path),
            "mode_report_sha256": _sha256(mode_report_path),
            "index_manifest_id": mode_report.get("index_manifest_id"),
            "hybrid_manifest_path": mode_report.get("hybrid_manifest_path"),
            "canonical_sha256": mode_report.get("canonical_sha256"),
            "annotations_path": str(annotations_path) if annotations_path else None,
            "annotations_sha256": _sha256(annotations_path) if annotations_path else None,
        },
        "coverage": {
            "positive_cases": positive_case_count,
            "id_gold_cases": len(gold_mrr_values),
            "required_evidence_cases": required_case_count,
            "required_evidence_annotation_rate": annotation_coverage,
        },
        "metrics": {
            "id_based_context_recall_at_10": {
                "retrieved_reference_ids": gold_retrieved,
                "reference_ids": gold_total,
                "rate": gold_retrieved / gold_total if gold_total else None,
            },
            "mrr_at_10_first_gold": {
                "cases": len(gold_mrr_values),
                "mean": sum(gold_mrr_values) / len(gold_mrr_values) if gold_mrr_values else None,
            },
            "required_evidence_recall_at_10": {
                "retrieved_requirements": required_retrieved,
                "requirements": required_total,
                "rate": required_retrieved / required_total if required_total else None,
                "complete_cases": required_case_complete,
                "cases": required_case_count,
            },
            "mrr_at_10_first_required_evidence": {
                "cases": len(required_mrr_values),
                "mean": sum(required_mrr_values) / len(required_mrr_values)
                if required_mrr_values else None,
            },
        },
        "publication_status": (
            "eligible_for_required_evidence_gate_review"
            if annotation_coverage == 1.0
            else "experimental_pending_required_evidence_annotation"
        ),
        "unresolved": (
            [] if annotation_coverage == 1.0 else [
                "Some positive cases have only legacy gold_evidence_ids. "
                "Annotate all required evidence before treating Required-evidence Recall as a release gate."
            ]
        ),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score ID recall, required-evidence recall, and MRR from a frozen mode report.")
    parser.add_argument("--suite", type=Path, default=Path("data/evaluation/regression_suite.v2.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.v2.jsonl"))
    parser.add_argument("--mode-report", type=Path, required=True)
    parser.add_argument("--required-annotations", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/required_evidence_runs"))
    args = parser.parse_args()

    report = build_report(
        suite_path=args.suite,
        questions_path=args.questions,
        mode_report_path=args.mode_report,
        annotations_path=args.required_annotations,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{report['evaluation_run_id']}.json"
    temporary = output.with_name(f".{output.name}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "metrics": report["metrics"], "coverage": report["coverage"], "publication_status": report["publication_status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
