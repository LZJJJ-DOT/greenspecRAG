"""Summarize run-bound Evidence Pack human reviews without changing annotations."""
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


ACCURACY_THRESHOLD = 0.95
COMPLETENESS_THRESHOLD = 0.90
HIT_THRESHOLD = 0.85


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rate(rows: list[dict], field: str) -> tuple[float | None, int, int, list[str]]:
    labeled = [row for row in rows if isinstance(row["annotation"].get(field), bool)]
    passed = [row for row in labeled if row["annotation"][field]]
    return (len(passed) / len(labeled) if labeled else None, len(passed), len(labeled), [row["eval_id"] for row in labeled if not row["annotation"][field]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize immutable Evidence Pack review annotations.")
    parser.add_argument("--review-run", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/evidence_pack_gate_runs"))
    args = parser.parse_args()

    review_manifest = json.loads((args.review_run / "review_manifest.json").read_text(encoding="utf-8"))
    rows = _load_jsonl(args.review_run / review_manifest["annotation_file"])
    retrieval = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
    expected = review_manifest["case_count"]
    errors = []
    if len(rows) != expected or len({row.get("eval_id") for row in rows}) != expected:
        errors.append("annotation count or eval_id uniqueness does not match the review manifest")
    for row in rows:
        if row.get("review_run_id") != review_manifest["review_run_id"]:
            errors.append(f"{row.get('eval_id')}: review_run_id mismatch")
        if row.get("index_manifest_id") != review_manifest["index_manifest_id"]:
            errors.append(f"{row.get('eval_id')}: index_manifest_id mismatch")
        if row.get("review_target") != "evidence_pack":
            errors.append(f"{row.get('eval_id')}: unsupported review target")
    if retrieval.get("index_manifest_id") != review_manifest["index_manifest_id"]:
        errors.append("retrieval report and review run use different index manifests")

    accuracy_rate, accuracy_passed, accuracy_labeled, accuracy_failed = _rate(rows, "citation_accuracy")
    completeness_rate, completeness_passed, completeness_labeled, completeness_failed = _rate(rows, "citation_completeness")
    supports_rate, supports_passed, supports_labeled, supports_failed = _rate(rows, "evidence_pack_supports_question")
    manual_rate, manual_passed, manual_labeled, manual_failed = _rate(rows, "needs_manual_review")
    # Do not confuse "错误召回" (an irrelevant candidate) with an incorrect
    # citation locator.  Only explicit page-locator defects are a Citation
    # Accuracy consistency warning.
    locator_warning = re.compile(
        r"printed_page(?:_start|_end)?[^，。；]{0,16}(?:出现|存在|有)?[^，。；]{0,8}(?:问题|错误)"
        r"|(?:PDF|印刷)?页码(?:出现|存在|有)[^，。；]{0,8}(?:问题|错误)",
        re.I,
    )
    consistency_warnings = [
        {
            "eval_id": row["eval_id"],
            "field": "citation_accuracy",
            "annotation": True,
            "reason": "notes describe a page-locator problem; verify the boolean against the frozen Accuracy rubric",
        }
        for row in rows
        if row["annotation"].get("citation_accuracy") is True and locator_warning.search(str(row["annotation"].get("notes") or ""))
    ]
    retrieval_hybrid = retrieval.get("hit_at_10", {}).get("hybrid", {})
    retrieval_gate = bool(retrieval.get("retrieval_gate_passed")) and float(retrieval_hybrid.get("rate", 0)) >= HIT_THRESHOLD
    annotation_coverage_passed = accuracy_labeled == expected and completeness_labeled == expected
    support_coverage_passed = supports_labeled == expected
    support_gate_passed = support_coverage_passed and supports_passed == expected
    human_gate = (
        annotation_coverage_passed
        and accuracy_rate is not None
        and completeness_rate is not None
        and accuracy_rate >= ACCURACY_THRESHOLD
        and completeness_rate >= COMPLETENESS_THRESHOLD
    )
    baseline_eligible = not errors and not consistency_warnings and retrieval_gate and human_gate and support_gate_passed
    report = {
        "gate_run_id": f"evidence_pack_gate_{uuid.uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_target": "evidence_pack",
        "review_run_id": review_manifest["review_run_id"],
        "index_manifest_id": review_manifest["index_manifest_id"],
        "retrieval_report_path": str(args.retrieval_report),
        "case_count": expected,
        "metrics": {
            "clause_hit_at_10_hybrid": retrieval_hybrid,
            "citation_accuracy": {"rate": accuracy_rate, "passed": accuracy_passed, "labeled": accuracy_labeled, "failed_eval_ids": accuracy_failed, "threshold": ACCURACY_THRESHOLD},
            "citation_completeness": {"rate": completeness_rate, "passed": completeness_passed, "labeled": completeness_labeled, "failed_eval_ids": completeness_failed, "threshold": COMPLETENESS_THRESHOLD},
            "citation_annotation_coverage": {"expected": expected, "accuracy_labeled": accuracy_labeled, "completeness_labeled": completeness_labeled, "passed": annotation_coverage_passed},
            "evidence_pack_supports_question": {"rate": supports_rate, "passed": supports_passed, "labeled": supports_labeled, "failed_eval_ids": supports_failed},
            "evidence_pack_support_coverage": {"expected": expected, "labeled": supports_labeled, "passed": support_coverage_passed},
            "needs_manual_review": {"rate": manual_rate, "passed": manual_passed, "labeled": manual_labeled, "false_eval_ids": manual_failed},
        },
        "retrieval_gate_passed": retrieval_gate,
        "human_citation_gate_passed": human_gate,
        "annotation_consistency_warnings": consistency_warnings,
        "validation_errors": errors,
        "publication_status": "baseline_eligible" if baseline_eligible else "experimental",
        "unresolved": (
            ([] if retrieval_gate else ["hybrid retrieval gate is not satisfied"])
            + ([] if annotation_coverage_passed else ["Citation Accuracy and Citation Completeness must be labeled for every Evidence Pack in the review run"])
            + ([] if human_gate else ["Citation Accuracy and/or Citation Completeness threshold is not satisfied"])
            + ([] if support_gate_passed else ["Every reviewed Evidence Pack must support its frozen question before baseline publication"])
            + (["annotation notes conflict with Citation Accuracy booleans; resolve before publication"] if consistency_warnings else [])
            + errors
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{report['gate_run_id']}.json"
    temporary = output.with_name(f".{output.name}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"gate_run_id": report["gate_run_id"], "report_path": str(output), "publication_status": report["publication_status"], "metrics": report["metrics"], "consistency_warning_count": len(consistency_warnings)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
