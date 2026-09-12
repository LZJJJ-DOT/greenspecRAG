"""Create, but never overwrite, a reviewer-owned required-evidence template."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _questions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a non-destructive template for required evidence annotation.")
    parser.add_argument("--suite", type=Path, default=Path("data/evaluation/regression_suite.v2.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.v2.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approve-legacy-single-gold",
        action="store_true",
        help="Approve each positive case with exactly one legacy gold ID as one required canonical evidence item.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite reviewer-owned template: {args.output}")

    suite = json.loads(args.suite.read_text(encoding="utf-8-sig"))
    annotations = []
    diagnostics = []
    for question in _questions(args.questions):
        gold = [str(value) for value in question.get("gold_evidence_ids", [])]
        required = question.get("required_evidence", []) or []
        if gold:
            requirements = []
            status = "needs_manual_annotation"
            if required and len(required) == len(gold):
                status = "approved"
                requirements = [
                    {
                        "requirement_id": f"REQ-{index:03d}",
                        "kind": requirement.get("kind"),
                        "value": requirement.get("value"),
                        "acceptable_evidence_ids": [gold[index - 1]],
                    }
                    for index, requirement in enumerate(required, start=1)
                ]
            elif args.approve_legacy_single_gold and len(gold) == 1:
                status = "approved"
                requirements = [
                    {
                        "requirement_id": "REQ-001",
                        "kind": "canonical_evidence_id",
                        "value": gold[0],
                        "acceptable_evidence_ids": [gold[0]],
                    }
                ]
            annotations.append(
                {
                    "eval_id": question["eval_id"],
                    "question": question["question"],
                    "status": status,
                    "requirements": requirements,
                    "legacy_gold_evidence_ids": gold,
                    "reviewer": None,
                    "reviewed_at": None,
                    "notes": (
                        "Existing explicit one-to-one requirement preserved."
                        if required else
                        "User-approved: the legacy single gold ID is the unique required evidence for this case."
                        if status == "approved" else
                        "Fill every indispensable clause/table/formula and all acceptable canonical evidence IDs; do not promote legacy gold automatically."
                    ),
                }
            )
        else:
            diagnostics.append(
                {
                    "eval_id": question["eval_id"],
                    "question": question["question"],
                    "expected_answer_type": question.get("expected_answer_type"),
                    "manual_review_reason": question.get("manual_review_reason"),
                    "reviewer": None,
                    "reviewed_at": None,
                    "expected_safe_behavior": None,
                    "notes": "Use this row to label a difficult negative, ambiguity, version-boundary, or missing-fact behavior; do not invent a normative answer.",
                }
            )

    payload = {
        "schema_version": "greenspec.required_evidence_annotation_template.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": suite.get("suite_id"),
        "frozen_inputs": {
            "suite_path": str(args.suite),
            "suite_sha256": _sha256(args.suite),
            "questions_path": str(args.questions),
            "questions_sha256": _sha256(args.questions),
        },
        "annotation_policy": {
            "required_evidence": "Each requirement must list one or more canonical evidence IDs that satisfy it.",
            "status": ["approved", "needs_manual_annotation"],
            "release_rule": "Only approved annotations count toward Required-evidence Recall.",
        },
        "annotations": annotations,
        "diagnostic_case_template": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "positive_annotations": len(annotations), "approved": sum(row["status"] == "approved" for row in annotations), "diagnostic_templates": len(diagnostics)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
