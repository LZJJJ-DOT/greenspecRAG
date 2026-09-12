"""Promote source-checked candidate questions into an auditable eval set.

This tool intentionally does not mutate the candidate pool.  It writes an
accepted-only JSONL plus a full review ledger, so rejected questions retain
their evidence-backed reason for exclusion.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REJECTED: dict[str, str] = {
    "C044": "Underspecified: satisfying all control items does not uniquely determine a green-building grade; score and table conditions are also required.",
    "C086": "Incorrect comparison target: the cited shallow buried-pipe clause requires a pressure test, while recharge testing belongs to groundwater-source wells.",
    "C087": "Incorrect gold evidence: the cited groundwater-source-well clause requires pumping/recharge tests, not a water-pressure test.",
}

CORRECTIONS: dict[str, dict[str, str]] = {
    "C054": {"category": "table", "expected_content_type": "normative_table"},
    "C066": {"category": "threshold", "expected_content_type": "normative_clause"},
    "C067": {"category": "single_clause", "expected_content_type": "normative_clause"},
    "C068": {"category": "single_clause", "expected_content_type": "normative_clause"},
    "C069": {"category": "single_clause", "expected_content_type": "normative_clause"},
    # These two Green Building Standard tables are stored as commentary_table
    # nodes in the current frozen canonical; retain the source fact and expose
    # the actual canonical type instead of rejecting a valid cross-page case.
    "C059": {"expected_content_type": "commentary_table"},
    "C060": {"expected_content_type": "commentary_table"},
    "C061": {"expected_content_type": "commentary_table"},
    "C062": {"expected_content_type": "commentary_table"},
}

ADVERSARIAL_NEAR_MISSES: dict[str, list[str]] = {
    "C088": ["GB50378-2019:9.2.8"],
    "C089": ["GB50378-2019:9.2.7"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--accepted-output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--reviewer", default="liang")
    args = parser.parse_args()

    nodes = {row["clause_id"]: row for row in read_jsonl(args.canonical)}
    reviewed_at = datetime.now(timezone.utc).isoformat()
    accepted: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for source in read_jsonl(args.candidates):
        candidate = dict(source)
        candidate.update(CORRECTIONS.get(candidate["candidate_id"], {}))
        gold_ids = list(candidate.get("gold_evidence_ids") or [])
        supporting_ids = list(candidate.get("supporting_evidence_ids") or [])
        all_ids = gold_ids + supporting_ids
        missing_ids = [evidence_id for evidence_id in all_ids if evidence_id not in nodes]
        rejection = REJECTED.get(candidate["candidate_id"])
        if missing_ids:
            rejection = f"Missing canonical evidence IDs: {', '.join(missing_ids)}."

        is_negative = candidate.get("category") == "insufficient_evidence"
        if not is_negative and not gold_ids:
            rejection = "Positive retrieval candidate has no gold evidence."

        verdict = "rejected" if rejection else "accepted"
        review = {
            "schema_version": "greenspec.candidate_review.v1",
            "candidate_id": candidate["candidate_id"],
            "status": verdict,
            "reviewer": args.reviewer,
            "reviewed_at": reviewed_at,
            "canonical_evidence_ids_checked": all_ids,
            "notes": rejection or "Question, gold IDs, canonical existence, content type, and structural evidence relations checked; accepted for frozen-manifest retrieval evaluation.",
        }
        review_rows.append(review)
        if rejection:
            continue

        required_ids = [] if is_negative else list(dict.fromkeys(gold_ids + supporting_ids))
        candidate.update({
            "schema_version": "greenspec.accepted_candidate.v1",
            "source_candidate_id": candidate["candidate_id"],
            "accepted": True,
            "required_evidence_ids": required_ids,
            "required_evidence": list(candidate.get("suggested_required_evidence") or []),
            "canonical_content_types": {
                evidence_id: nodes[evidence_id].get("content_type")
                for evidence_id in all_ids
                if evidence_id in nodes
            },
            "forbidden_near_misses": list(dict.fromkeys(candidate.get("forbidden_near_misses") or ADVERSARIAL_NEAR_MISSES.get(candidate["candidate_id"], []))),
            "review": review,
        })
        accepted.append(candidate)

    write_jsonl(args.accepted_output, accepted)
    write_jsonl(args.review_output, review_rows)
    print(json.dumps({"accepted": len(accepted), "rejected": len(review_rows) - len(accepted), "accepted_output": str(args.accepted_output), "review_output": str(args.review_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
