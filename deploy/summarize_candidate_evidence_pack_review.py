"""Summarize completed candidate EvidencePack human reviews."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = ("citation_accuracy", "citation_completeness", "evidence_pack_supports_question")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(review_run: Path) -> dict[str, Any]:
    manifest = json.loads((review_run / "review_manifest.json").read_text(encoding="utf-8"))
    rows = _rows(review_run / manifest["files"]["annotations"])
    expected_ids = {case["candidate_id"] for case in manifest["cases"]}
    observed_ids = [row.get("candidate_id") for row in rows]
    if len(rows) != manifest["case_count"] or set(observed_ids) != expected_ids or len(set(observed_ids)) != len(observed_ids):
        raise ValueError("review annotations do not match the immutable review manifest")
    if any(row.get("review_run_id") != manifest["review_run_id"] or row.get("index_manifest_id") != manifest["index_manifest_id"] for row in rows):
        raise ValueError("review annotations are bound to a different run or manifest")
    metrics: dict[str, dict[str, Any]] = {}
    for field in METRICS:
        labeled = [row for row in rows if isinstance((row.get("annotation") or {}).get(field), bool)]
        passed = [row for row in labeled if row["annotation"][field]]
        metrics[field] = {
            "passed": len(passed),
            "labeled": len(labeled),
            "expected": len(rows),
            "coverage": f"{len(labeled)}/{len(rows)}",
            "rate": len(passed) / len(labeled) if labeled else None,
            "failed_candidate_ids": [row["candidate_id"] for row in labeled if not row["annotation"][field]],
        }
    fully_reviewed = all(metric["labeled"] == len(rows) for metric in metrics.values())
    return {
        "schema_version": "greenspec.candidate_evidence_pack_review_summary.v1",
        "review_run_id": manifest["review_run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_manifest_id": manifest["index_manifest_id"],
        "case_count": len(rows),
        "metrics": metrics,
        "publication_status": "complete_human_review" if fully_reviewed else "pending_human_review",
        "unreviewed_candidate_ids": sorted(row["candidate_id"] for row in rows if any(not isinstance((row.get("annotation") or {}).get(field), bool) for field in METRICS)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize candidate EvidencePack review annotations.")
    parser.add_argument("--review-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/candidate_evidence_pack_review_summaries"))
    args = parser.parse_args()
    report = summarize(args.review_run)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{report['review_run_id']}.json"
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "publication_status": report["publication_status"], "metrics": report["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
