"""Carry forward human citation annotations only across semantically identical Evidence Packs."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


ITEM_FIELDS = {
    "evidence_id", "clause_id", "parent_id", "content_type", "text", "parent_context",
    "standard_id", "standard_name", "standard_version", "clause_no", "pdf_page_start",
    "pdf_page_end", "printed_page_start", "printed_page_end", "source_file", "source_sha256",
    "missing_facts", "table_id", "formula_id",
}
CITATION_FIELDS = {
    "evidence_id", "standard_number", "standard_name", "standard_version", "clause_no",
    "pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end", "table_id", "formula_id",
}
CARRIED_FIELDS = ("citation_accuracy", "citation_completeness", "evidence_pack_supports_question", "annotator", "annotated_at", "notes")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fingerprint(pack: dict) -> str:
    value = {
        "items": [{key: item.get(key) for key in sorted(ITEM_FIELDS)} for item in pack["items"]],
        "citations": [{key: citation.get(key) for key in sorted(CITATION_FIELDS)} for citation in pack["citations"]],
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Carry forward human Evidence Pack citation annotations after exact semantic comparison.")
    parser.add_argument("--source-review-run", type=Path, required=True)
    parser.add_argument("--target-review-run", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="carry only exact-match packs and leave changed packs unannotated")
    parser.add_argument("--exclude-eval-id", action="append", default=[], help="leave an exact-match pack unannotated for renewed review")
    args = parser.parse_args()

    source_manifest = _json(args.source_review_run / "review_manifest.json")
    target_manifest = _json(args.target_review_run / "review_manifest.json")
    source_rows = {row["eval_id"]: row for row in _jsonl(args.source_review_run / source_manifest["annotation_file"])}
    target_path = args.target_review_run / target_manifest["annotation_file"]
    target_rows = _jsonl(target_path)
    target_ids = {row["eval_id"] for row in target_rows}
    if set(source_rows) != target_ids:
        raise ValueError("source and target review runs do not contain the same evaluation IDs")

    excluded = set(args.exclude_eval_id)
    changed = []
    for eval_id in sorted(target_ids):
        source_pack = _json(args.source_review_run / source_rows[eval_id]["evidence_pack_file"])
        target_row = next(row for row in target_rows if row["eval_id"] == eval_id)
        target_pack = _json(args.target_review_run / target_row["evidence_pack_file"])
        if _fingerprint(source_pack) != _fingerprint(target_pack):
            changed.append(eval_id)
    if changed and not args.allow_partial:
        raise ValueError(f"cannot carry annotations: reviewable Evidence Pack content changed for {changed}")

    carried = []
    for target_row in target_rows:
        if target_row["eval_id"] in changed or target_row["eval_id"] in excluded:
            continue
        source_annotation = source_rows[target_row["eval_id"]]["annotation"]
        target_annotation = target_row["annotation"]
        for field in CARRIED_FIELDS:
            target_annotation[field] = source_annotation.get(field)
        # Source verification may legitimately change whether a pack itself flags
        # manual review. Do not overwrite that field with an older review result.
        target_row["review_provenance"] = {
            "kind": "carried_forward_equivalent_evidence_pack_review",
            "source_review_run_id": source_manifest["review_run_id"],
            "carried_at": datetime.now(timezone.utc).isoformat(),
            "comparison": "items and citations exact after excluding trace and verification metadata",
        }
        carried.append(target_row["eval_id"])

    temporary = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in target_rows), encoding="utf-8")
    os.replace(temporary, target_path)
    print(json.dumps({"target_review_run_id": target_manifest["review_run_id"], "carried_eval_ids": carried, "changed_eval_ids": changed, "excluded_eval_ids": sorted(excluded), "source_review_run_id": source_manifest["review_run_id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
