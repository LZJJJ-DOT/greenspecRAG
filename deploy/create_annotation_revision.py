"""Create an immutable corrected review-annotation revision."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "evaluation" / "evidence_pack_review_runs" / "pack_review_db17753b26774cd8bc9947b2517766e2" / "review_annotations.jsonl"
OUTPUT_ROOT = ROOT / "data" / "evaluation" / "evidence_pack_annotation_revisions"
ACCURACY_FALSE_COMPLETENESS_TRUE = {"E006", "E014", "E015", *{f"E{i:03d}" for i in range(18, 27)}}
COMPLETENESS_FALSE = {"E029", "E031"}


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--revision-id", default=None)
    args = parser.parse_args()
    revision_id = args.revision_id or f"annotation_correction_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    output = OUTPUT_ROOT / revision_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    changes: list[dict] = []
    new_rows: list[dict] = []
    for row in rows:
        item = dict(row)
        evidence_id = item.get("eval_id")
        before = dict(item)
        annotation = dict(item.get("annotation") or {})
        if evidence_id in ACCURACY_FALSE_COMPLETENESS_TRUE:
            annotation["citation_accuracy"] = False
            annotation["citation_completeness"] = True
        if evidence_id in COMPLETENESS_FALSE:
            annotation["citation_completeness"] = False
        if ",annotated_at" in annotation:
            annotation["annotated_at"] = annotation.pop(",annotated_at")
        item["annotation"] = annotation
        if item != before:
            changed = sorted(key for key in set(before) | set(item) if before.get(key) != item.get(key))
            changes.append({"evidence_id": evidence_id, "fields": changed})
        new_rows.append(item)
    expected = ACCURACY_FALSE_COMPLETENESS_TRUE | COMPLETENESS_FALSE | {"E012"}
    found = {entry["evidence_id"] for entry in changes}
    missing = expected - found
    if missing:
        raise ValueError(f"expected corrections missing: {sorted(missing)}")
    output_file = output / "review_annotations.jsonl"
    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metric = lambda key: sum(bool(row["annotation"].get(key)) for row in new_rows) / len(new_rows)
    manifest = {
        "annotation_revision_id": revision_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_review_annotations": str(args.source.relative_to(ROOT)).replace("\\", "/"),
        "parent_review_run_id": "pack_review_db17753b26774cd8bc9947b2517766e2",
        "scope": "Corrected historical adjudication. It does not certify evidence packs regenerated from a later canonical revision.",
        "annotation_count": len(new_rows),
        "changes": changes,
        "metrics": {
            "citation_accuracy": metric("citation_accuracy"),
            "citation_completeness": metric("citation_completeness"),
            "evidence_pack_supports_question": metric("evidence_pack_supports_question"),
        },
    }
    write_json(output / "annotation_revision_manifest.json", manifest)
    print(json.dumps({"annotation_revision_id": revision_id, "path": str(output), "metrics": manifest["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
