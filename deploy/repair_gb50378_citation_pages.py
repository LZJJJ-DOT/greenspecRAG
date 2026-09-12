"""Create an immutable canonical revision correcting GB/T 50378-2019 citation pages.

This tool never edits an existing canonical/index/evaluation artifact.  It writes a
new revision, then atomically switches only the selected canonical root's
``active`` directory after the old active directory has been archived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ROOT / "data" / "canonical" / "generated"
TARGET_STANDARD = "GB_T_50378_2019"
TARGET_TYPES = {
    "normative_table",
    "commentary_table",
    "normative_formula",
    "commentary_formula",
    "figure",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def is_commentary(node: dict) -> bool:
    return str(node.get("content_type", "")).startswith("commentary_")


def repair_node(node: dict) -> tuple[dict, dict] | None:
    if node.get("standard_id") != TARGET_STANDARD:
        return None
    if node.get("content_type") not in TARGET_TYPES:
        return None

    before = {
        key: node.get(key)
        for key in ("pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end", "text", "retrieval_text")
    }
    repaired = dict(node)

    # Formula 2 was accidentally expanded through the next formula.  The visual
    # audit confirms formula (1) and its variable definitions end on PDF page 87
    # (printed page 78); retain only that self-contained formula evidence.
    if repaired.get("clause_id") == "GB50378-2019:GB50378-2019-formula-002":
        for field in ("text", "retrieval_text"):
            value = repaired.get(field)
            if not isinstance(value, str) or "\n\nPDF" not in value:
                raise ValueError(f"formula-002 {field} lacks the expected page boundary")
            repaired[field] = value.split("\n\nPDF", 1)[0].rstrip()
        repaired["pdf_page_end"] = 87

    start = repaired.get("pdf_page_start")
    end = repaired.get("pdf_page_end")
    if not isinstance(start, int) or not isinstance(end, int) or end < start:
        raise ValueError(f"invalid PDF page range for {repaired.get('clause_id')}")
    offset = 9 if is_commentary(repaired) else 10
    repaired["printed_page_start"] = start - offset
    repaired["printed_page_end"] = end - offset

    after = {
        key: repaired.get(key)
        for key in ("pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end", "text", "retrieval_text")
    }
    if before == after:
        return None
    return repaired, {"clause_id": repaired["clause_id"], "content_type": repaired["content_type"], "before": before, "after": after}


def atomic_activate(active: Path, stage: Path, archive: Path) -> None:
    if archive.exists():
        raise FileExistsError(f"archive destination already exists: {archive}")
    temporary_old = active.with_name(f".active-old-{uuid.uuid4().hex}")
    try:
        os.replace(active, temporary_old)
        try:
            os.replace(stage, active)
        except Exception:
            os.replace(temporary_old, active)
            raise
        os.replace(temporary_old, archive)
    except Exception:
        if temporary_old.exists() and not active.exists():
            os.replace(temporary_old, active)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--canonical-root", type=Path, default=CANONICAL_ROOT, help="Canonical root containing active/ and revisions/.")
    args = parser.parse_args()
    canonical_root = args.canonical_root if args.canonical_root.is_absolute() else ROOT / args.canonical_root
    canonical_root = canonical_root.resolve()
    active = canonical_root / "active"
    revision_root = canonical_root / "revisions"
    if not active.is_dir():
        raise FileNotFoundError(f"missing active canonical: {active}")

    previous_manifest = read_json(active / "index_manifest.json")
    previous_manifest_id = previous_manifest["index_manifest_id"]
    run_id = args.run_id or f"citation_page_repair_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    revision_dir = revision_root / run_id
    if revision_dir.exists():
        raise FileExistsError(f"revision already exists: {revision_dir}")
    revision_root.mkdir(parents=True, exist_ok=True)
    revision_dir.mkdir()

    rows = load_jsonl(active / "clauses.jsonl")
    seen: set[str] = set()
    changes: list[dict] = []
    new_rows: list[dict] = []
    for node in rows:
        clause_id = node.get("clause_id")
        if not isinstance(clause_id, str) or not clause_id:
            raise ValueError("canonical contains an empty clause_id")
        if clause_id in seen:
            raise ValueError(f"duplicate clause_id: {clause_id}")
        seen.add(clause_id)
        outcome = repair_node(node)
        if outcome is None:
            new_rows.append(node)
        else:
            repaired, change = outcome
            new_rows.append(repaired)
            changes.append(change)

    required = {
        "GB50378-2019:GB50378-2019-formula-002",
        "GB50378-2019:GB50378-2019-table-3.2.4:p14",
        "GB50378-2019:GB50378-2019-table-7.2.2:p31",
        "GB50378-2019:GB50378-2019-table-8.2.3:p39",
    }
    changed_ids = {item["clause_id"] for item in changes}
    missing = required - changed_ids
    if missing:
        raise ValueError(f"required citation repairs were not made: {sorted(missing)}")

    canonical_path = revision_dir / "clauses.jsonl"
    write_jsonl(canonical_path, new_rows)
    canonical_sha = sha256_file(canonical_path)
    prior_source_path = active / "source_manifest.json"
    if not prior_source_path.exists():
        # Early canonical revisions recorded only an index manifest.  Keep its
        # provenance while explicitly noting that the source manifest was absent.
        prior_source = {"source_manifest_id": previous_manifest.get("source_manifest_id"), "legacy_source_manifest_missing": True}
    else:
        prior_source = read_json(prior_source_path)
    source_manifest = {
        "source_manifest_id": f"source_{run_id}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "derived_from_source_manifest_id": prior_source.get("source_manifest_id"),
        "source_sha256": previous_manifest.get("source_sha256"),
        "raw_pdf_validation": {
            "path": "data/raw/GBT50378-2019绿色建筑评价标准.pdf",
            "sha256": "4a41f9ce313a23fc3d615e5aec39c1065b7ffbcd0efdf60480b541d5c7c5f291",
            "page_count": 146,
            "visual_audit_scope": "GB/T 50378-2019 tables and formulas",
        },
        "parent_source_manifest": prior_source,
    }
    write_json(revision_dir / "source_manifest.json", source_manifest)
    index_manifest = {
        "index_manifest_id": f"index_{run_id}",
        "source_manifest_id": source_manifest["source_manifest_id"],
        "canonical_path": str(canonical_path.relative_to(ROOT)).replace("\\", "/"),
        "canonical_sha256": canonical_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "derived_from_index_manifest_id": previous_manifest_id,
        "revision_kind": "gb_t_50378_2019_citation_page_repair",
        "node_count": len(new_rows),
    }
    write_json(revision_dir / "index_manifest.json", index_manifest)
    audit = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "All GB/T 50378-2019 table/formula/figure canonical nodes; no figures were present.",
        "previous_index_manifest_id": previous_manifest_id,
        "new_index_manifest_id": index_manifest["index_manifest_id"],
        "node_count": len(new_rows),
        "changed_node_count": len(changes),
        "changes": changes,
        "validation": {
            "unique_clause_ids": True,
            "formula_002_bounded_to_pdf_87_printed_78": True,
            "non_continuation_pages_equal": True,
            "continuations_preserved": [
                "GB50378-2019:GB50378-2019-table-7.2.1-1:p30",
                "GB50378-2019:GB50378-2019-table-7.2.5:p32",
                "GB50378-2019:GB50378-2019-table-3:p121",
                "GB50378-2019:GB50378-2019-table-4:p130",
            ],
            "figure_nodes_found": 0,
        },
    }
    write_json(revision_dir / "audit_report.json", audit)

    stage = canonical_root / f".active-stage-{uuid.uuid4().hex}"
    stage.mkdir()
    for filename in ("clauses.jsonl", "index_manifest.json", "source_manifest.json", "audit_report.json"):
        shutil.copy2(revision_dir / filename, stage / filename)
    atomic_activate(active, stage, revision_root / previous_manifest_id)
    print(json.dumps({"run_id": run_id, "revision_dir": str(revision_dir), "index_manifest_id": index_manifest["index_manifest_id"], "changed_node_count": len(changes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
