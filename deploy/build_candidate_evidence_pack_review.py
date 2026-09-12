"""Build an immutable human-review queue from frozen candidate EvidencePacks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANNOTATION = {
    "citation_accuracy": None,
    "citation_completeness": None,
    "evidence_pack_supports_question": None,
    "reviewer": None,
    "reviewed_at": None,
    "notes": None,
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_review_run(*, accepted_path: Path, retrievals_path: Path, output_root: Path) -> Path:
    accepted, retrievals = _rows(accepted_path), _rows(retrievals_path)
    accepted_by_id = {row.get("candidate_id"): row for row in accepted}
    retrieval_by_id = {row.get("candidate_id"): row for row in retrievals}
    if len(accepted_by_id) != len(accepted) or len(retrieval_by_id) != len(retrievals) or set(accepted_by_id) != set(retrieval_by_id):
        raise ValueError("accepted candidates and retrievals must have identical unique candidate_id sets")
    missing_packs = [candidate_id for candidate_id, row in retrieval_by_id.items() if not isinstance(row.get("evidence_pack"), dict)]
    errors = [candidate_id for candidate_id, row in retrieval_by_id.items() if row.get("retrieval", {}).get("error")]
    manifests = {row.get("retrieval", {}).get("index_manifest_id") for row in retrievals}
    if missing_packs or errors or len(manifests) != 1 or None in manifests:
        raise ValueError(f"frozen retrievals need one complete manifest; missing_packs={missing_packs}, errors={errors}, manifests={sorted(manifests, key=str)}")

    run_id = f"candidate_pack_review_{uuid.uuid4().hex}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=output_root))
    try:
        pack_dir = temporary / "evidence_packs"
        pack_dir.mkdir()
        annotations: list[dict[str, Any]] = []
        cases: list[dict[str, Any]] = []
        for candidate_id in sorted(accepted_by_id):
            candidate, retrieval = accepted_by_id[candidate_id], retrieval_by_id[candidate_id]
            pack = retrieval["evidence_pack"]
            pack_file = pack_dir / f"{candidate_id}.json"
            pack_file.write_text(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            case = {
                "candidate_id": candidate_id,
                "question": candidate["question"],
                "category": candidate.get("category"),
                "required_evidence_ids": candidate.get("required_evidence_ids", []),
                "forbidden_near_misses": candidate.get("forbidden_near_misses", []),
                "evidence_pack_file": str(pack_file.relative_to(temporary)).replace("\\", "/"),
            }
            cases.append(case)
            annotations.append({
                "schema_version": "greenspec.candidate_evidence_pack_review.v1",
                "review_run_id": run_id,
                "review_target": "evidence_pack",
                "candidate_id": candidate_id,
                "index_manifest_id": next(iter(manifests)),
                "question": candidate["question"],
                "evidence_pack_file": case["evidence_pack_file"],
                "annotation": dict(ANNOTATION),
            })
        _write_jsonl(temporary / "review_annotations.jsonl", annotations)
        manifest = {
            "schema_version": "greenspec.candidate_evidence_pack_review_run.v1",
            "review_run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(annotations),
            "index_manifest_id": next(iter(manifests)),
            "source": {"accepted_path": str(accepted_path), "accepted_sha256": _sha256(accepted_path), "retrievals_path": str(retrievals_path), "retrievals_sha256": _sha256(retrievals_path)},
            "files": {"annotations": "review_annotations.jsonl", "evidence_packs": "evidence_packs"},
            "cases": cases,
            "rubric": {
                "citation_accuracy": "Every cited standard, version, clause/table/formula and page locator is correct for its EvidencePack item.",
                "citation_completeness": "The EvidencePack includes all material conditions, values, units, exceptions and required table/formula continuations needed to answer the question.",
                "evidence_pack_supports_question": "The EvidencePack alone supports an answer to the frozen question; do not use outside knowledge.",
            },
        }
        (temporary / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        final = output_root / run_id
        os.replace(temporary, final)
        return final
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 98-case EvidencePack human-review queue from frozen candidate retrievals.")
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--retrievals", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/candidate_evidence_pack_review_runs"))
    args = parser.parse_args()
    final = build_review_run(accepted_path=args.accepted, retrievals_path=args.retrievals, output_root=args.output_root)
    manifest = json.loads((final / "review_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"review_run_id": manifest["review_run_id"], "review_path": str(final), "case_count": manifest["case_count"], "index_manifest_id": manifest["index_manifest_id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
