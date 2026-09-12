"""Prepare separate retrieval and answer RAGAS datasets from frozen candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _item_context(item: dict[str, Any]) -> str:
    parent = "\n".join(str(value.get("text", "")) for value in item.get("parent_context", []) if isinstance(value, dict) and value.get("text"))
    return "\n".join(part for part in (parent, str(item.get("text", ""))) if part).strip()


def _dataset_manifest(*, dataset_id: str, kind: str, cases: list[dict[str, Any]], diagnostic_cases: list[dict[str, Any]], index_manifest_id: str, accepted_path: Path, retrievals_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "greenspec.candidate_ragas_dataset.v1",
        "dataset_id": dataset_id,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scored_sample_count": len(cases),
        "diagnostic_case_count": len(diagnostic_cases),
        "index_manifest_id": index_manifest_id,
        "files": {"samples": "samples.jsonl", "diagnostics": "diagnostic_cases.jsonl"},
        "source": {"accepted_path": str(accepted_path), "accepted_sha256": _sha256(accepted_path), "retrievals_path": str(retrievals_path), "retrievals_sha256": _sha256(retrievals_path)},
        "scope": "Retrieval samples contain real rerank Top10 primary evidence. Answer samples contain only the budgeted EvidencePack items that the downstream consumer receives.",
    }


def prepare_datasets(*, accepted_path: Path, retrievals_path: Path, canonical_path: Path, output_root: Path) -> Path:
    accepted, retrievals, canonical = _rows(accepted_path), _rows(retrievals_path), _rows(canonical_path)
    candidate_by_id = {row.get("candidate_id"): row for row in accepted}
    retrieval_by_id = {row.get("candidate_id"): row for row in retrievals}
    canonical_by_id = {row.get("clause_id"): row for row in canonical}
    if len(candidate_by_id) != len(accepted) or len(retrieval_by_id) != len(retrievals) or set(candidate_by_id) != set(retrieval_by_id):
        raise ValueError("accepted candidates and retrievals must have identical unique candidate_id sets")
    manifests = {row.get("retrieval", {}).get("index_manifest_id") for row in retrievals}
    if len(manifests) != 1 or None in manifests:
        raise ValueError("retrievals must be frozen to one non-empty index manifest")
    if any(row.get("retrieval", {}).get("error") or not isinstance(row.get("evidence_pack"), dict) for row in retrievals):
        raise ValueError("every frozen retrieval needs an error-free EvidencePack")

    index_manifest_id = next(iter(manifests))
    retrieval_cases: list[dict[str, Any]] = []
    retrieval_diagnostics: list[dict[str, Any]] = []
    answer_cases: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_by_id):
        candidate, retrieval = candidate_by_id[candidate_id], retrieval_by_id[candidate_id]
        required_ids = [str(value) for value in candidate.get("required_evidence_ids", [])]
        missing = [value for value in required_ids if value not in canonical_by_id]
        if missing:
            raise ValueError(f"{candidate_id}: required evidence is absent from canonical: {missing}")
        ranked_items = [item for item in retrieval["retrieval"].get("top_k_items", []) if item.get("rerank_score") is not None][:10]
        pack_items = retrieval["evidence_pack"].get("items", [])
        common = {
            "schema_version": "greenspec.candidate_ragas_sample.v1",
            "eval_id": candidate_id,
            "user_input": candidate["question"],
            "reference_context_ids": required_ids,
            "reference_contexts": [str(canonical_by_id[value].get("text", "")) for value in required_ids],
            "metadata": {
                "category": candidate.get("category"),
                "difficulty_label": candidate.get("difficulty_label"),
                "gold_evidence_ids": candidate.get("gold_evidence_ids", []),
                "required_evidence_ids": required_ids,
                "forbidden_near_misses": candidate.get("forbidden_near_misses", []),
                "index_manifest_id": index_manifest_id,
            },
        }
        retrieval_case = {
            **common,
            "retrieved_context_ids": [item["evidence_id"] for item in ranked_items],
            "retrieved_contexts": [_item_context(item) for item in ranked_items],
            "evidence_pack": {"source": "rerank_top10_primary", "items": [{key: item.get(key) for key in ("evidence_id", "clause_no", "content_type", "pdf_page_start", "pdf_page_end", "table_id", "formula_id")} for item in ranked_items]},
        }
        if candidate.get("category") == "insufficient_evidence":
            retrieval_diagnostics.append(retrieval_case)
        else:
            retrieval_cases.append(retrieval_case)
        answer_cases.append({
            **common,
            "retrieved_context_ids": [item["evidence_id"] for item in pack_items],
            "retrieved_contexts": [_item_context(item) for item in pack_items],
            "evidence_pack": {"source": "budgeted_evidence_pack", "evidence_pack_id": retrieval["evidence_pack"].get("evidence_pack_id"), "context_budget": retrieval["evidence_pack"].get("context_budget"), "items": [{key: item.get(key) for key in ("evidence_id", "clause_no", "content_type", "pdf_page_start", "pdf_page_end", "table_id", "formula_id")} for item in pack_items]},
        })

    fingerprint = hashlib.sha256((_sha256(accepted_path) + _sha256(retrievals_path) + _sha256(canonical_path)).encode("ascii")).hexdigest()[:12]
    destination = output_root / f"candidate_ragas_v2_{fingerprint}"
    if destination.exists():
        raise FileExistsError(f"dataset already exists: {destination}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=output_root))
    try:
        for kind, cases, diagnostics in (("retrieval", retrieval_cases, retrieval_diagnostics), ("answer", answer_cases, [])):
            folder = temporary / kind
            folder.mkdir()
            _write_jsonl(folder / "samples.jsonl", cases)
            _write_jsonl(folder / "diagnostic_cases.jsonl", diagnostics)
            (folder / "manifest.json").write_text(json.dumps(_dataset_manifest(dataset_id=f"{destination.name}_{kind}", kind=kind, cases=cases, diagnostic_cases=diagnostics, index_manifest_id=index_manifest_id, accepted_path=accepted_path, retrievals_path=retrievals_path), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return destination
    except Exception:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare separate RAGAS retrieval and answer datasets from frozen candidate retrievals.")
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--retrievals", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/ragas/candidate_datasets"))
    args = parser.parse_args()
    output = prepare_datasets(accepted_path=args.accepted, retrievals_path=args.retrievals, canonical_path=args.canonical, output_root=args.output_root)
    print(json.dumps({"output": str(output), "retrieval_dataset": str(output / "retrieval"), "answer_dataset": str(output / "answer")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
