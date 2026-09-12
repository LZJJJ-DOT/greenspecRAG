"""Atomically publish a candidate hybrid index only after its evidence gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HYBRID_ROOT = ROOT / "data" / "indexes" / "hybrid"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a gate-approved candidate hybrid index.")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--replace-active", action="store_true", help="Explicitly advance an existing baseline after a new candidate passes its gate.")
    args = parser.parse_args()

    candidate = load(args.candidate_manifest)
    gate = load(args.gate_report)
    manifest_id = candidate.get("index_manifest_id")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("candidate manifest has no index_manifest_id")
    if gate.get("publication_status") != "baseline_eligible":
        raise ValueError("gate report is not baseline_eligible")
    if gate.get("index_manifest_id") != manifest_id:
        raise ValueError("gate report and candidate manifest use different index manifests")
    active_canonical = ROOT / "data" / "canonical" / "generated" / "active" / "clauses.jsonl"
    if sha256(active_canonical) != candidate.get("canonical_sha256"):
        raise ValueError("candidate canonical SHA does not match current active canonical")
    active_path = HYBRID_ROOT / "active.json"
    if active_path.is_file():
        current = load(active_path)
        if current.get("index_manifest_id") != manifest_id and not args.replace_active:
            raise PermissionError("an active baseline already exists; pass --replace-active only for a newly gate-approved manifest")
    source_dir = args.candidate_manifest.parent
    destination = HYBRID_ROOT / manifest_id
    if destination.exists():
        existing = destination / "index_manifest.json"
        if not existing.is_file() or load(existing) != candidate:
            raise FileExistsError(f"published destination conflicts: {destination}")
    else:
        stage = HYBRID_ROOT / f".publish-{uuid.uuid4().hex}"
        shutil.copytree(source_dir, stage)
        os.replace(stage, destination)

    active = {
        "index_manifest_id": manifest_id,
        # API may run inside the Linux Docker container while publication runs
        # on Windows. Store a project-relative POSIX path, never a host path.
        "manifest_path": str((destination / "index_manifest.json").relative_to(ROOT)).replace("\\", "/"),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "publication_status": "baseline",
        "gate_report_path": str(args.gate_report),
    }
    write_atomic(active_path, active)
    release = {
        "release_id": f"baseline_release_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "publication_status": "baseline",
        "index_manifest_id": manifest_id,
        "candidate_manifest_path": str(args.candidate_manifest),
        "published_manifest_path": str((destination / "index_manifest.json").relative_to(ROOT)).replace("\\", "/"),
        "gate_report_path": str(args.gate_report),
        "canonical_sha256": candidate["canonical_sha256"],
        "qdrant_collection": candidate["qdrant_collection"],
    }
    releases = ROOT / "data" / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    write_atomic(releases / f"{release['release_id']}.json", release)
    print(json.dumps({"release_id": release["release_id"], "index_manifest_id": manifest_id, "active_manifest": str(HYBRID_ROOT / "active.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
