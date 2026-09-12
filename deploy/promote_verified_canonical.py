"""Atomically promote a fully verified canonical candidate without rewriting it."""
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


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(path: Path) -> tuple[int, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("verification_status") != "verified" or row.get("source_level") != "T0" for row in rows):
        raise ValueError("candidate canonical contains unverified or non-T0 nodes")
    return len(rows), _sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a verified canonical candidate and archive the previous active canonical.")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=Path("data/canonical/generated"))
    args = parser.parse_args()
    candidate_root = args.candidate_root if args.candidate_root.is_absolute() else ROOT / args.candidate_root
    target_root = args.target_root if args.target_root.is_absolute() else ROOT / args.target_root
    candidate_active, target_active = candidate_root.resolve() / "active", target_root.resolve() / "active"
    source = candidate_active / "clauses.jsonl"
    if not candidate_active.is_dir() or not target_active.is_dir():
        raise FileNotFoundError("both candidate and target active canonical directories must exist")
    node_count, canonical_sha = _verified(source)
    candidate_manifest = _load(candidate_active / "index_manifest.json")
    if candidate_manifest.get("canonical_sha256") != canonical_sha:
        raise ValueError("candidate manifest SHA does not match candidate canonical")
    previous_manifest = _load(target_active / "index_manifest.json")

    target_root = target_root.resolve()
    revisions = target_root / "revisions"
    revisions.mkdir(parents=True, exist_ok=True)
    archive = revisions / f"pre_verified_promotion_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{previous_manifest.get('index_manifest_id', 'unknown')}"
    stage = target_root / f".verified-stage-{uuid.uuid4().hex}"
    shutil.copytree(candidate_active, stage)
    temporary_old = target_root / f".active-old-{uuid.uuid4().hex}"
    try:
        os.replace(target_active, temporary_old)
        try:
            os.replace(stage, target_active)
        except Exception:
            os.replace(temporary_old, target_active)
            raise
        os.replace(temporary_old, archive)
    except Exception:
        if temporary_old.exists() and not target_active.exists():
            os.replace(temporary_old, target_active)
        if stage.exists():
            shutil.rmtree(stage)
        raise

    promotions = target_root / "promotions"
    promotions.mkdir(exist_ok=True)
    record = {
        "promotion_id": f"verified_canonical_{uuid.uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_root": str(candidate_root),
        "candidate_index_manifest_id": candidate_manifest["index_manifest_id"],
        "previous_index_manifest_id": previous_manifest.get("index_manifest_id"),
        "canonical_sha256": canonical_sha,
        "node_count": node_count,
        "archived_previous_active": str(archive),
    }
    (promotions / f"{record['promotion_id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
