"""Backfill top-level trace fields from already-frozen EvidencePacks.

No retrieval, model load, network request, or source mutation occurs here.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrievals", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, help="One-record JSONL retry result to merge before metadata backfill.")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.retrievals.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.replacement:
        replacements = [json.loads(line) for line in args.replacement.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(replacements) != 1:
            raise RuntimeError("replacement must contain exactly one JSONL record")
        replacement = replacements[0]
        candidate_id = replacement.get("candidate_id")
        matched = sum(row.get("candidate_id") == candidate_id for row in rows)
        if matched != 1:
            raise RuntimeError(f"replacement candidate must match exactly one existing row: {candidate_id!r}")
        rows = [replacement if row.get("candidate_id") == candidate_id else row for row in rows]
    if not rows:
        raise RuntimeError("retrieval export is empty")
    manifests: set[str] = set()
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        pack = row.get("evidence_pack")
        if not isinstance(pack, dict):
            raise RuntimeError(f"{row.get('candidate_id')}: missing EvidencePack")
        manifest = pack.get("index_manifest_id")
        run_id = pack.get("retrieval_run_id")
        if not isinstance(manifest, str) or not manifest or not isinstance(run_id, str) or not run_id:
            raise RuntimeError(f"{row.get('candidate_id')}: EvidencePack lacks frozen trace metadata")
        retrieval = dict(row.get("retrieval") or {})
        for key, value in {
            "request_id": pack.get("request_id"),
            "retrieval_run_id": run_id,
            "index_manifest_id": manifest,
        }.items():
            existing = retrieval.get(key)
            if existing not in (None, "", value):
                raise RuntimeError(f"{row.get('candidate_id')}: conflicting {key}: {existing!r} != {value!r}")
            retrieval[key] = value
        row["retrieval"] = retrieval
        row["metadata_backfilled_at"] = now
        row["metadata_source"] = "evidence_pack.v1"
        manifests.add(manifest)
    if len(manifests) != 1:
        raise RuntimeError(f"expected one frozen manifest, found {sorted(manifests)}")
    fd, temp_name = tempfile.mkstemp(prefix=args.retrievals.name + ".", suffix=".tmp", dir=args.retrievals.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp_name, args.retrievals)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    print(json.dumps({"retrievals": str(args.retrievals), "case_count": len(rows), "index_manifest_id": next(iter(manifests)), "metadata_source": "evidence_pack.v1"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
