"""Create an immutable local snapshot of the active Qdrant baseline collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _request(url: str, *, method: str = "GET") -> tuple[int, bytes]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, response.read()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"snapshot destination already exists: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Qdrant snapshot for the immutable active baseline.")
    parser.add_argument("--hybrid-root", type=Path, default=Path("data/indexes/hybrid"))
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--output-root", type=Path, default=Path("data/backups/qdrant"))
    args = parser.parse_args()
    active = json.loads((args.hybrid_root / "active.json").read_text(encoding="utf-8"))
    if active.get("publication_status") != "baseline":
        raise RuntimeError("only an active baseline may be backed up")
    manifest_path = ROOT / Path(str(active["manifest_path"]).replace("\\", "/"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = str(manifest["qdrant_collection"])
    root_url = args.qdrant_url.rstrip("/")
    status, created = _request(f"{root_url}/collections/{urllib.parse.quote(collection, safe='')}/snapshots", method="POST")
    if status != 200:
        raise RuntimeError("Qdrant refused snapshot creation")
    snapshot = json.loads(created.decode("utf-8")).get("result", {})
    snapshot_name = snapshot.get("name")
    if not isinstance(snapshot_name, str) or not snapshot_name:
        raise RuntimeError("Qdrant returned no snapshot name")
    _, payload = _request(f"{root_url}/collections/{urllib.parse.quote(collection, safe='')}/snapshots/{urllib.parse.quote(snapshot_name, safe='')}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = args.output_root / active["index_manifest_id"] / f"{timestamp}_{snapshot_name}"
    _write_atomic(snapshot_path, payload)
    metadata = {
        "backup_id": f"qdrant_snapshot_{uuid.uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_manifest_id": active["index_manifest_id"],
        "qdrant_collection": collection,
        "snapshot_name": snapshot_name,
        "snapshot_path": str(snapshot_path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    metadata_path = snapshot_path.with_suffix(snapshot_path.suffix + ".json")
    _write_atomic(metadata_path, (json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"backup_id": metadata["backup_id"], "snapshot_path": str(snapshot_path), "sha256": metadata["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
