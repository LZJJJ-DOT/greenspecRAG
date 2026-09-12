"""Fail-closed startup acceptance for the immutable active hybrid baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _json_request(url: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify(*, hybrid_root: Path, api_url: str, qdrant_url: str, expected_points: int) -> dict[str, Any]:
    active_path = hybrid_root / "active.json"
    if not active_path.is_file():
        raise RuntimeError("no active hybrid manifest")
    active = _load(active_path)
    if active.get("publication_status") != "baseline":
        raise RuntimeError("active hybrid manifest is not a baseline")
    manifest_path = ROOT / Path(str(active["manifest_path"]).replace("\\", "/"))
    manifest = _load(manifest_path)
    if manifest.get("index_manifest_id") != active.get("index_manifest_id"):
        raise RuntimeError("active manifest ID does not match its index manifest")
    if int(manifest.get("indexed_node_count", -1)) != expected_points:
        raise RuntimeError("baseline indexed node count differs from expected points")

    health_status, health = _json_request(f"{api_url.rstrip('/')}/health")
    if health_status != 200 or not health.get("ready") or health.get("active_index_manifest_id") != active["index_manifest_id"]:
        raise RuntimeError("API health is not ready for the active baseline")

    collection = str(manifest["qdrant_collection"])
    collection_status, collection_info = _json_request(f"{qdrant_url.rstrip('/')}/collections/{collection}")
    points = int(collection_info.get("result", {}).get("points_count", -1))
    if collection_status != 200 or points != expected_points:
        raise RuntimeError("Qdrant collection point count is invalid")
    samples = []
    for point_id in sorted({0, expected_points // 2, expected_points - 1}):
        status, point = _json_request(f"{qdrant_url.rstrip('/')}/collections/{collection}/points/{point_id}?with_payload=false&with_vector=true")
        vector = point.get("result", {}).get("vector")
        if status != 200 or not isinstance(vector, list) or len(vector) != int(manifest["qdrant_vector_size"]):
            raise RuntimeError(f"Qdrant vector probe {point_id} has an invalid dimension")
        norm = _norm(vector)
        if not 0.999 <= norm <= 1.001 or not any(abs(float(value)) > 1e-12 for value in vector):
            raise RuntimeError(f"Qdrant vector probe {point_id} is not a non-zero unit vector")
        samples.append({"point_id": point_id, "dimension": len(vector), "norm": round(norm, 8)})

    retrieve_body = {
        "query": "围护结构热工性能",
        "project_profile": {},
        "filters": {"standard_ids": [], "clause_nos": [], "as_of_date": None, "must_be_current": False, "include_commentary": False},
        "top_k": 10,
        "allow_degraded": False,
    }
    retrieve_status, retrieve = _json_request(f"{api_url.rstrip('/')}/v1/retrieve", method="POST", body=retrieve_body)
    if retrieve_status != 200 or retrieve.get("index_manifest_id") != active["index_manifest_id"]:
        raise RuntimeError("strict retrieve did not return the active baseline")
    if retrieve.get("degraded_modes") != []:
        raise RuntimeError("strict retrieve returned a degraded mode")
    return {
        "health_status": health_status,
        "active_index_manifest_id": active["index_manifest_id"],
        "qdrant_collection": collection,
        "points_count": points,
        "vector_samples": samples,
        "retrieve_status": retrieve_status,
        "retrieve_item_count": len(retrieve.get("items", [])),
        "degraded_modes": retrieve.get("degraded_modes", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the active GreenSpec baseline after Docker/Qdrant restart.")
    parser.add_argument("--hybrid-root", type=Path, default=Path("data/indexes/hybrid"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8787")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--expected-points", type=int, default=460)
    parser.add_argument("--output-root", type=Path, default=Path("data/verification/startup_runs"))
    args = parser.parse_args()
    report = {"startup_acceptance_id": f"startup_{uuid.uuid4().hex}", "created_at": datetime.now(timezone.utc).isoformat(), **verify(hybrid_root=args.hybrid_root, api_url=args.api_url, qdrant_url=args.qdrant_url, expected_points=args.expected_points)}
    output = args.output_root / f"{report['startup_acceptance_id']}.json"
    _write_atomic(output, report)
    print(json.dumps({"startup_acceptance_id": report["startup_acceptance_id"], "report_path": str(output), "index_manifest_id": report["active_index_manifest_id"], "points_count": report["points_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
