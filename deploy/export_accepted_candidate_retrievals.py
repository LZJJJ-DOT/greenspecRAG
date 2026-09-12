"""Run accepted candidates through the active Hybrid API and freeze the evidence returned."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def request_for(candidate: dict[str, Any], top_k: int) -> dict[str, Any]:
    # Negative/evidence-insufficient cases must search the complete indexed
    # corpus; filtering them to a missing standard would make the expected
    # empty result tautological.
    standard_ids = [] if candidate.get("category") == "insufficient_evidence" else list(candidate.get("expected_standard_ids") or [])
    return {
        "query": candidate["question"],
        "project_profile": {},
        "filters": {
            "standard_ids": standard_ids,
            "clause_nos": [],
            "as_of_date": None,
            "must_be_current": False,
            "include_commentary": False,
        },
        "top_k": top_k,
        "allow_degraded": False,
    }


def post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return int(error.code), json.loads(error.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8787/v1/retrieve")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.top_k <= 100:
        parser.error("--top-k must be 1..100")

    rows: list[dict[str, Any]] = []
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", encoding="utf-8") as handle:
      for index, candidate in enumerate(read_jsonl(args.accepted), start=1):
        payload = request_for(candidate, args.top_k)
        try:
            status, response = post(args.api_url, payload, args.timeout_seconds)
        except Exception as exc:  # Preserve failures as auditable rows; continue remaining cases.
            status, response = 0, {"error": {"code": "transport_error", "message": f"{type(exc).__name__}: {exc}"}}
        row = {
            "schema_version": "greenspec.accepted_candidate_retrieval.v1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "candidate_id": candidate["candidate_id"],
            "question": candidate["question"],
            "category": candidate.get("category"),
            "difficulty_label": candidate.get("difficulty_label"),
            "required_evidence_ids": candidate.get("required_evidence_ids", []),
            "supporting_evidence_ids": candidate.get("supporting_evidence_ids", []),
            "forbidden_near_misses": candidate.get("forbidden_near_misses", []),
            "request": payload,
            "http_status": status,
            "retrieval": {
                "request_id": response.get("request_id"),
                "retrieval_run_id": response.get("retrieval_run_id"),
                "index_manifest_id": response.get("index_manifest_id"),
                "top_k_items": response.get("items", []),
                "filters_applied": response.get("filters_applied"),
                "warnings": response.get("warnings", []),
                "degraded_modes": response.get("degraded_modes", []),
                "error": response.get("error"),
            },
            "evidence_pack": response.get("evidence_pack"),
        }
        rows.append(row)
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        print(f"[{index}] {candidate['candidate_id']}: HTTP {status}", flush=True)

    partial.replace(args.output)
    succeeded = sum(row["http_status"] == 200 and row["evidence_pack"] is not None for row in rows)
    print(json.dumps({"output": str(args.output), "case_count": len(rows), "evidence_pack_count": succeeded, "failed_count": len(rows) - succeeded}, ensure_ascii=False))
    return 0 if succeeded == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
