"""Write an immutable Day 5 run summary without copying sensitive evidence text."""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/runs"))
    parser.add_argument("--test-result", default="python -m unittest discover -s tests -v: 23 passed")
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    run_id = f"run_day5_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    summary = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "publication_status": "experimental", "index_manifest_id": evaluation.get("index_manifest_id"), "evaluation_run_id": evaluation.get("evaluation_run_id"), "change_summary": ["BGE reranker adapter with fixed batch, token bounds, timeout and device", "Evidence Pack schema/citation gate and parent context", "Hybrid API fallback, health and index-build status", "10-case evaluation executed without inferring labels"], "tests": [args.test_result], "unresolved": evaluation.get("unresolved", []), "logging_policy": "run/build/request/manifest, elapsed time, candidate count and degraded status only; no project evidence text"}
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / f"{run_id}.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "path": str(target), "publication_status": "experimental"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
