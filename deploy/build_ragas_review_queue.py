"""Turn per-case RAGAS scores into an auditable low-score review queue."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PERFECT_ID_RECALL_REGRESSIONS = {"E006", "E029", "E031"}


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _threshold(value: str) -> tuple[str, float]:
    name, separator, raw = value.partition("=")
    if not separator or not name:
        raise argparse.ArgumentTypeError("threshold must be METRIC=VALUE")
    try:
        score = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold value must be numeric") from exc
    if not 0 <= score <= 1:
        raise argparse.ArgumentTypeError("threshold value must be between 0 and 1")
    return name, score


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a per-case RAGAS manual review queue.")
    parser.add_argument("--dataset", type=Path, required=True, help="RAGAS dataset directory created by prepare_ragas_retrieval_dataset")
    parser.add_argument("--scores", type=Path, required=True, help="JSONL rows: eval_id plus a metrics mapping")
    parser.add_argument("--threshold", action="append", type=_threshold, required=True, help="Repeat: id_context_precision=0.20")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    dataset, scores_path = _path(args.dataset), _path(args.scores)
    samples, scores = _read_jsonl(dataset / "samples.jsonl"), _read_jsonl(scores_path)
    score_by_id = {str(row.get("eval_id")): row for row in scores}
    if len(scores) != len(score_by_id):
        raise SystemExit("scores contain duplicate eval_id values")
    thresholds = dict(args.threshold)
    queue = []
    for sample in samples:
        score_row = score_by_id.get(sample["eval_id"])
        metrics = score_row.get("metrics", {}) if score_row else {}
        failed, missing = [], []
        for name, threshold in thresholds.items():
            value = metrics.get(name)
            if not isinstance(value, (int, float)):
                missing.append(name)
            elif value < threshold:
                failed.append({"metric": name, "value": value, "threshold": threshold})
        if sample["eval_id"] in PERFECT_ID_RECALL_REGRESSIONS:
            value = metrics.get("id_context_recall")
            if not isinstance(value, (int, float)):
                missing.append("id_context_recall_for_fixed_regression")
            elif value < 1.0:
                failed.append({"metric": "id_context_recall", "value": value, "threshold": 1.0, "critical": True})
        needs_review = bool(failed or missing)
        queue.append({"eval_id": sample["eval_id"], "status": "needs_manual_review" if needs_review else "passed_score_gate_pending_sampling_policy", "manual_review_required": needs_review, "failed_metrics": failed, "missing_metrics": missing, "metrics": metrics, "review_reason": "low_or_missing_ragas_score" if needs_review else None, "reviewer": None, "reviewed_at": None, "notes": None})
    output = _path(args.output) if args.output else dataset / "manual_review_queue.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queue), encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": len(queue), "manual_review_required": sum(row["manual_review_required"] for row in queue), "created_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
