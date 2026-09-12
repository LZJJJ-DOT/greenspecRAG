"""Run the RAGAS retrieval trial against a hash-bound prepared dataset.

The deterministic ID metrics are derived from stable evidence IDs.  The
LLM-based metrics in this retrieval-sidecar trial are RAGAS ContextPrecision,
ContextRecall, and ContextRelevance. They evaluate a query and retrieved
contexts but never invent a final answer. API secrets are read only from the
process environment and are never written to output artifacts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _id_metrics(sample: dict[str, Any]) -> tuple[float, float]:
    retrieved = set(map(str, sample["retrieved_context_ids"]))
    reference = set(map(str, sample["reference_context_ids"]))
    overlap = len(retrieved.intersection(reference))
    precision = overlap / len(retrieved) if retrieved else 0.0
    recall = overlap / len(reference) if reference else 0.0
    return precision, recall


def _is_transient_judge_error(exc: Exception) -> bool:
    """Return whether retrying the unchanged judge request is justified."""
    status_code = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, ConnectionError))
        or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
        or status_code in {408, 425, 429}
        or (isinstance(status_code, int) and status_code >= 500)
    )


async def _judge_retrieval_metrics(
    sample: dict[str, Any],
    model: str,
    base_url: str,
    judge_max_tokens: int,
    judge_attempts: int,
    metric_names: set[str] | None = None,
) -> tuple[dict[str, float | None], dict[str, str | None], dict[str, str | None]]:
    try:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall, ContextRelevance
    except ImportError as exc:  # pragma: no cover - runtime image installation gate
        raise RuntimeError("RAGAS evaluation dependencies are unavailable; rebuild the ragas-eval image") from exc
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required unless --skip-llm is set")
    # Judge calls are external and billable.  Bound a stalled request so it cannot
    # indefinitely block the frozen regression run.
    timeout_seconds = float(os.getenv("RAGAS_JUDGE_TIMEOUT_SECONDS", "30"))
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )
    llm = llm_factory(
        model,
        client=client,
        max_tokens=judge_max_tokens,
    )
    scorers = {
        "context_precision": (ContextPrecision(llm=llm), {
            "user_input": sample["user_input"],
            "reference": "\n\n".join(sample["reference_contexts"]),
            "retrieved_contexts": sample["retrieved_contexts"],
        }),
        "context_recall": (ContextRecall(llm=llm), {
            "user_input": sample["user_input"],
            "reference": "\n\n".join(sample["reference_contexts"]),
            "retrieved_contexts": sample["retrieved_contexts"],
        }),
        "context_relevance": (ContextRelevance(llm=llm), {
            "user_input": sample["user_input"],
            "retrieved_contexts": sample["retrieved_contexts"],
        }),
    }
    if metric_names is not None:
        scorers = {name: scorer for name, scorer in scorers.items() if name in metric_names}
    scores: dict[str, float | None] = {}
    reasons: dict[str, str | None] = {}
    errors: dict[str, str | None] = {}

    async def score_one(name: str, scorer: Any, inputs: dict[str, Any]) -> tuple[str, float | None, str | None, str | None]:
        for attempt in range(judge_attempts):
            try:
                result = await asyncio.wait_for(
                    scorer.ascore(**inputs),
                    timeout=timeout_seconds,
                )
                reason = getattr(result, "reason", None)
                return name, float(result.value), str(reason) if reason is not None else None, None
            except Exception as exc:  # a failed judge call must be visible to the review queue
                should_retry = (
                    attempt + 1 < judge_attempts
                    and _is_transient_judge_error(exc)
                )
                if not should_retry:
                    return name, None, None, f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(attempt + 1)

        raise RuntimeError("judge retry loop exited unexpectedly")

    try:
        results = await asyncio.gather(*(score_one(name, scorer, inputs) for name, (scorer, inputs) in scorers.items()))
        for name, score, reason, error in results:
            scores[name], reasons[name], errors[name] = score, reason, error
        return scores, reasons, errors
    finally:
        await client.close()


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _path(args.dataset)
    manifest = _read_json(dataset / "manifest.json")
    samples = _read_jsonl(dataset / "samples.jsonl")
    if len(samples) != manifest.get("scored_sample_count"):
        raise ValueError("dataset sample count does not match its manifest")
    retry_rows: dict[str, dict[str, Any]] = {}
    retry_path: Path | None = None
    if args.retry_scores:
        retry_path = _path(args.retry_scores)
        for row in _read_jsonl(retry_path):
            eval_id = str(row.get("eval_id", ""))
            if not eval_id or eval_id in retry_rows:
                raise ValueError("--retry-scores must contain one distinct non-empty eval_id per row")
            retry_rows[eval_id] = row
        sample_ids = {str(sample["eval_id"]) for sample in samples}
        if set(retry_rows) != sample_ids:
            raise ValueError("--retry-scores eval_ids do not match the prepared dataset")

    async def score_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        id_precision, id_recall = _id_metrics(sample)
        previous = retry_rows.get(str(sample["eval_id"]))
        judged_metrics: dict[str, float | None] = {
            "context_precision": None,
            "context_recall": None,
            "context_relevance": None,
        }
        judge_reason: dict[str, str | None] = {}
        judge_error: dict[str, str | None] = {}
        metric_names: set[str] | None = None
        if previous:
            previous_metrics = previous.get("metrics", {})
            previous_reason = previous.get("judge_reason", {})
            previous_error = previous.get("judge_error", {})
            judged_metrics = {name: previous_metrics.get(name) for name in judged_metrics}
            judge_reason = {name: previous_reason.get(name) for name in judged_metrics}
            judge_error = {name: previous_error.get(name) for name in judged_metrics}
            metric_names = {name for name, value in judged_metrics.items() if value is None}
        if not args.skip_llm:
            if metric_names is None or metric_names:
                new_metrics, new_reason, new_error = await _judge_retrieval_metrics(
                    sample,
                    args.judge_model,
                    args.judge_base_url,
                    args.judge_max_tokens,
                    args.judge_attempts,
                    metric_names,
                )
                judged_metrics.update(new_metrics)
                judge_reason.update(new_reason)
                judge_error.update(new_error)
        row = {
            "eval_id": sample["eval_id"],
            "metrics": {
                "id_context_precision": id_precision,
                "id_context_recall": id_recall,
                **judged_metrics,
            },
            "judge_reason": judge_reason,
            "judge_error": judge_error,
        }
        print(
            f"[{index}/{len(samples)}] {sample['eval_id']}: "
            f"context_precision={judged_metrics['context_precision']!r} "
            f"context_recall={judged_metrics['context_recall']!r} "
            f"context_relevance={judged_metrics['context_relevance']!r}",
            flush=True,
        )
        return row

    rows: list[dict[str, Any]] = []
    for start in range(0, len(samples), args.max_concurrency):
        batch = samples[start:start + args.max_concurrency]
        rows.extend(await asyncio.gather(*(score_sample(index, sample) for index, sample in enumerate(batch, start=start + 1))))
    run_id = f"ragas_trial_{uuid.uuid4().hex}"
    output_root = _path(args.output_root)
    output = output_root / f"{run_id}.jsonl"
    _write_jsonl_atomic(output, rows)
    numeric = lambda metric: [row["metrics"][metric] for row in rows if isinstance(row["metrics"][metric], (int, float))]
    metric_names = (
        "id_context_precision",
        "id_context_recall",
        "context_precision",
        "context_recall",
        "context_relevance",
    )
    metric_coverage = {
        metric: {
            "scored_case_count": len(numeric(metric)),
            "case_count": len(rows),
            "coverage": f"{len(numeric(metric))}/{len(rows)}",
            "rate": len(numeric(metric)) / len(rows) if rows else None,
        }
        for metric in metric_names
    }
    summary = {
        "schema_version": "greenspec.ragas_trial.v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_manifest_sha256": __import__("hashlib").sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "judge": None if args.skip_llm else {
            "provider": "deepseek",
            "model": args.judge_model,
            "base_url": args.judge_base_url,
            "max_tokens": args.judge_max_tokens,
            "attempts": args.judge_attempts,
        },
        "max_concurrency": args.max_concurrency,
        "case_count": len(rows),
        "metric_means": {metric: (fmean(values) if values else None) for metric in metric_names for values in [numeric(metric)]},
        "metric_coverage": metric_coverage,
        "judge_error_count": sum(any(error is not None for error in row["judge_error"].values()) for row in rows),
        "retried_from_scores_path": str(retry_path.relative_to(ROOT)) if retry_path else None,
        "scores_path": str(output.relative_to(ROOT)),
        "scope": "Retrieval-only. This run evaluates RAGAS Context Precision, Context Recall and Context Relevance; it does not evaluate final-answer faithfulness, answer relevance, or business manual-review recall.",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAGAS retrieval metrics over a prepared GreenSpec trial dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/ragas/runs"))
    parser.add_argument("--judge-model", default=os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--judge-base-url", default=os.environ.get("RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--judge-max-tokens", type=int, default=2048, help="Maximum tokens for each structured judge response.")
    parser.add_argument("--judge-attempts", type=int, default=2, help="Attempts per metric for transient judge errors only.")
    parser.add_argument("--max-concurrency", type=int, default=int(os.environ.get("RAGAS_MAX_CONCURRENCY", "4")))
    parser.add_argument("--retry-scores", type=Path, help="Retry only semantic metrics that are null in this prior scores JSONL, preserving successful scores.")
    parser.add_argument("--skip-llm", action="store_true", help="Run deterministic ID metrics only; does not read DEEPSEEK_API_KEY.")
    args = parser.parse_args()
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be positive")
    if args.judge_max_tokens < 1:
        parser.error("--judge-max-tokens must be positive")
    if args.judge_attempts < 1:
        parser.error("--judge-attempts must be positive")
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
