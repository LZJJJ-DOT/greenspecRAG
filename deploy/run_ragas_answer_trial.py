"""Generate frozen-evidence answers and score RAGAS answer-side metrics.

This evaluator is deliberately isolated from the serving API. It writes an
immutable answer artifact containing the generated response, its available
RAGAS scores, and metric-specific failures. It never changes an index manifest
or a production response.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import math
import os
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_METRICS = ("faithfulness", "answer_relevancy")
ANSWER_PROMPT_VERSION = "greenspec.answer_eval.v1"
DIAGNOSTICS_SCHEMA_VERSION = "greenspec.ragas_judge_diagnostics.v1"
_JUDGE_CALL_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "ragas_judge_call_context", default={}
)


def _path(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_value(value: Any) -> Any:
    """Convert SDK / Instructor objects to JSON-safe diagnostic data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_value(model_dump(mode="json"))
        except TypeError:
            return _json_value(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    return repr(value)


def _exception_payload(exc: Exception) -> dict[str, Any]:
    """Keep the partial Instructor completion and exception chain for diagnosis."""
    chain: list[dict[str, str]] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__ or current.__context__
    attributes = {
        name: _json_value(getattr(exc, name))
        for name in (
            "last_completion",
            "create_kwargs",
            "n_attempts",
            "total_usage",
            "failed_attempts",
        )
        if hasattr(exc, name)
    }
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "chain": chain,
        "traceback": "".join(traceback.format_exception(exc)),
        "attributes": attributes,
    }


class _JudgeDiagnostics:
    """Append-only Instructor hook log, enabled only by an explicit CLI option."""

    def __init__(self, path: Path, *, run_id: str, config: dict[str, Any]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.sequence = 0
        self._write("run_config", config=config)

    def bind(self, **context: Any) -> contextvars.Token[dict[str, Any]]:
        return _JUDGE_CALL_CONTEXT.set(context)

    @staticmethod
    def reset(token: contextvars.Token[dict[str, Any]]) -> None:
        _JUDGE_CALL_CONTEXT.reset(token)

    def _write(self, event: str, **payload: Any) -> None:
        self.sequence += 1
        record = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event": event,
            "call_context": _JUDGE_CALL_CONTEXT.get(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_json_value(record), ensure_ascii=False, sort_keys=True) + "\n")

    def on_completion_kwargs(self, *args: Any, **kwargs: Any) -> None:
        self._write("completion_kwargs", args=_json_value(args), kwargs=_json_value(kwargs))

    def on_completion_response(self, response: Any) -> None:
        self._write("completion_response", response=_json_value(response))

    def on_completion_error(self, error: Exception, **kwargs: Any) -> None:
        self._write("completion_error", error=_exception_payload(error), hook_metadata=_json_value(kwargs))

    def on_completion_last_attempt(self, error: Exception, **kwargs: Any) -> None:
        self._write("completion_last_attempt", error=_exception_payload(error), hook_metadata=_json_value(kwargs))

    def on_completion_usage(self, usage: Any, **kwargs: Any) -> None:
        self._write("completion_usage", usage=_json_value(usage), hook_metadata=_json_value(kwargs))

    def on_parse_error(self, error: Exception, **kwargs: Any) -> None:
        self._write("parse_error", error=_exception_payload(error), hook_metadata=_json_value(kwargs))

    def record_outer_error(self, error: Exception) -> None:
        self._write("metric_error", error=_exception_payload(error))


def _inject_judge_thinking(judge_llm: Any, thinking: str) -> Any:
    """Return the Instructor client after applying an explicit thinking override."""
    instructor_client = getattr(judge_llm, "client", None)
    defaults = getattr(instructor_client, "kwargs", None)
    if not isinstance(defaults, dict):
        raise RuntimeError("RAGAS judge client does not expose Instructor default request kwargs")
    if thinking == "disabled":
        extra_body = dict(defaults.get("extra_body") or {})
        extra_body["thinking"] = {"type": "disabled"}
        defaults["extra_body"] = extra_body
    return instructor_client


def _configure_judge_diagnostics(
    judge_llm: Any,
    *,
    thinking: str,
    diagnostics: _JudgeDiagnostics | None,
) -> None:
    """Inject an optional DeepSeek thinking override and Instructor hooks."""
    instructor_client = _inject_judge_thinking(judge_llm, thinking)
    if diagnostics is None:
        return
    hooks = getattr(instructor_client, "hooks", None)
    if hooks is None:
        raise RuntimeError("RAGAS judge client does not expose Instructor hooks")
    from instructor.v2.core.hooks import HookName

    hooks.on(HookName.COMPLETION_KWARGS, diagnostics.on_completion_kwargs)
    hooks.on(HookName.COMPLETION_RESPONSE, diagnostics.on_completion_response)
    hooks.on(HookName.COMPLETION_ERROR, diagnostics.on_completion_error)
    hooks.on(HookName.COMPLETION_LAST_ATTEMPT, diagnostics.on_completion_last_attempt)
    hooks.on(HookName.COMPLETION_USAGE, diagnostics.on_completion_usage)
    hooks.on(HookName.PARSE_ERROR, diagnostics.on_parse_error)


def _is_transient_judge_error(exc: Exception) -> bool:
    """Return whether retrying an unchanged external request is justified."""
    status_code = getattr(exc, "status_code", None)
    return (
        isinstance(exc, (TimeoutError, ConnectionError))
        or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
        or status_code in {408, 425, 429}
        or (isinstance(status_code, int) and status_code >= 500)
    )


def _is_retryable_answer_error(exc: Exception) -> bool:
    """Also retry an empty provider response; it has no semantic meaning."""
    return _is_transient_judge_error(exc) or (
        isinstance(exc, ValueError)
        and str(exc) in {
            "DeepSeek returned an empty answer",
            "DeepSeek returned a truncated answer (finish_reason=length)",
        }
    )


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _answer_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    evidence = "\n\n".join(
        f"[证据 {evidence_id}]\n{context}"
        for evidence_id, context in zip(sample["retrieved_context_ids"], sample["retrieved_contexts"])
    )
    return [
        {
            "role": "system",
            "content": (
                "你是建筑规范问答助手。只能依据提供的证据作答；不要补充常识、"
                "不要猜测。每一个规范性结论都应紧跟一个或多个 [证据 ID] 引用。"
                "若证据不足，明确说明不足。答案使用中文，简洁且不超过 400 字。"
            ),
        },
        {
            "role": "user",
            "content": f"问题：{sample['user_input']}\n\n已冻结检索证据：\n{evidence}",
        },
    ]


async def _generate_answer(
    client: Any,
    model: str,
    sample: dict[str, Any],
    timeout_seconds: float,
    answer_max_tokens: int,
    answer_attempts: int,
) -> tuple[str | None, str | None]:
    for attempt in range(answer_attempts):
        try:
            messages = _answer_messages(sample)
            if attempt:
                messages[0]["content"] += (
                    "上一版答案被输出长度截断。本次只保留直接结论和必要证据引用，"
                    "不要展开背景或举例，限 200 个汉字以内。"
                )
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=answer_max_tokens,
                ),
                timeout=timeout_seconds,
            )
            choice = completion.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("DeepSeek returned a truncated answer (finish_reason=length)")
            answer = choice.message.content
            if not answer or not answer.strip():
                raise ValueError("DeepSeek returned an empty answer")
            return answer.strip(), None
        except Exception as exc:
            should_retry = (
                attempt + 1 < answer_attempts
                and _is_retryable_answer_error(exc)
            )
            if not should_retry:
                return None, f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(attempt + 1)

    raise RuntimeError("answer retry loop exited unexpectedly")


async def _score_metric(
    name: str,
    scorer: Any,
    sample: dict[str, Any],
    answer: str,
    timeout_seconds: float,
    judge_attempts: int,
    diagnostics: _JudgeDiagnostics | None = None,
) -> tuple[float | None, str | None, str | None]:
    for attempt in range(judge_attempts):
        token = diagnostics.bind(
            case_id=str(sample["eval_id"]),
            metric=name,
            outer_attempt=attempt + 1,
        ) if diagnostics else None
        try:
            if name == "faithfulness":
                result = await asyncio.wait_for(
                    scorer.ascore(user_input=sample["user_input"], response=answer, retrieved_contexts=sample["retrieved_contexts"]),
                    timeout=timeout_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    scorer.ascore(user_input=sample["user_input"], response=answer),
                    timeout=timeout_seconds,
                )
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError(f"RAGAS returned a non-finite {name} score")
            reason = getattr(result, "reason", None)
            return value, str(reason) if reason is not None else None, None
        except Exception as exc:
            if diagnostics:
                diagnostics.record_outer_error(exc)
            should_retry = (
                attempt + 1 < judge_attempts
                and _is_transient_judge_error(exc)
            )
            if not should_retry:
                return None, None, f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(attempt + 1)
        finally:
            if diagnostics and token is not None:
                diagnostics.reset(token)

    raise RuntimeError("judge retry loop exited unexpectedly")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    except ImportError as exc:  # pragma: no cover - runtime image gate
        raise RuntimeError("RAGAS answer-evaluation dependencies are unavailable; rebuild the ragas-eval image") from exc

    run_id = f"ragas_answer_trial_{uuid.uuid4().hex}"
    diagnostics_path = _path(args.judge_diagnostics_dir) / f"{run_id}.jsonl" if args.judge_diagnostics_dir else None
    diagnostics = _JudgeDiagnostics(
        diagnostics_path,
        run_id=run_id,
        config={
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_attempts": args.judge_attempts,
            "judge_thinking": args.judge_thinking,
            "timeout_seconds": args.timeout_seconds,
        },
    ) if diagnostics_path else None
    judge_api_key = os.environ.get(args.judge_api_key_env)
    if not judge_api_key:
        raise RuntimeError(f"{args.judge_api_key_env} is required for RAGAS scoring")
    dataset = _path(args.dataset)
    manifest = _read_json(dataset / "manifest.json")
    samples = _read_jsonl(dataset / "samples.jsonl")
    if len(samples) != manifest.get("scored_sample_count"):
        raise ValueError("dataset sample count does not match its manifest")
    all_sample_ids = {str(sample["eval_id"]) for sample in samples}
    selected_eval_ids: list[str] | None = None
    if args.eval_ids:
        selected_eval_ids = [item.strip() for item in args.eval_ids.split(",") if item.strip()]
        if not selected_eval_ids:
            raise ValueError("--eval-ids must contain at least one non-empty eval_id")
        unknown_ids = sorted(set(selected_eval_ids) - all_sample_ids)
        if unknown_ids:
            raise ValueError(f"--eval-ids contains unknown eval_id values: {', '.join(unknown_ids)}")
        selected_set = set(selected_eval_ids)
        samples = [sample for sample in samples if str(sample["eval_id"]) in selected_set]

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
        if selected_eval_ids:
            retry_rows = {eval_id: row for eval_id, row in retry_rows.items() if eval_id in sample_ids}
        if set(retry_rows) != sample_ids:
            raise ValueError("--retry-scores eval_ids do not match the prepared dataset")

    forced_metrics = {item.strip() for item in (args.force_metrics or "").split(",") if item.strip()}
    unknown_metrics = sorted(forced_metrics - set(SEMANTIC_METRICS))
    if unknown_metrics:
        raise ValueError(f"--force-metrics contains unknown metric values: {', '.join(unknown_metrics)}")

    timeout_seconds = float(args.timeout_seconds)
    judge_client = AsyncOpenAI(
        api_key=judge_api_key,
        base_url=args.judge_base_url,
        timeout=timeout_seconds,
        max_retries=0,
    )
    answer_client: Any | None = None
    try:
        judge_llm = llm_factory(
            args.judge_model,
            client=judge_client,
            max_tokens=args.judge_max_tokens,
        )
        _configure_judge_diagnostics(
            judge_llm,
            thinking=args.judge_thinking,
            diagnostics=diagnostics,
        )
        embeddings = HuggingFaceEmbeddings(model=str(args.embedding_model_path), device=args.embedding_device)
        scorers = {
            "faithfulness": Faithfulness(llm=judge_llm),
            "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=embeddings, strictness=args.answer_relevancy_strictness),
        }
        rows: list[dict[str, Any]] = []
        for index, sample in enumerate(samples, start=1):
            previous = retry_rows.get(str(sample["eval_id"]))
            answer: str | None = None
            answer_error: str | None = None
            metrics: dict[str, float | None] = {name: None for name in SEMANTIC_METRICS}
            judge_reason: dict[str, str | None] = {name: None for name in SEMANTIC_METRICS}
            judge_error: dict[str, str | None] = {name: None for name in SEMANTIC_METRICS}
            metric_names: set[str] = set(SEMANTIC_METRICS)
            if previous:
                previous_answer = previous.get("answer")
                if isinstance(previous_answer, str) and previous_answer.strip():
                    answer = previous_answer.strip()
                    previous_metrics = previous.get("metrics", {})
                    previous_reason = previous.get("judge_reason", {})
                    previous_error = previous.get("judge_error", {})
                    metrics = {name: previous_metrics.get(name) for name in SEMANTIC_METRICS}
                    judge_reason = {name: previous_reason.get(name) for name in SEMANTIC_METRICS}
                    judge_error = {name: previous_error.get(name) for name in SEMANTIC_METRICS}
                    metric_names = {name for name, value in metrics.items() if value is None}
                    for name in forced_metrics:
                        metrics[name] = None
                        judge_reason[name] = None
                        judge_error[name] = None
                    metric_names.update(forced_metrics)

            if answer is None:
                if answer_client is None:
                    answer_api_key = os.environ.get(args.answer_api_key_env)
                    if not answer_api_key:
                        raise RuntimeError(f"{args.answer_api_key_env} is required when an answer must be generated")
                    answer_client = AsyncOpenAI(
                        api_key=answer_api_key,
                        base_url=args.answer_base_url,
                        timeout=timeout_seconds,
                        max_retries=0,
                    )
                answer, answer_error = await _generate_answer(
                    answer_client,
                    args.answer_model,
                    sample,
                    timeout_seconds,
                    args.answer_max_tokens,
                    args.answer_attempts,
                )
            if answer:
                for name in metric_names:
                    metrics[name], judge_reason[name], judge_error[name] = await _score_metric(
                        name,
                        scorers[name],
                        sample,
                        answer,
                        timeout_seconds,
                        args.judge_attempts,
                        diagnostics,
                    )
            row = {
                "eval_id": sample["eval_id"],
                "answer": answer,
                "answer_error": answer_error,
                "metrics": metrics,
                "judge_reason": judge_reason,
                "judge_error": judge_error,
            }
            rows.append(row)
            print(
                f"[{index}/{len(samples)}] {sample['eval_id']}: answer={'ok' if answer else 'failed'} "
                f"faithfulness={metrics['faithfulness']!r} answer_relevancy={metrics['answer_relevancy']!r}",
                flush=True,
            )
    finally:
        await judge_client.close()
        if answer_client is not None:
            await answer_client.close()

    output_root = _path(args.output_root)
    output = output_root / f"{run_id}.jsonl"
    _write_jsonl_atomic(output, rows)
    numeric = lambda metric: [row["metrics"][metric] for row in rows if isinstance(row["metrics"][metric], (int, float))]
    metric_coverage = {
        metric: {
            "scored_case_count": len(numeric(metric)),
            "case_count": len(rows),
            "coverage": f"{len(numeric(metric))}/{len(rows)}",
            "rate": len(numeric(metric)) / len(rows) if rows else None,
        }
        for metric in SEMANTIC_METRICS
    }
    summary = {
        "schema_version": "greenspec.ragas_answer_trial.v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        "case_count": len(rows),
        "diagnostic_eval_ids": selected_eval_ids,
        "diagnostic_force_metrics": sorted(forced_metrics) or None,
        "answer_generation": {
            "provider": args.answer_provider,
            "model": args.answer_model,
            "prompt_version": ANSWER_PROMPT_VERSION,
            "temperature": 0,
            "max_tokens": args.answer_max_tokens,
            "attempts": args.answer_attempts,
        },
        "judge": {
            "provider": args.judge_provider,
            "model": args.judge_model,
            "base_url": args.judge_base_url,
            "timeout_seconds": timeout_seconds,
            "max_tokens": args.judge_max_tokens,
            "attempts": args.judge_attempts,
            "thinking": args.judge_thinking,
        },
        "answer_relevancy_embedding": {"provider": "huggingface_local", "model_path": str(args.embedding_model_path), "device": args.embedding_device, "strictness": args.answer_relevancy_strictness},
        "metric_means": {metric: (fmean(values) if values else None) for metric in SEMANTIC_METRICS for values in [numeric(metric)]},
        "metric_coverage": metric_coverage,
        "answer_error_count": sum(row["answer_error"] is not None for row in rows),
        "judge_error_count": sum(any(error is not None for error in row["judge_error"].values()) for row in rows),
        "retried_from_scores_path": str(retry_path.relative_to(ROOT)) if retry_path else None,
        "scores_path": str(output.relative_to(ROOT)),
        "judge_diagnostics_path": str(diagnostics_path.relative_to(ROOT)) if diagnostics_path else None,
        "scope": "Frozen-evidence answer generation plus RAGAS Faithfulness and Answer Relevancy. This does not replace human citation accuracy/completeness review.",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate answers from frozen retrieved evidence and evaluate RAGAS answer metrics.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation/ragas/answer_runs"))
    parser.add_argument("--answer-provider", default=os.environ.get("RAGAS_ANSWER_PROVIDER", "deepseek"))
    parser.add_argument("--answer-model", default=os.environ.get("RAGAS_ANSWER_MODEL", os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4-flash")))
    parser.add_argument("--answer-base-url", default=os.environ.get("RAGAS_ANSWER_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--answer-api-key-env", default=os.environ.get("RAGAS_ANSWER_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--judge-provider", default=os.environ.get("RAGAS_JUDGE_PROVIDER", "deepseek"))
    parser.add_argument("--judge-model", default=os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--judge-base-url", default=os.environ.get("RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--judge-api-key-env", default=os.environ.get("RAGAS_JUDGE_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--embedding-model-path", type=Path, default=Path("/models/bge-base-zh-v1.5"))
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--answer-relevancy-strictness", type=int, default=3)
    parser.add_argument("--answer-max-tokens", type=int, default=700, help="Maximum tokens for each frozen-evidence answer.")
    parser.add_argument("--answer-attempts", type=int, default=2, help="Attempts per answer for transient failures or empty provider responses.")
    parser.add_argument("--judge-max-tokens", type=int, default=2048, help="Maximum tokens for each structured judge response.")
    parser.add_argument("--judge-attempts", type=int, default=2, help="Attempts per metric for transient judge errors only.")
    parser.add_argument("--judge-thinking", choices=("default", "disabled"), default="default", help="DeepSeek judge thinking mode; disabled is a diagnostic-only explicit request.")
    parser.add_argument("--judge-diagnostics-dir", type=Path, help="Append-only per-call Instructor diagnostics directory; includes prompts and partial model output.")
    parser.add_argument("--eval-ids", help="Comma-separated diagnostic subset of frozen eval IDs; omitted evaluates the whole dataset.")
    parser.add_argument("--force-metrics", help="Comma-separated metrics to recompute from --retry-scores while preserving frozen answers; diagnostic use only.")
    parser.add_argument("--retry-scores", type=Path, help="Retry only missing answers or null metrics from this prior scores JSONL, preserving successful answers and scores.")
    parser.add_argument("--timeout-seconds", type=float, default=float(os.environ.get("RAGAS_JUDGE_TIMEOUT_SECONDS", "60")))
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.answer_relevancy_strictness < 1:
        parser.error("--answer-relevancy-strictness must be positive")
    if args.answer_max_tokens < 1:
        parser.error("--answer-max-tokens must be positive")
    if args.answer_attempts < 1:
        parser.error("--answer-attempts must be positive")
    if args.judge_max_tokens < 1:
        parser.error("--judge-max-tokens must be positive")
    if args.judge_attempts < 1:
        parser.error("--judge-attempts must be positive")
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
