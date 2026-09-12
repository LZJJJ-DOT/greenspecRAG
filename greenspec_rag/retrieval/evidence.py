"""Day 5 reranking, Evidence Pack construction, citation gates, and evaluation."""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .bm25 import RetrievalError
from .hybrid import HybridRetriever

RERANK_BATCH_SIZE = 16
RERANK_QUERY_MAX_TOKENS = 128
RERANK_DOCUMENT_MAX_TOKENS = 384
RERANK_TIMEOUT_SECONDS = 60.0
RERANK_CANDIDATE_LIMIT = 30
EVIDENCE_PACK_MAX_ESTIMATED_TOKENS = 5000
EVIDENCE_PACK_PRIMARY_LIMIT = 8


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read an operational limit without making a malformed env fatal at import."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _bounded_env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(maximum, max(minimum, value))


def _estimate_tokens(value: str) -> int:
    """Conservative, tokenizer-independent estimate for Chinese downstream LLMs."""
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", str(value or "")))


class RerankerTimeout(TimeoutError):
    pass


class Reranker(Protocol):
    model_id: str
    device: str
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class BGEReranker:
    """Local bge-reranker-v2-m3 adapter with bounded input and timeout."""
    def __init__(self, model_path: str | Path, *, device: str = "cuda", batch_size: int = RERANK_BATCH_SIZE, timeout_seconds: float | None = None):
        try:
            from FlagEmbedding import FlagReranker
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - runtime image gate
            raise RetrievalError("dependency_unavailable", "FlagEmbedding and transformers are required for reranking", status_code=503, retryable=True) from exc
        self.model_id, self.device, self.batch_size = str(model_path), device, batch_size
        self.timeout_seconds = timeout_seconds or _bounded_env_float("RAG_RERANK_TIMEOUT_SECONDS", RERANK_TIMEOUT_SECONDS, minimum=5.0, maximum=180.0)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        try:
            self.model = FlagReranker(self.model_id, use_fp16=device.startswith("cuda"), devices=device)
        except TypeError:  # FlagEmbedding 1.3 compatibility
            self.model = FlagReranker(self.model_id, use_fp16=device.startswith("cuda"))
        # The first CUDA inference initializes kernels and can take longer than
        # the request budget on small local GPUs.  Pay that one-off cost while
        # constructing the dependency; every request still keeps the strict
        # timeout below and therefore remains fail-closed.
        try:
            self.model.compute_score([["warmup", "warmup"]], batch_size=1)
        except Exception as exc:
            raise RetrievalError(
                "dependency_unavailable",
                "reranker CUDA warmup failed",
                status_code=503,
                retryable=True,
            ) from exc

    def _truncate(self, text: str, limit: int) -> str:
        ids = self.tokenizer(str(text or ""), add_special_tokens=False, truncation=True, max_length=limit)["input_ids"]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def score(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[self._truncate(query, RERANK_QUERY_MAX_TOKENS), self._truncate(text, RERANK_DOCUMENT_MAX_TOKENS)] for text in texts]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.model.compute_score, pairs, batch_size=self.batch_size)
            try:
                scores = future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError as exc:
                raise RerankerTimeout(f"reranker exceeded {self.timeout_seconds}s") from exc
        # FlagEmbedding versions return either a Python list or a NumPy array
        # for batched pairs.  Normalize both before enforcing one score/pair.
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        values = [float(score) for score in (scores if isinstance(scores, (list, tuple)) else [scores])]
        if len(values) != len(texts):
            raise RetrievalError("dependency_unavailable", "reranker returned a score count different from candidates", status_code=503)
        return values


def _registry(path: Path) -> dict[str, dict[str, Any]]:
    return {item["standard_id"]: item for item in json.loads(path.read_text(encoding="utf-8")) if item.get("standard_id")}


class EvidencePackBuilder:
    def __init__(self, *, registry_path: Path, relations_path: Path | None = None, schema_path: Path | None = None, max_context_tokens: int | None = None, primary_item_limit: int | None = None):
        self.registry = _registry(Path(registry_path))
        self.relations_path = Path(relations_path) if relations_path else None
        self.schema_path = schema_path or Path(__file__).resolve().parents[2] / "contracts" / "schemas" / "evidence_pack.schema.json"
        self.max_context_tokens = max_context_tokens or _bounded_env_int("RAG_EVIDENCE_PACK_MAX_TOKENS", EVIDENCE_PACK_MAX_ESTIMATED_TOKENS, minimum=512, maximum=16000)
        self.primary_item_limit = primary_item_limit or _bounded_env_int("RAG_EVIDENCE_PACK_PRIMARY_LIMIT", EVIDENCE_PACK_PRIMARY_LIMIT, minimum=1, maximum=20)

    def build(self, *, request_id: str | None, retrieval_run_id: str, index_manifest_id: str | None, query: str, items: list[dict[str, Any]], filters_applied: Mapping[str, Any], warnings: list[dict[str, Any]], degraded_modes: list[str], parent_loader: Callable[[str], Mapping[str, Any]] | None = None) -> dict[str, Any]:
        enriched = [self._enrich_parent(item, parent_loader) for item in items]
        selected, context_budget, budget_warning = self._select_with_context_budget(query, enriched)
        effective_warnings = [*warnings, *([budget_warning] if budget_warning else [])]
        citations = [self._citation(item) for item in selected]
        pack = {"schema_version": "evidence_pack.v1", "evidence_pack_id": f"pack_{uuid.uuid4().hex}", "request_id": request_id, "retrieval_run_id": retrieval_run_id, "index_manifest_id": index_manifest_id, "query": query, "items": selected, "groups": self._groups(selected), "citations": citations, "warnings": effective_warnings, "needs_manual_review": bool(effective_warnings or degraded_modes or any(item["verification_status"] != "verified" for item in selected)), "missing_facts": self._missing_facts(selected), "degraded_modes": degraded_modes, "filters_applied": dict(filters_applied), "context_budget": context_budget}
        self.validate(pack)
        return pack

    @staticmethod
    def _is_attached_support(item: Mapping[str, Any]) -> bool:
        return bool((item.get("trace") or {}).get("evidence_expansion"))

    @staticmethod
    def _item_estimated_tokens(item: Mapping[str, Any]) -> int:
        parent_text = " ".join(str(parent.get("text", "")) for parent in item.get("parent_context", []) if isinstance(parent, Mapping))
        citation_locator = " ".join(str(item.get(field, "")) for field in ("standard_name", "standard_version", "clause_no", "pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end", "table_id", "formula_id"))
        return 32 + _estimate_tokens(item.get("text", "")) + _estimate_tokens(parent_text) + _estimate_tokens(citation_locator)

    def _select_with_context_budget(self, query: str, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
        """Keep the highest-ranked primary evidence and its structural dependencies.

        `required_evidence_ids` are evaluation-only labels and are deliberately
        unavailable to the serving API.  At runtime, direct table/formula
        children are the only evidence dependencies known deterministically.
        """
        primary = [item for item in items if not self._is_attached_support(item)]
        selected_primary = primary[:self.primary_item_limit]
        selected_primary_ids = {item["clause_id"] for item in selected_primary}
        support_by_parent: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            expansion = (item.get("trace") or {}).get("evidence_expansion") or {}
            parent_id = expansion.get("parent_clause_id")
            if self._is_attached_support(item) and parent_id in selected_primary_ids:
                support_by_parent.setdefault(parent_id, []).append(item)

        selected: list[dict[str, Any]] = []
        omitted_for_budget: list[str] = []
        used = _estimate_tokens(query)
        for primary_item in selected_primary:
            related = [primary_item, *support_by_parent.get(primary_item["clause_id"], [])]
            if used + self._item_estimated_tokens(primary_item) > self.max_context_tokens:
                omitted_for_budget.extend(item["evidence_id"] for item in related)
                continue
            selected.append(primary_item)
            used += self._item_estimated_tokens(primary_item)
            for support in related[1:]:
                cost = self._item_estimated_tokens(support)
                if used + cost <= self.max_context_tokens:
                    selected.append(support)
                    used += cost
                else:
                    omitted_for_budget.append(support["evidence_id"])

        selected_ids = {item["evidence_id"] for item in selected}
        omitted_by_primary_limit = [item["evidence_id"] for item in primary[self.primary_item_limit:]]
        for item in items:
            if self._is_attached_support(item) and item["evidence_id"] not in selected_ids and item["evidence_id"] not in omitted_for_budget:
                omitted_by_primary_limit.append(item["evidence_id"])
        budget = {
            "max_estimated_tokens": self.max_context_tokens,
            "estimated_tokens": used,
            "estimation": "conservative CJK-character and lexical-unit estimate; not a provider tokenizer count",
            "primary_item_limit": self.primary_item_limit,
            "input_item_count": len(items),
            "selected_item_count": len(selected),
            "omitted_by_primary_limit_evidence_ids": omitted_by_primary_limit,
            "omitted_for_budget_evidence_ids": omitted_for_budget,
        }
        warning = None
        if omitted_for_budget:
            warning = {
                "code": "evidence_pack_context_budget_exceeded",
                "message": "EvidencePack omitted lower-priority evidence because its bounded context budget was reached.",
                "severity": "warning",
                "evidence_ids": omitted_for_budget,
            }
        return selected, budget, warning

    def _enrich_parent(self, item: dict[str, Any], loader: Callable[[str], Mapping[str, Any]] | None) -> dict[str, Any]:
        copied = dict(item)
        parent_id = copied.get("parent_id")
        if parent_id and loader:
            parent = loader(parent_id)
            copied["parent_context"] = [{"clause_id": parent.get("clause_id"), "clause_no": parent.get("clause_no"), "content_type": parent.get("content_type"), "text": parent.get("text")}]
        else:
            copied["parent_context"] = list(copied.get("parent_context") or [])
        return copied

    def _citation(self, item: Mapping[str, Any]) -> dict[str, Any]:
        if item.get("evidence_id") != item.get("clause_id"):
            raise RetrievalError("index_build_failed", "evidence_id must equal clause_id", status_code=422, details={"clause_id": item.get("clause_id")})
        standard = self.registry.get(item.get("standard_id"))
        required = ["standard_id", "standard_name", "standard_version", "source_file", "source_sha256", "pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end"]
        missing = [field for field in required if item.get(field) in (None, "")]
        if not standard or missing or standard.get("standard_version") != item.get("standard_version"):
            raise RetrievalError("index_build_failed", "citation provenance is incomplete or standard version mismatches registry", status_code=422, details={"clause_id": item.get("clause_id"), "missing": missing})
        if item.get("content_type") in {"normative_table", "commentary_table"} and not item.get("table_id"):
            raise RetrievalError("index_build_failed", "table citation needs table_id", status_code=422)
        if item.get("content_type") == "normative_formula" and not item.get("formula_id"):
            raise RetrievalError("index_build_failed", "formula citation needs formula_id", status_code=422)
        return {"evidence_id": item["evidence_id"], "standard_number": standard["standard_number"], "standard_name": item["standard_name"], "standard_version": item["standard_version"], "clause_no": item.get("clause_no"), "pdf_page_start": item["pdf_page_start"], "pdf_page_end": item["pdf_page_end"], "printed_page_start": item["printed_page_start"], "printed_page_end": item["printed_page_end"], "table_id": item.get("table_id"), "formula_id": item.get("formula_id")}

    @staticmethod
    def _groups(items: list[Mapping[str, Any]]) -> dict[str, list[str]]:
        groups = {"primary_normative": [], "supporting_table_formula": [], "version_relation": [], "project_facts": [], "policy": []}
        for item in items:
            if item["content_type"] in {"normative_table", "commentary_table", "normative_formula"}:
                groups["supporting_table_formula"].append(item["evidence_id"])
            elif str(item["content_type"]).startswith("normative") or item["content_type"] in {"appendix", "appendix_clause"}:
                groups["primary_normative"].append(item["evidence_id"])
        return groups

    @staticmethod
    def _missing_facts(items: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            for fact in item.get("missing_facts", []):
                key = (fact["fact"], fact["reason"])
                target = unique.setdefault(key, {"fact": fact["fact"], "reason": fact["reason"], "required_for": []})
                target["required_for"].extend(fact.get("required_for", []))
        return [{**value, "required_for": sorted(set(value["required_for"]))} for value in unique.values()]

    def validate(self, pack: Mapping[str, Any]) -> None:
        try:
            import jsonschema
        except ImportError as exc:  # pragma: no cover
            raise RetrievalError("dependency_unavailable", "jsonschema is required to validate Evidence Pack", status_code=503) from exc
        schema = json.loads(Path(self.schema_path).read_text(encoding="utf-8"))
        errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(pack), key=lambda error: list(error.path))
        if errors:
            raise RetrievalError("index_build_failed", "Evidence Pack schema validation failed", status_code=422, details={"error": errors[0].message, "path": list(errors[0].path)})


class CompleteRetriever:
    def __init__(
            self,
            *,
            hybrid: HybridRetriever,
            reranker: Reranker,
            evidence_builder: EvidencePackBuilder,
    ):
        self.hybrid = hybrid
        self.reranker = reranker
        self.evidence_builder = evidence_builder

    def _attach_direct_supporting_evidence(
            self,
            items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Attach direct table/formula evidence after ranking.

        The attached items are structural dependencies of an already selected
        clause. They are not re-ranked similarity candidates, so retrieval
        rank/score fields remain null and the trace records their origin.
        """
        parent_ids = [
            item["clause_id"]
            for item in items
            if item.get("content_type") in {
                "normative_body",
                "normative_clause",
                "commentary",
                "commentary_clause",
                "appendix",
                "appendix_clause",
            }
        ]

        supporting_by_parent = self.hybrid.supporting_nodes_for(parent_ids)
        if not supporting_by_parent:
            return items, 0

        existing_ids = {
            item["clause_id"]
            for item in items
            if item.get("clause_id")
        }
        expanded: list[dict[str, Any]] = []
        attached_count = 0

        for item in items:
            expanded.append(item)
            parent_id = item.get("clause_id")

            for node in supporting_by_parent.get(parent_id, []):
                clause_id = node.get("clause_id")
                if not isinstance(clause_id, str) or clause_id in existing_ids:
                    continue

                supporting_item = HybridRetriever._to_item(node)
                supporting_item["trace"]["evidence_expansion"] = {
                    "kind": "direct_child_table_or_formula",
                    "parent_clause_id": parent_id,
                    "reason": (
                        "Attached as a direct normative table/formula child "
                        "of a retrieved clause."
                    ),
                }

                expanded.append(supporting_item)
                existing_ids.add(clause_id)
                attached_count += 1

        return expanded, attached_count

    def retrieve(
            self,
            request: Mapping[str, Any],
            *,
            request_id: str | None = None,
            retrieval_run_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        run_id = retrieval_run_id or f"retrieve_{uuid.uuid4().hex}"
        requested_top_k = request.get("top_k")
        configured_candidate_limit = _bounded_env_int(
            "RAG_RERANK_CANDIDATE_LIMIT",
            RERANK_CANDIDATE_LIMIT,
            minimum=1,
            maximum=100,
        )
        rerank_candidate_limit = max(
            configured_candidate_limit,
            requested_top_k if isinstance(requested_top_k, int) and not isinstance(requested_top_k, bool) else 1,
        )

        base = self.hybrid.retrieve(
            request,
            mode="hybrid",
            require_degraded=False,
            include_reranker_unavailable=False,
            result_limit=rerank_candidate_limit,
        )
        rerank_candidate_count = len(base["items"])

        try:
            if not base["items"]:
                base["warnings"].append({
                    "code": "no_retrieval_candidates",
                    "message": "No evidence matched the request filters; reranking was skipped.",
                    "severity": "warning",
                    "evidence_ids": [],
                })
            else:
                scores = self.reranker.score(
                    request["query"],
                    [item["text"] for item in base["items"]],
                )
                for item, score in zip(base["items"], scores):
                    item["rerank_score"] = score

                base["items"].sort(
                    key=lambda item: (
                        -float(item["rerank_score"]),
                        item["clause_id"],
                    )
                )

        except (RerankerTimeout, RetrievalError) as exc:
            if not request.get("allow_degraded"):
                code = (
                    "reranker_timeout"
                    if isinstance(exc, RerankerTimeout)
                    else "dependency_unavailable"
                )
                raise RetrievalError(
                    code,
                    str(exc),
                    status_code=503,
                    retryable=True,
                ) from exc

            base["degraded_modes"].append(
                "reranker_timeout"
                if isinstance(exc, RerankerTimeout)
                else "reranker_unavailable"
            )
            base["warnings"].append(
                {
                    "code": base["degraded_modes"][-1],
                    "message": str(exc),
                    "severity": "warning",
                    "evidence_ids": [],
                }
            )

        # The public retrieval response still returns precisely the requested
        # ranked TopK. Structural expansion happens only after this cut.
        base["items"] = base["items"][:requested_top_k]
        ranked_item_count = len(base["items"])
        expanded_items, attached_count = self._attach_direct_supporting_evidence(
            base["items"]
        )
        base["items"] = expanded_items
        base["filters_applied"]["evidence_expansion"] = {
            "ranked_item_count": ranked_item_count,
            "attached_supporting_item_count": attached_count,
            "rule": "direct_normative_table_formula_children",
        }
        base["filters_applied"]["rerank"] = {
            "candidate_limit": rerank_candidate_limit,
            "candidate_count": rerank_candidate_count,
            "returned_ranked_item_count": ranked_item_count,
        }

        pack = self.evidence_builder.build(
            request_id=request_id,
            retrieval_run_id=run_id,
            index_manifest_id=self.hybrid.manifest["index_manifest_id"],
            query=request["query"],
            items=base["items"],
            filters_applied=base["filters_applied"],
            warnings=base["warnings"],
            degraded_modes=base["degraded_modes"],
            parent_loader=self.hybrid._load_node,
        )
        return base, pack

    def retrieve(
            self,
            request: Mapping[str, Any],
            *,
            request_id: str | None = None,
            retrieval_run_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        run_id = retrieval_run_id or f"retrieve_{uuid.uuid4().hex}"
        requested_top_k = request.get("top_k")
        configured_candidate_limit = _bounded_env_int(
            "RAG_RERANK_CANDIDATE_LIMIT",
            RERANK_CANDIDATE_LIMIT,
            minimum=1,
            maximum=100,
        )
        rerank_candidate_limit = max(
            configured_candidate_limit,
            requested_top_k if isinstance(requested_top_k, int) and not isinstance(requested_top_k, bool) else 1,
        )

        base = self.hybrid.retrieve(
            request,
            mode="hybrid",
            require_degraded=False,
            include_reranker_unavailable=False,
            result_limit=rerank_candidate_limit,
        )
        rerank_candidate_count = len(base["items"])

        try:
            if not base["items"]:
                base["warnings"].append({
                    "code": "no_retrieval_candidates",
                    "message": "No evidence matched the request filters; reranking was skipped.",
                    "severity": "warning",
                    "evidence_ids": [],
                })
            else:
                scores = self.reranker.score(
                    request["query"],
                    [item["text"] for item in base["items"]],
                )
                for item, score in zip(base["items"], scores):
                    item["rerank_score"] = score

                base["items"].sort(
                    key=lambda item: (
                        -float(item["rerank_score"]),
                        item["clause_id"],
                    )
                )

        except (RerankerTimeout, RetrievalError) as exc:
            if not request.get("allow_degraded"):
                code = (
                    "reranker_timeout"
                    if isinstance(exc, RerankerTimeout)
                    else "dependency_unavailable"
                )
                raise RetrievalError(
                    code,
                    str(exc),
                    status_code=503,
                    retryable=True,
                ) from exc

            base["degraded_modes"].append(
                "reranker_timeout"
                if isinstance(exc, RerankerTimeout)
                else "reranker_unavailable"
            )
            base["warnings"].append(
                {
                    "code": base["degraded_modes"][-1],
                    "message": str(exc),
                    "severity": "warning",
                    "evidence_ids": [],
                }
            )

        # The public retrieval response still returns precisely the requested
        # ranked TopK. Structural expansion happens only after this cut.
        base["items"] = base["items"][:requested_top_k]
        ranked_item_count = len(base["items"])
        expanded_items, attached_count = self._attach_direct_supporting_evidence(
            base["items"]
        )
        base["items"] = expanded_items
        base["filters_applied"]["evidence_expansion"] = {
            "ranked_item_count": ranked_item_count,
            "attached_supporting_item_count": attached_count,
            "rule": "direct_normative_table_formula_children",
        }
        base["filters_applied"]["rerank"] = {
            "candidate_limit": rerank_candidate_limit,
            "candidate_count": rerank_candidate_count,
            "returned_ranked_item_count": ranked_item_count,
        }

        pack = self.evidence_builder.build(
            request_id=request_id,
            retrieval_run_id=run_id,
            index_manifest_id=self.hybrid.manifest["index_manifest_id"],
            query=request["query"],
            items=base["items"],
            filters_applied=base["filters_applied"],
            warnings=base["warnings"],
            degraded_modes=base["degraded_modes"],
            parent_loader=self.hybrid._load_node,
        )
        return base, pack


def evaluate_cases(cases: list[Mapping[str, Any]], retrieve_pack: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate only labeled metrics; empty gold/human fields stay explicitly unavailable."""
    outcomes, metric_samples = [], {"clause_hit_at_10": [], "citation_accuracy": [], "citation_completeness": [], "manual_review_recall": []}
    for case in cases:
        pack = retrieve_pack(case)
        ids = {item["evidence_id"] for item in pack["items"][:10]}
        gold = set(case.get("gold_evidence_ids", []))
        if gold:
            metric_samples["clause_hit_at_10"].append(bool(ids & gold))
        human = case.get("human_annotation") or {}
        for field, metric in (("citation_accuracy", "citation_accuracy"), ("citation_completeness", "citation_completeness"), ("needs_manual_review", "manual_review_recall")):
            if isinstance(human.get(field), bool):
                metric_samples[metric].append(human[field] == bool(pack["needs_manual_review"]) if metric == "manual_review_recall" else human[field])
        outcomes.append({"eval_id": case["eval_id"], "retrieval_run_id": pack["retrieval_run_id"], "item_count": len(pack["items"]), "gold_labeled": bool(gold), "human_labeled": any(isinstance(value, bool) for value in human.values())})
    metrics = {name: (sum(values) / len(values) if values else None) for name, values in metric_samples.items()}
    unmet = [name for name, threshold in {"clause_hit_at_10": .85, "citation_accuracy": .95, "citation_completeness": .90}.items() if metrics[name] is None or metrics[name] < threshold]
    unresolved = [message for message in ["gold_evidence_ids are required for Clause Hit@10" if metrics["clause_hit_at_10"] is None else "", "human citation annotations are required for Citation Accuracy/Completeness" if metrics["citation_accuracy"] is None or metrics["citation_completeness"] is None else ""] if message]
    return {"evaluation_run_id": f"eval_{uuid.uuid4().hex}", "created_at": datetime.now(timezone.utc).isoformat(), "case_count": len(cases), "outcomes": outcomes, "metrics": metrics, "publication_status": "baseline" if not unmet else "experimental", "unresolved": unresolved + [f"threshold not met: {name}" for name in unmet]}
