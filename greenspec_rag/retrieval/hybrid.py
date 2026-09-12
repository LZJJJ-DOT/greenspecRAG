"""Day 4 dense retrieval, deterministic RRF fusion, and diversity controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from .bm25 import BM25Retriever, RetrievalError, _read_jsonl, tokenize_chinese

RRF_K = 60
RECALL_LIMIT = 50
JACCARD_THRESHOLD = 0.92
SPECIAL_SIBLINGS = {"normative_table", "commentary_table", "normative_formula"}
SEARCHABLE_COMMENTARY_TYPES = {"commentary_table"}
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _portable_path(value: str | Path) -> Path:
    """Read relative manifest paths produced on either Windows or Linux."""
    path = Path(value)
    if path.exists() or "\\" not in str(value):
        return path
    return Path(str(value).replace("\\", "/"))


class Embedder(Protocol):
    model_id: str
    device: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local BGE adapter; loading weights remains an explicit runtime gate."""
    def __init__(self, model_path: str | Path, *, device: str | None = None):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on deployment image
            raise RetrievalError("dependency_unavailable", "sentence-transformers and torch are required for dense retrieval", status_code=503, retryable=True) from exc
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(str(model_path), device=selected_device)
        self.model_id, self.device = str(model_path), selected_device
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if self.dimension <= 0:
            raise RetrievalError("dependency_unavailable", "embedding model reported an invalid vector dimension", status_code=503)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        result = [[float(value) for value in vector] for vector in vectors]
        _validate_vectors(result, self.dimension)
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([BGE_QUERY_INSTRUCTION + query])[0]


def _validate_vectors(vectors: Iterable[Iterable[float]], dimension: int) -> None:
    for vector in vectors:
        values = list(vector)
        if len(values) != dimension:
            raise RetrievalError("index_build_failed", "embedding vector dimension changed", status_code=422, details={"expected": dimension, "actual": len(values)})
        norm = math.sqrt(sum(value * value for value in values))
        if not 0.999 <= norm <= 1.001:
            raise RetrievalError("index_build_failed", "embedding vector is not normalized", status_code=422, details={"norm": norm})


def _sha256_text(text: str) -> str:
    return hashlib.sha256(" ".join(str(text or "").split()).encode("utf-8")).hexdigest()


def _payload(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "clause_id": node["clause_id"], "parent_id": node.get("parent_id"), "standard_id": node.get("standard_id"),
        "content_type": node["content_type"], "document_status": node.get("document_status"), "indexable": bool(node.get("indexable")),
        "pdf_page_start": node.get("pdf_page_start"), "pdf_page_end": node.get("pdf_page_end"),
        "printed_page_start": node.get("printed_page_start"), "printed_page_end": node.get("printed_page_end"),
        "verification_status": node.get("verification_status"), "clause_no": node.get("clause_no"),
        "text_sha256": _sha256_text(node.get("text", "")), "token_count": len(tokenize_chinese(node.get("retrieval_text", ""))),
    }


def _qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RetrievalError("dependency_unavailable", "qdrant-client is required for dense retrieval", status_code=503, retryable=True) from exc
    return models


@dataclass(frozen=True)
class BuiltHybridIndex:
    index_manifest_id: str
    manifest_path: Path
    collection_name: str


class HybridIndexBuilder:
    def __init__(self, *, canonical_path: Path, bm25_manifest_path: Path, output_root: Path, qdrant_client: Any, embedder: Embedder, batch_size: int = 32, qdrant_models: Any | None = None):
        self.canonical_path, self.bm25_manifest_path, self.output_root = Path(canonical_path), Path(bm25_manifest_path), Path(output_root)
        self.qdrant_client, self.embedder, self.batch_size = qdrant_client, embedder, batch_size
        self.qdrant_models = qdrant_models

    def build(self) -> BuiltHybridIndex:
        bm25_manifest = json.loads(self.bm25_manifest_path.read_text(encoding="utf-8"))
        canonical_sha = hashlib.sha256(self.canonical_path.read_bytes()).hexdigest()
        if bm25_manifest.get("canonical_sha256") != canonical_sha:
            raise RetrievalError("index_build_failed", "BM25 and dense indexes do not share canonical JSONL", status_code=422)
        nodes = [node for node in _read_jsonl(self.canonical_path) if node.get("indexable") and (not str(node.get("content_type", "")).startswith("commentary") or node.get("content_type") in SEARCHABLE_COMMENTARY_TYPES)]
        if not nodes or len({node["clause_id"] for node in nodes}) != len(nodes):
            raise RetrievalError("index_build_failed", "canonical indexable nodes are empty or have duplicate IDs", status_code=422)
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_id = f"hybrid_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        collection = f"greenspec_{manifest_id}".lower()
        temp_dir = Path(tempfile.mkdtemp(prefix=".building-", dir=self.output_root))
        created = False
        try:
            self._create_collection(collection)
            created = True
            point_count = self._upsert(collection, nodes)
            if point_count != len(nodes):
                raise RetrievalError("index_build_failed", "Qdrant point count does not match canonical nodes", status_code=422, details={"expected": len(nodes), "actual": point_count})
            manifest = {
                "index_manifest_id": manifest_id, "kind": "hybrid_bm25_qdrant", "created_at": datetime.now(timezone.utc).isoformat(),
                "canonical_path": str(self.canonical_path), "canonical_sha256": canonical_sha,
                "bm25_manifest_id": bm25_manifest["index_manifest_id"], "bm25_manifest_path": str(self.bm25_manifest_path),
                "qdrant_collection": collection, "qdrant_vector_size": self.embedder.dimension,
                "embedding": {"model_id": self.embedder.model_id, "device": self.embedder.device, "normalized": True, "query_instruction": BGE_QUERY_INSTRUCTION},
                "indexed_node_count": len(nodes), "recall_limit": RECALL_LIMIT, "rrf_k": RRF_K,
                "diversity": {"text_hash": "sha256", "token_jaccard_threshold": JACCARD_THRESHOLD, "parent_limit": 3, "parent_limit_exempt_content_types": sorted(SPECIAL_SIBLINGS)},
                "build_log": {"canonical_nodes_seen": len(nodes), "dense_points_written": point_count, "duplicate_clause_ids": 0, "commentary_indexed": sum(1 for node in nodes if str(node.get("content_type", "")).startswith("commentary"))},
            }
            (temp_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            final_dir = self.output_root / manifest_id
            os.replace(temp_dir, final_dir)
            active_temp = self.output_root / f".active-{uuid.uuid4().hex}.json"
            active_temp.write_text(json.dumps({"index_manifest_id": manifest_id, "manifest_path": str(final_dir / "index_manifest.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(active_temp, self.output_root / "active.json")
            return BuiltHybridIndex(manifest_id, final_dir / "index_manifest.json", collection)
        except Exception:
            if created:
                self._delete_collection(collection)
            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    item.unlink()
                temp_dir.rmdir()
            raise

    def _create_collection(self, collection: str) -> None:
        models = self.qdrant_models or _qdrant_models()
        self.qdrant_client.create_collection(collection_name=collection, vectors_config=models.VectorParams(size=self.embedder.dimension, distance=models.Distance.COSINE))

    def _delete_collection(self, collection: str) -> None:
        try:
            self.qdrant_client.delete_collection(collection_name=collection)
        except Exception:
            pass

    def _upsert(self, collection: str, nodes: list[dict[str, Any]]) -> int:
        models = self.qdrant_models or _qdrant_models()
        for start in range(0, len(nodes), self.batch_size):
            batch = nodes[start:start + self.batch_size]
            vectors = self.embedder.embed_documents([node["retrieval_text"] for node in batch])
            _validate_vectors(vectors, self.embedder.dimension)
            points = [models.PointStruct(id=index + start, vector=vector, payload=_payload(node)) for index, (node, vector) in enumerate(zip(batch, vectors))]
            self.qdrant_client.upsert(collection_name=collection, points=points, wait=True)
        info = self.qdrant_client.get_collection(collection_name=collection)
        point_count = int(getattr(info, "points_count", getattr(getattr(info, "result", None), "points_count", 0)))
        # A successful upsert is not enough: a client/server serialization
        # mismatch can otherwise publish an all-zero dense collection.
        if hasattr(self.qdrant_client, "retrieve"):
            stored = self.qdrant_client.retrieve(collection_name=collection, ids=[0], with_vectors=True)
            if not stored:
                raise RetrievalError("index_build_failed", "Qdrant did not return the dense write probe", status_code=422)
            vector = getattr(stored[0], "vector", None)
            if isinstance(vector, Mapping):
                vector = next(iter(vector.values()), None)
            try:
                _validate_vectors([vector or []], self.embedder.dimension)
            except RetrievalError as exc:
                raise RetrievalError("index_build_failed", "Qdrant dense write probe is invalid", status_code=422, details=exc.details) from exc
        return point_count


def _dedupe(items: Iterable[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        clause_id = item["clause_id"]
        if clause_id not in seen or item.get(score_key, float("-inf")) > seen[clause_id].get(score_key, float("-inf")):
            seen[clause_id] = item
    return list(seen.values())


def rrf_merge(bm25_items: list[dict[str, Any]], dense_items: list[dict[str, Any]], *, k: int = RRF_K) -> list[dict[str, Any]]:
    """Deduplicate each channel, use 1-based ranks, and expose both raw scores."""
    bm25 = _dedupe(bm25_items, "bm25_score")[:RECALL_LIMIT]
    dense = _dedupe(dense_items, "dense_score")[:RECALL_LIMIT]
    merged: dict[str, dict[str, Any]] = {}
    for rank, item in enumerate(bm25, 1):
        target = merged.setdefault(item["clause_id"], {**item, "bm25_rank": None, "dense_rank": None, "bm25_score": None, "dense_score": None, "rrf_score": 0.0})
        target["bm25_rank"], target["bm25_score"] = rank, item.get("bm25_score")
        target["rrf_score"] += 1.0 / (k + rank)
    for rank, item in enumerate(dense, 1):
        target = merged.setdefault(item["clause_id"], {**item, "bm25_rank": None, "dense_rank": None, "bm25_score": None, "dense_score": None, "rrf_score": 0.0})
        target["dense_rank"], target["dense_score"] = rank, item.get("dense_score")
        target["rrf_score"] += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda item: (-item["rrf_score"], item["clause_id"]))


def apply_diversity(items: Iterable[dict[str, Any]], *, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, removed, hashes, parent_counts, token_sets = [], [], set(), {}, []
    for item in items:
        text_hash = _sha256_text(item.get("text", ""))
        tokens = set(tokenize_chinese(item.get("retrieval_text", item.get("text", ""))))
        reason = None
        if text_hash in hashes:
            reason = "exact_text_hash"
        elif tokens and any(len(tokens & existing) / len(tokens | existing) >= JACCARD_THRESHOLD for existing in token_sets if tokens | existing):
            reason = "token_jaccard"
        parent_id = item.get("parent_id")
        if not reason and parent_id and item.get("content_type") not in SPECIAL_SIBLINGS and parent_counts.get(parent_id, 0) >= 3:
            reason = "parent_cap"
        if reason:
            removed.append({"clause_id": item["clause_id"], "reason": reason})
            continue
        hashes.add(text_hash)
        token_sets.append(tokens)
        if parent_id and item.get("content_type") not in SPECIAL_SIBLINGS:
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        accepted.append(item)
        if len(accepted) >= limit:
            break
    return accepted, removed


class HybridRetriever:
    def __init__(self, *, hybrid_manifest_path: Path, qdrant_client: Any, embedder: Embedder, registry_path: Path):
        self.manifest_path = Path(hybrid_manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.qdrant_client, self.embedder = qdrant_client, embedder
        self.bm25 = BM25Retriever(manifest_path=_portable_path(self.manifest["bm25_manifest_path"]), registry_path=registry_path)
        if self.embedder.dimension != self.manifest["qdrant_vector_size"]:
            raise RetrievalError("dependency_unavailable", "embedding dimension differs from active Qdrant collection", status_code=503, retryable=True)

    def retrieve(self, request: Mapping[str, Any], *, mode: str = "hybrid", require_degraded: bool = True, include_reranker_unavailable: bool = True, result_limit: int | None = None) -> dict[str, Any]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise RetrievalError("invalid_request", "mode must be bm25, dense, or hybrid")
        validated = self.bm25._validate_request(request, require_degraded=require_degraded)
        bm25_result = self.bm25.retrieve(request, result_limit=RECALL_LIMIT, require_degraded=require_degraded)
        bm25_items = [{**item, "bm25_score": item["trace"]["bm25_score"], "retrieval_text": item["text"]} for item in bm25_result["items"]]
        dense_items = self._dense_items(validated) if mode in {"dense", "hybrid"} else []
        if mode == "bm25":
            merged = rrf_merge(bm25_items, [])
        elif mode == "dense":
            merged = rrf_merge([], dense_items)
        else:
            merged = rrf_merge(bm25_items, dense_items)
        # CompleteRetriever uses a larger internal limit before reranking.
        # Public callers retain the request top_k behaviour by default.
        limit = validated["top_k"] if result_limit is None else result_limit
        if isinstance(limit, bool) or not isinstance(limit, int) or not validated["top_k"] <= limit <= 100:
            raise RetrievalError("invalid_request", "result_limit must be an integer from request top_k through 100", details={"field": "result_limit"})
        diverse, removed = apply_diversity(merged, limit=limit)
        output = [self._to_item(item) for item in diverse]
        fusion_trace = {"rrf_k": RRF_K, "dedupe": {"bm25_input": len(bm25_items), "dense_input": len(dense_items), "merged": len(merged)}, "diversity_limit": limit, "diversity_removed": removed}
        warnings = [{"code": "reranker_unavailable", "message": "Day 4 hybrid retrieval has no reranker yet.", "severity": "warning", "evidence_ids": []}] if include_reranker_unavailable else []
        degraded_modes = ["reranker_unavailable"] if include_reranker_unavailable else []
        return {"items": output, "filters_applied": {**bm25_result["filters_applied"], "bm25_candidate_count": len(bm25_items), "dense_candidate_count": len(dense_items), "mode": mode, "fusion_trace": fusion_trace}, "warnings": warnings, "degraded_modes": degraded_modes}

    def _dense_items(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        vector = self.embedder.embed_query(request["query"])
        _validate_vectors([vector], self.embedder.dimension)
        collection = self.manifest["qdrant_collection"]
        if hasattr(self.qdrant_client, "query_points"):
            raw = self.qdrant_client.query_points(collection_name=collection, query=vector, limit=RECALL_LIMIT * 4).points
        else:
            raw = self.qdrant_client.search(collection_name=collection, query_vector=vector, limit=RECALL_LIMIT * 4)
        items = []
        for point in raw:
            payload = dict(point.payload or {})
            node = self._load_node(payload["clause_id"])
            if self.bm25._hard_match(node, request):
                items.append({**node, "dense_score": float(point.score)})
            if len(items) >= RECALL_LIMIT:
                break
        return _dedupe(items, "dense_score")

    def _load_node(self, clause_id: str) -> dict[str, Any]:
        import sqlite3
        connection = sqlite3.connect(self.bm25.database_path)
        try:
            row = connection.execute("SELECT raw_json FROM nodes WHERE clause_id = ?", (clause_id,)).fetchone()
            if not row:
                raise RetrievalError("dependency_unavailable", "Qdrant point is absent from active BM25 index", status_code=503, details={"clause_id": clause_id})
            return json.loads(row[0])
        finally:
            connection.close()

    def supporting_nodes_for(
            self,
            parent_clause_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Load direct normative table/formula evidence for selected clauses.

        A returned table continuation is attached to the clause that owns the
        same table_id. This is structural evidence expansion, not a new
        similarity-search candidate.
        """
        parent_ids = sorted(
            {
                str(clause_id)
                for clause_id in parent_clause_ids
                if isinstance(clause_id, str) and clause_id
            }
        )
        if not parent_ids:
            return {}

        import sqlite3

        parent_placeholders = ", ".join("?" for _ in parent_ids)
        connection = sqlite3.connect(self.bm25.database_path)
        try:
            direct_rows = connection.execute(
                f"""
                SELECT raw_json
                FROM nodes
                WHERE parent_id IN ({parent_placeholders})
                  AND content_type IN ('normative_table', 'commentary_table', 'normative_formula')
                ORDER BY clause_id
                """,
                parent_ids,
            ).fetchall()

            direct_nodes = [json.loads(row[0]) for row in direct_rows]
            grouped: dict[str, list[dict[str, Any]]] = {
                parent_id: [] for parent_id in parent_ids
            }

            # First include tables/formulas directly owned by a retrieved clause.
            table_owners: dict[str, set[str]] = {}
            for node in direct_nodes:
                parent_id = node.get("parent_id")
                if parent_id in grouped:
                    grouped[parent_id].append(node)

                table_id = node.get("table_id")
                if isinstance(table_id, str) and table_id and parent_id in grouped:
                    table_owners.setdefault(table_id, set()).add(parent_id)

            # Then include all segments of the same table_id, including continuations.
            table_ids = sorted(table_owners)
            if table_ids:
                # ``table_id`` lives in canonical ``raw_json`` rather than as a
                # materialized SQLite column.  Keep the BM25 schema stable and
                # filter the small normative-table set in Python instead of
                # issuing a query against a nonexistent column.
                continuation_rows = connection.execute(
                    """
                    SELECT raw_json
                    FROM nodes
                    WHERE content_type IN ('normative_table', 'commentary_table')
                    ORDER BY clause_id
                    """,
                ).fetchall()

                for row in continuation_rows:
                    node = json.loads(row[0])
                    if node.get("table_id") not in table_owners:
                        continue
                    for owner_id in sorted(table_owners.get(node.get("table_id"), set())):
                        grouped[owner_id].append(node)

            # Deduplicate deterministically because the direct table itself can
            # also appear in the table_id continuation query.
            for parent_id, nodes in grouped.items():
                unique = {
                    node["clause_id"]: node
                    for node in nodes
                    if node.get("clause_id")
                }
                grouped[parent_id] = [
                    unique[clause_id]
                    for clause_id in sorted(unique)
                ]

            return {
                parent_id: nodes
                for parent_id, nodes in grouped.items()
                if nodes
            }
        finally:
            connection.close()

    @staticmethod
    def _to_item(item: Mapping[str, Any]) -> dict[str, Any]:
        return {"evidence_id": item["clause_id"], "clause_id": item["clause_id"], "parent_id": item.get("parent_id"), "content_type": item["content_type"], "text": item["text"], "parent_context": [], "standard_id": item.get("standard_id"), "standard_name": item.get("standard_name"), "standard_version": item.get("standard_version"), "clause_no": item.get("clause_no"), "pdf_page_start": item.get("pdf_page_start"), "pdf_page_end": item.get("pdf_page_end"), "printed_page_start": item.get("printed_page_start"), "printed_page_end": item.get("printed_page_end"), "source_file": item.get("source_file"), "source_sha256": item.get("source_sha256"), "verification_status": item["verification_status"], "missing_facts": [], "bm25_rank": item.get("bm25_rank"), "dense_rank": item.get("dense_rank"), "rrf_score": item.get("rrf_score"), "rerank_score": None, "table_id": item.get("table_id"), "formula_id": item.get("formula_id"), "trace": {"bm25_rank": item.get("bm25_rank"), "dense_rank": item.get("dense_rank"), "bm25_score": item.get("bm25_score"), "dense_score": item.get("dense_score"), "rrf_score": item.get("rrf_score"), "rrf_k": RRF_K}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Qdrant dense index paired with an immutable BM25 manifest.")
    parser.add_argument("canonical", type=Path)
    parser.add_argument("--bm25-manifest", type=Path)
    parser.add_argument("--bm25-active-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/indexes/hybrid"))
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    if bool(args.bm25_manifest) == bool(args.bm25_active_root):
        parser.error("provide exactly one of --bm25-manifest or --bm25-active-root")
    bm25_manifest = args.bm25_manifest
    if args.bm25_active_root:
        active = json.loads((args.bm25_active_root / "active.json").read_text(encoding="utf-8"))
        bm25_manifest = _portable_path(active["manifest_path"])
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SystemExit("qdrant-client is required; run this in the GreenSpec runtime container") from exc
    built = HybridIndexBuilder(canonical_path=args.canonical, bm25_manifest_path=bm25_manifest, output_root=args.output_root, qdrant_client=QdrantClient(url=args.qdrant_url), embedder=SentenceTransformerEmbedder(args.model_path, device=args.device)).build()
    print(json.dumps({"index_manifest_id": built.index_manifest_id, "manifest_path": str(built.manifest_path), "collection": built.collection_name}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
