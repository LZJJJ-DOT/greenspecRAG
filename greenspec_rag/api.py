"""Day 3 BM25-only HTTP API. Dense and reranker modes arrive in later days."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from .retrieval.bm25 import BM25Retriever, RetrievalError


logger = logging.getLogger(__name__)


def _error(error: RetrievalError, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": {"code": error.code, "message": str(error), "request_id": request_id, "retryable": error.retryable, "details": error.details}})


def create_app(*, index_root: str | Path = "data/indexes/bm25", hybrid_root: str | Path = "data/indexes/hybrid", registry_path: str | Path = "data/registry/standard_registry.json", source_manifest_root: str | Path = "data/canonical/generated", trace_path: str | Path = "data/traces/retrieve.jsonl", evidence_root: str | Path = "data/evidence_packs", qdrant_url: str | None = None, embedding_model_path: str | Path | None = None, reranker_model_path: str | Path | None = None, embedding_device: str | None = None, qdrant_client_factory: Callable[[], Any] | None = None, embedder_factory: Callable[[], Any] | None = None, reranker_factory: Callable[[], Any] | None = None) -> FastAPI:
    app = FastAPI(title="GreenSpec RAG sidecar API", version="0.1.0")
    index_root, hybrid_root, registry_path, source_manifest_root, trace_path, evidence_root = Path(index_root), Path(hybrid_root), Path(registry_path), Path(source_manifest_root), Path(trace_path), Path(evidence_root)
    qdrant_url = qdrant_url or os.getenv("RAG_QDRANT_URL", "http://127.0.0.1:6333")
    embedding_model_path = Path(embedding_model_path or os.getenv("RAG_BGE_MODEL_PATH", "/models/bge-base-zh-v1.5"))
    reranker_model_path = Path(reranker_model_path or os.getenv("RAG_RERANKER_MODEL_PATH", "/models/bge-reranker-v2-m3"))
    builds: dict[str, dict[str, Any]] = {}
    runtime_lock = threading.Lock()
    runtime_cache: dict[str, tuple[Any, Any]] = {}

    def complete_retriever_for(active: Mapping[str, Any]) -> tuple[Any, Any]:
        """Keep local models resident; invalidate automatically on manifest swap."""
        manifest_path = str(active["manifest_path"])
        with runtime_lock:
            cached = runtime_cache.get(manifest_path)
            if cached:
                return cached
            from .retrieval.evidence import BGEReranker, CompleteRetriever, EvidencePackBuilder
            from .retrieval.hybrid import HybridRetriever, SentenceTransformerEmbedder

            if qdrant_client_factory:
                qdrant = qdrant_client_factory()
            else:
                from qdrant_client import QdrantClient
                qdrant = QdrantClient(url=qdrant_url)
            embedder = embedder_factory() if embedder_factory else SentenceTransformerEmbedder(embedding_model_path, device=embedding_device)
            retriever = HybridRetriever(hybrid_manifest_path=Path(manifest_path), qdrant_client=qdrant, embedder=embedder, registry_path=registry_path)
            reranker = reranker_factory() if reranker_factory else BGEReranker(reranker_model_path, device=embedding_device or "cuda")
            complete = CompleteRetriever(hybrid=retriever, reranker=reranker, evidence_builder=EvidencePackBuilder(registry_path=registry_path, relations_path=registry_path.parent / "clause_relations.jsonl"))
            runtime_cache.clear()
            runtime_cache[manifest_path] = (retriever, complete)
            return retriever, complete

    def error_payload(code: str, message: str, request_id: str, *, status_code: int = 400, retryable: bool = False, details: Mapping[str, Any] | None = None) -> JSONResponse:
        return _error(RetrievalError(code, message, status_code=status_code, retryable=retryable, details=details), request_id)

    def find_source_manifest(source_manifest_id: str) -> Path | None:
        for candidate in source_manifest_root.glob("source_manifest.*.json"):
            try:
                if json.loads(candidate.read_text(encoding="utf-8")).get("source_manifest_id") == source_manifest_id:
                    return candidate
            except json.JSONDecodeError:
                continue
        return None

    def run_build(build_id: str, source_manifest_id: str, dry_run: bool) -> None:
        record = builds[build_id]
        record["status"] = "running"
        try:
            source_manifest = find_source_manifest(source_manifest_id)
            if not source_manifest:
                raise RetrievalError("index_build_failed", "source manifest does not exist", status_code=422)
            canonical = source_manifest_root / "active" / "clauses.jsonl"
            if not canonical.is_file():
                raise RetrievalError("index_build_failed", "active canonical JSONL does not exist", status_code=422)
            if dry_run:
                record.update({"status": "succeeded", "extraction_run_id": source_manifest_id, "warnings": [{"code": "dry_run", "message": "Source manifest and active canonical JSONL passed preflight; no index was changed.", "severity": "info", "evidence_ids": []}]})
                return
            from .retrieval.bm25 import BM25IndexBuilder
            from .retrieval.hybrid import HybridIndexBuilder, SentenceTransformerEmbedder
            from qdrant_client import QdrantClient
            bm25 = BM25IndexBuilder(canonical_path=canonical, output_root=index_root, source_manifest_path=source_manifest, registry_path=registry_path).build()
            qdrant = qdrant_client_factory() if qdrant_client_factory else QdrantClient(url=qdrant_url)
            embedder = embedder_factory() if embedder_factory else SentenceTransformerEmbedder(embedding_model_path, device=embedding_device)
            hybrid = HybridIndexBuilder(canonical_path=canonical, bm25_manifest_path=bm25.manifest_path, output_root=hybrid_root, qdrant_client=qdrant, embedder=embedder).build()
            record.update({"status": "succeeded", "extraction_run_id": source_manifest_id, "index_manifest_id": hybrid.index_manifest_id})
        except Exception as exc:
            code = exc.code if isinstance(exc, RetrievalError) else "index_build_failed"
            record.update({"status": "failed", "failure": {"code": code, "message": str(exc), "retryable": isinstance(exc, RetrievalError) and exc.retryable}})

    @app.post("/v1/index/build")
    async def start_index_build(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
        request_id = str(uuid.uuid4())
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return error_payload("invalid_request", "request body must be valid JSON", request_id)
        required = {"source_manifest_id", "rebuild", "dry_run"}
        if not isinstance(body, Mapping) or set(body) != required or not isinstance(body.get("source_manifest_id"), str) or not isinstance(body.get("rebuild"), bool) or not isinstance(body.get("dry_run"), bool):
            return error_payload("invalid_request", "build request does not match the frozen contract", request_id)
        if not find_source_manifest(body["source_manifest_id"]):
            return error_payload("index_build_failed", "source manifest does not exist", request_id, status_code=422)
        if any(record["status"] in {"queued", "running"} for record in builds.values()):
            return error_payload("build_conflict", "another index build is active", request_id, status_code=409, retryable=True)
        build_id = f"build_{uuid.uuid4().hex}"
        builds[build_id] = {"request_id": request_id, "build_id": build_id, "status": "queued", "source_manifest_id": body["source_manifest_id"], "extraction_run_id": None, "index_manifest_id": None, "failure": None, "warnings": []}
        background_tasks.add_task(run_build, build_id, body["source_manifest_id"], body["dry_run"])
        return JSONResponse(status_code=202, content={"request_id": request_id, "build_id": build_id, "status": "accepted", "accepted_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})

    @app.get("/v1/index/build/{build_id}")
    async def get_index_build(build_id: str) -> JSONResponse:
        request_id = str(uuid.uuid4())
        record = builds.get(build_id)
        if not record:
            return error_payload("build_not_found", "build ID does not exist", request_id, status_code=404)
        return JSONResponse(status_code=200, content={**record, "request_id": request_id})

    @app.get("/health")
    async def health() -> JSONResponse:
        dependencies: dict[str, dict[str, Any]] = {"sqlite": {"status": "ready" if (index_root / "active.json").is_file() else "unavailable", "message": "active BM25 manifest found" if (index_root / "active.json").is_file() else "no active BM25 manifest"}, "qdrant": {"status": "unavailable", "message": "not checked"}, "embedding": {"status": "ready" if embedding_model_path.is_dir() else "unavailable", "message": "local BGE path found" if embedding_model_path.is_dir() else "local BGE path missing"}, "reranker": {"status": "ready" if reranker_model_path.is_dir() else "unavailable", "message": "local reranker path found" if reranker_model_path.is_dir() else "local reranker path missing"}}
        try:
            from qdrant_client import QdrantClient
            (qdrant_client_factory() if qdrant_client_factory else QdrantClient(url=qdrant_url)).get_collections()
            dependencies["qdrant"] = {"status": "ready", "message": "Qdrant is reachable"}
        except Exception:
            dependencies["qdrant"] = {"status": "unavailable", "message": "Qdrant is unreachable"}
        ready = all(value["status"] == "ready" for value in dependencies.values()) and (hybrid_root / "active.json").is_file()
        active_id = None
        if (hybrid_root / "active.json").is_file():
            active_id = json.loads((hybrid_root / "active.json").read_text(encoding="utf-8")).get("index_manifest_id")
        body = {"service": "greenspec-rag", "version": "0.1.0", "live": True, "ready": ready, "active_index_manifest_id": active_id, "dependencies": dependencies, "warnings": [] if ready else [{"code": "dependency_unavailable", "message": "one or more full-hybrid dependencies are not ready", "severity": "warning", "evidence_ids": []}]}
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.post("/v1/retrieve")
    async def retrieve(request: Request) -> JSONResponse:
        request_id, started = str(uuid.uuid4()), time.perf_counter()
        retrieval_run_id = f"retrieve_{uuid.uuid4().hex}"
        trace = {"mode": "unknown", "index_manifest_id": None, "candidate_count": None, "diversity_removed": 0}
        try:
            body = await request.json()
            if not isinstance(body, Mapping):
                raise RetrievalError("invalid_request", "request body must be a JSON object")
            active_hybrid = hybrid_root / "active.json"
            if active_hybrid.is_file():
                try:
                    active = json.loads(active_hybrid.read_text(encoding="utf-8"))
                    retriever, complete = complete_retriever_for(active)
                    result, pack = complete.retrieve(body, request_id=request_id, retrieval_run_id=retrieval_run_id)
                    evidence_root.mkdir(parents=True, exist_ok=True)
                    pack_temp = evidence_root / f".{pack['retrieval_run_id']}.json"
                    pack_temp.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(pack_temp, evidence_root / f"{pack['retrieval_run_id']}.json")
                    trace.update({"mode": "hybrid", "index_manifest_id": retriever.manifest["index_manifest_id"], "candidate_count": len(result["items"]), "diversity_removed": len(result["filters_applied"]["fusion_trace"]["diversity_removed"])})
                    index_manifest_id = retriever.manifest["index_manifest_id"]
                except RetrievalError as exc:
                    if exc.code in {"invalid_request", "invalid_query", "missing_as_of_date", "degraded_not_allowed", "index_build_failed"} or not body.get("allow_degraded"):
                        raise
                    retriever = BM25Retriever.from_active(index_root, registry_path=registry_path)
                    result = retriever.retrieve(body)
                    result["warnings"].append({"code": "dense_unavailable", "message": "Qdrant or embedding model is unavailable; returned BM25-only fallback.", "severity": "warning", "evidence_ids": []})
                    trace.update({"mode": "bm25_fallback", "index_manifest_id": retriever.manifest["index_manifest_id"], "candidate_count": len(result["items"])})
                    index_manifest_id = retriever.manifest["index_manifest_id"]
                except Exception as exc:
                    logger.exception("unexpected full-hybrid retrieval failure")
                    if not body.get("allow_degraded"):
                        raise RetrievalError("dependency_unavailable", "hybrid dependencies are unavailable", status_code=503, retryable=True) from exc
                    retriever = BM25Retriever.from_active(index_root, registry_path=registry_path)
                    result = retriever.retrieve(body)
                    result["warnings"].append({"code": "dense_unavailable", "message": "Qdrant or embedding model is unavailable; returned BM25-only fallback.", "severity": "warning", "evidence_ids": []})
                    trace.update({"mode": "bm25_fallback", "index_manifest_id": retriever.manifest["index_manifest_id"], "candidate_count": len(result["items"])})
                    index_manifest_id = retriever.manifest["index_manifest_id"]
            else:
                retriever = BM25Retriever.from_active(index_root, registry_path=registry_path)
                result = retriever.retrieve(body)
                trace.update({"mode": "bm25", "index_manifest_id": retriever.manifest["index_manifest_id"], "candidate_count": len(result["items"])})
                index_manifest_id = retriever.manifest["index_manifest_id"]
            result.update({"request_id": request_id, "retrieval_run_id": retrieval_run_id, "index_manifest_id": index_manifest_id})
            # Keep the ranked-list response backward compatible while making
            # the already schema-validated artifact available to an external
            # report agent.  Fallback modes deliberately omit it because they
            # did not construct a complete EvidencePack.
            if "pack" in locals():
                result["evidence_pack"] = pack
            return JSONResponse(status_code=200, content=result)
        except RetrievalError as exc:
            return _error(exc, request_id)
        except json.JSONDecodeError:
            return _error(RetrievalError("invalid_request", "request body must be valid JSON"), request_id)
        except FileNotFoundError:
            return _error(RetrievalError("dependency_unavailable", "no active BM25 index is available", status_code=503, retryable=True), request_id)
        finally:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as handle:
                trace.update({"request_id": request_id, "route": "/v1/retrieve", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
                handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

    return app


app = create_app()
