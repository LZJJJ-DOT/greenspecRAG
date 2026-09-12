import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from greenspec_rag.api import create_app
from greenspec_rag.retrieval.bm25 import BM25IndexBuilder
from greenspec_rag.retrieval.hybrid import HybridIndexBuilder, HybridRetriever, apply_diversity, rrf_merge


def node(clause_id, text, *, parent_id=None, content_type="normative_clause", clause_no="4.1.1"):
    return {"schema_version": "clause.v1", "clause_id": clause_id, "parent_id": parent_id, "document_id": "GB55015-2021", "standard_id": "GB_55015_2021", "standard_name": "建筑节能与可再生能源利用通用规范", "standard_version": "2021", "source_file": "source.pdf", "source_sha256": "a" * 64, "document_status": "current", "content_type": content_type, "clause_type": "clause", "clause_no": clause_no, "clause_title": None, "hierarchy": [4, 1, 1], "text": text, "retrieval_text": text, "region": "全国", "building_type": None, "design_phase": None, "green_target_scope": None, "requires_project_facts": False, "requires_calculation": False, "requires_manual_review": False, "pdf_page_start": 12, "pdf_page_end": 12, "printed_page_start": 3, "printed_page_end": 3, "source_level": "T0_CANDIDATE", "verification_status": "source_unverified", "indexable": True, "table_id": "table" if content_type == "normative_table" else None, "formula_id": "formula" if content_type == "normative_formula" else None, "asset_ids": [], "supersession_ids": [], "provenance": {"adapter": "test", "adapter_version": "1"}}


class FakeModels:
    class Distance:
        COSINE = "cosine"
    class VectorParams:
        def __init__(self, *, size, distance): self.size, self.distance = size, distance
    class PointStruct:
        def __init__(self, *, id, vector, payload): self.id, self.vector, self.payload = id, vector, payload


class FakeQdrant:
    def __init__(self): self.collections = {}
    def create_collection(self, *, collection_name, vectors_config): self.collections[collection_name] = {"size": vectors_config.size, "points": {}}
    def delete_collection(self, *, collection_name): self.collections.pop(collection_name, None)
    def upsert(self, *, collection_name, points, wait): self.collections[collection_name]["points"].update({point.id: point for point in points})
    def get_collection(self, *, collection_name): return SimpleNamespace(points_count=len(self.collections[collection_name]["points"]))
    def query_points(self, *, collection_name, query, limit):
        points = self.collections[collection_name]["points"].values()
        found = [SimpleNamespace(payload=point.payload, score=sum(a * b for a, b in zip(point.vector, query))) for point in points]
        return SimpleNamespace(points=sorted(found, key=lambda point: (-point.score, point.payload["clause_id"]))[:limit])


class FakeEmbedder:
    model_id, device, dimension = "fake-bge", "cpu", 3
    def vector(self, text):
        raw = [1.0 if "节能" in text else 0.1, 1.0 if "可再生" in text else 0.1, 1.0 if "围护" in text else 0.1]
        norm = math.sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw]
    def embed_documents(self, texts): return [self.vector(text) for text in texts]
    def embed_query(self, query): return self.vector(query)


class FakeReranker:
    model_id, device = "fake-reranker", "cpu"
    def score(self, query, texts): return [float(len(text)) for text in texts]


class Day4HybridTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.canonical = root / "clauses.jsonl"
        self.canonical.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in [node("GB55015:4.1.1", "建筑节能设计应降低能耗"), node("GB55015:4.1.2", "可再生能源利用应满足要求"), node("GB55015:4.1.3", "围护结构节能性能应符合规定")]) + "\n", encoding="utf-8")
        self.registry = root / "registry.json"
        self.registry.write_text(json.dumps([{"standard_id": "GB_55015_2021", "standard_number": "GB 55015-2021", "standard_name": "建筑节能与可再生能源利用通用规范", "standard_version": "2021", "status": "current", "effective_date": "2022-04-01", "relation_verification_status": "pending_manual_review"}]), encoding="utf-8")
        self.bm25_root = root / "bm25"
        self.bm25 = BM25IndexBuilder(canonical_path=self.canonical, output_root=self.bm25_root, registry_path=self.registry).build()
        self.qdrant, self.embedder = FakeQdrant(), FakeEmbedder()
        self.hybrid_root = root / "hybrid"
        self.built = HybridIndexBuilder(canonical_path=self.canonical, bm25_manifest_path=self.bm25.manifest_path, output_root=self.hybrid_root, qdrant_client=self.qdrant, embedder=self.embedder, qdrant_models=FakeModels).build()
        self.retriever = HybridRetriever(hybrid_manifest_path=self.built.manifest_path, qdrant_client=self.qdrant, embedder=self.embedder, registry_path=self.registry)

    def tearDown(self): self.tmp.cleanup()
    def request(self, query): return {"query": query, "project_profile": {}, "filters": {"standard_ids": [], "clause_nos": [], "as_of_date": None, "must_be_current": False, "include_commentary": False}, "top_k": 3, "allow_degraded": True}

    def test_dense_payload_manifest_and_three_modes_are_reproducible(self):
        collection = self.qdrant.collections[self.built.collection_name]
        payload = next(iter(collection["points"].values())).payload
        self.assertTrue({"clause_id", "parent_id", "standard_id", "content_type", "document_status", "indexable", "pdf_page_start", "pdf_page_end", "verification_status"}.issubset(payload))
        self.assertEqual(json.loads(self.built.manifest_path.read_text(encoding="utf-8"))["qdrant_vector_size"], 3)
        for mode in ("bm25", "dense", "hybrid"):
            first = self.retriever.retrieve(self.request("建筑节能设计"), mode=mode)
            second = self.retriever.retrieve(self.request("建筑节能设计"), mode=mode)
            self.assertEqual([item["clause_id"] for item in first["items"]], [item["clause_id"] for item in second["items"]])
        hybrid = self.retriever.retrieve(self.request("建筑节能设计"), mode="hybrid")
        self.assertEqual(hybrid["items"][0]["clause_id"], "GB55015:4.1.1")
        self.assertEqual(hybrid["items"][0]["rrf_score"], 2 / 61)

    def test_rrf_dedup_uses_one_based_rank_and_preserves_raw_scores(self):
        merged = rrf_merge([{"clause_id": "a", "bm25_score": -1.0}, {"clause_id": "a", "bm25_score": -2.0}, {"clause_id": "b", "bm25_score": -3.0}], [{"clause_id": "b", "dense_score": 0.9}, {"clause_id": "c", "dense_score": 0.8}])
        self.assertEqual([item["clause_id"] for item in merged], ["b", "a", "c"])
        self.assertEqual(merged[0]["bm25_rank"], 2)
        self.assertEqual(merged[0]["dense_rank"], 1)
        self.assertAlmostEqual(merged[0]["rrf_score"], 1 / 62 + 1 / 61)

    def test_diversity_caps_only_ordinary_siblings_not_table_formula(self):
        commentary_table = node("commentary-table", "说明表独立内容", parent_id="parent", content_type="commentary_table")
        commentary_table["table_id"] = "commentary-table"
        items = [node(f"ordinary:{index}", f"不同条文内容 {index}", parent_id="parent") for index in range(4)] + [node("table", "表格独立内容", parent_id="parent", content_type="normative_table"), commentary_table, node("formula", "公式独立内容", parent_id="parent", content_type="normative_formula")]
        accepted, removed = apply_diversity(items, limit=10)
        self.assertEqual(len(accepted), 6)
        self.assertIn("table", [item["clause_id"] for item in accepted])
        self.assertIn("commentary-table", [item["clause_id"] for item in accepted])
        self.assertIn("formula", [item["clause_id"] for item in accepted])
        self.assertEqual(removed, [{"clause_id": "ordinary:3", "reason": "parent_cap"}])

    def test_supporting_nodes_loads_direct_table_and_its_continuation(self):
        root = Path(self.tmp.name)
        canonical = root / "supporting.jsonl"
        clause = node("GB55015:3.1.4", "窗墙面积比应符合表3.1.4的规定", clause_no="3.1.4")
        table = node("GB55015:table-3.1.4:p14", "表3.1.4 主表", parent_id=clause["clause_id"], content_type="normative_table", clause_no="表3.1.4")
        continuation = node("GB55015:table-3.1.4:p15", "续表3.1.4", parent_id=clause["clause_id"], content_type="normative_table", clause_no="表3.1.4")
        table["table_id"] = continuation["table_id"] = "GB55015-table-3.1.4"
        canonical.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in [clause, table, continuation]) + "\n", encoding="utf-8")

        bm25 = BM25IndexBuilder(canonical_path=canonical, output_root=root / "supporting-bm25", registry_path=self.registry).build()
        qdrant = FakeQdrant()
        hybrid = HybridIndexBuilder(canonical_path=canonical, bm25_manifest_path=bm25.manifest_path, output_root=root / "supporting-hybrid", qdrant_client=qdrant, embedder=self.embedder, qdrant_models=FakeModels).build()
        retriever = HybridRetriever(hybrid_manifest_path=hybrid.manifest_path, qdrant_client=qdrant, embedder=self.embedder, registry_path=self.registry)

        supporting = retriever.supporting_nodes_for([clause["clause_id"]])
        self.assertEqual(
            {item["clause_id"] for item in supporting[clause["clause_id"]]},
            {table["clause_id"], continuation["clause_id"]},
        )

    def test_http_endpoint_uses_active_hybrid_and_records_mode_trace(self):
        trace_path = Path(self.tmp.name) / "trace.jsonl"
        client = TestClient(create_app(index_root=self.bm25_root, hybrid_root=self.hybrid_root, registry_path=self.registry, trace_path=trace_path, evidence_root=Path(self.tmp.name) / "packs", qdrant_client_factory=lambda: self.qdrant, embedder_factory=lambda: self.embedder, reranker_factory=FakeReranker))
        response = client.post("/v1/retrieve", json=self.request("建筑节能设计"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["index_manifest_id"], self.built.index_manifest_id)
        self.assertIsNotNone(response.json()["items"][0]["dense_rank"])
        self.assertEqual(json.loads(trace_path.read_text(encoding="utf-8"))["mode"], "hybrid")
