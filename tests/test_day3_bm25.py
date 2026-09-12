import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from greenspec_rag.api import create_app
from greenspec_rag.retrieval.bm25 import BM25IndexBuilder, BM25Retriever, RetrievalError, build_retrieval_text


def clause(clause_id, *, content_type="normative_clause", indexable=True, clause_no="3.2.4", text="围护结构热工性能应满足 500m 范围要求，比例不低于 10%", **extra):
    node = {
        "schema_version": "clause.v1", "clause_id": clause_id, "parent_id": None,
        "document_id": "GB50378-2019", "standard_id": "GB_T_50378_2019", "standard_name": "绿色建筑评价标准", "standard_version": "2019",
        "source_file": "source/GB50378-2019.pdf", "source_sha256": "a" * 64, "document_status": "current",
        "content_type": content_type, "clause_type": "clause", "clause_no": clause_no, "clause_title": None, "hierarchy": [3, 2, 4], "text": text, "retrieval_text": text,
        "region": "全国", "building_type": None, "design_phase": None, "green_target_scope": None,
        "requires_project_facts": False, "requires_calculation": False, "requires_manual_review": False,
        "pdf_page_start": 12, "pdf_page_end": 12, "printed_page_start": 3, "printed_page_end": 3,
        "source_level": "T0_CANDIDATE", "verification_status": "source_unverified", "indexable": indexable,
        "table_id": None, "formula_id": None, "asset_ids": [], "supersession_ids": [], "provenance": {"adapter": "test", "adapter_version": "1"},
    }
    node.update(extra)
    return node


class Day3BM25Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.canonical = root / "clauses.jsonl"
        nodes = [
            clause("GB50378:3.2.4", text="3.2.4 围护结构热工性能应符合国家现行标准。"),
            clause("GB50378:table-3.2.4", clause_no="3.2.4", content_type="normative_table", text="表3.2.4 围护结构传热系数限值", table_id="GB50378-table-3.2.4"),
            clause("GB50378:formula-3.2.5", clause_no="3.2.5", content_type="normative_formula", text="公式3.2.5 建筑能耗计算公式", formula_id="GB50378-formula-3.2.5"),
            clause("GB50378:beijing", clause_no="5.1.1", text="北京地区绿色建筑场地应满足要求", region="北京", building_type=["住宅"], design_phase=["方案"], green_target_scope=["two_star"]),
            clause("GB50378:unknown-metadata", clause_no="5.1.2", text="绿色建筑场地要求", region=None),
            clause("GB50378:commentary", content_type="commentary_clause", indexable=False, text="围护结构说明"),
            clause("GB50378:commentary-table", clause_no="3.2.6", content_type="commentary_table", text="表3.2.6 说明表中的计算值", table_id="GB50378-table-3.2.6"),
        ]
        self.canonical.write_text("".join(json.dumps(node, ensure_ascii=False) + "\n" for node in nodes), encoding="utf-8")
        self.registry = root / "registry.json"
        self.registry.write_text(json.dumps([{"standard_id": "GB_T_50378_2019", "status": "current", "effective_date": "2019-08-01"}]), encoding="utf-8")
        self.index_root = root / "indexes"
        self.built = BM25IndexBuilder(canonical_path=self.canonical, output_root=self.index_root, registry_path=self.registry).build()
        self.retriever = BM25Retriever.from_active(self.index_root, registry_path=self.registry)

    def tearDown(self):
        self.tmp.cleanup()

    def request(self, query, **overrides):
        payload = {"query": query, "project_profile": {}, "filters": {"standard_ids": [], "clause_nos": [], "as_of_date": None, "must_be_current": False, "include_commentary": False}, "top_k": 10, "allow_degraded": True}
        payload.update(overrides)
        return payload

    def test_retrieval_text_preserves_identifiers_units_and_fixed_jieba(self):
        text = build_retrieval_text(clause("GB50378:3.2.4", text="GB/T 50378-2019 中 500m 范围比例不低于10%"))
        self.assertIn("gb/t50378-2019", text)
        self.assertIn("3.2.4", text)
        self.assertIn("500m", text)
        self.assertIn("10%", text)
        self.assertIn("公式", build_retrieval_text(clause("formula", content_type="normative_formula", formula_id="GB50378-formula-3.2.5")))

    def test_fixed_clause_table_formula_and_topic_queries_are_repeatable(self):
        cases = [("3.2.4", "GB50378:table-3.2.4"), ("表3.2.4", "GB50378:table-3.2.4"), ("公式3.2.5", "GB50378:formula-3.2.5"), ("围护结构热工性能", "GB50378:3.2.4")]
        for query, expected in cases:
            first = [item["clause_id"] for item in self.retriever.retrieve(self.request(query))["items"]]
            second = [item["clause_id"] for item in self.retriever.retrieve(self.request(query))["items"]]
            self.assertEqual(first, second)
            self.assertEqual(first[0], expected)

    def test_rebuild_does_not_modify_canonical_and_indexes_only_searchable_commentary_tables(self):
        before = hashlib.sha256(self.canonical.read_bytes()).hexdigest()
        manifest = json.loads(self.built.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(before, hashlib.sha256(self.canonical.read_bytes()).hexdigest())
        self.assertEqual(manifest["indexed_node_count"], 6)
        self.assertEqual(manifest["excluded_commentary_or_nonindexable_count"], 1)
        self.assertFalse(any("commentary" in item["clause_id"] for item in self.retriever.retrieve(self.request("围护结构"))["items"]))
        ids = [item["clause_id"] for item in self.retriever.retrieve(self.request("表3.2.6"))["items"]]
        self.assertEqual(ids[0], "GB50378:commentary-table")

    def test_hard_filters_and_unknown_metadata_safe_soft_filters(self):
        profile = {"location": {"city": "北京"}, "building": {"building_category": "住宅", "design_phase": "方案"}, "green_building_target": {"precheck": "two_star"}}
        result = self.retriever.retrieve(self.request("绿色建筑场地", project_profile=profile, filters={"standard_ids": ["GB_T_50378_2019"], "clause_nos": [], "as_of_date": "2026-08-28", "must_be_current": True, "include_commentary": False}))
        ids = [item["clause_id"] for item in result["items"]]
        self.assertIn("GB50378:beijing", ids)
        self.assertIn("GB50378:unknown-metadata", ids)
        self.assertNotIn("GB50378:3.2.4", ids)
        self.assertEqual(result["items"][0]["clause_id"], "GB50378:beijing")

    def test_zero_results_and_contract_validation(self):
        self.assertEqual(self.retriever.retrieve(self.request("不存在主题"))["items"], [])
        with self.assertRaises(RetrievalError) as bad_top_k:
            self.retriever.retrieve(self.request("围护结构", top_k=101))
        self.assertEqual(bad_top_k.exception.code, "invalid_request")
        payload = self.request("围护结构", filters={"standard_ids": [], "clause_nos": [], "as_of_date": None, "must_be_current": True, "include_commentary": False})
        with self.assertRaises(RetrievalError) as missing_date:
            self.retriever.retrieve(payload)
        self.assertEqual(missing_date.exception.code, "missing_as_of_date")

    def test_long_natural_language_query_has_deterministic_or_fallback(self):
        first = self.retriever.retrieve(self.request("围护结构与不存在的附加描述"))
        second = self.retriever.retrieve(self.request("围护结构与不存在的附加描述"))
        self.assertEqual(first["filters_applied"]["query_match_mode"], "or_fallback")
        self.assertEqual([item["clause_id"] for item in first["items"]], [item["clause_id"] for item in second["items"]])
        self.assertIn("GB50378:3.2.4", [item["clause_id"] for item in first["items"]])

    def test_http_error_envelope_request_id_and_trace(self):
        trace_path = Path(self.tmp.name) / "traces" / "retrieve.jsonl"
        client = TestClient(create_app(index_root=self.index_root, registry_path=self.registry, trace_path=trace_path))
        response = client.post("/v1/retrieve", json=self.request("围护结构"))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["request_id"])
        self.assertEqual(payload["degraded_modes"], ["bm25_only"])
        invalid = client.post("/v1/retrieve", json=self.request("围护结构", top_k=0)).json()
        self.assertEqual(invalid["error"]["code"], "invalid_request")
        self.assertTrue(invalid["error"]["request_id"])
        self.assertEqual(len(trace_path.read_text(encoding="utf-8").splitlines()), 2)
