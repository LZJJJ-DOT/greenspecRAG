import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from greenspec_rag.api import create_app
from greenspec_rag.retrieval.evidence import CompleteRetriever, EvidencePackBuilder, RerankerTimeout, evaluate_cases
from greenspec_rag.retrieval.bm25 import RetrievalError


def item(clause_id="GB55015:4.1.1", *, content_type="normative_clause", parent_id=None):
    return {"evidence_id": clause_id, "clause_id": clause_id, "parent_id": parent_id, "content_type": content_type, "text": "建筑节能条文", "parent_context": [], "standard_id": "GB_55015_2021", "standard_name": "建筑节能与可再生能源利用通用规范", "standard_version": "2021", "clause_no": "4.1.1", "pdf_page_start": 12, "pdf_page_end": 12, "printed_page_start": 3, "printed_page_end": 3, "source_file": "source.pdf", "source_sha256": "a" * 64, "verification_status": "source_unverified", "missing_facts": [], "bm25_rank": 1, "dense_rank": 1, "rrf_score": 2 / 61, "rerank_score": None, "table_id": "table-1" if content_type == "normative_table" else None, "formula_id": "formula-1" if content_type == "normative_formula" else None, "trace": {"bm25_score": -1.0, "dense_score": .9}}


class FakeHybrid:
    manifest = {"index_manifest_id": "hybrid-test"}
    def __init__(self, items, supporting=None):
        self.items = items
        self.supporting = supporting or {}
        self.calls = []
    def retrieve(self, request, **kwargs):
        self.calls.append(kwargs)
        limit = kwargs.get("result_limit", len(self.items))
        return {"items": [dict(value) for value in self.items[:limit]], "filters_applied": {"fusion_trace": {"diversity_removed": []}}, "warnings": [], "degraded_modes": []}
    def _load_node(self, clause_id): return {"clause_id": clause_id, "clause_no": "4.1", "content_type": "normative_body", "text": "父节点"}
    def supporting_nodes_for(self, parent_clause_ids):
        return {
            parent_id: [dict(node) for node in self.supporting.get(parent_id, [])]
            for parent_id in parent_clause_ids
            if parent_id in self.supporting
        }

class FakeReranker:
    model_id, device = "fake-reranker", "cpu"
    def __init__(self): self.last_texts = []
    def score(self, query, texts):
        self.last_texts = list(texts)
        return [float(index) for index, _ in enumerate(texts)]


class TimeoutReranker(FakeReranker):
    def score(self, query, texts): raise RerankerTimeout("timeout")


class Day5EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = Path(self.tmp.name) / "registry.json"
        self.registry.write_text(json.dumps([{"standard_id": "GB_55015_2021", "standard_number": "GB 55015-2021", "standard_name": "建筑节能与可再生能源利用通用规范", "standard_version": "2021"}]), encoding="utf-8")
        self.builder = EvidencePackBuilder(registry_path=self.registry)
        self.request = {"query": "建筑节能", "project_profile": {}, "filters": {"standard_ids": [], "clause_nos": [], "as_of_date": None, "must_be_current": False, "include_commentary": False}, "top_k": 10, "allow_degraded": True}

    def tearDown(self): self.tmp.cleanup()

    def test_evidence_pack_schema_groups_citations_and_parent_context(self):
        child, table, formula = item(parent_id="GB55015:4.1"), item("GB55015:table", content_type="normative_table"), item("GB55015:formula", content_type="normative_formula")
        pack = self.builder.build(request_id="r1", retrieval_run_id="run1", index_manifest_id="index1", query="建筑节能", items=[child, table, formula], filters_applied={}, warnings=[], degraded_modes=[], parent_loader=lambda _: {"clause_id": "GB55015:4.1", "clause_no": "4.1", "content_type": "normative_body", "text": "父节点"})
        self.assertEqual(pack["schema_version"], "evidence_pack.v1")
        self.assertEqual(pack["groups"]["primary_normative"], [child["evidence_id"]])
        self.assertEqual(pack["groups"]["supporting_table_formula"], [table["evidence_id"], formula["evidence_id"]])
        self.assertEqual(pack["citations"][0]["standard_number"], "GB 55015-2021")
        self.assertEqual(pack["items"][0]["parent_context"][0]["clause_id"], "GB55015:4.1")

    def test_complete_retriever_attaches_a_direct_table_absent_from_ranked_items(self):
        clause = item("GB55015:3.1.4")
        table = item(
            "GB55015:table-3.1.4",
            content_type="normative_table",
            parent_id=clause["clause_id"],
        )
        complete = CompleteRetriever(
            hybrid=FakeHybrid([clause], supporting={clause["clause_id"]: [table]}),
            reranker=FakeReranker(),
            evidence_builder=self.builder,
        )

        result, pack = complete.retrieve(self.request)

        self.assertEqual(
            [item["clause_id"] for item in result["items"]],
            [clause["clause_id"], table["clause_id"]],
        )
        self.assertEqual(pack["groups"]["supporting_table_formula"], [table["clause_id"]])
        self.assertEqual(
            result["items"][1]["trace"]["evidence_expansion"]["kind"],
            "direct_child_table_or_formula",
        )

    def test_reranker_scores_a_larger_pool_before_returning_request_top_k(self):
        hybrid = FakeHybrid([item(f"GB55015:4.1.{index}") for index in range(35)])
        reranker = FakeReranker()
        complete = CompleteRetriever(hybrid=hybrid, reranker=reranker, evidence_builder=self.builder)

        result, pack = complete.retrieve(self.request)

        self.assertEqual(hybrid.calls[0]["result_limit"], 30)
        self.assertEqual(len(reranker.last_texts), 30)
        self.assertEqual(result["filters_applied"]["rerank"], {"candidate_limit": 30, "candidate_count": 30, "returned_ranked_item_count": 10})
        self.assertEqual(len([entry for entry in result["items"] if not entry["trace"].get("evidence_expansion")]), 10)
        self.assertEqual(pack["context_budget"]["selected_item_count"], 8)

    def test_context_budget_keeps_structural_support_before_later_primary_evidence(self):
        clause = item("GB55015:3.1.4")
        clause["text"] = "主条文" * 10
        table = item("GB55015:table-3.1.4", content_type="normative_table", parent_id=clause["clause_id"])
        table["text"] = "续表" * 10
        table["trace"]["evidence_expansion"] = {"kind": "direct_child_table_or_formula", "parent_clause_id": clause["clause_id"]}
        later = item("GB55015:3.1.5")
        later["text"] = "后续条文" * 150
        budgeted = EvidencePackBuilder(registry_path=self.registry, max_context_tokens=220, primary_item_limit=2)

        pack = budgeted.build(request_id="r1", retrieval_run_id="run1", index_manifest_id="index1", query="建筑节能", items=[clause, table, later], filters_applied={}, warnings=[], degraded_modes=[])

        self.assertEqual([entry["evidence_id"] for entry in pack["items"]], [clause["evidence_id"], table["evidence_id"]])
        self.assertLessEqual(pack["context_budget"]["estimated_tokens"], 220)
        self.assertEqual(pack["context_budget"]["omitted_for_budget_evidence_ids"], [later["evidence_id"]])
        self.assertEqual(pack["warnings"][0]["code"], "evidence_pack_context_budget_exceeded")

    def test_empty_candidate_pool_skips_reranker_and_returns_auditable_pack(self):
        reranker = FakeReranker()
        complete = CompleteRetriever(hybrid=FakeHybrid([]), reranker=reranker, evidence_builder=self.builder)

        result, pack = complete.retrieve(self.request)

        self.assertEqual(reranker.last_texts, [])
        self.assertEqual(result["items"], [])
        self.assertEqual(pack["warnings"][0]["code"], "no_retrieval_candidates")

    def test_invalid_evidence_id_is_rejected(self):
        bad = item(); bad["evidence_id"] = "wrong"
        with self.assertRaises(RetrievalError) as caught:
            self.builder.build(request_id=None, retrieval_run_id="run", index_manifest_id="index", query="建筑节能", items=[bad], filters_applied={}, warnings=[], degraded_modes=[])
        self.assertEqual(caught.exception.code, "index_build_failed")

    def test_reranker_timeout_fails_closed_or_returns_rrf_when_allowed(self):
        complete = CompleteRetriever(hybrid=FakeHybrid([item()]), reranker=TimeoutReranker(), evidence_builder=self.builder)
        result, pack = complete.retrieve(self.request)
        self.assertEqual(result["degraded_modes"], ["reranker_timeout"])
        self.assertEqual(pack["degraded_modes"], ["reranker_timeout"])
        strict = dict(self.request, allow_degraded=False)
        with self.assertRaises(RetrievalError) as caught:
            complete.retrieve(strict)
        self.assertEqual(caught.exception.code, "reranker_timeout")

    def test_evaluation_without_gold_is_experimental_and_labeled_metrics_can_pass(self):
        unlabeled = [{"eval_id": f"E{index:03}", "gold_evidence_ids": [], "human_annotation": {}} for index in range(1, 11)]
        report = evaluate_cases(unlabeled, lambda case: {"retrieval_run_id": case["eval_id"], "items": [item()], "needs_manual_review": True})
        self.assertEqual(report["publication_status"], "experimental")
        self.assertIsNone(report["metrics"]["clause_hit_at_10"])
        labeled = [{"eval_id": "E001", "gold_evidence_ids": ["GB55015:4.1.1"], "human_annotation": {"citation_accuracy": True, "citation_completeness": True, "needs_manual_review": True}}]
        baseline = evaluate_cases(labeled, lambda case: {"retrieval_run_id": case["eval_id"], "items": [item()], "needs_manual_review": True})
        self.assertEqual(baseline["publication_status"], "baseline")

    def test_health_and_dry_run_build_status_follow_contract(self):
        root = Path(self.tmp.name) / "canonical"
        (root / "active").mkdir(parents=True)
        (root / "active" / "clauses.jsonl").write_text(json.dumps({"clause_id": "x"}) + "\n", encoding="utf-8")
        (root / "source_manifest.test.json").write_text(json.dumps({"source_manifest_id": "source-test"}), encoding="utf-8")
        client = TestClient(create_app(index_root=Path(self.tmp.name) / "missing-bm25", hybrid_root=Path(self.tmp.name) / "missing-hybrid", registry_path=self.registry, source_manifest_root=root, trace_path=Path(self.tmp.name) / "trace.jsonl"))
        health = client.get("/health")
        self.assertEqual(health.status_code, 503)
        self.assertEqual(set(health.json()["dependencies"]), {"sqlite", "qdrant", "embedding", "reranker"})
        accepted = client.post("/v1/index/build", json={"source_manifest_id": "source-test", "rebuild": True, "dry_run": True})
        self.assertEqual(accepted.status_code, 202)
        status = client.get(f"/v1/index/build/{accepted.json()['build_id']}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "succeeded")
