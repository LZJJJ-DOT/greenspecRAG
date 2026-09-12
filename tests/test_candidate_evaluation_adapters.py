import json
import tempfile
import unittest
from pathlib import Path

from deploy.build_candidate_evidence_pack_review import build_review_run
from deploy.prepare_candidate_ragas_datasets import prepare_datasets
from deploy.summarize_candidate_evidence_pack_review import summarize


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _item(evidence_id: str, *, score: float | None = 1.0) -> dict:
    return {
        "evidence_id": evidence_id,
        "clause_id": evidence_id,
        "clause_no": "3.1.1",
        "content_type": "normative_clause",
        "text": "规范条文原文",
        "parent_context": [],
        "pdf_page_start": 12,
        "pdf_page_end": 12,
        "table_id": None,
        "formula_id": None,
        "rerank_score": score,
    }


class CandidateEvaluationAdapterTests(unittest.TestCase):
    def test_candidate_review_and_split_ragas_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            accepted = tmp_path / "accepted.jsonl"
            retrievals = tmp_path / "retrievals.jsonl"
            canonical = tmp_path / "canonical.jsonl"
            _write_jsonl(accepted, [{"candidate_id": "C001", "question": "条文要求是什么？", "category": "single_clause", "difficulty_label": "direct", "gold_evidence_ids": ["A"], "required_evidence_ids": ["A"], "forbidden_near_misses": []}])
            _write_jsonl(canonical, [{"clause_id": "A", "text": "规范条文原文"}])
            ranked, attached = _item("A"), _item("T", score=None)
            _write_jsonl(retrievals, [{"candidate_id": "C001", "retrieval": {"index_manifest_id": "hybrid-test", "error": None, "top_k_items": [ranked, attached]}, "evidence_pack": {"evidence_pack_id": "pack-1", "items": [ranked], "context_budget": {"max_estimated_tokens": 5000}}}])

            review = build_review_run(accepted_path=accepted, retrievals_path=retrievals, output_root=tmp_path / "reviews")
            annotations_path = review / "review_annotations.jsonl"
            annotation = json.loads(annotations_path.read_text(encoding="utf-8"))
            annotation["annotation"].update({"citation_accuracy": True, "citation_completeness": True, "evidence_pack_supports_question": True, "reviewer": "liang"})
            _write_jsonl(annotations_path, [annotation])
            report = summarize(review)
            self.assertEqual(report["publication_status"], "complete_human_review")
            self.assertEqual(report["metrics"]["citation_accuracy"]["coverage"], "1/1")

            datasets = prepare_datasets(accepted_path=accepted, retrievals_path=retrievals, canonical_path=canonical, output_root=tmp_path / "datasets")
            retrieval_sample = json.loads((datasets / "retrieval" / "samples.jsonl").read_text(encoding="utf-8"))
            answer_sample = json.loads((datasets / "answer" / "samples.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(retrieval_sample["retrieved_context_ids"], ["A"])
            self.assertEqual(answer_sample["evidence_pack"]["source"], "budgeted_evidence_pack")
