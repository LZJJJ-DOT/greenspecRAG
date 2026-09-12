from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from greenspec_rag.compat.clauses_adapter import CompatibilityError, adapt_jsonl, canonicalize_records


class ClauseAdapterTests(unittest.TestCase):
    def test_legacy_protocols_are_preserved_and_page_is_mapped(self) -> None:
        record = {
            "id": "legacy-5.1.1",
            "channel": "local_clause",
            "clause_no": "5.1.1",
            "clause_text": "空气质量应符合规定。",
            "clause_summary": "室内空气质量",
            "source_page": "42-43",
            "applicability_results": {"applicable": None},
            "risk_register": [{"risk_id": "R-1"}],
            "evidence_verification": {"status": "pending"},
        }
        node = canonicalize_records([record], defaults={"document_id": "doc-1", "standard_id": "GB_T_50378_2019"})[0]
        self.assertEqual(node["clause_id"], "legacy-5.1.1")
        self.assertEqual(node["content_type"], "normative_clause")
        self.assertEqual((node["pdf_page_start"], node["pdf_page_end"]), (42, 43))
        self.assertEqual(node["applicability_results"], record["applicability_results"])
        self.assertEqual(node["risk_register"], record["risk_register"])
        self.assertEqual(node["evidence_verification"], record["evidence_verification"])
        self.assertEqual(node["legacy_compat"]["applicability_results"], record["applicability_results"])
        self.assertEqual(node["legacy_compat"]["risk_register"], record["risk_register"])
        self.assertEqual(node["legacy_compat"]["evidence_verification"], record["evidence_verification"])
        self.assertTrue(node["requires_manual_review"])
        self.assertEqual(node["verification_status"], "needs_manual_review")

    def test_missing_parent_is_manual_review_and_sort_is_deterministic(self) -> None:
        records = [
            {"clause_id": "z", "parent_id": "does-not-exist", "clause_text": "z"},
            {"clause_id": "a", "clause_text": "a"},
        ]
        result = canonicalize_records(records)
        self.assertEqual([item["clause_id"] for item in result], ["a", "z"])
        self.assertTrue(result[1]["requires_manual_review"])

    def test_empty_source_is_manual_review_and_unknown_fields_are_retained(self) -> None:
        node = canonicalize_records([{"clause_id": "unknown", "clause_text": "text", "new_legacy_field": {"x": 1}}])[0]
        self.assertIsNone(node["source_file"])
        self.assertIsNone(node["source_sha256"])
        self.assertTrue(node["requires_manual_review"])
        self.assertEqual(node["legacy_extra"]["new_legacy_field"], {"x": 1})

    def test_duplicate_ids_fail_closed(self) -> None:
        with self.assertRaises(CompatibilityError):
            canonicalize_records([{"clause_id": "same", "clause_text": "a"}, {"id": "same", "clause_text": "b"}])

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clauses.jsonl"
            destination = Path(directory) / "canonical.jsonl"
            source.write_text(json.dumps({"clause_id": "x", "clause_text": "text"}) + "\n", encoding="utf-8")
            result = adapt_jsonl(source, destination)
            self.assertEqual(len(result), 1)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["schema_version"], "clause.v1")


if __name__ == "__main__":
    unittest.main()
