import json
import unittest
from pathlib import Path

from deploy.run_evaluation import _load_cases


ROOT = Path(__file__).resolve().parents[1]


class EvaluationQuestionDatasetTests(unittest.TestCase):
    def test_positive_cases_have_schema_valid_requests_and_canonical_gold_ids(self):
        cases = _load_cases(ROOT / "data" / "evaluation" / "questions.jsonl")
        positive_cases = [case for case in cases if case["gold_evidence_ids"]]
        self.assertGreaterEqual(len(positive_cases), 20)

        canonical_ids = {
            json.loads(line)["clause_id"]
            for line in (ROOT / "data" / "canonical" / "generated" / "active" / "clauses.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for case in positive_cases:
            with self.subTest(eval_id=case["eval_id"]):
                self.assertTrue(case["request"]["query"])
                self.assertTrue(case["request"]["filters"]["as_of_date"])
                self.assertGreaterEqual(case["request"]["top_k"], 1)
                self.assertTrue(set(case["gold_evidence_ids"]).issubset(canonical_ids))
