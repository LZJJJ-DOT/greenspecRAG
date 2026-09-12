import json
import tempfile
import unittest
from pathlib import Path

from greenspec_rag.ingestion.canonical_builder import build


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    ROOT / "data" / "processed" / "清洗后GB55015-2021_建筑节能与可再生能源利用通用规范.md",
    ROOT / "data" / "processed" / "清洗后GBT50378-2019_绿色建筑评价标准 .md",
]


class SourceVerificationTests(unittest.TestCase):
    def test_verified_ledger_promotes_every_node_without_warnings(self):
        with tempfile.TemporaryDirectory() as name:
            report = build(INPUTS, ROOT, Path(name), require_verified_sources=True)
            self.assertTrue(report["publishable"], report["hard_failures"])
            self.assertEqual(report["warnings"], [])
            nodes = [json.loads(line) for line in (Path(name) / "active" / "clauses.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(nodes), 915)
        self.assertTrue(all(node["verification_status"] == "verified" for node in nodes))
        self.assertTrue(all(node["source_level"] == "T0" for node in nodes))

    def test_changed_markdown_hash_blocks_a_verified_build(self):
        ledger = json.loads((ROOT / "data" / "governance" / "source_verifications.json").read_text(encoding="utf-8"))
        ledger["sources"][0]["processed_markdown"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            ledger_path = root / "ledger.json"
            ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
            report = build(INPUTS, ROOT, root / "output", source_verifications_path=ledger_path, require_verified_sources=True)
        self.assertFalse(report["publishable"])
        self.assertIn("verified_markdown_changed", {item["code"] for item in report["hard_failures"]})
