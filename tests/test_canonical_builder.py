from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from greenspec_rag.ingestion.canonical_builder import build, parse_markdown


ROOT = Path(__file__).resolve().parents[1]


class CanonicalBuilderTests(unittest.TestCase):
    def _write(self, directory: Path, body: str) -> Path:
        path = directory / "source.md"
        path.write_text(body, encoding="utf-8")
        return path

    def test_node_metadata_overrides_page_region(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            directory = Path(name)
            source = self._write(directory, """---
standard_number: GB/T 50378-2019
---
## PDF 阅读器第 1 页（原文印刷页：1）
<!-- content_type: normative_body
pdf_page: 1
printed_page: 1
-->
**3.2.5** 总得分应按下式计算。
<!-- formula_metadata
formula_id: formula-3.2.5
equation_no: 3.2.5
pdf_page: 1
printed_page: 1
content_type: normative_formula
-->
$$Q=1$$
""")
            nodes, _ = parse_markdown(source, ROOT)
            formula = next(item for item in nodes if item["formula_id"] == "formula-3.2.5")
            self.assertEqual(formula["content_type"], "normative_formula")
            self.assertEqual(formula["pdf_page_start"], 1)

    def test_searchable_commentary_table_continuation_stays_on_its_own_page(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            directory = Path(name)
            source = self._write(directory, """---
standard_number: GB/T 50378-2019
---
## PDF 阅读器第 1 页（原文印刷页：1）
<!-- content_type: commentary
pdf_page: 1
printed_page: 1
-->
<!-- table_metadata
table_id: GB50378-2019-table-3
table_no: 3
content_type: commentary_table_continuation
indexable: true
pdf_page_start: 1
pdf_page_end: 1
-->
#### 续表 3 示例
| 项目 | 分值 |
| --- | --- |
| A | 10 |
## PDF 阅读器第 2 页（原文印刷页：2）
<!-- content_type: commentary
pdf_page: 2
printed_page: 2
-->
第 2 页无关条文说明。
<!-- formula_metadata
formula_id: formula-9
equation_no: 9.1
content_type: commentary_formula
-->
$$Q=1$$
""")
            nodes, _ = parse_markdown(source, ROOT)
            table = next(item for item in nodes if item["table_id"] == "GB50378-2019-table-3")
            self.assertEqual(table["content_type"], "commentary_table")
            self.assertEqual(table["clause_type"], "table")
            self.assertTrue(table["indexable"])
            self.assertEqual((table["pdf_page_start"], table["pdf_page_end"]), (1, 1))
            self.assertNotIn("第 2 页无关", table["text"])
            report = build([source], ROOT, directory / "output", source_verifications_path=directory / "missing.json")
            self.assertTrue(report["publishable"], report["hard_failures"])

    def test_structured_metadata_stops_before_next_clause_on_same_page(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            directory = Path(name)
            source = self._write(directory, """---
standard_number: GB/T 50378-2019
---
## PDF 阅读器第 1 页（原文印刷页：1）
<!-- content_type: normative_body
pdf_page: 1
printed_page: 1
-->
<!-- table_metadata
table_id: GB50378-2019-table-3.1.1
table_no: 3.1.1
content_type: normative_table
pdf_page_start: 1
pdf_page_end: 1
-->
#### 表 3.1.1 示例
| 项目 | 分值 |
| --- | --- |
| A | 10 |
**3.1.2** 此条文必须成为独立节点。
""")
            nodes, _ = parse_markdown(source, ROOT)
            table = next(item for item in nodes if item["table_id"] == "GB50378-2019-table-3.1.1")
            clause = next(item for item in nodes if item["clause_id"] == "GB50378-2019:3.1.2")
            self.assertIn("表 3.1.1 示例", table["text"])
            self.assertNotIn("3.1.2", table["text"])
            self.assertIn("3.1.2", clause["text"])

    def test_duplicate_commentary_blocks_publish(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            directory = Path(name)
            source = self._write(directory, """---
standard_number: GB/T 50378-2019
---
## PDF 阅读器第 1 页（原文印刷页：1）
<!-- content_type: commentary
pdf_page: 1
printed_page: 1
-->
**5.2.99** 第一段。
## PDF 阅读器第 2 页（原文印刷页：2）
<!-- content_type: commentary
pdf_page: 2
printed_page: 2
-->
无关内容。
## PDF 阅读器第 3 页（原文印刷页：3）
<!-- content_type: commentary
pdf_page: 3
printed_page: 3
-->
**5.2.99** 不同段。
""")
            output = directory / "output"
            report = build([source], ROOT, output)
            self.assertFalse(report["publishable"])
            self.assertFalse((output / "active").exists())
            self.assertTrue(any(item["code"] == "duplicate_clause_id" for item in report["hard_failures"]))

    def test_build_syncs_document_status_from_standard_registry(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as name:
            directory = Path(name)
            source = self._write(directory, """---
standard_number: GB/T 50378-2019
---
## PDF 阅读器第 1 页（原文印刷页：1）
<!-- content_type: normative_body
pdf_page: 1
printed_page: 1
-->
**3.2.5** 总得分应按下式计算。
""")
            registry = directory / "standard_registry.json"
            registry.write_text(json.dumps([{"standard_id": "GB_T_50378_2019", "status": "current"}]), encoding="utf-8")
            output = directory / "output"
            report = build(
                [source],
                ROOT,
                output,
                source_verifications_path=directory / "no_source_verifications.json",
                standard_registry_path=registry,
            )
            self.assertTrue(report["publishable"])
            nodes = [json.loads(line) for line in (output / "active" / "clauses.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(nodes)
            self.assertEqual({item["document_status"] for item in nodes}, {"current"})


if __name__ == "__main__":
    unittest.main()
