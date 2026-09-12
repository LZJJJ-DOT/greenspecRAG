from __future__ import annotations

from pathlib import Path

from deploy.prepare_ragas_retrieval_dataset import build_ragas_rows, load_frozen_cases, load_suite


ROOT = Path(__file__).resolve().parents[1]


def test_regression_suite_locks_the_32_case_set() -> None:
    suite = load_suite(ROOT / "data/evaluation/regression_suite.v1.json")
    cases = load_frozen_cases(suite)
    assert len(cases) == 32
    assert sum(bool(case["gold_evidence_ids"]) for case in cases) == 24
    assert [case["eval_id"] for case in cases] == suite["case_ids"]


def test_regression_suite_v2_preserves_case_set_and_corrects_e028_wording() -> None:
    suite = load_suite(ROOT / "data/evaluation/regression_suite.v2.json")
    cases = load_frozen_cases(suite)
    e028 = next(case for case in cases if case["eval_id"] == "E028")
    assert len(cases) == 32
    assert e028["question"] == "表3.1.2规定的居住建筑体形系数限值是什么？"
    assert e028["gold_evidence_ids"] == ["GB55015-2021:GB55015-2021-table-3.1.2:13"]
    e006 = next(case for case in cases if case["eval_id"] == "E006")
    assert e006["gold_evidence_ids"] == ["GB50378-2019:GB50378-2019-table-3:p122"]
    assert e006["required_evidence"] == [{"kind": "table_id", "value": "GB50378-2019-table-3"}]


def test_adapter_separates_gold_scoring_from_no_gold_diagnostics() -> None:
    suite = load_suite(ROOT / "data/evaluation/regression_suite.v1.json")
    cases = load_frozen_cases(suite)
    canonical = {}
    for case in cases:
        for evidence_id in case["gold_evidence_ids"]:
            canonical.setdefault(evidence_id, {"clause_id": evidence_id, "retrieval_text": f"evidence text for {evidence_id}", "standard_id": "TEST", "clause_no": "1", "content_type": "normative_body"})
    report = {"cases": [{"eval_id": case["eval_id"], "modes": {"hybrid": {"returned_evidence_ids": case["gold_evidence_ids"][:1]}}} for case in cases if case["gold_evidence_ids"]]}
    rows, diagnostics = build_ragas_rows(cases, canonical, report, "hybrid")
    assert len(rows) == 24
    assert len(diagnostics) == 8
    assert all(row["reference_context_ids"] for row in rows)
    assert all("diagnostic_no_gold" in row["tags"] for row in diagnostics)
    assert "cross_page_table" in next(row for row in rows if row["eval_id"] == "E006")["metadata"]["tags"]
