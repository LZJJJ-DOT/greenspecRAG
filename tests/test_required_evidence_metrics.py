import json
from pathlib import Path

from deploy.evaluate_required_evidence import build_report


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_required_metrics_do_not_promote_legacy_gold_without_explicit_requirement(tmp_path):
    suite = tmp_path / "suite.json"
    questions = tmp_path / "questions.jsonl"
    report = tmp_path / "mode.json"
    write_json(suite, {"suite_id": "suite"})
    questions.write_text(
        "\n".join([
            json.dumps({"eval_id": "E001", "gold_evidence_ids": ["A"], "required_evidence": []}),
            json.dumps({"eval_id": "E002", "gold_evidence_ids": ["B"], "required_evidence": [{"kind": "table_id", "value": "T"}]}),
        ]),
        encoding="utf-8",
    )
    write_json(report, {
        "index_manifest_id": "index", "canonical_sha256": "canonical",
        "cases": [
            {"eval_id": "E001", "modes": {"hybrid": {"retrieval_trace": [{"evidence_id": "Z"}, {"evidence_id": "A"}]}}},
            {"eval_id": "E002", "modes": {"hybrid": {"retrieval_trace": [{"evidence_id": "B"}]}}},
        ],
    })

    result = build_report(suite_path=suite, questions_path=questions, mode_report_path=report, annotations_path=None)

    assert result["metrics"]["id_based_context_recall_at_10"]["rate"] == 1.0
    assert result["metrics"]["mrr_at_10_first_gold"]["mean"] == 0.75
    assert result["metrics"]["required_evidence_recall_at_10"]["cases"] == 1
    assert result["coverage"]["required_evidence_annotation_rate"] == 0.5
    assert result["cases"][0]["required_evidence_label_source"] == "needs_manual_annotation"


def test_approved_annotation_can_define_multiple_acceptable_ids(tmp_path):
    suite = tmp_path / "suite.json"
    questions = tmp_path / "questions.jsonl"
    report = tmp_path / "mode.json"
    annotations = tmp_path / "annotations.json"
    write_json(suite, {"suite_id": "suite"})
    questions.write_text(json.dumps({"eval_id": "E001", "gold_evidence_ids": ["A"], "required_evidence": []}) + "\n", encoding="utf-8")
    write_json(report, {"cases": [{"eval_id": "E001", "modes": {"hybrid": {"retrieval_trace": [{"evidence_id": "ALT"}]}}}]})
    write_json(annotations, {"annotations": [{"eval_id": "E001", "status": "approved", "requirements": [{"requirement_id": "REQ-001", "kind": "clause", "value": "5.1.1", "acceptable_evidence_ids": ["A", "ALT"]}]}]})

    result = build_report(suite_path=suite, questions_path=questions, mode_report_path=report, annotations_path=annotations)

    assert result["metrics"]["required_evidence_recall_at_10"]["rate"] == 1.0
    assert result["metrics"]["mrr_at_10_first_required_evidence"]["mean"] == 1.0
