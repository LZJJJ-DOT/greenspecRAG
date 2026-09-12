"""Convert legacy clauses JSONL records to canonical clause.v1.

The adapter is conservative: it never guesses a source hash, printed page,
parent, or version relationship. Missing provenance is retained as JSON null
and raises the manual-review flags required by the protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

CONTENT_TYPES = {
    "publication_info", "reference_standards", "toc", "normative_front_matter",
    "normative_body", "normative_clause", "normative_table", "normative_formula",
    "appendix", "appendix_clause", "commentary_front_matter", "commentary",
    "commentary_clause", "commentary_table", "commentary_formula", "commentary_appendix",
    "figure", "non_content",
}
DOCUMENT_STATUSES = {"current", "superseded", "partially_superseded", "pending_manual_review"}
VERIFICATION_STATUSES = {"verified", "needs_manual_review", "source_unverified"}
SOURCE_LEVELS = {"T0", "T0_CANDIDATE", "T1", "T2", "T3", "PROJECT"}
_MISSING = object()
_KNOWN_INPUT_FIELDS = {
    "clause_id", "id", "parent_id", "parent_clause_id", "document_id", "standard_id",
    "standard_name", "standard_version", "source_file", "source_sha256", "document_status",
    "content_type", "channel", "clause_type", "clause_no", "clause_title", "hierarchy",
    "text", "clause_text", "retrieval_text", "clause_summary", "region", "building_type",
    "design_phase", "green_target_scope", "requires_project_facts", "requires_calculation",
    "requires_manual_review", "source_page", "pdf_page", "pdf_page_start", "pdf_page_end",
    "printed_page_start", "printed_page_end", "source_level", "verification_status", "indexable",
    "table_id", "formula_id", "asset_ids", "supersession_ids", "provenance",
    "applicability_results", "risk_register", "evidence_verification", "legacy_compat",
}

class CompatibilityError(ValueError):
    """Raised when a legacy JSONL record cannot be converted safely."""

def _value(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = record.get(name, _MISSING)
        if value is not _MISSING and value is not None and value != "":
            return value
    return default

def _text(value: Any) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()

def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in {"1", "true", "yes", "y", "是"}

def _int(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", _text(value))
    return int(match.group(0)) if match else None

def _page_range(value: Any) -> tuple[int | None, int | None]:
    if isinstance(value, Mapping):
        return _int(value.get("start")), _int(value.get("end", value.get("start")))
    if isinstance(value, (list, tuple)) and value:
        start = _int(value[0])
        return start, _int(value[1]) if len(value) > 1 else start
    if value is None:
        return None, None
    numbers = [int(n) for n in re.findall(r"\d+", _text(value))]
    return (numbers[0], numbers[-1]) if numbers else (None, None)

def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]

def _stable_id(value: Any, *, fallback: str) -> str:
    raw = _text(value) or fallback
    safe = re.sub(r"[^A-Za-z0-9._:-]+", "_", raw).strip("_")
    return safe[:256] or fallback

def _legacy_id(record: Mapping[str, Any], defaults: Mapping[str, Any], text: str) -> str:
    explicit = _value(record, "clause_id", "id")
    if explicit is not None:
        return _stable_id(explicit, fallback="legacy:clause")
    document_id = _value(record, "document_id", default=defaults.get("document_id", "legacy"))
    clause_no = _value(record, "clause_no")
    suffix = _text(clause_no) if clause_no is not None else hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return _stable_id(f"{document_id}:{suffix}", fallback="legacy:clause")

def _content_type(record: Mapping[str, Any]) -> str:
    raw = _text(_value(record, "content_type", "channel", default="normative_clause")).lower()
    aliases = {"local_clause": "normative_clause", "clause": "normative_clause", "normative": "normative_clause", "table": "normative_table", "formula": "normative_formula", "comment": "commentary"}
    result = aliases.get(raw, raw)
    return result if result in CONTENT_TYPES else "normative_clause"

def adapt_record(record: Mapping[str, Any], *, defaults: Mapping[str, Any] | None = None, known_clause_ids: set[str] | None = None) -> dict[str, Any]:
    """Return one canonical clause while preserving legacy protocol fields."""
    defaults = dict(defaults or {})
    text = _text(_value(record, "text", "clause_text"))
    content_type = _content_type(record)
    source_file = _value(record, "source_file", default=defaults.get("source_file"))
    source_sha256 = _value(record, "source_sha256", default=defaults.get("source_sha256"))
    pdf_start, pdf_end = _page_range(_value(record, "source_page", "pdf_page"))
    pdf_start = _int(_value(record, "pdf_page_start", default=pdf_start))
    pdf_end = _int(_value(record, "pdf_page_end", default=pdf_end if pdf_end is not None else pdf_start))
    printed_start = _int(_value(record, "printed_page_start"))
    printed_end = _int(_value(record, "printed_page_end"))
    parent_id = _value(record, "parent_id", "parent_clause_id")
    parent_id = _stable_id(parent_id, fallback="legacy:parent") if parent_id is not None else None
    clause_id = _legacy_id(record, defaults, text)
    missing_provenance = not source_file or not source_sha256
    missing_printed_page = printed_start is None or printed_end is None
    missing_parent = parent_id is not None and known_clause_ids is not None and parent_id not in known_clause_ids
    requires_manual_review = _bool(record.get("requires_manual_review"), False) or missing_provenance or missing_printed_page or missing_parent
    explicit_verification = _value(record, "verification_status")
    verification_status = explicit_verification if explicit_verification in VERIFICATION_STATUSES else ("needs_manual_review" if requires_manual_review else "source_unverified")
    document_status = _value(record, "document_status", default=defaults.get("document_status", "pending_manual_review"))
    if document_status not in DOCUMENT_STATUSES:
        document_status = "pending_manual_review"
    source_level = _value(record, "source_level", default=defaults.get("source_level", "T0_CANDIDATE"))
    if source_level not in SOURCE_LEVELS:
        source_level = "T0_CANDIDATE"
    retrieval_text = _value(record, "retrieval_text")
    if retrieval_text is None:
        retrieval_text = " ".join(part for part in [_text(_value(record, "clause_no")), _text(_value(record, "clause_title")), _text(_value(record, "clause_summary")), text] if part)
    legacy_compat = dict(_value(record, "legacy_compat", default={}) or {})
    for key in ("applicability_results", "risk_register", "evidence_verification"):
        if key in record:
            legacy_compat[key] = record[key]
    provenance = dict(_value(record, "provenance", default={}) or {})
    provenance.setdefault("adapter", "greenspec_rag.compat.clauses_adapter")
    provenance.setdefault("adapter_version", "1")
    provenance.setdefault("legacy_clause_id", record.get("clause_id", record.get("id")))
    return {
        "schema_version": "clause.v1", "clause_id": clause_id, "parent_id": parent_id,
        "document_id": _value(record, "document_id", default=defaults.get("document_id")),
        "standard_id": _value(record, "standard_id", default=defaults.get("standard_id")),
        "standard_name": _value(record, "standard_name", default=defaults.get("standard_name")),
        "standard_version": _value(record, "standard_version", default=defaults.get("standard_version")),
        "source_file": source_file, "source_sha256": source_sha256, "document_status": document_status,
        "content_type": content_type, "clause_type": _value(record, "clause_type", default="clause"),
        "clause_no": _value(record, "clause_no"), "clause_title": _value(record, "clause_title"), "clause_summary": _value(record, "clause_summary"),
        "hierarchy": _list(_value(record, "hierarchy", default=[])), "text": text,
        "retrieval_text": _text(retrieval_text), "region": _value(record, "region", default=defaults.get("region")),
        "building_type": _value(record, "building_type", default=defaults.get("building_type")),
        "design_phase": _value(record, "design_phase", default=defaults.get("design_phase")),
        "green_target_scope": _value(record, "green_target_scope", default=defaults.get("green_target_scope")),
        "requires_project_facts": _bool(record.get("requires_project_facts")),
        "requires_calculation": _bool(record.get("requires_calculation")),
        "requires_manual_review": requires_manual_review,
        "pdf_page_start": pdf_start, "pdf_page_end": pdf_end,
        "printed_page_start": printed_start,
        "printed_page_end": printed_end, "source_level": source_level,
        "verification_status": verification_status,
        "indexable": _bool(record.get("indexable"), not content_type.startswith("commentary") and content_type != "non_content"),
        "table_id": _value(record, "table_id"), "formula_id": _value(record, "formula_id"),
        "asset_ids": _list(_value(record, "asset_ids", default=[])),
        "supersession_ids": _list(_value(record, "supersession_ids", default=[])), "provenance": provenance,
        "applicability_results": record.get("applicability_results"), "risk_register": record.get("risk_register"),
        "evidence_verification": record.get("evidence_verification"), "legacy_compat": legacy_compat,
        "legacy_extra": {key: value for key, value in record.items() if key not in _KNOWN_INPUT_FIELDS},
    }

def canonicalize_records(records: Iterable[Mapping[str, Any]], *, defaults: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Convert and deterministically sort legacy records, rejecting duplicate IDs."""
    raw_records = [dict(record) for record in records]
    defaults = dict(defaults or {})
    tentative_ids = {_legacy_id(record, defaults, _text(_value(record, "text", "clause_text"))) for record in raw_records}
    if len(tentative_ids) != len(raw_records):
        raise CompatibilityError("duplicate clause_id after legacy ID normalization")
    result = [adapt_record(record, defaults=defaults, known_clause_ids=tentative_ids) for record in raw_records]
    result.sort(key=lambda item: item["clause_id"])
    return result

def adapt_jsonl(input_path: str | Path, output_path: str | Path | None = None, *, defaults: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Read legacy JSONL and optionally write canonical JSONL."""
    source = Path(input_path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompatibilityError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise CompatibilityError(f"expected an object at {source}:{line_number}")
        records.append(dict(value))
    result = canonicalize_records(records, defaults=defaults)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in result), encoding="utf-8")
    return result

def main() -> int:
    parser = argparse.ArgumentParser(description="Adapt legacy clauses.jsonl to canonical clause.v1 JSONL")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--standard-id")
    parser.add_argument("--standard-name")
    parser.add_argument("--standard-version")
    parser.add_argument("--document-id")
    parser.add_argument("--source-file")
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    defaults = {key: value for key, value in {"standard_id": args.standard_id, "standard_name": args.standard_name, "standard_version": args.standard_version, "document_id": args.document_id, "source_file": args.source_file, "source_sha256": args.source_sha256}.items() if value is not None}
    adapt_jsonl(args.input, args.output, defaults=defaults)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
