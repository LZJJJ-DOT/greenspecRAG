"""Build canonical clause.v1 drafts from annotated Markdown without publishing unsafe data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

COMMENT = re.compile(r"<!--(?P<body>.*?)-->", re.DOTALL)
PAGE = re.compile(r"^##\s+PDF(?:\s*阅读器)?第\s*(\d+)\s*页（(?:原书页码|原文印刷页)：([^）]+)）\s*$", re.MULTILINE)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CLAUSE = re.compile(r"^\*\*(\d+\.\d+\.\d+(?:\.\d+){0,2})\*\*\s*(.*)$", re.MULTILINE)
TABLE = re.compile(r"^#{3,6}\s+((?:续表\s*|表\s*)[^\n]+)", re.MULTILINE)
IMAGE = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")
TYPES = {"publication_info", "reference_standards", "toc", "normative_front_matter", "normative_body", "normative_clause", "normative_table", "normative_formula", "appendix", "appendix_clause", "commentary_front_matter", "commentary", "commentary_clause", "commentary_table", "commentary_formula", "commentary_appendix", "figure", "non_content"}
DOCUMENT_STATUSES = {"current", "superseded", "partially_superseded", "pending_manual_review"}


class BuildError(ValueError):
    pass


def scalar(value: str) -> Any:
    value = value.strip().strip('"\'')
    if value.lower() in {"null", "none", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def number(value: Any) -> int | None:
    match = re.search(r"\d+", str(value)) if value is not None else None
    return int(match.group()) if match else None


def truth(value: Any, default: bool = False) -> bool:
    return default if value is None else str(value).lower() in {"true", "1", "yes", "y", "是"}


def safe_id(value: str) -> str:
    return (re.sub(r"[^A-Za-z0-9._:-]+", "_", value).strip("_") or "node")[:256]


def clean(value: str) -> str:
    value = COMMENT.sub("", value)
    value = re.sub(r"^#{1,6}\s+", "", value, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def parse_front(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        raise BuildError("unterminated front matter")
    return {key.strip(): scalar(value) for line in text[4:end].splitlines() if ":" in line for key, value in [line.split(":", 1)]}


def parse_comment(body: str) -> tuple[dict[str, Any], str | None]:
    values: dict[str, Any] = {}
    kind = None
    for line in body.splitlines():
        line = line.strip()
        if line in {"table_metadata", "formula_metadata", "figure_metadata"}:
            kind = line
        elif ":" in line:
            # Early page annotations place several key/value pairs on one line,
            # e.g. ``pdf_page: 1 source_page: null``.
            pairs = re.findall(r"([A-Za-z_]+)\s*:\s*(.*?)(?=\s+[A-Za-z_]+\s*:|$)", line)
            for key, value in pairs:
                values[key.strip()] = scalar(value)
    return values, kind


def annotations(text: str) -> list[tuple[int, int, dict[str, Any], str | None]]:
    return [(item.start(), item.end(), *parse_comment(item.group("body"))) for item in COMMENT.finditer(text)]


def page_regions(text: str, marks: list[tuple[int, int, dict[str, Any], str | None]]) -> list[dict[str, Any]]:
    headings = list(PAGE.finditer(text))
    regions: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        start, end = heading.start(), headings[index + 1].start() if index + 1 < len(headings) else len(text)
        meta: dict[str, Any] = {"pdf_page": number(heading.group(1)), "printed_page": number(heading.group(2))}
        for mark_start, _, values, kind in marks:
            if start <= mark_start < end and kind is None:
                meta.update(values)
        meta["pdf_page"] = number(meta.get("pdf_page"))
        meta["printed_page"] = number(meta.get("printed_page"))
        regions.append({"start": start, "end": end, "meta": meta})
    if not regions:
        raise BuildError("missing PDF page headings")
    return regions


def region_for(regions: list[dict[str, Any]], position: int) -> dict[str, Any] | None:
    return next((item for item in regions if item["start"] <= position < item["end"]), None)



def pages(regions: list[dict[str, Any]], start: int, end: int, meta: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    first = region_for(regions, start)
    covered = [
        item
        for item in regions
        if item["start"] < end and item["end"] > start
    ]

    first_meta = first["meta"] if first else {}
    last_meta = covered[-1]["meta"] if covered else {}

    pdf_page_start = number(
        meta.get(
            "pdf_page_start",
            meta.get("pdf_page", first_meta.get("pdf_page")),
        )
    )
    pdf_page_end = number(
        meta.get(
            "pdf_page_end",
            last_meta.get("pdf_page"),
        )
    )
    def printed_page_for_pdf(regions, pdf_page):
        if pdf_page is None:
            return None
        for region in regions:
            meta = region["meta"]
            if number(meta.get("pdf_page")) == pdf_page:
                return number(meta.get("printed_page"))
        return None
    printed_page_start = number(
        meta.get(
            "printed_page_start",
            meta.get("printed_page"),
        )
    )
    if printed_page_start is None:
        printed_page_start = printed_page_for_pdf(regions, pdf_page_start)
    if printed_page_start is None:
        printed_page_start = number(first_meta.get("printed_page"))

    printed_page_end = number(meta.get("printed_page_end"))
    if printed_page_end is None:
        printed_page_end = printed_page_for_pdf(regions, pdf_page_end)
    if printed_page_end is None:
        printed_page_end = number(last_meta.get("printed_page"))

    return (
        pdf_page_start,
        pdf_page_end,
        printed_page_start,
        printed_page_end,
    )





def standard(front: dict[str, Any]) -> dict[str, str]:
    value = str(front.get("standard_number", "")).replace(" ", "")
    if "55015" in value:
        return {"document_id": "GB55015-2021", "standard_id": "GB_55015_2021", "standard_name": "建筑节能与可再生能源利用通用规范", "standard_version": "2021"}
    if "50378" in value:
        return {"document_id": "GB50378-2019", "standard_id": "GB_T_50378_2019", "standard_name": "绿色建筑评价标准", "standard_version": "2019"}
    raise BuildError(f"unsupported standard number: {value}")


def hierarchy(clause_no: str | None) -> list[int]:
    return [int(part) for part in (clause_no or "").split(".") if part.isdigit()]


def node(*, info: dict[str, str], source_file: str, sha: str, region_meta: dict[str, Any], node_id: str, parent_id: str | None, content_type: str, clause_type: str, clause_no: str | None, title: str | None, text: str, page_values: tuple[int | None, int | None, int | None, int | None], table_id: str | None = None, formula_id: str | None = None, asset_ids: list[str] | None = None, indexable: bool | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    commentary = content_type.startswith("commentary")
    indexable = truth(region_meta.get("indexable"), not commentary) if indexable is None else indexable
    # Commentary prose stays archived.  A manually marked commentary table is
    # an exception: it remains visibly typed as commentary, but is searchable
    # so a caller can retrieve the table values and its exact locator.
    if commentary and content_type != "commentary_table":
        indexable = False
    text = text or title or "[empty source node]"
    return {"schema_version": "clause.v1", "clause_id": safe_id(node_id), "parent_id": parent_id, **info, "source_file": source_file, "source_sha256": sha, "document_status": "pending_manual_review", "content_type": content_type, "clause_type": clause_type, "clause_no": clause_no, "clause_title": title, "hierarchy": hierarchy(clause_no), "text": text, "retrieval_text": " ".join(part for part in [clause_no or "", title or "", text] if part), "region": "全国", "building_type": None, "design_phase": None, "green_target_scope": None, "requires_project_facts": False, "requires_calculation": content_type in {"normative_table", "normative_formula"}, "requires_manual_review": False, "pdf_page_start": page_values[0], "pdf_page_end": page_values[1], "printed_page_start": page_values[2], "printed_page_end": page_values[3], "source_level": "T0_CANDIDATE", "verification_status": "source_unverified", "indexable": indexable, "table_id": table_id, "formula_id": formula_id, "asset_ids": asset_ids or [], "supersession_ids": [], "provenance": {"adapter": "greenspec_rag.ingestion.canonical_builder", "adapter_version": "1", **(extra or {})}, "applicability_results": None, "risk_register": None, "evidence_verification": None}


def parse_markdown(path: Path, project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = path.resolve()
    raw, text = path.read_bytes(), path.read_text(encoding="utf-8")
    front, info, sha = parse_front(text), standard(parse_front(text)), hashlib.sha256(raw).hexdigest()
    marks, regions = annotations(text), page_regions(text, annotations(text))
    source_file = path.relative_to(project_root).as_posix()
    nodes: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    occupied: list[tuple[int, int]] = []
    commentary_positions = [match.start() for match in HEADING.finditer(text) if "条文说明" in match.group(2)]
    commentary_start = min(commentary_positions) if commentary_positions else len(text) + 1
    for match in HEADING.finditer(text):
        if PAGE.fullmatch(match.group()):
            continue
        title = match.group(2).strip()
        found = re.match(r"(\d+(?:\.\d+)*)\s+(.+)", title) or re.match(r"(附录[A-Z])\s*(.*)", title)
        if not found or title.startswith(("表", "续表")) or found.group(1).count(".") >= 2:
            continue
        clause_no, label = found.group(1), found.group(2) or title
        region = region_for(regions, match.start())
        if not region:
            continue
        if region["meta"].get("content_type") in {"toc", "publication_info", "reference_standards", "normative_front_matter", "non_content"}:
            continue
        commentary = match.start() >= commentary_start or "条文说明" in title or str(region["meta"].get("content_type", "")).startswith("commentary")
        ctype = "commentary_appendix" if commentary and clause_no.startswith("附录") else ("commentary" if commentary else ("appendix" if clause_no.startswith("附录") else "normative_body"))
        parent_no = clause_no.rsplit(".", 1)[0] if "." in clause_no else None
        nid = f"{info['document_id']}:{clause_no}{':commentary' if commentary else ''}"
        item = node(info=info, source_file=source_file, sha=sha, region_meta=region["meta"], node_id=nid, parent_id=ids.get(parent_no), content_type=ctype, clause_type="appendix" if clause_no.startswith("附录") else "section", clause_no=clause_no, title=label, text=title, page_values=pages(regions, match.start(), match.end(), {}), indexable=False if commentary else None)
        nodes.append(item); occupied.append((match.start(), match.end()))
        if not commentary: ids[clause_no] = item["clause_id"]
    clauses, node_marks = list(CLAUSE.finditer(text)), [item for item in marks if item[3] in {"table_metadata", "formula_metadata", "figure_metadata"}]
    stops = sorted([item.start() for item in clauses] + [item.start() for item in TABLE.finditer(text)] + [item[0] for item in node_marks] + [item["end"] for item in regions])
    for match in clauses:
        end = next((position for position in stops if position > match.start()), len(text))
        region = region_for(regions, match.start())
        if not region: continue
        clause_no, body = match.group(1), clean(text[match.start():end])
        commentary = match.start() >= commentary_start or str(region["meta"].get("content_type", "")).startswith("commentary")
        parent_no = clause_no.rsplit(".", 1)[0] if "." in clause_no else None
        item = node(info=info, source_file=source_file, sha=sha, region_meta=region["meta"], node_id=f"{info['document_id']}:{clause_no}{':commentary' if commentary else ''}", parent_id=ids.get(parent_no), content_type="commentary_clause" if commentary else "normative_clause", clause_type="commentary" if commentary else "clause", clause_no=clause_no, title=None, text=body, page_values=pages(regions, match.start(), end, {}), indexable=not commentary)
        nodes.append(item); occupied.append((match.start(), end))
        if not commentary: ids[clause_no] = item["clause_id"]
    for index, mark in enumerate(node_marks):
        start, comment_end, meta, kind = mark
        region = region_for(regions, start)
        if not region: continue
        # A node-level table/formula annotation cannot consume text from a
        # different PDF page, nor from the next clause on the same page.  A
        # multi-page table has one annotation per page and is linked through
        # table_id/continuation_of, while a same-page clause marker always
        # begins a separate canonical node.
        next_mark = node_marks[index + 1][0] if index + 1 < len(node_marks) else region["end"]
        next_clause = next((item.start() for item in clauses if comment_end < item.start() < region["end"]), region["end"])
        end = min(next_mark, next_clause, region["end"])
        body, heading = clean(text[comment_end:end]), HEADING.search(text[comment_end:end])
        ctype = str(meta.get("content_type") or {"table_metadata": "normative_table", "formula_metadata": "normative_formula", "figure_metadata": "figure"}[kind])
        if ctype == "normative_table_continuation": ctype = "normative_table"
        if ctype == "commentary_table_continuation": ctype = "commentary_table"
        if start >= commentary_start and ctype == "normative_table": ctype = "commentary_table"
        if start >= commentary_start and ctype == "normative_formula": ctype = "commentary_formula"
        table_id, formula_id = (str(meta["table_id"]) if meta.get("table_id") else None), (str(meta["formula_id"]) if meta.get("formula_id") else None)
        clause_no = str(meta.get("table_no") or meta.get("equation_no") or "") or None
        parent_key = (clause_no or "").replace("表", "").strip()
        parent_id = ids.get(parent_key) or ids.get(parent_key.rsplit(".", 1)[0] if "." in parent_key else "")
        stable_suffix = f"{table_id}:p{number(meta.get('pdf_page_start', meta.get('pdf_page', region['meta'].get('pdf_page'))))}" if table_id else (formula_id or "figure")
        item = node(info=info, source_file=source_file, sha=sha, region_meta=region["meta"], node_id=f"{info['document_id']}:{stable_suffix}", parent_id=parent_id, content_type=ctype if ctype in TYPES else "non_content", clause_type={"normative_table": "table", "commentary_table": "table", "normative_formula": "formula", "figure": "figure"}.get(ctype, "node"), clause_no=clause_no, title=heading.group(2).strip() if heading else None, text=body, page_values=pages(regions, start, end, meta), table_id=table_id, formula_id=formula_id, indexable=truth(meta.get("indexable"), not ctype.startswith("commentary")), extra={"continuation_of": meta.get("continuation_of")})
        nodes.append(item); occupied.append((start, end))
    table_matches = list(TABLE.finditer(text))
    for table_index, match in enumerate(table_matches):
        if any(start <= match.start() < end for start, end in occupied): continue
        region = region_for(regions, match.start())
        table_no = re.search(r"(?:续表\s*|表\s*)([\d.\-]+)", match.group(1))
        if not region or not table_no: continue
        next_start = table_matches[table_index + 1].start() if table_index + 1 < len(table_matches) else region["end"]
        end, no, title = min(region["end"], next_start), table_no.group(1), match.group(1).strip()
        tid = f"{info['document_id']}-table-{no}"
        nodes.append(node(info=info, source_file=source_file, sha=sha, region_meta=region["meta"], node_id=f"{info['document_id']}:{tid}:{region['meta'].get('pdf_page')}", parent_id=ids.get(no) or ids.get(no.rsplit('.', 1)[0] if '.' in no else ''), content_type="commentary_table" if match.start() >= commentary_start else "normative_table", clause_type="table", clause_no=f"表{no}", title=title, text=clean(text[match.start():end]), page_values=pages(regions, match.start(), end, {}), table_id=tid, extra={"continuation_of": tid if title.startswith("续表") else None}))
        occupied.append((match.start(), end))
    for match in IMAGE.finditer(text):
        region = region_for(regions, match.start())
        if not region: continue
        asset = match.group(2).strip()
        asset_id = f"{info['document_id']}:asset:{hashlib.sha256(asset.encode('utf-8')).hexdigest()[:16]}"
        nodes.append(node(info=info, source_file=source_file, sha=sha, region_meta=region["meta"], node_id=f"{asset_id}:node", parent_id=None, content_type="figure", clause_type="figure", clause_no=None, title=match.group(1) or None, text=match.group(0), page_values=pages(regions, match.start(), match.end(), {}), asset_ids=[asset_id], indexable=False, extra={"asset_path": asset}))
    for region in regions:
        if any(region["start"] <= start < region["end"] for start, _ in occupied): continue
        body, ctype = clean(text[region["start"]:region["end"]]), str(region["meta"].get("content_type", "non_content"))
        if body:
            nodes.append(node(info=info, source_file=source_file, sha=sha, region_meta=region["meta"], node_id=f"{info['document_id']}:pdf:{region['meta'].get('pdf_page')}", parent_id=None, content_type=ctype if ctype in TYPES else "non_content", clause_type="page_region", clause_no=None, title=None, text=body, page_values=pages(regions, region["start"], region["end"], {}), indexable=truth(region["meta"].get("indexable"), False)))
    # The source sometimes repeats a clause marker after a page/table break.
    # Merge only contiguous, same-type fragments; distant same-number nodes stay
    # separate so validation can block a potentially incorrect publication.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in nodes:
        grouped.setdefault(item["clause_id"], []).append(item)
    merged: list[dict[str, Any]] = []
    for _, items in grouped.items():
        items.sort(key=lambda item: (item["pdf_page_start"] or 0, item["pdf_page_end"] or 0))
        current = items[0]
        for candidate in items[1:]:
            contiguous = (current["pdf_page_end"] is not None and candidate["pdf_page_start"] is not None and candidate["pdf_page_start"] <= current["pdf_page_end"] + 1)
            if current["content_type"] == candidate["content_type"] and contiguous:
                current["text"] = current["text"] + "\n\n" + candidate["text"]
                current["retrieval_text"] = current["retrieval_text"] + "\n\n" + candidate["text"]
                current["pdf_page_end"] = max(current["pdf_page_end"], candidate["pdf_page_end"] or current["pdf_page_end"])
                current["printed_page_end"] = candidate["printed_page_end"] or current["printed_page_end"]
                current["provenance"].setdefault("merged_source_fragments", 1)
                current["provenance"]["merged_source_fragments"] += 1
            else:
                merged.append(current); current = candidate
        merged.append(current)
    return sorted(merged, key=lambda item: item["clause_id"]), {"front": front, "sha": sha, "source_file": source_file, "info": info, "regions": regions}


def load_source_verifications(path: Path) -> dict[str, dict[str, Any]]:
    """Load the small, human-maintained document verification ledger."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("sources", [])
    if not isinstance(records, list):
        raise BuildError("source verification ledger must contain a sources list")
    verified: dict[str, dict[str, Any]] = {}
    for record in records:
        standard_id = record.get("standard_id")
        if not isinstance(standard_id, str) or not standard_id:
            raise BuildError("source verification record has no standard_id")
        if standard_id in verified:
            raise BuildError(f"duplicate source verification record: {standard_id}")
        verified[standard_id] = record
    return verified


def load_document_statuses(path: Path) -> dict[str, str]:
    """Load the registry's canonical document status by standard ID."""
    if not path.is_file():
        raise BuildError(f"standard registry not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise BuildError("standard registry must contain a list")
    statuses: dict[str, str] = {}
    for record in records:
        standard_id, status = record.get("standard_id"), record.get("status")
        if not isinstance(standard_id, str) or not standard_id:
            raise BuildError("standard registry record has no standard_id")
        if status not in DOCUMENT_STATUSES:
            raise BuildError(f"invalid document status for {standard_id}: {status!r}")
        if standard_id in statuses:
            raise BuildError(f"duplicate standard registry record: {standard_id}")
        statuses[standard_id] = status
    return statuses


def apply_document_status(nodes: list[dict[str, Any]], standard_id: str, statuses: dict[str, str]) -> list[dict[str, Any]]:
    """Make the registry, rather than the Markdown adapter default, authoritative."""
    status = statuses.get(standard_id)
    if status is None:
        return [{"code": "missing_registry_document_status", "standard_id": standard_id}]
    for item in nodes:
        item["document_status"] = status
    return []


def apply_source_verification(nodes: list[dict[str, Any]], meta: dict[str, Any], project_root: Path, record: dict[str, Any] | None, *, require_verified: bool) -> list[dict[str, Any]]:
    """Promote a document only when its manually verified PDF and Markdown are unchanged."""
    if record is None:
        if require_verified:
            return [{"code": "missing_source_verification", "standard_id": meta["info"]["standard_id"]}]
        return []
    if record.get("status") != "verified":
        if require_verified:
            return [{"code": "source_not_verified", "standard_id": meta["info"]["standard_id"]}]
        return []
    processed = record.get("processed_markdown", {})
    raw_pdf = record.get("raw_pdf", {})
    expected_markdown = processed.get("sha256")
    expected_pdf = raw_pdf.get("sha256")
    markdown_path = project_root / str(processed.get("path", ""))
    pdf_path = project_root / str(raw_pdf.get("path", ""))
    errors: list[dict[str, Any]] = []
    if markdown_path.resolve() != (project_root / meta["source_file"]).resolve() or expected_markdown != meta["sha"]:
        errors.append({"code": "verified_markdown_changed", "standard_id": meta["info"]["standard_id"]})
    if not pdf_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", str(expected_pdf or "")) or hashlib.sha256(pdf_path.read_bytes()).hexdigest() != expected_pdf:
        errors.append({"code": "verified_pdf_changed", "standard_id": meta["info"]["standard_id"]})
    expected_pages = raw_pdf.get("page_count")
    actual_pages = [item["meta"].get("pdf_page") for item in meta["regions"]]
    if not isinstance(expected_pages, int) or expected_pages < 1 or actual_pages != list(range(1, expected_pages + 1)):
        errors.append({"code": "verified_page_map_changed", "standard_id": meta["info"]["standard_id"], "expected_page_count": expected_pages, "actual_page_count": len(actual_pages)})
    if errors:
        return errors
    verification = {
        "verification_id": record.get("verification_id"),
        "reviewed_by": record.get("reviewed_by"),
        "reviewed_at": record.get("reviewed_at"),
        "raw_pdf_sha256": expected_pdf,
    }
    for node in nodes:
        node["source_level"] = "T0"
        node["verification_status"] = "verified"
        node["provenance"]["source_verification"] = verification
    meta["source_verification"] = verification
    return []


def validate(nodes: list[dict[str, Any]], meta: dict[str, Any], project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings = [] if meta.get("source_verification") else [{"code": "source_unverified", "message": "The supplied Markdown is hashed; source PDF provenance remains unverified."}]
    for key, count in Counter(item["clause_id"] for item in nodes).items():
        if count > 1: errors.append({"code": "duplicate_clause_id", "clause_id": key, "count": count})
    actual_pages = [item["meta"].get("pdf_page") for item in meta["regions"]]
    if actual_pages != list(range(1, len(actual_pages) + 1)): errors.append({"code": "non_contiguous_pdf_pages", "actual": actual_pages})
    for item in nodes:
        if item["source_sha256"] != meta["sha"] or not re.fullmatch(r"[0-9a-f]{64}", item["source_sha256"]): errors.append({"code": "invalid_source_sha256", "clause_id": item["clause_id"]})
        if item["pdf_page_start"] is None or item["pdf_page_end"] is None: errors.append({"code": "untraceable_node_page", "clause_id": item["clause_id"]})
        if item["indexable"] and (item["content_type"] not in TYPES or not item["source_file"]): errors.append({"code": "invalid_indexable_node", "clause_id": item["clause_id"]})
        if item["content_type"].startswith("commentary") and item["indexable"] and item["content_type"] != "commentary_table": errors.append({"code": "commentary_indexable", "clause_id": item["clause_id"]})
        if item["content_type"].endswith("table") and not item["table_id"]: errors.append({"code": "table_without_id", "clause_id": item["clause_id"]})
        if item["content_type"].endswith("formula") and not item["formula_id"]: errors.append({"code": "formula_without_id", "clause_id": item["clause_id"]})
        if item["asset_ids"]:
            asset = item["provenance"].get("asset_path")
            if not asset or Path(asset).is_absolute() or ".." in Path(asset).parts or not (project_root / "data" / "processed" / asset).is_file(): errors.append({"code": "missing_or_unsafe_asset", "clause_id": item["clause_id"]})
    for key, count in Counter(item["formula_id"] for item in nodes if item["formula_id"]).items():
        if count > 1: errors.append({"code": "duplicate_formula_id", "formula_id": key, "count": count})
    by_table: dict[str, list[dict[str, Any]]] = {}
    for item in nodes:
        if item["table_id"]: by_table.setdefault(item["table_id"], []).append(item)
    for key, items in by_table.items():
        if any(item["provenance"].get("continuation_of") for item in items) and not any(not item["provenance"].get("continuation_of") for item in items): errors.append({"code": "orphan_table_continuation", "table_id": key})
    appendix = any(item["content_type"] in {"appendix", "appendix_clause", "commentary_appendix"} for item in nodes)
    if meta["front"].get("appendices_detected") is not None and truth(meta["front"]["appendices_detected"]) != appendix: errors.append({"code": "appendices_detected_mismatch", "declared": truth(meta["front"]["appendices_detected"]), "actual": appendix})
    return errors, warnings


def apply_duplicate_approvals(nodes: list[dict[str, Any]], approved: set[str]) -> None:
    """Apply only explicitly approved same-number commentary disambiguations."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in nodes:
        grouped.setdefault(item["clause_id"], []).append(item)
    for clause_id, items in grouped.items():
        if len(items) < 2 or clause_id not in approved:
            continue
        for item in items:
            page = item["pdf_page_start"]
            if page is None:
                raise BuildError(f"approved duplicate {clause_id} lacks a PDF page")
            item["clause_id"] = f"{clause_id}:p{page}"
            item["provenance"]["duplicate_clause_number_approval"] = "data/governance/approved_duplicate_clause_nodes.json"
            item["provenance"]["original_clause_id"] = clause_id


def build(inputs: Iterable[Path], project_root: Path, output_root: Path, *, source_verifications_path: Path | None = None, standard_registry_path: Path | None = None, require_verified_sources: bool = False) -> dict[str, Any]:
    run = f"extract_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"; nodes: list[dict[str, Any]] = []; sources = []; errors: list[dict[str, Any]] = []; warnings: list[dict[str, Any]] = []
    approval_path = project_root / "data" / "governance" / "approved_duplicate_clause_nodes.json"
    approved = set(json.loads(approval_path.read_text(encoding="utf-8")).get("approved_clause_ids", [])) if approval_path.is_file() else set()
    verification_ledger = load_source_verifications(source_verifications_path or project_root / "data" / "governance" / "source_verifications.json")
    document_statuses = load_document_statuses(standard_registry_path or project_root / "data" / "registry" / "standard_registry.json")
    for path in inputs:
        current, meta = parse_markdown(path, project_root); apply_duplicate_approvals(current, approved)
        status_errors = apply_document_status(current, meta["info"]["standard_id"], document_statuses)
        verification_errors = apply_source_verification(current, meta, project_root, verification_ledger.get(meta["info"]["standard_id"]), require_verified=require_verified_sources)
        current_errors, current_warnings = validate(current, meta, project_root)
        nodes += current; errors += [{**item, "source_file": meta["source_file"]} for item in status_errors + verification_errors + current_errors]; warnings += [{**item, "source_file": meta["source_file"]} for item in current_warnings]; sources.append({**meta["info"], "source_file": meta["source_file"], "source_sha256": meta["sha"], "page_count": len(meta["regions"]), "node_count": len(current), "document_status": document_statuses.get(meta["info"]["standard_id"]), "verification_status": current[0]["verification_status"] if current else "source_unverified", "source_verification": meta.get("source_verification")})
    for key, count in Counter(item["clause_id"] for item in nodes).items():
        if count > 1: errors.append({"code": "duplicate_clause_id_cross_document", "clause_id": key, "count": count})
    output_root.mkdir(parents=True, exist_ok=True); draft = output_root / f"clauses.{run}.draft.jsonl"
    draft.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in sorted(nodes, key=lambda item: item["clause_id"])), encoding="utf-8")
    manifest = {"source_manifest_id": f"source_{run}", "created_at": datetime.now(timezone.utc).isoformat(), "sources": sources}
    (output_root / f"source_manifest.{run}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"extraction_run_id": run, "draft": draft.name, "node_count": len(nodes), "hard_failures": errors, "warnings": warnings, "publishable": not errors}
    (output_root / f"audit_report.{run}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / f"failed_samples.{run}.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in errors), encoding="utf-8")
    if not errors:
        stage = Path(tempfile.mkdtemp(prefix="greenspec-publish-", dir=output_root)); active = output_root / "active"
        try:
            shutil.copy2(draft, stage / "clauses.jsonl"); (stage / "index_manifest.json").write_text(json.dumps({"index_manifest_id": f"index_{run}", "source_manifest_id": manifest["source_manifest_id"], "canonical_sha256": hashlib.sha256(draft.read_bytes()).hexdigest()}, indent=2) + "\n", encoding="utf-8")
            if active.exists(): shutil.rmtree(active)
            os.replace(stage, active)
        finally:
            if stage.exists(): shutil.rmtree(stage)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("inputs", nargs="+", type=Path); parser.add_argument("--project-root", type=Path, default=Path.cwd()); parser.add_argument("--output-root", type=Path, default=Path("data/canonical/generated")); parser.add_argument("--source-verifications", type=Path, default=Path("data/governance/source_verifications.json")); parser.add_argument("--standard-registry", type=Path, default=Path("data/registry/standard_registry.json")); parser.add_argument("--require-verified-sources", action="store_true") ; args = parser.parse_args()
    project_root = args.project_root.resolve()
    report = build(args.inputs, project_root, (project_root / args.output_root).resolve(), source_verifications_path=(project_root / args.source_verifications).resolve(), standard_registry_path=(project_root / args.standard_registry).resolve(), require_verified_sources=args.require_verified_sources); print(json.dumps(report, ensure_ascii=False, indent=2)); return 0 if report["publishable"] else 2


if __name__ == "__main__": raise SystemExit(main())
