"""Deterministic Chinese SQLite FTS5 BM25 baseline for canonical clause.v1."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import jieba

JIEBA_VERSION = "0.42.1"
COMMENTARY_PREFIX = "commentary"
SEARCHABLE_COMMENTARY_TYPES = {"commentary_table"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/%+\-]*|[\u4e00-\u9fff]+")
_IDENTIFIER_RE = re.compile(r"(?:GB\s*/?\s*T?|JGJ|DB)\s*[-_/A-Za-z]*\s*\d+(?:\.\d+)?\s*[-—]\s*\d{4}|\d+(?:\.\d+){1,5}|\d+(?:\.\d+)?\s*(?:%|㎡|m²|m2|m³|m3|mm|cm|km|m|kW|W|Pa|MPa|℃|°C)", re.I)


class RetrievalError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 400, retryable: bool = False, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code, self.status_code, self.retryable = code, status_code, retryable
        self.details = dict(details or {})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _standard_variants(value: Any) -> list[str]:
    raw = str(value or "").strip()
    variants = [raw]
    match = re.match(r"^(GB)_T_(\d+)_(\d{4})$", raw, re.I)
    if match:
        variants.extend([f"{match.group(1)}/T {match.group(2)}-{match.group(3)}", f"{match.group(1)}T{match.group(2)}{match.group(3)}"])
    return variants


def tokenize_chinese(text: str) -> list[str]:
    """Tokenize with the locked jieba version and retain identifiers verbatim."""
    if jieba.__version__ != JIEBA_VERSION:
        raise RuntimeError(f"jieba version must be {JIEBA_VERSION}, got {jieba.__version__}")
    source = str(text or "")
    tokens = [piece.strip() for piece in jieba.lcut(source, HMM=False) if piece.strip()]
    tokens.extend(match.group(0).replace(" ", "") for match in _IDENTIFIER_RE.finditer(source))
    return [token.lower() for token in tokens if _TOKEN_RE.fullmatch(token)]


def build_retrieval_text(node: Mapping[str, Any]) -> str:
    """Create FTS text while explicitly retaining standard/clause/table/formula IDs."""
    parts: list[str] = []
    for standard in _standard_variants(node.get("standard_id")):
        parts.append(standard)
    for key in ("standard_name", "standard_version", "clause_no", "table_id", "formula_id", "clause_title", "retrieval_text", "text"):
        value = node.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    # Markdown table/formula labels vary across source documents.  These stable
    # aliases make a query such as "表 3.2.4" or "公式 3.2.5" reproducible.
    if node.get("table_id"):
        parts.extend(["表", str(node.get("clause_no") or ""), "表" + str(node.get("clause_no") or "")])
    if node.get("formula_id"):
        parts.extend(["公式", str(node.get("clause_no") or ""), "公式" + str(node.get("clause_no") or "")])
    return " ".join(tokenize_chinese(" ".join(parts)))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_registry(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.is_file():
        return {}
    return {entry["standard_id"]: entry for entry in json.loads(path.read_text(encoding="utf-8")) if entry.get("standard_id")}


@dataclass(frozen=True)
class BuiltIndex:
    index_manifest_id: str
    manifest_path: Path
    database_path: Path


class BM25IndexBuilder:
    def __init__(self, *, canonical_path: Path, output_root: Path, source_manifest_path: Path | None = None, registry_path: Path | None = None):
        self.canonical_path = Path(canonical_path)
        self.output_root = Path(output_root)
        self.source_manifest_path = Path(source_manifest_path) if source_manifest_path else None
        self.registry_path = Path(registry_path) if registry_path else None

    def build(self) -> BuiltIndex:
        if not self.canonical_path.is_file():
            raise RetrievalError("index_build_failed", "canonical JSONL does not exist", status_code=422)
        nodes = _read_jsonl(self.canonical_path)
        canonical_sha256 = _sha256(self.canonical_path)
        included, skipped = [], []
        for node in nodes:
            content_type = str(node.get("content_type") or "")
            if not node.get("indexable") or (content_type.startswith(COMMENTARY_PREFIX) and content_type not in SEARCHABLE_COMMENTARY_TYPES):
                skipped.append(node.get("clause_id"))
                continue
            if not node.get("clause_id") or not node.get("source_file") or not node.get("source_sha256") or not node.get("pdf_page_start"):
                raise RetrievalError("index_build_failed", "indexable node is not traceable", status_code=422, details={"clause_id": node.get("clause_id")})
            included.append(node)
        if len({node["clause_id"] for node in included}) != len(included):
            raise RetrievalError("index_build_failed", "duplicate clause_id in canonical JSONL", status_code=422)

        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_id = f"bm25_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        final_dir = self.output_root / manifest_id
        temp_dir = Path(tempfile.mkdtemp(prefix=".building-", dir=self.output_root))
        db_path = temp_dir / "fts.sqlite"
        try:
            self._write_database(db_path, included)
            source_manifest_id = None
            if self.source_manifest_path and self.source_manifest_path.is_file():
                source_manifest_id = json.loads(self.source_manifest_path.read_text(encoding="utf-8")).get("source_manifest_id")
            manifest = {
                "index_manifest_id": manifest_id,
                "kind": "sqlite_fts5_bm25",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "canonical_path": str(self.canonical_path),
                "canonical_sha256": canonical_sha256,
                "source_manifest_id": source_manifest_id,
                "database_file": "fts.sqlite",
                "database_sha256": _sha256(db_path),
                "jieba_version": jieba.__version__,
                "tokenizer": {"name": "jieba", "version": jieba.__version__, "hmm": False, "identifier_aliases": ["table", "formula"]},
                "fts5": True,
                "indexed_node_count": len(included),
                "excluded_commentary_or_nonindexable_count": len(skipped),
                "registry_path": str(self.registry_path) if self.registry_path else None,
            }
            (temp_dir / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp_dir, final_dir)
            active_temp = self.output_root / f".active-{uuid.uuid4().hex}.json"
            active_temp.write_text(json.dumps({"index_manifest_id": manifest_id, "manifest_path": str(final_dir / "index_manifest.json")}, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(active_temp, self.output_root / "active.json")
            return BuiltIndex(manifest_id, final_dir / "index_manifest.json", final_dir / "fts.sqlite")
        except Exception:
            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    item.unlink()
                temp_dir.rmdir()
            raise

    @staticmethod
    def _write_database(path: Path, nodes: Iterable[Mapping[str, Any]]) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("CREATE TABLE nodes (clause_id TEXT PRIMARY KEY, parent_id TEXT, standard_id TEXT, standard_name TEXT, standard_version TEXT, clause_no TEXT, content_type TEXT NOT NULL, region TEXT, building_type TEXT, design_phase TEXT, green_target_scope TEXT, document_status TEXT, text TEXT NOT NULL, retrieval_text TEXT NOT NULL, raw_json TEXT NOT NULL)")
            connection.execute("CREATE VIRTUAL TABLE fts_nodes USING fts5(search_tokens, clause_id UNINDEXED, tokenize='unicode61')")
            for node in sorted(nodes, key=lambda item: item["clause_id"]):
                connection.execute("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    node["clause_id"], node.get("parent_id"), node.get("standard_id"), node.get("standard_name"), node.get("standard_version"), node.get("clause_no"), node["content_type"], node.get("region"), json.dumps(node.get("building_type"), ensure_ascii=False), json.dumps(node.get("design_phase"), ensure_ascii=False), json.dumps(node.get("green_target_scope"), ensure_ascii=False), node.get("document_status"), node["text"], node["retrieval_text"], json.dumps(node, ensure_ascii=False, sort_keys=True),
                ))
                connection.execute("INSERT INTO fts_nodes(search_tokens, clause_id) VALUES (?, ?)", (build_retrieval_text(node), node["clause_id"]))
            connection.commit()
        finally:
            connection.close()


class BM25Retriever:
    def __init__(self, *, manifest_path: Path, registry_path: Path | None = None):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.database_path = self.manifest_path.parent / self.manifest["database_file"]
        self.registry = _read_registry(Path(registry_path) if registry_path else None)

    @classmethod
    def from_active(cls, output_root: Path, *, registry_path: Path | None = None) -> "BM25Retriever":
        active = json.loads((Path(output_root) / "active.json").read_text(encoding="utf-8"))
        return cls(manifest_path=Path(active["manifest_path"]), registry_path=registry_path)

    def retrieve(self, request: Mapping[str, Any], *, result_limit: int | None = None, require_degraded: bool = True) -> dict[str, Any]:
        validated = self._validate_request(request, require_degraded=require_degraded)
        terms = tokenize_chinese(validated["query"])
        if not terms:
            raise RetrievalError("invalid_query", "query contains no searchable token", status_code=422)
        quoted_terms = [f'"{term.replace(chr(34), "")}"' for term in dict.fromkeys(terms)]
        match = " AND ".join(quoted_terms)
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT n.raw_json, bm25(fts_nodes) AS bm25_score FROM fts_nodes JOIN nodes n ON n.clause_id = fts_nodes.clause_id WHERE fts_nodes MATCH ? ORDER BY bm25(fts_nodes), n.clause_id", (match,)).fetchall()
            match_mode = "and"
            if not rows and len(quoted_terms) > 1:
                rows = connection.execute("SELECT n.raw_json, bm25(fts_nodes) AS bm25_score FROM fts_nodes JOIN nodes n ON n.clause_id = fts_nodes.clause_id WHERE fts_nodes MATCH ? ORDER BY bm25(fts_nodes), n.clause_id", (" OR ".join(quoted_terms),)).fetchall()
                match_mode = "or_fallback"
        finally:
            connection.close()
        hard_filtered = [json.loads(row["raw_json"]) | {"_bm25": float(row["bm25_score"])} for row in rows if self._hard_match(json.loads(row["raw_json"]), validated)]
        ranked = []
        for node in hard_filtered:
            soft_score, soft_trace = self._soft_score(node, validated["project_profile"])
            node["_soft_score"], node["_soft_trace"] = soft_score, soft_trace
            node["_structured_match"] = self._structured_match(node, validated["query"])
            ranked.append(node)
        # A matching project tag is a ranking preference, not a candidate gate.
        ranked.sort(key=lambda node: (not node["_structured_match"], node["_bm25"] - (node["_soft_score"] * 10.0), node["clause_id"]))
        limit = validated["top_k"] if result_limit is None else result_limit
        items = [self._item(node, index + 1, len(terms)) for index, node in enumerate(ranked[:limit], 1)]
        warnings = [{"code": "bm25_diagnostic_only", "message": "This is the SQLite FTS5 BM25 diagnostic baseline; dense retrieval and reranking are disabled.", "severity": "warning", "evidence_ids": []}]
        if validated["filters"]["include_commentary"]:
            warnings.append({"code": "commentary_not_indexed", "message": "Commentary is archived but intentionally absent from the Day 3 BM25 index.", "severity": "info", "evidence_ids": []})
        return {"items": items, "filters_applied": {"hard": {"standard_ids": validated["filters"]["standard_ids"], "clause_nos": validated["filters"]["clause_nos"], "region": sorted(self._profile_regions(validated["project_profile"])), "must_be_current": validated["filters"]["must_be_current"]}, "soft": ["building_type", "design_phase", "green_target_scope"], "candidate_count_before_hard_filters": len(rows), "candidate_count_after_hard_filters": len(hard_filtered), "query_token_count": len(terms), "query_match_mode": match_mode}, "warnings": warnings, "degraded_modes": ["bm25_only"]}

    @staticmethod
    def _validate_request(request: Mapping[str, Any], *, require_degraded: bool = True) -> dict[str, Any]:
        allowed = {"query", "project_profile", "filters", "top_k", "allow_degraded"}
        missing = allowed - set(request)
        if missing or set(request) - allowed:
            raise RetrievalError("invalid_request", "request fields do not match the frozen contract", details={"missing": sorted(missing), "unknown": sorted(set(request) - allowed)})
        query, top_k, filters = request["query"], request["top_k"], request["filters"]
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise RetrievalError("invalid_query", "query must be a non-empty string of at most 4000 characters", status_code=422)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise RetrievalError("invalid_request", "top_k must be an integer from 1 to 100", details={"field": "top_k"})
        if not isinstance(request["project_profile"], Mapping) or not isinstance(filters, Mapping) or not isinstance(request["allow_degraded"], bool):
            raise RetrievalError("invalid_request", "project_profile, filters, and allow_degraded have invalid types")
        filter_keys = {"standard_ids", "clause_nos", "as_of_date", "must_be_current", "include_commentary"}
        if set(filters) != filter_keys or not all(isinstance(filters[key], list) for key in ("standard_ids", "clause_nos")) or not isinstance(filters["must_be_current"], bool) or not isinstance(filters["include_commentary"], bool):
            raise RetrievalError("invalid_request", "filters do not match the frozen contract")
        as_of = filters["as_of_date"]
        if as_of is not None:
            try:
                date.fromisoformat(as_of)
            except (TypeError, ValueError) as exc:
                raise RetrievalError("invalid_request", "as_of_date must be ISO-8601 date or null", details={"field": "as_of_date"}) from exc
        if filters["must_be_current"] and not as_of:
            raise RetrievalError("missing_as_of_date", "must_be_current requires as_of_date", details={"field": "as_of_date"})
        if require_degraded and not request["allow_degraded"]:
            raise RetrievalError("degraded_not_allowed", "BM25-only diagnostic mode requires allow_degraded=true", status_code=503, retryable=True)
        return {"query": query, "project_profile": dict(request["project_profile"]), "filters": dict(filters), "top_k": top_k}

    def _hard_match(self, node: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
        filters = request["filters"]
        if filters["standard_ids"] and node.get("standard_id") not in set(filters["standard_ids"]):
            return False
        if filters["clause_nos"] and str(node.get("clause_no") or "") not in set(map(str, filters["clause_nos"])):
            return False
        regions = self._profile_regions(request["project_profile"])
        node_region = _normalise(node.get("region"))
        if regions and node_region and node_region not in {"全国", "national"} and node_region not in regions:
            return False
        if filters["must_be_current"]:
            standard = self.registry.get(node.get("standard_id"))
            # The registry explicitly forbids automatic supersession decisions
            # until the relation has been human-verified. Unknown/pending data
            # therefore remains a candidate rather than becoming a false miss.
            if standard and standard.get("relation_verification_status") == "verified" and (standard.get("status") != "current" or standard.get("effective_date") > filters["as_of_date"]):
                return False
        return True

    @staticmethod
    def _profile_regions(profile: Mapping[str, Any]) -> set[str]:
        location = profile.get("location", {}) if isinstance(profile.get("location", {}), Mapping) else {}
        return {_normalise(location.get(key)) for key in ("region", "province", "city") if _normalise(location.get(key))}

    @staticmethod
    def _structured_match(node: Mapping[str, Any], query: str) -> bool:
        """Prefer an exact table/formula node only for an explicit structured query."""
        numbers = re.findall(r"\d+(?:\.\d+){1,5}", query)
        if not numbers:
            return False
        number = numbers[0]
        if "表" in query and node.get("content_type") in {"normative_table", "commentary_table"}:
            return str(node.get("clause_no") or "") == number
        if "公式" in query and node.get("content_type") == "normative_formula":
            return str(node.get("clause_no") or "") == number
        return False

    def _soft_score(self, node: Mapping[str, Any], profile: Mapping[str, Any]) -> tuple[int, dict[str, str]]:
        building = profile.get("building", {}) if isinstance(profile.get("building", {}), Mapping) else {}
        target = profile.get("green_building_target", {}) if isinstance(profile.get("green_building_target", {}), Mapping) else {}
        expected = {"building_type": [building.get("building_category"), building.get("building_type")], "design_phase": [building.get("design_phase")], "green_target_scope": list(target.values())}
        score, trace = 0, {}
        for field, wanted in expected.items():
            expected_values = {_normalise(value) for value in wanted if _normalise(value)}
            actual_values = {_normalise(value) for value in _as_list(node.get(field)) if _normalise(value)}
            if not expected_values:
                trace[field] = "not_requested"
            elif not actual_values:
                trace[field] = "unknown_kept"
            elif actual_values & expected_values:
                score, trace[field] = score + 1, "matched"
            else:
                trace[field] = "not_matched_kept"
        return score, trace

    @staticmethod
    def _item(node: Mapping[str, Any], rank: int, token_count: int) -> dict[str, Any]:
        missing = []
        if node.get("requires_project_facts"):
            missing.append({"fact": "project_facts", "reason": "clause requires project-side facts", "required_for": [node["clause_id"]]})
        return {"evidence_id": node["clause_id"], "clause_id": node["clause_id"], "parent_id": node.get("parent_id"), "content_type": node["content_type"], "text": node["text"], "parent_context": [], "standard_id": node.get("standard_id"), "standard_name": node.get("standard_name"), "standard_version": node.get("standard_version"), "clause_no": node.get("clause_no"), "pdf_page_start": node.get("pdf_page_start"), "pdf_page_end": node.get("pdf_page_end"), "printed_page_start": node.get("printed_page_start"), "printed_page_end": node.get("printed_page_end"), "source_file": node.get("source_file"), "source_sha256": node.get("source_sha256"), "verification_status": node["verification_status"], "missing_facts": missing, "bm25_rank": rank, "dense_rank": None, "rrf_score": None, "rerank_score": None, "table_id": node.get("table_id"), "formula_id": node.get("formula_id"), "trace": {"bm25_score": node["_bm25"], "soft_filter_score": node["_soft_score"], "soft_filter": node["_soft_trace"], "structured_identifier_match": node["_structured_match"], "query_token_count": token_count}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an immutable SQLite FTS5 BM25 index from canonical clause.v1 JSONL.")
    parser.add_argument("canonical", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/indexes/bm25"))
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("data/registry/standard_registry.json"))
    args = parser.parse_args()
    built = BM25IndexBuilder(canonical_path=args.canonical, output_root=args.output_root, source_manifest_path=args.source_manifest, registry_path=args.registry).build()
    print(json.dumps({"index_manifest_id": built.index_manifest_id, "manifest_path": str(built.manifest_path), "database_path": str(built.database_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
