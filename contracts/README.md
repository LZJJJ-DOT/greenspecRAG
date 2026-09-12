# Frozen contracts

All schemas use JSON Schema Draft 2020-12. A schema failure, duplicate ID, or untraceable indexable node is a hard build failure. Warnings such as missing printed pages, unverified OCR, and pending version relations are returned structurally and do not become authoritative evidence.

The OpenAPI contract uses these error codes in error.code: invalid_request, invalid_query, missing_as_of_date, dependency_unavailable, degraded_not_allowed, build_conflict, build_not_found, and index_build_failed. Every error has message, request_id, retryable, and an object details field.

A successful retrieval response always includes request_id, retrieval_run_id, index_manifest_id, items, filters_applied, warnings, and degraded_modes.
