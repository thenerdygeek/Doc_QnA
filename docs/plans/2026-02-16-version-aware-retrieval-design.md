# Version-Aware Retrieval Design

## Problem

2908+ unorganized doc files with no naming convention. Multiple versions of the same API/topic exist as separate files. The system returns chunks from all versions indiscriminately, causing the LLM to mix outdated and current information.

## Decision: Retrieval-Time Dedup (Approach A)

### 1. Schema — Add `doc_date` field

- Type: `pa.float64()` — Unix timestamp (seconds since epoch)
- Sentinel: `0.0` means "unknown date, treat as oldest"
- Migration: existing tables get column added with default `0.0`; no full re-index needed

### 2. Date Extraction — `doc_qa/parsers/date_extractor.py`

Single function: `extract_doc_date(file_path: str) -> float`

Priority chain:
1. **PDF**: `pdfplumber` metadata → `ModDate` or `CreationDate` (format `D:YYYYMMDDHHmmSS`)
2. **Markdown**: YAML frontmatter → `date` key
3. **AsciiDoc**: `:revdate:` or `:date:` attribute in first 50 lines
4. **Fallback**: `os.path.getmtime()` — always available

### 3. Indexing Integration

- `_parse_and_chunk()` calls `extract_doc_date()` once per file, returns in result dict
- `_bulk_add_chunks()` passes `doc_dates: dict[str, float]` to `add_chunks()`
- `add_chunks()` stores `doc_date` per record
- `_copy_unchanged_chunks()` — column travels with Arrow rows, no special handling

### 4. Retrieval-Time Near-Duplicate Dedup

New function: `deduplicate_near_duplicates(chunks, threshold=0.95) -> list[RetrievedChunk]`

Location: `doc_qa/retrieval/dedup.py`

Algorithm:
1. Extract vectors from candidate chunks (already loaded by retriever)
2. Build cosine similarity matrix (N×N where N ≤ candidate_pool, typically 20)
3. For each pair with similarity > 0.95 from different files:
   - Keep the chunk with higher `doc_date`
   - Mark the other for removal
4. Return filtered list preserving original order

Runs AFTER retrieval, BEFORE reranking in `QueryPipeline._process_single()`.

Why 0.95: high enough to catch true duplicates (same paragraph, minor formatting differences) without false positives on topically similar but distinct content.

### 5. Prompt Changes

`_build_context()` adds date to source headers:
```
[Source 1: api.pdf (2025-03-15) > Authentication] (score: 0.854)
```

`SYSTEM_PROMPT` gets additional instruction:
```
When sources contain conflicting information, prefer the most recently dated source.
```

### 6. `RetrievedChunk` — Add `doc_date` field

```python
@dataclass
class RetrievedChunk:
    ...
    doc_date: float = 0.0  # Unix timestamp
```

Populated from Arrow results in `_arrow_to_chunks()`.

### 7. Files Modified

| File | Change |
|------|--------|
| `parsers/date_extractor.py` | NEW — date extraction from PDF/MD/AsciiDoc/mtime |
| `indexing/indexer.py` | Schema: add `doc_date`; migration for existing tables; `add_chunks` accepts dates |
| `indexing/job.py` | `_parse_and_chunk` extracts date; passes through pipeline |
| `retrieval/dedup.py` | NEW — near-duplicate dedup function |
| `retrieval/retriever.py` | `RetrievedChunk.doc_date`; populate in `_arrow_to_chunks` |
| `retrieval/query_pipeline.py` | Insert dedup step; add date to `_build_context` |
| `llm/prompt_templates.py` | System prompt: prefer recent sources |
| `config.py` | `dedup_threshold: float = 0.95` on RetrievalConfig |
| `tests/test_date_extractor.py` | NEW — unit tests |
| `tests/test_dedup.py` | NEW — unit tests |
