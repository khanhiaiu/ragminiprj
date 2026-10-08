# Production RAG ingestion implementation plan

Audit date: 2026-10-08

## Repository facts

- Parsing is already implemented for PDF/image OCR, DOCX and XLSX and emits
  `CanonicalDocument`. The ingestion work must consume these files and must not
  alter the parser outputs.
- `parsed_test_document/all_documents_gpu` currently contains 28 canonical
  documents: 5 openpyxl, 4 docling and 19 PaddleOCR documents.
- Actual canonical element counts are 2,154 paragraphs, 330 headings, 61 lists,
  88 tables, 89 images, 125 headers, 226 footers and 10 unknown elements.
- All 89 image `asset_path` references resolve. There are 273 asset files when
  rendered source pages are included.
- The existing retrieval preprocessor is non-mutating and preserves basic legal
  heading context, but it intentionally drops images and most provenance.
- The existing chunker is explicitly a character-count fallback. It can emit
  image placeholders and does not enforce tokenizer limits or preserve complete
  source spans.
- The existing BGE-M3 adapter uses sentence-transformers dense vectors only.
  Sparse lexical vectors, revision pinning, validation and caching are absent.
- The existing Qdrant store has one named dense vector. `auto` silently falls
  back to a local store, has no generation collection/alias publication flow,
  and does not run sparse or hybrid verification.
- The ingestion CLI writes a single mutable output directory and a separate
  BM25 index. Captioning, prepare-only mode, checkpoints and versioned run
  manifests are absent.

## Implementation order

1. Add a pinned BGE-M3 tokenizer service and exact 200-token image context
   extraction with deterministic provenance, hashing, diagnostics and tests.
2. Add direct Gemini Developer API preflight, egress guard, structured caption
   schema/validation, cache, retry, checkpoint/resume and pilot reports.
3. Extend the derived retrieval representation and add tokenizer-aware text,
   table and figure chunkers without changing canonical JSON.
4. Replace the production embedding path with a single-load FlagEmbedding
   dense+sparse adapter, streaming validation and content-addressed cache.
5. Add Qdrant generation collections with dense+sparse vectors, deterministic
   points, payload indexes, verification and atomic `docs_current` publication.
6. Wire caption-only, prepare-only, full-ingest, resume and dry-run commands;
   run deterministic tests and document opt-in external checks separately.
