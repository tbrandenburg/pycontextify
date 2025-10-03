# WARP.md

Development guidance for PyContextify - a Python MCP server for semantic search with lightweight knowledge graph capabilities.

## Setup

```bash
# Install UV and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

## Essential Commands

```bash
# Run server
uv run pycontextify --verbose

# Run tests with coverage
uv run pytest --cov=pycontextify

# Code quality
uv run black . && uv run isort . && uv run flake8
uv run mypy pycontextify

# Dependency management
uv add package-name              # Runtime
uv add --dev package-name        # Development
uv sync --reinstall              # Reset environment
```

## Architecture

### Key Components
- **IndexManager**: Central orchestrator with lazy loading (`manager.py`)
- **VectorStore**: FAISS wrapper with persistence (`vector_store.py`)
- **EmbedderFactory**: Provider system (`embedders/factory.py`) — currently ships with the sentence-transformers implementation and validation stubs for future providers
- **HybridSearchEngine**: Vector + keyword search (`hybrid_search.py`)
- **Content Chunkers**: Code/document/web-aware processing (`chunker.py`)
- **Relationship Extraction**: Lightweight knowledge graph integrated into `chunker.py` and `models.py`

### Pipeline: Load → Chunk → Embed → Store → Search

## Chunking Strategies by Resource Type

- **Code files** → `CodeChunker`
  - Detects natural boundaries using function/class signatures and visibility keywords before falling back to token windows, ensuring structural cohesion for languages such as Python, JS/TS, Java, C-family, and Go.【F:pycontextify/index/chunker.py†L208-L315】
  - Captures relationships like function/class names, imports, and simple variable declarations to enrich the lightweight knowledge graph for downstream search.【F:pycontextify/index/chunker.py†L317-L359】

- **Documents (Markdown/Text/PDF)** → `DocumentChunker`
  - Breaks content along Markdown-style headers while enforcing minimum section length, with token-based fallback for unstructured prose.【F:pycontextify/index/chunker.py†L361-L475】
  - Extracts contextual relationships from links, citations, emphasized terms, and section titles; PDF files are first converted to text via `DocumentLoader`/`PDFLoader` and then treated like other documents.【F:pycontextify/index/chunker.py†L477-L525】【F:pycontextify/index/loaders.py†L178-L262】

- **Web pages** → `WebPageChunker`
  - Reuses the document strategy after HTML cleanup, augmenting each chunk with metadata such as external links, domain, and URL path segments.【F:pycontextify/index/chunker.py†L528-L613】
  - Adds web-specific relationship tags (domains, external links, contacts) on top of document-level extraction for richer search pivots.【F:pycontextify/index/chunker.py†L646-L683】

- **Fallback/Unknown sources** → `SimpleChunker`
  - Applies configurable token windows with overlap and basic capitalized-entity extraction when the source type is not recognized or lacks structure.【F:pycontextify/index/chunker.py†L168-L205】

`IndexManager` selects the appropriate strategy at runtime through `ChunkerFactory`, and all chunkers honor the shared configuration for chunk size, overlap, and relationship extraction limits defined in `Config`.【F:pycontextify/index/manager.py†L571-L602】【F:pycontextify/index/chunker.py†L685-L707】【F:pycontextify/index/config.py†L85-L95】

### Search Methods
- **Vector**: FAISS IndexFlatIP (cosine similarity)
- **Keyword**: TF-IDF + BM25 with configurable weighting
- **Reranking**: Not currently implemented (placeholders remain in models for future cross-encoder support)

## MCP Interface

6 essential functions:
1. `index_code(path)` - Codebase with relationship extraction
2. `index_document(path)` - Documents (PDF, MD, TXT)
3. `index_webpage(url, recursive=False, max_depth=1)` - Web content with optional recursion
4. `search(query, top_k=5)` - Hybrid semantic + keyword search
5. `reset_index(remove_files=True, confirm=False)` - Clear index data
6. `status()` - System statistics

### Search Result JSON Outline

**IndexManager API (`IndexManager.search`)**
- Returns a `SearchResponse` object with a consistent envelope: `success`, `query`, `search_config`, `results`, `total_results`, and optional `performance` / `query_analysis` data. Each `SearchResult` includes `text`, `relevance_score`, a structured `scores` dictionary (vector/keyword/combined), optional `position`, `metadata`, and provenance fields.
- Calling `SearchResponse.to_dict()` or using `display_format="structured"` yields a deterministic JSON object—ideal for integrations that use the Python package directly.

**MCP `search` tool (default `display_format="structured"`)**
- Returns a simplified list of dictionaries for tool consumers. Each item includes:
  - `chunk_id`
  - `chunk_text`
  - `similarity_score`
  - `source_path`
  - `source_type`
  - `metadata` (structured when available)
  - `scores` (keys such as `vector`, `keyword`, `combined`, depending on search mode)
- On error (validation failure, empty index, etc.) the structured format falls back to an empty list. Readable/summary formats return formatted strings generated by `SearchResponse.format_for_display()`.

**Notes**
- Hybrid search enriches `scores` with both semantic and keyword contributions. Reranking support was removed to simplify the code paths, so only vector and keyword signals remain.
- Relationship context is not injected into search results yet; see “Knowledge Graph” below for access patterns.

## Configuration

**Priority**: CLI args > Environment variables > Defaults

### Key Settings
- `PYCONTEXTIFY_EMBEDDING_PROVIDER`: sentence_transformers (default)
- `PYCONTEXTIFY_EMBEDDING_MODEL`: all-MiniLM-L6-v2 (default)
- `PYCONTEXTIFY_AUTO_PERSIST`: true (default)
- `PYCONTEXTIFY_INDEX_DIR`: ./index_data (default)
- `PYCONTEXTIFY_CHUNK_SIZE`: 512 (default)

## Knowledge Graph

**Lightweight approach** - no external database required

**Relationship signals extracted today**
- **Tags**: `import`, markdown section names, `citation`, domains, URL path fragments, `external_link`, `contact`
- **References**: Proper nouns, markdown links, citations, URLs, email addresses, detected code symbols
- **Code symbols**: Function, class, and variable names identified by the chunkers

**Access**
- Relationship data lives on each `ChunkMetadata` instance (`tags`, `references`, `code_symbols`). It is not yet surfaced automatically in MCP search responses.
- Retrieve relationship details via `IndexManager.metadata_store` or by extending the search result formatting to include `ChunkMetadata.get_relationships()`.

## Testing

**247 tests, 69% coverage - 100% pass rate**
- Core components: 85-87% coverage
- Run: `uv run pytest --cov=pycontextify`
- MCP test runner: `uv run python scripts/run_mcp_tests.py`
- Quick smoke test: `uv run python scripts/run_mcp_tests.py --smoke`
- Consolidated test files: 15 test files
- All persistence tests passing after auto-load fixes

## File Support

**Code**: Python, JS/TS, Java, C/C++, Rust, Go, etc.  
**Documents**: PDF (PyPDF2/pdfplumber), Markdown, Text  
**Web**: HTML with recursive crawling, content filtering

## Performance

**Models**:
- `all-MiniLM-L6-v2`: Fast, 384D
- `all-mpnet-base-v2`: Better quality, 768D

**Scaling**: Memory scales with corpus size. Use IndexIVFFlat for >100k chunks.

## Performance Scripts

**Available utilities**:
- `python scripts/measure_startup_time.py` - Startup performance measurement
- `python scripts/debug_lazy_loading.py` - Lazy loading verification  
- `python scripts/detailed_perf.py` - Component performance analysis
- `python scripts/fast_startup_test.py` - Startup optimization testing
- `python scripts/test_hf_connectivity.py` - HuggingFace connectivity test

## Troubleshooting

**Model loading**: Ensure internet for first download  
**Memory issues**: Use `all-MiniLM-L6-v2` model  
**Dependencies**: `uv sync --reinstall`  
**Permissions**: Check index directory write access  
**Debug**: Use `--verbose` flag for detailed logging
**Windows**: Set `$env:PYTHONPATH = "."` before running scripts

## Design Decisions

**UV**: Fast dependency management, lockfiles, optional dependencies  
**FAISS**: High-performance vector search with CPU/GPU support  
**Lazy Loading**: Fast startup, components loaded on demand  
**Simplified MCP**: 6 essential functions, clean and focused API
**No External DB**: File-based persistence, lightweight knowledge graph
