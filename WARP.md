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
- **EmbedderFactory**: Provider system (`embedders/factory.py`)
- **HybridSearchEngine**: Vector + keyword search (`hybrid_search.py`)
- **RelationshipStore**: Knowledge graph without external DB (`relationship_store.py`)
- **Content Chunkers**: Code/document/web-aware processing (`chunker.py`)

### Pipeline: Load → Chunk → Embed → Store → Search

### Search Methods
- **Vector**: FAISS IndexFlatIP (cosine similarity)
- **Keyword**: TF-IDF + BM25 with configurable weighting
- **Reranking**: Cross-encoder neural reranking

## MCP Interface

6 essential functions:
1. `index_code(path)` - Codebase with relationship extraction
2. `index_document(path)` - Documents (PDF, MD, TXT)
3. `index_webpage(url, recursive=False, max_depth=1)` - Web content with optional recursion
4. `search(query, top_k=5)` - Hybrid semantic + keyword search
5. `search_with_context(query, top_k=5, include_related=False)` - With relationships
6. `status()` - System statistics

### Search Result JSON Outline

**Current Structure** (Multiple Shapes - Inconsistent)  
Both `search` and `search_with_context` return a list of result objects (JSON). The exact shape depends on enabled features.

- Always-present fields (across all modes):
  - `score: float` – primary relevance score
  - `source_path: string` – file path or URL of the source
  - `source_type: string` – one of `codebase`, `document`, `webpage`
  - `chunk_text: string` – the matched content chunk
  - `chunk_id: string` – unique chunk identifier

- Standard vector search (default):
  - Adds: `start_char: int`, `end_char: int`, `created_at: ISO-8601 string`

- Hybrid search (vector + keyword), when enabled:
  - Adds: `vector_score: float`, `keyword_score: float`
  - Adds: `metadata: object` with keys such as `source_type`, `source_path`, `chunk_id`, `created_at`, and optionally `file_extension`, `embedding_provider`, `embedding_model`

- Reranking (cross-encoder), when enabled:
  - Adds: `original_score: float`, `rerank_score: float` (and `score` becomes the final combined score)

- Context enrichment via `search_with_context(..., include_related=True)` (if relationships are enabled):
  - Adds: `relationships: object` (e.g., `references`, `imports`, `entities`)
  - Adds: `related_chunks: array` of up to 3 objects: `{ source_path, chunk_text (truncated), relationship_type }`

- Error behavior: Both functions return `[]` (empty list) on error or no results.

**🚀 Proposed Improved Structure** (Based on analysis of rust-local-rag + mcp-crawl4ai-rag)  
Consistent response envelope with standardized result objects:

```json
{
  "success": true,
  "query": "user search query",
  "search_config": {
    "top_k": 5,
    "hybrid_search": true,
    "reranking": true,
    "relationships": false
  },
  "performance": {
    "search_time_ms": 45,
    "total_candidates": 50
  },
  "results": [{
    "chunk_id": "chunk_12345",
    "source_path": "/path/to/file.py", 
    "source_type": "codebase",
    "text": "def main():\n    print('Hello')",
    "relevance_score": 0.847,
    "position": {
      "start_char": 150,
      "end_char": 280
    },
    "scores": {
      "vector": 0.823,
      "keyword": 0.654,
      "rerank": 0.891,
      "combined": 0.847
    },
    "metadata": {
      "created_at": "2025-09-21T18:07:32Z",
      "embedding_model": "all-MiniLM-L6-v2",
      "word_count": 45
    },
    "context": {
      "relationships": {...},
      "related_chunks": [...]
    }
  }],
  "total_results": 5
}
```

**Key Improvements:**  
- Consistent response wrapper with query metadata
- Unified `relevance_score` field (instead of multiple score names)
- Structured `scores` object for detailed score breakdown
- Structured `position` object for location info
- Consistent `text` field name (not `chunk_text`)
- Performance metrics and search configuration details
- Optional `context` object for relationships (cleaner nesting)

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

**Relationship types**: function_call, import, reference, link, hierarchy, tag  
**Access**: `search_with_context(query, include_related=True)`

## Testing

**230+ tests, 71% coverage**
- Core components: 87-91% coverage
- Run: `uv run pytest --cov=pycontextify`
- MCP test runner: `uv run python scripts/run_mcp_tests.py`
- Quick smoke test: `uv run python scripts/run_mcp_tests.py --smoke`
- Consolidated test files: `test_*_consolidated.py`
- Main test files: `test_mcp_simple.py` (8 tests), `test_integration.py` (4 tests)

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
**Simplified MCP**: 6 essential functions, relationships via enhanced search  
**No External DB**: File-based persistence, lightweight knowledge graph
