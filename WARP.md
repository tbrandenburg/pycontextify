# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

PyContextify is a Python-based MCP (Model Context Protocol) server for semantic search over codebases, documents, and webpages. It provides semantic search capabilities using FAISS vector similarity search with multiple embedding providers and lightweight knowledge graph functionality.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- UV package manager

### Initial Setup
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# For development with additional tools
uv sync --extra dev

# Set up environment variables
cp .env.example .env
# Edit .env with your preferred settings
```

## Essential Commands

### Running the MCP Server
```bash
# Production mode
uv run pycontextify

# Direct FastMCP execution
uv run fastmcp run pycontextify/mcp_server.py

# Development mode with tools
uv run --extra dev fastmcp run pycontextify/mcp_server.py
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test files
uv run pytest tests/test_basic.py
uv run pytest tests/test_config.py

# Run smoke tests (minimal dependencies)
python3 tests/smoke/test_mcp_server.py
python3 tests/smoke/test_mcp_functionality.py

# Run smoke tests with pytest if available
python3 -m pytest tests/smoke/ -v

# Simple test runner (no pytest required)
python3 tests/smoke/run_smoke_tests.py
```

### Code Quality
```bash
# Format code
uv run black .

# Sort imports
uv run isort .

# Lint code
uv run flake8

# Type checking
uv run mypy pycontextify
```

### Dependency Management
```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add optional dependency
uv add --optional embedding-provider package-name

# Recreate environment
uv sync --reinstall

# Clear cache
uv cache clean
```

## Core Architecture

PyContextify follows a modular architecture with clear separation of concerns:

### Main Components

1. **IndexManager** (`pycontextify/index/manager.py`): Central orchestrator handling all operations
2. **VectorStore** (`pycontextify/index/vector_store.py`): FAISS wrapper with persistence and backup
3. **EmbedderFactory** (`pycontextify/index/embedders/factory.py`): Extensible embedding provider system
4. **Chunkers** (`pycontextify/index/chunker.py`): Content-specific processing (Code, Document, WebPage)
5. **RelationshipStore** (`pycontextify/index/relationship_store.py`): Lightweight knowledge graph storage
6. **MetadataStore** (`pycontextify/index/metadata.py`): Chunk metadata with relationship tracking

### Content Processing Pipeline

1. **Load** → Content loaded by specialized loaders (CodeLoader, DocumentLoader, WebpageLoader)
2. **Chunk** → Content processed by appropriate chunker based on source type
3. **Embed** → Chunks converted to vectors using selected embedding provider
4. **Store** → Vectors stored in FAISS, metadata in MetadataStore
5. **Extract Relationships** → Entity relationships extracted and stored
6. **Auto-Persist** → All components automatically saved after successful indexing

### Embedding Provider Architecture

The system supports multiple embedding providers through an extensible factory pattern:

- **Current**: Sentence Transformers (default)
- **Planned**: Ollama, OpenAI
- **Location**: `pycontextify/index/embedders/`
- **Interface**: All providers implement `BaseEmbedder`

To switch providers, modify the `PYCONTEXTIFY_EMBEDDING_PROVIDER` environment variable.

### Specialized Content Processing

- **CodeChunker**: Function/class boundary awareness, symbol extraction, import relationships
- **DocumentChunker**: Section hierarchy, citation extraction, cross-references  
- **WebPageChunker**: HTML structure awareness, link extraction, navigation hierarchy, content filtering

## MCP Server Interface

The server exposes 6 essential MCP functions:

1. **`index_code(path)`** - Index a codebase directory with relationship extraction
2. **`index_document(path)`** - Index individual documents (PDF, Markdown, text)
3. **`index_webpage(url, recursive=False, max_depth=1)`** - Index web content with HTML-aware processing
4. **`search(query, top_k=5)`** - Basic semantic search across all indexed content
5. **`search_with_context(query, top_k=5, include_related=False)`** - Enhanced search with relationship context
6. **`status()`** - System status and comprehensive statistics

### Key Implementation Notes

- All indexing operations trigger immediate auto-persistence if enabled
- The server maintains a global `IndexManager` instance shared across tool calls
- Error handling includes detailed validation and meaningful error messages
- Graceful shutdown handler saves index on server termination

## Configuration System

Configuration is managed through environment variables in `.env` file:

### Critical Settings
- `PYCONTEXTIFY_EMBEDDING_PROVIDER`: Embedding provider selection
- `PYCONTEXTIFY_EMBEDDING_MODEL`: Specific model to use
- `PYCONTEXTIFY_AUTO_PERSIST`: Enable automatic saving after indexing
- `PYCONTEXTIFY_INDEX_DIR`: Directory for storing index files
- `PYCONTEXTIFY_ENABLE_RELATIONSHIPS`: Enable knowledge graph features

### Chunking Parameters
- `PYCONTEXTIFY_CHUNK_SIZE`: Text chunk size in tokens (default: 512)
- `PYCONTEXTIFY_CHUNK_OVERLAP`: Overlap between chunks (default: 50)

### Performance Tuning
- `PYCONTEXTIFY_MAX_FILE_SIZE_MB`: Skip files larger than this
- `PYCONTEXTIFY_BATCH_SIZE`: Embedding batch size
- `PYCONTEXTIFY_CRAWL_DELAY_SECONDS`: Delay between web requests

## Knowledge Graph Features

PyContextify implements lightweight knowledge graph capabilities without requiring external graph databases:

### Relationship Types
- `function_call`: Code function calls
- `import`: Code imports and dependencies  
- `reference`: Document cross-references
- `link`: Web page links
- `hierarchy`: Parent-child relationships
- `tag`: Tag-based relationships

### Accessing Relationships
- Use `search_with_context(query, include_related=True)` for relationship-aware search
- Relationships are automatically extracted during indexing
- No separate MCP functions for relationships (simplified interface)

## Testing Strategy

### Unit Tests
- **Location**: `tests/`
- **Coverage**: Configuration, metadata, embeddings, relationships
- **Run with**: `uv run pytest`

### Smoke Tests
- **Location**: `tests/smoke/`
- **Purpose**: Verify system health without heavy dependencies
- **Benefits**: Quick CI/CD checks, test MCP server initialization
- **Run with**: `python3 tests/smoke/test_mcp_server.py`

## Supported File Types

### Code Files
- Python (`.py`), JavaScript/TypeScript (`.js`, `.ts`)
- Java (`.java`), C/C++ (`.c`, `.cpp`, `.h`)
- Rust (`.rs`), Go (`.go`), PHP (`.php`)
- Ruby (`.rb`), Swift (`.swift`), Kotlin (`.kt`)

### Documents
- PDF files (`.pdf`) - uses PyPDF2 + pdfplumber fallback
- Markdown (`.md`) - preserves structure
- Text files (`.txt`) - direct UTF-8 processing

### Web Content
- HTML pages with optional recursive crawling
- Automatic content filtering and structure extraction
- Respects robots.txt and includes rate limiting

## Performance Considerations

### Memory Usage
- Scales with corpus size and embedding model selection
- FAISS IndexFlatIP used for exact search (good for MVP)
- Auto-persistence has minimal overhead with compression

### Model Selection Impact
- `all-MiniLM-L6-v2`: Fast, 384 dimensions, lower accuracy
- `all-mpnet-base-v2`: Higher quality, 768 dimensions, slower

### Optimization Areas
- Consider IndexIVFFlat for larger datasets (>100k chunks)
- Batch processing optimizations for large file sets
- GPU support available via `faiss-gpu` optional dependency

## Troubleshooting

### Common Issues

**Model Loading Errors**: Ensure sufficient memory and stable internet for first-time model download

**Memory Issues**: Use smaller embedding models (`all-MiniLM-L6-v2`) or reduce batch sizes

**Import Errors**: Verify UV virtual environment with `uv sync --reinstall`

**Permission Issues**: Check index directory write permissions

### UV-Specific Issues
```bash
# Recreate environment completely
uv sync --reinstall

# Check dependency resolution
uv lock --check

# Clear all caches
uv cache clean
```

### Debugging
- Enable debug logging by modifying logging level in `mcp_server.py`
- Check index directory for persistence files (`.faiss`, `.pkl`)
- Use smoke tests to verify system health without full dependencies

## Architecture Decisions

### Why UV?
- Fast dependency resolution and installation
- Built-in virtual environment management
- Lockfile support for reproducible builds
- Optional dependencies for different embedding providers

### Why FAISS?
- Excellent performance for vector similarity search
- Supports both CPU and GPU execution
- Mature library with extensive documentation
- Easy persistence and loading

### Why Simplified MCP Interface?
- Focus on essential functionality
- Easier for users to understand and integrate
- Relationships accessible through enhanced search
- Reduced complexity without losing capabilities

### Why Lightweight Knowledge Graph?
- No external database dependencies
- Fast relationship extraction and storage
- Suitable for most use cases without full graph database overhead
- Easy to extend to full graph database later if needed