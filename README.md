# PyContextify ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-247_passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-69%25-green)

**One-line:** Semantic search server with relationship-aware discovery across codebases and documents.

PyContextify is a Python-based MCP (Model Context Protocol) server that provides intelligent semantic search capabilities over diverse knowledge sources. It combines vector similarity search with basic relationship tracking to help developers, researchers, and technical writers discover contextually relevant information across codebases and documentation.

**Main Features:**
- 🔍 **Semantic Search**: Vector similarity with FAISS + hybrid keyword search
- 📚 **Multi-Source**: Index code and documents (PDF/MD/TXT)
- 🧠 **Smart Chunking**: Content-aware processing (code boundaries, document hierarchy)
- ⚡ **Pre-loaded Models**: Embedders initialize at startup for fast first requests
- 🔗 **Relationship Tracking**: Basic relationship extraction (tags, references, code symbols)
- 🛠️ **MCP Protocol**: 5 essential functions for seamless AI assistant integration

---

## Quickstart

```bash
# Install UV and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run MCP server
uv run pycontextify --verbose
```

### Run with `uvx`

If you want to execute the published CLI without modifying your current
environment, `uvx` can resolve `pycontextify` from PyPI and run its console
entry point directly:

```bash
uvx pycontextify -- --help
```

The double dash (`--`) ensures any following arguments are forwarded to
PyContextify itself. This requires a released version of the package to be
available on PyPI, which the manual publishing workflow now provides.

## System Requirements

- **Python**: Python 3.10 or newer for the MCP server core (full test suite currently targets Python 3.13+).【F:pyproject.toml†L1-L35】【F:tests/README.md†L57-L63】
- **Package management**: Ability to install dependencies via [UV](https://docs.astral.sh/uv/) and resolve all runtime libraries, including FAISS, sentence-transformers, PDF processors, and supporting utilities.【F:README.md†L18-L55】【F:pyproject.toml†L21-L35】
- **CPU**: 64-bit multi-core processor (4+ cores recommended) so FAISS vector search and sentence-transformers embedding generation can run locally without bottlenecks.【F:pyproject.toml†L21-L35】【F:WARP.md†L122-L150】
- **Memory**: 8 GB RAM minimum (16 GB recommended for larger corpora) because embeddings and FAISS indexes reside in-process and scale with corpus size; switch to the lighter `all-MiniLM-L6-v2` model if constrained.【F:WARP.md†L122-L141】
- **Network access**: Internet connectivity on first run to download sentence-transformers models and other remote assets.【F:WARP.md†L139-L150】
- **Storage & filesystem**: At least 5 GB of free disk space to install Python dependencies, download embedding models, and persist FAISS indexes in `PYCONTEXTIFY_INDEX_DIR`, along with write access for temporary working folders during indexing and testing.【F:pyproject.toml†L21-L35】【F:tests/README.md†L57-L66】【F:README.md†L130-L162】
- **Optional acceleration**: CUDA-capable GPU support is available by installing the optional `gpu` dependency group (`faiss-gpu`) alongside the default CPU build.【F:pyproject.toml†L37-L56】【F:WARP.md†L148-L150】

## Table of Contents
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
- [Chunking Techniques](#chunking-techniques)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Tests & CI](#tests--ci)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)
- [Security](#security)
- [Maintainers](#maintainers)

## Installation

### PyPI (recommended)

```bash
pip install pycontextify
```

Extras are available for specific workflows:

- `pip install "pycontextify[dev]"` – testing, linting, and packaging helpers
- `pip install "pycontextify[nlp]"` – optional spaCy language model
- `pip install "pycontextify[ollama]"` / `[openai]` – alternative embedding providers

### From Source with UV

**Requirements:** Python 3.10+ and the [UV](https://github.com/astral-sh/uv) package manager

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone https://github.com/pycontextify/pycontextify.git
cd pycontextify
uv sync

# Optional: Install development + release tooling
uv sync --extra dev
```

To update dependencies from scratch use `uv sync --reinstall`.

## Usage

Minimal example:

```bash
# Start the MCP server
uv run pycontextify

# Index content (via MCP client/AI assistant)
# The server exposes 5 MCP functions:
# - index_filebase(path, topic) - Unified indexing for code & docs
# - discover() - List indexed topics
# - search(query, top_k=5) - Semantic search
# - reset_index(remove_files=True, confirm=False) - Clear index data
# - status() - Get system status and statistics
```

Expected output:

```
Starting PyContextify MCP Server...
Server provides 5 essential MCP functions:
  - index_filebase(path, topic): Unified filebase indexing
  - discover(): List indexed topics
  - search(query, top_k): Basic semantic search
  - reset_index(confirm=True): Clear all indexed content
  - status(): Get system status and statistics
MCP server ready and listening for requests...
```

## Chunking Techniques

PyContextify employs a **hierarchical chunking system** with specialized processors for different content types, optimizing semantic search while preserving structural integrity.

### Content-Aware Chunking Strategies

#### **Code Chunking** (`CodeChunker`)
- **Primary Strategy**: Structure-aware splitting by function/class boundaries
- **Language Support**: Python, JavaScript, TypeScript, Java, C/C++, Rust, Go, and more
- **Boundary Detection**: `def`, `class`, `function`, `const`, `var`, `let`, `public`, `private`, `protected`
- **Relationship Extraction**: Functions, classes, imports, variable assignments
- **Fallback**: Token-based splitting when code blocks exceed size limits

#### **Document Chunking** (`DocumentChunker`)
- **Primary Strategy**: Markdown header hierarchy preservation (`#`, `##`, `###`)
- **Section Tracking**: Maintains parent-section relationships for context
- **Content Filtering**: Requires minimum 50 characters per meaningful chunk
- **Relationship Extraction**: Links `[text](url)`, citations `[1]`, `(Smith 2020)`, emphasized terms
- **Fallback**: Token-based splitting when no structure is detected

#### **Simple Chunking** (`SimpleChunker`)
- **Fallback Strategy**: Pure token-based chunking for unstructured content
- **Basic Relationships**: Capitalized word extraction for entity hints
- **Universal Compatibility**: Handles any text format as last resort

### Technical Configuration

```python
chunk_size: int = 512        # Target tokens per chunk (configurable)
chunk_overlap: int = 64      # Overlap between adjacent chunks  
enable_relationships: bool   # Extract lightweight knowledge graph data
max_relationships_per_chunk: int  # Limit relationships to avoid noise
```

### Key Features

- **Smart Selection**: Automatic chunker selection via `ChunkerFactory` based on content type
- **Token Estimation**: `words × 1.3` heuristic for English text with automatic oversized chunk splitting
- **Position Tracking**: Maintains precise character start/end positions for all chunks
- **Metadata Preservation**: Source path, embedding info, creation timestamps, and custom metadata
- **Relationship Graph**: Lightweight knowledge extraction (imports, references, citations, links)

**Bottom Line**: PyContextify's chunking system intelligently adapts to content structure—respecting code boundaries and document hierarchy—while maintaining configurable token limits and extracting contextual relationships for enhanced semantic search.

## Configuration

Required environment variables / config:
- `PYCONTEXTIFY_EMBEDDING_MODEL` — string — default: `all-MiniLM-L6-v2` — Embedding model for semantic search
- `PYCONTEXTIFY_EMBEDDING_PROVIDER` — string — default: `sentence_transformers` — Embedding provider (sentence_transformers, ollama, openai)
- `PYCONTEXTIFY_INDEX_DIR` — string — default: `./index_data` — Directory for storing search indices
- `PYCONTEXTIFY_AUTO_PERSIST` — boolean — default: `true` — Automatically save after indexing
- `PYCONTEXTIFY_AUTO_LOAD` — boolean — default: `true` — Automatically load index on startup
- `PYCONTEXTIFY_CHUNK_SIZE` — integer — default: `512` — Text chunk size for processing
- `PYCONTEXTIFY_USE_HYBRID_SEARCH` — boolean — default: `false` — Enable hybrid vector + keyword search

**Priority**: CLI arguments > Environment variables > Defaults

Copy `.env.example` to `.env` and customize as needed.

## API Reference

PyContextify exposes 5 MCP (Model Context Protocol) functions for semantic search and indexing:

1. **`index_filebase(path, topic)`** - Unified indexing for codebases and documents with relationship extraction
2. **`discover()`** - List indexed topics for browsing and filtering
3. **`search(query, top_k=5)`** - Hybrid semantic + keyword search
4. **`reset_index(remove_files=True, confirm=False)`** - Clear index data
5. **`status()`** - Get system statistics and health

Full docs: See [WARP.md](./WARP.md) for development guidance and architecture details

## Tests & CI

Run tests:

```bash
# Run all tests with coverage (requires uv >= 0.4.20 for dependency groups)
uv run --extra dev --group dev pytest --cov=pycontextify

# Run MCP-specific tests
uv run python scripts/run_mcp_tests.py

# Quick smoke test
uv run python scripts/run_mcp_tests.py --smoke
```

CI: Manual testing ![Tests](https://img.shields.io/badge/tests-247_passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-69%25-green)

## Publishing to PyPI

Use the dedicated release checklist in [RELEASING.md](./RELEASING.md) when preparing a public build.

Quick reference:

1. Bump `version` in `pyproject.toml` (and ensure changelog coverage)
2. Run the full test suite or `uv run python scripts/run_mcp_tests.py --smoke`
3. Build distributables and run metadata checks:

   ```bash
   python scripts/build_package.py
   ```

4. Upload to TestPyPI or PyPI with Twine once validation passes:

   ```bash
   twine upload dist/*
   ```

## Changelog

Detailed release history lives in [CHANGELOG.md](./CHANGELOG.md). Update the changelog alongside any version bump so users can track notable changes between releases.

## Contributing

Please read [CONTRIBUTING.md](./CONTRIBUTING.md) (or follow the short flow below):

1. Fork the project
2. Create a branch `feature/your-feature`
3. Add tests and documentation
4. Open a pull request

## Security

Please report security issues to: Create an issue in this repository (or see [SECURITY.md](./SECURITY.md))

## License

![MIT License](https://img.shields.io/badge/license-MIT-green)  
This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

## Maintainers

- PyContextify Project — contact: Create an issue for questions or support

