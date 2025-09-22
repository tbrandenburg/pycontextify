# PyContextify ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

**One-line:** Semantic search server with relationship-aware discovery across codebases, documents, and webpages.

PyContextify is a Python-based MCP (Model Context Protocol) server that provides intelligent semantic search capabilities over diverse knowledge sources. It combines vector similarity search with lightweight knowledge graph features to help developers, researchers, and technical writers discover contextually relevant information across codebases, documentation, and web resources.

---

## Quickstart

```bash
# Install UV and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run MCP server
uv run pycontextify --verbose
```

## Table of Contents
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
- [Chunking Techniques](#chunking-techniques)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Tests & CI](#tests--ci)
- [Contributing](#contributing)
- [License](#license)
- [Security](#security)
- [Maintainers](#maintainers)

## Installation

**Requirements:** Python 3.10+ and UV package manager

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone <repository-url>
cd pycontextify
uv sync

# Optional: Install development tools
uv sync --extra dev
```

(Alternative: build from source: `uv sync --reinstall`)

## Usage

Minimal example:

```bash
# Start the MCP server
uv run pycontextify

# Index a codebase (in another terminal or via MCP client)
# The server exposes 6 MCP functions:
# - index_code(path)
# - index_document(path) 
# - index_webpage(url, recursive=False, max_depth=1)
# - search(query, top_k=5)
# - reset_index(remove_files=True, confirm=False)
# - status()
```

Expected output:

```
🚀 Starting PyContextify MCP Server...
Server provides 6 MCP functions:
  - index_code(path): Index codebase directory
  - index_document(path): Index document files (PDF, MD, TXT)
  - index_webpage(url, recursive, max_depth): Index web content
  - search(query, top_k): Semantic search
  - reset_index(remove_files, confirm): Clear index data
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

#### **Webpage Chunking** (`WebPageChunker`)
- **Primary Strategy**: HTML semantic structure awareness (extends DocumentChunker)
- **Web-Specific Processing**: Domain extraction, URL path analysis, link discovery
- **Content Processing**: Works with cleaned HTML text content
- **Relationship Extraction**: External links (max 10), email addresses, domain tags
- **Metadata Enhancement**: URL parsing, path segmentation, contact information

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

**Bottom Line**: PyContextify's chunking system intelligently adapts to content structure—respecting code boundaries, document hierarchy, and web semantics—while maintaining configurable token limits and extracting contextual relationships for enhanced semantic search.

## Configuration

Required environment variables / config:
- `PYCONTEXTIFY_EMBEDDING_MODEL` — string — default: `all-MiniLM-L6-v2` — Embedding model for semantic search
- `PYCONTEXTIFY_INDEX_DIR` — string — default: `./index_data` — Directory for storing search indices
- `PYCONTEXTIFY_AUTO_PERSIST` — boolean — default: `true` — Automatically save after indexing
- `PYCONTEXTIFY_CHUNK_SIZE` — integer — default: `512` — Text chunk size for processing

Copy `.env.example` to `.env` and customize as needed.

## API Reference

PyContextify exposes 6 MCP (Model Context Protocol) functions for semantic search and indexing:
Full docs: See [WARP.md](./WARP.md) for development guidance and architecture details

## Tests & CI

Run tests:

```bash
# Run all tests with coverage
uv run pytest --cov=pycontextify

# Run MCP-specific tests
uv run python scripts/run_mcp_tests.py

# Quick smoke test
uv run python scripts/run_mcp_tests.py --smoke
```

CI: Manual testing ![Tests](https://img.shields.io/badge/tests-100%25_pass-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-67%25-yellow)

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

