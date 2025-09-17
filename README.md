# PyContextify

A Python-based MCP (Model Context Protocol) server for semantic search over codebases, documents, and webpages with lightweight knowledge graph capabilities.

## Features

- **Semantic Search**: FAISS vector similarity search with hybrid keyword matching (TF-IDF + BM25)
- **Multi-Source Support**: Index codebases, documents (PDF, Markdown, text), and webpages
- **Lightweight Knowledge Graph**: Relationship extraction without external graph databases
- **Content-Aware Processing**: Specialized chunkers for code structure, document hierarchy, and web content
- **Advanced Search**: Vector similarity, keyword search, and neural reranking
- **Auto-Persistence**: Automatic saving with compressed backups
- **Lazy Loading**: Fast startup with on-demand component initialization
- **CLI Configuration**: Command-line arguments with environment variable overrides
- **Extensible Architecture**: Factory pattern for embedding providers and content processors

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager

### Project Setup

1. Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone and install:
```bash
git clone <repository-url>
cd pycontextify
uv sync
```

3. Optional - development tools:
```bash
uv sync --extra dev
```

## Quick Start

### Start the MCP Server

**Basic startup:**
```bash
# Production mode
uv run pycontextify

# With verbose logging
uv run pycontextify --verbose
```

**CLI Arguments:**
```bash
# Start with custom index path
uv run pycontextify --index-path ./my_project_index

# Start with custom index name
uv run pycontextify --index-name project_search

# Start and index initial documents
uv run pycontextify --initial-documents README.md docs/api.md

# Start and index initial codebase
uv run pycontextify --initial-codebase src tests

# Full example with multiple options
uv run pycontextify \
  --index-path ./project_index \
  --index-name my_search \
  --initial-documents README.md \
  --initial-codebase src tests \
  --embedding-provider sentence_transformers \
  --verbose

# Show all available options
uv run pycontextify --help
```

### MCP Functions

The server provides 6 essential MCP functions:

1. **`index_code(path)`** - Index codebase with relationship extraction
2. **`index_document(path)`** - Index documents (PDF, MD, TXT)
3. **`index_webpage(url, recursive=False, max_depth=1)`** - Index web content
4. **`search(query, top_k=5)`** - Hybrid semantic + keyword search
5. **`search_with_context(query, top_k=5, include_related=False)`** - Search with relationship context
6. **`status()`** - System status and statistics

### CLI Usage Examples

```bash
# Quick setup for a Python project
uv run pycontextify \
  --index-name python_project \
  --initial-codebase ./src ./tests \
  --initial-documents README.md docs/

# Documentation-focused setup
uv run pycontextify \
  --index-path ./docs_index \
  --initial-documents *.md docs/ \
  --no-auto-persist \
  --verbose

# Multi-language codebase
uv run pycontextify \
  --initial-codebase frontend/ backend/ mobile/ \
  --embedding-provider sentence_transformers \
  --embedding-model all-mpnet-base-v2

# Research paper analysis with related websites
uv run pycontextify \
  --index-name research \
  --initial-documents papers/*.pdf references.md \
  --initial-webpages https://arxiv.org/abs/1234.5678 \
  --index-path ./research_index

# Documentation site with recursive crawling
uv run pycontextify \
  --index-name docs_site \
  --initial-webpages https://docs.myproject.com \
  --recursive-crawling --max-crawl-depth 2 \
  --crawl-delay 2

# Comprehensive knowledge base
uv run pycontextify \
  --index-name knowledge_base \
  --initial-documents ./knowledge/*.md \
  --initial-codebase ./examples \
  --initial-webpages https://api-docs.com https://tutorials.com \
  --recursive-crawling --max-crawl-depth 1
```

## Configuration

### CLI Arguments

Command-line arguments for startup configuration:

```bash
# Index configuration
--index-path DIR        # Directory for vector storage (default: ./index_data)
--index-name NAME       # Custom index name (default: semantic_index)

# Initial indexing
--initial-documents FILES...  # Documents to index at startup
--initial-codebase DIRS...    # Codebases to index at startup
--initial-webpages URLS...    # Webpages to index at startup (http/https only)

# Webpage crawling options
--recursive-crawling    # Enable recursive crawling for webpages
--max-crawl-depth N     # Maximum crawl depth (1-3, default: 1)
--crawl-delay N         # Delay between requests in seconds (default: 1)

# Server options  
--no-auto-persist       # Disable automatic persistence
--no-auto-load         # Disable automatic index loading

# Embedding configuration
--embedding-provider PROVIDER  # Provider: sentence_transformers, ollama, openai
--embedding-model MODEL        # Model name for the provider

# Logging
--verbose, -v          # Enable verbose logging (DEBUG level)
--quiet               # Minimize logging (WARNING level only)

# Help
--help, -h            # Show all options with examples
```

**Configuration Priority:** CLI arguments > Environment variables > Defaults

### Environment Variables

Copy `.env.example` to `.env` and customize:

### Embedding Providers

**Sentence Transformers (Default)**:
```bash
PYCONTEXTIFY_EMBEDDING_PROVIDER=sentence_transformers
PYCONTEXTIFY_EMBEDDING_MODEL=all-mpnet-base-v2
```

**Future Providers**:
```bash
# Ollama (coming soon)
PYCONTEXTIFY_EMBEDDING_PROVIDER=ollama
PYCONTEXTIFY_EMBEDDING_MODEL=nomic-embed-text

# OpenAI (coming soon)  
PYCONTEXTIFY_EMBEDDING_PROVIDER=openai
PYCONTEXTIFY_EMBEDDING_MODEL=text-embedding-3-small
```

### Storage & Persistence

```bash
PYCONTEXTIFY_INDEX_DIR=./index_data
PYCONTEXTIFY_AUTO_PERSIST=true
PYCONTEXTIFY_BACKUP_INDICES=false
```

### Text Processing

```bash
PYCONTEXTIFY_CHUNK_SIZE=512
PYCONTEXTIFY_CHUNK_OVERLAP=50
PYCONTEXTIFY_ENABLE_RELATIONSHIPS=true
```

## Architecture

### Core Components

- **IndexManager**: Central orchestrator with lazy loading
- **VectorStore**: FAISS wrapper with persistence and backup
- **EmbedderFactory**: Extensible embedding provider system
- **HybridSearchEngine**: Combines vector similarity with TF-IDF + BM25
- **CrossEncoderReranker**: Neural reranking for improved results
- **RelationshipStore**: Lightweight knowledge graph without external databases
- **MetadataStore**: Chunk metadata with FAISS ID mapping

### Content Processing Pipeline

1. **Load**: Content-specific loaders (PDF, web, file)
2. **Chunk**: Content-aware chunking with relationship extraction
3. **Embed**: Lazy-loaded embedding providers
4. **Store**: Parallel storage in vector, metadata, and relationship stores
5. **Search**: Multi-modal search with optional reranking

### Search Capabilities

- **Vector Similarity**: FAISS IndexFlatIP for cosine similarity
- **Keyword Search**: TF-IDF and BM25 with configurable weighting
- **Neural Reranking**: Cross-encoder for improved relevance
- **Relationship Context**: Knowledge graph traversal for related content

## MCP Client Integration

### Claude Desktop Configuration

**Basic configuration:**
```json
{
  "mcpServers": {
    "pycontextify": {
      "command": "uv",
      "args": ["run", "pycontextify"],
      "cwd": "/path/to/pycontextify"
    }
  }
}
```

**Project-specific setup:**
```json
{
  "mcpServers": {
    "my-project-search": {
      "command": "uv",
      "args": [
        "run", "pycontextify",
        "--index-path", "./project_index",
        "--initial-codebase", "src", "tests",
        "--initial-documents", "README.md"
      ],
      "cwd": "/path/to/pycontextify"
    }
  }
}
```

## Development

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

# Run tests
uv run pytest

# Run smoke tests (minimal dependencies)
python3 tests/smoke/test_mcp_server.py
python3 tests/smoke/test_mcp_functionality.py
```

### Testing

**Unit Tests**: Comprehensive test suite with 230+ tests:
```bash
# Run all tests with coverage
uv run pytest --cov=pycontextify

# Run specific test categories
uv run pytest tests/test_*_consolidated.py
```

**Current Test Coverage**: 71% overall with excellent coverage of core components:
- Vector Store: 87%
- Reranker: 91% 
- Metadata: 91%
- Relationship Store: 84%

### Adding Dependencies

```bash
# Add runtime dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Add optional dependency
uv add --optional embedding-provider package-name
```

## Supported File Types

### Code Files
- Python (`.py`)
- JavaScript/TypeScript (`.js`, `.ts`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`)
- Rust (`.rs`)
- Go (`.go`)
- And more...

### Documents
- PDF files (`.pdf`)
- Markdown (`.md`)
- Text files (`.txt`)

### Web Content
- HTML pages with optional recursive crawling (CLI: `--recursive-crawling`)
- Configurable crawl depth (CLI: `--max-crawl-depth`, max: 3 for safety)
- Respectful crawling with delays (CLI: `--crawl-delay`, min: 1 second)
- Automatic content filtering and structure extraction

## Performance

- **Memory**: Scales with corpus size and embedding model choice
- **Search**: Sub-second with FAISS IndexFlatIP
- **Startup**: Lazy loading reduces initialization time
- **Models**: 
  - `all-MiniLM-L6-v2`: Fast, 384 dimensions
  - `all-mpnet-base-v2`: Higher quality, 768 dimensions
- **Auto-persistence**: Compressed storage with minimal overhead

## Troubleshooting

**Model Loading**: Ensure stable internet for first-time model download
**Memory Issues**: Use `all-MiniLM-L6-v2` model or reduce batch sizes
**Permissions**: Check write access to index directory
**Dependencies**: Recreate environment with `uv sync --reinstall`

## Contributing

1. Set up development environment:
```bash
uv sync --extra dev
```

2. Follow code style guidelines:
```bash
uv run black .
uv run isort .
uv run flake8
```

3. Run tests:
```bash
uv run pytest
```

4. Add type hints and documentation

### Adding Embedding Providers

1. Implement `BaseEmbedder` interface
2. Register with `EmbedderFactory`
3. Add configuration validation
4. Update documentation

### Extending Chunkers

1. Inherit from `BaseChunker`
2. Implement content-specific logic
3. Add to `ChunkerFactory`
4. Include relationship extraction

## Roadmap

- [ ] Additional embedding providers (Ollama, OpenAI)
- [ ] Advanced relationship queries
- [ ] Performance optimizations for large corpora
- [ ] Web UI for index management

## License

MIT License - see LICENSE file for details

## Support

- GitHub Issues: Report bugs and feature requests
- Documentation: This README and inline docstrings
- Examples: See `examples/` directory (coming soon)