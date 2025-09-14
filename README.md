# PyContextify

A Python-based MCP (Model Context Protocol) server for semantic search over codebases, documents, and webpages with lightweight knowledge graph capabilities.

## Features

- **Semantic Search**: Vector similarity search using FAISS with multiple embedding providers
- **Multi-Source Support**: Index codebases, documents (PDF, Markdown, text), and webpages
- **Lightweight Knowledge Graph**: Relationship extraction and entity linking without complex graph databases
- **Specialized Content Processing**: Dedicated chunkers for code, documents, and web content
- **Auto-Persistence**: Automatic saving with configurable backup management
- **Extensible Embedding System**: Support for sentence-transformers with plans for Ollama and OpenAI
- **Modern UV Project**: Fast dependency management and reproducible builds
- **Simplified MCP Interface**: Essential functions only for ease of use

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager

### Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pycontextify
```

2. Install dependencies:
```bash
uv sync
```

3. For development with additional tools:
```bash
uv sync --extra dev
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Quick Start

### Start the MCP Server

```bash
# Production mode
uv run pycontextify

# Or directly with FastMCP
uv run fastmcp run pycontextify/mcp_server.py

# Development mode with tools
uv run --extra dev fastmcp run pycontextify/mcp_server.py
```

### Example Usage

The server provides 6 MCP functions:

1. **`index_code(path)`** - Index a codebase directory
2. **`index_document(path)`** - Index a single document (PDF, MD, TXT)
3. **`index_webpage(url, recursive=False, max_depth=1)`** - Index web content
4. **`search(query, top_k=5)`** - Basic semantic search
5. **`status()`** - Get system status and statistics
6. **`search_with_context(query, top_k=5, include_related=False)`** - Enhanced search with relationship context

## Configuration

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

- **IndexManager**: Central orchestrator handling all operations
- **VectorStore**: FAISS wrapper with persistence and backup
- **EmbedderFactory**: Extensible embedding provider system
- **Specialized Chunkers**: Content-specific processing
- **RelationshipStore**: Lightweight knowledge graph storage
- **MetadataStore**: Chunk metadata with relationship tracking

### Content Processing

- **CodeChunker**: Function/class boundary awareness, symbol extraction
- **DocumentChunker**: Section hierarchy, citation extraction
- **WebPageChunker**: HTML structure awareness, link extraction
- **Automatic Selection**: Based on content type

### Knowledge Graph Features

- **Lightweight Approach**: No external graph database required
- **Relationship Types**: Function calls, imports, references, links, hierarchies
- **Entity Extraction**: Automatic from code, documents, and web content
- **Relationship-Aware Search**: Enhanced search via `search_with_context`

## MCP Client Integration

### Claude Desktop Configuration

Add to your Claude Desktop config:

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

**Unit Tests**: Comprehensive test suite with 55+ tests covering:
- Configuration system and environment variables
- Metadata and relationship management
- Content loading and chunking
- Embedding system factories
- Index management and persistence

**Smoke Tests**: Quick integration tests that work with minimal dependencies:
```bash
# Test MCP server interface and imports
python3 tests/smoke/test_mcp_server.py

# Test MCP functionality without heavy dependencies
python3 tests/smoke/test_mcp_functionality.py

# Run all smoke tests with pytest (if pytest available)
python3 -m pytest tests/smoke/ -v

# Or use the simple test runner (no pytest required)
python3 tests/smoke/run_smoke_tests.py
```

Smoke tests are perfect for:
- ✅ Verifying system health without installing torch/sentence-transformers
- ✅ Testing MCP server initialization and tool registration
- ✅ Validating configuration and core module functionality
- ✅ Quick CI/CD pipeline checks

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
- HTML pages with optional recursive crawling
- Automatic content filtering and structure extraction

## Performance Considerations

- **Memory**: Scales with corpus size and embedding model
- **Embedding Models**: 
  - `all-MiniLM-L6-v2`: Fast, 384 dimensions
  - `all-mpnet-base-v2`: High quality, 768 dimensions
- **Chunking Strategy**: Affects search granularity
- **Auto-persistence**: Minimal overhead with compression
- **Relationship Extraction**: Configurable complexity

## Troubleshooting

### Common Issues

**Model Loading Errors**: Ensure sufficient memory and stable internet for first download

**Memory Issues**: Use smaller embedding models or reduce batch sizes

**File Permission Problems**: Check index directory write permissions

**Import Errors**: Verify UV virtual environment activation

### UV-Specific Issues

```bash
# Recreate environment
uv sync --reinstall

# Clear cache
uv cache clean

# Check lock file
uv lock --check
```

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

- [ ] Ollama embedding provider
- [ ] OpenAI embedding provider  
- [ ] Advanced relationship queries
- [ ] Vector database alternatives
- [ ] Performance optimizations
- [ ] Web UI for management

## License

MIT License - see LICENSE file for details

## Support

- GitHub Issues: Report bugs and feature requests
- Documentation: This README and inline docstrings
- Examples: See `examples/` directory (coming soon)