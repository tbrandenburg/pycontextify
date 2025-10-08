# MCP Server Guide

Complete guide to running and using the PyContextify MCP server.

## Quick Start

### Start the Server
```bash
uv run pycontextify
```

### With Custom Index
```bash
uv run pycontextify --index-path ./my_index
```

### With Initial Content
```bash
uv run pycontextify \
  --initial-documents ./README.md \
  --initial-codebase ./src \
  --initial-webpages https://docs.example.com
```

---

## MCP Tools

The server exposes 6 tools for semantic search and indexing:

| Tool | Purpose | Parameters |
|------|---------|------------|
| `status` | Get index statistics | None |
| `index_document` | Index a document file | `file_path` (required) |
| `index_code` | Index a codebase directory | `directory_path` (required) |
| `index_webpage` | Index a webpage/website | `url` (required), `recursive`, `max_depth` |
| `search` | Semantic search | `query` (required), `top_k`, `output_format` |
| `reset_index` | Clear the index | `remove_files`, `confirm` (required) |

### Tool Examples

**Get Status:**
```json
{
  "tool": "status",
  "arguments": {}
}
```

**Index Document:**
```json
{
  "tool": "index_document",
  "arguments": {
    "file_path": "/path/to/document.pdf"
  }
}
```

**Index Codebase:**
```json
{
  "tool": "index_code",
  "arguments": {
    "directory_path": "/path/to/codebase"
  }
}
```

**Search:**
```json
{
  "tool": "search",
  "arguments": {
    "query": "how to authenticate",
    "top_k": 5
  }
}
```

**Reset Index:**
```json
{
  "tool": "reset_index",
  "arguments": {
    "remove_files": true,
    "confirm": true
  }
}
```

---

## Command-Line Options

### Index Configuration
| Option | Description | Default |
|--------|-------------|---------|
| `--index-path DIR` | Index storage directory | `~/.pycontextify` |
| `--index-name NAME` | Index name | `semantic_index` |
| `--no-auto-persist` | Disable auto-save | Enabled |
| `--no-auto-load` | Disable auto-load on startup | Enabled |

### Initial Indexing
| Option | Description |
|--------|-------------|
| `--initial-documents [FILES...]` | Documents to index at startup |
| `--initial-codebase [DIRS...]` | Codebases to index at startup |
| `--initial-webpages [URLS...]` | Webpages to index at startup |

### Web Crawling
| Option | Description | Default |
|--------|-------------|---------|
| `--recursive-crawling` | Enable recursive crawling | Disabled |
| `--max-crawl-depth N` | Maximum crawl depth (1-3) | 1 |
| `--crawl-delay N` | Delay between requests (seconds) | 1.0 |

### Embedding Configuration
| Option | Description | Default |
|--------|-------------|---------|
| `--embedding-provider` | Provider: sentence_transformers, ollama, openai | sentence_transformers |
| `--embedding-model NAME` | Model name | all-MiniLM-L6-v2 |

### Logging
| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose logging (DEBUG) |
| `--quiet` | Minimize logging (WARNING only) |

---

## Environment Variables

All CLI options can be set via environment variables:

| Variable | CLI Equivalent |
|----------|----------------|
| `PYCONTEXTIFY_INDEX_DIR` | `--index-path` |
| `PYCONTEXTIFY_INDEX_NAME` | `--index-name` |
| `PYCONTEXTIFY_AUTO_PERSIST` | `--no-auto-persist` (inverted) |
| `PYCONTEXTIFY_AUTO_LOAD` | `--no-auto-load` (inverted) |
| `PYCONTEXTIFY_EMBEDDING_PROVIDER` | `--embedding-provider` |
| `PYCONTEXTIFY_EMBEDDING_MODEL` | `--embedding-model` |
| `PYCONTEXTIFY_CRAWL_DELAY_SECONDS` | `--crawl-delay` |

**Priority**: CLI arguments > Environment variables > Defaults

---

## Integration with Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

### Basic Configuration
```json
{
  "mcpServers": {
    "pycontextify": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\pycontextify",
        "run",
        "pycontextify"
      ]
    }
  }
}
```

### With Custom Index
```json
{
  "mcpServers": {
    "pycontextify": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\pycontextify",
        "run",
        "pycontextify",
        "--index-path",
        "C:\\path\\to\\index",
        "--index-name",
        "my_project"
      ]
    }
  }
}
```

### With Initial Content
```json
{
  "mcpServers": {
    "pycontextify": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\path\\to\\pycontextify",
        "run",
        "pycontextify",
        "--index-path",
        "C:\\path\\to\\index",
        "--initial-codebase",
        "C:\\path\\to\\project\\src",
        "--initial-documents",
        "C:\\path\\to\\project\\README.md"
      ]
    }
  }
}
```

---

## Usage Examples

### Example 1: Index Documentation
```bash
uv run pycontextify \
  --index-path ./docs_index \
  --initial-documents ./docs/*.md \
  --quiet
```

### Example 2: Index Codebase with Docs
```bash
uv run pycontextify \
  --index-path ./project_index \
  --index-name my_project \
  --initial-codebase ./src \
  --initial-documents ./README.md ./docs/api.md
```

### Example 3: Crawl Documentation Website
```bash
uv run pycontextify \
  --index-path ./web_docs \
  --initial-webpages https://docs.python.org/3/ \
  --recursive-crawling \
  --max-crawl-depth 2 \
  --crawl-delay 2
```

### Example 4: Multi-Source Index
```bash
uv run pycontextify \
  --index-path ./knowledge_base \
  --initial-documents ./specs/*.pdf \
  --initial-codebase ./backend ./frontend \
  --initial-webpages https://api-docs.example.com \
  --recursive-crawling
```

---

## Server Status

Check if the server is properly configured:

```bash
# Show help
uv run pycontextify --help

# Verify command works
uv run python -c "from pycontextify import mcp; print('✅ OK')"

# List available tools
uv run python -c "from pycontextify import mcp; print(list(mcp.mcp._tool_manager._tools.keys()))"
```

**Expected tools:**
- `status`
- `index_document`
- `index_code`
- `index_webpage`
- `search`
- `reset_index`

---

## Troubleshooting

### Server Won't Start
```bash
# Check installation
uv run pycontextify --version

# Check dependencies
uv pip list | grep -E "fastmcp|faiss|sentence"

# Reinstall if needed
uv pip install -e .
```

### Import Errors
```bash
# Verify module loads
uv run python -c "import pycontextify; print('OK')"

# Check MCP module
uv run python -c "from pycontextify import mcp; print('MCP OK')"
```

### Tools Not Available
```bash
# List tools programmatically
uv run python -c "
from pycontextify import mcp
tools = list(mcp.mcp._tool_manager._tools.keys())
print(f'Available tools: {tools}')
print(f'Tool count: {len(tools)}')
"
```

Expected output:
```
Available tools: ['status', 'index_document', 'index_code', 'index_webpage', 'search', 'reset_index']
Tool count: 6
```

### Performance Issues
```bash
# Use faster embedding model
uv run pycontextify --embedding-model all-MiniLM-L6-v2

# Disable auto-persist for testing
uv run pycontextify --no-auto-persist --no-auto-load

# Reduce crawl load
uv run pycontextify --crawl-delay 3
```

---

## Testing the Server

### System Tests
Validate the MCP server works end-to-end:

```bash
# Run all system tests
uv run pytest tests/system/ -v

# Run complete user flow test
uv run pytest tests/system/test_mcp_system.py::TestMCPServerSystem::test_complete_user_flow_via_mcp -v -s
```

### Manual Testing
Test tools programmatically:

```python
from pycontextify import mcp

# Initialize manager
mcp.initialize_manager({"index_dir": "./test_index"})

# Get status
status = mcp.mcp._tool_manager._tools["status"].fn()
print(status)

# Index a document
result = mcp.mcp._tool_manager._tools["index_document"].fn("./README.md")
print(result)

# Search
results = mcp.mcp._tool_manager._tools["search"].fn("pycontextify", top_k=3)
print(results)

# Cleanup
mcp.reset_manager()
```

---

## Best Practices

### Production Deployment
1. Use dedicated index directory
2. Enable auto-persist for state preservation
3. Set reasonable crawl delays (2-3 seconds)
4. Use environment variables for configuration
5. Monitor memory usage with large indexes

### Development Setup
1. Use separate index per project
2. Disable auto-persist/auto-load for testing
3. Use smaller embedding models for speed
4. Clear index frequently during testing

### Security
1. Validate file paths before indexing
2. Sanitize URLs before crawling
3. Use confirm flag for destructive operations
4. Limit crawl depth to prevent abuse
5. Monitor disk usage for index growth

---

## Architecture

### Components
- **FastMCP**: MCP protocol implementation
- **IndexManager**: Core indexing and search logic
- **ChunkProcessor**: Text chunking and analysis
- **EmbeddingGenerator**: Vector embeddings
- **VectorStore**: FAISS-based similarity search
- **MetadataStore**: Chunk metadata and relationships

### Data Flow
```
Input (Document/Code/Webpage)
    ↓
Chunking & Analysis
    ↓
Embedding Generation
    ↓
Vector Storage (FAISS)
    ↓
Metadata Storage
    ↓
Search & Retrieval
```

---

## Summary

- ✅ **6 MCP tools** for indexing and search
- ✅ **Multiple content types** supported (documents, code, webpages)
- ✅ **Flexible configuration** via CLI and environment variables
- ✅ **Claude Desktop integration** ready
- ✅ **Comprehensive testing** included
- ✅ **Production-ready** with proper error handling

For more information:
- Documentation Index: See `docs/INDEX.md`
- Testing: See `docs/TESTING.md`
- Web Crawling: See `docs/WEB_CRAWLING.md`
- Index Bootstrap: See `docs/BOOTSTRAP.md`
