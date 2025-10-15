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
  --initial-filebase ./src \
  --tags my-project
```

---

## MCP Tools

The server exposes 5 tools for semantic search and indexing:

| Tool | Purpose | Parameters |
|------|---------|------------|
| `status` | Get index statistics | None |
| `index_filebase` | Index a filebase (code + docs) | `base_path` (required), `tags` (required), `include`, `exclude`, `exclude_dirs` |
| `discover` | List indexed tags | None |
| `search` | Semantic search | `query` (required), `top_k`, `display_format` |
| `reset_index` | Clear the index | `remove_files`, `confirm` (required) |

### Tool Examples

**Get Status:**
```json
{
  "tool": "status",
  "arguments": {}
}
```

**Index Filebase:**
```json
{
  "tool": "index_filebase",
  "arguments": {
    "base_path": "/path/to/repository",
    "tags_input": "documentation",
    "tags": ["documentation"],
    "include": ["*.py", "*.md"],
    "exclude_dirs": [".git", "node_modules"]
  }
}
```

**Discover Tags:**
```json
{
  "tool": "discover",
  "arguments": {}
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
| `--initial-filebase DIR` | Directory tree to index at startup |
| `--tags NAME` | Tag label for the initial indexing run (required with `--initial-filebase`) |
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
        "--initial-filebase",
        "C:\\path\\to\\project",
        "--tags",
        "project-docs"
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
  --initial-filebase ./docs \
  --tags documentation \
  --quiet
```

### Example 2: Index Codebase with Docs
```bash
uv run pycontextify \
  --index-path ./project_index \
  --index-name my_project \
  --initial-filebase ./project \
  --tags project-docs \
  --quiet
```

### Example 3: Multi-Source Index
```bash
uv run pycontextify \
  --index-path ./knowledge_base \
  --initial-filebase ./knowledge_base_sources \
  --tags unified-knowledge \
  --quiet
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
- `index_filebase`
- `discover`
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
Available tools: ['status', 'index_filebase', 'discover', 'search', 'reset_index']
Tool count: 5
```

### Performance Issues
```bash
# Use faster embedding model
uv run pycontextify --embedding-model all-MiniLM-L6-v2

# Disable auto-persist for testing
uv run pycontextify --no-auto-persist --no-auto-load
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

# Index a filebase (code or docs)
index_filebase = mcp.mcp._tool_manager._tools["index_filebase"].fn
result = index_filebase(base_path="./docs", tags="documentation")
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
3. Use environment variables for configuration
4. Monitor memory usage with large indexes

### Development Setup
1. Use separate index per project
2. Disable auto-persist/auto-load for testing
3. Use smaller embedding models for speed
4. Clear index frequently during testing

### Security
1. Validate file paths before indexing
2. Use confirm flag for destructive operations
3. Monitor disk usage for index growth

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
  Input (Document/Code)
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

  - ✅ **5 MCP tools** for indexing and search
  - ✅ **Multiple content types** supported (documents and code)
  - ✅ **Flexible configuration** via CLI and environment variables
  - ✅ **Claude Desktop integration** ready
  - ✅ **Comprehensive testing** included
  - ✅ **Production-ready** with proper error handling

  For more information:
  - Documentation Index: See `docs/INDEX.md`
  - Testing: See `docs/TESTING.md`
  - Index Bootstrap: See `docs/BOOTSTRAP.md`
