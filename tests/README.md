# PyContextify Testing Suite

This directory contains the automated regression tests for PyContextify.
They are organised by scope so contributors can quickly discover the
appropriate place to add coverage.

## Test file overview

### 🎯 End-to-end MCP coverage
- **`test_mcp_server.py`** – Consolidated MCP tests that exercise
  validation helpers, tool wiring, and a smoke-test workflow running
  through the FastMCP interface. These tests replace the earlier
  `test_mcp_system_*` modules while remaining lightweight and easy to
  maintain.

### 🔧 Integration tests
- **`test_integration.py`** – Exercises the indexing pipeline and search
  stack directly through the `IndexManager` to verify chunking, embedding,
  and retrieval logic with realistic content. Document and relationship
  scenarios live here while code ingestion is validated via the MCP suite.

### 📦 Unit tests
- **`test_chunker.py`**, **`test_cli_args.py`**, **`test_config.py`**,
  **`test_embeddings.py`**, **`test_loaders.py`**, **`test_metadata.py`**,
  **`test_models.py`**, **`test_pdf_loader.py`**, **`test_persistence.py`**,
  **`test_vector_store.py`**, and others cover the focused components that
  make up the indexing runtime.

## Running the suite

```bash
# Run every test
uv run python -m pytest tests -v

# Run only the MCP tests
uv run python -m pytest tests/test_mcp_server.py -v

# Run the integration pipeline checks
uv run python -m pytest tests/test_integration.py -v
```

## Conventions

- The `sentence-transformers` library is mocked globally for fast and
  deterministic execution. Add the `@pytest.mark.no_mock_st` marker to a
  test when real model behaviour is required.
- Temporary directories are used for isolation, and the MCP manager is
  reset between tests to avoid state bleed.
- Prefer extending the existing files listed above rather than adding new
  top-level modules unless the functionality under test is substantial.
