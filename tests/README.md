# PyContextify Testing Suite

This directory contains the automated regression tests for PyContextify.
They are organised by scope so contributors can quickly discover the
appropriate place to add coverage. Two sub-packages now separate the
fast unit suite from slower integration coverage, and each directory
applies the appropriate `@pytest.mark.unit` or `@pytest.mark.integration`
marker via a local `conftest.py`.

## Directory overview

```
tests/
├── unit/          # Fast component-level checks marked as unit tests
└── integration/   # Scenario and system tests marked as integration tests
```

### 📦 Unit tests (`tests/unit`)
- `test_chunker.py`, `test_cli_args.py`, `test_config.py`,
  `test_embeddings.py`, `test_metadata.py`,
  `test_models.py`, `test_pdf_loader.py`, `test_persistence.py`,
  `test_vector_store.py`, and related helpers cover the focused
  components that make up the indexing runtime.

### 🔧 Integration tests (`tests/integration`)
- `test_bootstrap_integration.py`, `test_hybrid_search.py`,
  `test_integration.py`, `test_mcp_server.py`,
  `test_persistence.py` exercise end-to-end behaviour such as
  bootstrap archives, metadata-backed search, and MCP workflows.

## Running the suite

```bash
# Run every test
uv run python -m pytest tests -v

# Run only the unit suite
uv run python -m pytest tests/unit -v

# Run only the integration scenarios
uv run python -m pytest tests/integration -v
```

## Conventions

- The `sentence-transformers` library is mocked globally for fast and
  deterministic execution. Add the `@pytest.mark.no_mock_st` marker to a
  test when real model behaviour is required.
- Temporary directories are used for isolation, and the MCP manager is
  reset between tests to avoid state bleed.
- Prefer extending the existing files listed above rather than adding new
  top-level modules unless the functionality under test is substantial.
