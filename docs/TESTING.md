# Testing Guide

Complete guide to testing PyContextify, including test execution, performance optimization, and validation strategies.

## Quick Start

### Run All Tests
```bash
uv run pytest tests/
```

### Run Fast Tests (Recommended for Development)
```bash
uv run pytest tests/ -m "not slow"
```

### Run with Coverage
```bash
uv run pytest tests/ --cov=pycontextify --cov-report=term-missing
```

---

## Test Organization

### Test Categories

| Category | Location | Tests | Duration | Purpose |
|----------|----------|-------|----------|---------|
| **Unit** | `tests/unit/` | 228 | ~10s | Core logic validation |
| **Integration** | `tests/integration/` | 51 | ~20s | Component interaction |
| **System** | `tests/system/` | 6 | ~15s | End-to-end MCP server |
| **Total** | `tests/` | **285** | **~45s (fast)** | Complete coverage |

### Test Markers

Tests are organized with pytest markers for selective execution:

```python
@pytest.mark.unit          # Fast unit tests
@pytest.mark.integration   # Integration tests  
@pytest.mark.system        # System-level tests
@pytest.mark.slow          # Slow tests (long-running scenarios)
@pytest.mark.embedding     # Tests requiring embedding models
```

**Run by marker:**
```bash
# Run only unit tests
uv run pytest -m "unit"

# Run integration tests
uv run pytest -m "integration"

# Run system tests
uv run pytest -m "system"

# Exclude slow tests
uv run pytest -m "not slow"
```

---

## Test Performance

### Execution Times

#### Full Suite
- **Time**: 98.10 seconds (1:38)
- **Tests**: 285 tests
- **Coverage**: 71% overall

#### Fast Mode (Recommended)
- **Time**: 20.88 seconds (~21s)
- **Tests**: 281 tests (4 slow tests excluded)
- **Coverage**: 71% (same as full suite)
- **Speed Improvement**: **78% faster**

### Slowest Tests

The following tests are marked `@pytest.mark.slow` and can be skipped:

| Test | Duration | Type | Skip Impact |
|------|----------|------|-------------|
| Embedding generation tests | Variable | Integration | Minimal |

### Development Workflow

**During Active Development:**
```bash
# Fast feedback loop (~21s)
uv run pytest tests/ -m "not slow" -q
```

**Before Committing:**
```bash
# Full validation (~98s)
uv run pytest tests/ -v
```

**Specific Module Testing:**
```bash
# Test specific module
uv run pytest tests/unit/test_chunker.py -v

# Test specific function
uv run pytest -k "test_chunk_text" -v
```

---

## Test Types Explained

### 1. Unit Tests (`tests/unit/`)

Test individual components in isolation.

**Coverage:**
- Chunking logic (SimpleChunker, CodeChunker)
- Metadata storage (ChunkMetadata, MetadataStore)
- Configuration handling
- Embedding functionality
- Vector storage operations

**Characteristics:**
- Fast execution (~0.07s per test)
- Heavy use of mocks
- No external dependencies
- Focus on logic correctness

**Example:**
```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific unit test file
uv run pytest tests/unit/test_metadata.py -v
```

### 2. Integration Tests (`tests/integration/`)

Test component interactions and workflows.

**Coverage:**
- Complete indexing pipeline
- Document → chunks → embeddings → search
- MCP server tools integration
- Persistence and loading
- Multi-document workflows

**Characteristics:**
- Moderate execution time (~20s total)
- Real embeddings generated
- Temporary file operations
- End-to-end pipelines

**Example:**
```bash
# Run all integration tests
uv run pytest tests/integration/ -v

# Run specific integration test
uv run pytest tests/integration/test_user_flow.py -v -s
```

### 3. System Tests (`tests/system/`)

Test the actual MCP server as users would experience it.

**Coverage:**
- MCP server command functionality
- Tool registration and availability
- Complete user workflow via MCP tools
- Server startup and configuration

**Characteristics:**
- Tests real MCP server behavior
- Calls actual tool functions
- Validates end-to-end system
- ~15s execution time

**Example:**
```bash
# Run all system tests
uv run pytest tests/system/ -v

# Run complete user flow system test
uv run pytest tests/system/test_mcp_system.py::TestMCPServerSystem::test_complete_user_flow_via_mcp -v -s
```

---

## Complete User Flow Validation

### Integration Test: `test_complete_user_workflow`

Tests the complete workflow through IndexManager:

1. ✅ Get initial status (empty index)
2. ✅ Index document (API documentation)
3. ✅ Index codebase (Python + JavaScript)
4. ✅ Get status after indexing
5. ✅ Perform searches (7 diverse queries)
6. ✅ Reset index
7. ✅ Verify clean state

**Result**: 16 chunks indexed, 100% search success rate

**Run:**
```bash
uv run pytest tests/integration/test_user_flow.py::TestCompleteUserFlow::test_complete_user_workflow -v -s
```

### System Test: `test_complete_user_flow_via_mcp`

Tests the same workflow via actual MCP tools:

1. ✅ Check tool availability
2. ✅ Get status via `status` tool
3. ✅ Index document via `index_document` tool
4. ✅ Index code via `index_code` tool
5. ✅ Search via `search` tool
6. ✅ Reset via `reset_index` tool

**Result**: Validates MCP server works for real users

**Run:**
```bash
uv run pytest tests/system/test_mcp_system.py::TestMCPServerSystem::test_complete_user_flow_via_mcp -v -s
```

---

## Coverage Analysis

### Current Coverage: 71%

#### High Coverage Modules (>90%)
- ✅ `storage_metadata.py` - 98%
- ✅ `search_hybrid.py` - 89%
- ✅ `embedder.py` - 89%
- ✅ `storage_vector.py` - 85%

#### Good Coverage Modules (70-89%)
- ✅ `config.py` - 79%
- ✅ `index_codebase.py` - 74%
- ✅ `embedder_factory.py` - 71%

#### Coverage Focus Areas
- `chunker.py` - 69%
- `search_models.py` - 70%
- `indexer.py` - 64%
- `index_document.py` - 65%

### Measuring Coverage

**Quick coverage check:**
```bash
uv run pytest tests/ -m "not slow" --cov=pycontextify --cov-report=term-missing:skip-covered
```

**HTML coverage report:**
```bash
uv run pytest tests/ --cov=pycontextify --cov-report=html
# Open htmlcov/index.html
```

**Coverage with specific tests:**
```bash
uv run pytest tests/unit/ --cov=pycontextify --cov-report=term
```

---

## CI/CD Recommendations

### Pull Request CI
```yaml
- name: Fast tests
  run: uv run pytest tests/ -m "not slow" --cov=pycontextify --cov-report=xml
```
**Time**: ~21 seconds

### Pre-Merge CI
```yaml
- name: Full test suite
  run: uv run pytest tests/ --cov=pycontextify --cov-report=xml
```
**Time**: ~98 seconds

### Nightly/Release CI
```yaml
- name: Comprehensive validation
  run: |
    uv run pytest tests/ -v --cov=pycontextify --cov-report=html
    uv run pytest tests/system/ -v
```
**Time**: ~2 minutes

---

## Troubleshooting

### Tests Running Too Slow
```bash
# Skip slow tests
uv run pytest tests/ -m "not slow"

# Run specific test file
uv run pytest tests/unit/test_chunker.py

# Run specific test
uv run pytest -k "test_chunk_text"
```

### Coverage Not Updating
```bash
# Clear coverage cache
rm -rf .coverage htmlcov/

# Run with fresh coverage
uv run pytest tests/ --cov=pycontextify --cov-report=term
```

### Import Errors
```bash
# Reinstall in development mode
uv pip install -e .

# Check imports
uv run python -c "import pycontextify; print('OK')"
```

### Test Failures
```bash
# Run with verbose output
uv run pytest tests/unit/test_metadata.py -vv

# Run with stdout/stderr visible
uv run pytest tests/unit/test_metadata.py -s

# Run with debugging
uv run pytest tests/unit/test_metadata.py -vv -s --pdb
```

---

## Best Practices

### During Development
1. Run fast tests frequently: `pytest -m "not slow"`
2. Focus on affected tests: `pytest tests/unit/test_module.py`
3. Use coverage to find gaps: `--cov=pycontextify`

### Before Committing
1. Run full test suite: `pytest tests/`
2. Check coverage: `--cov-report=term-missing`
3. Verify no regressions: All tests pass

### Code Review
1. Ensure new code has tests
2. Verify tests are meaningful (not just coverage)
3. Check test performance (avoid slow tests where possible)

### Test Writing Guidelines
1. Keep unit tests fast (<0.1s each)
2. Use fixtures for common setup
3. Test edge cases and error conditions
4. Prefer black-box testing for public APIs
5. Mock external dependencies in unit tests
6. Use integration tests for workflows

---

## Command Reference

### Essential Commands

| Command | Description | Time |
|---------|-------------|------|
| `uv run pytest tests/` | Run all tests | ~98s |
| `uv run pytest tests/ -m "not slow"` | Run fast tests | ~21s |
| `uv run pytest tests/unit/` | Run unit tests only | ~10s |
| `uv run pytest tests/integration/` | Run integration tests | ~20s |
| `uv run pytest tests/system/` | Run system tests | ~15s |
| `uv run pytest --durations=10` | Show 10 slowest tests | - |
| `uv run pytest -k "search"` | Run tests matching "search" | Variable |
| `uv run pytest -v` | Verbose output | - |
| `uv run pytest -s` | Show print statements | - |
| `uv run pytest -x` | Stop on first failure | - |
| `uv run pytest --lf` | Run last failed tests | - |

### Coverage Commands

| Command | Description |
|---------|-------------|
| `--cov=pycontextify` | Measure coverage |
| `--cov-report=term` | Terminal report |
| `--cov-report=term-missing` | Show uncovered lines |
| `--cov-report=html` | Generate HTML report |
| `--cov-report=xml` | Generate XML report (CI) |

---

## Summary

- ✅ **285 total tests** validating all functionality
- ✅ **71% code coverage** with focus on critical paths
- ✅ **21-second fast test suite** for rapid development
- ✅ **3 test levels**: unit, integration, system
- ✅ **Complete user flow validated** at all levels
- ✅ **MCP server fully tested** including actual command

PyContextify has comprehensive test coverage ensuring reliability and correctness for production use.

For detailed information:
- Documentation Index: See `docs/INDEX.md`
- MCP Server Guide: See `docs/MCP_SERVER.md`
- System tests: Run `pytest tests/system/ -v`
- Integration tests: Run `pytest tests/integration/ -v`
- Unit tests: Run `pytest tests/unit/ -v`
