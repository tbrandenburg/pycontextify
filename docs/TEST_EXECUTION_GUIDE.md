# Test Execution Guide

This guide explains how to run the PyContextify test suite efficiently using pytest markers.

## Quick Reference

| Command | Description | Time | Tests |
|---------|-------------|------|-------|
| `uv run pytest tests/` | Run all tests | ~98s | 278 tests |
| `uv run pytest tests/ -m "not slow"` | Run fast tests only | ~21s | 275 tests |
| `uv run pytest tests/ -m "slow"` | Run slow tests only | ~79s | 3 tests |
| `uv run pytest tests/ -m "integration"` | Run integration tests | ~90s | ~50 tests |
| `uv run pytest tests/ -m "unit"` | Run unit tests | ~10s | ~228 tests |

## Test Markers

The test suite uses the following pytest markers to organize and filter tests:

### `@pytest.mark.slow`
**Purpose**: Marks tests that take a long time to execute (typically >3 seconds)

**Use Cases**:
- Web crawling tests (real network calls)
- Large file processing
- Tests with significant I/O operations

**Currently Marked Tests**:
- `test_recursive_crawl_simple_site` (75.68s) - Recursive web crawling
- `test_single_page_non_recursive` (3.16s) - Single page crawling
- `test_markdown_document_embedding_generation` - Full embedding pipeline

### `@pytest.mark.integration`
**Purpose**: Marks tests that test multiple components together

**Use Cases**:
- End-to-end workflows
- Tests requiring external dependencies
- Real file system operations
- Network operations

### `@pytest.mark.unit`
**Purpose**: Marks fast, isolated unit tests

**Use Cases**:
- Testing individual functions/methods
- Tests with mocked dependencies
- Quick validation during development

### `@pytest.mark.embedding`
**Purpose**: Marks tests that require embedding models

**Use Cases**:
- Tests that load transformer models
- Embedding generation tests
- May be slow due to model loading

## Common Workflows

### Development Workflow (Fast Feedback)
Run fast tests only during active development:
```bash
uv run pytest tests/ -m "not slow"
```
**Result**: ~21 seconds, 275 tests

### Pre-Commit Testing
Run all tests before committing:
```bash
uv run pytest tests/
```
**Result**: ~98 seconds, 278 tests

### Unit Tests Only
Focus on unit tests during TDD:
```bash
uv run pytest tests/unit/
```
**Result**: ~10 seconds, 228 tests

### Integration Tests Only
Test system integration:
```bash
uv run pytest tests/integration/
```
**Result**: ~90 seconds, 50 tests

### Skip Web Crawling Tests
Skip slow web crawling during local development:
```bash
uv run pytest tests/ -m "not slow" --co -q
```
This excludes:
- Recursive web crawling (75s)
- Single page crawling (3s)
- Full embedding generation (variable)

## Coverage Testing

### Fast Tests with Coverage
```bash
uv run pytest tests/ -m "not slow" --cov=pycontextify --cov-report=term-missing
```
**Result**: ~21 seconds with coverage report

### Full Coverage
```bash
uv run pytest tests/ --cov=pycontextify --cov-report=term-missing
```
**Result**: ~98 seconds with comprehensive coverage

### Coverage HTML Report
```bash
uv run pytest tests/ --cov=pycontextify --cov-report=html
# Open htmlcov/index.html in browser
```

## Performance Analysis

### Find Slowest Tests
```bash
uv run pytest tests/ --durations=10
```

### Find All Slow Tests
```bash
uv run pytest tests/ --durations=0 | grep -E "^[0-9]+\.[0-9]+s"
```

## CI/CD Recommendations

### Pull Request CI
Run fast tests for quick feedback:
```yaml
- name: Fast tests
  run: uv run pytest tests/ -m "not slow" --cov=pycontextify
```

### Main Branch CI
Run full test suite:
```yaml
- name: Full test suite
  run: uv run pytest tests/ --cov=pycontextify --cov-report=xml
```

### Nightly CI
Run all tests including slow integration tests:
```yaml
- name: Comprehensive tests
  run: uv run pytest tests/ --cov=pycontextify --cov-report=html
```

## Test Performance Breakdown

### Total Test Suite: 98.10s (278 tests)
- **Slow tests** (3 tests): ~79s (80% of time)
  - `test_recursive_crawl_simple_site`: 75.68s
  - `test_single_page_non_recursive`: 3.16s
  - Embedding generation test: variable
  
- **Fast tests** (275 tests): ~19s (20% of time)
  - Average: ~0.07s per test
  - Unit tests are highly optimized
  
- **Bootstrap teardowns**: 0.4-0.5s each
  - File cleanup operations
  - Temporary directory removal

## Best Practices

1. **During Development**: Use `-m "not slow"` for rapid iteration
2. **Before Commit**: Run full test suite to ensure no regressions
3. **Code Review**: Focus on tests affected by your changes
4. **CI Pipeline**: Use fast tests for PRs, full tests for merges
5. **Coverage Monitoring**: Always enable coverage reporting

## Troubleshooting

### Tests Running Too Slow
- Use `-m "not slow"` to skip web crawling tests
- Run specific test files: `pytest tests/unit/test_chunker.py`
- Use `-k` to run specific test names: `pytest -k "test_chunk_text"`

### Coverage Not Updating
- Clear coverage cache: `rm -rf .coverage htmlcov/`
- Run with `--cov-report=term-missing` to see uncovered lines

### Integration Tests Failing
- Check network connectivity
- Verify external dependencies are available
- Review test logs for specific failures

## Adding New Slow Tests

When adding tests that take >3 seconds:

```python
import pytest

@pytest.mark.integration
@pytest.mark.slow
class TestMySlowFeature:
    def test_slow_operation(self):
        # Test code here
        pass
```

This ensures developers can skip your test during rapid development cycles.
