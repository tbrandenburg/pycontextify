# Smoke Tests

This directory contains smoke tests for PyContextify that verify basic system functionality without requiring heavy dependencies.

## Purpose

Smoke tests are designed to:
- ✅ Verify all core modules can be imported
- ✅ Test configuration system functionality  
- ✅ Validate MCP server initialization
- ✅ Check system integration without embeddings
- ✅ Provide quick feedback on system health

## Running Smoke Tests

### Individual Tests

```bash
# Test MCP server interface and imports
python3 tests/smoke/test_mcp_server.py

# Test MCP functionality and tool calls
python3 tests/smoke/test_mcp_functionality.py
```

### All Smoke Tests

```bash
# Run all smoke tests with pytest (from project root)
python3 -m pytest tests/smoke/ -v

# Or use the simple test runner (from project root)
python3 tests/smoke/run_smoke_tests.py

# Or run them individually as scripts
python3 tests/smoke/test_mcp_server.py
python3 tests/smoke/test_mcp_functionality.py

# From within the smoke tests directory
cd tests/smoke
python3 run_smoke_tests.py
```

## Test Files

### `test_mcp_server.py`
- Tests optional dependency imports
- Validates configuration creation
- Tests mock MCP server interface
- Verifies actual MCP server initialization

### `test_mcp_functionality.py` 
- Tests MCP tool function imports
- Validates configuration-only mode
- Tests individual MCP tool calls
- Checks graceful error handling

### `run_smoke_tests.py`
- Simple test runner that doesn't require pytest
- Runs all smoke tests in sequence
- Provides clear pass/fail summary
- Works from any directory (project root or smoke tests directory)

## Expected Behavior

These tests are designed to work even when heavy dependencies (like `torch` and `sentence-transformers`) are not installed. They will:

- ✅ Pass when core functionality is working
- ⚠️ Show expected warnings about missing embedding providers
- ❌ Only fail if there are actual import or configuration issues

The tests provide clear feedback about what's working and what requires additional setup.