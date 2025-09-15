# PyContextify Scripts

This directory contains utility scripts and runners for the PyContextify project.

## Scripts

### 🧪 **run_mcp_tests.py**
Comprehensive test runner for the PyContextify MCP server.

**Usage:**
```bash
# Full test suite with coverage reporting
uv run python scripts/run_mcp_tests.py

# Quick smoke test for faster feedback
uv run python scripts/run_mcp_tests.py --smoke
```

**Features:**
- ✅ Tests all 6 MCP functions
- ✅ Multiple document types (Markdown, Text, Codebase)
- ✅ Coverage reporting with missing lines
- ✅ Performance timing and slowest test identification
- ✅ Clean success/failure reporting
- ✅ Quick smoke test option

**Output Example:**
```
🚀 PyContextify MCP Server Test Suite
==================================================
📁 Running tests from: tests\test_mcp_simple.py
📊 Testing all 6 MCP functions with multiple document types

✅ status() - System status reporting
✅ index_document() - Single file indexing  
✅ index_code() - Codebase directory indexing
✅ search() - Basic semantic search
✅ search_with_context() - Enhanced search
✅ Error handling - Invalid input testing
✅ Full workflow - End-to-end pipeline
✅ Function availability - Direct access verification

📄 Document types tested:
• Markdown (.md) - Documentation, guides
• Text (.txt) - Code files, configs, general content
• Codebase indexing - Multi-file directory processing

🎉 All MCP tests passed successfully!
```

## Adding New Scripts

When adding new utility scripts:

1. **Place in this directory** (`scripts/`)
2. **Make executable** if needed: `chmod +x script_name.py`
3. **Add to this README** with usage instructions
4. **Use proper error handling** and exit codes
5. **Include help text** with `--help` option

## Script Conventions

- **Entry point**: Use `if __name__ == "__main__":` pattern
- **Error handling**: Return proper exit codes (0 for success, 1 for failure)
- **Output**: Use emoji and clear formatting for user feedback
- **Paths**: Use relative paths from project root
- **Dependencies**: Import from project modules as needed