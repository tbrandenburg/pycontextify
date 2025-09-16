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

### ⚡ **measure_startup_time.py**
Comprehensive performance measurement for MCP server startup and operations.

**Usage:**
```bash
$env:PYTHONPATH = "."; python scripts/measure_startup_time.py
```

**Features:**
- ✅ Config initialization timing
- ✅ IndexManager startup measurement
- ✅ Model loading performance
- ✅ Document indexing benchmarks
- ✅ Search operation timing
- ✅ Performance targets and recommendations

### 🐛 **debug_lazy_loading.py**
Debug script for testing lazy loading implementation in IndexManager.

**Usage:**
```bash
$env:PYTHONPATH = "."; python scripts/debug_lazy_loading.py
```

**Features:**
- ✅ Tests instant IndexManager initialization
- ✅ Verifies embedder lazy loading
- ✅ Validates vector store initialization
- ✅ Checks status query performance

### 🚀 **fast_startup_test.py**
Tests optimized configuration for fastest possible startup times.

**Usage:**
```bash
$env:PYTHONPATH = "."; python scripts/fast_startup_test.py
```

**Features:**
- ✅ Optimized config testing
- ✅ Component timing analysis
- ✅ Lazy loading demonstration
- ✅ Performance recommendations

### 📊 **detailed_perf.py**
Detailed component-by-component performance analysis for startup optimization.

**Usage:**
```bash
$env:PYTHONPATH = "."; python scripts/detailed_perf.py
```

**Features:**
- ✅ Individual component timing
- ✅ Bottleneck identification
- ✅ Timing breakdown analysis
- ✅ Optimization targets

### 🌐 **test_hf_connectivity.py**
Tests connectivity to HuggingFace Hub for model downloads and API access.

**Usage:**
```bash
$env:PYTHONPATH = "."; python scripts/test_hf_connectivity.py
```

**Features:**
- ✅ DNS resolution testing
- ✅ SSL connection validation
- ✅ HuggingFace API access
- ✅ Model download testing
- ✅ Tokenizer loading verification

## Adding New Scripts

When adding new utility scripts:

1. **Place in this directory** (`scripts/`)
2. **Make executable** if needed: `chmod +x script_name.py`
3. **Add to this README** with usage instructions
4. **Use proper error handling** and exit codes
5. **Include help text** with `--help` option

## Running Scripts

**From project root directory:**
```bash
# Windows PowerShell
$env:PYTHONPATH = "."; python scripts/script_name.py

# Linux/Mac
PYTHONPATH=. python scripts/script_name.py

# Alternative: use -m flag
python -m scripts.script_name
```

## Script Conventions

- **Entry point**: Use `if __name__ == "__main__":` pattern
- **Error handling**: Return proper exit codes (0 for success, 1 for failure)
- **Output**: Use emoji and clear formatting for user feedback
- **Paths**: Use relative paths from project root
- **Dependencies**: Import from project modules as needed
- **Python path**: Scripts assume they're run from project root with PYTHONPATH set
