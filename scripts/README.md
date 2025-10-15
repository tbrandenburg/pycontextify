# PyContextify Scripts

Utility and debug scripts for the PyContextify project.

## Build & Test Scripts

### 🧪 **run_mcp_tests.py**
Test runner with coverage reporting.

```bash
# Full test suite 
python scripts/run_mcp_tests.py

# Quick smoke test
python scripts/run_mcp_tests.py --smoke
```

### 📦 **build_package.py**
Builds distribution packages.

```bash
python scripts/build_package.py
```

## Debug Scripts

### 🔍 **debug_mcp_system.py** - MCP System Testing
Tests the complete MCP pipeline with detailed reports.

```bash
# Basic usage
uv run python scripts/debug_mcp_system.py

# Custom search query
uv run python scripts/debug_mcp_system.py --search-query "integration testing"
```

### 📄 **debug_pdf_indexing.py** - PDF Processing Analysis
Analyzes PDF-to-markdown conversion and chunking with full content review.

```bash
# Basic usage
uv run python scripts/debug_pdf_indexing.py

# Custom PDF file
uv run python scripts/debug_pdf_indexing.py --pdf-path path/to/document.pdf
```

## Output

Debug reports are generated in `.debug/` folder with timestamped HTML files containing:
- Complete operation logs and timing
- Full chunk content with expandable sections
- Visual chunk boundaries and statistics
- Search results and metadata analysis

## Options

Debug scripts support:
- `--output-dir` - Custom output directory (default: `.debug/`)
- `--help` - Full options and usage

All scripts should be run from the project root directory.
