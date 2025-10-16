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

### 🧹 **run_vulture.py**
Runs the Vulture dead-code scanner with project defaults (`pycontextify/` with `tests/` excluded).

```bash
# Scan the main package with default settings
python scripts/run_vulture.py

# Include additional directories and tweak confidence threshold
python scripts/run_vulture.py pycontextify tests --min-confidence 70

# Forward advanced options directly to Vulture
python scripts/run_vulture.py -- --ignore-names "test_*"
```

### 🔢 **bump_version.py**
Increments the semantic version in `pyproject.toml` (and syncs the fallback in `pycontextify/__init__.py`).

```bash
# Bump patch version (default)
python scripts/bump_version.py

# Bump minor version
python scripts/bump_version.py minor

# Preview without writing changes
python scripts/bump_version.py --dry-run
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

### 🧭 **inspect_faiss_index.py** - FAISS Index Explorer
Loads a `.faiss` index file, prints its schema, and previews stored vectors.

```bash
# Inspect an index file from the project root
uv run python scripts/inspect_faiss_index.py path/to/index.faiss

# Show more vectors without printing stored IDs
uv run python scripts/inspect_faiss_index.py path/to/index.faiss --max-vectors 25 --no-ids
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
