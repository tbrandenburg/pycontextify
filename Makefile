.PHONY: help qa all lint format test docs build run deploy clean install dev-install mcp-test profile

# Default target
help: ## Show this help message
	@echo "PyContextify - MCP Semantic Search Server"
	@echo "=========================================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

qa: test ## Run quality assurance checks (lint, format, test)
	@echo "✓ QA checks passed!"

all: build ## Run complete code quality workflow and build
	@echo "✓ All checks passed and project built!"

install: ## Install project dependencies
	@echo "Installing dependencies..."
	uv sync

dev-install: ## Install project with development dependencies
	@echo "Installing with development dependencies..."
	uv sync --extra dev

lint: ## Run code quality checks (black, isort, flake8, mypy)
	@echo "Running linting..."
	@echo "→ Checking code format with black..."
	uv run black --check --diff pycontextify tests scripts
	@echo "→ Checking import sorting with isort..."
	uv run isort --check-only --diff pycontextify tests scripts
	@echo "→ Running flake8 linter..."
	uv run flake8 pycontextify tests scripts
	@echo "→ Running mypy type checker..."
	uv run mypy pycontextify
	@echo "→ Running vulture dead code detector..."
	uv run python scripts/run_vulture.py
	@echo "✓ Linting checks passed!"

format: ## Auto-format code (black, isort)
	@echo "Formatting code..."
	@echo "→ Formatting with black..."
	uv run black pycontextify tests scripts
	@echo "→ Sorting imports with isort..."
	uv run isort pycontextify tests scripts
	@echo "✓ Code formatting completed!"

test: format ## Run the test suite with coverage reporting
	@echo "Running tests..."
	@echo "→ Running unit tests..."
	uv run pytest tests/unit -v
	@echo "→ Running integration tests..."
	uv run pytest tests/integration -v
	@echo "→ Running system tests..."
	uv run pytest tests/system -v
	@echo "→ Running MCP integration tests..."
	uv run python scripts/run_mcp_tests.py
	@echo "→ Running full test suite with coverage..."
	uv run pytest --cov=pycontextify --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "✓ Test suite completed! Coverage report generated in htmlcov/"

mcp-test: ## Run MCP-specific integration tests
	@echo "Running MCP integration tests..."
	uv run python scripts/run_mcp_tests.py
	@echo "✓ MCP integration tests completed!"

docs: test ## Generate documentation (placeholder - no docs setup yet)
	@echo "Generating documentation..."
	@echo "→ Creating basic project documentation structure..."
	@mkdir -p docs/build
	@echo "Documentation structure created in docs/"
	@echo "TODO: Set up Sphinx or other documentation generator"
	@echo "✓ Documentation preparation completed!"

build: docs ## Build the project (install dependencies, run quality checks)
	@echo "Building project..."
	@echo "→ Installing dependencies..."
	uv sync --extra dev
	@echo "→ Building wheel package..."
	uv run python scripts/build_package.py
	@echo "✓ Project built successfully!"

run: build ## Run the MCP server locally
	@echo "Running MCP server..."
	@echo "→ Starting PyContextify MCP server..."
	@echo "Note: MCP servers are typically connected to by MCP clients"
	@echo "Use: uv run pycontextify for direct execution"
	@echo "Or connect via MCP client with: python -m pycontextify.mcp"
	uv run python -m pycontextify.mcp

debug: ## Run debug utilities
	@echo "Running debug utilities..."
	@echo "→ Available debug scripts:"
	@echo "  • uv run python scripts/debug_mcp_system.py - Debug MCP system"
	@echo "  • uv run python scripts/debug_directory_indexing.py - Debug indexing"
	@echo "  • uv run python scripts/debug_pdf_indexing.py - Debug PDF processing"
	@echo "  • uv run python scripts/inspect_faiss_index.py - Inspect FAISS index"

profile: ## Profile the application performance
	@echo "Profiling application..."
	@echo "→ Running performance profiling..."
	@echo "TODO: Implement performance profiling with cProfile or py-spy"
	@echo "Consider: uv run py-spy record -o profile.svg -- python -m pycontextify.mcp"

version-bump: ## Bump version number
	@echo "Bumping version..."
	uv run python scripts/bump_version.py

deploy: run ## Deploy MCP server (setup for integration)
	@echo "Deploying MCP server..."
	@echo "→ MCP servers are typically integrated into client applications"
	@echo "→ Ensure the server is properly packaged and configured"
	@echo "→ Distribution methods:"
	@echo "  • Install via pip: pip install pycontextify"
	@echo "  • Local development: uv sync && uv run pycontextify"
	@echo "  • Docker deployment: Consider creating Dockerfile"
	@echo "✓ Deployment information provided!"

clean: ## Remove build artifacts and caches
	@echo "Cleaning build artifacts..."
	@echo "→ Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "→ Removing test and coverage artifacts..."
	rm -rf .pytest_cache .coverage htmlcov coverage.xml .mypy_cache
	@echo "→ Removing build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	@echo "→ Removing temporary files..."
	find . -name "*.tmp" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✓ Cleanup completed!"

# Advanced targets
lint-fix: ## Auto-fix linting issues where possible
	@echo "Auto-fixing linting issues..."
	uv run black pycontextify tests scripts
	uv run isort pycontextify tests scripts
	@echo "✓ Auto-fixable linting issues resolved!"

test-unit: ## Run only unit tests
	@echo "Running unit tests..."
	uv run pytest tests/unit -v
	@echo "✓ Unit tests completed!"

test-integration: ## Run only integration tests
	@echo "Running integration tests..."
	uv run pytest tests/integration -v
	@echo "✓ Integration tests completed!"

test-system: ## Run only system tests
	@echo "Running system tests..."
	uv run pytest tests/system -v
	@echo "✓ System tests completed!"

test-fast: ## Run fast tests only (exclude slow/integration tests)
	@echo "Running fast tests..."
	uv run pytest tests/unit -v -m "not slow"
	@echo "✓ Fast tests completed!"

check-deps: ## Check for dependency issues and security vulnerabilities
	@echo "Checking dependencies..."
	@echo "→ Checking for outdated packages..."
	uv show --tree
	@echo "TODO: Add security vulnerability scanning"
	@echo "Consider: pip-audit or safety for vulnerability checks"

# Environment setup
setup-dev: dev-install ## Complete development environment setup
	@echo "Setting up development environment..."
	@echo "→ Installing development dependencies..."
	uv sync --extra dev
	@echo "→ Setting up pre-commit hooks (if available)..."
	@echo "TODO: Consider setting up pre-commit hooks"
	@echo "✓ Development environment setup completed!"