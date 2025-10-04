# PyContextify Release Guide

This document captures the steps required to publish a new version of PyContextify to PyPI (or TestPyPI).

## Pre-flight Checklist

1. **Decide on the version number**
   - Follow [Semantic Versioning](https://semver.org/)
   - Update the `version` field in `pyproject.toml`
2. **Update documentation**
   - Ensure `README.md` reflects new features or breaking changes
   - Add or revise release notes in `CHANGELOG.md`
3. **Verify dependencies**
   - Run `uv sync --extra dev` to install the `dev` extra (includes `build` and `twine`)
   - Review `pyproject.toml` for any dependency adjustments needed for the release

## Quality Gates

Run the relevant checks from the project root:

```bash
# Core test suites
uv run python scripts/run_mcp_tests.py
uv run python -m pytest tests -v

# Static analysis (optional but recommended)
uv run black --check .
uv run isort --check-only .
uv run flake8
uv run mypy pycontextify
```

If runtime models or external services are unavailable, execute the smoke tests instead:

```bash
uv run python scripts/run_mcp_tests.py --smoke
```

## Build & Validate Artifacts

Use the helper script to produce fresh distributions and validate metadata:

```bash
python scripts/build_package.py
```

This command:

1. Removes existing `build/` and `dist/` directories
2. Builds both the wheel and source distribution via `python -m build`
3. Runs `python -m twine check` against the produced archives

## Upload

1. Optionally publish to TestPyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```
2. Publish to PyPI when satisfied:
   ```bash
   twine upload dist/*
   ```
3. Tag the release and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

## Post-Release

- Announce the release (blog, social, etc.)
- Monitor download stats and error reports
- Create follow-up issues for any known limitations or future improvements
