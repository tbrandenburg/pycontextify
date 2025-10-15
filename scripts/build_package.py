#!/usr/bin/env python3
"""Build wheel/sdist artifacts and validate metadata before publishing."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def _check_tool(module: str, install_hint: str) -> None:
    """Ensure the required build tooling is available."""

    try:
        __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            f"Missing required dependency '{module}'. Install it with `{install_hint}` before running this script."
        ) from exc


def _run(command: list[str]) -> None:
    """Run a subprocess, streaming output and raising on failure."""

    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


def _clean_directories() -> None:
    """Remove build artefacts from previous runs."""

    for path in (DIST_DIR, BUILD_DIR):
        if path.exists():
            print(f"Removing {path} ...")
            shutil.rmtree(path)


def main() -> None:
    """Build and validate distribution artefacts."""

    _check_tool("build", "pip install build")
    _check_tool("twine", "pip install twine")

    _clean_directories()

    print("\nBuilding wheel and source distribution...\n")
    _run([sys.executable, "-m", "build"])

    built_files = sorted(DIST_DIR.glob("*"))
    if not built_files:
        raise SystemExit(
            "No files were produced in dist/. Ensure the build step completed successfully."
        )

    print("\nRunning twine check...\n")
    _run([sys.executable, "-m", "twine", "check", *(str(path) for path in built_files)])

    print("\n✅ Build artefacts are ready in dist/ and metadata validation passed.")


if __name__ == "__main__":
    main()
