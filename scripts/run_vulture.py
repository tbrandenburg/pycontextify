#!/usr/bin/env python3
"""Run the Vulture dead-code scanner with project defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_PATHS = ("pycontextify",)
DEFAULT_EXCLUDES = ("tests",)
DEFAULT_MIN_CONFIDENCE = 80


def ensure_project_root() -> None:
    """Ensure the script is executed from the repository root."""

    if not Path("pyproject.toml").exists():
        raise SystemExit(
            "pyproject.toml not found. Please run this script from the repository root."
        )


def build_command(
    paths: Sequence[str],
    *,
    min_confidence: int,
    excludes: Sequence[str],
    extra_args: Sequence[str],
) -> list[str]:
    """Construct the command used to invoke Vulture."""

    command: list[str] = [sys.executable, "-m", "vulture", *paths]
    command.extend(["--min-confidence", str(min_confidence)])

    if excludes:
        command.extend(["--exclude", ",".join(excludes)])

    command.extend(extra_args)
    return command


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments and forward unknown parameters to Vulture."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Vulture with sensible project defaults. Unknown arguments are "
            "forwarded directly to Vulture so you can use advanced options when "
            "needed."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help=(
            "Target paths to analyze (defaults to the main pycontextify package). "
            "Multiple paths are supported."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum confidence threshold reported by Vulture (default: 80).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(DEFAULT_EXCLUDES),
        help=(
            "Directories or files to exclude from scanning. Accepts multiple "
            "values (default: tests)."
        ),
    )

    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the script."""

    ensure_project_root()
    args, extra = parse_args(list(argv) if argv is not None else sys.argv[1:])

    if not args.paths:
        raise SystemExit("At least one path must be provided to scan for dead code.")

    command = build_command(
        args.paths,
        min_confidence=args.min_confidence,
        excludes=args.exclude,
        extra_args=extra,
    )

    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
