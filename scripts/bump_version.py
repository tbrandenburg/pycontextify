"""Utility to bump the project version in pyproject.toml and related files."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"
INIT_FILE = PROJECT_ROOT / "pycontextify" / "__init__.py"


class VersionBumpError(RuntimeError):
    """Raised when the version bump cannot be completed."""


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        try:
            major, minor, patch = (int(part) for part in version_str.split("."))
        except ValueError as exc:  # pragma: no cover - defensive, depends on data file
            raise VersionBumpError(f"Invalid semantic version: {version_str}") from exc
        return cls(major=major, minor=minor, patch=patch)

    def bump(self, part: str) -> "Version":
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise VersionBumpError(f"Unknown part to bump: {part}")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.major}.{self.minor}.{self.patch}"


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        raise VersionBumpError(f"Unable to read {path}") from exc


def _write_text(path: pathlib.Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        raise VersionBumpError(f"Unable to write {path}") from exc


def _extract_version(content: str) -> str:
    pattern = re.compile(r'^(version\s*=\s*")(?P<version>[^"\n]+)(")', re.MULTILINE)
    match = pattern.search(content)
    if not match:
        raise VersionBumpError("Could not locate version field in pyproject.toml")
    return match.group("version")


def _update_pyproject(content: str, new_version: str) -> str:
    pattern = re.compile(r'^(version\s*=\s*")([^"\n]+)(")', re.MULTILINE)
    if not pattern.search(content):
        raise VersionBumpError("Could not locate version field in pyproject.toml")
    return pattern.sub(rf'\g<1>{new_version}\3', content, count=1)


def _update_init(content: str, new_version: str) -> str:
    pattern = re.compile(r'(__version__\s*=\s*")(?P<version>[^"\n]+)(")')
    if not pattern.search(content):
        return content  # No fallback version present; nothing to update
    return pattern.sub(rf'\g<1>{new_version}\3', content, count=1)


def bump_version(part: str, *, dry_run: bool = False) -> Version:
    pyproject_content = _read_text(PYPROJECT_FILE)
    current_version = Version.parse(_extract_version(pyproject_content))
    new_version = current_version.bump(part)

    updated_pyproject = _update_pyproject(pyproject_content, str(new_version))
    init_content = _read_text(INIT_FILE)
    updated_init = _update_init(init_content, str(new_version))

    if not dry_run:
        _write_text(PYPROJECT_FILE, updated_pyproject)
        _write_text(INIT_FILE, updated_init)

    return new_version


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bump the project version")
    parser.add_argument(
        "part",
        choices=("major", "minor", "patch"),
        nargs="?",
        default="patch",
        help="Which part of the semantic version to increment (default: patch)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the new version without modifying files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        new_version = bump_version(args.part, dry_run=args.dry_run)
    except VersionBumpError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"Dry run successful. New version would be: {new_version}")
    else:
        print(f"Version bumped to {new_version}")
    return 0


if __name__ == "__main__":  # pragma: no cover - command line entry point
    sys.exit(main())
