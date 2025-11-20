"""CLI utility for inspecting FAISS index files.

This script is intended to be executed from the project root using:

    uv run python scripts/inspect_faiss_index.py <path-to-index.faiss>

It prints summary information about the index (schema) and optionally dumps a
preview of the stored vectors and their IDs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Inspect a FAISS index file and print its schema and contents.",
    )
    parser.add_argument(
        "index_path",
        type=Path,
        help="Path to the .faiss file to inspect.",
    )
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=10,
        help="Maximum number of vectors to display (default: 10).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places when printing vector values (default: 6).",
    )
    parser.add_argument(
        "--no-ids",
        action="store_true",
        help="Do not attempt to load explicit vector IDs even if they are stored in the index.",
    )
    return parser


def import_faiss():  # type: ignore[no-untyped-def]
    """Safely import the FAISS Python bindings with a friendly error message."""
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via CLI
        msg = (
            "FAISS is required for this script. Install the 'faiss-cpu' optional "
            "dependency group or run via 'uv run' to auto-resolve dependencies."
        )
        raise SystemExit(msg) from exc

    return faiss


def format_metric(metric_type: int, faiss_module) -> str:
    """Return a symbolic name for the FAISS metric type."""
    metric_map = {
        getattr(faiss_module, name): name
        for name in dir(faiss_module)
        if name.startswith("METRIC_")
    }
    return metric_map.get(metric_type, f"UNKNOWN({metric_type})")


def unwrap_index(index, faiss_module):  # type: ignore[no-untyped-def]
    """Unwrap known FAISS index decorators (e.g. ID maps, pre-transform wrappers).

    Returns a tuple of `(base_index, wrappers)` where `wrappers` is an ordered
    list describing the wrappers that were removed while drilling down to the
    base index instance.
    """
    wrappers: List[str] = []
    current = index

    # Helper to attempt a downcast function and gracefully handle failures.
    def try_downcast(name: str, obj):  # type: ignore[no-untyped-def]
        downcast_fn = getattr(faiss_module, f"downcast_{name}", None)
        if downcast_fn is None:
            return None
        try:
            return downcast_fn(obj)
        except Exception:  # pragma: no cover - depends on FAISS features
            return None

    while True:
        id_map = try_downcast("IndexIDMap", current)
        if id_map is not None:
            wrappers.append("IndexIDMap")
            current = id_map.index
            continue

        pre_transform = try_downcast("IndexPreTransform", current)
        if pre_transform is not None:
            wrappers.append("IndexPreTransform")
            current = pre_transform.index
            continue

        # No known wrappers remaining.
        break

    return current, wrappers


def gather_schema(index_path: Path, index, faiss_module) -> str:  # type: ignore[no-untyped-def]
    """Compile human-readable schema information for the FAISS index."""
    base_index, wrappers = unwrap_index(index, faiss_module)
    metrics = format_metric(getattr(base_index, "metric_type", -1), faiss_module)
    is_trained_attr = getattr(base_index, "is_trained", "unknown")
    if callable(is_trained_attr):  # pragma: no cover - defensive guard
        try:
            is_trained_value = is_trained_attr()
        except Exception:  # pragma: no cover - depends on FAISS bindings
            is_trained_value = "error"
    else:
        is_trained_value = is_trained_attr

    schema_lines = [
        f"File: {index_path}",
        f"Index type: {type(index).__name__}",
        f"Base index type: {type(base_index).__name__}",
        f"Vector dimension: {getattr(base_index, 'd', 'unknown')}",
        f"Metric: {metrics}",
        f"Is trained: {is_trained_value}",
        f"Total vectors: {getattr(index, 'ntotal', 'unknown')}",
    ]

    if wrappers:
        schema_lines.append(f"Wrappers: {' -> '.join(wrappers)}")

    # Include IVF-specific parameters if available.
    nlist = getattr(base_index, "nlist", None)
    if nlist is not None:
        schema_lines.append(f"Inverted lists (nlist): {nlist}")

    nprobe = getattr(base_index, "nprobe", None)
    if nprobe is not None:
        schema_lines.append(f"Search probes (nprobe): {nprobe}")

    return "\n".join(schema_lines)


def extract_ids(index, faiss_module, count: int) -> Sequence[int]:  # type: ignore[no-untyped-def]
    """Return stored IDs if available, otherwise fall back to positional IDs."""
    downcast = getattr(faiss_module, "downcast_IndexIDMap", None)
    if downcast is None:
        return list(range(count))

    try:
        id_map = downcast(index)
    except Exception:
        return list(range(count))

    if not hasattr(id_map, "id_map"):
        return list(range(count))

    try:
        raw_ids = faiss_module.vector_long_to_array(id_map.id_map)
        ids = raw_ids.tolist()
        if len(ids) >= count:
            return ids[:count]
        # Fallback if the ID map is shorter than expected.
        ids.extend(list(range(len(ids), count)))
        return ids
    except Exception:  # pragma: no cover - depends on FAISS bindings
        return list(range(count))


def reconstruct_vectors(index, count: int) -> Tuple[List[np.ndarray], List[str]]:
    """Reconstruct up to ``count`` vectors from the index."""
    vectors: List[np.ndarray] = []
    errors: List[str] = []
    for i in range(count):
        try:
            vector = index.reconstruct(i)
        except Exception as exc:  # pragma: no cover - depends on FAISS support
            errors.append(f"Failed to reconstruct vector {i}: {exc}")
            break
        else:
            vectors.append(np.asarray(vector, dtype=np.float32))
    return vectors, errors


def format_vector(vector: np.ndarray, precision: int, max_elements: int = 16) -> str:
    """Format a vector for human-readable output."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.flatten()

    if arr.size > max_elements:
        head = ", ".join(f"{value:.{precision}f}" for value in arr[:max_elements])
        return f"[{head}, ...] (len={arr.size})"

    return "[" + ", ".join(f"{value:.{precision}f}" for value in arr) + "]"


def render_vectors(
    index,
    faiss_module,
    max_vectors: int,
    precision: int,
    include_ids: bool,
) -> str:  # type: ignore[no-untyped-def]
    """Generate formatted text for the stored vectors."""
    total_vectors = getattr(index, "ntotal", 0)
    if total_vectors == 0:
        return "(Index contains no vectors.)"

    limit = min(max_vectors, total_vectors)
    vectors, errors = reconstruct_vectors(index, limit)

    if not vectors:
        if errors:
            return "\n".join(errors)
        return "No vectors could be reconstructed from the index."

    if include_ids:
        ids = extract_ids(index, faiss_module, len(vectors))
    else:
        ids = list(range(len(vectors)))

    lines = [f"Showing {len(vectors)} of {total_vectors} vectors:"]
    for idx, (vector_id, vector) in enumerate(zip(ids, vectors)):
        formatted_vector = format_vector(vector, precision)
        lines.append(f"{idx:>4}: FAISS ID={vector_id}  {formatted_vector}")

    if limit < total_vectors:
        lines.append(f"... (omitted {total_vectors - limit} additional vectors)")

    if errors:
        lines.extend(errors)

    return "\n".join(lines)


def validate_args(args: argparse.Namespace) -> None:
    """Perform basic argument validation."""
    if not args.index_path.exists():
        raise SystemExit(f"Index file not found: {args.index_path}")

    if not args.index_path.is_file():
        raise SystemExit(f"Path is not a file: {args.index_path}")

    if args.max_vectors < 0:
        raise SystemExit("--max-vectors must be non-negative")

    if args.precision < 0:
        raise SystemExit("--precision must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    """Entrypoint for the CLI script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    faiss_module = import_faiss()

    index = faiss_module.read_index(str(args.index_path))
    schema_text = gather_schema(args.index_path, index, faiss_module)
    vector_text = render_vectors(
        index,
        faiss_module,
        max_vectors=args.max_vectors,
        precision=args.precision,
        include_ids=not args.no_ids,
    )

    print("FAISS Index Schema")
    print("===================")
    print(schema_text)
    print()
    print("Vector Contents")
    print("================")
    print(vector_text)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
