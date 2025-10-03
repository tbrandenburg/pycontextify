"""System-level MCP test for PDF indexing and search.

This test spins up the in-process FastMCP server, indexes a generated
PDF file, and verifies that semantic search returns the expected
results. It exercises the full MCP stack rather than calling the tool
implementations directly.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest
from fastmcp import Client

from pycontextify import mcp_server


def _install_sentence_transformers_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight sentence_transformers stub for the test."""

    stub_module = types.ModuleType("sentence_transformers")

    class _StubSentenceTransformer:
        """Minimal stub that emulates the required API surface."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.max_seq_length = 512
            self.device = kwargs.get("device", "cpu")

        def encode(self, texts: Any, *args: Any, **kwargs: Any) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = np.random.rand(len(texts), 384).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embeddings / norms

        def get_sentence_embedding_dimension(self) -> int:
            return 384

    stub_module.SentenceTransformer = _StubSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub_module)


def _create_test_pdf(pdf_path: Path) -> None:
    """Generate a multi-page PDF large enough to create several chunks."""

    import fitz  # Imported lazily to avoid test collection side-effects

    document = fitz.open()

    base_paragraph = (
        "PyContextify system test validates knowledge graph aware searches within PDFs. "
        "This section describes how embeddings, chunking, and metadata tracking operate together "
        "to build reliable semantic recall. It emphasizes deterministic behavior, reproducible "
        "results, and the importance of hybrid strategies that blend keyword recall with dense "
        "vector similarity."
    )

    paragraphs = []
    for section in range(1, 13):
        paragraphs.append(
            (
                f"Section {section} explores PyContextify orchestration layers, detailing how the "
                f"manager coordinates the vector store, metadata repository, and knowledge graph "
                f"features. The narrative highlights knowledge graph enrichment, pragmatic caching, "
                f"and resilience strategies for indexing large document batches. "
                f"{base_paragraph}"
            )
        )

    # Group paragraphs into batches so text flows across multiple pages.
    for start in range(0, len(paragraphs), 4):
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
            "\n\n".join(paragraphs[start : start + 4]),
            fontsize=11,
            lineheight=1.2,
        )

    document.save(pdf_path)
    document.close()


def _extract_structured_results(search_result: Any) -> List[dict[str, Any]]:
    """Convert an MCP search response into structured JSON."""

    if getattr(search_result, "structured_content", None):
        return list(search_result.structured_content)  # type: ignore[misc]

    blocks: List[str] = [
        block.text
        for block in getattr(search_result, "content", [])
        if hasattr(block, "text") and block.text
    ]
    combined = "".join(blocks).strip()
    return json.loads(combined) if combined else []


def _collect_text_output(search_result: Any) -> str:
    """Gather textual content from an MCP search response."""

    if isinstance(getattr(search_result, "data", None), str):
        return str(search_result.data)

    blocks: List[str] = [
        block.text
        for block in getattr(search_result, "content", [])
        if hasattr(block, "text") and block.text
    ]

    return "\n".join(blocks).strip()


@pytest.mark.no_mock_st
def test_mcp_server_indexes_pdf_and_supports_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end verification of PDF indexing and search via MCP."""

    _install_sentence_transformers_stub(monkeypatch)

    index_dir = tmp_path / "index"
    monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
    monkeypatch.setenv("PYCONTEXTIFY_AUTO_LOAD", "false")

    pdf_path = tmp_path / "system_test.pdf"
    _create_test_pdf(pdf_path)

    mcp_server.reset_manager()

    client = Client(mcp_server.mcp)

    async def _run_flow() -> None:
        try:
            async with client:
                index_result = await client.call_tool(
                    "index_document", {"path": str(pdf_path)}
                )
                assert not index_result.is_error

                payload = index_result.data or index_result.structured_content
                assert isinstance(payload, dict)
                assert payload.get("chunks_added", 0) >= 3
                assert Path(payload.get("file_processed", "")).name == pdf_path.name

                async def _search(display_format: str):
                    result = await client.call_tool(
                        "search",
                        {
                            "query": "knowledge graph",
                            "top_k": 5,
                            "display_format": display_format,
                        },
                    )
                    assert not result.is_error
                    return result

                structured_result = await _search("structured")

                results = _extract_structured_results(structured_result)
                assert results, "Expected at least one search result from indexed PDF"

                assert any(
                    Path(result.get("source_path", "")).name == pdf_path.name
                    for result in results
                )

                assert any(
                    "knowledge" in result.get("chunk_text", "").lower()
                    for result in results
                )

                scores = []
                for result in results:
                    score_value = result.get("similarity_score")
                    try:
                        scores.append(float(score_value))
                    except (TypeError, ValueError):
                        continue

                assert scores and max(scores) > 0, "Similarity scores should be positive"

                for display_format in ("readable", "summary"):
                    formatted_result = await _search(display_format)
                    formatted_text = _collect_text_output(formatted_result)
                    assert formatted_text, f"Expected non-empty {display_format} output"
                    assert pdf_path.name in formatted_text
                    assert "knowledge graph" in formatted_text.lower()
        finally:
            mcp_server.reset_manager()

    import asyncio

    asyncio.run(_run_flow())
