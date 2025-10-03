"""System-level tests for indexing the pycontextify codebase via MCP."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from fastmcp import Client

from pycontextify import mcp_server
from pycontextify.mcp_server import initialize_manager, reset_manager


def _unwrap_structured(payload: Any) -> Any:
    """Extract useful data from a FastMCP structured payload."""

    if isinstance(payload, dict) and "result" in payload:
        return payload["result"]
    return payload


def _parse_text_block(item: Any) -> list[dict[str, Any]]:
    if hasattr(item, "text"):
        try:
            parsed = json.loads(item.text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            return []
    return []


def _resolve_search_payload(result: Any, expected_format: str) -> list[dict[str, Any]]:
    """Normalize search tool output into a list of dictionaries."""

    data = _unwrap_structured(result)

    def _format_text_entry(text: str) -> dict[str, Any]:
        return {
            "text": text,
            "display_format": expected_format,
        }

    if isinstance(data, list):
        if data and hasattr(data[0], "text"):
            normalized: list[dict[str, Any]] = []
            fallbacks: list[str] = []
            for item in data:
                parsed = _parse_text_block(item)
                if parsed:
                    normalized.extend(parsed)
                else:
                    text = getattr(item, "text", "")
                    if text:
                        fallbacks.append(text)
            if normalized:
                return normalized
            if fallbacks:
                return [_format_text_entry(text) for text in fallbacks]
        return data  # Already a list of dicts

    if isinstance(data, dict):
        return [data]

    if isinstance(data, str):
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            if data.strip():
                return [_format_text_entry(data)]

    if isinstance(result, Iterable):
        normalized: list[dict[str, Any]] = []
        fallbacks: list[str] = []
        for item in result:
            parsed = _parse_text_block(item)
            if parsed:
                normalized.extend(parsed)
            else:
                text = getattr(item, "text", "") if hasattr(item, "text") else str(item)
                if text:
                    fallbacks.append(text)
        if normalized:
            return normalized
        if fallbacks:
            return [_format_text_entry(text) for text in fallbacks]

    raise AssertionError("Unable to interpret search results from MCP response")


@pytest.mark.parametrize(
    "display_format",
    ["structured", "readable", "summary", "invalid"],
)
def test_mcp_server_indexes_codebase_and_searches(tmp_path, display_format) -> None:
    """Index the pycontextify package through the live MCP server and search it."""

    async def _run() -> None:
        reset_manager()

        overrides = {
            "index_dir": str(tmp_path),
            "auto_persist": False,
            "auto_load": False,
            "embedding_model": "all-MiniLM-L6-v2",
        }

        initialize_manager(overrides)

        try:
            async with Client(mcp_server.mcp) as client:
                status_before = await client.call_tool("status", {})
                before_data = _unwrap_structured(status_before.structured_content)
                assert before_data["metadata"]["total_chunks"] == 0

                project_root = Path(__file__).resolve().parents[2]
                code_dir = project_root / "pycontextify"

                index_result = await client.call_tool(
                    "index_code", {"path": str(code_dir)}
                )
                index_data = _unwrap_structured(index_result.structured_content)
                assert index_data["chunks_added"] > 0
                assert index_data["files_processed"] > 0

                status_after = await client.call_tool("status", {})
                after_data = _unwrap_structured(status_after.structured_content)
                assert (
                    after_data["metadata"]["total_chunks"]
                    >= index_data["chunks_added"]
                )

                search_kwargs = {
                    "query": "IndexManager search pipeline",
                    "top_k": 5,
                }
                if display_format != "invalid":
                    search_kwargs["display_format"] = display_format
                else:
                    search_kwargs["display_format"] = "totally-unknown"

                search_result = await client.call_tool("search", search_kwargs)
                format_key = (
                    search_kwargs["display_format"]
                    if search_kwargs.get("display_format") != "totally-unknown"
                    else "readable"
                )
                search_data = _resolve_search_payload(
                    search_result.structured_content or search_result.content,
                    format_key,
                )

                assert isinstance(search_data, list)
                assert search_data, "search returned no results"

                first_result = search_data[0]

                if format_key == "structured":
                    assert "chunk_text" in first_result
                    assert "source_path" in first_result
                    assert first_result["source_path"].endswith(".py")
                    assert "pycontextify" in Path(first_result["source_path"]).parts
                else:
                    assert "text" in first_result
                    assert format_key in first_result.get("display_format", "")
                    assert "IndexManager" in first_result["text"]
        finally:
            reset_manager()

    asyncio.run(_run())
