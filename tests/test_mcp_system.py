"""System-level tests for the running PyContextify MCP server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Iterable

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


def _resolve_search_payload(result: Any) -> list[dict[str, Any]]:
    """Normalize search tool output into a list of dictionaries."""

    data = _unwrap_structured(result)

    if isinstance(data, list):
        if data and hasattr(data[0], "text"):
            normalized: list[dict[str, Any]] = []
            for item in data:
                normalized.extend(_parse_text_block(item))
            if normalized:
                return normalized
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
            pass

    if isinstance(result, Iterable):
        normalized: list[dict[str, Any]] = []
        for item in result:
            normalized.extend(_parse_text_block(item))
        if normalized:
            return normalized

    raise AssertionError("Unable to interpret search results from MCP response")


def test_mcp_server_indexes_codebase_and_searches(tmp_path) -> None:
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

                search_result = await client.call_tool(
                    "search",
                    {
                        "query": "IndexManager search pipeline",
                        "top_k": 5,
                        "display_format": "structured",
                    },
                )
                search_data = _resolve_search_payload(
                    search_result.structured_content or search_result.content
                )

                assert isinstance(search_data, list)
                assert search_data, "search returned no results"

                first_result = search_data[0]
                assert "chunk_text" in first_result
                assert "source_path" in first_result
                assert first_result["source_path"].endswith(".py")
                assert "pycontextify" in Path(first_result["source_path"]).parts
        finally:
            reset_manager()

    asyncio.run(_run())
