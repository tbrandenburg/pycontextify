"""Black-box system test covering all MCP tools via FastMCP STDIO transport."""

from __future__ import annotations

import asyncio
import json
import os
import site
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


def _normalize_payload(value: Any) -> Any:
    """Convert tool responses to plain Python data structures for assertions."""

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if isinstance(value, dict):
        return {key: _normalize_payload(val) for key, val in value.items()}

    if isinstance(value, list):
        return [_normalize_payload(item) for item in value]

    return value


def _generate_sample_value(name: str, schema: Dict[str, Any]) -> Any:
    """Generate placeholder data from a JSON schema definition."""

    schema_type = schema.get("type")

    if schema_type == "string":
        return f"sample_{name}"
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 1.0
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [_generate_sample_value(f"{name}_item", item_schema)]
    if schema_type == "object":
        properties = schema.get("properties", {})
        return {
            prop_name: _generate_sample_value(prop_name, prop_schema)
            for prop_name, prop_schema in properties.items()
        }

    return f"sample_{name}" if schema_type else {}


def _build_arguments(
    tool_name: str,
    schema: Dict[str, Any],
    *,
    base_path: Path,
    tags: str,
) -> Dict[str, Any]:
    """Create representative arguments for a tool based on its schema."""

    if tool_name == "index_filebase":
        return {
            "base_path": str(base_path),
            "tags": tags,
            # Remove include filter to allow PDF files
        }

    if tool_name == "search":
        return {
            "query": "ASPICE process automotive spice",
            "top_k": 3,
            "display_format": "structured",
        }

    if tool_name == "reset_index":
        return {
            "remove_files": True,
            "confirm": True,
        }

    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        return {
            name: _generate_sample_value(name, details)
            for name, details in properties.items()
        }

    return {}


@pytest.mark.system
def test_mcp_server_end_to_end_stdio() -> None:
    """Launch the MCP server via uv and exercise every available tool."""

    asyncio.run(_run_stdio_system_flow())


async def _run_stdio_system_flow() -> None:
    """Execute the asynchronous MCP tool discovery and validation flow."""

    # Prepare working directories and use real PDF resource for indexing
    runtime_dir = Path(tempfile.mkdtemp(prefix="mcp_stdio_system_test_"))
    index_dir = runtime_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Use the real PDF resource file instead of creating temporary content
    repo_root = Path(__file__).resolve().parents[2]
    pdf_file_path = repo_root / "tests" / "resources" / "Automotive_SPICE_PAM_30.pdf"

    if not pdf_file_path.exists():
        raise FileNotFoundError(f"Test PDF resource not found: {pdf_file_path}")

    tag_name = "ASPICE,engineering,process"  # Use descriptive tags for the PDF content

    cli_script = runtime_dir / "pycontextify"
    cli_script.write_text(
        "#!/usr/bin/env python3\n"
        "from pycontextify.mcp import main\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    cli_script.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "FOO_BAR": "123",
            "PYCONTEXTIFY_INDEX_DIR": str(index_dir),
            "PYCONTEXTIFY_INDEX_NAME": "system_test_index",
            "PYCONTEXTIFY_AUTO_PERSIST": "false",
            "PYCONTEXTIFY_AUTO_LOAD": "false",
            "PYCONTEXTIFY_ENABLE_RELATIONSHIPS": "false",
            "PYCONTEXTIFY_USE_HYBRID_SEARCH": "false",
            "UV_NO_SYNC": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )

    existing_path = env.get("PYTHONPATH", "")
    pythonpath_entries = [str(repo_root)]
    if hasattr(site, "getsitepackages"):
        for pkg_path in site.getsitepackages():
            if pkg_path and pkg_path not in pythonpath_entries:
                pythonpath_entries.append(pkg_path)

    user_site = site.getusersitepackages()
    if user_site and user_site not in pythonpath_entries:
        pythonpath_entries.append(user_site)
    if existing_path:
        pythonpath_entries.append(existing_path)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    env["PATH"] = os.pathsep.join([str(runtime_dir), env.get("PATH", "")])

    transport = StdioTransport(
        command="uv",
        args=[
            "run",
            "pycontextify",
            "--verbose",  # Enable DEBUG logging
        ],
        env=env,
        cwd=str(repo_root),
        keep_alive=False,
    )

    client = Client(transport)
    call_log: list[dict[str, Any]] = []
    tool_names: list[str] = []

    try:
        async with client:
            tools = None
            # Allow the subprocess ample time to start up (including model downloads)
            for attempt in range(5):
                try:
                    await asyncio.sleep(2 * (attempt + 1))
                    tools = await client.list_tools()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    continue

            assert tools is not None, "Failed to retrieve tool list from MCP server"
            tool_names = [tool.name for tool in tools]
            print(f"Discovered MCP tools: {json.dumps(tool_names)}")

            name_to_tool = {tool.name: tool for tool in tools}
            planned_order = [
                name_to_tool.pop("status", None),
                name_to_tool.pop("index_filebase", None),
                name_to_tool.pop("discover", None),
                name_to_tool.pop("search", None),
                name_to_tool.pop("reset_index", None),
            ]
            # Append any remaining tools that were not part of the expected set
            planned_order.extend(name_to_tool.values())
            tools_to_run = [tool for tool in planned_order if tool is not None]

            assert tools_to_run, "No MCP tools were discovered for testing."

            for tool in tools_to_run:
                args = _build_arguments(
                    tool.name,
                    tool.inputSchema or {},
                    base_path=pdf_file_path,
                    tags=tag_name,
                )
                print(f"Calling tool '{tool.name}' with arguments: {json.dumps(args)}")
                result = await client.call_tool(tool.name, args)
                payload = (
                    result.data
                    if result.data is not None
                    else (
                        result.structured_content
                        if result.structured_content is not None
                        else [
                            getattr(block, "text", str(block))
                            for block in (result.content or [])
                        ]
                    )
                )
                normalized = _normalize_payload(payload)

                # If the environment cannot download the default embedding model,
                # fall back to skipping rather than failing the entire suite.
                error_message = ""
                if result.is_error:
                    error_message = getattr(result, "message", "") or str(payload)
                elif isinstance(normalized, dict):
                    error_message = str(normalized.get("error", ""))

                if (
                    "Failed to load model" in error_message
                    or "403 Forbidden" in error_message
                ):
                    pytest.skip(
                        "Default sentence-transformers model is unavailable "
                        "in this environment"
                    )

                call_log.append(
                    {
                        "tool": tool.name,
                        "arguments": args,
                        "payload": normalized,
                    }
                )
                print(
                    f"Tool '{tool.name}' returned payload: "
                    f"{json.dumps(normalized, default=str)}"
                )

                assert not result.is_error, f"Tool {tool.name} reported an error"

                if tool.name == "status":
                    assert isinstance(normalized, dict), "Status payload must be a dict"
                    assert (
                        normalized.get("mcp_server", {}).get("name") == "PyContextify"
                    )
                elif tool.name == "index_filebase":
                    assert normalized.get("tags_input") == tag_name
                    expected_tags = ["ASPICE", "engineering", "process"]
                    assert normalized.get("tags") == expected_tags
                    assert normalized.get("files_crawled", 0) >= 1
                    assert normalized.get("files_loaded", 0) >= 1, (
                        f"Expected files_loaded >= 1, "
                        f"got {normalized.get('files_loaded', 0)}"
                    )
                    assert normalized.get("chunks_created", 0) > 0, (
                        f"Expected chunks_created > 0, "
                        f"got {normalized.get('chunks_created', 0)}. "
                        f"PDF content should be processed into chunks."
                    )
                elif tool.name == "discover":
                    discovered_tags = normalized.get("tags", [])
                    # Check that all our expected tags are present
                    expected_tags = ["ASPICE", "engineering", "process"]
                    for expected_tag in expected_tags:
                        assert expected_tag in discovered_tags, (
                            f"Expected tag '{expected_tag}' not found in "
                            f"discovered tags: {discovered_tags}"
                        )
                    assert normalized.get("count", 0) >= 3
                elif tool.name == "search":
                    assert isinstance(normalized, list)
                    assert normalized, "Search should return at least one result"
                    first_result = normalized[0]
                    if isinstance(first_result, dict):
                        assert "chunk_text" in first_result
                        assert "similarity_score" in first_result
                elif tool.name == "reset_index":
                    assert normalized.get("success") is True
                    assert normalized.get("after_reset", {}).get("total_chunks") == 0

    finally:
        await transport.close()

    # Ensure each discovered tool produced a well-formed payload
    assert all("payload" in entry for entry in call_log)
    assert {entry["tool"] for entry in call_log} == set(tool_names)
