"""System test for MCP server indexing and searching a webpage."""

import asyncio
import json
import socket
import threading
from contextlib import suppress
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest
from fastmcp.client import Client

from pycontextify import mcp_server


class QuietHandler(SimpleHTTPRequestHandler):
    """HTTP handler without console noise."""

    def log_message(self, format, *args):  # noqa: D401, A003 - signature fixed by base class
        """Suppress default logging."""
        pass


async def _wait_for_server(port: int, *, timeout: float = 10.0) -> None:
    """Wait until the MCP server is ready to accept connections."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            client = Client(f"http://127.0.0.1:{port}/mcp")
            async with client:
                await client.list_tools()
            return
        except Exception:
            if asyncio.get_event_loop().time() >= deadline:
                raise
            await asyncio.sleep(0.2)


def _start_local_http_server(root: Path) -> tuple[ThreadingHTTPServer, int]:
    """Start a simple HTTP server that serves files from ``root``."""
    handler = partial(QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _reserve_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_mcp_server_indexes_webpage_and_supports_search(monkeypatch, tmp_path):
    """Index a served HTML page via the MCP server and validate search results."""

    async def _run_test() -> None:
        web_root = tmp_path / "web"
        web_root.mkdir()
        html_path = web_root / "index.html"
        html_path.write_text(
            """
            <html><body>
            <h1>Local Example Knowledge Base</h1>
            <p>This local documentation page demonstrates semantic and keyword search capabilities within PyContextify.</p>
            <h2>Semantic Search Highlights</h2>
            <p>PyContextify combines embeddings with keyword intelligence to surface highly relevant results.</p>
            <p>The system excels at relationship-aware discovery, linking related concepts across documentation sections.</p>
            <h2>Usage Notes</h2>
            <p>Use the search interface to locate phrases like "semantic search capabilities" and confirm retrieval quality.</p>
            </body></html>
            """.strip(),
            encoding="utf-8",
        )

        httpd, web_port = _start_local_http_server(web_root)

        index_dir = tmp_path / "index"
        monkeypatch.setenv("PYCONTEXTIFY_INDEX_DIR", str(index_dir))
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_PERSIST", "false")
        monkeypatch.setenv("PYCONTEXTIFY_AUTO_LOAD", "false")

        mcp_port = _reserve_free_port()
        server_task = asyncio.create_task(
            mcp_server.mcp.run_async(
                transport="http",
                host="127.0.0.1",
                port=mcp_port,
                show_banner=False,
                log_level="error",
            )
        )

        try:
            await _wait_for_server(mcp_port)

            client = Client(f"http://127.0.0.1:{mcp_port}/mcp")
            async with client:
                index_result = await client.call_tool(
                    "index_webpage",
                    {
                        "url": f"http://127.0.0.1:{web_port}/index.html",
                        "recursive": False,
                        "max_depth": 1,
                    },
                )
                assert not index_result.is_error, index_result.content
                index_data = index_result.data
                assert isinstance(index_data, dict)
                assert index_data.get("chunks_added", 0) >= 1

                status_result = await client.call_tool("status", {})
                status_data = status_result.data
                assert status_data["metadata"]["total_chunks"] >= index_data["chunks_added"]

                def _collect_text_parts(tool_response):
                    return "".join(
                        getattr(part, "text", "")
                        for part in getattr(tool_response, "content", [])
                        if getattr(part, "text", "")
                    )

                for display_format in ("structured", "readable", "summary"):
                    search_result = await client.call_tool(
                        "search",
                        {
                            "query": "semantic search capabilities",
                            "top_k": 3,
                            "display_format": display_format,
                        },
                    )
                    assert not search_result.is_error, search_result.content
                    payload_text = _collect_text_parts(search_result)
                    assert payload_text, "search should return textual payload"

                    if display_format == "structured":
                        results_payload = json.loads(payload_text)
                        assert isinstance(results_payload, list) and results_payload
                        assert any(
                            "semantic and keyword search capabilities"
                            in entry["chunk_text"]
                            for entry in results_payload
                        )
                    else:
                        assert "semantic search capabilities" in payload_text
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task
            mcp_server.reset_manager()

    asyncio.run(_run_test())
