#!/usr/bin/env python3
"""
PyContextify MCP System Debug Script

This script runs a comprehensive test of the MCP system, capturing detailed logs,
intermediate outputs, and found chunks. It generates a readable HTML report for
debugging and analysis purposes.

Usage:
    python debug_mcp_system.py [--output-dir PATH] [--search-query TEXT]

Examples:
    python debug_mcp_system.py
    python debug_mcp_system.py --output-dir ./debug_reports
    python debug_mcp_system.py --search-query "integration testing"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import site
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


class MCPDebugger:
    """MCP system debugger with comprehensive logging and reporting."""

    def __init__(
        self, output_dir: Path, search_query: str = "ASPICE integration testing"
    ):
        self.output_dir = output_dir
        self.search_query = search_query
        self.call_log: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def _normalize_payload(self, value: Any) -> Any:
        """Convert tool responses to plain Python data structures."""
        if is_dataclass(value):
            return asdict(value)

        if hasattr(value, "model_dump"):
            return value.model_dump()

        if isinstance(value, dict):
            return {key: self._normalize_payload(val) for key, val in value.items()}

        if isinstance(value, list):
            return [self._normalize_payload(item) for item in value]

        return value

    def _build_arguments(
        self, tool_name: str, schema: Dict[str, Any], *, pdf_file_path: Path
    ) -> Dict[str, Any]:
        """Create representative arguments for a tool based on its schema."""
        if tool_name == "index_filebase":
            return {
                "base_path": str(pdf_file_path),
                "tags": "automotive-spice,pam,process-assessment,documentation",
            }

        if tool_name == "search":
            return {
                "query": self.search_query,
                "top_k": 5,
                "display_format": "structured",
            }

        if tool_name == "reset_index":
            return {
                "remove_files": True,
                "confirm": True,
            }

        # Default argument generation for other tools
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            return {
                name: self._generate_sample_value(name, details)
                for name, details in properties.items()
            }

        return {}

    def _generate_sample_value(self, name: str, schema: Dict[str, Any]) -> Any:
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
            return [self._generate_sample_value(f"{name}_item", item_schema)]
        if schema_type == "object":
            properties = schema.get("properties", {})
            return {
                prop_name: self._generate_sample_value(prop_name, prop_schema)
                for prop_name, prop_schema in properties.items()
            }

        return f"sample_{name}" if schema_type else {}

    async def run_debug_session(self) -> None:
        """Execute the MCP debug session with comprehensive logging."""
        print(f"🔍 Starting MCP debug session at {self.start_time}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔎 Search query: '{self.search_query}'")

        # Prepare working directories and use real PDF resource for indexing
        runtime_dir = Path(tempfile.mkdtemp(prefix="mcp_debug_"))
        index_dir = runtime_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)

        # Use the real PDF resource file
        repo_root = (
            Path(__file__).resolve().parent.parent
        )  # Go up from scripts/ to project root
        pdf_file_path = (
            repo_root / "tests" / "resources" / "Automotive_SPICE_PAM_30.pdf"
        )

        if not pdf_file_path.exists():
            raise FileNotFoundError(f"Test PDF resource not found: {pdf_file_path}")

        print(f"📄 Using PDF: {pdf_file_path}")

        # Setup MCP transport with proper environment
        env = self._setup_environment(repo_root, runtime_dir, index_dir)
        transport = self._create_transport(repo_root, env)
        client = Client(transport)

        try:
            async with client:
                await self._run_mcp_operations(client, pdf_file_path)

        finally:
            await transport.close()

        # Generate the debug report
        self._generate_report()
        print(f"✅ Debug session completed. Report saved to: {self.output_dir}")

    def _setup_environment(
        self, repo_root: Path, runtime_dir: Path, index_dir: Path
    ) -> Dict[str, str]:
        """Setup environment variables for MCP server."""
        env = os.environ.copy()
        env.update(
            {
                "PYCONTEXTIFY_INDEX_DIR": str(index_dir),
                "PYCONTEXTIFY_INDEX_NAME": "debug_index",
                "PYCONTEXTIFY_AUTO_PERSIST": "false",
                "PYCONTEXTIFY_AUTO_LOAD": "false",
                "PYCONTEXTIFY_ENABLE_RELATIONSHIPS": "true",
                "PYCONTEXTIFY_USE_HYBRID_SEARCH": "true",
                "UV_NO_SYNC": "1",
                "UV_PYTHON_DOWNLOADS": "never",
            }
        )

        # Setup PYTHONPATH
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

        return env

    def _create_transport(self, repo_root: Path, env: Dict[str, str]) -> StdioTransport:
        """Create MCP transport with proper configuration."""
        return StdioTransport(
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

    async def _run_mcp_operations(self, client: Client, pdf_file_path: Path) -> None:
        """Execute MCP operations with comprehensive logging."""
        tools = None

        # Allow the subprocess ample time to start up (including model downloads)
        for attempt in range(5):
            try:
                await asyncio.sleep(2 * (attempt + 1))
                tools = await client.list_tools()
                break
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt == 4:
                    raise
                continue

        if tools is None:
            raise RuntimeError("Failed to retrieve tool list from MCP server")

        tool_names = [tool.name for tool in tools]
        print(f"🛠️  Discovered tools: {json.dumps(tool_names, indent=2)}")

        # Plan execution order
        name_to_tool = {tool.name: tool for tool in tools}
        planned_order = [
            name_to_tool.pop("status", None),
            name_to_tool.pop("index_filebase", None),
            name_to_tool.pop("discover", None),
            name_to_tool.pop("search", None),
            name_to_tool.pop("status", None),  # Status after operations
            name_to_tool.pop("reset_index", None),
        ]

        # Append any remaining tools
        planned_order.extend(name_to_tool.values())
        tools_to_run = [tool for tool in planned_order if tool is not None]

        print(f"📋 Planned execution order: {[t.name for t in tools_to_run]}")

        # Execute each tool with timing and logging
        for i, tool in enumerate(tools_to_run, 1):
            print(f"\n🔧 [{i}/{len(tools_to_run)}] Executing: {tool.name}")

            operation_start = time.time()
            args = self._build_arguments(
                tool.name, tool.inputSchema or {}, pdf_file_path=pdf_file_path
            )

            print(f"📤 Arguments: {json.dumps(args, indent=2, default=str)}")

            try:
                result = await client.call_tool(tool.name, args)
                operation_duration = time.time() - operation_start

                payload = self._extract_payload(result)
                normalized = self._normalize_payload(payload)

                # Check for errors
                if result.is_error:
                    error_message = getattr(result, "message", "") or str(payload)
                    print(f"❌ Error: {error_message}")
                else:
                    print(f"✅ Success ({operation_duration:.2f}s)")

                # Log the operation
                self.call_log.append(
                    {
                        "step": i,
                        "tool": tool.name,
                        "arguments": args,
                        "payload": normalized,
                        "duration_seconds": operation_duration,
                        "timestamp": datetime.now().isoformat(),
                        "is_error": result.is_error,
                    }
                )

                # Print condensed output
                self._print_condensed_output(tool.name, normalized)

            except Exception as e:
                operation_duration = time.time() - operation_start
                print(f"💥 Exception: {e}")
                self.call_log.append(
                    {
                        "step": i,
                        "tool": tool.name,
                        "arguments": args,
                        "payload": {"error": str(e)},
                        "duration_seconds": operation_duration,
                        "timestamp": datetime.now().isoformat(),
                        "is_error": True,
                    }
                )

    def _extract_payload(self, result) -> Any:
        """Extract payload from MCP result."""
        return (
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

    def _print_condensed_output(self, tool_name: str, payload: Any) -> None:
        """Print condensed output for immediate feedback."""
        if tool_name == "status":
            if isinstance(payload, dict):
                chunks = payload.get("metadata", {}).get("total_chunks", 0)
                vectors = payload.get("vector_store", {}).get("total_vectors", 0)
                print(f"📊 Status: {chunks} chunks, {vectors} vectors")

        elif tool_name == "index_filebase":
            if isinstance(payload, dict):
                files_loaded = payload.get("files_loaded", 0)
                chunks_created = payload.get("chunks_created", 0)
                duration = payload.get("duration_seconds", 0)
                print(
                    f"📚 Indexed: {files_loaded} files → {chunks_created} chunks ({duration:.1f}s)"
                )

        elif tool_name == "search":
            if isinstance(payload, list):
                print(f"🔍 Found {len(payload)} search results")
                for i, result in enumerate(payload[:3]):  # Show first 3
                    if isinstance(result, dict):
                        score = result.get("similarity_score", 0)
                        text_preview = result.get("chunk_text", "")[:100] + "..."
                        print(f"  {i+1}. Score: {score:.3f} | {text_preview}")

        elif tool_name == "discover":
            if isinstance(payload, dict):
                tags = payload.get("tags", [])
                count = payload.get("count", 0)
                print(
                    f"🏷️  Discovered {count} tags: {tags[:10]}{'...' if len(tags) > 10 else ''}"
                )

    def _generate_report(self) -> None:
        """Generate comprehensive HTML debug report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"mcp_debug_report_{report_time}.html"

        html_content = self._generate_html_report()

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 Generated report: {report_file}")

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        total_duration = (datetime.now() - self.start_time).total_seconds()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PyContextify MCP Debug Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .operation {{ border: 1px solid #e1e5e9; margin: 20px 0; border-radius: 6px; }}
        .operation-header {{ background: #f8f9fa; padding: 15px; border-bottom: 1px solid #e1e5e9; }}
        .operation-body {{ padding: 15px; }}
        .success {{ border-left: 4px solid #28a745; }}
        .error {{ border-left: 4px solid #dc3545; }}
        .json-block {{ background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .search-results {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        pre {{ margin: 0; white-space: pre-wrap; }}
        .stats {{ display: flex; gap: 20px; }}
        .stat-box {{ background: white; border: 1px solid #e1e5e9; padding: 15px; border-radius: 6px; }}
        .collapsible {{ background: #007bff; color: white; cursor: pointer; padding: 10px; border: none; text-align: left; outline: none; font-size: 14px; border-radius: 4px; margin: 10px 0; width: 100%; }}
        .collapsible:hover {{ background: #0056b3; }}
        .collapsible:after {{ content: '+'; font-weight: bold; float: right; margin-left: 5px; }}
        .collapsible.active:after {{ content: '-'; }}
        .collapsible-content {{ max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; background: #f8f9fa; border-radius: 4px; margin-bottom: 10px; }}
        .collapsible-content.show {{ max-height: none; padding: 15px; border: 1px solid #e1e5e9; }}
    </style>
    <script>
        function toggleCollapsible(element) {{
            element.classList.toggle('active');
            var content = element.nextElementSibling;
            content.classList.toggle('show');
        }}
        document.addEventListener('DOMContentLoaded', function() {{
            var collapsibles = document.getElementsByClassName('collapsible');
            for (var i = 0; i < collapsibles.length; i++) {{
                collapsibles[i].addEventListener('click', function() {{
                    toggleCollapsible(this);
                }});
            }}
        }});
    </script>
</head>
<body>
    <div class="header">
        <h1>🔍 PyContextify MCP Debug Report</h1>
        <div class="stats">
            <div class="stat-box">
                <strong>Start Time:</strong><br>{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            <div class="stat-box">
                <strong>Total Duration:</strong><br>{total_duration:.2f}s
            </div>
            <div class="stat-box">
                <strong>Operations:</strong><br>{len(self.call_log)}
            </div>
            <div class="stat-box">
                <strong>Search Query:</strong><br>{self.search_query}
            </div>
        </div>
    </div>
"""

        # Add each operation
        for entry in self.call_log:
            css_class = "error" if entry["is_error"] else "success"
            status_icon = "❌" if entry["is_error"] else "✅"

            html += f"""
    <div class="operation {css_class}">
        <div class="operation-header">
            <h3>{status_icon} Step {entry['step']}: {entry['tool']} ({entry['duration_seconds']:.2f}s)</h3>
            <small>{entry['timestamp']}</small>
        </div>
        <div class="operation-body">
            <button class="collapsible">📤 View Arguments</button>
            <div class="collapsible-content">
                <div class="json-block">
                    <pre>{json.dumps(entry['arguments'], indent=2, default=str)}</pre>
                </div>
            </div>
            
            <button class="collapsible">📨 View Response</button>
            <div class="collapsible-content">
                <div class="json-block">
                    <pre>{json.dumps(entry['payload'], indent=2, default=str)}</pre>
                </div>
            </div>
"""

            # Special handling for search results
            if entry["tool"] == "search" and isinstance(entry["payload"], list):
                html += f"""
            <div class="search-results">
                <h4>🔍 Found Chunks ({len(entry['payload'])} results):</h4>
"""
                for i, result in enumerate(entry["payload"][:5]):  # Show first 5
                    if isinstance(result, dict):
                        score = result.get("similarity_score", 0)
                        chunk_text = result.get("chunk_text", "")[:500]
                        source_path = result.get("source_path", "Unknown")
                        html += f"""
                <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 4px;">
                    <strong>Result {i+1}</strong> (Score: {score:.4f})<br>
                    <small>Source: {source_path}</small><br>
                    <code>{chunk_text}...</code>
                </div>
"""
                html += "</div>"

            html += "</div></div>"

        html += """
</body>
</html>
"""
        return html


async def main():
    """Main entry point for the debug script."""
    parser = argparse.ArgumentParser(description="Debug PyContextify MCP system")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./.debug"),
        help="Output directory for debug reports",
    )
    parser.add_argument(
        "--search-query",
        type=str,
        default="ASPICE integration testing processes",
        help="Search query to test with",
    )

    args = parser.parse_args()

    debugger = MCPDebugger(args.output_dir, args.search_query)
    await debugger.run_debug_session()


if __name__ == "__main__":
    asyncio.run(main())
