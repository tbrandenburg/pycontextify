import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import anyio
import fitz
import pytest

from fastmcp.client import Client, UvStdioTransport


def test_pycontextify_cli_e2e(tmp_path):
    """Run an end-to-end MCP workflow through the `uv run pycontextify` CLI server."""

    project_root = Path(__file__).resolve().parents[1]

    # Provide a lightweight sentence-transformers shim so the real server can
    # run without downloading large models from Hugging Face. The stub mimics
    # the small portion of the API that PyContextify relies on while producing
    # deterministic embeddings for test assertions.
    stub_root = tmp_path / "stubs"
    st_package_dir = stub_root / "sentence_transformers"
    st_package_dir.mkdir(parents=True, exist_ok=True)
    (st_package_dir / "__init__.py").write_text(
        r"""
import hashlib
from typing import Iterable, List, Sequence, Union

import numpy as np


__all__ = ["SentenceTransformer"]
__version__ = "0.0.0-test"


class SentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or "cpu"
        self._dim = 128

    def _embed(self, text: str) -> np.ndarray:
        data = text.encode("utf-8")
        digest = hashlib.sha256(data).digest()
        repeats = (self._dim * 4 + len(digest) - 1) // len(digest)
        buffer = (digest * repeats)[: self._dim * 4]
        array = np.frombuffer(buffer, dtype=np.uint32)
        vector = array.astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            iterable: Iterable[str] = [sentences]
        else:
            iterable = sentences

        vectors: List[np.ndarray] = []
        for sentence in iterable:
            vector = self._embed(sentence or "")
            if not normalize_embeddings:
                vector = vector * self._dim
            vectors.append(vector)

        result = np.stack(vectors, axis=0)
        if convert_to_numpy:
            return result
        return result.tolist()
""",
        encoding="utf-8",
    )

    crawl_package = stub_root / "crawl4ai"
    crawl_package.mkdir(parents=True, exist_ok=True)
    (crawl_package / "__init__.py").write_text(
        r"""
import asyncio
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence


__all__ = [
    "AsyncWebCrawler",
    "BrowserConfig",
    "CacheMode",
    "CrawlResult",
    "CrawlerRunConfig",
    "html2text",
]


class CacheMode:
    BYPASS = "bypass"


class BrowserConfig:
    def __init__(self, browser_mode: str = "dedicated", headless: bool = True, verbose: bool = False) -> None:
        self.browser_mode = browser_mode
        self.headless = headless
        self.verbose = verbose


class CrawlerRunConfig:
    def __init__(
        self,
        *,
        cache_mode: str | None = None,
        excluded_tags: Sequence[str] | None = None,
        remove_overlay_elements: bool = True,
        verbose: bool = False,
    ) -> None:
        self.cache_mode = cache_mode
        self.excluded_tags = list(excluded_tags or [])
        self.remove_overlay_elements = remove_overlay_elements
        self.verbose = verbose
        self.mean_delay: float = 0.0
        self.max_range: float = 0.0

    def clone(self, **overrides):
        cloned = CrawlerRunConfig(
            cache_mode=overrides.get("cache_mode", self.cache_mode),
            excluded_tags=list(overrides.get("excluded_tags", self.excluded_tags)),
            remove_overlay_elements=overrides.get(
                "remove_overlay_elements", self.remove_overlay_elements
            ),
            verbose=overrides.get("verbose", self.verbose),
        )
        cloned.mean_delay = overrides.get("mean_delay", self.mean_delay)
        cloned.max_range = overrides.get("max_range", self.max_range)
        for key, value in overrides.items():
            setattr(cloned, key, value)
        return cloned


@dataclass
class CrawlResult:
    url: str
    text: str

    @property
    def success(self) -> bool:
        return True

    @property
    def markdown(self):
        return SimpleNamespace(fit_markdown=self.text)

    @property
    def cleaned_html(self) -> str:
        return self.text


def html2text(html: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", html)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class AsyncWebCrawler:
    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def arun(self, url: str, config: CrawlerRunConfig):
        import urllib.request

        try:
            with urllib.request.urlopen(url) as response:
                raw_bytes = response.read()
        except Exception:
            return []

        try:
            html = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            html = raw_bytes.decode("latin-1", errors="ignore")

        text = html2text(html)
        async def _result():
            return [CrawlResult(url=url, text=text)]

        return await _result()
""",
        encoding="utf-8",
    )

    (crawl_package / "deep_crawling.py").write_text(
        r"""
class BFSDeepCrawlStrategy:
    def __init__(self, *, max_depth, include_external, max_pages, logger=None):
        self.max_depth = max_depth
        self.include_external = include_external
        self.max_pages = max_pages
        self.logger = logger

    def __iter__(self):
        return iter(())
""",
        encoding="utf-8",
    )

    (crawl_package / "models.py").write_text(
        r"""
class CrawlResultContainer(list):
    pass
""",
        encoding="utf-8",
    )

    install_dir = crawl_package / "install.py"
    install_dir.write_text(
        r"""
def install_playwright():
    return True
""",
        encoding="utf-8",
    )

    (stub_root / "sitecustomize.py").write_text(
        r"""
import re
import urllib.request


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load(self, url: str, recursive: bool = False, max_depth: int | None = None):
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
    except Exception:
        return []

    try:
        html = data.decode("utf-8")
    except UnicodeDecodeError:
        html = data.decode("latin-1", errors="ignore")

    return [(url, _strip_html(html))]


try:
    from pycontextify.index import loaders as _loaders
except Exception:  # pragma: no cover - best-effort monkeypatch
    _loaders = None

if _loaders is not None:
    _loaders.WebpageLoader.load = _load
    _loaders.WebpageLoader._ensure_runtime = lambda self: None
    _loaders.WebpageLoader._requires_local_browser = lambda self: False
    _loaders.WebpageLoader._retry_after_runtime_error = lambda self, exc: False
""",
        encoding="utf-8",
    )

    pythonpath_parts = [str(stub_root), str(project_root)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # Prepare local webpage served over HTTP for the crawler stub
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    html_content = """
    <html>
      <head><title>PyContextify Test Page</title></head>
      <body>
        <h1>PyContextify MCP Webpage Fixture</h1>
        <p>This content validates the webpage indexing workflow.</p>
        <p>Key phrase: integration-harness-webpage.</p>
      </body>
    </html>
    """
    (web_dir / "index.html").write_text(html_content, encoding="utf-8")

    class QuietHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_dir), **kwargs)

        def log_message(self, *_args, **_kwargs):  # pragma: no cover - avoid noise
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    webpage_url = f"http://127.0.0.1:{server.server_port}/index.html"

    # Prepare a simple PDF using PyMuPDF so the full pipeline is exercised
    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            (72, 720),
            "PyContextify PDF integration document.\nKey phrase: integration-harness-pdf.",
            fontsize=16,
        )
        doc.save(pdf_path)

    env_vars = {
        "PYTHONPATH": ":".join(pythonpath_parts),
        "PYCONTEXTIFY_AUTO_PERSIST": "false",
        "PYCONTEXTIFY_AUTO_LOAD": "false",
        "PYCONTEXTIFY_USE_HYBRID_SEARCH": "false",
        "PYCONTEXTIFY_INDEX_DIR": str(index_dir),
        "PYCONTEXTIFY_EMBEDDING_PROVIDER": "sentence_transformers",
        "PYCONTEXTIFY_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    }

    transport = UvStdioTransport(
        "pycontextify",
        project_directory=str(project_root),
        env_vars=env_vars,
        keep_alive=False,
    )

    client = Client(transport, init_timeout=180)

    async def run_workflow() -> None:
        async with client:
            async def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
                result = await client.call_tool_mcp(name=name, arguments=arguments)
                assert not result.isError, f"{name} returned error: {result.content}"
                if result.structuredContent is not None:
                    data = result.structuredContent
                    if isinstance(data, dict) and "result" in data:
                        return data["result"]
                    if isinstance(data, str):
                        try:
                            parsed = json.loads(data)
                            if isinstance(parsed, (list, dict)):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                    return data
                blocks: List[str] = []
                for block in result.content or []:
                    text = getattr(block, "text", None)
                    if text:
                        blocks.append(text)
                combined = "\n".join(blocks)
                if combined:
                    try:
                        parsed = json.loads(combined)
                        if isinstance(parsed, (list, dict)):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                return combined

            status_before = await call_tool("status", {})
            assert status_before["index_stats"]["total_chunks"] == 0

            code_path = project_root / "pycontextify" / "index"
            code_result = await call_tool("index_code", {"path": str(code_path)})
            assert code_result["chunks_added"] > 0
            assert code_result["source_type"] == "code"

            webpage_result = await call_tool(
                "index_webpage",
                {"url": webpage_url, "recursive": False, "max_depth": 1},
            )
            assert "chunks_added" in webpage_result and webpage_result["chunks_added"] >= 0
            assert webpage_result.get("source_type") == "webpage" or "error" not in webpage_result

            pdf_result = await call_tool(
                "index_document",
                {"path": str(pdf_path)},
            )
            assert pdf_result["chunks_added"] > 0
            assert pdf_result["source_type"] == "document"

            status_after = await call_tool("status", {})
            index_stats = status_after["index_stats"]
            assert index_stats["total_documents"] >= 2
            source_stats = index_stats.get("source_types", {})
            assert source_stats.get("code", 0) >= 1
            assert source_stats.get("webpage", 0) >= 1
            assert source_stats.get("document", 0) >= 1

            web_search = await call_tool(
                "search",
                {"query": "integration-harness-webpage", "top_k": 5, "display_format": "structured"},
            )
            assert isinstance(web_search, list), f"unexpected web search payload: {web_search!r}"
            assert web_search, "web search returned no results"
            assert all(isinstance(item, dict) for item in web_search)
            assert all("source_type" in item for item in web_search)

            code_search = await call_tool(
                "search",
                {"query": "IndexManager semantic", "top_k": 5, "display_format": "structured"},
            )
            assert isinstance(code_search, list), f"unexpected code search payload: {code_search!r}"
            assert any(item["source_type"] == "code" for item in code_search)

            pdf_search = await call_tool(
                "search",
                {"query": "integration-harness-pdf", "top_k": 5, "display_format": "structured"},
            )
            assert isinstance(pdf_search, list), f"unexpected pdf search payload: {pdf_search!r}"
            assert pdf_search, "pdf search returned no results"
            assert all(isinstance(item, dict) for item in pdf_search)
            assert all("source_type" in item for item in pdf_search)

            reset_result = await call_tool(
                "reset_index", {"confirm": True, "remove_files": True}
            )
            assert reset_result["success"] is True

            status_final = await call_tool("status", {})
            assert status_final["index_stats"]["total_chunks"] == 0

    try:
        anyio.run(run_workflow, backend="asyncio")
    finally:
        server.shutdown()
        server.server_close()
