# Web Crawling

PyContextify delegates web crawling to
[Crawl4AI](https://github.com/unclecode/crawl4ai). The loader requests pages
through Crawl4AI's Playwright-powered browser so you get robust rendering and
readable markdown output without having to manage the crawling logic yourself.

> **Browser runtime**
>
> The first crawl downloads the Chromium runtime that Crawl4AI needs. This
> happens automatically and only once per environment. You can still run
> `crawl4ai-setup` manually if you prefer to pre-install the browsers.

## Depth behaviour

`max_depth` mirrors Crawl4AI's `BFSDeepCrawlStrategy`:

| `max_depth` | Pages visited                                |
|-------------|-----------------------------------------------|
| `0`         | No limit (use with caution)                   |
| `1`         | Starting URL only                             |
| `2`         | Root + direct children                        |
| `3`         | Root + children + grandchildren, and so on    |

The loader passes your value directly to Crawl4AI, so the behaviour matches the
upstream defaults.

## Basic usage

```python
from pycontextify.indexer.loaders import WebpageLoader

loader = WebpageLoader()
pages = loader.load(
    "https://example.com",
    recursive=True,
    max_depth=2,
)
```

Pass any Crawl4AI options directly to the loader, for example to disable
headless mode:

```python
loader = WebpageLoader(browser_mode="builtin", headless=False)
pages = loader.load("https://example.com", recursive=False)
```

## MCP server

The MCP tool exposes the same interface:

```python
index_webpage(
    url="https://example.com",
    recursive=True,
    max_depth=3,
)
```
