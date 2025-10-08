#!/usr/bin/env python3
"""Crawl a URL using PyContextify's Crawl4AI-backed loader."""

from __future__ import annotations

import argparse
import logging
from typing import List, Tuple

from pycontextify.index_webpage import WebpageLoader


def _format_page(index: int, page: Tuple[str, str]) -> str:
    url, content = page
    header = f"=== Page {index}: {url} ==="
    return f"{header}\n{content}\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a URL using PyContextify's Crawl4AI integration with an optional "
            "crawl depth."
        )
    )
    parser.add_argument("url", help="The starting URL to crawl")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help=(
            "Maximum crawl depth. Use 0 for no limit. Depth follows Crawl4AI semantics "
            "where 1 includes the starting URL and its direct children."
        ),
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Delay between Crawl4AI requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--browser-mode",
        default="dedicated",
        choices=["dedicated", "builtin", "api"],
        help=(
            "Crawl4AI browser mode. Use 'api' to delegate rendering to Crawl4AI's "
            "hosted browser when available."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode (default)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="Run the browser with a visible window",
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--single-page",
        action="store_true",
        help="Only fetch the starting URL (skip deep crawling)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("crawl_url")

    loader = WebpageLoader(
        delay_seconds=args.delay_seconds,
        max_depth=args.max_depth,
        headless=args.headless,
        browser_mode=args.browser_mode,
    )

    try:
        stop_after_first = args.single_page
        if args.single_page:
            run_config = loader._build_run_config()
            results = loader._execute_crawl(args.url, run_config)
        else:
            run_config = loader._build_run_config(
                deep_crawl_strategy=loader._create_deep_crawl_strategy(args.max_depth),
                stream=False,
            )
            results = loader._execute_crawl(args.url, run_config)

        pages: List[Tuple[str, str]] = loader._results_to_pages(
            results,
            args.url,
            stop_after_first=stop_after_first,
        )
    except ModuleNotFoundError as exc:
        logger.error("%s", exc)
        return
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        logger.error("Crawl4AI request failed: %s", exc)
        return

    if not pages:
        errors = [
            getattr(result, "error_message", None)
            for result in results
            if not getattr(result, "success", False)
        ]
        if errors:
            logger.warning("Crawl4AI reported %d errors. First: %s", len(errors), errors[0])
        logger.info("No pages were returned by the crawl.")
        return

    for index, page in enumerate(pages, start=1):
        print(_format_page(index, page))


if __name__ == "__main__":
    main()
