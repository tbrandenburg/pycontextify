import pytest

pytest.importorskip("crawl4ai")

from pycontextify.index_webpage import WebpageLoader


@pytest.mark.integration
@pytest.mark.slow
class TestRecursiveCrawlingRealWorld:
    """Real-world integration tests for recursive webpage crawling."""

    def test_recursive_crawl_simple_site(self):
        """
        Test recursive crawling on a simple multi-page website.

        Uses Python.org's download page which has a simple structure with links to different versions.
        Depth 1 means: start page + pages directly linked from start page.
        """
        loader = WebpageLoader()

        # Use Python.org's download page - simple structure with clear internal links
        start_url = "https://www.python.org/downloads/"

        print(f"\n[CRAWL] Starting recursive crawl from: {start_url}")
        print(
            "        This will follow internal links at depth 1 (start page + direct children)..."
        )

        # Crawl with depth 1 to keep test fast and simple
        # Depth 1 means: start page + pages directly linked from start page
        pages = loader.load(start_url, recursive=True, max_depth=1)

        # Extract URLs and content
        crawled_urls = [url for url, _ in pages]
        crawled_content = [content for _, content in pages]

        print(f"\n[RESULTS] Crawl results:")
        print(f"   - Total pages crawled: {len(pages)}")
        print(f"   - Crawled URLs:")
        for i, url in enumerate(crawled_urls[:10]):
            print(f"     {i+1}. {url}")
        if len(crawled_urls) > 10:
            print(f"     ... and {len(crawled_urls) - 10} more")

        # Verify we got at least one page (the start page)
        assert len(pages) >= 1, f"Expected at least 1 page from crawl, got {len(pages)}"

        # Verify the start URL is in the results
        assert any(
            "python.org/downloads" in url.lower() for url in crawled_urls
        ), f"Start URL should be in crawled results. Got: {crawled_urls[:3]}"

        # Verify we have actual content from pages
        for i, (url, content) in enumerate(pages[:3]):
            assert content.strip(), f"Page {i} ({url}) should have content"
            assert len(content) > 50, (
                f"Page {i} ({url}) content seems too short: {len(content)} chars. "
                "Expected some text content."
            )

        # Verify URLs are unique (no duplicates)
        unique_urls = set(crawled_urls)
        assert len(unique_urls) == len(crawled_urls), (
            f"Found duplicate URLs in crawl results. "
            f"Total: {len(crawled_urls)}, Unique: {len(unique_urls)}"
        )

        print(f"\n[SUCCESS] Recursive crawl test passed:")
        print(f"          - Crawled {len(pages)} unique pages")
        print(f"          - Start URL: {start_url}")
        print(f"          - All URLs are unique")

        # Verify content quality - should contain Python-related terms
        combined_content = " ".join(crawled_content).lower()
        expected_terms = ["python", "download"]
        for term in expected_terms:
            assert (
                term in combined_content
            ), f"Expected to find '{term}' in crawled content from Python.org pages"

        print(
            f"          - Content validation: Found expected terms ({', '.join(expected_terms)}) in crawled pages"
        )

    def test_single_page_non_recursive(self):
        """
        Test non-recursive crawling (single page only).

        This ensures basic crawling works before testing recursive functionality.
        """
        loader = WebpageLoader()

        # Use a simple, reliable page
        start_url = "https://www.python.org/about/"

        print(f"\n[CRAWL] Testing single-page crawl: {start_url}")

        # Non-recursive crawl - should only get the start page
        pages = loader.load(start_url, recursive=False)

        assert (
            len(pages) == 1
        ), f"Expected 1 page from non-recursive crawl, got {len(pages)}"

        url, content = pages[0]
        assert content.strip(), "Page should have content"
        assert len(content) > 100, f"Content seems too short: {len(content)} chars"

        print(f"\n[SUCCESS] Single-page crawl test passed:")
        print(f"          - URL: {url}")
        print(f"          - Content length: {len(content)} chars")
        print(f"          - Content preview: {content[:100]}...")
