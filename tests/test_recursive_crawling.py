"""Comprehensive tests for recursive web crawling to identify depth and link limit issues."""

from unittest.mock import Mock, patch

import pytest

from pycontextify.index.loaders import WebpageLoader


class TestRecursiveCrawlingDepth:
    """Tests to verify recursive crawling depth behavior."""

    def test_max_depth_2_should_crawl_two_levels(self):
        """Test that max_depth=2 actually crawls 2 levels deep (root + children)."""
        loader = WebpageLoader()

        # Setup mock responses for a tree structure:
        # root -> child1, child2
        # child1 -> grandchild1
        
        def mock_load_page(url):
            if url == "https://example.com/":
                return "<html><body><main>Root Page Content</main></body></html>"
            elif url == "https://example.com/child1":
                return "<html><body><main>Child 1 Content</main></body></html>"
            elif url == "https://example.com/child2":
                return "<html><body><main>Child 2 Content</main></body></html>"
            elif url == "https://example.com/grandchild1":
                return "<html><body><main>Grandchild 1 Content</main></body></html>"
            return None

        def mock_extract_links_from_soup(soup, base_url):
            text = soup.get_text(separator=" ", strip=True)
            if "Root Page" in text:
                return ["https://example.com/child1", "https://example.com/child2"]
            elif "Child 1" in text:
                return ["https://example.com/grandchild1"]
            elif "Child 2" in text:
                return ["https://example.com/grandchild2"]
            return []

        def mock_fetch_and_parse(url):
            html = mock_load_page(url)
            if html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                return soup, text
            return None, None

        with patch.object(loader, "_fetch_and_parse", side_effect=mock_fetch_and_parse), \
             patch.object(loader, "_extract_links_from_soup", side_effect=mock_extract_links_from_soup):
            # Test with max_depth=2, starting at depth=1
            # NEW BEHAVIOR: depth 1 = root, depth 2 = children
            result = loader._crawl_recursive("https://example.com/", max_depth=2, current_depth=1)

            urls = [url for url, content in result]
            print(f"\nURLs crawled with max_depth=2: {urls}")

            # EXPECTED: root (depth 1), child1 (depth 2), child2 (depth 2)
            # But NOT grandchild1 (would be depth 3)
            assert "https://example.com/" in urls, "Root should be crawled"
            assert "https://example.com/child1" in urls, "Child1 should be crawled"
            assert "https://example.com/child2" in urls, "Child2 should be crawled"
            
            # Grandchildren should NOT be crawled with max_depth=2
            # because they would be at depth 3, which exceeds max_depth
            assert "https://example.com/grandchild1" not in urls, \
                "Grandchild should NOT be crawled with max_depth=2"

    def test_max_depth_3_should_crawl_three_levels(self):
        """Test that max_depth=3 actually crawls 3 levels deep."""
        loader = WebpageLoader()

        def mock_load_page(url):
            if url == "https://example.com/":
                return "<html><body><main>Root Page Content</main></body></html>"
            elif url == "https://example.com/child1":
                return "<html><body><main>Child 1 Content</main></body></html>"
            elif url == "https://example.com/grandchild1":
                return "<html><body><main>Grandchild 1 Content</main></body></html>"
            return None

        def mock_extract_links_from_soup(soup, base_url):
            text = soup.get_text(separator=" ", strip=True)
            if "Root Page" in text:
                return ["https://example.com/child1"]
            elif "Child 1" in text:
                return ["https://example.com/grandchild1"]
            return []

        def mock_fetch_and_parse(url):
            html = mock_load_page(url)
            if html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                return soup, text
            return None, None

        with patch.object(loader, "_fetch_and_parse", side_effect=mock_fetch_and_parse), \
             patch.object(loader, "_extract_links_from_soup", side_effect=mock_extract_links_from_soup):
            # NEW BEHAVIOR: start at depth=1
            # depth 1 = root, depth 2 = children, depth 3 = grandchildren
            result = loader._crawl_recursive("https://example.com/", max_depth=3, current_depth=1)

            urls = [url for url, content in result]
            print(f"\nURLs crawled with max_depth=3: {urls}")

            assert "https://example.com/" in urls, "Root should be crawled"
            assert "https://example.com/child1" in urls, "Child should be crawled"
            # With max_depth=3, grandchild SHOULD be crawled
            # because it's at depth 3, which is within the limit
            assert "https://example.com/grandchild1" in urls, \
                "Grandchild SHOULD be crawled with max_depth=3"


class TestRecursiveCrawlingLinkLimit:
    """Tests to verify the 10-link limit per page."""

    def test_only_first_10_links_are_followed(self):
        """Test that only the first max_links_per_page links on a page are crawled."""
        # Create loader with default max_links_per_page=10
        loader = WebpageLoader(max_links_per_page=10)

        def mock_load_page(url):
            if url == "https://example.com/":
                return "<html><body><main>Root with many links</main></body></html>"
            else:
                return f"<html><body><main>Page {url}</main></body></html>"

        def mock_extract_links_from_soup(soup, base_url):
            text = soup.get_text(separator=" ", strip=True)
            if "Root with many links" in text:
                # Return 20 links
                return [f"https://example.com/page{i}" for i in range(20)]
            return []

        def mock_fetch_and_parse(url):
            html = mock_load_page(url)
            if html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                return soup, text
            return None, None

        with patch.object(loader, "_fetch_and_parse", side_effect=mock_fetch_and_parse), \
             patch.object(loader, "_extract_links_from_soup", side_effect=mock_extract_links_from_soup):
            # NEW BEHAVIOR: start at depth=1
            result = loader._crawl_recursive("https://example.com/", max_depth=2, current_depth=1)

            urls = [url for url, content in result]
            print(f"\nURLs crawled (with 20 available links): {urls}")
            print(f"Total pages crawled: {len(urls)}")

            # Should have: 1 root + 10 child pages = 11 total
            # (NOT 1 + 20 = 21)
            assert len(urls) == 11, \
                f"Should only crawl 11 pages (1 root + 10 children), got {len(urls)}"

            # Verify we got the first 10 links
            for i in range(10):
                assert f"https://example.com/page{i}" in urls

            # Verify we did NOT get links 10-19
            for i in range(10, 20):
                assert f"https://example.com/page{i}" not in urls

    def test_configurable_max_links_per_page(self):
        """Test that max_links_per_page can be configured."""
        # Create loader with custom max_links_per_page=5
        loader = WebpageLoader(max_links_per_page=5)

        def mock_load_page(url):
            if url == "https://example.com/":
                return "<html><body><main>Root with many links</main></body></html>"
            else:
                return f"<html><body><main>Page {url}</main></body></html>"

        def mock_extract_links_from_soup(soup, base_url):
            text = soup.get_text(separator=" ", strip=True)
            if "Root with many links" in text:
                # Return 20 links
                return [f"https://example.com/page{i}" for i in range(20)]
            return []

        def mock_fetch_and_parse(url):
            html = mock_load_page(url)
            if html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                return soup, text
            return None, None

        with patch.object(loader, "_fetch_and_parse", side_effect=mock_fetch_and_parse), \
             patch.object(loader, "_extract_links_from_soup", side_effect=mock_extract_links_from_soup):
            result = loader._crawl_recursive("https://example.com/", max_depth=2, current_depth=1)

            urls = [url for url, content in result]
            print(f"\nURLs crawled (with max_links_per_page=5): {urls}")
            print(f"Total pages crawled: {len(urls)}")

            # Should have: 1 root + 5 child pages = 6 total
            assert len(urls) == 6, \
                f"Should only crawl 6 pages (1 root + 5 children), got {len(urls)}"

            # Verify we got the first 5 links
            for i in range(5):
                assert f"https://example.com/page{i}" in urls

            # Verify we did NOT get links 5-19
            for i in range(5, 20):
                assert f"https://example.com/page{i}" not in urls


class TestRecursiveCrawlingActualBehavior:
    """Test to document the actual behavior vs expected behavior."""

    def test_document_new_depth_behavior(self):
        """Document how max_depth parameter works with new inclusive depth logic."""
        loader = WebpageLoader()

        test_cases = [
            # (max_depth, current_depth, should_continue_crawling)
            # NEW BEHAVIOR: depth starts at 1, condition is: max_depth == 0 or current_depth < max_depth
            (1, 1, False),  # 1 < 1 = False, don't crawl children of root
            (1, 2, False),  # 2 > 1, exceeded depth
            (2, 1, True),   # 1 < 2 = True, crawl children of root
            (2, 2, False),  # 2 < 2 = False, don't crawl grandchildren
            (3, 1, True),   # 1 < 3 = True, crawl children
            (3, 2, True),   # 2 < 3 = True, crawl grandchildren
            (3, 3, False),  # 3 < 3 = False, don't go deeper
            (0, 1, True),   # 0 = unlimited, always continue
            (0, 10, True),  # 0 = unlimited, always continue
        ]

        for max_depth, current_depth, expected in test_cases:
            actual = max_depth == 0 or current_depth < max_depth
            print(f"max_depth={max_depth}, current_depth={current_depth}: "
                  f"should_continue={actual}, expected={expected}")
            assert actual == expected, \
                f"Depth calculation mismatch at max_depth={max_depth}, current_depth={current_depth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
